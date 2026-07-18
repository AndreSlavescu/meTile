from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np

from metile.backends.mlx import _mlx_dtype_to_numpy, _mlx_kernel_body
from metile.codegen.msl_emitter import emit
from metile.compiler.block_scaled import lower_block_scaled_matmul
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.block_scaled import _quantize_block_scaled_arrays
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-block-scaled-autotune-v1.json"


@dataclass(frozen=True)
class MLXBlockScaledConfig:
    block_m: int
    block_n: int
    schedule: str
    fragment_type: str
    k_unroll: int


_CONFIGS = (
    MLXBlockScaledConfig(32, 64, "linear", "bfloat", 2),
    MLXBlockScaledConfig(64, 64, "linear", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "grouped4", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "hilbert", "float", 1),
)


@dataclass(frozen=True)
class MLXBlockScaledWeight:
    """K-major MXFP weight arrays owned by MLX."""

    values: object
    scales: object
    shape: tuple[int, int]
    format: str

    @property
    def bits(self):
        return 4 if self.format == "mxfp4" else 8

    @classmethod
    def quantize(cls, weight, *, format="mxfp8"):
        import mlx.core as mx

        dense = np.array(weight, dtype=np.float32)
        packed, scales = _quantize_block_scaled_arrays(dense, format)
        return cls(mx.array(packed), mx.array(scales), tuple(dense.shape), format)


@dataclass(frozen=True)
class _MLXBlockScaledKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    block_m: int
    block_n: int
    output_features: int
    description_bits: int

    def __call__(self, activations, weight):
        rows = activations.size // activations.shape[-1]
        output_shape = (*activations.shape[:-1], self.output_features)
        threadgroups_m = (rows + self.block_m - 1) // self.block_m
        return self.operation(
            inputs=[activations, weight.values, weight.scales],
            grid=(threadgroups_m * self.threadgroup[0], self.output_features // self.block_n, 1),
            threadgroup=self.threadgroup,
            output_shapes=[output_shape],
            output_dtypes=[activations.dtype],
        )[0]


def _candidate_configs(rows, output_features, reduction):
    return tuple(
        config
        for config in _CONFIGS
        if output_features % config.block_n == 0
        and reduction % (16 * config.k_unroll) == 0
        and (rows >= config.block_m or config.block_m == 32)
    )


def _compile_mlx_block_scaled(rows, reduction, output_features, dtype, format, config):
    import mlx.core as mx

    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    if numpy_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise TypeError("MLX block-scaled matmul requires float16 or float32 activations")
    key = (
        rows,
        reduction,
        output_features,
        numpy_dtype.str,
        format,
        config,
    )
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    function_name = f"metile_mlx_bsmm_{stable_digest(key)[:16]}"
    metal_ir = decompose_nax_fragments(
        optimize_tile_schedules(
            lower_block_scaled_matmul(
                function_name,
                rows,
                output_features,
                reduction,
                4 if format == "mxfp4" else 8,
                block_m=config.block_m,
                block_n=config.block_n,
                register_fragments=True,
                schedule=config.schedule,
                fragment_type=config.fragment_type,
                k_unroll=config.k_unroll,
                activation_type="f16" if numpy_dtype == np.dtype(np.float16) else "f32",
                output_type="f16" if numpy_dtype == np.dtype(np.float16) else "f32",
            )
        )
    )
    source = emit(metal_ir)
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "packed", "scales"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXBlockScaledKernel(
        operation,
        metal_ir.threadgroup_size,
        config.block_m,
        config.block_n,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def _persistent_key(rows, reduction, output_features, dtype, format, configs):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "format": format,
            "mlx": mx.__version__,
            "output_features": output_features,
            "reduction": reduction,
            "rows": rows,
            "source": inspect.getsource(lower_block_scaled_matmul),
            "tuner": 1,
        }
    )


def _read_config(key, configs):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next((config for config in configs if vars(config) == payload), None)


def _write_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_cache_path, {})
    payload[key] = vars(config)
    atomic_write_json(_cache_path, payload)


def _tune_config(activations, weight, configs):
    import mlx.core as mx

    kernels = []
    for config in configs:
        try:
            kernel = _compile_mlx_block_scaled(
                activations.size // activations.shape[-1],
                activations.shape[-1],
                weight.shape[1],
                activations.dtype,
                weight.format,
                config,
            )
            kernels.append((config, kernel))
            mx.eval(kernel(activations, weight))
        except (RuntimeError, TypeError, ValueError):
            continue
    if not kernels:
        raise RuntimeError("no MLX block-scaled kernel compiled for this shape")

    def measure(active, rounds):
        samples = {config: [] for config, _ in active}
        for round_index in range(rounds):
            shift = round_index % len(active)
            ordered = active[shift:] + active[:shift]
            if round_index & 1:
                ordered.reverse()
            for config, kernel in ordered:
                start = time.perf_counter_ns()
                mx.eval(kernel(activations, weight))
                samples[config].append((time.perf_counter_ns() - start) * 1e-9)
        return samples

    provisional = measure(kernels, 9)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [candidate for candidate in kernels if medians[candidate[0]] <= best * 1.08]
    final = measure(finalists, 21)
    return choose_mdl_tie(
        [
            (statistics.median(final[config]), kernel.description_bits, config)
            for config, kernel in finalists
        ]
    )


def mlx_block_scaled_matmul(activations, weight, *, autotune=True):
    """Run a zero-copy M5 MXFP matmul inside an MLX lazy graph."""
    if not isinstance(weight, MLXBlockScaledWeight):
        raise TypeError("weight must be an MLXBlockScaledWeight")
    if activations.ndim < 2 or activations.shape[-1] != weight.shape[0]:
        raise ValueError("expected activations[..., K] and a KxN block-scaled weight")
    if weight.format not in {"mxfp4", "mxfp8"}:
        raise ValueError("block-scaled weight format must be mxfp4 or mxfp8")
    _mlx_dtype_to_numpy(activations.dtype)
    rows = activations.size // activations.shape[-1]
    reduction, output_features = weight.shape
    configs = _candidate_configs(rows, output_features, reduction)
    if not configs:
        raise ValueError("no MLX block-scaled schedule supports this shape")
    schedule_key = (rows, reduction, output_features, str(activations.dtype), weight.format)
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(
                    rows,
                    reduction,
                    output_features,
                    activations.dtype,
                    weight.format,
                    configs,
                )
                selected = _read_config(key, configs)
            if selected is None:
                selected = _tune_config(activations, weight, configs) if autotune else configs[0]
                _write_config(key, selected)
            _schedule_cache[schedule_key] = selected
    kernel = _compile_mlx_block_scaled(
        rows,
        reduction,
        output_features,
        activations.dtype,
        weight.format,
        selected,
    )
    return kernel(activations, weight)


def mlx_block_scaled_dispatches():
    """Return in-process MLX block-scaled schedule decisions."""
    return tuple(
        {
            "rows": key[0],
            "reduction": key[1],
            "output_features": key[2],
            "dtype": key[3],
            "format": key[4],
            **vars(config),
        }
        for key, config in sorted(_schedule_cache.items())
    )


__all__ = [
    "MLXBlockScaledConfig",
    "MLXBlockScaledWeight",
    "mlx_block_scaled_dispatches",
    "mlx_block_scaled_matmul",
]
