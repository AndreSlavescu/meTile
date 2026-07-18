from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np

from metile.backends.mlx import _mlx_dtype_to_numpy, _mlx_kernel_body
from metile.backends.mlx_quantized import repack_mlx_affine_weight
from metile.codegen.msl_emitter import emit
from metile.compiler.affine_quantized import lower_affine_matmul
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-affine-matmul-autotune-v2.json"
_SWITCH_MARGIN = 0.03
_TUNER_VERSION = 2


@dataclass(frozen=True)
class MLXAffineMatmulConfig:
    algorithm: str
    block_n: int = 0
    schedule: str = ""


_CONFIGS = (
    MLXAffineMatmulConfig("mlx"),
    MLXAffineMatmulConfig("metile", 32, "morton"),
    MLXAffineMatmulConfig("metile", 64, "grouped4"),
    MLXAffineMatmulConfig("metile", 128, "hilbert"),
)


@dataclass(frozen=True)
class MLXAffineWeight:
    """Original and K-major views of one MLX affine uint4 weight."""

    native_weight: object
    native_scales: object
    native_biases: object
    packed: object
    scales: object
    biases: object
    shape: tuple[int, int]
    group_size: int = 64
    bits: int = 4

    @classmethod
    def from_mlx(cls, weight, scales, biases, *, group_size=64, bits=4):
        if group_size != 64 or bits != 4:
            raise ValueError("MLX affine NAX weights require group size 64 and 4 bits")
        packed, repacked_scales, repacked_biases = repack_mlx_affine_weight(
            weight,
            scales,
            biases,
        )
        input_features = weight.shape[1] * 32 // bits
        return cls(
            weight,
            scales,
            biases,
            packed,
            repacked_scales,
            repacked_biases,
            (input_features, weight.shape[0]),
            group_size,
            bits,
        )


@dataclass(frozen=True)
class _MLXAffineKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    block_n: int
    output_features: int
    description_bits: int

    def __call__(self, values, weight):
        rows = values.size // values.shape[-1]
        threadgroups_m = (rows + 31) // 32
        return self.operation(
            inputs=[values, weight.packed, weight.scales, weight.biases],
            grid=(
                threadgroups_m * self.threadgroup[0],
                self.output_features // self.block_n,
                1,
            ),
            threadgroup=self.threadgroup,
            output_shapes=[(*values.shape[:-1], self.output_features)],
            output_dtypes=[values.dtype],
        )[0]


def _native_affine_matmul(values, weight):
    import mlx.core as mx

    return mx.quantized_matmul(
        values,
        weight.native_weight,
        scales=weight.native_scales,
        biases=weight.native_biases,
        transpose=True,
        group_size=weight.group_size,
        bits=weight.bits,
        mode="affine",
    )


def _candidate_configs(output_features):
    return tuple(
        config
        for config in _CONFIGS
        if config.algorithm == "mlx" or output_features % config.block_n == 0
    )


def _compile_mlx_affine(rows, input_features, output_features, dtype, config):
    import mlx.core as mx

    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    if numpy_dtype != np.dtype(np.float16):
        raise TypeError("MLX affine NAX matmul requires float16 activations")
    if config.algorithm != "metile":
        raise ValueError("only meTile affine configs compile a Metal kernel")
    key = (rows, input_features, output_features, numpy_dtype.str, config)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    function_name = f"metile_mlx_affine_{stable_digest(key)[:16]}"
    metal_ir = decompose_nax_fragments(
        optimize_tile_schedules(
            lower_affine_matmul(
                function_name,
                rows,
                output_features,
                input_features,
                block_n=config.block_n,
                schedule=config.schedule,
            )
        )
    )
    source = emit(metal_ir)
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "packed", "scales", "biases"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXAffineKernel(
        operation,
        metal_ir.threadgroup_size,
        config.block_n,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def mlx_affine_backend_signature():
    """Return the code/config identity that can change affine dispatch decisions."""
    return stable_digest(
        {
            "accuracy": inspect.getsource(_accuracy_compatible),
            "candidates": inspect.getsource(_candidate_configs),
            "compile": inspect.getsource(_compile_mlx_affine),
            "configs": [vars(config) for config in _CONFIGS],
            "dispatch": inspect.getsource(mlx_affine_matmul),
            "lowering": inspect.getsource(lower_affine_matmul),
            "selection": inspect.getsource(_choose_config),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_config),
            "tuner": _TUNER_VERSION,
        }
    )


def _persistent_key(rows, weight, dtype, configs):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "bits": weight.bits,
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "group_size": weight.group_size,
            "input_features": weight.shape[0],
            "mlx": mx.__version__,
            "output_features": weight.shape[1],
            "rows": rows,
            "source": mlx_affine_backend_signature(),
            "switch_margin": _SWITCH_MARGIN,
            "tuner": _TUNER_VERSION,
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


def _accuracy_compatible(actual, reference):
    import mlx.core as mx

    maximum_error = float(mx.max(mx.abs(actual - reference)).item())
    reference_scale = float(mx.max(mx.abs(reference)).item())
    return maximum_error <= 0.02 + 0.02 * reference_scale


def _choose_config(results):
    native = next(result for result in results if result[2].algorithm == "mlx")
    alternatives = [result for result in results if result[2].algorithm == "metile"]
    if not alternatives:
        return native[2]
    fastest = min(alternatives, key=lambda result: result[0])
    if fastest[0] >= native[0] * (1.0 - _SWITCH_MARGIN):
        return native[2]
    cutoff = fastest[0] * 1.0025
    return choose_mdl_tie([result for result in alternatives if result[0] <= cutoff])


def _tune_config(values, weight, configs):
    import mlx.core as mx

    reference = _native_affine_matmul(values, weight)
    mx.eval(reference)
    dispatches = []
    for config in configs:
        if config.algorithm == "mlx":
            dispatches.append((config, lambda: _native_affine_matmul(values, weight), 0))
            continue
        try:
            kernel = _compile_mlx_affine(
                values.size // values.shape[-1],
                weight.shape[0],
                weight.shape[1],
                values.dtype,
                config,
            )
            actual = kernel(values, weight)
            mx.eval(actual)
            if _accuracy_compatible(actual, reference):
                dispatches.append(
                    (config, lambda kernel=kernel: kernel(values, weight), kernel.description_bits)
                )
        except (RuntimeError, TypeError, ValueError):
            continue

    def measure(active, rounds):
        samples = {config: [] for config, _, _ in active}
        for round_index in range(rounds):
            shift = round_index % len(active)
            ordered = active[shift:] + active[:shift]
            if round_index & 1:
                ordered.reverse()
            for config, dispatch, _ in ordered:
                start = time.perf_counter_ns()
                mx.eval(dispatch())
                samples[config].append((time.perf_counter_ns() - start) * 1e-9)
        return samples

    provisional = measure(dispatches, 9)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [
        candidate
        for candidate in dispatches
        if candidate[0].algorithm == "mlx" or medians[candidate[0]] <= best * 1.08
    ]
    final = measure(finalists, 21)
    return _choose_config(
        [
            (statistics.median(final[config]), description_bits, config)
            for config, _, description_bits in finalists
        ]
    )


def mlx_affine_matmul(values, weight, *, autotune=True):
    """Dispatch an MLX affine uint4 matmul to native MLX or generated M5 NAX."""
    if not isinstance(weight, MLXAffineWeight):
        raise TypeError("weight must be an MLXAffineWeight")
    if values.ndim < 2 or values.shape[-1] != weight.shape[0]:
        raise ValueError("expected values[..., K] and a KxN affine weight")
    if str(values.dtype) != "mlx.core.float16":
        raise TypeError("MLX affine NAX matmul requires float16 activations")

    rows = values.size // values.shape[-1]
    configs = _candidate_configs(weight.shape[1])
    schedule_key = (
        rows,
        weight.shape[0],
        weight.shape[1],
        str(values.dtype),
        weight.group_size,
        weight.bits,
    )
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(rows, weight, values.dtype, configs)
                selected = _read_config(key, configs)
            if selected is None:
                selected = (
                    _tune_config(values, weight, configs)
                    if autotune
                    else next(
                        (config for config in configs if config.algorithm == "metile"),
                        configs[0],
                    )
                )
                _write_config(key, selected)
            _schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        return _native_affine_matmul(values, weight)
    kernel = _compile_mlx_affine(
        rows,
        weight.shape[0],
        weight.shape[1],
        values.dtype,
        selected,
    )
    return kernel(values, weight)


def mlx_affine_matmul_dispatches():
    """Return in-process affine matmul schedule decisions."""
    return tuple(
        {
            "rows": key[0],
            "input_features": key[1],
            "output_features": key[2],
            "dtype": key[3],
            "group_size": key[4],
            "bits": key[5],
            **vars(config),
        }
        for key, config in sorted(_schedule_cache.items())
    )


__all__ = [
    "MLXAffineMatmulConfig",
    "MLXAffineWeight",
    "mlx_affine_backend_signature",
    "mlx_affine_matmul",
    "mlx_affine_matmul_dispatches",
]
