from __future__ import annotations

import inspect
import os
import statistics
import threading
from dataclasses import dataclass

import numpy as np

import metile
from kernels.gemm import matmul
from metile.backends.mlx import (
    _mlx_compiler_dtype,
    _mlx_kernel_body,
    _specialize_mlx_source,
    batched_measure,
    calibrate_tournament_batch,
)
from metile.compiler.lowering import _lower_tensor_ops_gemm
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import choose_mdl_tie, compressed_description_bits
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest
from metile.runtime.metal_device import MetalDevice
from metile.tuning import round_robin, select_fastest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-dense-matmul-autotune-v1.json"
_SWITCH_MARGIN = 0.03
_TUNER_VERSION = 1


@dataclass(frozen=True)
class MLXDenseMatmulConfig:
    algorithm: str
    block_m: int = 0
    block_n: int = 0
    wm: int = 0
    wn: int = 0
    schedule: str = ""
    outer_k: int = 0
    k_unroll: int = 1


_CONFIGS = (
    MLXDenseMatmulConfig("mlx"),
    MLXDenseMatmulConfig("metile", 128, 128, 4, 4, "grouped4", 128, 1),
    MLXDenseMatmulConfig("metile", 128, 128, 4, 4, "grouped4", 0, 2),
    MLXDenseMatmulConfig("metile", 128, 128, 4, 4, "hilbert", 0, 2),
    MLXDenseMatmulConfig("metile", 64, 128, 2, 4, "morton", 128, 2),
)


@dataclass(frozen=True)
class MLXDenseWeight:
    """Native output-major and AOT K-major views of one dense MLX weight."""

    native_weight: object
    k_major: object
    shape: tuple[int, int]

    @classmethod
    def from_mlx(cls, weight):
        import mlx.core as mx

        if weight.ndim != 2:
            raise ValueError("dense MLX weights must have shape [output, input]")
        if str(weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16"):
            raise TypeError("dense MLX weights must use bfloat16 or float16")
        k_major = mx.contiguous(weight.T)
        mx.eval(k_major)
        return cls(weight, k_major, (weight.shape[1], weight.shape[0]))


@dataclass(frozen=True)
class _MLXDenseKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    config: MLXDenseMatmulConfig
    output_features: int
    description_bits: int

    def __call__(self, values, weight):
        rows = values.size // values.shape[-1]
        threadgroups_m = (rows + self.config.block_m - 1) // self.config.block_m
        return self.operation(
            inputs=[values, weight.k_major],
            grid=(
                threadgroups_m * self.threadgroup[0],
                self.output_features // self.config.block_n,
                1,
            ),
            threadgroup=self.threadgroup,
            output_shapes=[(*values.shape[:-1], self.output_features)],
            output_dtypes=[values.dtype],
        )[0]


def _native_dense_matmul(values, weight):
    return values @ weight.native_weight.T


def _candidate_configs(rows, reduction, output_features):
    return tuple(
        config
        for config in _CONFIGS
        if config.algorithm == "mlx"
        or (
            output_features % config.block_n == 0
            and reduction % (16 * config.k_unroll) == 0
            and (not config.outer_k or reduction % config.outer_k == 0)
            and rows >= 32
        )
    )


def _compile_mlx_dense(rows, reduction, output_features, dtype, config):
    import mlx.core as mx

    numpy_dtype = _mlx_compiler_dtype(dtype)
    if numpy_dtype != np.dtype(np.float16):
        raise TypeError("dense NAX matmul requires bfloat16 or float16")
    if config.algorithm != "metile":
        raise ValueError("only meTile dense configs compile a Metal kernel")
    key = (rows, reduction, output_features, str(dtype), config)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    activations = metile.Buffer.empty((rows * reduction,), dtype=numpy_dtype)
    weight = metile.Buffer.empty((reduction * output_features,), dtype=numpy_dtype)
    output = metile.Buffer.empty((rows * output_features,), dtype=numpy_dtype)
    grid = (
        (rows + config.block_m - 1) // config.block_m,
        output_features // config.block_n,
    )
    launcher = matmul.kernel_fn[grid]
    compiler_options = {
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "BLOCK_K": 16,
        "WM": config.wm,
        "WN": config.wn,
        "SWIZZLE": config.schedule,
        "NAX_FRAGMENTS": True,
        "NAX_K_UNROLL": config.k_unroll,
    }
    if config.outer_k:
        compiler_options["NAX_OUTER_K"] = config.outer_k
    launcher(
        activations,
        weight,
        output,
        rows,
        output_features,
        reduction,
        **compiler_options,
    )
    MetalDevice.get().sync()
    compiled = launcher._last_compiled
    source = _specialize_mlx_source(compiled.msl_source, dtype)
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=f"metile_dense_{stable_digest(key)[:16]}",
        input_names=["A", "B"],
        output_names=["C"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXDenseKernel(
        operation,
        compiled.threadgroup_size,
        config,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def mlx_dense_backend_signature():
    """Return the code/config identity that can change dense dispatch."""
    return stable_digest(
        {
            "accuracy": inspect.getsource(_accuracy_compatible),
            "candidates": inspect.getsource(_candidate_configs),
            "compile": inspect.getsource(_compile_mlx_dense),
            "configs": [vars(config) for config in _CONFIGS],
            "decomposition": inspect.getsource(decompose_nax_fragments),
            "dispatch": inspect.getsource(mlx_dense_matmul),
            "lowering": inspect.getsource(_lower_tensor_ops_gemm),
            "measure": inspect.getsource(_measure_dispatches),
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
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "input_features": weight.shape[0],
            "mlx": mx.__version__,
            "output_features": weight.shape[1],
            "rows": rows,
            "source": mlx_dense_backend_signature(),
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

    difference = mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32))
    maximum_error = float(mx.max(difference).item())
    mean_error = float(mx.mean(difference).item())
    reference_scale = float(mx.max(mx.abs(reference)).item())
    return mean_error <= 0.002 and maximum_error <= 0.02 + 0.02 * reference_scale


def _choose_config(results):
    native = next(result for result in results if result[2].algorithm == "mlx")[2]
    return select_fastest(results, native, lambda _config: _SWITCH_MARGIN, tie_break=choose_mdl_tie)


def _measure_dispatches(dispatches, rounds, *, batch=None):
    if batch is None:
        batch = calibrate_tournament_batch(dispatches[0][1])
    return round_robin(dispatches, rounds, batched_measure(batch))


def _tune_config(values, weight, configs):
    import mlx.core as mx

    reference = _native_dense_matmul(values, weight)
    mx.eval(reference)
    dispatches = []
    for config in configs:
        if config.algorithm == "mlx":
            dispatches.append((config, lambda: _native_dense_matmul(values, weight), 0))
            continue
        try:
            kernel = _compile_mlx_dense(
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

    provisional = _measure_dispatches(dispatches, 9)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [
        candidate
        for candidate in dispatches
        if candidate[0].algorithm == "mlx" or medians[candidate[0]] <= best * 1.08
    ]
    final = _measure_dispatches(finalists, 31)
    return _choose_config(
        [
            (statistics.median(final[config]), description_bits, config)
            for config, _, description_bits in finalists
        ]
    )


def mlx_dense_matmul(values, weight, *, autotune=True):
    """Dispatch a dense MLX matmul to native MLX or generated M5 NAX."""
    if not isinstance(weight, MLXDenseWeight):
        raise TypeError("weight must be an MLXDenseWeight")
    if values.ndim < 2 or values.shape[-1] != weight.shape[0]:
        raise ValueError("expected values[..., K] and a KxN dense weight")
    if values.dtype != weight.native_weight.dtype:
        raise TypeError("dense MLX matmul requires matching activation and weight dtypes")
    if str(values.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("dense MLX matmul requires bfloat16 or float16")

    rows = values.size // values.shape[-1]
    configs = _candidate_configs(rows, weight.shape[0], weight.shape[1])
    schedule_key = (rows, weight.shape[0], weight.shape[1], str(values.dtype))
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
        return _native_dense_matmul(values, weight)
    kernel = _compile_mlx_dense(
        rows,
        weight.shape[0],
        weight.shape[1],
        values.dtype,
        selected,
    )
    return kernel(values, weight)


def mlx_dense_matmul_dispatches():
    """Return in-process dense matmul schedule decisions."""
    return tuple(
        {
            "rows": key[0],
            "input_features": key[1],
            "output_features": key[2],
            "dtype": key[3],
            **vars(config),
        }
        for key, config in sorted(_schedule_cache.items())
    )


__all__ = [
    "MLXDenseMatmulConfig",
    "MLXDenseWeight",
    "mlx_dense_backend_signature",
    "mlx_dense_matmul",
    "mlx_dense_matmul_dispatches",
]
