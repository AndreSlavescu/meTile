from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

from metile.backends.mlx import _mlx_kernel_body, _specialize_mlx_source
from metile.backends.mlx_dense import MLXDenseWeight, mlx_dense_matmul
from metile.codegen.msl_emitter import _emit_nax_binary_fragment, emit
from metile.compiler.dense import lower_dense_swiglu
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-dense-swiglu-autotune-v1.json"
_SWITCH_MARGIN = 0.03
_TUNER_VERSION = 1


@dataclass(frozen=True)
class MLXDenseSwiGLUConfig:
    algorithm: str
    block_m: int = 0
    block_n: int = 0
    schedule: str = ""
    k_unroll: int = 1


_CONFIGS = (
    MLXDenseSwiGLUConfig("mlx"),
    MLXDenseSwiGLUConfig("metile", 32, 128, "linear", 2),
    MLXDenseSwiGLUConfig("metile", 64, 64, "grouped8", 2),
    MLXDenseSwiGLUConfig("metile", 64, 128, "morton", 2),
    MLXDenseSwiGLUConfig("metile", 64, 128, "grouped4", 2),
    MLXDenseSwiGLUConfig("metile", 128, 128, "grouped4", 2),
    MLXDenseSwiGLUConfig("metile", 128, 128, "hilbert", 2),
)


@dataclass(frozen=True)
class _MLXDenseSwiGLUKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    config: MLXDenseSwiGLUConfig
    output_features: int
    description_bits: int

    def __call__(self, values, gate_weight, up_weight):
        rows = values.size // values.shape[-1]
        threadgroups_m = (rows + self.config.block_m - 1) // self.config.block_m
        return self.operation(
            inputs=[values, gate_weight.k_major, up_weight.k_major],
            grid=(
                threadgroups_m * self.threadgroup[0],
                self.output_features // self.config.block_n,
                1,
            ),
            threadgroup=self.threadgroup,
            output_shapes=[(*values.shape[:-1], self.output_features)],
            output_dtypes=[values.dtype],
        )[0]


def _native_dense_swiglu(values, gate_weight, up_weight):
    import mlx.nn as nn

    gate = values @ gate_weight.native_weight.T
    up = values @ up_weight.native_weight.T
    return nn.silu(gate) * up


def mlx_dense_swiglu_projected(values, gate_weight, up_weight):
    """Compose exact low-precision projections with MLX's native SwiGLU epilogue."""
    import mlx.nn as nn

    gate = mlx_dense_matmul(values, gate_weight)
    up = mlx_dense_matmul(values, up_weight)
    return nn.silu(gate) * up


def _candidate_configs(rows, reduction, output_features):
    return tuple(
        config
        for config in _CONFIGS
        if config.algorithm == "mlx"
        or (
            rows >= 32
            and output_features % config.block_n == 0
            and reduction % (16 * config.k_unroll) == 0
        )
    )


def _compile_mlx_dense_swiglu(rows, reduction, output_features, dtype, config):
    import mlx.core as mx

    if str(dtype) not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("dense SwiGLU requires bfloat16 or float16")
    if config.algorithm != "metile":
        raise ValueError("only meTile dense SwiGLU configs compile a Metal kernel")
    key = (rows, reduction, output_features, str(dtype), config)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    function_name = f"metile_dense_swiglu_{stable_digest(key)[:16]}"
    metal_ir = optimize_tile_schedules(
        lower_dense_swiglu(
            function_name,
            rows,
            output_features,
            reduction,
            block_m=config.block_m,
            block_n=config.block_n,
            schedule=config.schedule,
            k_unroll=config.k_unroll,
        )
    )
    source = _specialize_mlx_source(emit(metal_ir), dtype)
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "gate_weight", "up_weight"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXDenseSwiGLUKernel(
        operation,
        metal_ir.threadgroup_size,
        config,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def mlx_dense_swiglu_backend_signature():
    """Return the code/config identity that can change dense SwiGLU dispatch."""
    return stable_digest(
        {
            "accuracy": inspect.getsource(_accuracy_compatible),
            "candidates": inspect.getsource(_candidate_configs),
            "compile": inspect.getsource(_compile_mlx_dense_swiglu),
            "configs": [vars(config) for config in _CONFIGS],
            "dispatch": inspect.getsource(mlx_dense_swiglu),
            "epilogue_emitter": inspect.getsource(_emit_nax_binary_fragment),
            "lowering": inspect.getsource(lower_dense_swiglu),
            "selection": inspect.getsource(_choose_config),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_config),
            "tuner": _TUNER_VERSION,
        }
    )


def _persistent_key(rows, gate_weight, dtype, configs):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "input_features": gate_weight.shape[0],
            "mlx": mx.__version__,
            "output_features": gate_weight.shape[1],
            "rows": rows,
            "source": mlx_dense_swiglu_backend_signature(),
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
    return mean_error <= 0.003 and maximum_error <= 0.04 + 0.03 * reference_scale


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


def _measure_dispatches(dispatches, rounds):
    import mlx.core as mx

    samples = {config: [] for config, _, _ in dispatches}
    for round_index in range(rounds):
        shift = round_index % len(dispatches)
        ordered = dispatches[shift:] + dispatches[:shift]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval(dispatch())
            samples[config].append((time.perf_counter_ns() - start) * 1e-9)
    return samples


def _tune_config(values, gate_weight, up_weight, configs):
    import mlx.core as mx

    reference = _native_dense_swiglu(values, gate_weight, up_weight)
    mx.eval(reference)
    dispatches = []
    for config in configs:
        if config.algorithm == "mlx":
            dispatches.append(
                (
                    config,
                    lambda: _native_dense_swiglu(values, gate_weight, up_weight),
                    0,
                )
            )
            continue
        try:
            kernel = _compile_mlx_dense_swiglu(
                values.size // values.shape[-1],
                gate_weight.shape[0],
                gate_weight.shape[1],
                values.dtype,
                config,
            )
            actual = kernel(values, gate_weight, up_weight)
            mx.eval(actual)
            if _accuracy_compatible(actual, reference):
                dispatches.append(
                    (
                        config,
                        lambda kernel=kernel: kernel(values, gate_weight, up_weight),
                        kernel.description_bits,
                    )
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


def mlx_dense_swiglu(values, gate_weight, up_weight, *, autotune=True):
    """Dispatch dense gate/up projections and SwiGLU to native MLX or generated M5 NAX."""
    if not isinstance(gate_weight, MLXDenseWeight) or not isinstance(up_weight, MLXDenseWeight):
        raise TypeError("gate_weight and up_weight must be MLXDenseWeight values")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("dense SwiGLU gate and up weights must have matching shapes")
    if values.ndim < 2 or values.shape[-1] != gate_weight.shape[0]:
        raise ValueError("expected values[..., K] and matching KxN gate/up weights")
    if (
        values.dtype != gate_weight.native_weight.dtype
        or values.dtype != up_weight.native_weight.dtype
    ):
        raise TypeError("dense SwiGLU requires matching activation and weight dtypes")
    if str(values.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("dense SwiGLU requires bfloat16 or float16")

    rows = values.size // values.shape[-1]
    configs = _candidate_configs(rows, gate_weight.shape[0], gate_weight.shape[1])
    schedule_key = (rows, gate_weight.shape[0], gate_weight.shape[1], str(values.dtype))
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(rows, gate_weight, values.dtype, configs)
                selected = _read_config(key, configs)
                if selected is None:
                    selected = (
                        _tune_config(values, gate_weight, up_weight, configs)
                        if autotune
                        else next(
                            (config for config in configs if config.algorithm == "metile"),
                            configs[0],
                        )
                    )
                    _write_config(key, selected)
                _schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        return _native_dense_swiglu(values, gate_weight, up_weight)
    kernel = _compile_mlx_dense_swiglu(
        rows,
        gate_weight.shape[0],
        gate_weight.shape[1],
        values.dtype,
        selected,
    )
    return kernel(values, gate_weight, up_weight)


def mlx_dense_swiglu_dispatches():
    """Return in-process dense SwiGLU schedule decisions."""
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
    "MLXDenseSwiGLUConfig",
    "mlx_dense_swiglu",
    "mlx_dense_swiglu_backend_signature",
    "mlx_dense_swiglu_dispatches",
    "mlx_dense_swiglu_projected",
]
