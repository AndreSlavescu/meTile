from __future__ import annotations

import inspect
import statistics
import threading
from dataclasses import dataclass

from metile.backends.mlx import (
    _mlx_kernel_body,
    _specialize_mlx_source,
    batched_measure,
    calibrate_tournament_batch,
)
from metile.codegen import msl_emitter
from metile.codegen.msl_emitter import emit
from metile.compiler.dense import lower_dense_residual_qmv
from metile.compiler.schedule_search import choose_mdl_tie, compressed_description_bits
from metile.runtime.cache import (
    cache_root,
    read_cached_config,
    stable_digest,
    write_cached_config,
)
from metile.tuning import round_robin, select_fastest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-dense-residual-autotune-v1.json"
_SWITCH_MARGIN = 0.015
_TUNER_VERSION = 1


@dataclass(frozen=True)
class MLXDenseResidualConfig:
    algorithm: str
    outputs_per_simdgroup: int = 0
    simdgroups_per_threadgroup: int = 0


_CONFIGS = (
    MLXDenseResidualConfig("mlx"),
    MLXDenseResidualConfig("metile", 1, 1),
    MLXDenseResidualConfig("metile", 1, 2),
    MLXDenseResidualConfig("metile", 1, 4),
)


@dataclass(frozen=True)
class _MLXDenseResidualKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    config: MLXDenseResidualConfig
    output_features: int
    description_bits: int

    def __call__(self, values, weight, residual):
        outputs = self.config.outputs_per_simdgroup
        simdgroups = self.config.simdgroups_per_threadgroup
        threadgroups = (self.output_features // outputs + simdgroups - 1) // simdgroups
        return self.operation(
            inputs=[values, weight, residual],
            grid=(threadgroups * self.threadgroup[0], 1, 1),
            threadgroup=self.threadgroup,
            output_shapes=[residual.shape],
            output_dtypes=[values.dtype],
        )[0]


def _native_dense_residual(values, weight, residual):
    return values @ weight.T + residual


_MAX_QMV_ROWS = 31
_MAX_QMV_ACCUMULATORS = 32


def _candidate_configs(rows, reduction, output_features):
    return tuple(
        config
        for config in _CONFIGS
        if config.algorithm == "mlx"
        or (
            # One accumulator per (row, output) lives in registers for the whole K loop,
            # so bound the product to keep the SIMDgroup off the spill path.
            1 <= rows <= _MAX_QMV_ROWS
            and rows * config.outputs_per_simdgroup <= _MAX_QMV_ACCUMULATORS
            and reduction % 128 == 0
            and output_features % (config.outputs_per_simdgroup * config.simdgroups_per_threadgroup)
            == 0
        )
    )


def _compile_mlx_dense_residual(reduction, output_features, dtype, config, rows=1):
    import mlx.core as mx

    if config.algorithm != "metile":
        raise ValueError("only meTile dense residual configs compile a Metal kernel")
    key = (reduction, output_features, str(dtype), config, rows)
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    function_name = f"metile_dense_residual_{stable_digest(key)[:16]}"
    source = _specialize_mlx_source(
        emit(
            lower_dense_residual_qmv(
                function_name,
                output_features,
                reduction,
                outputs_per_simdgroup=config.outputs_per_simdgroup,
                simdgroups_per_threadgroup=config.simdgroups_per_threadgroup,
                rows=rows,
            )
        ),
        dtype,
    )
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "weight", "residual"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXDenseResidualKernel(
        operation,
        (config.simdgroups_per_threadgroup * 32, 1, 1),
        config,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def mlx_dense_residual_backend_signature():
    """Return the code/config identity that can change dense residual dispatch."""
    return stable_digest(
        {
            "accumulate_emitter": inspect.getsource(msl_emitter._emit_dot_accumulate),
            "candidates": inspect.getsource(_candidate_configs),
            "compile": inspect.getsource(_compile_mlx_dense_residual),
            "configs": [vars(config) for config in _CONFIGS],
            "dispatch": inspect.getsource(mlx_dense_residual_qmv),
            "init_emitter": inspect.getsource(msl_emitter._emit_dot_accumulator_init),
            "lowering": inspect.getsource(lower_dense_residual_qmv),
            "measure": inspect.getsource(_measure_dispatches),
            "selection": inspect.getsource(_choose_config),
            "store_emitter": inspect.getsource(msl_emitter._emit_dot_residual_store),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_config),
            "tuner": _TUNER_VERSION,
        }
    )


def _persistent_key(rows, reduction, output_features, dtype, configs):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "input_features": reduction,
            "mlx": mx.__version__,
            "output_features": output_features,
            "rows": rows,
            "source": mlx_dense_residual_backend_signature(),
            "switch_margin": _SWITCH_MARGIN,
            "tuner": _TUNER_VERSION,
        }
    )


def _choose_config(results):
    native = next(result for result in results if result[2].algorithm == "mlx")[2]
    return select_fastest(results, native, lambda _config: _SWITCH_MARGIN, tie_break=choose_mdl_tie)


def _measure_dispatches(dispatches, rounds, *, batch=None):
    if batch is None:
        batch = calibrate_tournament_batch(dispatches[0][1])
    return round_robin(dispatches, rounds, batched_measure(batch))


def _tune_config(values, weight, residual, configs, rows=1):
    import mlx.core as mx

    reference = _native_dense_residual(values, weight, residual)
    mx.eval(reference)
    # MLX switches to a tile kernel above one row, so its result is not bit-comparable.
    # Require instead that every row match MLX's own single-row projection exactly, which
    # is what a batched decode step must reproduce to stay equivalent to decoding those
    # tokens one at a time. Rows come from a flattened view because callers pass rank-3
    # [batch, sequence, hidden] as well as rank-2, and slicing axis 0 on the former
    # yields empty rows.
    if 1 < rows <= _MAX_QMV_ROWS:
        flat_values = values.reshape(rows, values.shape[-1])
        flat_residual = residual.reshape(rows, residual.shape[-1])
        per_row = mx.concatenate(
            [
                _native_dense_residual(
                    flat_values[row : row + 1], weight, flat_residual[row : row + 1]
                )
                for row in range(rows)
            ],
            axis=0,
        ).reshape(reference.shape)
        mx.eval(per_row)
        reference = per_row
    dispatches = []
    for config in configs:
        if config.algorithm == "mlx":
            dispatches.append(
                (
                    config,
                    lambda: _native_dense_residual(values, weight, residual),
                    0,
                )
            )
            continue
        try:
            kernel = _compile_mlx_dense_residual(
                values.shape[-1],
                weight.shape[0],
                values.dtype,
                config,
                rows,
            )
            actual = kernel(values, weight, residual)
            mx.eval(actual)
            if bool(mx.array_equal(actual, reference).item()):
                dispatches.append(
                    (
                        config,
                        lambda kernel=kernel: kernel(values, weight, residual),
                        kernel.description_bits,
                    )
                )
        except (RuntimeError, TypeError, ValueError):
            continue

    provisional = _measure_dispatches(dispatches, 11)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [
        candidate
        for candidate in dispatches
        if candidate[0].algorithm == "mlx" or medians[candidate[0]] <= best * 1.08
    ]
    final = _measure_dispatches(finalists, 63)
    return _choose_config(
        [
            (statistics.median(final[config]), description_bits, config)
            for config, _, description_bits in finalists
        ]
    )


def mlx_dense_residual_qmv(values, weight, residual, *, autotune=True):
    """Dispatch an exact dense down projection plus residual addition."""
    if values.ndim < 2 or weight.ndim != 2 or residual.ndim != values.ndim:
        raise ValueError("dense residual QMV requires rank-compatible values, weight, and residual")
    if values.shape[-1] != weight.shape[1]:
        raise ValueError("dense residual QMV input features must match the weight")
    if residual.shape != (*values.shape[:-1], weight.shape[0]):
        raise ValueError("dense residual QMV residual shape must match the projected output")
    if values.dtype != weight.dtype or values.dtype != residual.dtype:
        raise TypeError("dense residual QMV requires matching input dtypes")
    dtype_name = str(values.dtype)
    if dtype_name not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("dense residual QMV requires bfloat16 or float16")

    rows = values.size // values.shape[-1]
    schedule_key = (rows, weight.shape[1], weight.shape[0], dtype_name)
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        # Only needed on a cache miss; keep it off the steady-state decode path.
        configs = _candidate_configs(rows, weight.shape[1], weight.shape[0])
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(*schedule_key, configs)
                selected = read_cached_config(_cache_path, key, configs)
                if selected is None:
                    selected = (
                        _tune_config(values, weight, residual, configs, rows)
                        if autotune
                        else next(config for config in configs if config.algorithm == "metile")
                    )
                    write_cached_config(_cache_path, key, selected)
                _schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        return _native_dense_residual(values, weight, residual)
    kernel = _compile_mlx_dense_residual(
        weight.shape[1],
        weight.shape[0],
        values.dtype,
        selected,
        rows,
    )
    return kernel(values, weight, residual)


def mlx_dense_residual_dispatches():
    """Return in-process dense residual schedule decisions."""
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
    "MLXDenseResidualConfig",
    "mlx_dense_residual_backend_signature",
    "mlx_dense_residual_dispatches",
    "mlx_dense_residual_qmv",
]
