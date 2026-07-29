from __future__ import annotations

import inspect
import os
import statistics
import threading
from dataclasses import dataclass

from metile.backends.mlx import (
    _mlx_kernel_body,
    _specialize_mlx_source,
    batched_measure,
    calibrate_tournament_batch,
)
from metile.backends.mlx_dense import MLXDenseWeight, mlx_dense_matmul
from metile.codegen import msl_emitter
from metile.codegen.msl_emitter import _emit_nax_binary_fragment, emit
from metile.compiler.dense import lower_dense_swiglu, lower_dense_swiglu_qmv
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest
from metile.tuning import confirm_pairwise, round_robin, select_fastest

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-dense-swiglu-autotune-v5.json"
_SWITCH_MARGIN = 0.03
_EXACT_SWITCH_MARGIN = 0.015
_TUNER_VERSION = 5


@dataclass(frozen=True)
class MLXDenseSwiGLUConfig:
    algorithm: str
    block_m: int = 0
    block_n: int = 0
    schedule: str = ""
    k_unroll: int = 1
    implementation: str = "nax"
    outputs_per_simdgroup: int = 0
    simdgroups_per_threadgroup: int = 0


_CONFIGS = (
    MLXDenseSwiGLUConfig("mlx"),
    *(
        MLXDenseSwiGLUConfig(
            "metile",
            k_unroll=k_unroll,
            implementation=implementation,
            outputs_per_simdgroup=outputs,
            simdgroups_per_threadgroup=simdgroups,
        )
        for implementation in ("simdgroup", "simdgroup_paired")
        for outputs in (1, 2, 4)
        for simdgroups in (1, 2, 4, 8)
        for k_unroll in (1, 2)
    ),
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

    def __call__(self, values, gate_weight, up_weight, paired_weight=None):
        if self.config.implementation.startswith("simdgroup"):
            outputs = self.config.outputs_per_simdgroup
            simdgroups = self.config.simdgroups_per_threadgroup
            threadgroups = (self.output_features // outputs + simdgroups - 1) // simdgroups
            if self.config.implementation == "simdgroup_paired":
                if paired_weight is None:
                    raise ValueError("paired SIMDgroup SwiGLU requires an interleaved weight")
                inputs = [values, paired_weight]
            else:
                inputs = [
                    values,
                    gate_weight.native_weight,
                    up_weight.native_weight,
                ]
            return self.operation(
                inputs=inputs,
                grid=(threadgroups * self.threadgroup[0], 1, 1),
                threadgroup=self.threadgroup,
                output_shapes=[(*values.shape[:-1], self.output_features)],
                output_dtypes=[values.dtype],
            )[0]
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


_NAX_MIN_ROWS = 32
# Bounds rows * outputs_per_simdgroup as a proxy for live accumulator pairs. Raising it to
# 32 was tried, on the grounds that outputs_per_simdgroup=2 at 16 rows compiles to 115
# registers against the 140-register budget in metile.target.agx and so cannot spill. It made no
# difference: the tuner already reaches the same speed with outputs_per_simdgroup=1 at a
# larger simdgroup count, so the wider search bought nothing and 16 stands. Audited by
# benchmarks/agx_registers.py, which reports the worst admitted kernel at 99 of 140.
_MAX_QMV_ACCUMULATOR_PAIRS = 16


def _candidate_configs(rows, reduction, output_features, paired_available=False):
    return tuple(
        config
        for config in _CONFIGS
        if config.algorithm == "mlx"
        or (
            config.implementation.startswith("simdgroup")
            # Each row carries a gate and an up accumulator per output, so bound the
            # product rather than the row count to keep the SIMDgroup off the spill path.
            and 1 <= rows < _NAX_MIN_ROWS
            and rows * config.outputs_per_simdgroup <= _MAX_QMV_ACCUMULATOR_PAIRS
            and reduction % 128 == 0
            and (config.implementation != "simdgroup_paired" or paired_available)
            and (config.implementation != "simdgroup" or not paired_available)
            and output_features % (config.outputs_per_simdgroup * config.simdgroups_per_threadgroup)
            == 0
        )
        or (
            config.implementation == "nax"
            and rows >= _NAX_MIN_ROWS
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
    if config.implementation.startswith("simdgroup"):
        if rows < 1:
            raise ValueError("SIMDgroup dense SwiGLU requires at least one row")
        metal_ir = lower_dense_swiglu_qmv(
            function_name,
            output_features,
            reduction,
            outputs_per_simdgroup=config.outputs_per_simdgroup,
            simdgroups_per_threadgroup=config.simdgroups_per_threadgroup,
            interleaved=config.implementation == "simdgroup_paired",
            k_unroll=config.k_unroll,
            rows=rows,
        )
    else:
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
    input_names = (
        ["activations", "paired_weight"]
        if config.implementation == "simdgroup_paired"
        else ["activations", "gate_weight", "up_weight"]
    )
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=input_names,
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
            "qmv_layout_emitter": inspect.getsource(msl_emitter._emit_simdgroup_qmv_layout),
            "qmv_init_emitter": inspect.getsource(msl_emitter._emit_paired_dot_accumulator_init),
            "qmv_accumulate_emitter": inspect.getsource(msl_emitter._emit_paired_dot_accumulate),
            "qmv_store_emitter": inspect.getsource(msl_emitter._emit_paired_dot_swiglu_store),
            "exact_switch_margin": _EXACT_SWITCH_MARGIN,
            "lowering": inspect.getsource(lower_dense_swiglu),
            "qmv_lowering": inspect.getsource(lower_dense_swiglu_qmv),
            "confirm": inspect.getsource(_confirm_pairwise),
            "measure": inspect.getsource(_measure_dispatches),
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


def _native_config(results):
    return next(result for result in results if result[2].algorithm == "mlx")[2]


def _margin_for(config):
    # SIMDgroup candidates are held to per-row bit-exactness, so a win there is a real
    # kernel win and a smaller margin is enough to act on.
    return _EXACT_SWITCH_MARGIN if config.implementation.startswith("simdgroup") else _SWITCH_MARGIN


def _choose_config(results):
    return select_fastest(
        results,
        _native_config(results),
        _margin_for,
        tie_break=choose_mdl_tie,
    )


def _measure_dispatches(dispatches, rounds, *, batch=None):
    if batch is None:
        batch = calibrate_tournament_batch(dispatches[0][1])
    return round_robin(dispatches, rounds, batched_measure(batch))


def _confirm_pairwise(finalists, rounds, batch):
    """Time each finalist against native alone and return results ready for selection."""
    native = next(candidate for candidate in finalists if candidate[0].algorithm == "mlx")
    timings = confirm_pairwise(finalists, native[0], rounds, batched_measure(batch))
    return [
        (timings[config], description_bits, config)
        for config, _, description_bits in finalists
        if config in timings
    ]


def _tune_config(values, gate_weight, up_weight, paired_weight, configs):
    import mlx.core as mx

    reference = _native_dense_swiglu(values, gate_weight, up_weight)
    mx.eval(reference)
    rows = values.size // values.shape[-1]
    exact_reference = reference
    # MLX switches to a tile kernel above one row, so its result is not bit-comparable.
    # Hold the SIMDgroup candidates to the stronger property instead: every row must match
    # MLX's own single-row SwiGLU, so a batched step stays equivalent to decoding those
    # tokens one at a time. Only those candidates are gated on it and they exist only in
    # the QMV band, so building it at prefill sizes would cost one native call per row for
    # nothing. Rows are taken from a flattened view because callers pass rank-3
    # [batch, sequence, hidden] as well as rank-2, and slicing axis 0 on the former yields
    # empty rows.
    if 1 < rows < _NAX_MIN_ROWS:
        flat = values.reshape(rows, values.shape[-1])
        exact_reference = mx.concatenate(
            [
                _native_dense_swiglu(flat[row : row + 1], gate_weight, up_weight)
                for row in range(rows)
            ],
            axis=0,
        ).reshape(reference.shape)
        mx.eval(exact_reference)
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
                rows,
                gate_weight.shape[0],
                gate_weight.shape[1],
                values.dtype,
                config,
            )
            actual = kernel(values, gate_weight, up_weight, paired_weight)
            mx.eval(actual)
            exact_qmv = not config.implementation.startswith("simdgroup") or bool(
                mx.array_equal(actual, exact_reference).item()
            )
            if exact_qmv and _accuracy_compatible(actual, reference):
                dispatches.append(
                    (
                        config,
                        lambda kernel=kernel: kernel(
                            values,
                            gate_weight,
                            up_weight,
                            paired_weight,
                        ),
                        kernel.description_bits,
                    )
                )
        except (RuntimeError, TypeError, ValueError):
            continue

    qmv = values.size // values.shape[-1] == 1
    batch = calibrate_tournament_batch(dispatches[0][1])
    provisional = _measure_dispatches(dispatches, 11 if qmv else 9, batch=batch)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [
        candidate
        for candidate in dispatches
        if candidate[0].algorithm == "mlx" or medians[candidate[0]] <= best * 1.08
    ]
    # Confirm head to head rather than trusting the crowded rotation. This used to run
    # only for one-row decode, which left every multi-row shape picking its kernel from a
    # ranking that does not survive isolated measurement.
    return _choose_config(_confirm_pairwise(finalists, 63 if qmv else 31, batch))


def mlx_dense_swiglu(values, gate_weight, up_weight, *, paired_weight=None, autotune=True):
    """Dispatch dense SwiGLU across native, NAX, and exact SIMDgroup candidates."""
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
    dtype_name = str(values.dtype)
    if dtype_name not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("dense SwiGLU requires bfloat16 or float16")
    if paired_weight is not None and (
        paired_weight.shape != (gate_weight.shape[1], gate_weight.shape[0], 2)
        or paired_weight.dtype != values.dtype
    ):
        raise ValueError("paired dense SwiGLU weight must have shape [N, K, 2] and matching dtype")

    rows = values.size // values.shape[-1]
    paired_available = paired_weight is not None
    schedule_key = (
        rows,
        gate_weight.shape[0],
        gate_weight.shape[1],
        dtype_name,
        paired_available,
    )
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        # Candidate filtering is only needed when the schedule is unknown; keeping it
        # off the steady-state decode path saves ~5 us per dispatch per layer.
        configs = _candidate_configs(
            rows,
            gate_weight.shape[0],
            gate_weight.shape[1],
            paired_available,
        )
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(rows, gate_weight, values.dtype, configs)
                selected = _read_config(key, configs)
                if selected is None:
                    selected = (
                        _tune_config(values, gate_weight, up_weight, paired_weight, configs)
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
    return kernel(values, gate_weight, up_weight, paired_weight)


def mlx_dense_swiglu_dispatches():
    """Return in-process dense SwiGLU schedule decisions."""
    return tuple(
        {
            "rows": key[0],
            "input_features": key[1],
            "output_features": key[2],
            "dtype": key[3],
            "paired_available": key[4],
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
