"""Selection, tuning and dispatch across the affine-quantized kernels."""

from __future__ import annotations

import inspect
import statistics

from metile.backends.mlx import (
    _token_bucket,
    batched_measure,
    calibrate_tournament_batch,
)
from metile.backends.mlx_quantized.common import (
    _AFFINE_RESIDUAL_TUNER_VERSION,
    _AFFINE_SWIGLU_TUNER_VERSION,
    _COMPILED_SWITCH_MARGIN,
    _RESIDUAL_SWITCH_MARGIN,
    _SWITCH_MARGIN,
    _affine_cache_lock,
    _affine_residual_schedule_cache,
    _affine_swiglu_schedule_cache,
    _discard_repacked_affine_pair,
)
from metile.backends.mlx_quantized.residual import (
    _AFFINE_RESIDUAL_CONFIGS,
    MLXAffineResidualConfig,
    _affine_residual_dispatch,
    _choose_affine_residual_config,
    _make_affine_residual_executor,
    _mlx_compiled_affine_residual_qmv,
    _native_affine_residual_qmv,
    _read_affine_residual_config,
    _write_affine_residual_config,
)
from metile.backends.mlx_quantized.swiglu import (
    _AFFINE_SWIGLU_CONFIGS,
    MLXAffineSwiGLUConfig,
    _affine_swiglu_compatible,
    _affine_swiglu_configs,
    _affine_swiglu_dispatch,
    _choose_affine_swiglu_config,
    _compile_affine_swiglu_qmv,
    _compile_affine_swiglu_scratch_qmv,
    _compile_nax_affine_swiglu_qmv,
    _make_affine_swiglu_executor,
    _mlx_compiled_affine_swiglu,
    _native_affine_swiglu,
    _read_affine_swiglu_config,
    _write_affine_swiglu_config,
)
from metile.compiler.affine_quantized import lower_affine_swiglu_qmv
from metile.kernels.affine_qmv import (
    affine_residual_qmv,
    affine_swiglu_qmv,
    affine_swiglu_scratch_qmv,
)
from metile.runtime.cache import stable_digest
from metile.tuning import confirm_pairwise, round_robin


def _tune_affine_dispatches(configs, make_dispatch, choose_config):
    import mlx.core as mx

    kernels = []
    for config in configs:
        try:
            dispatch, description_bits = make_dispatch(config)
            result = dispatch()
            mx.eval(result)
        except (RuntimeError, TypeError, ValueError):
            if config.algorithm == "mlx":
                raise
            continue
        kernels.append((config, dispatch, description_bits))

    native_dispatch = next(dispatch for config, dispatch, _ in kernels if config.algorithm == "mlx")
    reference = native_dispatch()
    mx.eval(reference)
    compatible = []
    for config, dispatch, description_bits in kernels:
        result = dispatch()
        mx.eval(result)
        if config.algorithm == "mlx" or _affine_swiglu_compatible(result, reference):
            compatible.append((config, dispatch, description_bits))
    kernels = compatible

    # One eval per batch rather than per dispatch: the blocking round trip costs roughly
    # 200 us whatever the kernel does, so evaluating per dispatch adds that constant to
    # every candidate and compresses their ratios toward 1.0, letting the switch margins
    # admit a kernel that is actually slower than native MLX.
    batch = calibrate_tournament_batch(native_dispatch)
    measure = batched_measure(batch)
    samples = round_robin(kernels, 11, measure)

    provisional = {
        config: statistics.median(config_samples) for config, config_samples in samples.items()
    }
    configs = tuple(config for config, _, _ in kernels)
    native = next(config for config in configs if config.algorithm == "mlx")
    alternatives = tuple(config for config in configs if config.algorithm != "mlx")
    if not alternatives:
        return native
    fastest_alternative = min(alternatives, key=provisional.__getitem__)
    best = min(provisional.values())
    finalists = {
        config
        for config, latency in provisional.items()
        if latency <= best * 1.10 or config in {native, fastest_alternative}
    }
    finalist_kernels = [candidate for candidate in kernels if candidate[0] in finalists]
    # Confirm head to head rather than trusting the crowded rotation. A candidate's measured
    # time depends on how many others share the round-robin, and the mlx_compiled variant in
    # particular reads faster in the tournament than it runs afterwards: at one row it was
    # being selected and then measuring 0.93x of plain native MLX.
    timings = confirm_pairwise(finalist_kernels, native, 31, measure)
    return choose_config(
        [
            (timings[config], description_bits, config)
            for config, _, description_bits in finalist_kernels
            if config in timings
        ]
    )


def _tune_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    return _tune_affine_dispatches(
        _affine_swiglu_configs(values.dtype, bits),
        lambda config: _affine_swiglu_dispatch(
            config,
            values,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            group_size,
            bits,
        ),
        _choose_affine_swiglu_config,
    )


def _tune_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    group_size,
    bits,
):
    return _tune_affine_dispatches(
        _AFFINE_RESIDUAL_CONFIGS,
        lambda config: _affine_residual_dispatch(
            config,
            values,
            weight,
            scales,
            biases,
            residual,
            group_size,
            bits,
        ),
        _choose_affine_residual_config,
    )


def mlx_affine_swiglu_backend_signature():
    """Return the code/config identity that can change affine SwiGLU dispatch."""
    return stable_digest(
        {
            "compiled": inspect.getsource(_mlx_compiled_affine_swiglu),
            "compiled_switch_margin": _COMPILED_SWITCH_MARGIN,
            "configs": [vars(config) for config in _AFFINE_SWIGLU_CONFIGS],
            "config_filter": inspect.getsource(_affine_swiglu_configs),
            "dispatch": inspect.getsource(mlx_affine_swiglu),
            "executor": inspect.getsource(mlx_affine_swiglu_executor),
            "fidelity": inspect.getsource(_affine_swiglu_compatible),
            "lowering": inspect.getsource(lower_affine_swiglu_qmv),
            "native": inspect.getsource(_native_affine_swiglu),
            "nax": inspect.getsource(_compile_nax_affine_swiglu_qmv),
            "residual_compiled": inspect.getsource(_mlx_compiled_affine_residual_qmv),
            "residual_configs": [vars(config) for config in _AFFINE_RESIDUAL_CONFIGS],
            "residual_dispatch": inspect.getsource(_affine_residual_dispatch),
            "residual_kernel": inspect.getsource(affine_residual_qmv.fn),
            "residual_margin": _RESIDUAL_SWITCH_MARGIN,
            "residual_native": inspect.getsource(_native_affine_residual_qmv),
            "residual_selection": inspect.getsource(_choose_affine_residual_config),
            "residual_tune": inspect.getsource(_tune_affine_residual_qmv),
            "residual_tuner": _AFFINE_RESIDUAL_TUNER_VERSION,
            "runtime_executor": inspect.getsource(mlx_affine_mlp_executor),
            "scalar": inspect.getsource(affine_swiglu_qmv.fn),
            "scalar_compile": inspect.getsource(_compile_affine_swiglu_qmv),
            "scratch": inspect.getsource(affine_swiglu_scratch_qmv.fn),
            "scratch_compile": inspect.getsource(_compile_affine_swiglu_scratch_qmv),
            "selection": inspect.getsource(_choose_affine_swiglu_config),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_affine_swiglu),
            "tuning_measure": inspect.getsource(_tune_affine_dispatches),
            "tuner": _AFFINE_SWIGLU_TUNER_VERSION,
        }
    )


def _affine_swiglu_persistent_key(values, gate_weight, group_size, bits):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "bits": bits,
            "configs": [vars(config) for config in _affine_swiglu_configs(values.dtype, bits)],
            "dtype": str(values.dtype),
            "group_size": group_size,
            "input_features": values.shape[-1],
            "mlx": mx.__version__,
            "output_features": gate_weight.shape[0],
            "rows": _token_bucket(values.size // values.shape[-1]),
            "source": mlx_affine_swiglu_backend_signature(),
            "compiled_switch_margin": _COMPILED_SWITCH_MARGIN,
            "switch_margin": _SWITCH_MARGIN,
            "tuner": _AFFINE_SWIGLU_TUNER_VERSION,
        }
    )


def _affine_residual_persistent_key(values, weight, group_size, bits):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "bits": bits,
            "configs": [vars(config) for config in _AFFINE_RESIDUAL_CONFIGS],
            "dtype": str(values.dtype),
            "group_size": group_size,
            "input_features": values.shape[-1],
            "mlx": mx.__version__,
            "output_features": weight.shape[0],
            "source": mlx_affine_swiglu_backend_signature(),
            "switch_margin": _RESIDUAL_SWITCH_MARGIN,
            "tuner": _AFFINE_RESIDUAL_TUNER_VERSION,
        }
    )


def mlx_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    *,
    group_size=64,
    bits=4,
    autotune=True,
):
    """Dispatch affine QMV plus residual to the fastest compatible kernel."""
    if group_size != 64 or bits != 4:
        raise ValueError("affine residual QMV requires 4-bit weights with group size 64")
    if biases is None:
        raise ValueError("affine residual QMV requires affine quantization biases")
    input_features = values.shape[-1]
    output_features = weight.shape[0]
    parameter_shape = (output_features, input_features // group_size)
    if (
        weight.ndim != 2
        or weight.shape[1] * 32 // bits != input_features
        or scales.shape != parameter_shape
        or biases.shape != parameter_shape
    ):
        raise ValueError("affine residual QMV received incompatible quantization parameters")
    if scales.dtype != values.dtype or biases.dtype != values.dtype:
        raise ValueError("affine residual QMV parameters must match the input dtype")
    expected_shape = (*values.shape[:-1], output_features)
    if residual.shape != expected_shape or residual.dtype != values.dtype:
        raise ValueError("residual must match the affine QMV output shape and dtype")
    if values.size != values.shape[-1]:
        return _native_affine_residual_qmv(
            values,
            weight,
            scales,
            biases,
            residual,
            group_size,
            bits,
        )
    schedule_key = (
        values.shape[-1],
        weight.shape[0],
        str(values.dtype),
        group_size,
        bits,
    )
    selected = _affine_residual_schedule_cache.get(schedule_key)
    if selected is None:
        with _affine_cache_lock:
            selected = _affine_residual_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _affine_residual_persistent_key(
                    values,
                    weight,
                    group_size,
                    bits,
                )
                selected = _read_affine_residual_config(persistent_key)
            if selected is None:
                selected = (
                    _tune_affine_residual_qmv(
                        values,
                        weight,
                        scales,
                        biases,
                        residual,
                        group_size,
                        bits,
                    )
                    if autotune
                    else MLXAffineResidualConfig("metile", 64)
                )
                _write_affine_residual_config(persistent_key, selected)
            _affine_residual_schedule_cache[schedule_key] = selected
    dispatch, _ = _affine_residual_dispatch(
        selected,
        values,
        weight,
        scales,
        biases,
        residual,
        group_size,
        bits,
    )
    return dispatch()


def mlx_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    *,
    group_size=64,
    bits=4,
    autotune=True,
):
    """Dispatch affine SwiGLU to eager/compiled MLX, scalar, or M5 NAX kernels."""
    if not ((bits == 4 and group_size == 64) or (bits == 8 and group_size in {32, 64, 128})):
        raise ValueError("affine SwiGLU requires 4-bit group-64 or 8-bit group-32/64/128 weights")
    if gate_biases is None or up_biases is None:
        raise ValueError("affine SwiGLU requires affine quantization biases")
    if (
        values.dtype != gate_scales.dtype
        or values.dtype != gate_biases.dtype
        or gate_weight.shape != up_weight.shape
        or gate_scales.shape != up_scales.shape
        or gate_biases.shape != up_biases.shape
    ):
        raise ValueError("affine SwiGLU requires matching gate/up weights and parameters")
    schedule_key = (
        _token_bucket(values.size // values.shape[-1]),
        values.shape[-1],
        gate_weight.shape[0],
        str(values.dtype),
        group_size,
        bits,
    )
    selected = _affine_swiglu_schedule_cache.get(schedule_key)
    if selected is None:
        with _affine_cache_lock:
            selected = _affine_swiglu_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _affine_swiglu_persistent_key(
                    values, gate_weight, group_size, bits
                )
                selected = _read_affine_swiglu_config(persistent_key)
            if selected is None:
                selected = (
                    _tune_affine_swiglu(
                        values,
                        gate_weight,
                        gate_scales,
                        gate_biases,
                        up_weight,
                        up_scales,
                        up_biases,
                        group_size,
                        bits,
                    )
                    if autotune
                    else MLXAffineSwiGLUConfig("metile", "scalar", 32)
                )
                _write_affine_swiglu_config(persistent_key, selected)
            _affine_swiglu_schedule_cache[schedule_key] = selected
    if selected.implementation not in {"nax", "nax_scratch"}:
        _discard_repacked_affine_pair(gate_weight, up_weight)
    dispatch, _ = _affine_swiglu_dispatch(
        selected,
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )
    return dispatch()


def mlx_affine_swiglu_executor(
    sample_values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    *,
    group_size=64,
    bits=4,
):
    """Autotune once and return the selected shape-specialized SwiGLU callable."""
    mlx_affine_swiglu(
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
    )
    schedule_key = (
        _token_bucket(sample_values.size // sample_values.shape[-1]),
        sample_values.shape[-1],
        gate_weight.shape[0],
        str(sample_values.dtype),
        group_size,
        bits,
    )
    executor, _ = _make_affine_swiglu_executor(
        _affine_swiglu_schedule_cache[schedule_key],
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )
    return executor


def mlx_affine_mlp_executor(
    sample_values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    down_weight,
    down_scales,
    down_biases,
    sample_residual,
    *,
    group_size=64,
    bits=4,
):
    """Autotune once and return a shape-specialized affine MLP callable."""
    swiglu_executor = mlx_affine_swiglu_executor(
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
    )
    hidden = swiglu_executor(sample_values)
    residual_key = (
        hidden.shape[-1],
        down_weight.shape[0],
        str(hidden.dtype),
        group_size,
        bits,
    )
    if residual_key not in _affine_residual_schedule_cache:
        import mlx.core as mx

        mx.eval(hidden)
    mlx_affine_residual_qmv(
        hidden,
        down_weight,
        down_scales,
        down_biases,
        sample_residual,
        group_size=group_size,
        bits=bits,
    )
    residual_executor, _ = _make_affine_residual_executor(
        _affine_residual_schedule_cache[residual_key],
        hidden,
        down_weight,
        down_scales,
        down_biases,
        group_size,
        bits,
    )

    def execute(values, residual):
        return residual_executor(swiglu_executor(values), residual)

    return execute
