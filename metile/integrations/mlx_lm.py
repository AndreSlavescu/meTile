from __future__ import annotations

import inspect
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

from metile.backends.mlx import (
    _mlx_attention_decode_unchecked,
    mlx_add_rms_norm_dispatches,
    mlx_add_rms_norm_selection,
    mlx_attention_dispatches,
    mlx_rms_norm,
    mlx_rms_norm_dispatches,
)
from metile.backends.mlx_affine import (
    MLXAffineWeight,
    mlx_affine_backend_signature,
    mlx_affine_matmul,
    mlx_affine_matmul_dispatches,
)
from metile.backends.mlx_graph import compile_mlx_graph
from metile.backends.mlx_quantized import (
    mlx_affine_swiglu,
    mlx_affine_swiglu_backend_signature,
    mlx_affine_swiglu_dispatches,
)
from metile.compiler.schedule_search import choose_mdl_tie
from metile.ir.graph_ir import GraphBuilder, TensorSpec
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_graph_executable_cache = {}
_mlx_lm_plan_cache = {}
_mlx_lm_plan_lock = threading.RLock()
_mlx_lm_plan_cache_path = cache_root() / "mlx-lm-plan-autotune-v10.json"
_MODEL_SWITCH_MARGIN = 0.01
_MODEL_REGRESSION_MARGIN = 0.005
_MODEL_KL_LIMIT = 1e-3
_MODEL_MEAN_LOGIT_ERROR_LIMIT = 0.02
_MODEL_MAX_LOGIT_ERROR_LIMIT = 0.25
_QUANTIZED_MLP_MIN_ROWS = 32


@dataclass(frozen=True)
class MLXLMPlan:
    """A measured MLX-LM feature combination."""

    attention: bool = True
    rms_norm: bool = True
    graph_fusion: bool = True
    quantized_mlp: bool = True
    affine_prefill: bool = False

    @property
    def feature_count(self):
        return sum(vars(self).values())

    def as_dict(self):
        return dict(vars(self))


@dataclass
class MLXPatch:
    """A reversible set of MLX-LM module patches."""

    replacements: list[tuple[object, str, object]]
    replacement: object | None = None
    original: object | None = None

    def restore(self):
        if self.replacement is not None:
            for module in tuple(sys.modules.values()):
                if (
                    module is not None
                    and getattr(module, "__name__", "").startswith("mlx_lm.models")
                    and getattr(module, "scaled_dot_product_attention", None) is self.replacement
                ):
                    module.scaled_dot_product_attention = self.original
        for module, name, original in reversed(self.replacements):
            if name == "__class__":
                object.__setattr__(module, name, original)
            else:
                setattr(module, name, original)
        self.replacements.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


@dataclass
class MLXAffinePrefill:
    """AOT-repacked affine projections for one MLX-LM model."""

    model: object
    weights: dict[int, tuple[object, MLXAffineWeight]]
    min_rows: int = 32
    patched_classes: dict[int, type] = field(default_factory=dict)

    def weight_for(self, module):
        entry = self.weights.get(id(module))
        return entry[1] if entry is not None and entry[0] is module else None

    @property
    def projection_count(self):
        return len(self.weights)

    def patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(module)
        original_call = original_class.__call__
        weight = self.weight_for(module)
        min_rows = self.min_rows

        class MLXAffinePrefillLinear(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows < min_rows:
                    object.__setattr__(self, "__class__", original_class)
                    return original_call(self, values)
                output = mlx_affine_matmul(values, weight)
                if "bias" in self:
                    output = output + self["bias"]
                return output

        MLXAffinePrefillLinear.__name__ = f"MeTile{original_class.__name__}"
        MLXAffinePrefillLinear.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXAffinePrefillLinear
        return MLXAffinePrefillLinear


def _mlx_lm_plan_candidates(requested):
    names = tuple(name for name, enabled in requested.as_dict().items() if enabled)
    candidates = []
    for mask in range(1 << len(names)):
        enabled = {name for index, name in enumerate(names) if mask & (1 << index)}
        candidates.append(
            MLXLMPlan(
                attention="attention" in enabled,
                rms_norm="rms_norm" in enabled,
                graph_fusion="graph_fusion" in enabled,
                quantized_mlp="quantized_mlp" in enabled,
                affine_prefill="affine_prefill" in enabled,
            )
        )
    return tuple(
        sorted(candidates, key=lambda plan: (plan.feature_count, tuple(vars(plan).values())))
    )


def _effective_mlx_lm_plan(plan, affine_prefill=None):
    return MLXLMPlan(
        attention=plan.attention
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_attention_dispatches()),
        rms_norm=plan.rms_norm
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_rms_norm_dispatches()),
        graph_fusion=plan.graph_fusion
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_add_rms_norm_dispatches()),
        quantized_mlp=plan.quantized_mlp
        and any(dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_swiglu_dispatches()),
        affine_prefill=plan.affine_prefill
        and affine_prefill is not None
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_affine_matmul_dispatches()),
    )


def _mlx_lm_model_signature(model):
    layers = tuple(_model_layers(model))
    first_layer = layers[0] if layers else None
    attention = getattr(first_layer, "self_attn", None)
    norm = getattr(first_layer, "input_layernorm", None)
    weight = getattr(norm, "weight", None)
    return {
        "attention_class": (
            f"{type(attention).__module__}.{type(attention).__qualname__}"
            if attention is not None
            else None
        ),
        "head_dim": getattr(attention, "head_dim", None),
        "hidden": weight.shape[0] if weight is not None else None,
        "layers": len(layers),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "n_heads": getattr(attention, "n_heads", None),
        "n_kv_heads": getattr(attention, "n_kv_heads", None),
        "scale": getattr(attention, "scale", None),
    }


def _mlx_lm_plan_key(model, sample_tokens, requested, affine_prefill, decode_steps, trials):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "affine_prefill": (
                {
                    "min_rows": affine_prefill.min_rows,
                    "projections": affine_prefill.projection_count,
                }
                if affine_prefill is not None
                else None
            ),
            "decode_steps": decode_steps,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt_bucket": 1 << max(sample_tokens.shape[1] - 1, 0).bit_length(),
            "requested": requested.as_dict(),
            "source": stable_digest(
                {
                    "apply": inspect.getsource(apply_metile_to_mlx_lm),
                    "affine_backend": mlx_affine_backend_signature(),
                    "affine_prefill_class": inspect.getsource(MLXAffinePrefill.patched_class),
                    "affine_prefill": inspect.getsource(_patch_affine_prefill),
                    "affine_swiglu_backend": mlx_affine_swiglu_backend_signature(),
                    "choose": inspect.getsource(_choose_mlx_lm_plan),
                    "effective": inspect.getsource(_effective_mlx_lm_plan),
                    "fidelity": inspect.getsource(_plan_preserves_logits),
                    "finalists": inspect.getsource(_provisional_mlx_lm_finalists),
                    "quantized_mlp_patch": inspect.getsource(_patch_quantized_mlp),
                    "timing": inspect.getsource(_time_mlx_lm_plan),
                }
            ),
            "regression_margin": _MODEL_REGRESSION_MARGIN,
            "quantized_mlp_min_rows": _QUANTIZED_MLP_MIN_ROWS,
            "switch_margin": _MODEL_SWITCH_MARGIN,
            "trials": trials,
            "tuner": 10,
        }
    )


def _read_mlx_lm_plan(key):
    cached = _mlx_lm_plan_cache.get(key)
    if cached is not None:
        return cached
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_mlx_lm_plan_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    try:
        return MLXLMPlan(**{name: bool(payload[name]) for name in vars(MLXLMPlan())})
    except KeyError:
        return None


def _write_mlx_lm_plan(key, plan):
    _mlx_lm_plan_cache[key] = plan
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_mlx_lm_plan_cache_path, {})
    payload[key] = plan.as_dict()
    atomic_write_json(_mlx_lm_plan_cache_path, payload)


def _time_mlx_lm_plan(model, sample_tokens, plan, affine_prefill, decode_steps):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    decode_token = sample_tokens[:, -1:]
    with apply_metile_to_mlx_lm(model=model, plan=plan, affine_prefill=affine_prefill):
        total_start = time.perf_counter_ns()
        logits = model(sample_tokens, cache=cache)
        mx.eval(logits)
        ttft = (time.perf_counter_ns() - total_start) * 1e-9
        decode_start = time.perf_counter_ns()
        for _ in range(decode_steps):
            logits = model(decode_token, cache=cache)
            mx.eval(logits)
        decode = (time.perf_counter_ns() - decode_start) * 1e-9 / decode_steps
        total = (time.perf_counter_ns() - total_start) * 1e-9
    next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
    return (ttft, decode, total), next_token


def _logit_fidelity(reference, actual):
    import mlx.core as mx

    reference = reference[:, -1].astype(mx.float32)
    actual = actual[:, -1].astype(mx.float32)
    difference = mx.abs(reference - actual)
    reference_log_probs = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    actual_log_probs = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
    divergence = mx.sum(
        mx.exp(reference_log_probs) * (reference_log_probs - actual_log_probs),
        axis=-1,
    )
    mx.eval(difference, divergence)
    return {
        "next_token": int(mx.argmax(reference, axis=-1).item()),
        "actual_next_token": int(mx.argmax(actual, axis=-1).item()),
        "kl_divergence": max(0.0, float(mx.max(divergence).item())),
        "mean_logit_error": float(mx.mean(difference).item()),
        "max_logit_error": float(mx.max(difference).item()),
    }


def _fidelity_compatible(fidelity):
    return (
        fidelity["next_token"] == fidelity["actual_next_token"]
        and fidelity["kl_divergence"] <= _MODEL_KL_LIMIT
        and fidelity["mean_logit_error"] <= _MODEL_MEAN_LOGIT_ERROR_LIMIT
        and fidelity["max_logit_error"] <= _MODEL_MAX_LOGIT_ERROR_LIMIT
    )


def _plan_preserves_logits(model, sample_tokens, plan, affine_prefill, reference):
    import mlx.core as mx

    with apply_metile_to_mlx_lm(model=model, plan=plan, affine_prefill=affine_prefill):
        actual = model(sample_tokens)
        mx.eval(actual)
    fidelity = _logit_fidelity(reference, actual)
    return _fidelity_compatible(fidelity)


def _measure_mlx_lm_plans(
    model,
    sample_tokens,
    candidates,
    affine_prefill,
    decode_steps,
    rounds,
):
    import mlx.core as mx

    samples = {plan: [] for plan in candidates}
    expected_token = None
    compatible = set(candidates)
    reference = model(sample_tokens)
    mx.eval(reference)
    for plan in candidates:
        if not plan.feature_count:
            continue
        try:
            if not _plan_preserves_logits(model, sample_tokens, plan, affine_prefill, reference):
                compatible.remove(plan)
        except (RuntimeError, TypeError, ValueError):
            compatible.remove(plan)
    for round_index in range(rounds):
        shift = round_index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if round_index & 1:
            ordered = tuple(reversed(ordered))
        for plan in ordered:
            if plan not in compatible:
                continue
            try:
                measurement, next_token = _time_mlx_lm_plan(
                    model,
                    sample_tokens,
                    plan,
                    affine_prefill,
                    decode_steps,
                )
            except (RuntimeError, TypeError, ValueError):
                if plan.feature_count == 0:
                    raise
                compatible.remove(plan)
                continue
            if expected_token is None:
                expected_token = next_token
            if next_token != expected_token:
                compatible.remove(plan)
                continue
            samples[plan].append(measurement)
    return {plan: values for plan, values in samples.items() if values and plan in compatible}


def _choose_mlx_lm_plan(samples):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = samples[native]
    generated = []
    for plan, measurements in samples.items():
        if not plan.feature_count:
            continue
        ttft_ratios = _paired_plan_ratios(measurements, native_measurements, 0)
        decode_ratios = _paired_plan_ratios(measurements, native_measurements, 1)
        total_ratios = _paired_plan_ratios(measurements, native_measurements, 2)
        required_wins = max(1, (len(total_ratios) * 2 + 2) // 3)
        ttft_median = statistics.median(ttft_ratios)
        decode_median = statistics.median(decode_ratios)
        total_median = statistics.median(total_ratios)
        improves_total = total_median < 1.0 - _MODEL_SWITCH_MARGIN
        improves_ttft = ttft_median < 0.97
        decode_sensitive = any(
            (plan.attention, plan.rms_norm, plan.graph_fusion, plan.quantized_mlp)
        )
        decode_limit = 1.0 + (_MODEL_REGRESSION_MARGIN if decode_sensitive else 0.05)
        if (
            ttft_median <= 1.0 + _MODEL_REGRESSION_MARGIN
            and decode_median <= decode_limit
            and total_median <= 1.0 + _MODEL_REGRESSION_MARGIN
            and (improves_total or improves_ttft)
            and sum(ratio <= 1.01 for ratio in ttft_ratios) >= required_wins
            and sum(ratio <= 1.05 for ratio in decode_ratios) >= required_wins
            and (
                sum(ratio < 1.0 for ratio in total_ratios) >= required_wins
                if improves_total
                else sum(ratio < 0.98 for ratio in ttft_ratios) >= required_wins
            )
        ):
            objective = min(total_median, ttft_median)
            generated.append((objective, plan.feature_count * 64, plan))
    return choose_mdl_tie(generated) if generated else native


def _median_plan_measurement(measurements):
    return tuple(statistics.median(values) for values in zip(*measurements))


def _paired_plan_ratios(measurements, native_measurements, metric):
    return tuple(
        measurement[metric] / native[metric]
        for measurement, native in zip(measurements, native_measurements)
    )


def _provisional_mlx_lm_finalists(provisional, candidates):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = provisional[native]
    total_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 2))
        for plan, samples in provisional.items()
    }
    ttft_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 0))
        for plan, samples in provisional.items()
    }
    best_total = min(total_ratios.values())
    best_ttft = min(ttft_ratios.values())
    fastest_total = min(total_ratios, key=total_ratios.__getitem__)
    fastest_ttft = min(ttft_ratios, key=ttft_ratios.__getitem__)
    required = {native, fastest_total, fastest_ttft}
    return tuple(
        plan
        for plan in candidates
        if plan in total_ratios
        and (
            total_ratios[plan] <= best_total * 1.03
            or ttft_ratios[plan] <= best_ttft * 1.03
            or plan in required
        )
    )


def autotune_metile_for_mlx_lm(
    model,
    sample_tokens,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
    affine_prefill=None,
    decode_steps=8,
    trials=5,
):
    """Choose a persistent feature plan by timing the real MLX-LM decode graph."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if sample_tokens.ndim != 2 or sample_tokens.shape[1] < 1:
        raise ValueError("sample_tokens must have shape [batch, sequence]")
    if decode_steps < 1 or trials < 1:
        raise ValueError("decode_steps and trials must be positive")
    if affine_prefill is not None and not isinstance(affine_prefill, MLXAffinePrefill):
        raise TypeError("affine_prefill must be an MLXAffinePrefill")
    if affine_prefill is not None and affine_prefill.model is not model:
        raise ValueError("affine_prefill was prepared for a different model")
    requested = MLXLMPlan(
        attention,
        rms_norm,
        graph_fusion,
        quantized_mlp,
        affine_prefill is not None,
    )
    key = _mlx_lm_plan_key(
        model,
        sample_tokens,
        requested,
        affine_prefill,
        decode_steps,
        trials,
    )
    with _mlx_lm_plan_lock:
        cached = _read_mlx_lm_plan(key)
        if cached is not None:
            return cached

        candidates = _mlx_lm_plan_candidates(requested)
        _measure_mlx_lm_plans(
            model,
            sample_tokens,
            candidates,
            affine_prefill,
            decode_steps,
            1,
        )
        candidates = tuple(
            dict.fromkeys(_effective_mlx_lm_plan(plan, affine_prefill) for plan in candidates)
        )
        provisional = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            candidates,
            affine_prefill,
            decode_steps,
            3,
        )
        finalists = _provisional_mlx_lm_finalists(provisional, candidates)
        measured = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            finalists,
            affine_prefill,
            decode_steps,
            trials,
        )
        selected = _choose_mlx_lm_plan(measured)
        _write_mlx_lm_plan(key, selected)
        return selected


def _supports_metile_decode(queries, keys, values, cache, mask, sinks):
    return (
        not hasattr(cache, "bits")
        and mask is None
        and sinks is None
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[2] == 1
        and keys.shape == values.shape
        and queries.shape[0] == keys.shape[0]
        and queries.shape[1] % keys.shape[1] == 0
        and queries.shape[-1] == keys.shape[-1]
        and queries.shape[-1] % 32 == 0
        and queries.dtype == keys.dtype == values.dtype
        and str(queries.dtype) in ("mlx.core.float16", "mlx.core.float32")
    )


def _supports_metile_rms_norm(values, weight):
    return (
        values.ndim >= 1
        and weight.ndim == 1
        and values.shape[-1] == weight.shape[0]
        and values.dtype == weight.dtype
        and str(values.dtype) in ("mlx.core.float16", "mlx.core.float32")
    )


def _supports_metile_residual_rms_norm(values, residual, norm):
    weight = norm["weight"]
    return (
        values.shape == residual.shape
        and values.dtype == residual.dtype
        and _supports_metile_rms_norm(values, weight)
    )


def _tensor_spec(value):
    dtype = {
        "mlx.core.float16": "f16",
        "mlx.core.float32": "f32",
    }[str(value.dtype)]
    return TensorSpec(tuple(value.shape), dtype)


def _execute_residual_rms_graph(values, residual, norm):
    weight = norm["weight"]
    key = (tuple(values.shape), str(values.dtype), float(norm.eps))
    executable = _graph_executable_cache.get(key)
    if executable is None:
        builder = GraphBuilder()
        values_input = builder.input("values", _tensor_spec(values))
        residual_input = builder.input("residual", _tensor_spec(residual))
        weight_input = builder.input("weight", _tensor_spec(weight))
        summed = builder.add(values_input, residual_input, name="residual_add")
        normalized = builder.rms_norm(
            summed,
            weight_input,
            norm.eps,
            name="post_attention_rms_norm",
        )
        executable = compile_mlx_graph(builder.build((summed, normalized)))
        _graph_executable_cache[key] = executable
    return executable(values, residual, weight)


def _patch_graph_fusion(model, replacements):
    classes = []
    if model is not None:
        classes.extend(type(layer) for layer in _model_layers(model))
    else:
        from mlx_lm.models import llama

        classes.append(llama.TransformerBlock)

    for block_class in dict.fromkeys(classes):
        if (
            block_class.__module__ != "mlx_lm.models.llama"
            or block_class.__name__ != "TransformerBlock"
        ):
            continue
        original = block_class.__call__
        if getattr(original, "_metile_original", None) is not None:
            continue

        def make_replacement(original_call):
            def replacement(self, values, mask=None, cache=None):
                selected = mlx_add_rms_norm_selection(values, self.post_attention_layernorm.eps)
                if selected is not None and selected.algorithm == "mlx":
                    return original_call(self, values, mask, cache)

                attention_output = self.self_attn(self.input_layernorm(values), mask, cache)
                if _supports_metile_residual_rms_norm(
                    values, attention_output, self.post_attention_layernorm
                ):
                    hidden, normalized = _execute_residual_rms_graph(
                        values, attention_output, self.post_attention_layernorm
                    )
                else:
                    hidden = values + attention_output
                    normalized = self.post_attention_layernorm(hidden)
                return hidden + self.mlp(normalized)

            return replacement

        metile_transformer_block = make_replacement(original)

        metile_transformer_block._metile_original = original
        replacements.append((block_class, "__call__", original))
        block_class.__call__ = metile_transformer_block


def _model_layers(model):
    if model is None:
        return ()
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", ())
    return layers


def _supports_quantized_mlp(module, values, quantized_linear):
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    return (
        isinstance(gate, quantized_linear)
        and isinstance(up, quantized_linear)
        and gate.mode == up.mode == "affine"
        and gate.group_size == up.group_size == 64
        and gate.bits == up.bits == 4
        and gate.get("biases") is not None
        and up.get("biases") is not None
        and "bias" not in gate
        and "bias" not in up
        and str(values.dtype) == "mlx.core.float16"
    )


def _patch_quantized_mlp(model, replacements, quantized_linear, *, min_rows=1):
    if min_rows < 1:
        raise ValueError("quantized MLP minimum rows must be positive")
    classes = [type(layer.mlp) for layer in _model_layers(model) if hasattr(layer, "mlp")]
    if model is None:
        from mlx_lm.models import llama

        classes.append(llama.MLP)

    for mlp_class in dict.fromkeys(classes):
        if mlp_class.__module__ != "mlx_lm.models.llama" or mlp_class.__name__ != "MLP":
            continue
        original = mlp_class.__call__
        if getattr(original, "_metile_original", None) is not None:
            continue

        def make_replacement(original_call):
            def replacement(self, values):
                rows = values.size // values.shape[-1]
                if rows < min_rows:
                    type(self).__call__ = original_call
                    return original_call(self, values)
                if not _supports_quantized_mlp(self, values, quantized_linear):
                    return original_call(self, values)
                gate = self.gate_proj
                up = self.up_proj
                hidden = mlx_affine_swiglu(
                    values,
                    gate["weight"],
                    gate["scales"],
                    gate.get("biases"),
                    up["weight"],
                    up["scales"],
                    up.get("biases"),
                    group_size=gate.group_size,
                    bits=gate.bits,
                )
                return self.down_proj(hidden)

            return replacement

        metile_mlp = make_replacement(original)
        metile_mlp._metile_original = original
        replacements.append((mlp_class, "__call__", original))
        mlp_class.__call__ = metile_mlp


def _patch_affine_prefill(affine_prefill, replacements):
    if affine_prefill is None:
        return
    for module, _ in affine_prefill.weights.values():
        patched_class = affine_prefill.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", affine_prefill.patched_class(module))


def prepare_mlx_lm_affine_prefill(
    model,
    *,
    projections=("down_proj",),
    min_rows=32,
):
    """AOT-repack exact affine weights for generated prefill matmuls."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if not projections or not all(isinstance(name, str) and name for name in projections):
        raise ValueError("projections must contain at least one attribute name")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    try:
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Affine prefill preparation requires the optional 'mlx' package"
        ) from error

    weights = {}
    for layer in _model_layers(model):
        mlp = getattr(layer, "mlp", None)
        for name in projections:
            module = getattr(mlp, name, None)
            if (
                not isinstance(module, nn.QuantizedLinear)
                or module.mode != "affine"
                or module.group_size != 64
                or module.bits != 4
                or module.get("biases") is None
                or module.weight.shape[0] % 32
            ):
                continue
            weight = MLXAffineWeight.from_mlx(
                module.weight,
                module.scales,
                module.biases,
                group_size=module.group_size,
                bits=module.bits,
            )
            weights[id(module)] = (module, weight)
    if not weights:
        raise ValueError("model contains no supported affine prefill projections")
    return MLXAffinePrefill(model, weights, min_rows)


def apply_metile_to_mlx_lm(
    model=None,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
    affine_prefill=None,
    plan=None,
):
    """Patch MLX-LM with zero-copy, autotuned meTile primitives.

    Decode attention, RMSNorm, quantized SwiGLU, and compute-graph fusion are independently
    selectable. Unsupported calls preserve MLX-LM's original implementation.
    The returned handle can restore every changed module or be used as a context
    manager.
    """
    if model is not None and not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if plan is not None:
        if not isinstance(plan, MLXLMPlan):
            raise TypeError("plan must be an MLXLMPlan")
        attention = attention and plan.attention
        rms_norm = rms_norm and plan.rms_norm
        graph_fusion = graph_fusion and plan.graph_fusion
        quantized_mlp = quantized_mlp and plan.quantized_mlp
        if not plan.affine_prefill:
            affine_prefill = None
    if affine_prefill is not None:
        if not isinstance(affine_prefill, MLXAffinePrefill):
            raise TypeError("affine_prefill must be an MLXAffinePrefill")
        if model is not affine_prefill.model:
            raise ValueError("affine_prefill was prepared for a different model")
    if (
        not attention
        and not rms_norm
        and not graph_fusion
        and not quantized_mlp
        and affine_prefill is None
    ):
        return MLXPatch([])
    try:
        import mlx.nn as nn
        from mlx_lm.models import base
    except ImportError as error:
        raise ImportError(
            "The MLX-LM integration requires the optional 'mlx-lm' package"
        ) from error

    replacements = []
    attention_replacement = None
    attention_original = None
    if attention:
        attention_original = base.scaled_dot_product_attention
        if getattr(attention_original, "_metile_original", None) is None:

            def metile_scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache,
                scale,
                mask,
                sinks=None,
            ):
                if _supports_metile_decode(queries, keys, values, cache, mask, sinks):
                    return _mlx_attention_decode_unchecked(
                        queries,
                        keys,
                        values,
                        scale,
                    )
                return attention_original(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )

            metile_scaled_dot_product_attention._metile_original = attention_original
            attention_replacement = metile_scaled_dot_product_attention
            for module in tuple(sys.modules.values()):
                if module is None or not getattr(module, "__name__", "").startswith(
                    "mlx_lm.models"
                ):
                    continue
                if getattr(module, "scaled_dot_product_attention", None) is attention_original:
                    replacements.append(
                        (module, "scaled_dot_product_attention", attention_original)
                    )
                    module.scaled_dot_product_attention = attention_replacement

    if rms_norm:
        original_rms_norm = nn.RMSNorm.__call__
        if getattr(original_rms_norm, "_metile_original", None) is None:

            def metile_rms_norm(self, values):
                weight = self["weight"]
                if _supports_metile_rms_norm(values, weight):
                    return mlx_rms_norm(values, weight, self.eps)
                return original_rms_norm(self, values)

            metile_rms_norm._metile_original = original_rms_norm
            replacements.append((nn.RMSNorm, "__call__", original_rms_norm))
            nn.RMSNorm.__call__ = metile_rms_norm

    if graph_fusion:
        _patch_graph_fusion(model, replacements)
    if quantized_mlp:
        min_rows = _QUANTIZED_MLP_MIN_ROWS if affine_prefill is not None else 1
        _patch_quantized_mlp(model, replacements, nn.QuantizedLinear, min_rows=min_rows)
    _patch_affine_prefill(affine_prefill, replacements)

    return MLXPatch(replacements, attention_replacement, attention_original)


__all__ = [
    "MLXAffinePrefill",
    "MLXLMPlan",
    "MLXPatch",
    "apply_metile_to_mlx_lm",
    "autotune_metile_for_mlx_lm",
    "prepare_mlx_lm_affine_prefill",
]
