from __future__ import annotations

import inspect
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass

from metile.backends.mlx import (
    _mlx_attention_decode_unchecked,
    mlx_add_rms_norm_dispatches,
    mlx_add_rms_norm_selection,
    mlx_attention_dispatches,
    mlx_rms_norm,
    mlx_rms_norm_dispatches,
)
from metile.backends.mlx_graph import compile_mlx_graph
from metile.backends.mlx_quantized import mlx_affine_swiglu, mlx_affine_swiglu_dispatches
from metile.compiler.schedule_search import choose_mdl_tie
from metile.ir.graph_ir import GraphBuilder, TensorSpec
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_graph_executable_cache = {}
_mlx_lm_plan_cache = {}
_mlx_lm_plan_lock = threading.RLock()
_mlx_lm_plan_cache_path = cache_root() / "mlx-lm-plan-autotune-v3.json"
_MODEL_SWITCH_MARGIN = 0.01
_MODEL_REGRESSION_MARGIN = 0.005


@dataclass(frozen=True)
class MLXLMPlan:
    """A measured MLX-LM feature combination."""

    attention: bool = True
    rms_norm: bool = True
    graph_fusion: bool = True
    quantized_mlp: bool = True

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
            setattr(module, name, original)
        self.replacements.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


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
            )
        )
    return tuple(
        sorted(candidates, key=lambda plan: (plan.feature_count, tuple(vars(plan).values())))
    )


def _effective_mlx_lm_plan(plan):
    return MLXLMPlan(
        attention=plan.attention
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_attention_dispatches()),
        rms_norm=plan.rms_norm
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_rms_norm_dispatches()),
        graph_fusion=plan.graph_fusion
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_add_rms_norm_dispatches()),
        quantized_mlp=plan.quantized_mlp
        and any(dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_swiglu_dispatches()),
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


def _mlx_lm_plan_key(model, sample_tokens, requested, decode_steps, trials):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt_bucket": 1 << max(sample_tokens.shape[1] - 1, 0).bit_length(),
            "requested": requested.as_dict(),
            "source": stable_digest(
                {
                    "apply": inspect.getsource(apply_metile_to_mlx_lm),
                    "timing": inspect.getsource(_time_mlx_lm_plan),
                }
            ),
            "regression_margin": _MODEL_REGRESSION_MARGIN,
            "switch_margin": _MODEL_SWITCH_MARGIN,
            "trials": trials,
            "tuner": 3,
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


def _time_mlx_lm_plan(model, sample_tokens, plan, decode_steps):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    decode_token = sample_tokens[:, -1:]
    with apply_metile_to_mlx_lm(model=model, plan=plan):
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


def _measure_mlx_lm_plans(model, sample_tokens, candidates, decode_steps, rounds):
    samples = {plan: [] for plan in candidates}
    expected_token = None
    compatible = set(candidates)
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
        required_wins = max(1, (len(total_ratios) * 4 + 4) // 5)
        if (
            statistics.median(ttft_ratios) <= 1.0 + _MODEL_REGRESSION_MARGIN
            and statistics.median(decode_ratios) <= 1.0 + _MODEL_REGRESSION_MARGIN
            and statistics.median(total_ratios) < 1.0 - _MODEL_SWITCH_MARGIN
            and sum(ratio <= 1.01 for ratio in ttft_ratios) >= required_wins
            and sum(ratio <= 1.01 for ratio in decode_ratios) >= required_wins
            and sum(ratio < 1.0 for ratio in total_ratios) >= required_wins
        ):
            generated.append((statistics.median(total_ratios), plan.feature_count * 64, plan))
    return choose_mdl_tie(generated) if generated else native


def _median_plan_measurement(measurements):
    return tuple(statistics.median(values) for values in zip(*measurements))


def _paired_plan_ratios(measurements, native_measurements, metric):
    return tuple(
        measurement[metric] / native[metric]
        for measurement, native in zip(measurements, native_measurements)
    )


def autotune_metile_for_mlx_lm(
    model,
    sample_tokens,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
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
    requested = MLXLMPlan(attention, rms_norm, graph_fusion, quantized_mlp)
    key = _mlx_lm_plan_key(model, sample_tokens, requested, decode_steps, trials)
    with _mlx_lm_plan_lock:
        cached = _read_mlx_lm_plan(key)
        if cached is not None:
            return cached

        candidates = _mlx_lm_plan_candidates(requested)
        _measure_mlx_lm_plans(model, sample_tokens, candidates, decode_steps, 1)
        candidates = tuple(dict.fromkeys(_effective_mlx_lm_plan(plan) for plan in candidates))
        provisional = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            candidates,
            decode_steps,
            3,
        )
        native = MLXLMPlan(False, False, False, False)
        provisional_totals = {
            plan: statistics.median(_paired_plan_ratios(samples, provisional[native], 2))
            for plan, samples in provisional.items()
        }
        best = min(provisional_totals.values())
        fastest = min(provisional_totals, key=provisional_totals.__getitem__)
        finalists = tuple(
            plan
            for plan in candidates
            if plan in provisional_totals
            and (provisional_totals[plan] <= best * 1.03 or plan in {native, fastest})
        )
        measured = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            finalists,
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


def _patch_quantized_mlp(model, replacements, quantized_linear):
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


def apply_metile_to_mlx_lm(
    model=None,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
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
    if not attention and not rms_norm and not graph_fusion and not quantized_mlp:
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
        _patch_quantized_mlp(model, replacements, nn.QuantizedLinear)

    return MLXPatch(replacements, attention_replacement, attention_original)


__all__ = [
    "MLXLMPlan",
    "MLXPatch",
    "apply_metile_to_mlx_lm",
    "autotune_metile_for_mlx_lm",
]
