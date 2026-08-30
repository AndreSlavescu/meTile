"""plan model layer of the mlx_lm integration."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from itertools import combinations

from metile.backends.mlx import (
    mlx_add_rms_norm_dispatches,
    mlx_attention_dispatches,
    mlx_rms_norm_dispatches,
)
from metile.backends.mlx_affine import (
    mlx_affine_matmul_dispatches,
)
from metile.backends.mlx_dense import (
    mlx_dense_matmul_dispatches,
)
from metile.backends.mlx_dense_residual import (
    mlx_dense_residual_dispatches,
)
from metile.backends.mlx_dense_swiglu import (
    mlx_dense_swiglu_dispatches,
)
from metile.backends.mlx_quantized import (
    mlx_affine_residual_qmv_dispatches,
    mlx_affine_swiglu_dispatches,
)
from metile.compiler.schedule_search import choose_mdl_tie
from metile.integrations.mlx_lm._state import (
    _MODEL_BF16_MAX_LOGIT_ERROR_LIMIT,
    _MODEL_BF16_MEAN_LOGIT_ERROR_LIMIT,
    _MODEL_DECODE_SWITCH_MARGIN,
    _MODEL_KL_LIMIT,
    _MODEL_MAX_LOGIT_ERROR_LIMIT,
    _MODEL_MEAN_LOGIT_ERROR_LIMIT,
    _MODEL_PROVISIONAL_MAX_FINALISTS,
    _MODEL_PROVISIONAL_RELATIVE_MARGIN,
    _MODEL_REGRESSION_MARGIN,
    _MODEL_STRONG_DECODE_SWITCH_MARGIN,
    _MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN,
    _MODEL_SWITCH_MARGIN,
    _MODEL_TTFT_SWITCH_MARGIN,
    _MODEL_VALIDATION_MAX_SURVIVORS,
    _mlx_lm_plan_cache,
    _mlx_lm_plan_cache_path,
)
from metile.runtime.cache import atomic_write_json, read_json


@dataclass(frozen=True)
class MLXLMPlan:
    """A measured MLX-LM feature combination."""

    attention: bool = True
    rms_norm: bool = True
    graph_fusion: bool = True
    quantized_mlp: bool = True
    affine_prefill: bool = False
    dense_mlp: bool = False
    dense_residual: bool = False
    compressed_down: bool = False
    compressed_gate_up: bool = False
    compressed_vocab: bool = False
    compressed_attention: bool = False

    @property
    def feature_count(self):
        return sum(vars(self).values())

    @property
    def is_decode_only_compression(self):
        return any(
            (
                self.compressed_down,
                self.compressed_gate_up,
                self.compressed_vocab,
                self.compressed_attention,
            )
        ) and not any(
            (
                self.attention,
                self.rms_norm,
                self.graph_fusion,
                self.quantized_mlp,
                self.affine_prefill,
                self.dense_mlp,
                self.dense_residual,
            )
        )

    def as_dict(self):
        return dict(vars(self))


def _mlx_lm_plan_candidates(requested):
    requested_names = tuple(name for name, enabled in requested.as_dict().items() if enabled)
    compression_names = tuple(
        name
        for name in (
            "compressed_down",
            "compressed_gate_up",
            "compressed_vocab",
            "compressed_attention",
        )
        if name in requested_names
    )
    structural_names = tuple(name for name in requested_names if name not in compression_names)
    enabled_sets = {
        frozenset(name for index, name in enumerate(compression_names) if mask & (1 << index))
        for mask in range(1 << len(compression_names))
    }
    maximum_structural_order = len(structural_names) if len(structural_names) <= 3 else 2
    structural_sets = {
        frozenset(names)
        for order in range(maximum_structural_order + 1)
        for names in combinations(structural_names, order)
    }
    full_compression = frozenset(compression_names)
    for structural in structural_sets:
        enabled_sets.add(structural)
        enabled_sets.add(structural | full_compression)
    enabled_sets.add(frozenset(requested_names))

    candidates = []
    for enabled in enabled_sets:
        if "compressed_down" in enabled and "dense_residual" in enabled:
            continue
        if "compressed_gate_up" in enabled and "dense_mlp" in enabled:
            continue
        candidates.append(
            MLXLMPlan(
                attention="attention" in enabled,
                rms_norm="rms_norm" in enabled,
                graph_fusion="graph_fusion" in enabled,
                quantized_mlp="quantized_mlp" in enabled,
                affine_prefill="affine_prefill" in enabled,
                dense_mlp="dense_mlp" in enabled,
                dense_residual="dense_residual" in enabled,
                compressed_down="compressed_down" in enabled,
                compressed_gate_up="compressed_gate_up" in enabled,
                compressed_vocab="compressed_vocab" in enabled,
                compressed_attention="compressed_attention" in enabled,
            )
        )
    return tuple(
        sorted(candidates, key=lambda plan: (plan.feature_count, tuple(vars(plan).values())))
    )


def _mlx_lm_warmup_plans(candidates):
    """Select the linear-sized plans needed to populate primitive dispatch caches."""
    compile_features = (
        "attention",
        "rms_norm",
        "graph_fusion",
        "quantized_mlp",
        "affine_prefill",
        "dense_mlp",
        "dense_residual",
    )
    available = {name for plan in candidates for name, enabled in plan.as_dict().items() if enabled}
    required = {frozenset()}
    required.update(frozenset((name,)) for name in compile_features if name in available)
    for interaction in (
        frozenset(("graph_fusion", "quantized_mlp")),
        frozenset(("dense_mlp", "dense_residual")),
    ):
        if interaction <= available:
            required.add(interaction)
    return tuple(
        plan
        for plan in candidates
        if frozenset(name for name, enabled in plan.as_dict().items() if enabled) in required
    )


def _effective_mlx_lm_plan(
    plan,
    affine_prefill=None,
    dense_mlp=None,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    return MLXLMPlan(
        attention=plan.attention
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_attention_dispatches()),
        rms_norm=plan.rms_norm
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_rms_norm_dispatches()),
        graph_fusion=plan.graph_fusion
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_add_rms_norm_dispatches()),
        quantized_mlp=plan.quantized_mlp
        and (
            any(dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_swiglu_dispatches())
            or any(
                dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_residual_qmv_dispatches()
            )
        ),
        affine_prefill=plan.affine_prefill
        and affine_prefill is not None
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_affine_matmul_dispatches()),
        dense_mlp=plan.dense_mlp
        and dense_mlp is not None
        and (
            (
                dense_mlp.implementation == "fused"
                and any(
                    dispatch["algorithm"] == "metile" for dispatch in mlx_dense_swiglu_dispatches()
                )
            )
            or (
                dense_mlp.implementation == "projected"
                and any(
                    dispatch["algorithm"] == "metile" for dispatch in mlx_dense_matmul_dispatches()
                )
            )
        ),
        dense_residual=plan.dense_residual
        and dense_mlp is not None
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_dense_residual_dispatches()),
        compressed_down=plan.compressed_down
        and compressed_down is not None
        and compressed_down.projection_count > 0,
        compressed_gate_up=plan.compressed_gate_up
        and compressed_gate_up is not None
        and compressed_gate_up.projection_count > 0,
        compressed_vocab=plan.compressed_vocab
        and compressed_vocab is not None
        and compressed_vocab.projection_count > 0,
        compressed_attention=plan.compressed_attention
        and compressed_attention is not None
        and compressed_attention.projection_count > 0,
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


def _logit_fidelity(reference, actual):
    import mlx.core as mx

    reference_dtype = str(reference.dtype)
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
        "reference_dtype": reference_dtype,
        "next_token": int(mx.argmax(reference, axis=-1).item()),
        "actual_next_token": int(mx.argmax(actual, axis=-1).item()),
        "kl_divergence": max(0.0, float(mx.max(divergence).item())),
        "mean_logit_error": float(mx.mean(difference).item()),
        "max_logit_error": float(mx.max(difference).item()),
    }


def _fidelity_compatible(fidelity):
    is_bfloat16 = fidelity.get("reference_dtype") == "mlx.core.bfloat16"
    mean_limit = (
        _MODEL_BF16_MEAN_LOGIT_ERROR_LIMIT if is_bfloat16 else _MODEL_MEAN_LOGIT_ERROR_LIMIT
    )
    maximum_limit = (
        _MODEL_BF16_MAX_LOGIT_ERROR_LIMIT if is_bfloat16 else _MODEL_MAX_LOGIT_ERROR_LIMIT
    )
    return (
        fidelity["next_token"] == fidelity["actual_next_token"]
        and fidelity["kl_divergence"] <= _MODEL_KL_LIMIT
        and fidelity["mean_logit_error"] <= mean_limit
        and fidelity["max_logit_error"] <= maximum_limit
    )


def _rank_mlx_lm_plans(samples):
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
        improves_decode = decode_median < 1.0 - _MODEL_DECODE_SWITCH_MARGIN
        improves_ttft = ttft_median < 1.0 - _MODEL_TTFT_SWITCH_MARGIN
        decode_only = _is_decode_only_compression_plan(plan)
        decode_sensitive = any(
            (
                plan.attention,
                plan.rms_norm,
                plan.graph_fusion,
                plan.quantized_mlp,
                plan.dense_mlp,
                plan.dense_residual,
                plan.compressed_down,
                plan.compressed_gate_up,
                plan.compressed_vocab,
                plan.compressed_attention,
            )
        )
        decode_limit = 1.0 + (_MODEL_REGRESSION_MARGIN if decode_sensitive else 0.05)
        strong_decode_win = (
            decode_sensitive
            and decode_median <= 1.0 - _MODEL_STRONG_DECODE_SWITCH_MARGIN
            and sum(ratio < 1.0 for ratio in total_ratios) >= required_wins
        )
        stable_ttft = decode_only or (
            sum(ratio <= 1.01 for ratio in ttft_ratios) >= required_wins or strong_decode_win
        )
        if strong_decode_win:
            ttft_margin = _MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN
        else:
            ttft_margin = _MODEL_REGRESSION_MARGIN
        ttft_limit = 1.0 + ttft_margin
        if (
            (decode_only or ttft_median <= ttft_limit)
            and decode_median <= decode_limit
            and total_median <= 1.0 + _MODEL_REGRESSION_MARGIN
            and (improves_total or improves_decode or improves_ttft)
            and stable_ttft
            and sum(ratio <= 1.05 for ratio in decode_ratios) >= required_wins
            and any(
                (
                    improves_total and sum(ratio < 1.0 for ratio in total_ratios) >= required_wins,
                    improves_decode
                    and sum(ratio < 1.0 for ratio in decode_ratios) >= required_wins,
                    improves_ttft and sum(ratio < 0.98 for ratio in ttft_ratios) >= required_wins,
                )
            )
        ):
            objective = (
                min(total_median, decode_median)
                if decode_only
                else min(total_median, decode_median, ttft_median)
            )
            generated.append((objective, plan.feature_count * 64, plan))
    ranked = []
    while generated:
        selected = choose_mdl_tie(generated)
        ranked.append(selected)
        generated = [candidate for candidate in generated if candidate[2] != selected]
    return tuple(ranked)


def _choose_mlx_lm_plan(samples):
    ranked = _rank_mlx_lm_plans(samples)
    return ranked[0] if ranked else MLXLMPlan(False, False, False, False)


def _paired_plan_ratios(measurements, native_measurements, metric):
    return tuple(
        measurement[metric] / native[metric]
        for measurement, native in zip(measurements, native_measurements)
    )


def _is_decode_only_compression_plan(plan):
    return plan.is_decode_only_compression


def _compression_ladder(plans, decode_ratios):
    singleton_names = []
    for plan in sorted(
        (
            plan
            for plan in plans
            if _is_decode_only_compression_plan(plan) and plan.feature_count == 1
        ),
        key=decode_ratios.__getitem__,
    ):
        singleton_names.extend(name for name, enabled in plan.as_dict().items() if enabled)
    selected = []
    enabled = set()
    available = set(plans)
    for name in singleton_names:
        enabled.add(name)
        candidate = MLXLMPlan(**{feature: feature in enabled for feature in MLXLMPlan().as_dict()})
        if candidate in available:
            selected.append(candidate)
    return tuple(selected)


def _mlx_lm_validation_finalists(measured):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = measured[native]
    ranked = _rank_mlx_lm_plans(measured)
    decode_order = sorted(
        (plan for plan in measured if plan.feature_count),
        key=lambda plan: statistics.median(
            _paired_plan_ratios(measured[plan], native_measurements, 1)
        ),
    )
    compressed_order = sorted(
        (plan for plan in measured if _is_decode_only_compression_plan(plan)),
        key=lambda plan: (
            -plan.feature_count,
            statistics.median(_paired_plan_ratios(measured[plan], native_measurements, 1)),
        ),
    )
    decode_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 1))
        for plan, samples in measured.items()
    }
    return tuple(
        dict.fromkeys(
            (
                native,
                *ranked[:2],
                *decode_order[:3],
                *compressed_order[:1],
                *_compression_ladder(measured, decode_ratios),
            )
        )
    )


def _mlx_lm_validation_survivors(measured):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = measured[native]
    ranked = _rank_mlx_lm_plans(measured)
    metric_leaders = tuple(
        min(
            measured,
            key=lambda plan: statistics.median(
                _paired_plan_ratios(measured[plan], native_measurements, metric)
            ),
        )
        for metric in (1, 2, 0)
    )
    ordered = tuple(dict.fromkeys((native, *ranked[:2], *metric_leaders)))
    return ordered[:_MODEL_VALIDATION_MAX_SURVIVORS]


def _provisional_mlx_lm_finalists(
    provisional,
    candidates,
    *,
    max_finalists=_MODEL_PROVISIONAL_MAX_FINALISTS,
    relative_margin=_MODEL_PROVISIONAL_RELATIVE_MARGIN,
):
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
    decode_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 1))
        for plan, samples in provisional.items()
    }
    best_total = min(total_ratios.values())
    best_ttft = min(ttft_ratios.values())
    best_decode = min(decode_ratios.values())
    fastest_total = min(total_ratios, key=total_ratios.__getitem__)
    fastest_ttft = min(ttft_ratios, key=ttft_ratios.__getitem__)
    fastest_decode = min(decode_ratios, key=decode_ratios.__getitem__)
    compressed = tuple(
        plan
        for plan in candidates
        if plan in total_ratios and _is_decode_only_compression_plan(plan)
    )
    maximal_compression = (
        min(
            compressed,
            key=lambda plan: (-plan.feature_count, decode_ratios[plan]),
        )
        if compressed
        else native
    )
    required = {
        native,
        fastest_total,
        fastest_ttft,
        fastest_decode,
        maximal_compression,
        *_compression_ladder(provisional, decode_ratios),
        *(
            plan
            for plan in candidates
            if plan in total_ratios
            and plan.feature_count == 1
            and min(total_ratios[plan], ttft_ratios[plan], decode_ratios[plan]) < 0.99
        ),
    }
    eligible = sorted(
        (
            plan
            for plan in candidates
            if plan in total_ratios
            and (
                total_ratios[plan] <= best_total * (1.0 + relative_margin)
                or ttft_ratios[plan] <= best_ttft * (1.0 + relative_margin)
                or decode_ratios[plan] <= best_decode * (1.0 + relative_margin)
            )
        ),
        key=lambda plan: min(
            total_ratios[plan] / best_total,
            ttft_ratios[plan] / best_ttft,
            decode_ratios[plan] / best_decode,
        ),
    )
    selected = set(required)
    for plan in eligible:
        if len(selected) >= max_finalists:
            break
        selected.add(plan)
    return tuple(plan for plan in candidates if plan in selected and plan in total_ratios)
