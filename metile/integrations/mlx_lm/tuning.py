"""tuning layer of the mlx_lm integration."""

from __future__ import annotations

import copy
import gc
import inspect
import os
import statistics
import time

from metile.backends.mlx_affine import (
    mlx_affine_backend_signature,
)
from metile.backends.mlx_compressed_down import (
    mlx_compressed_down_backend_signature,
)
from metile.backends.mlx_dense import (
    mlx_dense_backend_signature,
)
from metile.backends.mlx_dense_residual import (
    mlx_dense_residual_backend_signature,
)
from metile.backends.mlx_dense_swiglu import (
    mlx_dense_swiglu_backend_signature,
)
from metile.backends.mlx_quantized import (
    mlx_affine_swiglu_backend_signature,
)
from metile.integrations.mlx_lm._state import (
    _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
    _COMPRESSED_GATE_UP_FUSION_MARGIN,
    _COMPRESSED_INTERVAL_DIRECTION_BUDGET,
    _COMPRESSED_SUBSET_AUGMENTATION_BUDGET,
    _MODEL_DECODE_SWITCH_MARGIN,
    _MODEL_PROVISIONAL_MAX_FINALISTS,
    _MODEL_PROVISIONAL_RELATIVE_MARGIN,
    _MODEL_PROVISIONAL_ROUNDS,
    _MODEL_REGRESSION_MARGIN,
    _MODEL_SCREEN_MAX_FINALISTS,
    _MODEL_SCREEN_RELATIVE_MARGIN,
    _MODEL_SCREEN_ROUNDS,
    _MODEL_SEARCH_MIN_DECODE_STEPS,
    _MODEL_STRONG_DECODE_SWITCH_MARGIN,
    _MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN,
    _MODEL_SWITCH_MARGIN,
    _MODEL_TTFT_SWITCH_MARGIN,
    _MODEL_VALIDATION_ATTEMPTS,
    _MODEL_VALIDATION_MAX_SURVIVORS,
    _MODEL_VALIDATION_MIN_DECODE_STEPS,
    _MODEL_VALIDATION_MIN_TRIALS,
    _MODEL_VALIDATION_SCREEN_TRIALS,
    _QUANTIZED_MLP_MIN_ROWS,
    _compressed_attention_calibration_lock,
    _compressed_attention_group_cache_path,
    _compressed_attention_group_lock,
    _compressed_down_calibration_lock,
    _compressed_gate_up_calibration_lock,
    _compressed_gate_up_group_cache_path,
    _compressed_gate_up_group_lock,
    _compressed_gate_up_implementation_cache_path,
    _compressed_gate_up_implementation_lock,
    _compressed_vocab_calibration_lock,
)
from metile.integrations.mlx_lm.apply import (
    apply_metile_to_mlx_lm,
)
from metile.integrations.mlx_lm.compressed import (
    MLXCompressedAttention,
    MLXCompressedDown,
    MLXCompressedGateUp,
    MLXCompressedVocab,
    _audit_larger_compressed_regions,
    _augment_compressed_subset,
    _compressed_attention_repack_bytes,
    _compressed_down_subset_candidates,
    _compressed_gate_up_repack_bytes,
    _prepare_compressed_calibration_reference,
    _repack_compressed_attention_group,
    _repack_compressed_gate_up_group,
    _restore_compressed_attention_calibration,
    _restore_compressed_down_calibration,
    _restore_compressed_gate_up_calibration,
    _restore_compressed_vocab_calibration,
    _select_compressed_region,
    _supports_compressed_gate_up_fusion,
    _write_compressed_attention_calibration,
    _write_compressed_down_calibration,
    _write_compressed_gate_up_calibration,
    _write_compressed_gate_up_implementation,
    _write_compressed_vocab_calibration,
)
from metile.integrations.mlx_lm.core import (
    _mlx_lm_model_signature,
    _prepare_mlx_lm_prompt,
)
from metile.integrations.mlx_lm.modules import (
    MLXAffinePrefill,
    MLXDenseMLP,
    _execute_dense_mlp,
    _select_model_affine8_group,
    _supports_dense_residual_mlp,
)
from metile.integrations.mlx_lm.patching import (
    _patch_affine_prefill,
    _patch_compressed_attention,
    _patch_compressed_down,
    _patch_compressed_gate_up,
    _patch_compressed_vocab,
    _patch_dense_mlp,
    _patch_graph_fusion,
    _patch_quantized_mlp,
)
from metile.integrations.mlx_lm.plan_model import (
    MLXLMPlan,
    _choose_mlx_lm_plan,
    _effective_mlx_lm_plan,
    _fidelity_compatible,
    _is_decode_only_compression_plan,
    _logit_fidelity,
    _mlx_lm_plan_candidates,
    _mlx_lm_validation_finalists,
    _mlx_lm_validation_survivors,
    _mlx_lm_warmup_plans,
    _provisional_mlx_lm_finalists,
    _rank_mlx_lm_plans,
)
from metile.runtime.cache import atomic_write_json, read_json, stable_digest


def _compressed_down_calibration_key(
    model,
    sample_tokens,
    compressed_down,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "allow_approximate": compressed_down.allow_approximate,
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "format": compressed_down.format,
            "group_size": compressed_down.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_down),
                    "class": inspect.getsource(MLXCompressedDown.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedDown.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_down),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_down_calibration),
                    "write": inspect.getsource(_write_compressed_down_calibration),
                }
            ),
            "weights": tuple(
                (weight.shape, weight.format, weight.group_size)
                for _, weight in compressed_down.weights.values()
            ),
        }
    )


def _mlx_lm_plan_key(
    model,
    sample_tokens,
    requested,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
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
            "compressed_down": (
                {
                    "allow_approximate": compressed_down.allow_approximate,
                    "format": compressed_down.format,
                    "group_size": compressed_down.group_size,
                    "layer_indices": compressed_down.layer_indices,
                    "projections": compressed_down.projection_count,
                    "repack_bytes": compressed_down.repack_bytes,
                    "selection": compressed_down.selection,
                }
                if compressed_down is not None
                else None
            ),
            "compressed_gate_up": (
                {
                    "group_size": compressed_gate_up.group_size,
                    "implementation": compressed_gate_up.implementation,
                    "layer_indices": compressed_gate_up.layer_indices,
                    "layers": compressed_gate_up.layer_count,
                    "projections": compressed_gate_up.projection_count,
                    "repack_bytes": compressed_gate_up.repack_bytes,
                    "selection": compressed_gate_up.selection,
                }
                if compressed_gate_up is not None
                else None
            ),
            "compressed_vocab": (
                {
                    "group_size": compressed_vocab.group_size,
                    "projections": compressed_vocab.projection_count,
                    "repack_bytes": compressed_vocab.repack_bytes,
                    "tied": compressed_vocab.tied,
                }
                if compressed_vocab is not None
                else None
            ),
            "compressed_attention": (
                {
                    "group_size": compressed_attention.group_size,
                    "layer_indices": compressed_attention.layer_indices,
                    "layers": compressed_attention.layer_count,
                    "projections": compressed_attention.projection_count,
                    "repack_bytes": compressed_attention.repack_bytes,
                    "selection": compressed_attention.selection,
                }
                if compressed_attention is not None
                else None
            ),
            "dense_mlp": (
                {
                    "min_rows": dense_mlp.min_rows,
                    "mlps": dense_mlp.mlp_count,
                    "repack_bytes": dense_mlp.repack_bytes,
                }
                if dense_mlp is not None
                else None
            ),
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
                    "plan_candidates": inspect.getsource(_mlx_lm_plan_candidates),
                    "compressed_down_backend": mlx_compressed_down_backend_signature(),
                    "compressed_down_class": inspect.getsource(MLXCompressedDown.patched_class),
                    "compressed_down_calibration": inspect.getsource(_calibrate_compressed_down),
                    "compressed_down_candidates": inspect.getsource(
                        _compressed_down_subset_candidates
                    ),
                    "compressed_down_patch": inspect.getsource(_patch_compressed_down),
                    "compressed_gate_up_class": inspect.getsource(
                        MLXCompressedGateUp.patched_class
                    ),
                    "compressed_gate_up_fused_class": inspect.getsource(
                        MLXCompressedGateUp.fused_patched_class
                    ),
                    "compressed_gate_up_fusion_guard": inspect.getsource(
                        _supports_compressed_gate_up_fusion
                    ),
                    "compressed_gate_up_calibration": inspect.getsource(
                        _calibrate_compressed_gate_up
                    ),
                    "compressed_gate_up_group": inspect.getsource(
                        _autotune_compressed_gate_up_group
                    ),
                    "compressed_gate_up_implementation": inspect.getsource(
                        _select_compressed_gate_up_implementation
                    ),
                    "compressed_gate_up_patch": inspect.getsource(_patch_compressed_gate_up),
                    "compressed_attention_class": inspect.getsource(
                        MLXCompressedAttention.patched_class
                    ),
                    "compressed_attention_calibration": inspect.getsource(
                        _calibrate_compressed_attention
                    ),
                    "compressed_attention_group": inspect.getsource(
                        _autotune_compressed_attention_group
                    ),
                    "compressed_attention_patch": inspect.getsource(_patch_compressed_attention),
                    "compressed_region_policy": _compressed_region_policy_signature(),
                    "compressed_vocab_calibration": inspect.getsource(_calibrate_compressed_vocab),
                    "compressed_vocab_class": inspect.getsource(MLXCompressedVocab.patched_class),
                    "compressed_vocab_patch": inspect.getsource(_patch_compressed_vocab),
                    "dense_backend": mlx_dense_swiglu_backend_signature(),
                    "dense_matmul_backend": mlx_dense_backend_signature(),
                    "dense_residual_backend": mlx_dense_residual_backend_signature(),
                    "dense_class": inspect.getsource(MLXDenseMLP.patched_class),
                    "dense_execute": inspect.getsource(_execute_dense_mlp),
                    "dense_patch": inspect.getsource(_patch_dense_mlp),
                    "dense_residual_support": inspect.getsource(_supports_dense_residual_mlp),
                    "dense_selection": inspect.getsource(_select_dense_mlp_implementation),
                    "decode_only_plan": inspect.getsource(_is_decode_only_compression_plan),
                    "effective": inspect.getsource(_effective_mlx_lm_plan),
                    "fidelity": inspect.getsource(_plan_preserves_logits),
                    "finalists": inspect.getsource(_provisional_mlx_lm_finalists),
                    "plan": inspect.getsource(MLXLMPlan),
                    "prompt": inspect.getsource(_prepare_mlx_lm_prompt),
                    "warmups": inspect.getsource(_mlx_lm_warmup_plans),
                    "graph_patch": inspect.getsource(_patch_graph_fusion),
                    "quantized_mlp_patch": inspect.getsource(_patch_quantized_mlp),
                    "rank": inspect.getsource(_rank_mlx_lm_plans),
                    "timing": inspect.getsource(_time_mlx_lm_plan),
                    "validation": inspect.getsource(_validate_mlx_lm_plan),
                    "validation_finalists": inspect.getsource(_mlx_lm_validation_finalists),
                    "validation_joint": inspect.getsource(_validate_mlx_lm_finalists_repeated),
                    "validation_retry": inspect.getsource(_validate_mlx_lm_plan_repeated),
                }
            ),
            "regression_margin": _MODEL_REGRESSION_MARGIN,
            "decode_switch_margin": _MODEL_DECODE_SWITCH_MARGIN,
            "strong_decode_switch_margin": _MODEL_STRONG_DECODE_SWITCH_MARGIN,
            "strong_decode_ttft_regression_margin": (_MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN),
            "dense_mlp_implementation": (
                dense_mlp.implementation if dense_mlp is not None else None
            ),
            "quantized_mlp_min_rows": _QUANTIZED_MLP_MIN_ROWS,
            "switch_margin": _MODEL_SWITCH_MARGIN,
            "ttft_switch_margin": _MODEL_TTFT_SWITCH_MARGIN,
            "trials": trials,
            "screen_max_finalists": _MODEL_SCREEN_MAX_FINALISTS,
            "screen_relative_margin": _MODEL_SCREEN_RELATIVE_MARGIN,
            "screen_rounds": _MODEL_SCREEN_ROUNDS,
            "provisional_max_finalists": _MODEL_PROVISIONAL_MAX_FINALISTS,
            "provisional_relative_margin": _MODEL_PROVISIONAL_RELATIVE_MARGIN,
            "provisional_rounds": _MODEL_PROVISIONAL_ROUNDS,
            "search_decode_steps": _MODEL_SEARCH_MIN_DECODE_STEPS,
            "validation_decode_steps": _MODEL_VALIDATION_MIN_DECODE_STEPS,
            "validation_attempts": _MODEL_VALIDATION_ATTEMPTS,
            "validation_max_survivors": _MODEL_VALIDATION_MAX_SURVIVORS,
            "validation_screen_trials": _MODEL_VALIDATION_SCREEN_TRIALS,
            "validation_trials": _MODEL_VALIDATION_MIN_TRIALS,
            "tuner": 47,
        }
    )


def _time_mlx_lm_plan(
    model,
    sample_tokens,
    plan,
    affine_prefill,
    dense_mlp,
    decode_steps,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    prepared_prompt=None,
    decode_tokens=None,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    decode_token = sample_tokens[:, -1:]
    if prepared_prompt is not None:
        cache = copy.deepcopy(prepared_prompt[0])
        ttft = prepared_prompt[1]
    else:
        cache = make_prompt_cache(model)
    with apply_metile_to_mlx_lm(
        model=model,
        plan=plan,
        affine_prefill=affine_prefill,
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
    ):
        if prepared_prompt is None:
            total_start = time.perf_counter_ns()
            logits = model(sample_tokens, cache=cache)
            mx.eval(logits)
            ttft = (time.perf_counter_ns() - total_start) * 1e-9
        if decode_tokens is None:
            decode_tokens = (decode_token,) * decode_steps
        elif len(decode_tokens) != decode_steps:
            raise ValueError("decode trajectory must match decode steps")
        decode_start = time.perf_counter_ns()
        for token in decode_tokens:
            logits = model(token, cache=cache)
            mx.eval(logits)
        decode_elapsed = (time.perf_counter_ns() - decode_start) * 1e-9
        decode = decode_elapsed / decode_steps
        total = ttft + decode_elapsed
    next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
    return (ttft, decode, total), next_token


def _run_compressed_calibration_candidate(
    model,
    sample_tokens,
    reference,
    steps,
    plan,
    **patches,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if steps < 1:
        raise ValueError("compressed calibration requires positive decode steps")
    actual_cache = (
        copy.deepcopy(reference.prompt_cache)
        if reference.prompt_cache is not None
        else make_prompt_cache(model)
    )
    with apply_metile_to_mlx_lm(model=model, plan=plan, **patches):
        if reference.prompt_cache is None:
            actual = model(sample_tokens, cache=actual_cache)
            mx.eval(actual)
        for _ in range(steps):
            actual = model(reference.decode_token, cache=actual_cache)
            mx.eval(actual)
    return actual


def _compressed_region_policy_signature():
    return stable_digest(
        {
            "candidate_trajectory": inspect.getsource(_run_compressed_calibration_candidate),
            "full_horizon": inspect.getsource(_audit_larger_compressed_regions),
            "interval": inspect.getsource(_select_compressed_region),
            "interval_direction_budget": _COMPRESSED_INTERVAL_DIRECTION_BUDGET,
            "reference_trajectory": inspect.getsource(_prepare_compressed_calibration_reference),
            "subset": inspect.getsource(_augment_compressed_subset),
            "subset_budget": _COMPRESSED_SUBSET_AUGMENTATION_BUDGET,
        }
    )


def _calibrate_compressed_down(model, sample_tokens, compressed_down, decode_steps):
    if compressed_down.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_down.weights.items())
    if not entries:
        compressed_down.calibrated = True
        compressed_down.selection = "native"
        return
    key = _compressed_down_calibration_key(
        model,
        sample_tokens,
        compressed_down,
        decode_steps,
    )
    with _compressed_down_calibration_lock:
        if _restore_compressed_down_calibration(compressed_down, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_down=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_down.patched_classes.clear()
            compressed_down.weights = {entries[index][0]: entries[index][1] for index in indices}
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_down=compressed_down,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_down.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_down.patched_classes.clear()
    compressed_down.weights = {entries[index][0]: entries[index][1] for index in selected_indices}
    compressed_down.repack_bytes = sum(
        weight.nbytes for _, weight in compressed_down.weights.values()
    )
    compressed_down.calibrated = True
    compressed_down.selection = selected_name
    compressed_down.layer_indices = selected_indices
    compressed_down.calibration_fidelity = selected_fidelity
    with _compressed_down_calibration_lock:
        _write_compressed_down_calibration(compressed_down, key)
    gc.collect()
    mx.clear_cache()


def _compressed_gate_up_group_key(model, sample_tokens, compressed_gate_up, decode_steps):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_tuning": {
                name: value
                for name, value in compressed_gate_up.group_tuning.items()
                if name != "cached"
            },
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "calibrate": inspect.getsource(_calibrate_compressed_gate_up),
                    "region_policy": _compressed_region_policy_signature(),
                    "repack": inspect.getsource(_repack_compressed_gate_up_group),
                    "select": inspect.getsource(_select_model_affine8_group),
                    "tune": inspect.getsource(_autotune_compressed_gate_up_group),
                }
            ),
            "weights": tuple(
                (gate_weight.shape, str(gate_weight.dtype), up_weight.shape)
                for _, _, gate_weight, _, up_weight in compressed_gate_up.source_layers.values()
            ),
        }
    )


def _autotune_compressed_group(
    model,
    sample_tokens,
    region,
    decode_steps,
    *,
    group_key,
    lock,
    cache_path,
    repack,
    calibrate,
):
    """Pick the group size for a compressed region by timing each candidate.

    The region-specific parts are injected: how its cache key is built, which lock and cache
    file it owns, how it repacks at a given group size, and how it is calibrated afterwards.
    """
    tuning = region.group_tuning
    if tuning is None or tuning.get("model_calibrated") or not region.source_layers:
        return
    timings_payload = tuning.get("median_nanoseconds")
    native_timing = tuning.get("native_median_nanoseconds")
    if not isinstance(timings_payload, dict) or not isinstance(native_timing, int):
        return
    timings = {int(group): int(value) for group, value in timings_payload.items()}
    key = group_key(
        model,
        sample_tokens,
        region,
        decode_steps,
    )
    cached = None
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with lock:
            cached = read_json(cache_path, {}).get(key)
    if isinstance(cached, dict) and cached.get("group_size") in timings:
        selected = cached["group_size"]
        repack(region, selected)
        calibrate(
            model,
            sample_tokens,
            region,
            decode_steps,
        )
        region.group_tuning = {
            **tuning,
            **cached,
            "cached": True,
            "group_size": selected,
            "micro_group_size": tuning["group_size"],
            "model_calibrated": True,
        }
        return

    total_layers = len(region.source_layers)
    candidates = {}
    layer_counts = {}
    for group in sorted(timings):
        repack(region, group)
        calibrate(
            model,
            sample_tokens,
            region,
            decode_steps,
        )
        layer_counts[group] = region.layer_count
        candidates[str(group)] = {
            "fidelity": region.calibration_fidelity,
            "layers": region.layer_count,
            "selection": region.selection,
        }
    selected, estimates = _select_model_affine8_group(
        total_layers,
        layer_counts,
        timings,
        native_timing,
    )
    if region.group_size != selected:
        repack(region, selected)
        calibrate(
            model,
            sample_tokens,
            region,
            decode_steps,
        )
    record = {
        "group_size": selected,
        "model_calibrated": True,
        "model_candidates": candidates,
        "predicted_nanoseconds": {str(group): round(value) for group, value in estimates.items()},
    }
    region.group_tuning = {
        **tuning,
        **record,
        "cached": False,
        "micro_group_size": tuning["group_size"],
    }
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with lock:
            payload = read_json(cache_path, {})
            payload[key] = record
            atomic_write_json(cache_path, payload)


def _autotune_compressed_gate_up_group(model, sample_tokens, compressed_gate_up, decode_steps):
    _autotune_compressed_group(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
        group_key=_compressed_gate_up_group_key,
        lock=_compressed_gate_up_group_lock,
        cache_path=_compressed_gate_up_group_cache_path,
        repack=_repack_compressed_gate_up_group,
        calibrate=_calibrate_compressed_gate_up,
    )


def _autotune_compressed_attention_group(model, sample_tokens, compressed_attention, decode_steps):
    _autotune_compressed_group(
        model,
        sample_tokens,
        compressed_attention,
        decode_steps,
        group_key=_compressed_attention_group_key,
        lock=_compressed_attention_group_lock,
        cache_path=_compressed_attention_group_cache_path,
        repack=_repack_compressed_attention_group,
        calibrate=_calibrate_compressed_attention,
    )


def _compressed_gate_up_calibration_key(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_gate_up.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_gate_up),
                    "class": inspect.getsource(MLXCompressedGateUp.patched_class),
                    "fused_backend": mlx_affine_swiglu_backend_signature(),
                    "fused_class": inspect.getsource(MLXCompressedGateUp.fused_patched_class),
                    "fused_guard": inspect.getsource(_supports_compressed_gate_up_fusion),
                    "fidelity": inspect.getsource(MLXCompressedGateUp.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_gate_up),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_gate_up_calibration),
                    "write": inspect.getsource(_write_compressed_gate_up_calibration),
                }
            ),
            "weights": tuple(
                (
                    gate_weight.shape,
                    gate_weight.group_size,
                    up_weight.shape,
                    up_weight.group_size,
                )
                for _, _, gate_weight, _, up_weight in compressed_gate_up.layers.values()
            ),
        }
    )


def _calibrate_compressed_gate_up(model, sample_tokens, compressed_gate_up, decode_steps):
    if compressed_gate_up.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_gate_up.layers.items())
    if not entries:
        compressed_gate_up.calibrated = True
        compressed_gate_up.selection = "native"
        return
    key = _compressed_gate_up_calibration_key(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
    )
    with _compressed_gate_up_calibration_lock:
        if _restore_compressed_gate_up_calibration(compressed_gate_up, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_gate_up=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_gate_up.patched_classes.clear()
            compressed_gate_up.layers = {entries[index][0]: entries[index][1] for index in indices}
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_gate_up=compressed_gate_up,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_gate_up.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_gate_up.patched_classes.clear()
    compressed_gate_up.layers = {entries[index][0]: entries[index][1] for index in selected_indices}
    compressed_gate_up.repack_bytes = _compressed_gate_up_repack_bytes(compressed_gate_up.layers)
    compressed_gate_up.calibrated = True
    compressed_gate_up.selection = selected_name
    compressed_gate_up.layer_indices = selected_indices
    compressed_gate_up.calibration_fidelity = selected_fidelity
    with _compressed_gate_up_calibration_lock:
        _write_compressed_gate_up_calibration(compressed_gate_up, key)
    gc.collect()
    mx.clear_cache()


def _compressed_gate_up_implementation_key(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_gate_up.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "selection": compressed_gate_up.selection,
            "source": stable_digest(
                {
                    "backend": mlx_affine_swiglu_backend_signature(),
                    "class": inspect.getsource(MLXCompressedGateUp.fused_patched_class),
                    "fidelity": inspect.getsource(MLXCompressedGateUp.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_gate_up),
                    "select": inspect.getsource(_select_compressed_gate_up_implementation),
                    "switch_margin": _COMPRESSED_GATE_UP_FUSION_MARGIN,
                }
            ),
            "weights": tuple(
                (
                    gate_weight.shape,
                    gate_weight.group_size,
                    up_weight.shape,
                    up_weight.group_size,
                )
                for _, _, gate_weight, _, up_weight in compressed_gate_up.layers.values()
            ),
        }
    )


def _select_compressed_gate_up_implementation(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
    trials,
):
    compressed_gate_up.implementation = "projected"
    if not compressed_gate_up.layers or not any(
        _supports_compressed_gate_up_fusion(module)
        for module, *_ in compressed_gate_up.layers.values()
    ):
        compressed_gate_up.implementation_tuning = {
            "implementation": "projected",
            "reason": "no_supported_fusion",
        }
        return
    key = _compressed_gate_up_implementation_key(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
    )
    cached = None
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_gate_up_implementation_lock:
            cached = read_json(_compressed_gate_up_implementation_cache_path, {}).get(key)
    if isinstance(cached, dict) and cached.get("implementation") in {"fused", "projected"}:
        compressed_gate_up.implementation = cached["implementation"]
        compressed_gate_up.implementation_tuning = {**cached, "cached": True}
        return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )
    plan = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    compressed_gate_up.implementation = "fused"
    actual = _run_compressed_calibration_candidate(
        model,
        sample_tokens,
        reference,
        decode_steps,
        plan,
        compressed_gate_up=compressed_gate_up,
    )
    fidelity = _logit_fidelity(reference.full_reference, actual)
    if not compressed_gate_up.fidelity_compatible(fidelity):
        record = {
            "cached": False,
            "fidelity": fidelity,
            "implementation": "projected",
            "reason": "fidelity",
        }
        compressed_gate_up.implementation = "projected"
        compressed_gate_up.implementation_tuning = record
        _write_compressed_gate_up_implementation(key, record)
        return

    prepared_prompt = _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps)
    decode_tokens = prepared_prompt[2]
    implementations = ("projected", "fused")
    samples = {implementation: [] for implementation in implementations}
    for round_index in range(max(3, trials)):
        order = implementations if round_index % 2 == 0 else tuple(reversed(implementations))
        for implementation in order:
            compressed_gate_up.implementation = implementation
            measurement, _ = _time_mlx_lm_plan(
                model,
                sample_tokens,
                plan,
                None,
                None,
                decode_steps,
                compressed_gate_up=compressed_gate_up,
                prepared_prompt=prepared_prompt,
                decode_tokens=decode_tokens,
            )
            samples[implementation].append(measurement[1])
    medians = {
        implementation: statistics.median(values) for implementation, values in samples.items()
    }
    selected = (
        "fused"
        if medians["fused"] < medians["projected"] * (1.0 - _COMPRESSED_GATE_UP_FUSION_MARGIN)
        else "projected"
    )
    record = {
        "cached": False,
        "fidelity": fidelity,
        "implementation": selected,
        "median_nanoseconds": {
            implementation: round(value * 1e9) for implementation, value in medians.items()
        },
        "reason": "timing",
    }
    compressed_gate_up.implementation = selected
    compressed_gate_up.implementation_tuning = record
    _write_compressed_gate_up_implementation(key, record)


def _compressed_attention_group_key(model, sample_tokens, compressed_attention, decode_steps):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_tuning": {
                name: value
                for name, value in compressed_attention.group_tuning.items()
                if name != "cached"
            },
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "calibrate": inspect.getsource(_calibrate_compressed_attention),
                    "region_policy": _compressed_region_policy_signature(),
                    "repack": inspect.getsource(_repack_compressed_attention_group),
                    "select": inspect.getsource(_select_model_affine8_group),
                    "tune": inspect.getsource(_autotune_compressed_attention_group),
                }
            ),
            "weights": tuple(
                tuple((weight.shape, str(weight.dtype)) for _, weight in projections)
                for _, projections in compressed_attention.source_layers.values()
            ),
        }
    )


def _compressed_attention_calibration_key(
    model,
    sample_tokens,
    compressed_attention,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_attention.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_attention),
                    "class": inspect.getsource(MLXCompressedAttention.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedAttention.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_attention),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_attention_calibration),
                    "write": inspect.getsource(_write_compressed_attention_calibration),
                }
            ),
            "weights": tuple(
                tuple((weight.shape, weight.group_size) for _, weight in projections)
                for _, projections in compressed_attention.layers.values()
            ),
        }
    )


def _calibrate_compressed_attention(model, sample_tokens, compressed_attention, decode_steps):
    if compressed_attention.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_attention.layers.items())
    if not entries:
        compressed_attention.calibrated = True
        compressed_attention.selection = "native"
        return
    key = _compressed_attention_calibration_key(
        model,
        sample_tokens,
        compressed_attention,
        decode_steps,
    )
    with _compressed_attention_calibration_lock:
        if _restore_compressed_attention_calibration(compressed_attention, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_attention=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_attention.patched_classes.clear()
            compressed_attention.layers = {
                entries[index][0]: entries[index][1] for index in indices
            }
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_attention=compressed_attention,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_attention.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
        augmentation_budget=0,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_attention.patched_classes.clear()
    compressed_attention.layers = {
        entries[index][0]: entries[index][1] for index in selected_indices
    }
    compressed_attention.repack_bytes = _compressed_attention_repack_bytes(
        compressed_attention.layers
    )
    compressed_attention.calibrated = True
    compressed_attention.selection = selected_name
    compressed_attention.layer_indices = selected_indices
    compressed_attention.calibration_fidelity = selected_fidelity
    with _compressed_attention_calibration_lock:
        _write_compressed_attention_calibration(compressed_attention, key)
    gc.collect()
    mx.clear_cache()


def _compressed_vocab_calibration_key(
    model,
    sample_tokens,
    compressed_vocab,
    decode_steps,
):
    import mlx.core as mx

    weight = compressed_vocab.weight
    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_vocab.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_vocab),
                    "class": inspect.getsource(MLXCompressedVocab.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedVocab.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_vocab),
                }
            ),
            "tied": compressed_vocab.tied,
            "weight": (weight.shape, weight.format, weight.group_size),
        }
    )


def _calibrate_compressed_vocab(model, sample_tokens, compressed_vocab, decode_steps):
    if compressed_vocab.calibrated or compressed_vocab.weight is None:
        return
    import mlx.core as mx

    key = _compressed_vocab_calibration_key(
        model,
        sample_tokens,
        compressed_vocab,
        decode_steps,
    )
    with _compressed_vocab_calibration_lock:
        if _restore_compressed_vocab_calibration(compressed_vocab, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )
    plan = MLXLMPlan(False, False, False, False, compressed_vocab=True)
    actual = _run_compressed_calibration_candidate(
        model,
        sample_tokens,
        reference,
        decode_steps,
        plan,
        compressed_vocab=compressed_vocab,
    )
    fidelity = _logit_fidelity(reference.full_reference, actual)
    if not compressed_vocab.fidelity_compatible(fidelity):
        compressed_vocab.weight = None
        compressed_vocab.repack_bytes = 0
    compressed_vocab.patched_classes.clear()
    compressed_vocab.calibrated = True
    compressed_vocab.calibration_fidelity = fidelity
    with _compressed_vocab_calibration_lock:
        _write_compressed_vocab_calibration(compressed_vocab, key)
    gc.collect()
    mx.clear_cache()


def _cache_aware_dense_fidelity(model, sample_tokens, dense_mlp, implementation):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    dense_mlp.implementation = implementation
    reference_cache = make_prompt_cache(model)
    actual_cache = make_prompt_cache(model)
    reference_prefix = model(sample_tokens[:, :-1], cache=reference_cache)
    mx.eval(reference_prefix)
    reference = model(sample_tokens[:, -1:], cache=reference_cache)
    mx.eval(reference)
    dense_plan = MLXLMPlan(False, False, False, False, False, True)
    with apply_metile_to_mlx_lm(model=model, plan=dense_plan, dense_mlp=dense_mlp):
        actual_prefix = model(sample_tokens[:, :-1], cache=actual_cache)
        mx.eval(actual_prefix)
        actual = model(sample_tokens[:, -1:], cache=actual_cache)
        mx.eval(actual)
    return _logit_fidelity(reference, actual)


def _time_dense_mlp_implementation(model, sample_tokens, dense_mlp, implementation):
    import mlx.core as mx

    dense_mlp.implementation = implementation
    dense_plan = MLXLMPlan(False, False, False, False, False, True)
    start = time.perf_counter_ns()
    with apply_metile_to_mlx_lm(model=model, plan=dense_plan, dense_mlp=dense_mlp):
        output = model(sample_tokens)
        mx.eval(output)
    return (time.perf_counter_ns() - start) * 1e-9


def _select_dense_mlp_implementation(model, sample_tokens, dense_mlp, trials):
    compatible = []
    for implementation in ("fused", "projected"):
        try:
            fidelity = _cache_aware_dense_fidelity(
                model,
                sample_tokens,
                dense_mlp,
                implementation,
            )
        except (RuntimeError, TypeError, ValueError):
            continue
        exact_fusion = implementation != "fused" or (
            fidelity["kl_divergence"] == 0.0
            and fidelity["mean_logit_error"] == 0.0
            and fidelity["max_logit_error"] == 0.0
        )
        if _fidelity_compatible(fidelity) and exact_fusion:
            compatible.append(implementation)
    if not compatible:
        dense_mlp.implementation = "native"
        return

    for implementation in compatible:
        _time_dense_mlp_implementation(model, sample_tokens, dense_mlp, implementation)
    samples = {implementation: [] for implementation in compatible}
    for round_index in range(max(3, min(trials, 7))):
        ordered = compatible if round_index % 2 == 0 else tuple(reversed(compatible))
        for implementation in ordered:
            samples[implementation].append(
                _time_dense_mlp_implementation(
                    model,
                    sample_tokens,
                    dense_mlp,
                    implementation,
                )
            )
    dense_mlp.implementation = min(
        compatible,
        key=lambda implementation: statistics.median(samples[implementation]),
    )


def _plan_preserves_logits(
    model,
    sample_tokens,
    plan,
    affine_prefill,
    dense_mlp,
    reference,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    import mlx.core as mx

    decode_compression = any(
        (
            plan.compressed_down,
            plan.compressed_gate_up,
            plan.compressed_vocab,
            plan.compressed_attention,
        )
    )
    if decode_compression and sample_tokens.shape[1] > 1:
        from mlx_lm.models.cache import make_prompt_cache

        reference_cache = make_prompt_cache(model)
        reference_prefix = model(sample_tokens[:, :-1], cache=reference_cache)
        mx.eval(reference_prefix)
        reference = model(sample_tokens[:, -1:], cache=reference_cache)
        mx.eval(reference)

    with apply_metile_to_mlx_lm(
        model=model,
        plan=plan,
        affine_prefill=affine_prefill,
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
    ):
        if decode_compression and sample_tokens.shape[1] > 1:
            actual_cache = make_prompt_cache(model)
            actual_prefix = model(sample_tokens[:, :-1], cache=actual_cache)
            mx.eval(actual_prefix)
            actual = model(sample_tokens[:, -1:], cache=actual_cache)
        else:
            actual = model(sample_tokens)
        mx.eval(actual)
    fidelity = _logit_fidelity(reference, actual)
    policies = []
    if plan.compressed_down and compressed_down is not None:
        policies.append(compressed_down.fidelity_compatible)
    if plan.compressed_gate_up and compressed_gate_up is not None:
        policies.append(compressed_gate_up.fidelity_compatible)
    if plan.compressed_vocab and compressed_vocab is not None:
        policies.append(compressed_vocab.fidelity_compatible)
    if plan.compressed_attention and compressed_attention is not None:
        policies.append(compressed_attention.fidelity_compatible)
    return (
        all(policy(fidelity) for policy in policies) if policies else _fidelity_compatible(fidelity)
    )


def _measure_mlx_lm_plans(
    model,
    sample_tokens,
    candidates,
    affine_prefill,
    dense_mlp,
    decode_steps,
    rounds,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    *,
    prepared_prompt=None,
    validate_fidelity=True,
):
    import mlx.core as mx

    samples = {plan: [] for plan in candidates}
    expected_token = None
    compatible = set(candidates)
    if validate_fidelity:
        reference = model(sample_tokens)
        mx.eval(reference)
        for plan in candidates:
            if not plan.feature_count:
                continue
            try:
                if not _plan_preserves_logits(
                    model,
                    sample_tokens,
                    plan,
                    affine_prefill,
                    dense_mlp,
                    reference,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                ):
                    compatible.remove(plan)
            except (RuntimeError, TypeError, ValueError):
                compatible.remove(plan)
    if prepared_prompt is None:
        prepared_prompt = (
            _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps)
            if sample_tokens.shape[1] > 1
            and any(
                not plan.feature_count or _is_decode_only_compression_plan(plan)
                for plan in compatible
            )
            else None
        )
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
                    dense_mlp,
                    decode_steps,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                    prepared_prompt=(
                        prepared_prompt
                        if not plan.feature_count or _is_decode_only_compression_plan(plan)
                        else None
                    ),
                    decode_tokens=(prepared_prompt[2] if prepared_prompt is not None else None),
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


def _extend_mlx_lm_measurements(
    model,
    sample_tokens,
    measured,
    candidates,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    *,
    prepared_prompt=None,
):
    existing_trials = min(len(measured[plan]) for plan in candidates)
    remaining_trials = max(0, trials - existing_trials)
    if not remaining_trials:
        return {plan: measured[plan] for plan in candidates}
    additional = _measure_mlx_lm_plans(
        model,
        sample_tokens,
        candidates,
        affine_prefill,
        dense_mlp,
        decode_steps,
        remaining_trials,
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
        prepared_prompt=prepared_prompt,
        validate_fidelity=False,
    )
    return {plan: measured[plan] + additional[plan] for plan in candidates if plan in additional}


def _validate_mlx_lm_plan(
    model,
    sample_tokens,
    selected,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    if not selected.feature_count:
        return selected
    native = MLXLMPlan(False, False, False, False)
    measured = _measure_mlx_lm_plans(
        model,
        sample_tokens,
        (native, selected),
        affine_prefill,
        dense_mlp,
        max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        max(_MODEL_VALIDATION_MIN_TRIALS, trials),
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
    )
    return _choose_mlx_lm_plan(measured)


def _validate_mlx_lm_plan_repeated(
    model,
    sample_tokens,
    selected,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    native = MLXLMPlan(False, False, False, False)
    for _ in range(_MODEL_VALIDATION_ATTEMPTS):
        validated = _validate_mlx_lm_plan(
            model,
            sample_tokens,
            selected,
            affine_prefill,
            dense_mlp,
            decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )
        if validated.feature_count:
            return validated
    return native


def _validate_mlx_lm_finalists_repeated(
    model,
    sample_tokens,
    finalists,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    native = MLXLMPlan(False, False, False, False)
    validation_decode_steps = max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4)
    validation_trials = max(_MODEL_VALIDATION_MIN_TRIALS, trials)
    prepared_prompt = (
        _prepare_mlx_lm_prompt(model, sample_tokens, validation_decode_steps)
        if getattr(sample_tokens, "ndim", None) == 2 and sample_tokens.shape[1] > 1
        else None
    )
    for _ in range(_MODEL_VALIDATION_ATTEMPTS):
        screening_trials = (
            min(_MODEL_VALIDATION_SCREEN_TRIALS, validation_trials)
            if len(finalists) > _MODEL_VALIDATION_MAX_SURVIVORS
            else validation_trials
        )
        screening = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            finalists,
            affine_prefill,
            dense_mlp,
            validation_decode_steps,
            screening_trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        survivors = (
            _mlx_lm_validation_survivors(screening)
            if screening_trials < validation_trials
            else tuple(screening)
        )
        measured = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            screening,
            survivors,
            affine_prefill,
            dense_mlp,
            validation_decode_steps,
            validation_trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        selected = _choose_mlx_lm_plan(measured)
        if selected.feature_count:
            return selected
    return native
