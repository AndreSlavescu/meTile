"""autotune layer of the mlx_lm integration."""

from __future__ import annotations

from metile.integrations.mlx_lm._state import (
    _MODEL_PROVISIONAL_ROUNDS,
    _MODEL_SCREEN_MAX_FINALISTS,
    _MODEL_SCREEN_RELATIVE_MARGIN,
    _MODEL_SCREEN_ROUNDS,
    _MODEL_SEARCH_MIN_DECODE_STEPS,
    _MODEL_VALIDATION_MIN_DECODE_STEPS,
    _mlx_lm_plan_lock,
)
from metile.integrations.mlx_lm.compressed import (
    MLXCompressedAttention,
    MLXCompressedDown,
    MLXCompressedGateUp,
    MLXCompressedVocab,
)
from metile.integrations.mlx_lm.core import (
    _prepare_mlx_lm_prompt,
)
from metile.integrations.mlx_lm.modules import (
    MLXAffinePrefill,
    MLXDenseMLP,
)
from metile.integrations.mlx_lm.plan_model import (
    MLXLMPlan,
    _effective_mlx_lm_plan,
    _is_decode_only_compression_plan,
    _mlx_lm_plan_candidates,
    _mlx_lm_validation_finalists,
    _mlx_lm_warmup_plans,
    _provisional_mlx_lm_finalists,
    _read_mlx_lm_plan,
    _write_mlx_lm_plan,
)
from metile.integrations.mlx_lm.tuning import (
    _autotune_compressed_attention_group,
    _autotune_compressed_gate_up_group,
    _calibrate_compressed_attention,
    _calibrate_compressed_down,
    _calibrate_compressed_gate_up,
    _calibrate_compressed_vocab,
    _extend_mlx_lm_measurements,
    _measure_mlx_lm_plans,
    _mlx_lm_plan_key,
    _select_compressed_gate_up_implementation,
    _select_dense_mlp_implementation,
    _validate_mlx_lm_finalists_repeated,
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
    dense_mlp=None,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
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
    if dense_mlp is not None and not isinstance(dense_mlp, MLXDenseMLP):
        raise TypeError("dense_mlp must be an MLXDenseMLP")
    if dense_mlp is not None and dense_mlp.model is not model:
        raise ValueError("dense_mlp was prepared for a different model")
    if compressed_down is not None and not isinstance(compressed_down, MLXCompressedDown):
        raise TypeError("compressed_down must be an MLXCompressedDown")
    if compressed_down is not None and compressed_down.model is not model:
        raise ValueError("compressed_down was prepared for a different model")
    if compressed_gate_up is not None and not isinstance(compressed_gate_up, MLXCompressedGateUp):
        raise TypeError("compressed_gate_up must be an MLXCompressedGateUp")
    if compressed_gate_up is not None and compressed_gate_up.model is not model:
        raise ValueError("compressed_gate_up was prepared for a different model")
    if compressed_vocab is not None and not isinstance(compressed_vocab, MLXCompressedVocab):
        raise TypeError("compressed_vocab must be an MLXCompressedVocab")
    if compressed_vocab is not None and compressed_vocab.model is not model:
        raise ValueError("compressed_vocab was prepared for a different model")
    if compressed_attention is not None and not isinstance(
        compressed_attention, MLXCompressedAttention
    ):
        raise TypeError("compressed_attention must be an MLXCompressedAttention")
    if compressed_attention is not None and compressed_attention.model is not model:
        raise ValueError("compressed_attention was prepared for a different model")
    if compressed_down is not None:
        _calibrate_compressed_down(
            model,
            sample_tokens,
            compressed_down,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if compressed_gate_up is not None:
        _autotune_compressed_gate_up_group(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _calibrate_compressed_gate_up(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _select_compressed_gate_up_implementation(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
            trials,
        )
    if compressed_vocab is not None:
        _calibrate_compressed_vocab(
            model,
            sample_tokens,
            compressed_vocab,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if compressed_attention is not None:
        _autotune_compressed_attention_group(
            model,
            sample_tokens,
            compressed_attention,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _calibrate_compressed_attention(
            model,
            sample_tokens,
            compressed_attention,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if dense_mlp is not None and sample_tokens.shape[1] >= dense_mlp.min_rows:
        _select_dense_mlp_implementation(model, sample_tokens, dense_mlp, trials)
    requested = MLXLMPlan(
        attention=attention,
        rms_norm=rms_norm,
        graph_fusion=graph_fusion,
        quantized_mlp=quantized_mlp,
        affine_prefill=affine_prefill is not None,
        dense_mlp=dense_mlp is not None,
        dense_residual=dense_mlp is not None,
        compressed_down=compressed_down is not None and compressed_down.projection_count > 0,
        compressed_gate_up=compressed_gate_up is not None
        and compressed_gate_up.projection_count > 0,
        compressed_vocab=compressed_vocab is not None and compressed_vocab.projection_count > 0,
        compressed_attention=compressed_attention is not None
        and compressed_attention.projection_count > 0,
    )
    key = _mlx_lm_plan_key(
        model,
        sample_tokens,
        requested,
        affine_prefill,
        dense_mlp,
        decode_steps,
        trials,
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
    )
    with _mlx_lm_plan_lock:
        cached = _read_mlx_lm_plan(key)
        if cached is not None:
            return cached

        search_decode_steps = max(_MODEL_SEARCH_MIN_DECODE_STEPS, decode_steps)
        candidates = _mlx_lm_plan_candidates(requested)
        prepared_prompt = (
            _prepare_mlx_lm_prompt(model, sample_tokens, search_decode_steps)
            if sample_tokens.shape[1] > 1
            and any(
                not plan.feature_count or _is_decode_only_compression_plan(plan)
                for plan in candidates
            )
            else None
        )
        _measure_mlx_lm_plans(
            model,
            sample_tokens,
            _mlx_lm_warmup_plans(candidates),
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            1,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        candidates = tuple(
            dict.fromkeys(
                _effective_mlx_lm_plan(
                    plan,
                    affine_prefill,
                    dense_mlp,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                )
                for plan in candidates
            )
        )
        screening = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            candidates,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            _MODEL_SCREEN_ROUNDS,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        screened = _provisional_mlx_lm_finalists(
            screening,
            candidates,
            max_finalists=_MODEL_SCREEN_MAX_FINALISTS,
            relative_margin=_MODEL_SCREEN_RELATIVE_MARGIN,
        )
        provisional = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            screening,
            screened,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            _MODEL_PROVISIONAL_ROUNDS,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        finalists = _provisional_mlx_lm_finalists(provisional, screened)
        measured = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            provisional,
            finalists,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        selected = _validate_mlx_lm_finalists_repeated(
            model,
            sample_tokens,
            _mlx_lm_validation_finalists(measured),
            affine_prefill,
            dense_mlp,
            decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )
        _write_mlx_lm_plan(key, selected)
        return selected
