"""apply layer of the mlx_lm integration."""

from __future__ import annotations

import sys

from metile.backends.mlx import (
    _mlx_attention_decode_unchecked,
    mlx_rms_norm,
)
from metile.integrations.mlx_lm._state import (
    _QUANTIZED_MLP_MIN_ROWS,
    _unsupported_decode_shapes,
)
from metile.integrations.mlx_lm.compressed import (
    MLXCompressedAttention,
    MLXCompressedDown,
    MLXCompressedGateUp,
    MLXCompressedVocab,
)
from metile.integrations.mlx_lm.core import (
    _decode_shape_key,
    _supports_metile_decode,
    _supports_metile_rms_norm,
)
from metile.integrations.mlx_lm.modules import (
    MLXAffinePrefill,
    MLXDenseMLP,
)
from metile.integrations.mlx_lm.patching import (
    MLXPatch,
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
)


def apply_metile_to_mlx_lm(
    model=None,
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
    plan=None,
):
    """Patch MLX-LM with zero-copy, autotuned meTile primitives.

    Decode attention, RMSNorm, dense/quantized SwiGLU, and compute-graph fusion are independently
    selectable. Unsupported calls preserve MLX-LM's original implementation.
    The returned handle can restore every changed module or be used as a context
    manager.
    """
    if model is not None and not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    dense_swiglu = dense_mlp
    dense_residual = dense_mlp
    active_compressed_down = compressed_down
    active_compressed_gate_up = compressed_gate_up
    active_compressed_vocab = compressed_vocab
    active_compressed_attention = compressed_attention
    if plan is not None:
        if not isinstance(plan, MLXLMPlan):
            raise TypeError("plan must be an MLXLMPlan")
        attention = attention and plan.attention
        rms_norm = rms_norm and plan.rms_norm
        graph_fusion = graph_fusion and plan.graph_fusion
        quantized_mlp = quantized_mlp and plan.quantized_mlp
        if not plan.affine_prefill:
            affine_prefill = None
        if not plan.dense_mlp:
            dense_swiglu = None
        if not plan.dense_residual:
            dense_residual = None
        if not plan.compressed_down:
            active_compressed_down = None
        if not plan.compressed_gate_up:
            active_compressed_gate_up = None
        if not plan.compressed_vocab:
            active_compressed_vocab = None
        if not plan.compressed_attention:
            active_compressed_attention = None
    if affine_prefill is not None:
        if not isinstance(affine_prefill, MLXAffinePrefill):
            raise TypeError("affine_prefill must be an MLXAffinePrefill")
        if model is not affine_prefill.model:
            raise ValueError("affine_prefill was prepared for a different model")
    if dense_mlp is not None:
        if not isinstance(dense_mlp, MLXDenseMLP):
            raise TypeError("dense_mlp must be an MLXDenseMLP")
        if model is not dense_mlp.model:
            raise ValueError("dense_mlp was prepared for a different model")
    if compressed_down is not None:
        if not isinstance(compressed_down, MLXCompressedDown):
            raise TypeError("compressed_down must be an MLXCompressedDown")
        if model is not compressed_down.model:
            raise ValueError("compressed_down was prepared for a different model")
    if compressed_gate_up is not None:
        if not isinstance(compressed_gate_up, MLXCompressedGateUp):
            raise TypeError("compressed_gate_up must be an MLXCompressedGateUp")
        if model is not compressed_gate_up.model:
            raise ValueError("compressed_gate_up was prepared for a different model")
    if compressed_vocab is not None:
        if not isinstance(compressed_vocab, MLXCompressedVocab):
            raise TypeError("compressed_vocab must be an MLXCompressedVocab")
        if model is not compressed_vocab.model:
            raise ValueError("compressed_vocab was prepared for a different model")
    if compressed_attention is not None:
        if not isinstance(compressed_attention, MLXCompressedAttention):
            raise TypeError("compressed_attention must be an MLXCompressedAttention")
        if model is not compressed_attention.model:
            raise ValueError("compressed_attention was prepared for a different model")
    if (
        not attention
        and not rms_norm
        and not graph_fusion
        and not quantized_mlp
        and affine_prefill is None
        and dense_swiglu is None
        and dense_residual is None
        and active_compressed_down is None
        and active_compressed_gate_up is None
        and active_compressed_vocab is None
        and active_compressed_attention is None
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
                    try:
                        return _mlx_attention_decode_unchecked(
                            queries,
                            keys,
                            values,
                            scale,
                        )
                    except RuntimeError:
                        # The kernel cannot be built for this shape on this device, most
                        # often because its threadgroup memory exceeds the limit. Record the
                        # shape so later tokens skip the attempt, and serve this one from MLX.
                        # Falling back is the whole point: a shape meTile cannot handle must
                        # cost speed, never correctness or a crash.
                        _unsupported_decode_shapes.add(_decode_shape_key(queries, keys))
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

    quantized_mlp_prefill_min_rows = _QUANTIZED_MLP_MIN_ROWS if affine_prefill is not None else 1
    quantized_mlp_prefill_max_rows = None if affine_prefill is not None else 1
    if graph_fusion or quantized_mlp or dense_residual is not None:
        _patch_graph_fusion(
            model,
            replacements,
            nn.QuantizedLinear if quantized_mlp else None,
            dense_mlp=dense_residual,
            dense_swiglu=dense_swiglu is not None,
            quantized_mlp_min_rows=1,
            quantized_mlp_max_rows=1,
            fuse_residual_rms=graph_fusion,
        )
    if quantized_mlp:
        _patch_quantized_mlp(
            model,
            replacements,
            nn.QuantizedLinear,
            min_rows=quantized_mlp_prefill_min_rows,
            max_rows=quantized_mlp_prefill_max_rows,
        )
    _patch_affine_prefill(affine_prefill, replacements)
    _patch_dense_mlp(dense_swiglu, replacements)
    _patch_compressed_down(active_compressed_down, replacements)
    _patch_compressed_gate_up(active_compressed_gate_up, replacements)
    _patch_compressed_vocab(active_compressed_vocab, replacements)
    _patch_compressed_attention(active_compressed_attention, replacements)

    return MLXPatch(replacements, attention_replacement, attention_original)
