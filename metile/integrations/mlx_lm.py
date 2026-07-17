from __future__ import annotations

import sys
from dataclasses import dataclass

from metile.backends.mlx import (
    mlx_add_rms_norm_selection,
    mlx_attention_decode,
    mlx_rms_norm,
)
from metile.backends.mlx_graph import compile_mlx_graph
from metile.ir.graph_ir import GraphBuilder, TensorSpec

_graph_executable_cache = {}


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


def _supports_metile_decode(queries, keys, values, cache, mask, sinks):
    return (
        not hasattr(cache, "bits")
        and mask is None
        and sinks is None
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[2] == 1
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
        classes.extend(type(layer) for layer in getattr(model, "layers", ()))
    else:
        module = sys.modules.get("mlx_lm.models.llama")
        if module is not None and hasattr(module, "TransformerBlock"):
            classes.append(module.TransformerBlock)

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


def apply_metile_to_mlx_lm(
    model=None,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
):
    """Patch MLX-LM with zero-copy, autotuned meTile primitives.

    Decode attention, RMSNorm, and compute-graph fusion are independently
    selectable. Unsupported calls preserve MLX-LM's original implementation.
    The returned handle can restore every changed module or be used as a context
    manager.
    """
    if model is not None and not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if not attention and not rms_norm and not graph_fusion:
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
                    return mlx_attention_decode(queries, keys, values, scale=scale)
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

    return MLXPatch(replacements, attention_replacement, attention_original)


__all__ = ["MLXPatch", "apply_metile_to_mlx_lm"]
