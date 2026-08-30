"""patching layer of the mlx_lm integration."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from metile.backends.mlx import (
    mlx_add_rms_norm_selection,
)
from metile.integrations.mlx_lm._state import (
    _FUSED_BLOCK_CLASSES,
    _GATED_MLP_CLASSES,
)
from metile.integrations.mlx_lm.compressed import (
    _supports_compressed_gate_up_fusion,
)
from metile.integrations.mlx_lm.core import (
    _attention_module,
    _model_layers,
    _recognised,
    _registry_classes,
)
from metile.integrations.mlx_lm.modules import (
    _execute_dense_mlp,
    _execute_quantized_mlp,
    _execute_residual_rms_graph,
    _supports_dense_residual_mlp,
    _supports_metile_residual_rms_norm,
    _supports_quantized_mlp,
    _supports_quantized_residual_mlp,
)


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


def _patch_graph_fusion(
    model,
    replacements,
    quantized_linear=None,
    dense_mlp=None,
    dense_swiglu=False,
    *,
    quantized_mlp_min_rows=1,
    quantized_mlp_max_rows=None,
    fuse_residual_rms=True,
):
    classes = []
    if model is not None:
        classes.extend(type(layer) for layer in _model_layers(model))
    else:
        classes.extend(_registry_classes(_FUSED_BLOCK_CLASSES))

    for block_class in dict.fromkeys(classes):
        if not _recognised(block_class, _FUSED_BLOCK_CLASSES):
            continue
        original = block_class.__call__
        if getattr(original, "_metile_original", None) is not None:
            continue
        quantized_support_cache = {}

        def make_replacement(original_call, support_cache):
            def replacement(self, values, mask=None, cache=None):
                supports_quantized_residual = False
                supports_dense_residual = False
                if (
                    quantized_linear is not None
                    and hasattr(values, "size")
                    and getattr(values, "shape", ())
                    and hasattr(self, "mlp")
                ):
                    rows = values.size // values.shape[-1]
                    support_key = id(self.mlp)
                    support = support_cache.get(support_key)
                    if (
                        support is None
                        or support[0] is not self.mlp
                        or support[1] != values.shape[-1]
                        or support[2] != values.dtype
                    ):
                        support = (
                            self.mlp,
                            values.shape[-1],
                            values.dtype,
                            _supports_quantized_residual_mlp(
                                self.mlp,
                                values,
                                values,
                                quantized_linear,
                            ),
                        )
                        support_cache[support_key] = support
                    supports_quantized_residual = (
                        rows >= quantized_mlp_min_rows
                        and (quantized_mlp_max_rows is None or rows <= quantized_mlp_max_rows)
                        and support[3]
                    )
                if (
                    dense_mlp is not None
                    and hasattr(values, "size")
                    and getattr(values, "shape", ())
                    and hasattr(self, "mlp")
                ):
                    supports_dense_residual = _supports_dense_residual_mlp(
                        self.mlp,
                        values,
                        values,
                        dense_mlp,
                    )
                selected = (
                    mlx_add_rms_norm_selection(values, self.post_attention_layernorm.eps)
                    if fuse_residual_rms
                    else None
                )
                if (
                    not fuse_residual_rms
                    and not supports_quantized_residual
                    and not supports_dense_residual
                ):
                    return original_call(self, values, mask, cache)
                if (
                    selected is not None
                    and selected.algorithm == "mlx"
                    and not supports_quantized_residual
                    and not supports_dense_residual
                ):
                    return original_call(self, values, mask, cache)

                # A block binding its attention to neither name is one this replacement cannot
                # reproduce, so hand it back rather than guess. Checked here rather than at patch
                # time because a hybrid architecture binds different names on different layers.
                attention = _attention_module(self)
                if attention is None:
                    return original_call(self, values, mask, cache)
                attention_output = attention(self.input_layernorm(values), mask, cache)
                if (
                    fuse_residual_rms
                    and (selected is None or selected.algorithm != "mlx")
                    and _supports_metile_residual_rms_norm(
                        values, attention_output, self.post_attention_layernorm
                    )
                ):
                    hidden, normalized = _execute_residual_rms_graph(
                        values, attention_output, self.post_attention_layernorm
                    )
                else:
                    hidden = values + attention_output
                    normalized = self.post_attention_layernorm(hidden)
                if supports_quantized_residual:
                    return _execute_quantized_mlp(self.mlp, normalized, hidden)
                if supports_dense_residual:
                    return _execute_dense_mlp(
                        self.mlp,
                        normalized,
                        hidden,
                        dense_mlp,
                        dense_swiglu,
                    )
                return hidden + self.mlp(normalized)

            return replacement

        metile_transformer_block = make_replacement(original, quantized_support_cache)

        metile_transformer_block._metile_original = original
        replacements.append((block_class, "__call__", original))
        block_class.__call__ = metile_transformer_block


def _patch_quantized_mlp(
    model,
    replacements,
    quantized_linear,
    *,
    min_rows=1,
    max_rows=None,
):
    if min_rows < 1:
        raise ValueError("quantized MLP minimum rows must be positive")
    if max_rows is not None and max_rows < min_rows:
        raise ValueError("quantized MLP maximum rows must not be smaller than its minimum")
    classes = [type(layer.mlp) for layer in _model_layers(model) if hasattr(layer, "mlp")]
    if model is None:
        classes.extend(_registry_classes(_GATED_MLP_CLASSES))

    for mlp_class in dict.fromkeys(classes):
        if not _recognised(mlp_class, _GATED_MLP_CLASSES):
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
                if max_rows is not None and rows > max_rows:
                    return original_call(self, values)
                if not _supports_quantized_mlp(self, values, quantized_linear):
                    return original_call(self, values)
                return _execute_quantized_mlp(self, values)

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


def _patch_dense_mlp(dense_mlp, replacements):
    if dense_mlp is None:
        return
    for module, *_ in dense_mlp.weights.values():
        patched_class = dense_mlp.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", dense_mlp.patched_class(module))


def _patch_compressed_down(compressed_down, replacements):
    if compressed_down is None:
        return
    for module, _ in compressed_down.weights.values():
        patched_class = compressed_down.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", compressed_down.patched_class(module))


def _patch_compressed_gate_up(compressed_gate_up, replacements):
    if compressed_gate_up is None:
        return
    for module, gate, _, up, _ in compressed_gate_up.layers.values():
        if compressed_gate_up.implementation == "fused" and _supports_compressed_gate_up_fusion(
            module
        ):
            patched_class = compressed_gate_up.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(
                module,
                "__class__",
                compressed_gate_up.fused_patched_class(module),
            )
            continue
        for module in (gate, up):
            patched_class = compressed_gate_up.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(module, "__class__", compressed_gate_up.patched_class(module))


def _patch_compressed_attention(compressed_attention, replacements):
    if compressed_attention is None:
        return
    for _, projections in compressed_attention.layers.values():
        for module, _ in projections:
            patched_class = compressed_attention.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(module, "__class__", compressed_attention.patched_class(module))


def _patch_compressed_vocab(compressed_vocab, replacements):
    if compressed_vocab is None or compressed_vocab.weight is None:
        return
    module = compressed_vocab.module
    patched_class = compressed_vocab.patched_classes.get(id(module))
    original_class = type(module)
    if original_class is patched_class:
        return
    replacements.append((module, "__class__", original_class))
    object.__setattr__(module, "__class__", compressed_vocab.patched_class())
