"""core layer of the mlx_lm integration."""

from __future__ import annotations

import copy
import time

from metile.integrations.mlx_lm._state import (
    _ATTENTION_ATTRIBUTES,
    _STRUCTURE,
    _unsupported_decode_shapes,
)
from metile.ir.graph_ir import TensorSpec


def _attention_module(block):
    """The attention a block will call, or None if it binds none this pass understands."""
    for name in _ATTENTION_ATTRIBUTES:
        found = getattr(block, name, None)
        if found is not None:
            return found
    return None


def _structurally_matches(cls, registry):
    """Whether a class carries the parts the registry's replacement needs.

    Read off the class, so it works for architectures nobody enumerated. Requires __call__ to be defined
    on the class itself rather than inherited, because a class that does not define one has nothing for
    meTile to replace and `getattr` would hand back a fresh method-wrapper from the metaclass.
    """
    required = _STRUCTURE.get(id(registry))
    if required is None:
        return False
    if not any("__call__" in vars(klass) for klass in cls.__mro__):
        return False
    if not all(
        hasattr(cls, name) or name in getattr(cls, "__annotations__", {}) for name in required
    ):
        # Attributes are usually set in __init__ rather than declared, so fall back to the source of
        # __call__: a replacement only works if the body actually reaches those names.
        source = _call_source(cls)
        return source is not None and all(name in source for name in required)
    return True


def _call_source(cls):
    import inspect as _inspect

    for klass in cls.__mro__:
        if "__call__" in vars(klass):
            try:
                return _inspect.getsource(vars(klass)["__call__"])
            except (OSError, TypeError):
                return None
    return None


def _recognised(cls, registry, structural=True):
    """Whether meTile is allowed to replace this class's __call__.

    The named pairs are the combinations whose arithmetic has been checked against MLX in the model
    matrix. Structural matches are candidates that `metile.compile` verifies at patch time.
    """
    if (cls.__module__, cls.__name__) in registry:
        return True
    return structural and _structurally_matches(cls, registry)


def _registry_classes(registry):
    """Import and return the classes in a registry, skipping any this mlx-lm does not have.

    Used when patching without a model in hand. Skipping rather than raising because the
    registry spans several mlx-lm versions and a missing architecture is not an error.
    """
    import importlib

    found = []
    for module_name, class_name in sorted(registry):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            found.append(cls)
    return found


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


def _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if decode_steps < 1:
        raise ValueError("prompt preparation requires positive decode steps")
    cache = make_prompt_cache(model)
    start = time.perf_counter_ns()
    logits = model(sample_tokens, cache=cache)
    mx.eval(logits)
    elapsed = (time.perf_counter_ns() - start) * 1e-9
    trajectory_cache = copy.deepcopy(cache)
    decode_tokens = []
    for step in range(decode_steps):
        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(token)
        decode_tokens.append(token)
        if step + 1 < decode_steps:
            logits = model(token, cache=trajectory_cache)
            mx.eval(logits)
    return cache, elapsed, tuple(decode_tokens)


def _decode_shape_key(queries, keys):
    return (queries.shape[1], keys.shape[1], queries.shape[-1], str(queries.dtype))


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
        and str(queries.dtype) in ("mlx.core.bfloat16", "mlx.core.float16", "mlx.core.float32")
        and _decode_shape_key(queries, keys) not in _unsupported_decode_shapes
    )


def _supports_metile_rms_norm(values, weight):
    return (
        values.ndim >= 1
        and weight.ndim == 1
        and values.shape[-1] == weight.shape[0]
        and values.dtype == weight.dtype
        and str(values.dtype) in ("mlx.core.bfloat16", "mlx.core.float16", "mlx.core.float32")
    )


def _tensor_spec(value):
    dtype = {
        "mlx.core.bfloat16": "bf16",
        "mlx.core.float16": "f16",
        "mlx.core.float32": "f32",
    }[str(value.dtype)]
    return TensorSpec(tuple(value.shape), dtype)


def _model_layers(model):
    if model is None:
        return ()
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", ())
    return layers
