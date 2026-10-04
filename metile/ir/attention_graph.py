"""Explicit native attention operators, without pattern-based semantic rewrites.

Softmax attention uses BHSD, while delta recurrences use BTHD and carry an
explicit state value between nodes. DCA inputs are already RoPE-transformed
branches. These helpers do not infer models, insert rotations/normalization,
or register framework autodiff. The native graph executor supplies explicit VJPs.
"""

from math import isfinite, prod

from metile.ir.graph_ir import GraphBuilder, GraphNode, GraphValue, TensorSpec

DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024
_INDEX_LIMIT = (1 << 31) - 1
_FP32_MAX = 3.4028234663852886e38
_FP32_MIN = 1.401298464324817e-45


def _spec(value, name):
    if not isinstance(value, GraphValue):
        raise TypeError(f"{name} must be a GraphValue")
    shape = value.spec.shape
    if any(type(size) is not int or size <= 0 for size in shape) or prod(shape) > _INDEX_LIMIT:
        raise ValueError(f"{name} requires positive integer dimensions and signed 32-bit indexing")
    return value.spec


def _scale(value, *, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("scale must be a real scalar")
    value = float(value)
    if not isfinite(value) or abs(value) > _FP32_MAX or (positive and value < _FP32_MIN):
        raise ValueError(
            "scale must be representable finite FP32" + (" and positive" if positive else "")
        )
    return value


def _workspace(limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_workspace_bytes must be a positive integer")
    return limit


def _attention_specs(query, key, value, mask):
    query_spec, key_spec, value_spec = (
        _spec(item, name) for item, name in ((query, "query"), (key, "key"), (value, "value"))
    )
    if len(query_spec.shape) != 4 or len(key_spec.shape) != 4 or key_spec != value_spec:
        raise ValueError("attention requires BHSD query/key/value and identical K/V specifications")
    if query_spec.dtype not in {"f16", "f32"} or query_spec.dtype != key_spec.dtype:
        raise ValueError("attention requires matching FP16 or FP32 dtypes")
    batch, query_heads, query_length, dimension = query_spec.shape
    if (
        batch != key_spec.shape[0]
        or dimension != key_spec.shape[-1]
        or query_heads % key_spec.shape[1]
    ):
        raise ValueError("attention requires matching batch/dimension and divisible GQA heads")
    if dimension < 32 or dimension > 256 or dimension % 32:
        raise ValueError("attention head dimension must be 32..256 in steps of 32")
    mask_shape = (batch, query_heads, query_length, key_spec.shape[2])
    if prod(mask_shape) > _INDEX_LIMIT:
        raise ValueError("attention mask indexing exceeds signed 32-bit range")
    if mask is not None:
        mask_spec = _spec(mask, "mask")
        if mask_spec.shape != mask_shape or mask_spec.dtype not in {"bool", "u8"}:
            raise ValueError("attention graph mask requires exact [B,Hq,Sq,Sk] bool/uint8 metadata")
    return query_spec, key_spec


def stable_attention(
    builder,
    query,
    key,
    value,
    *,
    scale=None,
    causal=False,
    causal_offset=None,
    mask=None,
    name=None,
):
    """Build dense/masked/causal BHSD attention; default causality is bottom-right."""
    query_spec, key_spec = _attention_specs(query, key, value, mask)
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    query_length, key_length = query_spec.shape[2], key_spec.shape[2]
    if causal_offset is None:
        causal_offset = key_length - query_length if causal else 0
    if (
        type(causal_offset) is not int
        or not -_INDEX_LIMIT <= causal_offset <= _INDEX_LIMIT - query_length
    ):
        raise ValueError("causal offset and query positions must fit signed 32-bit indexing")
    if not causal and causal_offset != 0:
        raise ValueError("causal_offset requires causal attention")
    scale = _scale(query_spec.shape[-1] ** -0.5 if scale is None else scale, positive=True)
    inputs = (query, key, value) + (() if mask is None else (mask,))
    return builder._node(
        "stable_attention",
        inputs,
        {
            "scale": scale,
            "causal": causal,
            "causal_offset": causal_offset,
            "has_mask": mask is not None,
        },
        (query_spec,),
        name,
    )[0]


def _delta_attention(
    builder,
    operation,
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    scale,
    max_workspace_bytes,
    name,
):
    query_spec = _spec(query, "query")
    key_spec = _spec(key, "key")
    value_spec = _spec(value, "value")
    if len(query_spec.shape) != 4 or key_spec != query_spec or len(value_spec.shape) != 4:
        raise ValueError("delta attention requires BTHD query/key/value and identical Q/K metadata")
    batch, sequence, heads, key_dimension = query_spec.shape
    value_dimension = value_spec.shape[-1]
    if value_spec.shape[:3] != (batch, sequence, heads):
        raise ValueError("delta attention batch, sequence, and head dimensions must match")
    if query_spec.dtype != "f32" or value_spec.dtype != "f32":
        raise ValueError("delta attention graph inputs must be FP32")
    if heads > 128 or key_dimension > 256 or value_dimension > 256:
        raise ValueError("delta attention requires H<=128 and K,V<=256")
    prefix = (batch, sequence, heads)
    decay_shape = (*prefix, key_dimension) if operation == "kimi_delta_attention" else prefix
    state_spec = TensorSpec((batch, heads, key_dimension, value_dimension), "f32")
    for item, label, expected in (
        (log_decay, "log_decay", TensorSpec(decay_shape, "f32")),
        (beta, "beta", TensorSpec(prefix, "f32")),
        (initial_state, "initial_state", state_spec),
    ):
        if _spec(item, label) != expected:
            raise ValueError(f"{operation} {label} requires {expected.shape}/{expected.dtype}")
    return builder._node(
        operation,
        (query, key, value, log_decay, beta, initial_state),
        {
            "scale": _scale(scale, positive=False),
            "max_workspace_bytes": _workspace(max_workspace_bytes),
        },
        (value_spec, state_spec),
        name,
    )


def gated_delta_attention(
    builder,
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    *,
    scale=1.0,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
    name=None,
):
    """Build scalar-decay GDN, returning (output, final_state); no hidden cache."""
    return _delta_attention(
        builder,
        "gated_delta_attention",
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        scale,
        max_workspace_bytes,
        name,
    )


def kimi_delta_attention(
    builder,
    query,
    key,
    value,
    log_decay,
    beta,
    initial_state,
    *,
    scale=1.0,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
    name=None,
):
    """Build channel-decay KDA core, returning (output, final_state), not a model."""
    return _delta_attention(
        builder,
        "kimi_delta_attention",
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        scale,
        max_workspace_bytes,
        name,
    )


def dual_chunk_attention(
    builder,
    query_intra,
    query_successive,
    query_inter,
    key,
    value,
    *,
    chunk_size,
    local_window,
    query_start=0,
    key_start=0,
    mask=None,
    scale=None,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
    name=None,
):
    """Build causal DCA from three explicitly RoPE-transformed query branches."""
    query_spec, key_spec = _attention_specs(query_intra, key, value, mask)
    if (
        _spec(query_successive, "query_successive") != query_spec
        or _spec(query_inter, "query_inter") != query_spec
    ):
        raise ValueError("DCA query branches must have identical metadata")
    if type(chunk_size) is not int or not 0 < chunk_size <= _INDEX_LIMIT:
        raise ValueError("chunk_size must be a positive signed 32-bit integer")
    if type(local_window) is not int or not 0 <= local_window < chunk_size:
        raise ValueError("local_window must be an integer in [0, chunk_size)")
    for start, length in ((query_start, query_spec.shape[2]), (key_start, key_spec.shape[2])):
        if type(start) is not int or start < 0 or start + length - 1 > _INDEX_LIMIT:
            raise ValueError("DCA global positions must fit nonnegative signed 32-bit indexing")
    if 3 * prod(query_spec.shape[:-1]) > _INDEX_LIMIT:
        raise ValueError("DCA statistics exceed signed 32-bit indexing")
    scale = _scale(query_spec.shape[-1] ** -0.5 if scale is None else scale, positive=True)
    inputs = (query_intra, query_successive, query_inter, key, value) + (
        () if mask is None else (mask,)
    )
    return builder._node(
        "dual_chunk_attention",
        inputs,
        {
            "chunk_size": chunk_size,
            "local_window": local_window,
            "query_start": query_start,
            "key_start": key_start,
            "has_mask": mask is not None,
            "scale": scale,
            "max_workspace_bytes": _workspace(max_workspace_bytes),
        },
        (query_spec,),
        name,
    )[0]


_BUILDERS = {
    "stable_attention": stable_attention,
    "gated_delta_attention": gated_delta_attention,
    "kimi_delta_attention": kimi_delta_attention,
    "dual_chunk_attention": dual_chunk_attention,
}
_ATTRIBUTES = {
    "stable_attention": {"scale", "causal", "causal_offset", "has_mask"},
    "gated_delta_attention": {"scale", "max_workspace_bytes"},
    "kimi_delta_attention": {"scale", "max_workspace_bytes"},
    "dual_chunk_attention": {
        "chunk_size",
        "local_window",
        "query_start",
        "key_start",
        "has_mask",
        "scale",
        "max_workspace_bytes",
    },
}


def validate_attention_node(node: GraphNode):
    """Recheck an explicit node, including manually constructed or mutated metadata."""
    if node.op not in _BUILDERS:
        raise ValueError(f"unsupported native attention graph operation: {node.op}")
    if node.side_effect:
        raise ValueError("native attention graph nodes must be functional, not effectful")
    if set(node.attrs) != _ATTRIBUTES[node.op]:
        raise ValueError(f"{node.op} requires exactly its declared semantic attributes")
    arguments = list(node.inputs)
    options = dict(node.attrs)
    if node.op in {"stable_attention", "dual_chunk_attention"}:
        base_count = 3 if node.op == "stable_attention" else 5
        has_mask = options.pop("has_mask", None)
        if type(has_mask) is not bool or len(arguments) != base_count + int(has_mask):
            raise ValueError("attention node input count must agree with bool has_mask metadata")
        options["mask"] = arguments.pop() if has_mask else None
    try:
        inferred = _BUILDERS[node.op](GraphBuilder(), *arguments, **options)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {node.op} node {node.name}: {error}") from error
    inferred = inferred if isinstance(inferred, tuple) else (inferred,)
    if tuple(value.spec for value in node.outputs) != tuple(value.spec for value in inferred):
        raise ValueError(f"{node.op} output specifications do not match its semantic contract")
    for index, output in enumerate(node.outputs):
        if output.producer is not node or output.output_index != index:
            raise ValueError("attention node outputs have invalid producer/index metadata")


__all__ = [
    "dual_chunk_attention",
    "gated_delta_attention",
    "kimi_delta_attention",
    "stable_attention",
    "validate_attention_node",
]
