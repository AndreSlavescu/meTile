"""Bounded causal Dual Chunk Attention over explicitly RoPE-transformed inputs.

The position helper follows HKUNLP/ChunkLlama's cache construction and branch
selection at this pinned revision (an independent implementation):
https://github.com/HKUNLP/ChunkLlama/blob/2add4d7c99d24dcc1ab03414cc602abb2e28cf2c/chunkllama_attn_replace.py
With c=chunk_size-local_window, positions are Q_intra=i%c,
Q_successive=min(i%c+c,chunk_size), Q_inter=min(2*c-1,chunk_size), K=j%c.
The local window preserves nearby cross-boundary positions; it does not mask
out distant keys. The clamp is inclusive, matching that implementation.

Qwen2.5-1M describes the three patterns and a separate YaRN temperature in
section 5.1: https://arxiv.org/html/2501.15383v1#S5.SS1 . This backend does not
choose model-specific RoPE frequencies, rotations, interpolation, or YaRN
scaling. Supply three transformed queries and a transformed key explicitly;
the backward result is with respect to those inputs, before their RoPE VJPs.
This is not a Qwen model adapter, sparse implementation, or long-context
performance claim.

Inputs are contiguous BHSD FP16/FP32 arrays. Three uint8 masks cost
3*B*Hq*Sq*Sk bytes. Every forward branch contributes to one global softmax;
backward uses its unrounded FP32 output and separate raw maximum/log sum.
The workspace budget bounds newly allocated outputs and temporaries for each
call, excluding NumPy input copies and already retained forward buffers.
Caller-owned device inputs and saved buffers must not change before backward.
Finite FP32 dot products and representable attention intermediates are required.
"""

from dataclasses import dataclass, field, replace
from math import isfinite, prod

import numpy as np

from metile.runtime.buffer import MtileBuffer

DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024
_INDEX_LIMIT = (1 << 31) - 1


@dataclass(frozen=True)
class DualChunkPositionIndices:
    query_intra: np.ndarray
    query_successive: np.ndarray
    query_inter: np.ndarray
    key: np.ndarray


@dataclass(frozen=True)
class DualChunkAttentionResult:
    output: MtileBuffer
    precise_output: MtileBuffer
    row_maximum: MtileBuffer
    row_log_denominator: MtileBuffer
    _contexts: tuple = field(repr=False)


@dataclass(frozen=True)
class DualChunkAttentionGradients:
    query_intra: MtileBuffer
    query_successive: MtileBuffer
    query_inter: MtileBuffer
    key: MtileBuffer
    value: MtileBuffer


def _chunk_length(chunk_size, local_window):
    if type(chunk_size) is not int or not 0 < chunk_size <= _INDEX_LIMIT:
        raise ValueError("chunk_size must be a positive signed 32-bit integer")
    if type(local_window) is not int or not 0 <= local_window < chunk_size:
        raise ValueError("local_window must be an integer in [0, chunk_size)")
    return chunk_size - local_window


def _positions(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values > _INDEX_LIMIT):
        raise ValueError(f"{name} must contain nonnegative signed 32-bit positions")
    return values.astype(np.int64)


def dual_chunk_position_indices(query_positions, key_positions, *, chunk_size, local_window):
    """Return integer RoPE cache positions; no rotation or temperature is applied."""
    chunk_length = _chunk_length(chunk_size, local_window)
    query_positions = _positions(query_positions, "query_positions")
    key_positions = _positions(key_positions, "key_positions")
    intra = query_positions % chunk_length
    return DualChunkPositionIndices(
        intra,
        np.minimum(intra + chunk_length, chunk_size),
        np.full_like(intra, min(2 * chunk_length - 1, chunk_size)),
        key_positions % chunk_length,
    )


def _array(value, name, shape=None, dtype=None):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    actual_shape = tuple(value.shape)
    if (
        not actual_shape
        or any(size <= 0 for size in actual_shape)
        or prod(actual_shape) > _INDEX_LIMIT
    ):
        raise ValueError(f"{name} requires positive dimensions and signed 32-bit indexing")
    if shape is not None and actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and np.dtype(value.dtype) != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return actual_shape, np.dtype(value.dtype)


def _budget(required, maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("max_workspace_bytes must be a positive integer")
    if required > maximum:
        raise ValueError(f"DCA needs up to {required} workspace/output bytes; budget is {maximum}")


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer.from_numpy(value)


def dual_chunk_attention_forward(
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
):
    """Return output and saved global statistics for a bounded DCA core.

    Query/key positions are consecutive intervals starting at the given global
    offsets, which also determine causal visibility. Mask, when supplied, has
    exact [B,Hq,Sq,Sk] shape and bool/uint8 entries (nonzero means visible).
    Scale is an explicit positive FP32 temperature, defaulting to 1/sqrt(D).
    """
    chunk_length = _chunk_length(chunk_size, local_window)
    query_shape, dtype = _array(query_intra, "query_intra")
    key_shape, _ = _array(key, "key", dtype=dtype)
    if len(query_shape) != 4 or len(key_shape) != 4:
        raise ValueError("DCA requires BHSD query and key tensors")
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("DCA requires matching FP16 or FP32 inputs")
    _array(query_successive, "query_successive", query_shape, dtype)
    _array(query_inter, "query_inter", query_shape, dtype)
    _array(value, "value", key_shape, dtype)
    batch, query_heads, query_length, dimension = query_shape
    kv_heads, key_length = key_shape[1:3]
    if batch != key_shape[0] or dimension != key_shape[-1] or query_heads % kv_heads:
        raise ValueError("DCA requires matching batch/dimension and divisible GQA heads")
    if dimension < 32 or dimension > 256 or dimension % 32:
        raise ValueError("DCA requires head dimension 32..256 in steps of 32")
    for start, length in ((query_start, query_length), (key_start, key_length)):
        if type(start) is not int or start < 0 or start + length - 1 > _INDEX_LIMIT:
            raise ValueError(
                "global query/key positions must fit nonnegative signed 32-bit integers"
            )
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("scale must be positive finite FP32")
    scale = dimension**-0.5 if scale is None else float(scale)
    if not isfinite(scale) or not float(np.finfo(np.float32).smallest_subnormal) <= scale <= float(
        np.finfo(np.float32).max
    ):
        raise ValueError("scale must be positive finite FP32")
    scale = float(np.float32(scale))
    rows = prod(query_shape[:-1])
    mask_shape = (*query_shape[:-1], key_length)
    if prod(mask_shape) > _INDEX_LIMIT or rows * 3 > _INDEX_LIMIT:
        raise ValueError("DCA masks/statistics exceed signed 32-bit indexing")
    if mask is not None:
        _, mask_dtype = _array(mask, "mask", mask_shape)
        if mask_dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
            raise TypeError("mask must be bool/uint8, not an additive bias")
        if isinstance(mask, MtileBuffer) and mask_dtype != np.dtype(np.uint8):
            raise TypeError("device mask must be uint8")
    required = 3 * prod(mask_shape) + (16 + 4 * dtype.itemsize) * prod(query_shape) + 44 * rows
    _budget(required, max_workspace_bytes)

    from metile.backends.attention import attention_forward
    from metile_kernels.dual_chunk_attention import dual_chunk_merge, dual_chunk_partition

    queries = tuple(_buffer(query) for query in (query_intra, query_successive, query_inter))
    key_buffer, value_buffer = _buffer(key), _buffer(value)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    mask_buffer = _buffer(mask if mask is not None else np.zeros(1, dtype=np.uint8))
    branch_masks = [MtileBuffer.empty(mask_shape, dtype=np.uint8) for _ in range(3)]
    live = MtileBuffer.empty((3, rows))
    dual_chunk_partition[(rows,)](
        mask_buffer,
        *branch_masks,
        live,
        query_length,
        key_length,
        rows,
        CHUNK_LEN=chunk_length,
        QUERY_START=query_start,
        KEY_START=key_start,
        HAS_MASK=mask is not None,
        BLOCK=32,
        STRICT_MATH=True,
    )
    branches = [
        attention_forward(query, key_buffer, value_buffer, scale=scale, mask=branch_mask)
        for query, branch_mask in zip(queries, branch_masks)
    ]
    contexts = [context for _, context in branches]
    output = MtileBuffer.empty(query_shape, dtype=dtype)
    precise_output = MtileBuffer.empty(query_shape)
    row_maximum = MtileBuffer.empty(query_shape[:-1])
    row_log_denominator = MtileBuffer.empty(query_shape[:-1])
    dual_chunk_merge[(rows,)](
        *(context.precise_output for context in contexts),
        *(context.row_maximum for context in contexts),
        *(context.row_log_denominator for context in contexts),
        live,
        output,
        precise_output,
        row_maximum,
        row_log_denominator,
        rows,
        scale,
        D=dimension,
        BLOCK=32,
        STRICT_MATH=True,
    )
    global_contexts = tuple(
        replace(
            context,
            precise_output=precise_output,
            row_maximum=row_maximum,
            row_log_denominator=row_log_denominator,
        )
        for context in contexts
    )
    return DualChunkAttentionResult(
        output, precise_output, row_maximum, row_log_denominator, global_contexts
    )


def dual_chunk_attention_backward(
    result,
    gradient,
    *,
    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
):
    """Return FP32 VJPs of all five transformed inputs, without RoPE adjoints."""
    if not isinstance(result, DualChunkAttentionResult) or len(result._contexts) != 3:
        raise TypeError("backward requires a saved DualChunkAttentionResult")
    context = result._contexts[0]
    _, dtype = _array(gradient, "gradient", context.query.shape)
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("gradient must be FP16 or FP32")
    key_elements = prod(context.key.shape)
    _budget(12 * prod(context.query.shape) + 32 * key_elements, max_workspace_bytes)

    from metile.backends.attention import attention_backward
    from metile_kernels.dual_chunk_attention import dual_chunk_sum_key_value_gradients

    gradient = _buffer(gradient)
    branch_gradients = [attention_backward(branch, gradient) for branch in result._contexts]
    grad_key = MtileBuffer.empty(context.key.shape)
    grad_value = MtileBuffer.empty(context.value.shape)
    dual_chunk_sum_key_value_gradients[((key_elements + 255) // 256,)](
        *(branch[1] for branch in branch_gradients),
        *(branch[2] for branch in branch_gradients),
        grad_key,
        grad_value,
        key_elements,
        BLOCK=256,
        STRICT_MATH=True,
    )
    return DualChunkAttentionGradients(
        *(branch[0] for branch in branch_gradients), grad_key, grad_value
    )
