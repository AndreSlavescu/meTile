"""Native attention orchestration; device kernels live in metile_kernels.

The training path uses separate raw maxima/log denominators and unrounded
FP32 outputs following KohakuFA's precision analysis:
https://github.com/KohakuBlueleaf/KohakuFA/blob/041ac512ac474709ad910e505c545a1ab94853c2/docs/precision.md

This is a bounded native backend, not an MLX autograd registration. Inputs are
contiguous BHSD FP16/FP32 tensors with finite FP32 dot products. Saved input
buffers must not be mutated before backward. Gradients accumulate into FP32.
"""

from dataclasses import dataclass
from math import isfinite, prod

import numpy as np

from metile.runtime.buffer import MtileBuffer
from metile_kernels.stable_attention import (
    stable_attention_backward_key_value,
    stable_attention_backward_query,
    stable_attention_forward,
)

_INDEX_LIMIT = (1 << 31) - 1


def _metadata(value, name):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    shape = tuple(value.shape)
    if not shape or any(size <= 0 for size in shape) or prod(shape) > _INDEX_LIMIT:
        raise ValueError(f"{name} must have positive dimensions and signed-32-bit indexing")
    return shape, np.dtype(value.dtype)


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer(data=value)


def _positive_scale(scale):
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("attention scale must be positive finite FP32")
    scale = float(scale)
    if (
        not isfinite(scale)
        or scale < float(np.finfo(np.float32).smallest_subnormal)
        or scale > float(np.finfo(np.float32).max)
    ):
        raise ValueError("attention scale must be positive finite FP32")
    return float(np.float32(scale))


@dataclass(frozen=True)
class AttentionContext:
    query: MtileBuffer
    key: MtileBuffer
    value: MtileBuffer
    mask: MtileBuffer
    precise_output: MtileBuffer
    row_maximum: MtileBuffer
    row_log_denominator: MtileBuffer
    scale: float
    causal: bool
    causal_offset: int
    has_mask: bool

    @property
    def kernel_options(self):
        return {
            "D": self.query.shape[-1],
            "Q_HEADS": self.query.shape[1],
            "KV_HEADS": self.key.shape[1],
            "CAUSAL": self.causal,
            "CAUSAL_OFFSET": self.causal_offset,
            "HAS_MASK": self.has_mask,
            "BLOCK": 32,
            "STRICT_MATH": True,
        }


def attention_forward(
    query, key, value, *, scale=None, causal=False, causal_offset=None, mask=None
):
    """Return ``(output, context)`` for stable dense, masked, or causal GQA.

    Input layouts are [batch, heads, sequence, dimension]; K/V heads divide
    query heads. Default causal alignment is bottom-right, including cached
    suffix queries. Set causal_offset=0 for top-left alignment. A mask is
    boolean/uint8 (nonzero means visible), not an additive score bias.
    """
    query_shape, dtype = _metadata(query, "query")
    key_shape, key_dtype = _metadata(key, "key")
    value_shape, value_dtype = _metadata(value, "value")
    if len(query_shape) != 4 or len(key_shape) != 4 or value_shape != key_shape:
        raise ValueError("attention requires BHSD query/key/value and identical K/V shapes")
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)) or not (
        dtype == key_dtype == value_dtype
    ):
        raise ValueError("attention requires matching FP16 or FP32 input dtypes")
    batch, query_heads, query_length, dimension = query_shape
    if query_shape[0] != key_shape[0] or dimension != key_shape[-1]:
        raise ValueError("attention batch and head dimensions must match")
    kv_heads, key_length = key_shape[1:3]
    if query_heads % kv_heads or dimension < 32 or dimension > 256 or dimension % 32:
        raise ValueError("attention requires divisible heads and dimension 32..256 in steps of 32")
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    if causal_offset is None:
        causal_offset = key_length - query_length if causal else 0
    if type(causal_offset) is not int or not (
        -_INDEX_LIMIT <= causal_offset <= _INDEX_LIMIT - query_length
    ):
        raise ValueError("causal_offset and query positions must fit signed-32-bit indexing")
    if not causal and causal_offset != 0:
        raise ValueError("causal_offset requires causal attention")
    scale = _positive_scale(dimension**-0.5 if scale is None else scale)
    mask_shape = (batch, query_heads, query_length, key_length)
    if prod(mask_shape) > _INDEX_LIMIT:
        raise ValueError("attention row/key indexing exceeds signed-32-bit indexing")
    if mask is not None:
        shape, mask_dtype = _metadata(mask, "mask")
        if mask_dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
            raise TypeError("attention mask must be boolean or uint8, not an additive bias")
        if isinstance(mask, MtileBuffer):
            if shape != mask_shape or mask_dtype != np.dtype(np.uint8):
                raise ValueError("device mask must be uint8 with shape [B,Hq,Sq,Sk]")
        else:
            try:
                mask = np.broadcast_to(mask, mask_shape).astype(np.uint8)
            except ValueError as error:
                raise ValueError("attention mask cannot broadcast to [B,Hq,Sq,Sk]") from error
    query_buffer, key_buffer, value_buffer = map(_buffer, (query, key, value))
    mask_buffer = _buffer(mask if mask is not None else np.zeros(1, dtype=np.uint8))
    output = MtileBuffer.empty(query_shape, dtype=dtype)
    precise_output = MtileBuffer.empty(query_shape, dtype=np.float32)
    row_maximum = MtileBuffer.empty(query_shape[:-1], dtype=np.float32)
    row_log_denominator = MtileBuffer.empty(query_shape[:-1], dtype=np.float32)
    context = AttentionContext(
        query_buffer,
        key_buffer,
        value_buffer,
        mask_buffer,
        precise_output,
        row_maximum,
        row_log_denominator,
        scale,
        causal,
        causal_offset,
        mask is not None,
    )
    stable_attention_forward[(batch * query_heads * query_length,)](
        query_buffer,
        key_buffer,
        value_buffer,
        mask_buffer,
        output,
        precise_output,
        row_maximum,
        row_log_denominator,
        query_length,
        key_length,
        scale,
        **context.kernel_options,
    )
    return output, context


def attention_backward(context, gradient):
    """Return FP32 ``(dquery, dkey, dvalue)`` without floating-point atomics."""
    if not isinstance(context, AttentionContext):
        raise TypeError("attention backward requires an AttentionContext")
    shape, dtype = _metadata(gradient, "gradient")
    if shape != context.query.shape or dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("gradient must match output shape and use FP16 or FP32")
    gradient = _buffer(gradient)
    grad_query = MtileBuffer.empty(context.query.shape, dtype=np.float32)
    grad_key = MtileBuffer.empty(context.key.shape, dtype=np.float32)
    grad_value = MtileBuffer.empty(context.value.shape, dtype=np.float32)
    batch, query_heads, query_length, _ = context.query.shape
    kv_heads, key_length = context.key.shape[1:3]
    arguments = (
        context.query,
        context.key,
        context.value,
        context.mask,
        context.precise_output,
        context.row_maximum,
        context.row_log_denominator,
        gradient,
    )
    stable_attention_backward_query[(batch * query_heads * query_length,)](
        *arguments, grad_query, query_length, key_length, context.scale, **context.kernel_options
    )
    stable_attention_backward_key_value[(batch * kv_heads * key_length,)](
        *arguments,
        grad_key,
        grad_value,
        query_length,
        key_length,
        context.scale,
        **context.kernel_options,
    )
    return grad_query, grad_key, grad_value
