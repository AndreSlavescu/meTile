"""Native training entry points for activations and pre-gathered rotary tables.

Returns explicit contexts and first-order adjoints, not framework autograd
registrations. NumPy inputs are copied; Buffer inputs are retained and must not
be mutated before backward. All returned gradients use FP32 storage.
"""

from dataclasses import dataclass
from math import prod

import numpy as np

from metile.runtime.buffer import MtileBuffer


def _array(value, name, shape=None):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    if np.dtype(value.dtype) not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(f"{name} must use FP16 or FP32")
    if not value.shape or any(size <= 0 for size in value.shape) or prod(value.shape) >= 2**31:
        raise ValueError(f"{name} requires positive dimensions and signed-32-bit indexing")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer(data=value)


@dataclass(frozen=True)
class ActivationContext:
    values: MtileBuffer
    up: MtileBuffer | None
    kind: str


def activation_forward(values, *, kind="silu", up=None):
    """Return (output, context); supplying ``up`` makes a gated activation."""
    from metile_kernels.training_activations import _validate
    from metile_kernels.training_activations import activation_forward as kernel

    _validate(kind, up is not None, 128)
    _array(values, "values")
    if up is not None:
        _array(up, "up", values.shape)
        if up.dtype != values.dtype:
            raise TypeError("up must have the same dtype as values")
    values = _buffer(values)
    up = _buffer(up) if up is not None else None
    output = MtileBuffer.empty(values.shape, values.dtype)
    count = prod(values.shape)
    kernel[((count + 127) // 128,)](
        values,
        up if up is not None else values,
        output,
        count,
        KIND=kind,
        GATED=up is not None,
        BLOCK=128,
        STRICT_MATH=True,
    )
    return output, ActivationContext(values, up, kind)


def activation_backward(context, gradient):
    """Return ``dvalues`` or ``(dvalues, dup)`` for a gated activation."""
    from metile_kernels.training_activations import activation_backward as kernel

    if not isinstance(context, ActivationContext):
        raise TypeError("activation backward requires an ActivationContext")
    _array(gradient, "gradient", context.values.shape)
    grad_values = MtileBuffer.empty(context.values.shape, np.float32)
    grad_up = (
        MtileBuffer.empty(context.values.shape, np.float32) if context.up is not None else None
    )
    count = prod(context.values.shape)
    kernel[((count + 127) // 128,)](
        context.values,
        context.up if context.up is not None else context.values,
        _buffer(gradient),
        grad_values,
        grad_up if grad_up is not None else grad_values,
        count,
        KIND=context.kind,
        GATED=context.up is not None,
        BLOCK=128,
        STRICT_MATH=True,
    )
    return grad_values if grad_up is None else (grad_values, grad_up)


@dataclass(frozen=True)
class RotaryContext:
    values: MtileBuffer
    cosine: MtileBuffer
    sine: MtileBuffer
    rotary_dim: int
    interleaved: bool
    block: int

    @property
    def options(self):
        return dict(
            DIM=self.values.shape[-1],
            ROTARY_DIM=self.rotary_dim,
            INTERLEAVED=self.interleaved,
            BLOCK=self.block,
            STRICT_MATH=True,
        )


def rope_forward(values, cosine, sine, *, rotary_dim=None, interleaved=False):
    """Return (output, context) for 2-D inputs and pre-expanded rotary tables.

    Values are [rows, dimension]; tables are [rows, rotary_dim/2]. Q and K
    with different head counts are separate calls. Position selection, YaRN,
    MRoPE table construction and summing broadcast table gradients are outside
    this operator, so their semantics cannot be accidentally conflated.
    """
    from metile_kernels.rope import _contract
    from metile_kernels.rope import rope_forward as kernel

    _array(values, "values")
    if len(values.shape) != 2:
        raise ValueError("RoPE requires values with shape [rows, dimension]")
    rows, dimension = values.shape
    rotary_dim = dimension if rotary_dim is None else rotary_dim
    block = max(32, 1 << (dimension - 1).bit_length())
    _contract(dimension, rotary_dim, interleaved, block)
    _array(cosine, "cosine", (rows, rotary_dim // 2))
    _array(sine, "sine", cosine.shape)
    if cosine.dtype != sine.dtype:
        raise TypeError("cosine and sine must use the same dtype")
    values, cosine, sine = map(_buffer, (values, cosine, sine))
    context = RotaryContext(values, cosine, sine, rotary_dim, interleaved, block)
    output = MtileBuffer.empty(values.shape, values.dtype)
    kernel[(rows,)](values, cosine, sine, output, rows, **context.options)
    return output, context


def rope_backward(context, gradient):
    """Return FP32 ``(dvalues, dcosine, dsine)`` for the expanded tables."""
    from metile_kernels.rope import rope_backward as kernel

    if not isinstance(context, RotaryContext):
        raise TypeError("RoPE backward requires a RotaryContext")
    _array(gradient, "gradient", context.values.shape)
    gradients = tuple(
        MtileBuffer.empty(value.shape, np.float32)
        for value in (context.values, context.cosine, context.sine)
    )
    rows = context.values.shape[0]
    kernel[(rows,)](
        context.values,
        context.cosine,
        context.sine,
        _buffer(gradient),
        *gradients,
        rows,
        **context.options,
    )
    return gradients
