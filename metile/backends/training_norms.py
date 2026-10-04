"""Bounded native normalization forward/backward APIs, independent of MLX.

Rows are independent; input widths are 1..8192. Storage is FP16/FP32,
statistics and all returned gradients are FP32. Backward uses per-row
parameter partials followed by a fixed-order reduction, never atomics.
Saved input buffers must remain unchanged until backward completes.
"""

from dataclasses import dataclass
from math import isfinite, prod

import numpy as np

import metile
from metile.runtime.buffer import MtileBuffer
from metile_kernels.training_norms import (
    norm_backward_rows_kernel,
    norm_forward_kernel,
    norm_parameter_reduce_kernel,
)


def _metadata(value, name, shape=None):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    if np.dtype(value.dtype) not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(f"{name} must use float16 or float32 storage")
    actual = tuple(value.shape)
    if shape is not None and actual != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return actual


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer(data=value)


def _options(columns, kind):
    block = max(64, metile.next_power_of_2(columns))
    return {
        "KIND": kind,
        "BLOCK": block,
        "LAYOUT": metile.ThreadLayout.identity(block, elements_per_thread=max(2, block // 256)),
        "STRICT_MATH": True,
    }


@dataclass(frozen=True)
class NormContext:
    kind: str
    source: MtileBuffer
    weight: MtileBuffer
    residual: MtileBuffer | None
    mean: MtileBuffer
    inverse_scale: MtileBuffer


@dataclass(frozen=True)
class NormGradients:
    source: MtileBuffer
    weight: MtileBuffer
    bias: MtileBuffer | None = None
    residual: MtileBuffer | None = None


def _forward(source, weight, bias, residual, epsilon, kind):
    shape = _metadata(source, "source")
    if len(shape) != 2 or shape[0] <= 0 or not 1 <= shape[1] <= 8192:
        raise ValueError("normalization requires [rows, columns] with rows > 0 and columns 1..8192")
    if prod(shape) * 8 > 128 * 1024**2:
        raise ValueError("normalization parameter partials exceed the 128 MiB scratch bound")
    rows, columns = shape
    _metadata(weight, "weight", (columns,))
    if bias is not None:
        _metadata(bias, "bias", (columns,))
    if residual is not None:
        _metadata(residual, "residual", shape)
        if residual.dtype != source.dtype:
            raise TypeError("residual and source must have matching storage dtypes")
    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isfinite(float(epsilon))
        or not (
            np.finfo(np.float32).smallest_subnormal <= float(epsilon) <= np.finfo(np.float32).max
        )
    ):
        raise ValueError("epsilon must be positive finite FP32")
    source_buffer, weight_buffer = _buffer(source), _buffer(weight)
    residual_buffer = _buffer(residual) if residual is not None else None
    bias_buffer = _buffer(bias) if bias is not None else weight_buffer
    output = MtileBuffer.empty(shape, dtype=source.dtype)
    residual_output = MtileBuffer.empty(shape, dtype=source.dtype) if residual is not None else None
    mean = MtileBuffer.empty((rows, 1))
    inverse = MtileBuffer.empty((rows, 1))
    norm_forward_kernel[(rows,)].prepare(
        source_buffer,
        residual_buffer or source_buffer,
        weight_buffer,
        bias_buffer,
        output,
        residual_output or output,
        mean,
        inverse,
        rows,
        columns,
        float(np.float32(epsilon)),
        **_options(columns, kind),
    )
    context = NormContext(kind, source_buffer, weight_buffer, residual_buffer, mean, inverse)
    return output, residual_output, context


def rms_norm_forward(source, weight, *, epsilon=1e-5):
    """Return (output, context) for x * rsqrt(mean(x*x) + epsilon) * weight."""
    output, _, context = _forward(source, weight, None, None, epsilon, "rms")
    return output, context


def layer_norm_forward(source, weight, bias, *, epsilon=1e-5):
    """Return (output, context), using FP32 centered population variance."""
    output, _, context = _forward(source, weight, bias, None, epsilon, "layer")
    return output, context


def add_rms_norm_forward(source, residual, weight, *, epsilon=1e-5):
    """Return (output, residual_sum, context).

    Normalize the unrounded FP32 sum of source and residual. Both returned
    outputs are cast to source storage dtype; backward ignores cast rounding.
    """
    return _forward(source, weight, None, residual, epsilon, "add_rms")


def norm_backward(context, output_gradient, *, residual_output_gradient=None):
    """Return FP32 dsource/dweight and optional dbias/dresidual.

    For add-RMSNorm, the optional second-output cotangent defaults to zero.
    Gradients for shared parameters are reduced across all rows in fixed order.
    This is an explicit native API, not a framework autograd registration.
    """
    if not isinstance(context, NormContext) or context.kind not in ("rms", "layer", "add_rms"):
        raise TypeError("norm_backward requires a NormContext from a forward call")
    shape = tuple(context.source.shape)
    rows, columns = shape
    _metadata(output_gradient, "output_gradient", shape)
    if residual_output_gradient is not None:
        if context.kind != "add_rms":
            raise ValueError("residual_output_gradient requires add-RMSNorm")
        _metadata(residual_output_gradient, "residual_output_gradient", shape)
    output_seed = _buffer(output_gradient)
    residual_seed = (
        _buffer(residual_output_gradient)
        if residual_output_gradient is not None
        else MtileBuffer.zeros(shape)
    )
    source_gradient = MtileBuffer.empty(shape)
    residual_gradient = MtileBuffer.empty(shape) if context.kind == "add_rms" else None
    weight_partials = MtileBuffer.empty(shape)
    bias_partials = MtileBuffer.empty(shape) if context.kind == "layer" else weight_partials
    weight_gradient = MtileBuffer.empty((columns,))
    bias_gradient = MtileBuffer.empty((columns,)) if context.kind == "layer" else None
    norm_backward_rows_kernel[(rows,)].prepare(
        context.source,
        context.residual or context.source,
        context.weight,
        context.mean,
        context.inverse_scale,
        output_seed,
        residual_seed,
        source_gradient,
        residual_gradient or source_gradient,
        weight_partials,
        bias_partials,
        rows,
        columns,
        **_options(columns, context.kind),
    )
    norm_parameter_reduce_kernel[(metile.cdiv(columns, 128),)].prepare(
        weight_partials,
        bias_partials,
        weight_gradient,
        bias_gradient or weight_gradient,
        rows,
        columns,
        HAS_BIAS=context.kind == "layer",
        BLOCK=128,
        STRICT_MATH=True,
    )
    return NormGradients(source_gradient, weight_gradient, bias_gradient, residual_gradient)
