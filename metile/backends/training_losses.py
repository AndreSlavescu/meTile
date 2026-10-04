"""Bounded native FP32 row normalization and cross entropy with explicit VJPs.

Normalize the final axis of contiguous arrays: 1..262144 columns, at most
65536 rows, and at most 256 MiB of newly allocated output/workspace per call.
Finite logits are a precondition; native calls do not scan floating inputs.
Cross-entropy targets are integer class indices, not probability distributions.
Targets are validated on the CPU (Buffer targets therefore synchronize).

Uniform smoothing uses (1-epsilon)*one_hot + epsilon/classes. The optional
nonnegative z_loss coefficient adds z_loss*logsumexp(logits)**2 to each valid
row, inside the chosen reduction. Ignored rows contribute neither loss nor
gradient; an all-ignored mean is exactly zero. Saved buffers must remain
unchanged until backward. These functions do not register framework autograd.
"""

from dataclasses import dataclass
from math import isfinite, prod

import numpy as np

from metile.runtime.buffer import MtileBuffer

_MAX_BYTES = 256 * 1024**2


def _array(value, name, shape=None, *, reference=False):
    if not isinstance(value, (np.ndarray, MtileBuffer)):
        raise TypeError(f"{name} must be a NumPy array or meTile Buffer")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    allowed = (np.dtype("float32"), np.dtype("float64")) if reference else (np.dtype("float32"),)
    if value.dtype not in allowed:
        raise TypeError(f"{name} must use FP32 (FP64 is reference-only)")
    if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return tuple(value.shape)


def _matrix(source, *, reference=False):
    shape = _array(source, "source", reference=reference)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("row normalization requires positive dimensions and a final axis")
    rows, columns = prod(shape[:-1]), shape[-1]
    if rows > 65536 or columns > 262144:
        raise ValueError("row normalization supports at most 65536 rows and 262144 columns")
    _budget(rows * columns)
    return shape, rows, columns


def _budget(elements):
    if 4 * elements > _MAX_BYTES:
        raise ValueError("training loss outputs/workspace exceed the 256 MiB bound")


def _buffer(value):
    return value if isinstance(value, MtileBuffer) else MtileBuffer(data=value)


def _normalization_forward(source, logarithmic):
    from metile_kernels.training_losses import row_normalization_forward_kernel

    shape, rows, columns = _matrix(source)
    output = MtileBuffer.empty(shape)
    row_normalization_forward_kernel[(rows,)](
        _buffer(source),
        output,
        rows,
        columns,
        LOGARITHMIC=logarithmic,
        BLOCK=256,
        STRICT_MATH=True,
    )
    return output


def softmax_forward(source):
    """Return FP32 final-axis softmax; keep the output for the explicit VJP."""
    return _normalization_forward(source, False)


def log_softmax_forward(source):
    """Return stable FP32 final-axis log-softmax, including very negative logits."""
    return _normalization_forward(source, True)


def _normalization_backward(output, output_gradient, logarithmic):
    from metile_kernels.training_losses import row_normalization_backward_kernel

    shape, rows, columns = _matrix(output)
    _array(output_gradient, "output_gradient", shape)
    gradient = MtileBuffer.empty(shape)
    row_normalization_backward_kernel[(rows,)](
        _buffer(output),
        _buffer(output_gradient),
        gradient,
        rows,
        columns,
        LOGARITHMIC=logarithmic,
        BLOCK=256,
        STRICT_MATH=True,
    )
    return gradient


def softmax_backward(output, output_gradient):
    """Return output * (cotangent - sum(output*cotangent))."""
    return _normalization_backward(output, output_gradient, False)


def log_softmax_backward(output, output_gradient):
    """Return cotangent - exp(output)*sum(cotangent)."""
    return _normalization_backward(output, output_gradient, True)


def _attributes(reduction, ignore_index, label_smoothing, z_loss):
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("ignore_index must be an integer")
    if not -(2**31) <= ignore_index < 2**31:
        raise ValueError("ignore_index must fit int32")
    for name, value in (("label_smoothing", label_smoothing), ("z_loss", z_loss)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real scalar")
        if not isfinite(value) or value < 0 or value > float(np.finfo(np.float32).max):
            raise ValueError(f"{name} must be nonnegative finite FP32")
    if label_smoothing > 1:
        raise ValueError("label_smoothing must be in [0, 1]")


def _targets(target, shape, columns, ignore_index):
    if not isinstance(target, (np.ndarray, MtileBuffer)) or tuple(target.shape) != shape:
        raise ValueError(f"target must have shape {shape}")
    if np.dtype(target.dtype) not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("target must use int32 or int64 class indices")
    values = target.numpy() if isinstance(target, MtileBuffer) else target
    valid = values != ignore_index
    if np.any(valid & ((values < 0) | (values >= columns))):
        raise ValueError("target class is outside [0, classes) and is not ignore_index")
    return np.ascontiguousarray(values, dtype=np.int32).reshape(shape), int(np.count_nonzero(valid))


@dataclass(frozen=True)
class CrossEntropyContext:
    source: object
    target: object
    maxima: object
    log_totals: object
    log_normalizer: object
    valid_count: int
    reduction: str
    ignore_index: int
    label_smoothing: float
    z_loss: float

    @property
    def normalizer(self):
        return max(1, self.valid_count) if self.reduction == "mean" else 1


def cross_entropy_forward(
    source,
    target,
    *,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
    z_loss=0.0,
):
    """Return (loss, context); loss is scalar unless reduction='none'."""
    from metile_kernels.training_losses import (
        cross_entropy_forward_kernel,
        cross_entropy_reduce_kernel,
    )

    shape, rows, columns = _matrix(source)
    _attributes(reduction, ignore_index, label_smoothing, z_loss)
    target_array, valid_count = _targets(target, shape[:-1], columns, ignore_index)
    _budget(4 * rows + (reduction != "none"))
    source_buffer = _buffer(source)
    target_buffer = _buffer(target_array)
    row_loss, maxima, log_totals, log_normalizer = [MtileBuffer.empty(shape[:-1]) for _ in range(4)]
    smoothing = float(np.float32(label_smoothing))
    coefficient = float(np.float32(z_loss))
    cross_entropy_forward_kernel[(rows,)](
        source_buffer,
        target_buffer,
        row_loss,
        log_normalizer,
        maxima,
        log_totals,
        rows,
        columns,
        int(ignore_index),
        smoothing,
        coefficient,
        SMOOTH=smoothing != 0.0,
        BLOCK=256,
        STRICT_MATH=True,
    )
    context = CrossEntropyContext(
        source_buffer,
        target_buffer,
        maxima,
        log_totals,
        log_normalizer,
        valid_count,
        reduction,
        int(ignore_index),
        smoothing,
        coefficient,
    )
    loss = row_loss
    if reduction != "none":
        loss = MtileBuffer.empty(())
        cross_entropy_reduce_kernel[(1,)](
            row_loss,
            loss,
            rows,
            float(context.normalizer),
            BLOCK=256,
            STRICT_MATH=True,
        )
    return loss, context


def cross_entropy_backward(context, cotangent=None):
    """Return dlogits; labels, smoothing, z-loss coefficient are nondifferentiated."""
    from metile_kernels.training_losses import cross_entropy_backward_kernel

    if not isinstance(context, CrossEntropyContext):
        raise TypeError("context must come from cross_entropy_forward")
    shape, rows, columns = _matrix(context.source)
    seed_shape = shape[:-1] if context.reduction == "none" else ()
    if cotangent is None:
        cotangent = np.ones(seed_shape, dtype=np.float32)
    _array(cotangent, "cotangent", seed_shape)
    _array(context.maxima, "maxima", shape[:-1])
    _array(context.log_totals, "log_totals", shape[:-1])
    gradient = MtileBuffer.empty(shape)
    cross_entropy_backward_kernel[(rows,)](
        context.source,
        context.target,
        context.maxima,
        context.log_totals,
        _buffer(cotangent),
        gradient,
        rows,
        columns,
        context.ignore_index,
        context.label_smoothing,
        context.z_loss,
        float(context.normalizer),
        REDUCED=context.reduction != "none",
        BLOCK=256,
        STRICT_MATH=True,
    )
    return gradient


def cross_entropy_reference(
    source,
    target,
    *,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
    z_loss=0.0,
):
    """FP64 NumPy oracle with the same loss semantics, without Metal initialization."""
    if not isinstance(source, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("the CPU reference accepts only NumPy arrays")
    shape, _rows, columns = _matrix(source, reference=True)
    if not np.all(np.isfinite(source)):
        raise ValueError("the CPU reference requires finite logits")
    _attributes(reduction, ignore_index, label_smoothing, z_loss)
    target_array, valid_count = _targets(target, shape[:-1], columns, ignore_index)
    logits = source.astype(np.float64)
    maxima = logits.max(axis=-1)
    centered = logits - maxima[..., None]
    log_totals = np.log(np.exp(centered).sum(axis=-1))
    valid = target_array != ignore_index
    safe_target = np.where(valid, target_array, 0)
    selected = np.take_along_axis(centered, safe_target[..., None], axis=-1)[..., 0]
    losses = (1 - label_smoothing) * (log_totals - selected)
    losses += label_smoothing * (log_totals - centered.mean(axis=-1))
    log_normalizer = maxima + log_totals
    losses = np.where(valid, losses + z_loss * log_normalizer**2, 0)
    context = CrossEntropyContext(
        logits,
        target_array,
        maxima,
        log_totals,
        log_normalizer,
        valid_count,
        reduction,
        int(ignore_index),
        float(label_smoothing),
        float(z_loss),
    )
    loss = losses if reduction == "none" else np.asarray(losses.sum() / context.normalizer)
    return loss, context


def cross_entropy_reference_backward(context, cotangent=None):
    """FP64 oracle VJP; accepts only a CPU-reference context."""
    if not isinstance(context, CrossEntropyContext) or not isinstance(context.source, np.ndarray):
        raise TypeError("context must come from cross_entropy_reference")
    logits = context.source
    seed_shape = logits.shape[:-1] if context.reduction == "none" else ()
    seed = np.ones(seed_shape) if cotangent is None else np.asarray(cotangent, dtype=np.float64)
    if seed.shape != seed_shape:
        raise ValueError(f"cotangent must have shape {seed_shape}")
    probability = np.exp((logits - context.maxima[..., None]) - context.log_totals[..., None])
    gradient = (1 + 2 * context.z_loss * context.log_normalizer[..., None]) * probability
    gradient -= context.label_smoothing / logits.shape[-1]
    valid = context.target != context.ignore_index
    target = np.where(valid, context.target, 0)
    flat_gradient = gradient.reshape(-1, logits.shape[-1])
    flat_gradient[np.arange(target.size), target.ravel()] -= 1 - context.label_smoothing
    gradient *= np.asarray(seed / context.normalizer)[..., None]
    return np.where(valid[..., None], gradient, 0.0)
