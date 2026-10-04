import inspect

import numpy as np
import pytest

from metile.backends.training_losses import (
    cross_entropy_forward,
    cross_entropy_reference,
    cross_entropy_reference_backward,
    softmax_backward,
    softmax_forward,
)
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType
from metile_kernels.training_losses import (
    cross_entropy_backward_kernel,
    cross_entropy_forward_kernel,
    cross_entropy_reduce_kernel,
    row_normalization_backward_kernel,
    row_normalization_forward_kernel,
)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("z_loss", [0.0, 0.03])
def test_cross_entropy_vjp_finite_differences(reduction, smoothing, z_loss):
    random = np.random.default_rng(306)
    logits = random.normal(size=(2, 3, 5))
    targets = np.array([[0, 4, -100], [2, -100, 1]], dtype=np.int64)
    options = dict(reduction=reduction, label_smoothing=smoothing, z_loss=z_loss)
    loss, context = cross_entropy_reference(logits, targets, **options)
    cotangent = np.asarray(random.normal(size=loss.shape))
    actual = cross_entropy_reference_backward(context, cotangent)
    epsilon = 1e-6
    expected = np.empty_like(logits)
    for index in np.ndindex(logits.shape):
        original = logits[index]
        logits[index] = original + epsilon
        plus = np.sum(cross_entropy_reference(logits, targets, **options)[0] * cotangent)
        logits[index] = original - epsilon
        minus = np.sum(cross_entropy_reference(logits, targets, **options)[0] * cotangent)
        logits[index] = original
        expected[index] = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=4e-9)
    np.testing.assert_array_equal(actual[targets == -100], 0.0)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_all_ignored_loss_and_gradient_are_zero(reduction):
    logits = np.array([[1000.0, -1000.0], [-2.0, 3.0]])
    targets = np.array([-100, -100], dtype=np.int32)
    loss, context = cross_entropy_reference(
        logits, targets, reduction=reduction, label_smoothing=0.3, z_loss=0.5
    )
    np.testing.assert_array_equal(loss, 0.0)
    np.testing.assert_array_equal(cross_entropy_reference_backward(context), 0.0)
    assert context.valid_count == 0


def test_numpy_scalar_loss_configuration():
    source = np.array([[0.1, 0.3]], dtype=np.float64)
    target = np.array([1], dtype=np.int32)
    loss, _context = cross_entropy_reference(
        source, target, label_smoothing=np.float32(0.2), z_loss=np.float32(0.01)
    )
    assert np.isfinite(loss)


def test_mean_counts_valid_rows_and_large_common_shift_is_stable():
    logits = np.full((3, 7), 1e8, dtype=np.float64)
    targets = np.array([0, -100, 6], dtype=np.int32)
    loss, context = cross_entropy_reference(logits, targets, label_smoothing=0.7)
    np.testing.assert_allclose(loss, np.log(7), rtol=1e-14)
    assert context.valid_count == 2
    np.testing.assert_allclose(
        cross_entropy_reference_backward(context).sum(axis=-1), 0, atol=1e-16
    )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"reduction": "batchmean"}, ValueError),
        ({"label_smoothing": -0.1}, ValueError),
        ({"label_smoothing": 1.1}, ValueError),
        ({"label_smoothing": float("nan")}, ValueError),
        ({"z_loss": -0.1}, ValueError),
        ({"z_loss": float("inf")}, ValueError),
        ({"ignore_index": 1.5}, TypeError),
        ({"ignore_index": 2**40}, ValueError),
    ],
)
def test_invalid_loss_attributes_fail_before_gpu(options, error):
    with pytest.raises(error):
        cross_entropy_forward(
            np.zeros((2, 3), dtype=np.float32), np.array([0, 1], dtype=np.int32), **options
        )


def test_invalid_shapes_dtypes_targets_and_strides_fail_before_gpu():
    logits = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        cross_entropy_forward(logits, np.array([0, 3], dtype=np.int32))
    with pytest.raises(TypeError, match="class indices"):
        cross_entropy_forward(logits, np.array([0.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="target must have shape"):
        cross_entropy_forward(logits, np.zeros((2, 1), dtype=np.int32))
    with pytest.raises(TypeError, match="FP32"):
        softmax_forward(logits.astype(np.float16))
    with pytest.raises(ValueError, match="contiguous"):
        softmax_forward(logits[:, ::-1])
    with pytest.raises(ValueError, match="positive"):
        softmax_forward(np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="262144"):
        softmax_forward(np.zeros((1, 262145), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        softmax_backward(logits, np.zeros((2, 1), dtype=np.float32))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "kernel",
    [
        row_normalization_forward_kernel,
        row_normalization_backward_kernel,
        cross_entropy_forward_kernel,
        cross_entropy_reduce_kernel,
        cross_entropy_backward_kernel,
    ],
)
def test_loss_kernels_trace_with_predicated_tensor_accesses(kernel, enabled):
    constants = dict(LOGARITHMIC=enabled, SMOOTH=enabled, REDUCED=enabled, BLOCK=256)
    arguments, keywords = [], {}
    with TracingContext(kernel.name) as context:
        for name in inspect.signature(kernel.fn).parameters:
            if name in constants:
                keywords[name] = constants[name]
            else:
                if name in ("rows", "columns", "ignore_index"):
                    dtype = ScalarType("i32")
                elif name in ("smoothing", "z_loss", "normalizer"):
                    dtype = ScalarType("f32")
                else:
                    dtype = PtrType("i32" if name == "target" else "f32")
                context.func.params.append(tir.Param(name, dtype))
                arguments.append(TracingProxy(tir.Value(name, dtype)))
        context.func.constexprs = {**keywords, "STRICT_MATH": True}
        kernel.fn(*arguments, **keywords)
    pending = list(context.func.ops)
    while pending:
        operation = pending.pop()
        if isinstance(operation, (tir.Load, tir.Store)):
            assert operation.tensor is not None
        pending.extend(getattr(operation, "body", ()))
    source = emit(lower(context.func))
    assert "[[kernel]]" in source
    assert "atomic" not in source
    assert "for (" in source
