import sys

import numpy as np
import pytest

from metile.backends.training_losses import (
    cross_entropy_backward,
    cross_entropy_forward,
    cross_entropy_reference,
    cross_entropy_reference_backward,
    log_softmax_backward,
    log_softmax_forward,
    softmax_backward,
    softmax_forward,
)

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


@pytest.mark.parametrize("columns", [1, 7, 257, 8193, 262144])
@pytest.mark.parametrize("logarithmic", [False, True])
def test_normalization_forward_and_vjp_match_fp64(columns, logarithmic):
    random = np.random.default_rng(308)
    values = random.normal(0, 3, (2, columns)).astype(np.float32)
    seed = random.normal(size=values.shape).astype(np.float32)
    centered = values.astype(np.float64) - values.max(axis=-1, keepdims=True)
    expected_log = centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))
    probability = np.exp(expected_log)
    if logarithmic:
        output = log_softmax_forward(values)
        gradient = log_softmax_backward(output, seed)
        expected = expected_log
        expected_gradient = seed - probability * seed.astype(np.float64).sum(axis=-1, keepdims=True)
    else:
        output = softmax_forward(values)
        gradient = softmax_backward(output, seed)
        expected = probability
        expected_gradient = probability * (seed - (seed * probability).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(output.numpy(), expected, rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=3e-5, atol=5e-6)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("smoothing,z_loss", [(0.0, 0.0), (0.2, 0.01), (1.0, 0.0)])
@pytest.mark.parametrize("columns", [1, 7, 513, 8193])
def test_cross_entropy_forward_backward_match_fp64(reduction, smoothing, z_loss, columns):
    random = np.random.default_rng(309)
    values = random.normal(0, 2, (2, 3, columns)).astype(np.float32)
    targets = random.integers(0, columns, size=(2, 3), dtype=np.int64)
    targets[0, 1] = -100
    options = dict(reduction=reduction, label_smoothing=smoothing, z_loss=z_loss)
    expected, reference_context = cross_entropy_reference(values, targets, **options)
    cotangent = np.asarray(random.normal(size=expected.shape), dtype=np.float32)
    loss, context = cross_entropy_forward(values, targets, **options)
    gradient = cross_entropy_backward(context, cotangent)
    expected_gradient = cross_entropy_reference_backward(reference_context, cotangent)
    np.testing.assert_allclose(loss.numpy(), expected, rtol=4e-6, atol=4e-6)
    np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=5e-5, atol=4e-6)
    np.testing.assert_array_equal(gradient.numpy()[targets == -100], 0.0)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_native_all_ignored_mean_and_gradient_are_zero(reduction):
    values = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]], dtype=np.float32)
    loss, context = cross_entropy_forward(
        values,
        np.array([-100, -100], dtype=np.int32),
        reduction=reduction,
        label_smoothing=0.7,
        z_loss=0.03,
    )
    np.testing.assert_array_equal(loss.numpy(), 0.0)
    np.testing.assert_array_equal(cross_entropy_backward(context).numpy(), 0.0)


def test_native_large_shift_and_extreme_log_softmax():
    values = np.full((2, 257), 1e8, dtype=np.float32)
    loss, context = cross_entropy_forward(
        values, np.array([0, 256], dtype=np.int32), label_smoothing=0.5
    )
    np.testing.assert_allclose(loss.numpy(), np.log(257), rtol=2e-6)
    np.testing.assert_allclose(softmax_forward(values).numpy(), 1 / 257, rtol=3e-6)
    np.testing.assert_allclose(cross_entropy_backward(context).numpy().sum(axis=-1), 0.0, atol=2e-6)
    extreme = np.array([[1000.0, -1000.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(log_softmax_forward(extreme).numpy(), [[0.0, -2000.0, -1000.0]])


def test_single_row_scalar_target_and_loss():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target = np.array(1, dtype=np.int32)
    expected, reference_context = cross_entropy_reference(values, target)
    actual, context = cross_entropy_forward(values, target)
    assert actual.shape == ()
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6)
    np.testing.assert_allclose(
        cross_entropy_backward(context).numpy(),
        cross_entropy_reference_backward(reference_context),
        rtol=3e-6,
    )
