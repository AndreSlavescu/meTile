import numpy as np
import pytest

from metile.backends.training_matmul import matmul_backward, matmul_forward
from tests.kernels.test_training_activations import _reference


@pytest.mark.parametrize("shape", [(3, 5, 7), (33, 35, 37)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("activation", [None, "relu", "silu", "quick_gelu", "gelu_tanh"])
def test_gpu_matmul_backward_and_activation_epilogues(shape, dtype, activation):
    rows, reduction, columns = shape
    generator = np.random.default_rng(310)
    left = generator.normal(scale=0.4, size=(rows, reduction)).astype(dtype)
    right = generator.normal(scale=0.4, size=(reduction, columns)).astype(dtype)
    seed = generator.normal(size=(rows, columns)).astype(np.float32)
    output, context = matmul_forward(left, right, activation=activation)
    grad_left, grad_right = matmul_backward(context, seed)
    expected = left.astype(np.float64) @ right.astype(np.float64)
    derivative = np.ones_like(expected)
    if activation is not None:
        step = 1e-5
        derivative = (
            _reference(expected + step, activation) - _reference(expected - step, activation)
        ) / (2 * step)
        expected = _reference(expected, activation)
    expected_seed = seed * derivative
    np.testing.assert_allclose(output.numpy(), expected, rtol=5e-5, atol=3e-6)
    np.testing.assert_allclose(
        grad_left.numpy(), expected_seed @ right.astype(np.float64).T, rtol=8e-5, atol=6e-6
    )
    np.testing.assert_allclose(
        grad_right.numpy(), left.astype(np.float64).T @ expected_seed, rtol=8e-5, atol=6e-6
    )
