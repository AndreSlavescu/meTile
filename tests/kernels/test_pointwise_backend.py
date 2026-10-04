import numpy as np

from metile.backends.pointwise import (
    activation_backward,
    activation_forward,
    rope_backward,
    rope_forward,
)


def test_gpu_gated_native_backend_preserves_inputs_and_fp32_gradients():
    values = np.linspace(-2, 2, 35, dtype=np.float16).reshape(5, 7)
    multiplier = np.full_like(values, 0.5)
    saved_values = values.copy()
    output, context = activation_forward(values, kind="sigmoid", up=multiplier)
    first, second = activation_backward(context, np.ones_like(values))
    sigmoid = 1 / (1 + np.exp(-values.astype(np.float64)))
    np.testing.assert_allclose(output.numpy(), sigmoid * 0.5, atol=2e-4)
    np.testing.assert_allclose(first.numpy(), 0.5 * sigmoid * (1 - sigmoid), atol=1e-7)
    np.testing.assert_allclose(second.numpy(), sigmoid, atol=1e-7)
    np.testing.assert_array_equal(values, saved_values)


def test_gpu_rotary_native_backend_accepts_fp32_tables_with_fp16_values():
    values = np.linspace(-1, 1, 70, dtype=np.float16).reshape(2, 35)
    cosine = np.full((2, 16), 0.8, dtype=np.float32)
    sine = np.full_like(cosine, 0.6)
    output, context = rope_forward(values, cosine, sine, rotary_dim=32, interleaved=True)
    grad_values, grad_cosine, grad_sine = rope_backward(context, np.ones_like(values))
    expected = np.ones(values.shape, dtype=np.float32)
    expected[:, :32:2] = 1.4
    expected[:, 1:32:2] = 0.2
    np.testing.assert_allclose(grad_values.numpy(), expected, atol=1e-7)
    np.testing.assert_array_equal(output.numpy()[:, 32:], values[:, 32:])
    np.testing.assert_allclose(
        grad_cosine.numpy(),
        values[:, :32:2].astype(np.float32) + values[:, 1:32:2].astype(np.float32),
    )
    np.testing.assert_allclose(
        grad_sine.numpy(),
        values[:, :32:2].astype(np.float32) - values[:, 1:32:2].astype(np.float32),
    )
