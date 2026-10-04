import numpy as np
import pytest

from metile.backends.attention import attention_backward, attention_forward
from tests.kernels.test_stable_attention import _reference


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_gpu_native_attention_context_and_gradients(dtype, causal):
    generator = np.random.default_rng(301)
    query = generator.normal(size=(1, 4, 3, 32)).astype(dtype)
    key = generator.normal(size=(1, 2, 5, 32)).astype(dtype)
    value = generator.normal(size=key.shape).astype(dtype)
    gradient = generator.normal(size=query.shape).astype(np.float32)
    mask = np.ones((3, 5), dtype=bool)
    mask[0] = False
    output, context = attention_forward(query, key, value, causal=causal, mask=mask)
    expected = _reference(query, key, value, gradient, 32**-0.5, mask, causal, 2 if causal else 0)
    actual_gradients = attention_backward(context, gradient)
    np.testing.assert_allclose(
        output.numpy(),
        expected["output"],
        rtol=2e-3 if dtype == np.float16 else 3e-5,
        atol=5e-4 if dtype == np.float16 else 2e-6,
    )
    for actual, name in zip(actual_gradients, ("grad_query", "grad_key", "grad_value")):
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual.numpy(), expected[name], rtol=3e-4, atol=3e-6)
    assert context.causal_offset == (2 if causal else 0)
    assert context.kernel_options["STRICT_MATH"] is True
