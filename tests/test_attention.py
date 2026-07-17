import numpy as np
import pytest

import metile
from kernels.attention import attention_decode_kernel
from metile.runtime.metal_device import MetalDevice


def _reference_attention(query, key, value, scale):
    scores = np.einsum("hd,hnd->hn", query, key) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hn,hnd->hd", weights, value)


def _run_attention(heads, tokens, dimension, block):
    random = np.random.default_rng(heads * 1000 + tokens + dimension + block)
    query = random.standard_normal((heads, dimension), dtype=np.float32)
    key = random.standard_normal((heads, tokens, dimension), dtype=np.float32)
    value = random.standard_normal((heads, tokens, dimension), dtype=np.float32)
    output = metile.Buffer.zeros((heads * dimension,))
    scale = float(dimension**-0.5)

    attention_decode_kernel[(heads,)](
        metile.Buffer(data=query.ravel()),
        metile.Buffer(data=key.ravel()),
        metile.Buffer(data=value.ravel()),
        output,
        tokens,
        scale,
        D=dimension,
        BLOCK=block,
    )
    MetalDevice.get().sync()

    expected = _reference_attention(query, key, value, scale)
    actual = output.numpy().reshape(heads, dimension)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("block", [64, 128, 256, 512, 1024])
def test_decode_attention_supports_runtime_schedule_shapes(block):
    _run_attention(heads=2, tokens=129, dimension=128, block=block)


@pytest.mark.parametrize("tokens", [1, 7, 32, 64, 513])
def test_decode_attention_online_softmax_edges(tokens):
    _run_attention(heads=2, tokens=tokens, dimension=64, block=1024)
