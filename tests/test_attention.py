import threading

import numpy as np
import pytest

import metile
from kernels.attention import attention_decode_kernel
from metile.runtime import attention as attention_runtime
from metile.runtime.attention import AttentionDecodeConfig, _prepare_two_pass, attention_decode
from metile.runtime.metal_device import MetalDevice


def _reference_attention(query, key, value, scale):
    scores = np.einsum("hd,hnd->hn", query, key) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hn,hnd->hd", weights, value)


def _reference_grouped_attention(query, key, value, scale):
    query_heads = query.shape[1]
    key_value_heads = key.shape[1]
    key_value_indices = np.arange(query_heads) // (query_heads // key_value_heads)
    expanded_key = key[:, key_value_indices]
    expanded_value = value[:, key_value_indices]
    scores = np.einsum("bhd,bhnd->bhn", query, expanded_key) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhn,bhnd->bhd", weights, expanded_value)


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
        Q_HEADS=heads,
        KV_HEADS=heads,
        BLOCK=block,
    )
    MetalDevice.get().sync()

    expected = _reference_attention(query, key, value, scale)
    actual = output.numpy().reshape(heads, dimension)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def _run_two_pass_attention(heads, tokens, dimension, config):
    random = np.random.default_rng(heads * 1000 + tokens + dimension)
    query = random.standard_normal((heads, dimension), dtype=np.float32)
    key = random.standard_normal((heads, tokens, dimension), dtype=np.float32)
    value = random.standard_normal((heads, tokens, dimension), dtype=np.float32)
    output = metile.Buffer.zeros((heads * dimension,))
    scale = float(dimension**-0.5)

    dispatch = _prepare_two_pass(
        metile.Buffer(data=query.ravel()),
        metile.Buffer(data=key.ravel()),
        metile.Buffer(data=value.ravel()),
        output,
        1,
        heads,
        heads,
        tokens,
        scale,
        dimension,
        config,
    )
    dispatch()
    MetalDevice.get().sync()

    expected = _reference_attention(query, key, value, scale)
    actual = output.numpy().reshape(heads, dimension)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("block", [32, 64, 128, 256, 512, 1024])
def test_decode_attention_supports_runtime_schedule_shapes(block):
    _run_attention(heads=2, tokens=129, dimension=128, block=block)


@pytest.mark.parametrize("tokens", [1, 7, 32, 64, 513])
def test_decode_attention_online_softmax_edges(tokens):
    _run_attention(heads=2, tokens=tokens, dimension=64, block=1024)


@pytest.mark.parametrize("tokens", [129, 513, 8192])
def test_two_pass_decode_attention_merges_unnormalized_partials(tokens):
    _run_two_pass_attention(
        heads=1,
        tokens=tokens,
        dimension=128,
        config=AttentionDecodeConfig(
            "two_pass",
            tokens_per_block=256,
            partial_block=256,
            merge_block=128,
        ),
    )


def test_two_pass_decode_attention_supports_multiple_heads():
    _run_two_pass_attention(
        heads=2,
        tokens=513,
        dimension=64,
        config=AttentionDecodeConfig(
            "two_pass",
            tokens_per_block=256,
            partial_block=128,
            merge_block=128,
        ),
    )


@pytest.mark.parametrize(("batch,query_heads,key_value_heads"), [(2, 4, 2), (2, 4, 1)])
def test_decode_attention_supports_batched_gqa_and_mqa(batch, query_heads, key_value_heads):
    tokens = 129
    dimension = 128
    random = np.random.default_rng(batch * 1000 + query_heads * 100 + key_value_heads)
    query = random.standard_normal((batch, query_heads, dimension), dtype=np.float32)
    key = random.standard_normal((batch, key_value_heads, tokens, dimension), dtype=np.float32)
    value = random.standard_normal(key.shape, dtype=np.float32)
    output = metile.Buffer.zeros((batch * query_heads * dimension,))
    scale = float(dimension**-0.5)

    attention_decode_kernel[(batch * query_heads,)](
        metile.Buffer(data=query.ravel()),
        metile.Buffer(data=key.ravel()),
        metile.Buffer(data=value.ravel()),
        output,
        tokens,
        scale,
        D=dimension,
        Q_HEADS=query_heads,
        KV_HEADS=key_value_heads,
        BLOCK=256,
    )
    MetalDevice.get().sync()

    expected = _reference_grouped_attention(query, key, value, scale)
    actual = output.numpy().reshape(batch, query_heads, dimension)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_two_pass_decode_attention_supports_batched_gqa():
    batch = 2
    query_heads = 4
    key_value_heads = 2
    tokens = 4096
    dimension = 64
    random = np.random.default_rng(2026)
    query = random.standard_normal((batch, query_heads, dimension), dtype=np.float32)
    key = random.standard_normal((batch, key_value_heads, tokens, dimension), dtype=np.float32)
    value = random.standard_normal(key.shape, dtype=np.float32)
    output = metile.Buffer.zeros((batch * query_heads * dimension,))
    scale = float(dimension**-0.5)

    dispatch = _prepare_two_pass(
        metile.Buffer(data=query.ravel()),
        metile.Buffer(data=key.ravel()),
        metile.Buffer(data=value.ravel()),
        output,
        batch,
        query_heads,
        key_value_heads,
        tokens,
        scale,
        dimension,
        AttentionDecodeConfig("two_pass", 512, 256, 128),
    )
    dispatch()
    MetalDevice.get().sync()

    expected = _reference_grouped_attention(query, key, value, scale)
    actual = output.numpy().reshape(batch, query_heads, dimension)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("grid,tokens,dimension,error"),
    [
        ((0,), 32, 64, ValueError),
        ((1,), 0, 64, ValueError),
        ((1,), 32, 48, ValueError),
    ],
)
def test_decode_attention_validates_static_shape(grid, tokens, dimension, error):
    query = metile.Buffer.empty((max(grid[0], 1) * max(dimension, 64),))
    key = metile.Buffer.empty((max(grid[0], 1) * max(tokens, 1) * max(dimension, 64),))
    value = metile.Buffer.empty(key.shape)
    output = metile.Buffer.empty(query.shape)

    with pytest.raises(error):
        attention_decode[grid].prepare(
            query,
            key,
            value,
            output,
            tokens,
            0.125,
            D=dimension,
        )


def test_decode_attention_rejects_non_float_storage():
    query = metile.Buffer.empty((64,), dtype=np.float16)
    key = metile.Buffer.empty((64,), dtype=np.float16)
    value = metile.Buffer.empty((64,), dtype=np.float16)
    output = metile.Buffer.empty((64,), dtype=np.float16)

    with pytest.raises(TypeError, match="float32"):
        attention_decode[(1,)].prepare(query, key, value, output, 1, 0.125, D=64)


def test_decode_attention_validates_grouped_head_count():
    query = metile.Buffer.empty((6 * 64,))
    key = metile.Buffer.empty((4 * 32 * 64,))
    value = metile.Buffer.empty(key.shape)
    output = metile.Buffer.empty(query.shape)

    with pytest.raises(ValueError, match="positive divisor"):
        attention_decode[(1, 6)].prepare(query, key, value, output, 32, 0.125, D=64, KV_HEADS=4)


def test_decode_attention_validates_batched_grouped_buffer_sizes():
    query = metile.Buffer.empty((2 * 4 * 64,))
    key = metile.Buffer.empty((2 * 2 * 32 * 64 - 1,))
    value = metile.Buffer.empty((2 * 2 * 32 * 64,))
    output = metile.Buffer.empty(query.shape)

    with pytest.raises(ValueError, match="key requires"):
        attention_decode[(2, 4)].prepare(query, key, value, output, 32, 0.125, D=64, KV_HEADS=2)


def test_dynamic_two_pass_candidates_fit_one_simdgroup_merge():
    candidates = attention_runtime._two_pass_candidates(65536)
    assert candidates
    assert all(metile.cdiv(65536, candidate.tokens_per_block) <= 32 for candidate in candidates)


def test_two_pass_dispatcher_batches_both_stages_under_one_lock():
    class Device:
        def __init__(self):
            self._dispatch_lock = threading.RLock()

    class Dispatch:
        def __init__(self, device, description_bits):
            self._dev = device
            self._concurrent = True
            self._completion_spin_ns = 0
            self.description_bits = description_bits
            self.calls = 0

        def _encode_unlocked(self, device):
            assert device is self._dev
            self.calls += 1

    device = Device()
    first = Dispatch(device, 10)
    second = Dispatch(device, 20)
    dispatch = attention_runtime._TwoPassDispatcher(first, second)

    dispatch.repeat(3)

    assert (first.calls, second.calls) == (3, 3)
    assert dispatch.description_bits == 30
    assert second._concurrent is False


def test_attention_config_persistence_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    monkeypatch.setattr(
        attention_runtime,
        "_attention_cache_path",
        tmp_path / "attention.json",
    )
    config = AttentionDecodeConfig("two_pass", 512, 256, 128)

    attention_runtime._write_attention_config("shape", config, 0.0002)

    assert attention_runtime._read_attention_config("shape", [config]) == (config, 0.0002)
