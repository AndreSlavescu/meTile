"""Advancing multi-dispatch execution must preserve logits, cache, and greedy IDs."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.megakernels.qwen3_staged_runtime import Qwen3Staged


@pytest.fixture
def host_packing_candidate(monkeypatch):
    def buffer(values):
        return SimpleNamespace(numpy=lambda: values, nbytes=values.nbytes, dtype=values.dtype)

    monkeypatch.setattr(
        "benchmarks.megakernels.qwen3_staged_runtime.metile.Buffer.empty",
        lambda shape, dtype: buffer(np.empty(shape, dtype=dtype)),
    )
    candidate = object.__new__(Qwen3Staged)
    candidate.dtype = np.float32
    candidate.config = {"hidden_size": 4, "intermediate_size": 8}
    candidate.layer_weights = buffer(np.arange(-8, 8, dtype=np.float32).reshape(2, 8))
    candidate.embedding = buffer(np.arange(-6, 6, dtype=np.float32).reshape(3, 4))
    candidate.decode_layer_weights = candidate.layer_weights
    candidate.decode_embedding = candidate.embedding
    candidate._packed_decode_weights = False
    candidate._prepared = False
    return candidate


def test_lossless_pack_records_exact_storage_and_preserves_original_buffers(host_packing_candidate):
    candidate = host_packing_candidate
    original = candidate.layer_weights, candidate.embedding
    expected = tuple(buffer.numpy().copy() for buffer in original)
    assert candidate.pack_decode_weights()
    assert candidate._packed_decode_weights
    packed = candidate.decode_layer_weights, candidate.decode_embedding
    assert candidate.layer_weights is original[0]
    assert candidate.embedding is original[1]
    for source, destination, values in zip(original, packed, expected, strict=True):
        assert destination is not source
        assert destination.dtype == np.uint32
        assert destination.nbytes * 2 == source.nbytes
        words = destination.numpy()
        decoded = np.empty(values.size, dtype=np.uint32)
        decoded[::2] = words << np.uint32(16)
        decoded[1::2] = words & np.uint32(0xFFFF0000)
        np.testing.assert_array_equal(decoded, values.ravel().view(np.uint32))
        np.testing.assert_array_equal(source.numpy(), values)
    assert candidate.config["decode_weight_packing"] == {
        "enabled": True,
        "format": "fp32_high16x2_u32_lossless",
        "decoded_dtype": "float32",
        "storage_dtype": "uint32",
        "values_per_word": 2,
        "scopes": ["layer_weights", "embedding"],
        "consumers": ["decode_projections", "first_token_vocabulary_projection"],
        "element_count": 28,
        "unpacked_bytes": 112,
        "packed_bytes": 56,
        "verification": {
            "finite": True,
            "even_elements": True,
            "zero_low16_bits": True,
            "roundtrip_bitwise": True,
        },
        "original_buffers_retained": True,
    }
    assert candidate.pack_decode_weights()
    assert candidate.decode_layer_weights is packed[0]
    assert candidate.decode_embedding is packed[1]


@pytest.mark.parametrize("name", ["layer_weights", "embedding"])
def test_lossless_pack_falls_back_without_rounding_random_fp32(host_packing_candidate, name):
    candidate = host_packing_candidate
    values = getattr(candidate, name).numpy()
    values[:] = np.random.default_rng(177).normal(size=values.shape).astype(np.float32)
    before = values.copy()
    assert not candidate.pack_decode_weights()
    assert not candidate._packed_decode_weights
    assert candidate.decode_layer_weights is candidate.layer_weights
    assert candidate.decode_embedding is candidate.embedding
    assert "decode_weight_packing" not in candidate.config
    np.testing.assert_array_equal(values.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("invalid", ["float16", "hidden_size", "intermediate_size"])
def test_lossless_pack_declines_unsupported_storage_or_geometry(host_packing_candidate, invalid):
    candidate = host_packing_candidate
    if invalid == "float16":
        candidate.dtype = np.float16
    else:
        candidate.config[invalid] = 3
    assert not candidate.pack_decode_weights()
    assert candidate.decode_layer_weights is candidate.layer_weights
    assert candidate.decode_embedding is candidate.embedding
    assert "decode_weight_packing" not in candidate.config


@pytest.mark.parametrize("already_packed", [False, True])
def test_lossless_pack_rejects_changes_after_preparation(host_packing_candidate, already_packed):
    candidate = host_packing_candidate
    if already_packed:
        assert candidate.pack_decode_weights()
    original = candidate.decode_layer_weights, candidate.decode_embedding
    candidate._prepared = True
    with pytest.raises(RuntimeError, match="before preparing"):
        candidate.pack_decode_weights()
    assert candidate.decode_layer_weights is original[0]
    assert candidate.decode_embedding is original[1]
    assert candidate._packed_decode_weights is already_packed


@pytest.fixture(params=[("float32", 32), ("float32", 128), ("float16", 128)])
def staged_model(request):
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    dtype, dimension = request.param
    if dtype == "float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("strict FP32 batched prefill requires MLX_ENABLE_TF32=0 before importing MLX")
    core = pytest.importorskip("mlx.core")
    qwen = pytest.importorskip("mlx_lm.models.qwen3")
    core.random.seed(381)
    model = qwen.Model(
        qwen.ModelArgs(
            model_type="qwen3",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=dimension,
            vocab_size=67,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
    )
    for layer in model.layers:
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_norm,
            layer.self_attn.k_norm,
        ):
            norm.weight = core.random.uniform(0.25, 1.75, shape=norm.weight.shape)
    model.set_dtype(getattr(core, dtype))
    core.eval(model.parameters())
    return model, dtype


@pytest.mark.parametrize("threads", [32, 256])
def test_batched_prefill_then_advancing_decode_matches_native(staged_model, threads):
    from benchmarks.megakernels.qwen3_end_to_end import _validate_workload

    model, dtype = staged_model
    candidate = Qwen3Staged(model, capacity=7, threads=threads).prepare()
    assert candidate.cached_tokens == 0
    assert not np.any(candidate.cache.numpy())
    result = _validate_workload(model, candidate, [3, 7, 11], 4, dtype)
    assert result["passed"]
    assert result["actual_output_tokens"] == 4
    assert len(result["checks"]) == 6
    assert candidate.cached_tokens == 6
    assert candidate.dispatches_per_forward() == 19
    assert candidate.dispatches_per_forward(include_greedy=True) == 21
    assert candidate.dispatches_per_forward(project=False) == 17
    assert len(candidate._body) == 17
    assert len(candidate._projection) == len(candidate._selection) == 2
    assert all(not dispatch._concurrent for dispatch in candidate._body)
    assert candidate._body[2]._grid.width > candidate._body[2]._tg.width
    vocabulary = candidate._projection[-1]
    assert vocabulary._grid.width // vocabulary._tg.width == candidate.max_projection_threadgroups


def test_skipped_projection_reset_and_imported_prefix(staged_model):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from metile.runtime.metal_device import MetalDevice

    model, dtype = staged_model
    candidate = Qwen3Staged(model, capacity=5).prepare()
    cache = make_prompt_cache(model)
    output = model(mx.array([[3, 7, 11]]), cache=cache)
    mx.eval(output, [layer.state for layer in cache])
    for position, token in enumerate((3, 7)):
        candidate.forward(token, position, project=False)
        assert MetalDevice.get()._pending_dispatches == 17
        with pytest.raises(RuntimeError, match="projected"):
            candidate.greedy()
    candidate.forward(11, 2)
    assert MetalDevice.get()._pending_dispatches == 19
    tolerance = 1e-4 if dtype == "float32" else 0.02
    np.testing.assert_allclose(
        candidate.logits.numpy(),
        np.asarray(output)[0, -1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert candidate.greedy() == int(mx.argmax(output[0, -1]).item())
    previous = candidate.logits.numpy().copy()
    candidate.reset()
    assert candidate.cached_tokens == 0
    assert not np.any(candidate.cache.numpy())
    with pytest.raises(RuntimeError, match="projected"):
        candidate.greedy()
    candidate.load_cache(cache)
    candidate.forward(11, 2)
    np.testing.assert_allclose(candidate.logits.numpy(), previous, rtol=tolerance, atol=tolerance)
    candidate.prepare()
    assert candidate.cached_tokens == 0


def test_forward_contract_rejects_uninitialized_or_invalid_state(staged_model):
    model, _ = staged_model
    candidate = Qwen3Staged(model, capacity=3)
    with pytest.raises(RuntimeError, match="prepare"):
        candidate.forward(3, 0)
    candidate.prepare()
    for token, position in ((-1, 0), (67, 0), (True, 0), (0, 3), (0, True), (0, 1)):
        with pytest.raises(ValueError):
            candidate.forward(token, position)
    with pytest.raises(ValueError, match="boolean"):
        candidate.forward(0, 0, project=1)
    with pytest.raises(ValueError, match="projection"):
        candidate.dispatches_per_forward(project=False, include_greedy=True)
