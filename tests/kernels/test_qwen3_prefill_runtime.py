"""Complete chunked-prefill state must match unmodified native Qwen3."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.megakernels.qwen3_prefill_runtime import Qwen3ChunkedPrefill


@pytest.fixture
def host_request(monkeypatch):
    events = []

    def buffer(values):
        return SimpleNamespace(
            numpy=lambda: values, shape=values.shape, dtype=values.dtype, nbytes=values.nbytes
        )

    monkeypatch.setattr(
        "benchmarks.megakernels.qwen3_prefill_runtime.metile.Buffer.empty",
        lambda shape, dtype: buffer(np.empty(shape, dtype=dtype)),
    )
    monkeypatch.setattr(
        "benchmarks.megakernels.qwen3_prefill_runtime.metile.Buffer.zeros",
        lambda shape, dtype: buffer(np.zeros(shape, dtype=dtype)),
    )
    monkeypatch.setattr(
        "benchmarks.megakernels.qwen3_prefill_runtime.MetalDevice.get",
        lambda: SimpleNamespace(sync=lambda: events.append("sync")),
    )
    request = object.__new__(Qwen3ChunkedPrefill)
    request.config = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "num_hidden_layers": 2,
        "vocab_size": 67,
        "prefill_attention": {"kind": "matrix", "tile": [32, 16]},
        "decode_weight_packing": {"verification": {"roundtrip_bitwise": True}},
    }
    request.dtype = np.dtype(np.float32)
    request._mx_dtype = "float32"
    request.capacity = 12
    request.threads = 256
    request.chunk_size = 3
    request.scratch_bytes = 1024
    request.parameter_count = 100
    request.projection_backend = "simdgroup"
    request.projection_tile = (64, 64, 32)
    request.attention_backend = "matrix"
    request.prefill_weight_bytes = 16
    request._packed_decode_weights = True
    for name in ("embedding", "layer_weights", "final_norm", "rotary"):
        setattr(request, name, buffer(np.arange(4, dtype=np.float32)))
    for name in ("decode_layer_weights", "decode_embedding"):
        setattr(request, name, buffer(np.arange(2, dtype=np.uint32)))
    request.prefill_weights = ({"qkv": buffer(np.arange(4, dtype=np.float32))},)
    request._allocate_cache_state()
    request._allocate_decode_state()
    request._allocate_prefill_state()
    request._prepared = request._prefill_prepared = request._rotary_prepared = True
    request._prefill_cache_body = (lambda: events.append("cache"),)
    request._prefill_body = (*request._prefill_cache_body, lambda: events.append("tail"))
    request._projection = (lambda: events.append("projection"),)
    request._last_hidden = lambda: events.append("last_hidden")
    request.cached_tokens = 5
    request._logits_valid = request._last_prefill_available = True
    request.prefill_tokens.numpy().fill(9)
    request.prefill_control.numpy()[:] = (2, 3)
    request.cache.numpy().fill(7)
    return request, events


def test_fork_request_shares_only_finalized_model_storage(host_request):
    parent, events = host_request
    child = parent.fork_request()
    assert events == ["sync"]
    for name in (
        "embedding",
        "layer_weights",
        "final_norm",
        "rotary",
        "decode_layer_weights",
        "decode_embedding",
    ):
        assert getattr(child, name) is getattr(parent, name)
    assert child.prefill_weights is not parent.prefill_weights
    assert child.prefill_weights[0] is not parent.prefill_weights[0]
    assert child.prefill_weights[0]["qkv"] is parent.prefill_weights[0]["qkv"]
    child.prefill_weights[0]["unused"] = None
    assert "unused" not in parent.prefill_weights[0]
    child.config["prefill_attention"]["tile"][0] = 8
    child.config["decode_weight_packing"]["verification"]["roundtrip_bitwise"] = False
    assert parent.config["prefill_attention"]["tile"] == [32, 16]
    assert parent.config["decode_weight_packing"]["verification"]["roundtrip_bitwise"]
    for name in (
        "cache",
        "logits",
        "control",
        "hidden",
        "residual",
        "normalized",
        "qkv",
        "queries",
        "attention",
        "intermediate",
        "prefill_tokens",
        "prefill_control",
        "prefill_hidden",
        "prefill_residual",
        "prefill_normalized",
        "prefill_qkv",
        "prefill_queries",
        "prefill_attention",
        "prefill_projected",
        "prefill_gate_up",
        "prefill_intermediate",
    ):
        assert getattr(child, name) is not getattr(parent, name)
        assert not np.shares_memory(getattr(child, name).numpy(), getattr(parent, name).numpy())
    assert child.capacity == parent.capacity
    assert child.chunk_size == parent.chunk_size
    assert child._packed_decode_weights
    assert child._rotary_prepared
    assert not child._prepared
    assert not child._prefill_prepared
    assert not child._logits_valid
    assert not child._last_prefill_available
    assert child._body == child._projection == child._selection == ()
    assert child._prefill_body == child._prefill_cache_body == ()
    assert child.cached_tokens == 0
    assert not np.any(child.cache.numpy())
    assert parent.cached_tokens == 5
    assert parent._logits_valid and parent._last_prefill_available
    np.testing.assert_array_equal(parent.cache.numpy(), 7)


@pytest.mark.parametrize("flag", ["_prepared", "_prefill_prepared", "_rotary_prepared"])
def test_fork_request_requires_a_fully_prepared_source(host_request, flag):
    parent, events = host_request
    setattr(parent, flag, False)
    with pytest.raises(RuntimeError, match="prepare the source"):
        parent.fork_request()
    assert events == []


@pytest.mark.parametrize(
    "tokens,final_chunk",
    [([], True), ([1, 2, 3, 4], True), ([True], True), ([-1], True), ([67], True), ([1], 1)],
)
def test_prefill_chunk_validation_preserves_state(host_request, tokens, final_chunk):
    request, events = host_request
    with pytest.raises(ValueError):
        request.prefill_chunk(tokens, final_chunk=final_chunk)
    assert events == []
    assert request.cached_tokens == 5
    assert request._logits_valid and request._last_prefill_available
    np.testing.assert_array_equal(request.prefill_tokens.numpy(), 9)
    np.testing.assert_array_equal(request.prefill_control.numpy(), [2, 3])


def test_prefill_chunk_preserves_cache_only_pruning_and_final_projection(host_request):
    request, events = host_request
    request.prefill_chunk([3, 7], final_chunk=False)
    assert events == ["cache"]
    assert request.cached_tokens == 7
    assert not request._logits_valid
    assert not request._last_prefill_available
    np.testing.assert_array_equal(request.prefill_tokens.numpy(), [3, 7, 0])
    np.testing.assert_array_equal(request.prefill_control.numpy(), [5, 2])
    with pytest.raises(RuntimeError, match="nonempty prefill"):
        request.project_last_prefill()
    request.prefill_chunk([11], final_chunk=True)
    assert request.cached_tokens == 8
    assert not request._logits_valid
    assert request._last_prefill_available
    request.project_last_prefill()
    assert request._logits_valid
    assert events == ["cache", "cache", "tail", "last_hidden", "projection"]


def test_prefill_wrapper_validates_whole_prompt_before_mutating(host_request):
    request, events = host_request
    for tokens in ([3, 7, 11, 67], [1] * 8):
        with pytest.raises(ValueError):
            request.prefill(tokens)
    assert events == []
    assert request.cached_tokens == 5
    request.prefill([])
    assert request._logits_valid and request._last_prefill_available
    request.prefill([3, 7, 11, 5])
    assert events == ["cache", "cache", "tail"]
    assert request.cached_tokens == 9


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_reset_policy_rejects_nonbooleans_without_mutating_state(host_request, invalid):
    request, events = host_request
    with pytest.raises(ValueError, match="clear_cache must be bool"):
        request.reset(clear_cache=invalid)
    assert request.cached_tokens == 5
    assert request._logits_valid and request._last_prefill_available
    assert events == []
    np.testing.assert_array_equal(request.cache.numpy(), 7)


def test_logical_reset_never_touches_cache_or_synchronizes(host_request, monkeypatch):
    request, events = host_request
    values = request.cache.numpy()
    values[:] = np.nan

    def forbidden_access():
        raise AssertionError("logical reset must not access physical cache storage")

    monkeypatch.setattr(request.cache, "numpy", forbidden_access)
    request.reset(clear_cache=False)
    assert request.cached_tokens == 0
    assert not request._logits_valid
    assert not request._last_prefill_available
    assert events == []
    assert np.all(np.isnan(values))


def test_default_reset_still_clears_every_cache_slot(host_request):
    request, _ = host_request
    request.cache.numpy().fill(np.nan)
    request.reset()
    assert request.cached_tokens == 0
    assert not request._logits_valid
    assert not request._last_prefill_available
    np.testing.assert_array_equal(request.cache.numpy(), 0)


@pytest.fixture(params=[("float32", 32), ("float32", 128), ("float16", 128)])
def small_model(request):
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    dtype, dimension = request.param
    if dtype == "float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("strict FP32 reference requires MLX_ENABLE_TF32=0 before importing MLX")
    core = pytest.importorskip("mlx.core")
    qwen = pytest.importorskip("mlx_lm.models.qwen3")
    core.random.seed(982)
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
    return model, 1e-4 if dtype == "float32" else 0.02


def _compare(candidate, reference, cache, valid, tolerance):
    np.testing.assert_allclose(
        candidate.logits.numpy(),
        np.asarray(reference)[0, -1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert candidate.greedy() == int(np.argmax(np.asarray(reference)[0, -1]))
    actual = candidate.cache.numpy()
    for layer, entry in enumerate(cache):
        for kind, expected in enumerate((entry.keys, entry.values)):
            np.testing.assert_allclose(
                actual[kind, layer, :, :valid],
                np.asarray(expected)[0, :, :valid],
                rtol=tolerance,
                atol=tolerance,
            )


@pytest.mark.parametrize("chunk_size", [3, 8, 32])
@pytest.mark.parametrize("attention_backend", ["tiled", "matrix"])
def test_chunk_boundaries_padding_prefix_append_and_autoregressive_decode(
    small_model, chunk_size, attention_backend
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model, tolerance = small_model
    candidate = Qwen3ChunkedPrefill(
        model, 12, chunk_size=chunk_size, attention_backend=attention_backend
    ).prepare()
    assert candidate.config["threads"] == 256
    assert candidate.config["prefill_strict_math"] is True
    assert candidate.config["prefill_projection_relaxed_precision"] is False
    assert candidate.config["prefill_prunes_unused_final_layer"] is True
    assert candidate.config["fused_row_arithmetic"] is (attention_backend == "matrix")
    if attention_backend == "tiled":
        assert candidate.config["prefill_attention"] == {
            "kind": "query_tiled_shared_kv",
            "query_rows_per_threadgroup": 8,
            "key_tile": 16,
            "softmax_partitions_per_query": 8,
        }
    else:
        assert candidate.config["prefill_attention"] == {
            "kind": "matrix_tiled_online_softmax",
            "query_rows_per_threadgroup": 32,
            "key_tile": 16,
            "threads": 128,
            "shared_padding": 0,
            "unroll_mma": True,
            "cached_query_fragments": candidate.config["head_dim"] // 8,
            "query_kv_shared_storage": False,
            "direct_device_memory": True,
            "masked_device_tile_scratch_bytes": 256 * (2 if candidate.dtype == np.float16 else 4),
            "softmax_lanes": 8,
            "transpose_keys": True,
            "register_stats": True,
            "unroll_softmax": True,
            "load_vector": 16,
            "softmax_base2": True,
            "softmax_exponential": "fast_exp2",
            "denominator_fma": True,
            "normalization": "divide",
            "compiler_backend": "simdgroup_inline",
        }
    cache = make_prompt_cache(model)
    assert candidate.cached_tokens == 0
    candidate.cache.numpy().fill(np.nan)
    tokens = [3, 7, 11, 5, 9, 13, 18]
    for chunk in (tokens[:5], tokens[5:]):
        reference = model(mx.array([chunk], dtype=mx.int32), cache=cache)
        mx.eval(reference, [layer.state for layer in cache])
        candidate.prefill(chunk)
        with pytest.raises(RuntimeError, match="projected"):
            candidate.greedy()
        candidate.project_last_prefill()
        _compare(candidate, reference, cache, candidate.cached_tokens, tolerance)
        assert np.all(np.isnan(candidate.cache.numpy()[:, :, :, candidate.cached_tokens :]))
    assert candidate.cached_tokens == 7
    for position in range(7, 10):
        token = candidate.greedy()
        reference = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(reference, [layer.state for layer in cache])
        candidate.forward(token, position)
        _compare(candidate, reference, cache, position + 1, tolerance)
        with pytest.raises(RuntimeError, match="nonempty prefill"):
            candidate.project_last_prefill()
    assert len(candidate._prefill_body) == 23
    assert candidate._prefill_cache_body == candidate._prefill_body[:-8]
    chunks = (7 + chunk_size - 1) // chunk_size
    assert candidate.dispatches_per_prefill(7) == chunks * 23 - 8 * (chunks - 1)
    assert candidate.dispatches_per_prefill(0) == 0
    assert all(not dispatch._concurrent for dispatch in candidate._prefill_body)
    assert candidate._prefill_body[2]._grid.width == 1


@pytest.mark.parametrize("chunk_size", [2048, 4096])
def test_large_prefill_capacity_masks_padding(small_model, chunk_size):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model, tolerance = small_model
    candidate = Qwen3ChunkedPrefill(
        model, 12, chunk_size=chunk_size, attention_backend="matrix"
    ).prepare()
    cache = make_prompt_cache(model)
    tokens = [3, 7, 11, 5, 9]
    reference = model(mx.array([tokens], dtype=mx.int32), cache=cache)
    mx.eval(reference, [layer.state for layer in cache])
    candidate.cache.numpy().fill(np.nan)
    candidate.prefill(tokens)
    candidate.project_last_prefill()
    _compare(candidate, reference, cache, len(tokens), tolerance)
    assert np.all(np.isnan(candidate.cache.numpy()[:, :, :, len(tokens) :]))


@pytest.mark.parametrize("attention_backend", ["tiled", "matrix"])
def test_changing_chunk_size_reuses_weights_cache_and_decoder(small_model, attention_backend):
    model, tolerance = small_model
    candidate = Qwen3ChunkedPrefill(
        model, 12, chunk_size=3, attention_backend=attention_backend
    ).prepare()
    original = (
        candidate.embedding,
        candidate.layer_weights,
        candidate.prefill_weights,
        candidate.cache,
    )
    candidate.prefill([3, 7, 11, 5, 9])
    candidate.project_last_prefill()
    expected = candidate.logits.numpy().copy()
    for size in (8, 8, 2):
        candidate.set_chunk_size(size).prepare()
        assert candidate.cached_tokens == 0
        assert not np.any(candidate.cache.numpy())
        assert all(
            old is new
            for old, new in zip(
                original,
                (
                    candidate.embedding,
                    candidate.layer_weights,
                    candidate.prefill_weights,
                    candidate.cache,
                ),
                strict=True,
            )
        )
        candidate.prefill([3, 7, 11, 5, 9])
        candidate.project_last_prefill()
        np.testing.assert_allclose(
            candidate.logits.numpy(), expected, rtol=tolerance, atol=tolerance
        )


def test_prefill_contracts_fail_before_state_changes(small_model):
    model, _ = small_model
    candidate = Qwen3ChunkedPrefill(model, 4, chunk_size=3)
    with pytest.raises(RuntimeError, match="prepare"):
        candidate.prefill([3])
    candidate.prepare()
    with pytest.raises(RuntimeError, match="nonempty prefill"):
        candidate.project_last_prefill()
    for tokens in ([True], [-1], [67], [3, 7, 11, 5, 9]):
        with pytest.raises(ValueError):
            candidate.prefill(tokens)
        assert candidate.cached_tokens == 0
    candidate.prefill([])
    assert candidate.cached_tokens == 0
    candidate.prefill([3, 7, 11, 5])
    candidate.project_last_prefill()
    candidate.greedy()
    with pytest.raises(ValueError, match="capacity"):
        candidate.prefill([0])
    candidate.reset()
    with pytest.raises(RuntimeError, match="nonempty prefill"):
        candidate.project_last_prefill()
    for invalid in (True, 0, -1, 4097):
        with pytest.raises(ValueError, match="chunk_size"):
            candidate.set_chunk_size(invalid)


@pytest.mark.parametrize("chunk_size", [True, 0, -1, 1.5, 4097])
def test_invalid_chunk_size_is_rejected_before_loading_a_model(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        Qwen3ChunkedPrefill(None, 8, chunk_size=chunk_size)


@pytest.mark.parametrize("dimension", [288, 512, 1024])
def test_oversized_attention_tiles_are_rejected_before_allocating_model_buffers(dimension):
    model = SimpleNamespace(args=SimpleNamespace(head_dim=dimension))
    with pytest.raises(ValueError, match="head_dim <= 256"):
        Qwen3ChunkedPrefill(model, 8)


@pytest.mark.parametrize("dimension", [160, 192, 224, 256, 512])
def test_oversized_matrix_tiles_are_rejected_before_allocating_model_buffers(dimension):
    model = SimpleNamespace(args=SimpleNamespace(head_dim=dimension))
    with pytest.raises(ValueError, match="head_dim <= 128"):
        Qwen3ChunkedPrefill(model, 8, attention_backend="matrix")


def test_unknown_attention_backend_is_rejected_before_loading_a_model():
    with pytest.raises(ValueError, match="attention_backend"):
        Qwen3ChunkedPrefill(None, 8, attention_backend="native")


@pytest.mark.parametrize("invalid", [None, 0, 1, "auto"])
def test_lossless_decode_policy_requires_a_boolean_before_loading_model(invalid):
    with pytest.raises(ValueError, match="lossless_decode_weights must be bool"):
        Qwen3ChunkedPrefill(None, 8, lossless_decode_weights=invalid)


@pytest.mark.parametrize("small_model", [("float32", 32), ("float32", 128)], indirect=True)
@pytest.mark.parametrize("attention_backend", ["tiled", "matrix"])
def test_lossless_decode_packing_preserves_prefill_projection_and_advancing_decode(
    small_model, attention_backend
):
    import mlx.core as mx

    model, _ = small_model
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    options = dict(capacity=12, chunk_size=3, attention_backend=attention_backend)
    packed = Qwen3ChunkedPrefill(model, **options)
    unpacked = Qwen3ChunkedPrefill(model, lossless_decode_weights=False, **options)
    assert packed._packed_decode_weights
    assert not unpacked._packed_decode_weights
    assert unpacked.decode_layer_weights is unpacked.layer_weights
    assert unpacked.decode_embedding is unpacked.embedding
    assert "decode_weight_packing" not in unpacked.config
    assert packed.decode_layer_weights.dtype == np.uint32
    assert packed.decode_embedding.dtype == np.uint32
    metadata = packed.config["decode_weight_packing"]
    assert metadata["packed_bytes"] == (
        packed.decode_layer_weights.nbytes + packed.decode_embedding.nbytes
    )
    assert metadata["unpacked_bytes"] == packed.layer_weights.nbytes + packed.embedding.nbytes
    assert metadata["packed_bytes"] * 2 == metadata["unpacked_bytes"]
    for packed_layer, original_layer in zip(
        packed.prefill_weights, unpacked.prefill_weights, strict=True
    ):
        for name in packed_layer:
            assert packed_layer[name].dtype == np.float32
            np.testing.assert_array_equal(packed_layer[name].numpy(), original_layer[name].numpy())
    packed.prepare()
    unpacked.prepare()

    def assert_same_state():
        assert packed.cached_tokens == unpacked.cached_tokens
        for name in ("hidden", "logits", "cache"):
            np.testing.assert_array_equal(
                getattr(packed, name).numpy().view(np.uint32),
                getattr(unpacked, name).numpy().view(np.uint32),
            )
        assert packed.greedy() == unpacked.greedy()

    for tokens in ([3, 7, 11, 5, 9], [13, 18]):
        packed.prefill(tokens)
        unpacked.prefill(tokens)
        for candidate in (packed, unpacked):
            with pytest.raises(RuntimeError, match="projected"):
                candidate.greedy()
            candidate.project_last_prefill()
        assert_same_state()
    for position in range(7, 10):
        token = unpacked.greedy()
        packed.forward(token, position)
        unpacked.forward(token, position)
        assert_same_state()
    for candidate in (packed, unpacked):
        with pytest.raises(RuntimeError, match="before preparing"):
            candidate.pack_decode_weights()


@pytest.mark.parametrize("small_model", [("float32", 32)], indirect=True)
def test_lossless_decode_auto_policy_keeps_random_fp32_weights_unpacked(small_model):
    model, _ = small_model
    candidate = Qwen3ChunkedPrefill(model, 4, chunk_size=3)
    assert not candidate._packed_decode_weights
    assert not candidate.pack_decode_weights()
    assert candidate.decode_layer_weights is candidate.layer_weights
    assert candidate.decode_embedding is candidate.embedding
    assert "decode_weight_packing" not in candidate.config


@pytest.mark.parametrize("attention_backend", ["tiled", "matrix"])
def test_forked_requests_interleave_without_changing_results_or_parent_state(
    small_model, attention_backend, monkeypatch
):
    import mlx.core as mx

    model, _ = small_model
    if model.model.embed_tokens.weight.dtype == mx.float32:
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    parent = Qwen3ChunkedPrefill(
        model, 12, chunk_size=3, attention_backend=attention_backend
    ).prepare()
    parent.prefill([2, 4])
    parent.project_last_prefill()
    initial_cache = parent.cache.numpy().copy()
    initial_logits = parent.logits.numpy().copy()
    shared_weights = {
        name: getattr(parent, name).numpy().copy()
        for name in ("embedding", "layer_weights", "final_norm", "rotary")
    }
    monkeypatch.setattr("benchmarks.megakernels.qwen3_prefill_runtime.qwen3_rotary_table", None)
    child = parent.fork_request()
    with pytest.raises(RuntimeError, match="prepare"):
        child.prefill_chunk([3], final_chunk=True)
    child.prepare()
    assert parent.cached_tokens == 2
    np.testing.assert_array_equal(parent.cache.numpy(), initial_cache)
    np.testing.assert_array_equal(parent.logits.numpy(), initial_logits)
    assert parent._logits_valid and parent._last_prefill_available
    assert child.cached_tokens == 0
    assert not np.any(child.cache.numpy())
    assert child.config == parent.config
    for name in ("partial_values", "partial_indices", "selected"):
        assert getattr(child, name) is not getattr(parent, name)
    for name in ("_body", "_projection", "_selection", "_prefill_body"):
        assert all(
            first is not second
            for first, second in zip(getattr(parent, name), getattr(child, name), strict=True)
        )
    reference_parent = parent.fork_request().prepare()
    reference_child = parent.fork_request().prepare()
    reference_parent.prefill([3, 7, 11])
    reference_parent.project_last_prefill()
    for position in (3, 4):
        reference_parent.forward(reference_parent.greedy(), position)
    reference_child.prefill_chunk([5, 9], final_chunk=False)
    reference_child.prefill_chunk([13, 18, 21], final_chunk=True)
    reference_child.project_last_prefill()
    reference_child.forward(reference_child.greedy(), 5)
    parent.reset()
    parent.prefill([3, 7, 11])
    parent.project_last_prefill()
    child.prefill_chunk([5, 9], final_chunk=False)
    with pytest.raises(RuntimeError, match="nonempty prefill"):
        child.project_last_prefill()
    parent.forward(parent.greedy(), 3)
    child.prefill_chunk([13, 18, 21], final_chunk=True)
    child.project_last_prefill()
    parent.forward(parent.greedy(), 4)
    child.forward(child.greedy(), 5)
    for actual, expected in ((parent, reference_parent), (child, reference_child)):
        assert actual.cached_tokens == expected.cached_tokens
        for name in ("hidden", "logits", "cache"):
            np.testing.assert_array_equal(
                getattr(actual, name).numpy().view(np.uint8),
                getattr(expected, name).numpy().view(np.uint8),
            )
        assert actual.greedy() == expected.greedy()
    child.reset()
    child.set_chunk_size(2).prepare()
    assert child.chunk_size == 2 and parent.chunk_size == 3
    assert child.config["prefill_chunk_size"] == 2
    assert parent.config["prefill_chunk_size"] == 3
    assert child.cached_tokens == 0 and parent.cached_tokens == 5
    for name in ("cache", "logits"):
        np.testing.assert_array_equal(
            getattr(parent, name).numpy(), getattr(reference_parent, name).numpy()
        )
    for name, expected in shared_weights.items():
        np.testing.assert_array_equal(getattr(parent, name).numpy(), expected)


@pytest.mark.parametrize("attention_backend", ["tiled", "matrix"])
def test_logical_reset_handles_nan_tails_shorter_prefixes_and_pending_decode(
    small_model, attention_backend, monkeypatch
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from metile.runtime.metal_device import MetalDevice

    model, tolerance = small_model
    if model.model.embed_tokens.weight.dtype == mx.float32:
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    logical = Qwen3ChunkedPrefill(
        model, 12, chunk_size=3, attention_backend=attention_backend
    ).prepare()
    cleared = logical.fork_request().prepare()
    if logical.dtype == np.float32:
        assert logical._packed_decode_weights

    def compare_requests():
        assert logical.cached_tokens == cleared.cached_tokens
        for name in ("hidden", "logits"):
            np.testing.assert_array_equal(
                getattr(logical, name).numpy().view(np.uint8),
                getattr(cleared, name).numpy().view(np.uint8),
            )
        np.testing.assert_array_equal(
            logical.cache.numpy()[:, :, :, : logical.cached_tokens].view(np.uint8),
            cleared.cache.numpy()[:, :, :, : cleared.cached_tokens].view(np.uint8),
        )
        assert logical.greedy() == cleared.greedy()

    logical.prefill([2, 4, 6, 8, 10, 12, 14, 16, 18])
    logical.project_last_prefill()
    logical.forward(logical.greedy(), 9)
    old_cache = logical.cache.numpy().copy()
    logical.reset(clear_cache=False)
    np.testing.assert_array_equal(logical.cache.numpy().view(np.uint8), old_cache.view(np.uint8))
    logical.cache.numpy().fill(np.nan)
    logical.reset(clear_cache=False)
    assert np.all(np.isnan(logical.cache.numpy()))
    cleared.reset()
    for candidate in (logical, cleared):
        with pytest.raises(RuntimeError, match="projected"):
            candidate.greedy()
        with pytest.raises(RuntimeError, match="nonempty prefill"):
            candidate.project_last_prefill()
        candidate.prefill_chunk([5, 7], final_chunk=False)
        with pytest.raises(RuntimeError, match="nonempty prefill"):
            candidate.project_last_prefill()
        candidate.prefill_chunk([11], final_chunk=True)
        candidate.project_last_prefill()
    compare_requests()
    native_cache = make_prompt_cache(model)
    native = model(mx.array([[5, 7, 11]], dtype=mx.int32), cache=native_cache)
    mx.eval(native, [layer.state for layer in native_cache])
    _compare(logical, native, native_cache, 3, tolerance)
    assert np.all(np.isnan(logical.cache.numpy()[:, :, :, 3:]))
    for position in (3, 4):
        token = logical.greedy()
        logical.forward(token, position)
        cleared.forward(token, position)
        compare_requests()
        native = model(mx.array([[token]], dtype=mx.int32), cache=native_cache)
        mx.eval(native, [layer.state for layer in native_cache])
        _compare(logical, native, native_cache, position + 1, tolerance)
        assert np.all(np.isnan(logical.cache.numpy()[:, :, :, position + 1 :]))

    old_cache = logical.cache.numpy().copy()
    logical.reset(clear_cache=False)
    cleared.reset()
    logical.forward(3, 0)
    cleared.forward(3, 0)
    compare_requests()
    np.testing.assert_array_equal(
        logical.cache.numpy()[:, :, :, 1:].view(np.uint8), old_cache[:, :, :, 1:].view(np.uint8)
    )
    cleared.forward(7, 1)
    pending_expected = cleared.cache.numpy()[:, :, :, 1].copy()
    cleared.reset()
    logical.forward(7, 1)
    device = MetalDevice.get()
    pending_dispatches = device._pending_dispatches
    assert pending_dispatches > 0
    original_sync = device.sync
    synchronization_calls = []

    def counted_sync():
        synchronization_calls.append(True)
        original_sync()

    with monkeypatch.context() as patch:
        patch.setattr(device, "sync", counted_sync)
        logical.reset(clear_cache=False)
        assert not synchronization_calls
        assert device._pending_dispatches == pending_dispatches
        logical.forward(11, 0)
        assert synchronization_calls
    cleared.forward(11, 0)
    compare_requests()
    np.testing.assert_array_equal(
        logical.cache.numpy()[:, :, :, 1].view(np.uint8), pending_expected.view(np.uint8)
    )
