"""Portable contracts and small Metal numerical checks for chunked Qwen3 rows."""

import inspect
import os
import sys

import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels import qwen3_prefill as stages
from metile_kernels.megakernels.qwen3 import qwen3_layer_offsets

STAGES = tuple(
    getattr(stages, f"qwen3_prefill_{name}")
    for name in (
        "embedding",
        "rmsnorm",
        "qk_rope",
        "attention",
        "residual",
        "swiglu",
        "last_hidden",
    )
)
metal = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


def _trace(stage, **overrides):
    geometry = (
        dict(
            CHUNK=4,
            HIDDEN=32,
            INTERMEDIATE=64,
            QUERY_HEADS=3,
            KV_HEADS=1,
            HEAD_DIM=96,
            LAYERS=2,
            VOCAB=67,
            MAX_CONTEXT=16,
            WIDTH=32,
            WEIGHT_OFFSET=32,
            WEIGHT_STRIDE=24576,
            EPS=1e-6,
            BLOCK=128,
            STORAGE_DTYPE="f32",
            DECODE=False,
            FUSED_ARITHMETIC=False,
        )
        | overrides
    )
    signature = inspect.signature(stage.fn)
    constants = {
        name: geometry[name]
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }
    storage = geometry["STORAGE_DTYPE"]
    parameters = []
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if name == "layer":
            dtype = I32
        elif name in ("Tokens", "Control"):
            dtype = PtrType("i32")
        elif name == "Rotary":
            dtype = PtrType("f32")
        else:
            dtype = PtrType(storage if storage in ("f16", "f32") else "f32")
        parameters.append((name, dtype))
    with TracingContext(stage.fn.__name__) as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        stage.fn(*[TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters], **constants)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("stage", [stages.qwen3_prefill_rmsnorm, stages.qwen3_prefill_qk_rope])
@pytest.mark.parametrize("fused", [False, True])
def test_prefill_native_rounding_uses_only_explicit_fma(stage, fused):
    source = emit(lower(_trace(stage, FUSED_ARITHMETIC=fused)))
    assert ("fma(" in source) == fused


@pytest.mark.parametrize(
    "stage",
    [stages.qwen3_prefill_rmsnorm, stages.qwen3_prefill_qk_rope, stages.qwen3_prefill_swiglu],
)
def test_prefill_native_rounding_rejects_nonboolean_mode(stage):
    with pytest.raises(ValueError, match="FUSED_ARITHMETIC"):
        _trace(stage, FUSED_ARITHMETIC=1)


@pytest.mark.parametrize("fused", [False, True])
def test_prefill_native_swiglu_uses_explicit_fast_exp_without_fma(fused):
    source = emit(lower(_trace(stages.qwen3_prefill_swiglu, FUSED_ARITHMETIC=fused)))
    assert ("fast::exp(" in source) == fused
    assert "fma(" not in source


def _native_mlx():
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("strict FP32 reference requires MLX_ENABLE_TF32=0 before importing MLX")
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available():
        pytest.skip("requires MLX Metal support")
    return core


@metal
@pytest.mark.parametrize("width", [128, 1024])
def test_gpu_prefill_fused_rms_matches_native_bitwise(width):
    core = _native_mlx()
    generator = np.random.default_rng(7231)
    source = generator.normal(0, 1.5, (39, width)).astype(np.float32)
    weights = generator.uniform(0.2, 2.5, width).astype(np.float32)
    expected = np.asarray(core.fast.rms_norm(core.array(source), core.array(weights), 1e-6))
    actual = np.empty_like(source)
    stages.qwen3_prefill_rmsnorm[(39,)].prepare(
        source,
        weights,
        actual,
        0,
        CHUNK=39,
        WIDTH=width,
        BLOCK=256,
        STORAGE_DTYPE="f32",
        FUSED_ARITHMETIC=True,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@metal
def test_gpu_prefill_native_swiglu_matches_mlx_bitwise():
    core = _native_mlx()
    activation = pytest.importorskip("mlx_lm.models.activations").swiglu
    generator = np.random.default_rng(93742)
    gate_up = generator.normal(0, 3, (35, 256)).astype(np.float32)
    expected = np.asarray(activation(core.array(gate_up[:, :128]), core.array(gate_up[:, 128:])))
    actual = np.empty_like(expected)
    stages.qwen3_prefill_swiglu[(35,)].prepare(
        gate_up,
        actual,
        CHUNK=35,
        INTERMEDIATE=128,
        BLOCK=128,
        STORAGE_DTYPE="f32",
        FUSED_ARITHMETIC=True,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@metal
def test_gpu_prefill_fused_qk_rope_matches_native_bitwise():
    core = _native_mlx()
    generator = np.random.default_rng(9863)
    rows, valid_rows, prefix, capacity = 39, 35, 513, 553
    dimension, query_heads, kv_heads = 128, 4, 2
    layout = qwen3_layer_offsets(32, 64, query_heads, kv_heads, dimension)
    weights = np.zeros(layout["layer_size"], dtype=np.float32)
    query_weight = generator.uniform(0.2, 2.5, dimension).astype(np.float32)
    key_weight = generator.uniform(0.2, 2.5, dimension).astype(np.float32)
    weights[layout["q_norm"] : layout["q_norm"] + dimension] = query_weight
    weights[layout["k_norm"] : layout["k_norm"] + dimension] = key_weight
    qkv = generator.normal(0, 1.5, (rows, query_heads + 2 * kv_heads, dimension)).astype(np.float32)
    qkv[valid_rows:] = np.nan
    basis = np.zeros((1, capacity, dimension), dtype=np.float32)
    basis[:, :, : dimension // 2] = 1.0
    rotated = np.asarray(
        core.fast.rope(
            core.array(basis),
            dims=dimension,
            traditional=False,
            base=1_000_000.0,
            scale=1.0,
            offset=0,
        )
    )[0]
    rotary = np.stack((rotated[:, : dimension // 2], rotated[:, dimension // 2 :]), axis=-1)
    expected = []
    for first, count, weight in (
        (0, query_heads, query_weight),
        (query_heads, kv_heads, key_weight),
    ):
        normalized = core.fast.rms_norm(
            core.array(qkv[None, :valid_rows, first : first + count]), core.array(weight), 1e-6
        )
        result = core.fast.rope(
            normalized.transpose(0, 2, 1, 3),
            dims=dimension,
            traditional=False,
            base=1_000_000.0,
            scale=1.0,
            offset=prefix,
        )
        expected.append(np.ascontiguousarray(np.asarray(result)[0].transpose(1, 0, 2)))
    queries = np.empty((rows, query_heads, dimension), dtype=np.float32)
    cache = np.full((2, 1, kv_heads, capacity, dimension), -7.0, dtype=np.float32)
    stages.qwen3_prefill_qk_rope[(1, rows)].prepare(
        qkv,
        weights,
        rotary,
        np.array([prefix, valid_rows], dtype=np.int32),
        queries,
        cache,
        0,
        CHUNK=rows,
        HIDDEN=32,
        INTERMEDIATE=64,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=dimension,
        LAYERS=1,
        MAX_CONTEXT=capacity,
        BLOCK=256,
        STORAGE_DTYPE="f32",
        FUSED_ARITHMETIC=True,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(queries[:valid_rows].view(np.uint32), expected[0].view(np.uint32))
    actual_keys = np.ascontiguousarray(
        cache[0, 0, :, prefix : prefix + valid_rows].transpose(1, 0, 2)
    )
    np.testing.assert_array_equal(actual_keys.view(np.uint32), expected[1].view(np.uint32))
    np.testing.assert_array_equal(queries[valid_rows:], 0.0)
    np.testing.assert_array_equal(cache[:, :, :, :prefix], -7.0)
    np.testing.assert_array_equal(cache[:, :, :, prefix + valid_rows :], -7.0)


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.fn.__name__)
@pytest.mark.parametrize("storage", ["f16", "f32"])
@pytest.mark.parametrize("block", [32, 128, 256])
def test_prefill_stages_lower_pure_dsl_with_bounded_local_synchronization(stage, storage, block):
    function = _trace(stage, STORAGE_DTYPE=storage, BLOCK=block)
    lowered = lower(function)
    source = emit(lowered)
    assert source.count("[[kernel") == 1
    assert lowered.threadgroup_size == (block, 1, 1)
    assert "atomic_" not in source and "mem_device" not in source
    barriers = [
        operation for operation in _walk(function.ops) if isinstance(operation, tir.Barrier)
    ]
    allocations = [
        operation
        for operation in _walk(lowered.ops)
        if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    if stage is stages.qwen3_prefill_attention:
        assert len(barriers) == 1
        assert sum(allocation.size * 4 for allocation in allocations) == 4 * (96 + 2) * (
            block // 32
        )
    elif stage is stages.qwen3_prefill_rmsnorm:
        assert len(barriers) == 1
        assert sum(allocation.size * 4 for allocation in allocations) == 4
    else:
        assert not barriers and not allocations
    if storage == "f32":
        assert "half" not in source
    assert "threadgroup_position_in_grid" in source
    assert not any(
        isinstance(operation, (tir.Dot, tir.PersistentRange)) for operation in _walk(function.ops)
    )


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.fn.__name__)
@pytest.mark.parametrize(
    "overrides", [{"CHUNK": 0}, {"CHUNK": True}, {"BLOCK": 31}, {"STORAGE_DTYPE": "bf16"}]
)
def test_invalid_execution_contract_is_rejected(stage, overrides):
    with pytest.raises(ValueError):
        _trace(stage, **overrides)


@pytest.mark.parametrize("stage", [stages.qwen3_prefill_attention, stages.qwen3_prefill_qk_rope])
@pytest.mark.parametrize("overrides", [{"DECODE": True}, {"DECODE": 1, "CHUNK": 1}])
def test_decode_control_requires_boolean_mode_and_one_row(stage, overrides):
    with pytest.raises(ValueError, match="DECODE"):
        _trace(stage, **overrides)


def test_decode_mode_accepts_shared_staged_control():
    assert "[[kernel" in emit(lower(_trace(stages.qwen3_prefill_attention, DECODE=True, CHUNK=1)))


@metal
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_embedding_zeroes_padding_and_last_hidden_uses_last_valid_row(dtype):
    embeddings = np.arange(67 * 32).reshape(67, 32).astype(dtype)
    tokens = np.array([3, 7, -(2**31), 2**31 - 1], dtype=np.int32)
    control = np.array([8, 2], dtype=np.int32)
    hidden = np.full((4, 32), np.nan, dtype=dtype)
    selected = np.full(32, np.nan, dtype=dtype)
    storage = "f16" if dtype == np.float16 else "f32"
    stages.qwen3_prefill_embedding[(1,)].prepare(
        tokens,
        control,
        embeddings,
        hidden,
        CHUNK=4,
        HIDDEN=32,
        VOCAB=67,
        BLOCK=128,
        STORAGE_DTYPE=storage,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(hidden[:2], embeddings[tokens[:2]])
    np.testing.assert_array_equal(hidden[2:], 0)
    stages.qwen3_prefill_last_hidden[(1,)].prepare(
        hidden,
        control,
        selected,
        CHUNK=4,
        HIDDEN=32,
        BLOCK=128,
        STORAGE_DTYPE=storage,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(selected, embeddings[7])


@metal
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_batched_norm_and_residual_do_not_mix_rows(dtype):
    generator = np.random.default_rng(712)
    source = generator.normal(size=(4, 160)).astype(dtype)
    source[-1] = 0
    weights = generator.uniform(0.5, 1.5, size=181).astype(dtype)
    destination = np.empty_like(source)
    storage = "f16" if dtype == np.float16 else "f32"
    values = source.astype(np.float32)
    expected = (
        (values / np.sqrt(np.mean(values**2, axis=1, keepdims=True) + 1e-6))
        .astype(dtype)
        .astype(np.float32)
        * weights[21:].astype(np.float32)
    ).astype(dtype)
    stages.qwen3_prefill_rmsnorm[(4,)].prepare(
        source,
        weights,
        destination,
        1,
        CHUNK=4,
        WIDTH=160,
        WEIGHT_OFFSET=5,
        WEIGHT_STRIDE=16,
        BLOCK=128,
        STORAGE_DTYPE=storage,
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(destination, expected, rtol=2e-3, atol=1e-6)
    projected = generator.normal(size=(4, 160)).astype(dtype)
    stages.qwen3_prefill_residual[(5,)].prepare(
        projected,
        source,
        destination,
        CHUNK=4,
        WIDTH=160,
        BLOCK=128,
        STORAGE_DTYPE=storage,
        STRICT_MATH=True,
    )
    np.testing.assert_array_equal(
        destination, (source.astype(np.float32) + projected.astype(np.float32)).astype(dtype)
    )


@metal
def test_batched_swiglu_uses_gate_up_halves_with_storage_rounding():
    generator = np.random.default_rng(251)
    gate_up = generator.normal(size=(3, 128)).astype(np.float32)
    result = np.empty((3, 64), dtype=np.float32)
    gate, up = np.split(gate_up, 2, axis=1)
    expected = gate / (1 + np.exp(-gate)) * up
    stages.qwen3_prefill_swiglu[(2,)].prepare(
        gate_up, result, CHUNK=3, INTERMEDIATE=64, BLOCK=128, STORAGE_DTYPE="f32", STRICT_MATH=True
    )
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-6)


@metal
@pytest.mark.parametrize("dimension", [32, 96, 128])
def test_chunk_qk_rope_writes_only_valid_rows_and_preserves_prefix(dimension):
    generator = np.random.default_rng(424)
    geometry = dict(HIDDEN=32, INTERMEDIATE=64, QUERY_HEADS=2, KV_HEADS=1, HEAD_DIM=dimension)
    offsets = qwen3_layer_offsets(*geometry.values())
    weights = generator.uniform(0.5, 1.5, size=(2, offsets["layer_size"])).astype(np.float32)
    qkv = generator.normal(size=(4, 4, dimension)).astype(np.float32)
    qkv[2:] = np.nan
    angles = generator.uniform(-1, 1, size=(7, dimension // 2)).astype(np.float32)
    rotary = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    queries = np.full((4, 2, dimension), np.nan, dtype=np.float32)
    cache = np.full((2, 2, 1, 7, dimension), 321, dtype=np.float32)
    control = np.array([5, 2], dtype=np.int32)
    expected_cache = cache.copy()
    normalized = qkv[:2, :3].copy()
    normalized /= np.sqrt(np.mean(normalized**2, axis=-1, keepdims=True) + 1e-6)
    query_norm = weights[1, offsets["q_norm"] : offsets["k_norm"]]
    key_norm = weights[1, offsets["k_norm"] :]
    normalized *= np.stack((query_norm, query_norm, key_norm))[None]
    left, right = np.split(normalized, 2, axis=-1)
    cosine, sine = rotary[5:7, None, :, 0], rotary[5:7, None, :, 1]
    rotated = np.concatenate((left * cosine - right * sine, right * cosine + left * sine), axis=-1)
    expected_cache[0, 1, 0, 5:7] = rotated[:, 2]
    expected_cache[1, 1, 0, 5:7] = qkv[:2, 3]
    stages.qwen3_prefill_qk_rope[(1, 4)].prepare(
        qkv,
        weights,
        rotary,
        control,
        queries,
        cache,
        1,
        CHUNK=4,
        **geometry,
        LAYERS=2,
        MAX_CONTEXT=7,
        BLOCK=128,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(queries[:2], rotated[:, :2], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(queries[2:], 0)
    np.testing.assert_allclose(cache, expected_cache, rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(cache[:, :, :, :5], expected_cache[:, :, :, :5])


@metal
@pytest.mark.parametrize("block", [32, 96, 128, 256])
@pytest.mark.parametrize("dimension", [32, 96, 128])
def test_key_parallel_attention_is_causal_includes_prefix_and_zeroes_padding(block, dimension):
    generator = np.random.default_rng(973)
    query = generator.normal(size=(4, 3, dimension)).astype(np.float32)
    query[-1] = np.nan
    cache = generator.normal(size=(2, 2, 1, 11, dimension)).astype(np.float32)
    cache[:, :, :, 6:] = np.nan
    control = np.array([3, 3], dtype=np.int32)
    output = np.full_like(query, np.nan)
    expected = np.zeros_like(query)
    for row in range(3):
        keys = cache[0, 1, 0, : 4 + row]
        values = cache[1, 1, 0, : 4 + row]
        scores = query[row] @ keys.T / np.sqrt(dimension)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected[row] = probabilities @ values
    stages.qwen3_prefill_attention[(3, 4)].prepare(
        query,
        cache,
        control,
        output,
        1,
        CHUNK=4,
        QUERY_HEADS=3,
        KV_HEADS=1,
        HEAD_DIM=dimension,
        LAYERS=2,
        MAX_CONTEXT=11,
        BLOCK=block,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(output[-1], 0)


@metal
@pytest.mark.parametrize("position", [0, 9])
def test_decode_mode_matches_prefill_row_with_existing_control_layout(position):
    generator = np.random.default_rng(91)
    queries = generator.normal(size=(1, 4, 128)).astype(np.float32)
    cache = generator.normal(size=(2, 1, 2, 10, 128)).astype(np.float32)
    prefill_control = np.array([position, 1], dtype=np.int32)
    decode_control = np.array([52, position], dtype=np.int32)
    expected, actual = np.empty_like(queries), np.empty_like(queries)
    common = dict(
        CHUNK=1,
        QUERY_HEADS=4,
        KV_HEADS=2,
        HEAD_DIM=128,
        LAYERS=1,
        MAX_CONTEXT=10,
        BLOCK=256,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    stages.qwen3_prefill_attention[(4, 1)].prepare(
        queries, cache, prefill_control, expected, 0, **common
    )
    stages.qwen3_prefill_attention[(4, 1)].prepare(
        queries, cache, decode_control, actual, 0, DECODE=True, **common
    )
    np.testing.assert_array_equal(actual, expected)
