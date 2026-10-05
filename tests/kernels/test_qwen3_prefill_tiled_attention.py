import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels.qwen3_prefill import qwen3_prefill_attention
from metile_kernels.megakernels.qwen3_prefill_tiled_attention import qwen3_prefill_tiled_attention


def _trace(**overrides):
    constants = {
        "CHUNK": 9,
        "QUERY_HEADS": 4,
        "KV_HEADS": 2,
        "HEAD_DIM": 128,
        "LAYERS": 2,
        "MAX_CONTEXT": 17,
        "BLOCK": 256,
        "KEY_TILE": 16,
        "PARTITIONS": 8,
        "STORAGE_DTYPE": "f32",
    } | overrides
    storage = constants["STORAGE_DTYPE"]
    parameters = [
        ("Queries", PtrType(storage if storage in ("f16", "f32") else "f32")),
        ("KVCache", PtrType(storage if storage in ("f16", "f32") else "f32")),
        ("Control", PtrType("i32")),
        ("Attention", PtrType(storage if storage in ("f16", "f32") else "f32")),
        ("layer", I32),
    ]
    with TracingContext(qwen3_prefill_tiled_attention.name) as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        qwen3_prefill_tiled_attention.fn(*proxies, **constants)
    _mark_outputs(context.func)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("block", [32, 96, 256, 1024])
@pytest.mark.parametrize("dimension,key_tile", [(32, 8), (96, 16), (128, 16), (128, 32)])
def test_tiled_attention_uses_only_bounded_local_memory_and_two_key_tile_barriers(
    block, dimension, key_tile
):
    function = _trace(BLOCK=block, HEAD_DIM=dimension, KEY_TILE=key_tile)
    metal = lower(function)
    source = emit(metal)
    assert metal.threadgroup_size == (block, 1, 1)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(operation.size * 4 for operation in allocations) == 2 * key_tile * dimension * 4
    assert sum(isinstance(operation, tir.Barrier) for operation in _walk(function.ops)) == 2
    assert "atomic_" not in source and "mem_device" not in source and "half" not in source
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Attention"]
    cache = [tensor for tensor in function.tensors if tensor.ptr.name == "KVCache"]
    assert len(cache) == 1 and cache[0].access == "read"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"CHUNK": 0}, "dimensions"),
        ({"BLOCK": 31}, "BLOCK"),
        ({"HEAD_DIM": 33}, "HEAD_DIM"),
        ({"QUERY_HEADS": 3}, "QUERY_HEADS"),
        ({"KEY_TILE": 0}, "KEY_TILE"),
        ({"KEY_TILE": True}, "KEY_TILE"),
        ({"KEY_TILE": 40}, "32 KiB"),
        ({"PARTITIONS": 0}, "PARTITIONS"),
        ({"PARTITIONS": True}, "PARTITIONS"),
        ({"PARTITIONS": 3}, "PARTITIONS"),
        ({"PARTITIONS": 64}, "PARTITIONS"),
        ({"KEY_TILE": 12}, "PARTITIONS"),
        ({"STORAGE_DTYPE": "bf16"}, "STORAGE_DTYPE"),
    ],
)
def test_tiled_attention_rejects_invalid_shapes_and_oversized_shared_tiles(overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(**overrides)


def _reference(queries, cache, prefix, valid_rows):
    result = np.zeros_like(queries)
    dimension = queries.shape[-1]
    grouping = queries.shape[1] // cache.shape[2]
    for row in range(valid_rows):
        for head in range(queries.shape[1]):
            keys = cache[0, 1, head // grouping, : prefix + row + 1].astype(np.float64)
            values = cache[1, 1, head // grouping, : prefix + row + 1].astype(np.float64)
            scores = keys @ queries[row, head].astype(np.float64) / np.sqrt(dimension)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            result[row, head] = probabilities @ values
    return result


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("block", [32, 96, 256])
@pytest.mark.parametrize("dimension", [32, 96, 128])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_tiled_attention_matches_causal_gqa_with_prefix_and_nan_padding(
    block, dimension, dtype
):
    generator = np.random.default_rng(654)
    query = generator.normal(size=(9, 4, dimension)).astype(dtype)
    cache_values = generator.normal(size=(2, 2, 2, 17, dimension)).astype(dtype)
    prefix, valid_rows = 5, 6
    query[valid_rows:] = np.nan
    cache_values[:, :, :, prefix + valid_rows :] = np.nan
    expected = _reference(query, cache_values, prefix, valid_rows)
    queries = metile.Buffer(data=query)
    cache = metile.Buffer(data=cache_values)
    control = metile.Buffer(data=np.array([prefix, valid_rows], dtype=np.int32))
    output = metile.Buffer.empty(query.shape, dtype=dtype)
    dispatch = qwen3_prefill_tiled_attention[(metile.cdiv(9, block // 32), 4)].prepare(
        queries,
        cache,
        control,
        output,
        1,
        CHUNK=9,
        QUERY_HEADS=4,
        KV_HEADS=2,
        HEAD_DIM=dimension,
        LAYERS=2,
        MAX_CONTEXT=17,
        BLOCK=block,
        KEY_TILE=8,
        STORAGE_DTYPE="f16" if dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    tolerance = 0.002 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(output.numpy(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(output.numpy()[valid_rows:], 0)
    np.testing.assert_array_equal(cache.numpy(), cache_values)
    control.numpy()[:] = (0, 0)
    dispatch()
    np.testing.assert_array_equal(output.numpy(), 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
def test_gpu_tiled_attention_handles_large_scores_and_partial_last_key_tile():
    generator = np.random.default_rng(883)
    query = generator.normal(0, 80, size=(7, 4, 128)).astype(np.float32)
    cache = generator.normal(0, 80, size=(2, 2, 2, 45, 128)).astype(np.float32)
    cache[:, :, :, 44:] = np.nan
    expected = _reference(query, cache, 37, 7)
    output = np.empty_like(query)
    qwen3_prefill_tiled_attention[(1, 4)].prepare(
        query,
        cache,
        np.array([37, 7], dtype=np.int32),
        output,
        1,
        CHUNK=7,
        QUERY_HEADS=4,
        KV_HEADS=2,
        HEAD_DIM=128,
        LAYERS=2,
        MAX_CONTEXT=45,
        BLOCK=256,
        KEY_TILE=16,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("prefix", [0, 57, 513])
@pytest.mark.parametrize("block", [128, 256, 512])
def test_gpu_eight_partitions_match_key_parallel_attention_bitwise(prefix, block):
    generator = np.random.default_rng(667)
    capacity = prefix + 11
    queries = generator.normal(0, 3, size=(9, 4, 128)).astype(np.float32)
    cache = generator.normal(size=(2, 2, 2, capacity, 128)).astype(np.float32)
    queries[7:] = np.nan
    cache[:, :, :, prefix + 7 :] = np.nan
    control = np.array([prefix, 7], dtype=np.int32)
    expected = np.empty_like(queries)
    actual = np.empty_like(queries)
    constants = dict(
        CHUNK=9,
        QUERY_HEADS=4,
        KV_HEADS=2,
        HEAD_DIM=128,
        LAYERS=2,
        MAX_CONTEXT=capacity,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    qwen3_prefill_attention[(4, 9)].prepare(
        queries, cache, control, expected, 1, BLOCK=256, **constants
    )
    qwen3_prefill_tiled_attention[(metile.cdiv(9, block // 32), 4)].prepare(
        queries, cache, control, actual, 1, BLOCK=block, KEY_TILE=16, PARTITIONS=8, **constants
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
