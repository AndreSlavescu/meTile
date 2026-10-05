import os
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
from metile_kernels.megakernels.qwen3_prefill_matrix_attention import (
    qwen3_prefill_matrix_attention,
)


def _trace(**overrides):
    constants = {
        "CHUNK": 9,
        "QUERY_HEADS": 4,
        "KV_HEADS": 2,
        "HEAD_DIM": 128,
        "LAYERS": 2,
        "MAX_CONTEXT": 45,
        "BLOCK": 128,
        "QUERY_TILE": 8,
        "KEY_TILE": 32,
        "SHARED_PADDING": 0,
        "UNROLL_MMA": False,
        "SOFTMAX_LANES": 32,
        "TRANSPOSE_KEYS": False,
        "REGISTER_STATS": False,
        "UNROLL_SOFTMAX": False,
        "LOAD_VECTOR": 1,
        "SOFTMAX_BASE2": False,
        "DIRECT_MEMORY": False,
        "STORAGE_DTYPE": "f32",
    } | overrides
    storage = constants["STORAGE_DTYPE"]
    dtype = storage if storage in ("f16", "f32") else "f32"
    parameters = [
        ("Queries", PtrType(dtype)),
        ("KVCache", PtrType(dtype)),
        ("Control", PtrType("i32")),
        ("Attention", PtrType(dtype)),
        ("layer", I32),
    ]
    with TracingContext(qwen3_prefill_matrix_attention.name) as context:
        context.func.constexprs.update(constants)
        context.func.constexprs.update(
            SCHEDULE=metile.Schedule(backend="simdgroup_inline"), STRICT_MATH=True
        )
        context.func.params = [tir.Param(name, kind) for name, kind in parameters]
        proxies = [TracingProxy(tir.Value(name, kind)) for name, kind in parameters]
        qwen3_prefill_matrix_attention.fn(*proxies, **constants)
    _mark_outputs(context.func)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("dimension", [32, 96, 128, 192])
@pytest.mark.parametrize("storage", ["f16", "f32"])
def test_matrix_attention_lowers_public_dot_to_bounded_fp32_matrix_fragments(dimension, storage):
    function = _trace(HEAD_DIM=dimension, STORAGE_DTYPE=storage)
    metal = lower(function)
    source = emit(metal)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(operation.size * 4 for operation in allocations) == 4 * (40 * dimension + 336)
    assert metal.threadgroup_size == (128, 1, 1)
    assert "simdgroup_multiply_accumulate" in source
    assert "atomic_" not in source and "mem_device" not in source
    if storage == "f32":
        assert "half" not in source
    operations = list(_walk(function.ops))
    assert sum(isinstance(operation, tir.Dot) for operation in operations) == dimension // 32 + 1
    assert sum(isinstance(operation, tir.Barrier) for operation in operations) == 7
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Attention"]
    cache = next(tensor for tensor in function.tensors if tensor.ptr.name == "KVCache")
    assert cache.access == "read"


@pytest.mark.parametrize("chunk,capacity", [(1, 1), (128, 4096), (1024, 5119)])
def test_matrix_attention_workspace_does_not_grow_with_prompt_or_chunk(chunk, capacity):
    metal = lower(_trace(CHUNK=chunk, MAX_CONTEXT=capacity))
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(operation.size * 4 for operation in allocations) == 21824


@pytest.mark.parametrize("query_tile,key_tile", [(8, 32), (32, 16)])
@pytest.mark.parametrize("padding", [0, 4])
@pytest.mark.parametrize("unroll", [False, True])
def test_matrix_attention_query_ownership_padding_and_unrolling(
    query_tile, key_tile, padding, unroll
):
    function = _trace(
        QUERY_TILE=query_tile, KEY_TILE=key_tile, SHARED_PADDING=padding, UNROLL_MMA=unroll
    )
    metal = lower(function)
    source = emit(metal)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    expected = 4 * (
        (query_tile + key_tile) * (128 + padding) + query_tile * (key_tile + padding + 10)
    )
    if query_tile == 32 and padding == 0 and unroll:
        expected -= 4 * key_tile * 128
    assert sum(operation.size * 4 for operation in allocations) == expected
    score_fragments = 1 if query_tile == 8 else 2
    output_fragments = 4 if query_tile == 8 else 16
    expected_dots = (
        score_fragments * 16 + output_fragments * (key_tile // 8)
        if unroll
        else score_fragments + output_fragments
    )
    assert sum(isinstance(operation, tir.Dot) for operation in _walk(function.ops)) == expected_dots
    assert source.count("simdgroup_multiply_accumulate") == expected_dots


@pytest.mark.parametrize(
    "query_tile,key_tile,padding,unroll,reuses_query",
    [
        (8, 32, 0, True, False),
        (32, 16, 4, True, False),
        (32, 16, 0, False, False),
        (32, 16, 0, True, True),
    ],
)
def test_matrix_attention_query_cache_is_derived_from_geometry(
    query_tile, key_tile, padding, unroll, reuses_query
):
    function = _trace(
        QUERY_TILE=query_tile, KEY_TILE=key_tile, SHARED_PADDING=padding, UNROLL_MMA=unroll
    )
    matrices = [tensor for tensor in function.tensors if tensor.block_shape == (8, 8)]
    query_matrix, key_matrix = matrices[:2]
    assert (query_matrix.ptr.name == key_matrix.ptr.name) == reuses_query
    query_loads = [
        operation
        for operation in function.ops
        if isinstance(operation, tir.TileLoad) and operation.tensor is query_matrix
    ]
    assert len(query_loads) == (16 if reuses_query else 0)
    if reuses_query:
        last_load = next(
            index for index, operation in enumerate(function.ops) if operation is query_loads[-1]
        )
        assert isinstance(function.ops[last_load + 1], tir.Barrier)
        assert not any(
            isinstance(operation, tir.TileLoad) and operation.tensor is query_matrix
            for top_level in function.ops
            for operation in _walk(getattr(top_level, "body", ()))
        )
    lower(function)


def test_matrix_attention_query_reuse_keeps_all_shared_allocations_bounded():
    constants = dict(
        QUERY_TILE=32, KEY_TILE=16, SHARED_PADDING=0, UNROLL_MMA=True, REGISTER_STATS=True
    )
    for dimension in (32, 128, 224):
        metal = lower(_trace(HEAD_DIM=dimension, **constants))
        allocations = [
            operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
        ]
        assert sum(operation.size * 4 for operation in allocations) == 4 * (32 * dimension + 768)
    with pytest.raises(ValueError, match="32 KiB"):
        _trace(HEAD_DIM=256, **constants)


@pytest.mark.parametrize("storage", ["f16", "f32"])
@pytest.mark.parametrize("dimension", [32, 128, 224])
def test_matrix_attention_direct_views_use_bounded_exclusive_scratch(storage, dimension):
    function = _trace(
        QUERY_TILE=32,
        KEY_TILE=16,
        SHARED_PADDING=0,
        UNROLL_MMA=True,
        REGISTER_STATS=True,
        SOFTMAX_BASE2=True,
        DIRECT_MEMORY=True,
        STORAGE_DTYPE=storage,
        HEAD_DIM=dimension,
    )
    device_operations = [
        operation
        for operation in _walk(function.ops)
        if isinstance(operation, (tir.TileLoad, tir.TileStore))
        and operation.tensor.address_space == "device"
    ]
    assert device_operations
    scratches = {operation.scratch.ptr.name for operation in device_operations}
    assert len(scratches) == 1
    assert all(operation.scratch.block_shape is None for operation in device_operations)
    assert not any(
        isinstance(operation, (tir.Load, tir.Store))
        and operation.tensor is not None
        and operation.tensor.ptr.name in scratches
        for operation in _walk(function.ops)
    )
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Attention"]
    metal = lower(function)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert len(allocations) == 3
    assert sum(
        operation.size * (2 if operation.elem_type == "half" else 4) for operation in allocations
    ) == (3584 if storage == "f16" else 4096)
    source = emit(metal)
    assert "simdgroup_load" in source and "simdgroup_store" in source
    assert "simdgroup_barrier" in source
    assert "atomic_" not in source and "mem_device" not in source


@pytest.mark.parametrize("lanes", [8, 16, 32])
@pytest.mark.parametrize("transpose_keys", [False, True])
@pytest.mark.parametrize("padding", [0, 4])
def test_matrix_attention_subgroups_and_transposed_staging_remain_bounded(
    lanes, transpose_keys, padding
):
    function = _trace(
        QUERY_TILE=32,
        KEY_TILE=16,
        SHARED_PADDING=padding,
        UNROLL_MMA=True,
        SOFTMAX_LANES=lanes,
        TRANSPOSE_KEYS=transpose_keys,
    )
    metal = lower(function)
    source = emit(metal)
    value_elements = 16 * (128 + padding)
    key_elements = 128 * (16 + padding) if transpose_keys else value_elements
    expected = 4 * (32 * (128 + padding) + max(value_elements, key_elements) + 32 * (26 + padding))
    if padding == 0:
        expected -= 4 * max(value_elements, key_elements)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(operation.size * 4 for operation in allocations) == expected
    assert expected <= 32768
    assert ("simd_shuffle_xor" in source) == (lanes < 32)
    assert ("ulong2(0), true" in source) == (not transpose_keys)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"CHUNK": 0}, "dimensions"),
        ({"HEAD_DIM": 33}, "HEAD_DIM"),
        ({"HEAD_DIM": 224}, "32 KiB"),
        ({"HEAD_DIM": 192, "SHARED_PADDING": 4}, "32 KiB"),
        ({"QUERY_HEADS": 3}, "QUERY_HEADS"),
        ({"BLOCK": 31}, "BLOCK"),
        ({"BLOCK": 256}, "BLOCK=128"),
        ({"QUERY_TILE": 16}, "QUERY_TILE/KEY_TILE"),
        ({"QUERY_TILE": True}, "QUERY_TILE/KEY_TILE"),
        ({"KEY_TILE": 16}, "QUERY_TILE/KEY_TILE"),
        ({"KEY_TILE": True}, "QUERY_TILE/KEY_TILE"),
        ({"QUERY_TILE": 32, "KEY_TILE": 16, "HEAD_DIM": 160}, "32 KiB"),
        ({"SHARED_PADDING": -1}, "SHARED_PADDING"),
        ({"SHARED_PADDING": True}, "SHARED_PADDING"),
        ({"UNROLL_MMA": 1}, "UNROLL_MMA"),
        ({"SOFTMAX_LANES": 8}, "SOFTMAX_LANES"),
        ({"SOFTMAX_LANES": 4}, "SOFTMAX_LANES"),
        ({"SOFTMAX_LANES": True}, "SOFTMAX_LANES"),
        ({"TRANSPOSE_KEYS": 1}, "TRANSPOSE_KEYS"),
        ({"REGISTER_STATS": 1}, "REGISTER_STATS"),
        ({"UNROLL_SOFTMAX": 1}, "UNROLL_SOFTMAX"),
        ({"LOAD_VECTOR": 2}, "LOAD_VECTOR"),
        ({"LOAD_VECTOR": True}, "LOAD_VECTOR"),
        ({"SOFTMAX_BASE2": 1}, "SOFTMAX_BASE2"),
        ({"DIRECT_MEMORY": 1}, "DIRECT_MEMORY"),
        ({"DIRECT_MEMORY": True}, "DIRECT_MEMORY requires"),
        (
            {
                "DIRECT_MEMORY": True,
                "QUERY_TILE": 32,
                "KEY_TILE": 16,
                "SHARED_PADDING": 4,
                "UNROLL_MMA": True,
            },
            "DIRECT_MEMORY requires",
        ),
        ({"DIRECT_MEMORY": True, "QUERY_TILE": 32, "KEY_TILE": 16}, "DIRECT_MEMORY requires"),
        ({"STORAGE_DTYPE": "bf16"}, "STORAGE_DTYPE"),
    ],
)
def test_matrix_attention_rejects_unsupported_shapes(overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(**overrides)


def _reference(queries, cache, prefix, valid_rows):
    result = np.zeros_like(queries)
    grouping = queries.shape[1] // cache.shape[2]
    for row in range(valid_rows):
        for head in range(queries.shape[1]):
            keys = cache[0, 1, head // grouping, : prefix + row + 1].astype(np.float64)
            values = cache[1, 1, head // grouping, : prefix + row + 1].astype(np.float64)
            scores = keys @ queries[row, head].astype(np.float64) / np.sqrt(queries.shape[-1])
            probabilities = np.exp(scores - scores.max())
            result[row, head] = (probabilities / probabilities.sum()) @ values
    return result


def _prepare(
    queries,
    cache,
    control,
    output,
    *,
    chunk,
    query_heads,
    kv_heads,
    dimension,
    capacity,
    storage,
    query_tile=8,
    key_tile=32,
    shared_padding=4,
    unroll=False,
    softmax_lanes=32,
    transpose_keys=False,
    register_stats=False,
    unroll_softmax=False,
    load_vector=1,
    softmax_base2=False,
    direct_memory=False,
):
    return qwen3_prefill_matrix_attention[(metile.cdiv(chunk, query_tile), query_heads)].prepare(
        queries,
        cache,
        control,
        output,
        1,
        CHUNK=chunk,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=dimension,
        LAYERS=2,
        MAX_CONTEXT=capacity,
        BLOCK=128,
        QUERY_TILE=query_tile,
        KEY_TILE=key_tile,
        SHARED_PADDING=shared_padding,
        UNROLL_MMA=unroll,
        SOFTMAX_LANES=softmax_lanes,
        TRANSPOSE_KEYS=transpose_keys,
        REGISTER_STATS=register_stats,
        UNROLL_SOFTMAX=unroll_softmax,
        LOAD_VECTOR=load_vector,
        SOFTMAX_BASE2=softmax_base2,
        DIRECT_MEMORY=direct_memory,
        STORAGE_DTYPE=storage,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
        STRICT_MATH=True,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dimension", [32, 128])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("query_heads,kv_heads", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize("query_tile,key_tile", [(8, 32), (32, 16)])
def test_gpu_matrix_attention_matches_causal_heads_prefix_padding_and_control_reuse(
    dimension, dtype, query_heads, kv_heads, query_tile, key_tile
):
    generator = np.random.default_rng(7741)
    query_values = generator.normal(0, 0.5, (11, query_heads, dimension)).astype(dtype)
    cache_values = generator.normal(0, 0.5, (2, 2, kv_heads, 45, dimension)).astype(dtype)
    prefix, valid_rows = 29, 9
    query_values[valid_rows:] = np.nan
    cache_values[:, :, :, prefix + valid_rows :] = np.nan
    expected = _reference(query_values, cache_values, prefix, valid_rows)
    queries = metile.Buffer(data=query_values)
    cache = metile.Buffer(data=cache_values)
    control = metile.Buffer(data=np.array([prefix, valid_rows], dtype=np.int32))
    output = metile.Buffer.empty(query_values.shape, dtype=dtype)
    dispatch = _prepare(
        queries,
        cache,
        control,
        output,
        chunk=11,
        query_heads=query_heads,
        kv_heads=kv_heads,
        dimension=dimension,
        capacity=45,
        storage="f16" if dtype == np.float16 else "f32",
        query_tile=query_tile,
        key_tile=key_tile,
    )
    tolerance = 0.002 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(output.numpy(), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(output.numpy()[valid_rows:], 0)
    np.testing.assert_array_equal(cache.numpy(), cache_values)
    np.testing.assert_array_equal(queries.numpy(), query_values)
    control.numpy()[:] = (0, 0)
    dispatch()
    np.testing.assert_array_equal(output.numpy(), 0)
    control.numpy()[:] = (prefix, 1)
    dispatch()
    np.testing.assert_allclose(
        output.numpy(),
        _reference(query_values, cache_values, prefix, 1),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("prefix,valid_rows", [(0, 1), (0, 7), (63, 7), (4091, 5)])
@pytest.mark.parametrize("query_tile,key_tile", [(8, 32), (32, 16)])
def test_gpu_matrix_attention_handles_first_token_and_long_context(
    prefix, valid_rows, query_tile, key_tile
):
    generator = np.random.default_rng(1418)
    capacity = prefix + valid_rows + 3
    queries = generator.normal(0, 0.3, (7, 16, 128)).astype(np.float32)
    cache = generator.normal(0, 0.3, (2, 2, 8, capacity, 128)).astype(np.float32)
    queries[valid_rows:] = np.nan
    cache[:, :, :, prefix + valid_rows :] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([prefix, valid_rows], dtype=np.int32),
        output,
        chunk=7,
        query_heads=16,
        kv_heads=8,
        dimension=128,
        capacity=capacity,
        storage="f32",
        query_tile=query_tile,
        key_tile=key_tile,
    )
    np.testing.assert_allclose(
        output, _reference(queries, cache, prefix, valid_rows), rtol=2e-5, atol=2e-5
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
def test_gpu_matrix_attention_preserves_fp32_values_below_half_precision():
    queries = np.ones((8, 2, 128), dtype=np.float32)
    cache = np.ones((2, 2, 1, 33, 128), dtype=np.float32)
    cache[1] += np.float32(2**-12)
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([25, 8], dtype=np.int32),
        output,
        chunk=8,
        query_heads=2,
        kv_heads=1,
        dimension=128,
        capacity=33,
        storage="f32",
    )
    np.testing.assert_allclose(output, 1 + 2**-12, rtol=0, atol=2e-6)
    assert np.min(output) > 1


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("query_tile,key_tile", [(8, 32), (32, 16)])
@pytest.mark.parametrize("padding", [0, 4])
@pytest.mark.parametrize("unroll", [False, True])
def test_gpu_matrix_attention_tuning_layouts_agree_across_multiple_query_tiles(
    query_tile, key_tile, padding, unroll
):
    generator = np.random.default_rng(5978)
    queries = generator.normal(0, 0.5, (35, 4, 128)).astype(np.float32)
    cache = generator.normal(0, 0.5, (2, 2, 2, 99, 128)).astype(np.float32)
    queries[33:] = np.nan
    cache[:, :, :, 94:] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([61, 33], dtype=np.int32),
        output,
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=128,
        capacity=99,
        storage="f32",
        query_tile=query_tile,
        key_tile=key_tile,
        shared_padding=padding,
        unroll=unroll,
    )
    np.testing.assert_allclose(output, _reference(queries, cache, 61, 33), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("lanes", [8, 16, 32])
@pytest.mark.parametrize("transpose_keys", [False, True])
@pytest.mark.parametrize("padding", [0, 4])
def test_gpu_matrix_attention_subgroups_and_transposed_staging_match_causal_reference(
    lanes, transpose_keys, padding
):
    generator = np.random.default_rng(6199)
    queries = generator.normal(0, 0.5, (35, 4, 128)).astype(np.float32)
    cache = generator.normal(0, 0.5, (2, 2, 2, 133, 128)).astype(np.float32)
    queries[33:] = np.nan
    cache[:, :, :, 130:] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([97, 33], dtype=np.int32),
        output,
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=128,
        capacity=133,
        storage="f32",
        query_tile=32,
        key_tile=16,
        shared_padding=padding,
        unroll=True,
        softmax_lanes=lanes,
        transpose_keys=transpose_keys,
    )
    np.testing.assert_allclose(output, _reference(queries, cache, 97, 33), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("register_stats", [False, True])
@pytest.mark.parametrize("vector", [1, 4, 16])
def test_matrix_attention_register_statistics_and_staging_have_explicit_memory_contracts(
    register_stats, vector
):
    function = _trace(
        QUERY_TILE=32,
        KEY_TILE=16,
        SOFTMAX_LANES=8,
        TRANSPOSE_KEYS=True,
        UNROLL_MMA=True,
        REGISTER_STATS=register_stats,
        UNROLL_SOFTMAX=True,
        LOAD_VECTOR=vector,
    )
    metal = lower(function)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(operation.size * 4 for operation in allocations) == 19712 - (
        256 if register_stats else 0
    )
    assert sum(isinstance(operation, tir.Barrier) for operation in _walk(function.ops)) == 8


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dimension", [32, 128])
@pytest.mark.parametrize("register_stats", [False, True])
@pytest.mark.parametrize("vector", [1, 4, 16])
def test_gpu_matrix_attention_register_statistics_and_grouped_loads_preserve_causality(
    dimension, register_stats, vector
):
    generator = np.random.default_rng(9265)
    queries = generator.normal(0, 0.5, (35, 4, dimension)).astype(np.float32)
    cache = generator.normal(0, 0.5, (2, 2, 2, 133, dimension)).astype(np.float32)
    queries[33:] = np.nan
    cache[:, :, :, 130:] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([97, 33], dtype=np.int32),
        output,
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=dimension,
        capacity=133,
        storage="f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        softmax_lanes=8,
        transpose_keys=True,
        register_stats=register_stats,
        unroll_softmax=True,
        load_vector=vector,
    )
    np.testing.assert_allclose(output, _reference(queries, cache, 97, 33), rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("lanes", [8, 16])
@pytest.mark.parametrize("register_stats", [False, True])
def test_gpu_matrix_attention_adjacent_subgroup_tree_is_bitwise_equal_to_full_simd_sum(
    lanes, register_stats
):
    generator = np.random.default_rng(9611)
    queries = generator.normal(0, 3, (35, 4, 128)).astype(np.float32)
    cache = generator.normal(size=(2, 2, 2, 563, 128)).astype(np.float32)
    queries[33:] = np.nan
    cache[:, :, :, 546:] = np.nan
    control = np.array([513, 33], dtype=np.int32)
    reference = np.empty_like(queries)
    output = np.empty_like(queries)
    constants = dict(
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=128,
        capacity=563,
        storage="f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        transpose_keys=True,
    )
    _prepare(queries, cache, control, reference, **constants)
    _prepare(
        queries,
        cache,
        control,
        output,
        softmax_lanes=lanes,
        register_stats=register_stats,
        unroll_softmax=True,
        load_vector=16,
        **constants,
    )
    np.testing.assert_array_equal(output.view(np.uint32), reference.view(np.uint32))


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("dimension", [32, 128])
@pytest.mark.parametrize("register_stats", [False, True])
def test_gpu_matrix_attention_cached_queries_preserve_bits_and_control_reuse(
    dtype, dimension, register_stats
):
    generator = np.random.default_rng(73229)
    query_data = generator.normal(0, 1.5, (35, 4, dimension)).astype(dtype)
    cache_data = generator.normal(0, 1.5, (2, 2, 2, 133, dimension)).astype(dtype)
    query_data[33:] = np.nan
    cache_data[:, :, :, 130:] = np.nan
    queries = metile.Buffer(data=query_data)
    cache = metile.Buffer(data=cache_data)
    control = metile.Buffer(data=np.array([97, 33], dtype=np.int32))
    reference = metile.Buffer.empty(query_data.shape, dtype=dtype)
    output = metile.Buffer.empty(query_data.shape, dtype=dtype)
    constants = dict(
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=dimension,
        capacity=133,
        storage="f16" if dtype == np.float16 else "f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        softmax_lanes=8,
        transpose_keys=True,
        register_stats=register_stats,
        unroll_softmax=True,
        load_vector=16,
        softmax_base2=True,
    )
    uncached = _prepare(queries, cache, control, reference, unroll=False, **constants)
    cached = _prepare(queries, cache, control, output, unroll=True, **constants)
    bits = np.uint16 if dtype == np.float16 else np.uint32
    for valid_rows in (33, 0, 1, 33):
        control.numpy()[:] = (97, valid_rows)
        uncached()
        cached()
        np.testing.assert_array_equal(output.numpy().view(bits), reference.numpy().view(bits))
        np.testing.assert_array_equal(output.numpy()[valid_rows:], 0)
    np.testing.assert_array_equal(queries.numpy(), query_data)
    np.testing.assert_array_equal(cache.numpy(), cache_data)


@pytest.mark.parametrize("base2", [False, True])
def test_matrix_attention_softmax_base_and_scaling_order_are_explicit(base2):
    function = _trace(
        QUERY_TILE=32,
        KEY_TILE=16,
        SOFTMAX_LANES=8,
        TRANSPOSE_KEYS=True,
        REGISTER_STATS=True,
        UNROLL_MMA=True,
        SOFTMAX_BASE2=base2,
    )
    metal = lower(function)
    source = emit(metal)
    assert ("fast::exp2(" in source) == base2
    assert ("fma(" in source) == base2
    assert ("exp(" in source) == (not base2)
    assert ("1.4426950408889634f" in source) == base2
    assert (
        any(
            isinstance(operation, mir.MFragmentElementwise) and operation.operation == "div"
            for operation in _walk(metal.ops)
        )
        == base2
    )
    assert "half" not in source


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("dimension", [32, 128])
@pytest.mark.parametrize("prefix,valid_rows", [(0, 1), (61, 33), (4090, 29)])
def test_gpu_matrix_attention_direct_memory_matches_shared_bits_and_masks(
    dtype, dimension, prefix, valid_rows, base2=True, register_stats=True
):
    generator = np.random.default_rng(67182)
    capacity = prefix + valid_rows + 3
    query_data = generator.normal(0, 1.5, (35, 4, dimension)).astype(dtype)
    cache_data = generator.normal(0, 1.5, (2, 2, 2, capacity, dimension)).astype(dtype)
    query_data[valid_rows:] = np.nan
    cache_data[:, :, :, prefix + valid_rows :] = np.nan
    queries = metile.Buffer(data=query_data)
    cache = metile.Buffer(data=cache_data)
    control = metile.Buffer(data=np.array([prefix, valid_rows], dtype=np.int32))
    reference = metile.Buffer.empty(query_data.shape, dtype=dtype)
    output = metile.Buffer.empty(query_data.shape, dtype=dtype)
    constants = dict(
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=dimension,
        capacity=capacity,
        storage="f16" if dtype == np.float16 else "f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        softmax_lanes=8,
        transpose_keys=True,
        register_stats=register_stats,
        unroll_softmax=True,
        load_vector=16,
        softmax_base2=base2,
    )
    shared = _prepare(queries, cache, control, reference, **constants)
    direct = _prepare(queries, cache, control, output, direct_memory=True, **constants)
    bits = np.uint16 if dtype == np.float16 else np.uint32
    for active_rows in (valid_rows, 0, 1, valid_rows):
        control.numpy()[:] = (prefix, active_rows)
        shared()
        direct()
        np.testing.assert_array_equal(output.numpy().view(bits), reference.numpy().view(bits))
        np.testing.assert_array_equal(output.numpy()[active_rows:], 0)
    np.testing.assert_array_equal(queries.numpy(), query_data)
    np.testing.assert_array_equal(cache.numpy(), cache_data)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("base2,register_stats", [(False, False), (False, True), (True, False)])
def test_gpu_matrix_attention_direct_memory_preserves_alternate_softmax_state(
    base2, register_stats
):
    test_gpu_matrix_attention_direct_memory_matches_shared_bits_and_masks(
        np.float32, 128, 61, 33, base2=base2, register_stats=register_stats
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
def test_gpu_matrix_attention_query_reuse_supports_largest_shared_dimension():
    generator = np.random.default_rng(37288)
    queries = generator.normal(0, 0.5, (35, 4, 224)).astype(np.float32)
    cache = generator.normal(0, 0.5, (2, 2, 2, 133, 224)).astype(np.float32)
    queries[33:] = np.nan
    cache[:, :, :, 130:] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([97, 33], dtype=np.int32),
        output,
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=224,
        capacity=133,
        storage="f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        softmax_lanes=8,
        transpose_keys=True,
        register_stats=True,
        unroll_softmax=True,
        load_vector=16,
        softmax_base2=True,
    )
    np.testing.assert_allclose(output, _reference(queries, cache, 97, 33), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(output[33:], 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dimension", [32, 128])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("lanes", [8, 32])
def test_gpu_matrix_attention_base2_post_dot_scale_matches_causal_reference(
    dimension, dtype, lanes
):
    generator = np.random.default_rng(6832)
    queries = generator.normal(0, 1, (35, 4, dimension)).astype(dtype)
    cache = generator.normal(0, 1, (2, 2, 2, 563, dimension)).astype(dtype)
    queries[33:] = np.nan
    cache[:, :, :, 546:] = np.nan
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([513, 33], dtype=np.int32),
        output,
        chunk=35,
        query_heads=4,
        kv_heads=2,
        dimension=dimension,
        capacity=563,
        storage="f16" if dtype == np.float16 else "f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        softmax_lanes=lanes,
        transpose_keys=True,
        register_stats=True,
        unroll_softmax=True,
        load_vector=16,
        softmax_base2=True,
    )
    tolerance = 0.002 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(
        output, _reference(queries, cache, 513, 33), rtol=tolerance, atol=tolerance
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("prefix", [0, 513])
@pytest.mark.parametrize("direct_memory", [False, True])
def test_gpu_matrix_attention_explicit_arithmetic_matches_native_sdpa_bitwise(
    prefix, direct_memory
):
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.skip("strict FP32 reference requires MLX_ENABLE_TF32=0 before importing MLX")
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available():
        pytest.skip("requires MLX Metal support")
    generator = np.random.default_rng(81753)
    valid_rows, chunk, query_heads, kv_heads, dimension = 65, 69, 4, 2, 128
    valid_keys = prefix + valid_rows
    capacity = valid_keys + 7
    queries = generator.normal(0, 1.5, (chunk, query_heads, dimension)).astype(np.float32)
    cache = generator.normal(0, 1.5, (2, 2, kv_heads, capacity, dimension)).astype(np.float32)
    queries[valid_rows:] = np.nan
    cache[:, :, :, valid_keys:] = np.nan
    native = core.fast.scaled_dot_product_attention(
        core.array(queries[:valid_rows].transpose(1, 0, 2)[None]),
        core.array(cache[0, 1, :, :valid_keys][None]),
        core.array(cache[1, 1, :, :valid_keys][None]),
        scale=dimension**-0.5,
        mask="causal",
    )
    core.eval(native)
    expected = np.ascontiguousarray(np.asarray(native)[0].transpose(1, 0, 2))
    output = np.empty_like(queries)
    _prepare(
        queries,
        cache,
        np.array([prefix, valid_rows], dtype=np.int32),
        output,
        chunk=chunk,
        query_heads=query_heads,
        kv_heads=kv_heads,
        dimension=dimension,
        capacity=capacity,
        storage="f32",
        query_tile=32,
        key_tile=16,
        shared_padding=0,
        unroll=True,
        softmax_lanes=8,
        transpose_keys=True,
        register_stats=True,
        unroll_softmax=True,
        load_vector=16,
        softmax_base2=True,
        direct_memory=direct_memory,
    )
    np.testing.assert_array_equal(output[:valid_rows].view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(output[valid_rows:], 0.0)
