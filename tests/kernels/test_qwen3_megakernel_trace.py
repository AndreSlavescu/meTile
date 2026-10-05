import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels import qwen3_decode_megakernel, qwen3_layer_offsets


def _trace(**overrides):
    geometry = dict(
        HIDDEN=32,
        INTERMEDIATE=64,
        QUERY_HEADS=2,
        KV_HEADS=1,
        HEAD_DIM=32,
        LAYERS=2,
        VOCAB=67,
        MAX_CONTEXT=8,
        BLOCK=32,
        STORAGE_DTYPE="f16",
    )
    geometry.update(overrides)
    storage = geometry["STORAGE_DTYPE"]
    pointer_type = storage if storage in ("f16", "f32") else "f16"
    parameters = [
        ("Embedding", PtrType(pointer_type)),
        ("LayerWeights", PtrType(pointer_type)),
        ("FinalNorm", PtrType(pointer_type)),
        ("Rotary", PtrType("f32")),
        ("KVCache", PtrType(pointer_type)),
        ("Logits", PtrType(pointer_type)),
        ("token", I32),
        ("position", I32),
    ]
    with TracingContext("qwen3_megakernel") as context:
        context.func.constexprs.update(geometry)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        qwen3_decode_megakernel.fn(*proxies, **geometry)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("block", [32, 64, 128, 256])
@pytest.mark.parametrize("dimension", [32, 64, 128])
@pytest.mark.parametrize("storage", ["f16", "f32"])
def test_qwen3_is_one_dsl_kernel_with_runtime_layers(block, dimension, storage):
    function = _trace(BLOCK=block, HEAD_DIM=dimension, STORAGE_DTYPE=storage)
    metal = lower(function)
    source = emit(metal)
    assert source.count("[[kernel") == 1
    assert metal.threadgroup_size == (block, 1, 1)
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "mem_device" not in source
    assert "atomic_" not in source
    if storage == "f32":
        assert not any(
            isinstance(operation, tir.Cast) and operation.dtype == "f16"
            for operation in _walk(function.ops)
        )
        assert "half" not in source
    assert not any(
        isinstance(operation, (tir.Dot, tir.PersistentRange)) for operation in _walk(function.ops)
    )
    layer_loops = [
        operation
        for operation in function.ops
        if isinstance(operation, tir.ForRange)
        and isinstance(operation.end.defining_op, tir.Constant)
        and operation.end.defining_op.value == 2
    ]
    assert len(layer_loops) == 1
    assert any(
        isinstance(operation, tir.AssignLoopState) for operation in _walk(layer_loops[0].body)
    )


def test_qwen3_official_small_geometry_uses_24kib_shared_memory():
    function = _trace(
        HIDDEN=1024,
        INTERMEDIATE=3072,
        QUERY_HEADS=16,
        KV_HEADS=8,
        HEAD_DIM=128,
        LAYERS=28,
        VOCAB=151936,
        MAX_CONTEXT=128,
        BLOCK=256,
    )
    metal = lower(function)
    allocations = [
        operation for operation in _walk(metal.ops) if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert len(allocations) == 1
    assert allocations[0].size * 4 == 24576
    assert len(function.tensors) == 32
    assert "[[kernel" in emit(metal)


def test_qwen3_current_cache_is_written_only_after_all_cache_reads():
    function = _trace()
    operations = list(_walk(function.ops))
    cache_reads = [
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, tir.Load)
        and operation.tensor is not None
        and operation.tensor.ptr.name == "KVCache"
    ]
    cache_writes = [
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, tir.Store)
        and operation.tensor is not None
        and operation.tensor.ptr.name == "KVCache"
    ]
    assert len(cache_reads) == 2
    assert len(cache_writes) == 2
    assert max(cache_reads) < min(cache_writes)
    assert any(
        isinstance(operation, tir.Barrier)
        for operation in operations[max(cache_reads) : min(cache_writes)]
    )


def test_qwen3_layer_offsets_cover_disjoint_packed_weights():
    offsets = qwen3_layer_offsets(32, 64, 2, 1, 32)
    sizes = (32, 2048, 1024, 1024, 2048, 32, 2048, 2048, 2048, 32, 32)
    names = (
        "input_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "post_norm",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_norm",
        "k_norm",
        "layer_size",
    )
    assert tuple(offsets) == names
    assert offsets["input_norm"] == 0
    for index, size in enumerate(sizes):
        assert offsets[names[index + 1]] - offsets[names[index]] == size
    assert offsets["layer_size"] == sum(sizes)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"HIDDEN": 31}, "multiples"),
        ({"HEAD_DIM": 16}, "multiples"),
        ({"QUERY_HEADS": 3, "KV_HEADS": 2}, "divisible"),
        ({"BLOCK": 31}, "BLOCK"),
        ({"LAYERS": 0}, "positive"),
        ({"EPS": float("nan")}, "finite"),
        ({"EPS": float("inf")}, "finite"),
        ({"EPS": True}, "finite"),
        ({"INTERMEDIATE": 8192}, "32 KiB"),
        ({"STORAGE_DTYPE": "bf16"}, "STORAGE_DTYPE"),
    ],
)
def test_qwen3_rejects_unsupported_geometry(overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(**overrides)
