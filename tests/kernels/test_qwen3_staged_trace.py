import inspect

import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels import qwen3_staged

STAGES = (
    qwen3_staged.qwen3_staged_embedding,
    qwen3_staged.qwen3_staged_rmsnorm,
    qwen3_staged.qwen3_staged_qkv,
    qwen3_staged.qwen3_staged_qk_rope,
    qwen3_staged.qwen3_staged_attention,
    qwen3_staged.qwen3_staged_gemv,
    qwen3_staged.qwen3_staged_residual,
    qwen3_staged.qwen3_staged_swiglu,
)


def _trace(stage, **overrides):
    geometry = dict(
        HIDDEN=32,
        INTERMEDIATE=64,
        QUERY_HEADS=2,
        KV_HEADS=1,
        HEAD_DIM=32,
        LAYERS=2,
        VOCAB=67,
        MAX_CONTEXT=8,
        WIDTH=32,
        ROWS=67,
        COLUMNS=32,
        WEIGHT_OFFSET=32,
        WEIGHT_STRIDE=12448,
        EPS=1e-6,
        BLOCK=128,
        STORAGE_DTYPE="f32",
        PACKED_WEIGHTS=False,
    )
    geometry.update(overrides)
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
        elif name == "Control":
            dtype = PtrType("i32")
        elif name == "Rotary":
            dtype = PtrType("f32")
        else:
            dtype = PtrType(storage if storage in ("f16", "f32") else "f32")
        parameters.append((name, dtype))
    with TracingContext(stage.fn.__name__) as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        stage.fn(*proxies, **constants)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.fn.__name__)
@pytest.mark.parametrize("storage", ["f16", "f32"])
@pytest.mark.parametrize("block", [32, 128, 256])
def test_staged_kernels_lower_without_cross_threadgroup_synchronization(stage, storage, block):
    function = _trace(stage, STORAGE_DTYPE=storage, BLOCK=block)
    metal = lower(function)
    source = emit(metal)
    assert source.count("[[kernel") == 1
    assert metal.threadgroup_size == (block, 1, 1)
    assert "atomic_" not in source
    assert "threadgroup_barrier" not in source
    assert "mem_device" not in source
    assert not any(isinstance(operation, mir.MThreadgroupAlloc) for operation in _walk(metal.ops))
    assert not any(
        isinstance(operation, (tir.Dot, tir.PersistentRange)) for operation in _walk(function.ops)
    )
    if storage == "f32":
        assert "half" not in source
    if stage is not qwen3_staged.qwen3_staged_rmsnorm:
        assert "threadgroup_position_in_grid" in source


@pytest.mark.parametrize("dimension", [32, 64, 96, 128])
def test_qk_and_attention_support_partial_simdgroup_head_shapes(dimension):
    for stage in (qwen3_staged.qwen3_staged_qk_rope, qwen3_staged.qwen3_staged_attention):
        source = emit(lower(_trace(stage, HEAD_DIM=dimension)))
        assert "[[kernel" in source


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.fn.__name__)
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"BLOCK": 31}, "BLOCK"),
        ({"BLOCK": 1056}, "BLOCK"),
        ({"STORAGE_DTYPE": "bf16"}, "STORAGE_DTYPE"),
    ],
)
def test_staged_kernels_reject_unsupported_execution_geometry(stage, overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(stage, **overrides)


@pytest.mark.parametrize(
    "stage", [qwen3_staged.qwen3_staged_rmsnorm, qwen3_staged.qwen3_staged_qk_rope]
)
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf"), True])
def test_staged_norms_reject_invalid_epsilon(stage, epsilon):
    with pytest.raises(ValueError, match="EPS"):
        _trace(stage, EPS=epsilon)


def test_qk_stage_only_writes_cache_and_attention_only_reads_cache():
    writer = _trace(qwen3_staged.qwen3_staged_qk_rope)
    reader = _trace(qwen3_staged.qwen3_staged_attention)
    for function, access in ((writer, "write"), (reader, "read")):
        cache = [tensor for tensor in function.tensors if tensor.ptr.name == "KVCache"]
        assert len(cache) == 1
        assert cache[0].access == access
    assert not any(
        isinstance(operation, tir.Load)
        and operation.tensor is not None
        and operation.tensor.ptr.name == "KVCache"
        for operation in _walk(writer.ops)
    )
    assert not any(
        isinstance(operation, tir.Store)
        and operation.tensor is not None
        and operation.tensor.ptr.name == "KVCache"
        for operation in _walk(reader.ops)
    )


@pytest.mark.parametrize(
    "stage",
    [
        qwen3_staged.qwen3_staged_embedding,
        qwen3_staged.qwen3_staged_qk_rope,
        qwen3_staged.qwen3_staged_attention,
    ],
)
def test_mutable_token_and_position_are_device_reads_not_scalar_specializations(stage):
    function = _trace(stage)
    assert not any(parameter.name in ("token", "position") for parameter in function.params)
    assert any(
        isinstance(operation, tir.Load)
        and operation.tensor is not None
        and operation.tensor.ptr.name == "Control"
        for operation in _walk(function.ops)
    )


def test_official_small_model_stages_do_not_allocate_model_sized_shared_scratch():
    for stage in STAGES:
        function = _trace(
            stage,
            HIDDEN=1024,
            INTERMEDIATE=3072,
            QUERY_HEADS=16,
            KV_HEADS=8,
            HEAD_DIM=128,
            LAYERS=28,
            VOCAB=151936,
            MAX_CONTEXT=128,
            WIDTH=1024,
            ROWS=151936,
            COLUMNS=1024,
        )
        metal = lower(function)
        assert "[[kernel" in emit(metal)
        assert not any(
            isinstance(operation, mir.MThreadgroupAlloc) for operation in _walk(metal.ops)
        )
