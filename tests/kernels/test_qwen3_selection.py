import inspect
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
from metile.ir.types import PtrType
from metile_kernels.megakernels.qwen3_selection import (
    qwen3_argmax_finalize,
    qwen3_argmax_partials,
)


def _trace(stage, **overrides):
    geometry = dict(VOCAB=67, CHUNKS=3, BLOCK=32, STORAGE_DTYPE="f32")
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
        if name in ("PartialIndices", "Output"):
            dtype = PtrType("i32")
        elif name == "Logits":
            dtype = PtrType(storage if storage in ("f16", "f32") else "f32")
        else:
            dtype = PtrType("f32")
        parameters.append((name, dtype))
    with TracingContext(stage.name) as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        stage.fn(*proxies, **constants)
    _mark_outputs(context.func)
    return context.func


@pytest.mark.parametrize("stage", [qwen3_argmax_partials, qwen3_argmax_finalize])
@pytest.mark.parametrize("block", [32, 96, 256, 1024])
@pytest.mark.parametrize("storage", ["f16", "f32"])
def test_selection_lowers_to_dsl_simd_reductions_and_bounded_shared_memory(stage, block, storage):
    function = _trace(stage, BLOCK=block, STORAGE_DTYPE=storage)
    metal = lower(function)
    source = emit(metal)
    assert metal.threadgroup_size == (block, 1, 1)
    assert "simd_max(" in source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "atomic_" not in source
    assert "mem_device" not in source
    assert not any(isinstance(operation, mir.MThreadgroupReduce) for operation in metal.ops)
    allocated = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert len(allocated) == 2
    assert sum(operation.size for operation in allocated) == 2 * block // 32
    outputs = {parameter.name for parameter in function.params if parameter.is_output}
    assert outputs == (
        {"PartialValues", "PartialIndices"} if stage is qwen3_argmax_partials else {"Output"}
    )


@pytest.mark.parametrize("stage", [qwen3_argmax_partials, qwen3_argmax_finalize])
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"VOCAB": 0}, "VOCAB"),
        ({"VOCAB": True}, "VOCAB"),
        ({"VOCAB": 2**31}, "VOCAB"),
        ({"CHUNKS": 0}, "CHUNKS"),
        ({"CHUNKS": False}, "CHUNKS"),
        ({"CHUNKS": 2**31}, "CHUNKS"),
        ({"BLOCK": 31}, "BLOCK"),
        ({"BLOCK": 33}, "BLOCK"),
        ({"BLOCK": True}, "BLOCK"),
        ({"BLOCK": 1056}, "BLOCK"),
    ],
)
def test_invalid_selection_geometry_fails_before_lowering(stage, overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(stage, **overrides)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"CHUNKS": 2}, "cover VOCAB"),
        ({"CHUNKS": 2**30}, "32-bit addressing"),
        ({"STORAGE_DTYPE": "bf16"}, "STORAGE_DTYPE"),
    ],
)
def test_partial_stage_rejects_incomplete_or_unrepresentable_launches(overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(qwen3_argmax_partials, **overrides)


@pytest.fixture
def metal_device():
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    from metile.runtime.metal_device import MetalDevice

    return MetalDevice.get()


def _prepare(values, block, extra_chunks=0, final_block=None):
    vocabulary = values.size
    chunks = metile.cdiv(vocabulary, block) + extra_chunks
    logits = metile.Buffer(data=values)
    partial_values = metile.Buffer.empty((chunks,), dtype=np.float32)
    partial_indices = metile.Buffer.empty((chunks,), dtype=np.int32)
    output = metile.Buffer(data=np.full(3, -17, dtype=np.int32))
    partials = qwen3_argmax_partials[(chunks,)].prepare(
        logits,
        partial_values,
        partial_indices,
        VOCAB=vocabulary,
        CHUNKS=chunks,
        BLOCK=block,
        STORAGE_DTYPE="f16" if values.dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    finalize = qwen3_argmax_finalize[(1,)].prepare(
        partial_values,
        partial_indices,
        output,
        VOCAB=vocabulary,
        CHUNKS=chunks,
        BLOCK=block if final_block is None else final_block,
        STRICT_MATH=True,
    )
    return logits, partial_values, partial_indices, output, partials, finalize


@pytest.mark.parametrize("block", [32, 96, 256, 1024])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_selection_matches_numpy_for_negative_logits_ties_and_padded_chunks(
    metal_device, block, dtype
):
    values = -np.arange(1, block * 3 + 18, dtype=np.float32).astype(dtype)
    values[block - 1] = values[block + 3] = values[-1] = -0.5
    logits, partial_values, partial_indices, output, partials, finalize = _prepare(
        values, block, extra_chunks=2, final_block=32
    )
    assert output.numpy().tolist() == [int(np.argmax(values)), -17, -17]
    for chunk in range(metile.cdiv(values.size, block)):
        piece = values[chunk * block : (chunk + 1) * block]
        assert partial_values.numpy()[chunk] == np.max(piece)
        assert partial_indices.numpy()[chunk] == chunk * block + np.argmax(piece)
    assert np.isneginf(partial_values.numpy()[-2:]).all()
    assert (partial_indices.numpy()[-2:] == values.size).all()
    logits.numpy()[-1] = 9
    partials()
    finalize()
    assert output.numpy().tolist() == [values.size - 1, -17, -17]


@pytest.mark.parametrize("value", [-float("inf"), -2.0, 0.0, float("inf")])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_selection_handles_one_element_and_nonfinite_extrema(metal_device, value, dtype):
    values = np.array([value], dtype=dtype)
    *_buffers, output, partials, finalize = _prepare(values, 128, extra_chunks=2)
    assert output.numpy().tolist() == [0, -17, -17]
    partials()
    finalize()
    assert output.numpy().tolist() == [0, -17, -17]


@pytest.mark.parametrize("value", [-float("inf"), -3.0, float("inf")])
def test_gpu_selection_breaks_global_ties_by_lowest_token_id(metal_device, value):
    values = np.full(1031, value, dtype=np.float32)
    output = _prepare(values, 128, extra_chunks=1)[3]
    assert output.numpy()[0] == 0


def test_gpu_final_stage_loops_across_more_chunks_than_threads(metal_device):
    values = np.full(32 * 70 + 3, -9, dtype=np.float32)
    values[32 * 68 + 7] = values[-1] = -1
    output = _prepare(values, 32, extra_chunks=2, final_block=32)[3]
    assert output.numpy()[0] == 32 * 68 + 7
