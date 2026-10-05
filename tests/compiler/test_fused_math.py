import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import fold_constants, split_elementwise_loops, vectorize_elementwise
from metile.compiler.scheduling import reorder_for_latency
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType


def _trace(dtype, *, matrix=False, layout=None, loop=False):
    context = TracingContext("fused_math")
    context.func.params = [
        tir.Param("source", PtrType(dtype)),
        tir.Param("destination", PtrType(dtype), is_output=True),
    ]
    context.func.constexprs.update(BLOCK=32, STRICT_MATH=True)
    if matrix:
        context.func.constexprs["SCHEDULE"] = metile.Schedule(backend="simdgroup_inline")
    with context:
        source = TracingProxy(tir.Value("source", PtrType(dtype)))
        destination = TracingProxy(tir.Value("destination", PtrType(dtype)))
        inputs = metile.tensor(source, shape=(256,), access="read")
        outputs = metile.tensor(destination, shape=(256,), access="write")
        if matrix:
            allocation = metile.shared(64, dtype=dtype)
            values = metile.tensor(allocation, shape=(64,))
            fragments = metile.tensor(allocation, shape=(8, 8), block_shape=(8, 8))
            thread = metile.thread_id()
            for index in metile.tile_range(thread, 64, 32):
                values.store((index,), inputs.load((index,)))
            metile.barrier()
            fragment = fragments.load((0, 0))
            result = metile.fast_exp2(metile.fma(fragment, 0.5, fragment))
            metile.barrier()
            fragments.store((0, 0), result)
            metile.barrier()
            for index in metile.tile_range(thread, 64, 32):
                outputs.store((index,), values.load((index,)))
        elif loop:
            for offset in metile.tile_range(0, 256, 32):
                indices = offset + metile.arange(0, 32)
                value = inputs.load((indices,))
                outputs.store((indices,), metile.fast_exp2(metile.fma(value, 0.5, value)))
        else:
            indices = metile.arange(0, 128, layout=layout)
            value = inputs.load((indices,))
            outputs.store((indices,), metile.fast_exp2(metile.fma(value, 0.5, value)))
    return context.func


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("mode", ["scalarized", "registers", "matrix", "vector_loop"])
def test_explicit_math_survives_supported_lowering_and_optimization_paths(dtype, mode):
    function = _trace(
        dtype,
        matrix=mode == "matrix",
        layout=metile.ThreadLayout.identity(128, elements_per_thread=4)
        if mode == "registers"
        else None,
        loop=mode == "vector_loop",
    )
    metal = lower(function)
    split_elementwise_loops(metal)
    vectorize_elementwise(metal)
    fold_constants(metal)
    reorder_for_latency(metal)
    source = emit(metal)
    assert "fma(" in source
    assert "fast::exp2(" in source
    if mode == "matrix":
        assert ".thread_elements()[0] = fma(" in source
    if mode == "registers":
        assert source.count(" = fma(") == 4
    if mode == "vector_loop":
        assert f"fma({'half' if dtype == 'f16' else 'float'}4(" in source


def test_explicit_fma_constants_are_not_folded_using_separate_python_arithmetic():
    function = mir.MFunction("explicit_fused")
    operands = [
        function.add_op(mir.MConstant(value=value, dtype="f32"), name)
        for name, value in (
            ("left", 1.0 + 2.0**-13),
            ("right", 1.0 - 2.0**-13),
            ("addend", -1.0),
        )
    ]
    operation = mir.MFma(left=operands[0], right=operands[1], addend=operands[2])
    result = function.add_op(operation, "fused")
    fold_constants(function)
    assert result.defining_op is operation
    assert function.ops == [operation]


def test_ordinary_multiply_add_does_not_become_explicit_fma():
    function = mir.MFunction("unfused")
    left, right, addend = [
        mir.MValue(name, ScalarType("f32")) for name in ("left", "right", "addend")
    ]
    product = function.add_op(mir.MBinOp(op="mul", lhs=left, rhs=right), "product")
    function.add_op(mir.MBinOp(op="add", lhs=product, rhs=addend), "sum")
    fold_constants(function)
    reorder_for_latency(function)
    assert [operation.op for operation in function.ops] == ["mul", "add"]
