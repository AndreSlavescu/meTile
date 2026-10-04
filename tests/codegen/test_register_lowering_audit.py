import operator

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.ownership import validate_register_reductions
from metile.compiler.passes import fold_constants
from metile.compiler.scheduling import reorder_for_latency
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.rmsnorm import rmsnorm_register

_DTYPES = {"i32": np.int32, "u32": np.uint32, "f16": np.float16, "f32": np.float32, "bool": bool}
_BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "mod": operator.mod,
    "bitand": operator.and_,
    "bitor": operator.or_,
    "bitxor": operator.xor,
    "shl": operator.lshift,
    "shr": operator.rshift,
    "min": np.minimum,
    "max": np.maximum,
}
_COMPARE = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


def _interpret(function, arguments, *, group=0):
    """Evaluate ordinary scalar MIR across physical threads without executing Metal."""
    environment = dict(arguments)
    thread = np.arange(function.threadgroup_size[0], dtype=np.uint32)

    def value(operand):
        operand = mir.resolve(operand)
        if isinstance(operand.defining_op, mir.MConstant):
            return np.asarray(operand.defining_op.value, dtype=_DTYPES[operand.type.dtype])
        return environment[operand.name]

    for operation in function.ops:
        if isinstance(operation, mir.MConstant):
            result = operation.value
        elif isinstance(operation, mir.ThreadPositionInThreadgroup):
            result = thread
        elif isinstance(operation, mir.ThreadgroupPositionInGrid):
            result = group if operation.axis == 0 else 0
        elif isinstance(operation, mir.MSimdgroupId):
            result = thread // 32
        elif isinstance(operation, mir.MThreadInSimdgroup):
            result = thread % 32
        elif isinstance(operation, mir.MCast):
            result = value(operation.value)
        elif isinstance(operation, mir.MThreadIndexMap):
            packed = value(operation.thread)
            result = np.zeros_like(packed)
            for logical_bit, physical_bit in enumerate(operation.layout.bit_order):
                result |= ((packed >> physical_bit) & 1) << logical_bit
            result ^= operation.layout.xor_mask
        elif isinstance(operation, mir.MBinOp):
            result = _BINARY[operation.op](value(operation.lhs), value(operation.rhs))
        elif isinstance(operation, mir.MCompare):
            result = _COMPARE[operation.predicate](value(operation.lhs), value(operation.rhs))
        elif isinstance(operation, mir.MSelect):
            result = np.where(
                value(operation.condition), value(operation.true_val), value(operation.false_val)
            )
        elif isinstance(operation, mir.MUnary):
            result = {"sqrt": np.sqrt, "abs": np.abs, "neg": operator.neg}[operation.op](
                value(operation.operand)
            )
        elif isinstance(operation, mir.DeviceLoad):
            indices = value(operation.index)
            mask = value(operation.mask) if operation.mask is not None else True
            fill = value(operation.other) if operation.other is not None else 0
            result = np.where(mask, value(operation.ptr)[np.where(mask, indices, 0)], fill)
        elif isinstance(operation, mir.DeviceStore):
            indices, stored, mask = np.broadcast_arrays(
                value(operation.index),
                value(operation.value),
                value(operation.mask) if operation.mask is not None else True,
            )
            value(operation.ptr)[indices[mask]] = stored[mask]
            continue
        elif isinstance(operation, mir.MThreadgroupReduce):
            assert operation.reduce_op == "sum"
            result = np.sum(
                np.broadcast_to(value(operation.operand), thread.shape), dtype=np.float32
            )
        elif isinstance(operation, mir.MThreadgroupAlloc):
            continue
        else:
            raise AssertionError(f"Unexpected operation in scalarized program: {operation}")
        environment[operation.result.name] = np.asarray(
            result, dtype=_DTYPES[operation.result.type.dtype]
        )


def _trace(body, *, dtype="f32", names=("source", "output")):
    with TracingContext("register_audit") as context:
        source, output = names
        context.func.params = [
            tir.Param(source, PtrType(dtype)),
            tir.Param(output, PtrType(dtype), is_output=True),
            tir.Param("width", I32),
        ]
        arguments = [
            TracingProxy(tir.Value(parameter.name, parameter.type))
            for parameter in context.func.params
        ]
        body(*arguments)
    return context.func


def _checked_lower(function, *, optimized):
    function.constexprs.setdefault("SCHEDULE", metile.Schedule(vector_width=1))
    original = repr(function)
    lowered = lower(function)
    assert repr(function) == original
    if optimized:
        lowered = reorder_for_latency(fold_constants(lowered))
    validate_register_reductions(lowered)
    emit(lowered)
    names = [operation.result.name for operation in lowered.ops if operation.result is not None]
    assert len(names) == len(set(names))
    return lowered


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("permuted", [False, True])
def test_multiple_register_reductions_broadcast_uniform_results_without_losing_live_values(
    optimized, dtype, permuted
):
    layout = metile.ThreadLayout.identity(256, elements_per_thread=4)
    if permuted:
        layout = metile.ThreadLayout(tuple(reversed(layout.bit_order)), 37, elements_per_thread=4)

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, 256, layout=layout)
        values = metile.cast(inputs.load((indices - 1,), other=-2), "f32")
        total = metile.sum(values)
        centered = values - total / 256
        square_sum = metile.sum(centered * centered)
        scaled = centered / metile.sqrt(square_sum / 256 + 1e-5)
        result = metile.where(square_sum > 0, scaled, 0.0) + values * 0.25
        outputs.store((indices,), metile.convert_layout(result, layout))

    lowered = _checked_lower(_trace(body, dtype=dtype), optimized=optimized)
    source = np.linspace(-3, 4, 237).astype(_DTYPES[dtype])
    output = np.full_like(source, np.nan)
    _interpret(lowered, {"source": source, "output": output, "width": source.size})
    padded = np.full(256, -2, dtype=np.float32)
    padded[1 : source.size + 1] = source.astype(np.float32)
    centered = padded - np.sum(padded, dtype=np.float32) / 256
    expected = centered / np.sqrt(np.sum(centered * centered, dtype=np.float32) / 256 + 1e-5)
    expected += padded * 0.25
    np.testing.assert_allclose(
        output, expected[: source.size].astype(source.dtype), rtol=2e-3, atol=1e-5
    )
    assert len(lowered.register_reductions) == 2
    assert len({record.scratch for record in lowered.register_reductions}) == 2
    assert sum(isinstance(operation, mir.DeviceLoad) for operation in lowered.ops) == 4


@pytest.mark.parametrize("optimized", [False, True])
def test_each_register_memory_access_keeps_its_own_mask_and_fill(optimized):
    layout = metile.ThreadLayout((5, 6, 0, 1, 2, 3, 4), elements_per_thread=4)

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, 128, layout=layout)
        previous = inputs.load((indices - 1,), other=-9)
        upcoming = inputs.load((indices + 3,), other=13)
        outputs.store((indices,), previous + upcoming)

    lowered = _checked_lower(_trace(body), optimized=optimized)
    source = np.arange(113, dtype=np.float32)
    output = np.full_like(source, np.nan)
    _interpret(lowered, {"source": source, "output": output, "width": source.size})
    expected = np.r_[-9, source[:-1]] + np.r_[source[3:], 13, 13, 13]
    np.testing.assert_array_equal(output, expected)
    assert sum(isinstance(operation, mir.DeviceLoad) for operation in lowered.ops) == 8
    assert sum(isinstance(operation, mir.DeviceStore) for operation in lowered.ops) == 4


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_register_rmsnorm_preserves_uniform_row_origins_and_original_input_registers(dtype):
    width = 1009
    with TracingContext("register_rmsnorm") as context:
        context.func.params = [
            tir.Param("source", PtrType(dtype)),
            tir.Param("weight", PtrType(dtype)),
            tir.Param("output", PtrType(dtype), is_output=True),
        ]
        arguments = [
            TracingProxy(tir.Value(parameter.name, parameter.type))
            for parameter in context.func.params
        ]
        rmsnorm_register.fn(*arguments, 1e-5, N=width, BLOCK=1024)
    lowered = _checked_lower(context.func, optimized=True)
    source = np.linspace(-2, 3, width * 2).astype(_DTYPES[dtype])
    weight = np.linspace(0.5, 1.5, width).astype(_DTYPES[dtype])
    output = np.full_like(source, np.nan)
    for row in range(2):
        _interpret(lowered, {"source": source, "weight": weight, "output": output}, group=row)
    floating = source.astype(np.float32).reshape(2, width)
    expected = floating / np.sqrt(np.mean(floating * floating, axis=-1, keepdims=True) + 1e-5)
    expected *= weight.astype(np.float32)
    np.testing.assert_allclose(
        output.reshape(2, width), expected.astype(output.dtype), rtol=2e-3, atol=1e-5
    )
    sources = [
        operation.ptr.name for operation in lowered.ops if isinstance(operation, mir.DeviceLoad)
    ]
    assert sources.count("source") == sources.count("weight") == 4


def test_generated_register_names_do_not_alias_user_pointer_parameters():
    layout = metile.ThreadLayout.identity(128, elements_per_thread=4)

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, 128, layout=layout)
        outputs.store((indices,), inputs.load((indices,)))

    names = ("_metile_register_index", "_metile_register_thread")
    lowered = _checked_lower(_trace(body, names=names), optimized=True)
    source = np.arange(123, dtype=np.float32)
    output = np.full_like(source, np.nan)
    _interpret(lowered, {names[0]: source, names[1]: output, "width": source.size})
    np.testing.assert_array_equal(output, source)
    assert not {operation.result.name for operation in lowered.ops if operation.result} & set(names)
