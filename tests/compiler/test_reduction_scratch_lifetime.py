import re
import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.codegen.msl_emitter.elementwise import _emit_threadgroup_reduce
from metile.compiler.lowering import lower
from metile.compiler.passes import (
    fold_constants,
    split_elementwise_loops,
    vectorize_elementwise,
)
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile_kernels.rmsnorm import rmsnorm_register

BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"


@metile.kernel
def _repeated_reduction(
    Source,
    Output,
    ROWS,
    COLUMNS,
    *,
    BLOCK: metile.constexpr,
    OPERATION: metile.constexpr,
):
    source = metile.tensor(Source, shape=(ROWS, COLUMNS), access="read")
    output = metile.tensor(Output, shape=(ROWS, BLOCK), access="write")
    row = metile.program_id(0)
    lanes = metile.arange(0, BLOCK)
    total = metile.loop_state(0.0)
    for start in metile.tile_range(0, COLUMNS, BLOCK):
        if OPERATION == "sum":
            value = metile.sum(source.load((row, start + lanes)))
        elif OPERATION == "max":
            value = metile.max(source.load((row, start + lanes), other=float("-inf")))
        else:
            value = metile.min(source.load((row, start + lanes), other=float("inf")))
        total.update(total.value + value)
    output.store((row, lanes), total.value)


@metile.kernel
def _tile_extrema(Source, Output, *, BLOCK: metile.constexpr, OPERATION: metile.constexpr):
    source = metile.tensor(Source, shape=(BLOCK,), access="read")
    output = metile.tensor(Output, shape=(BLOCK,), access="write")
    lanes = metile.arange(0, BLOCK)
    values = source.load((lanes,))
    reduced = metile.max(values) if OPERATION == "max" else metile.min(values)
    output.store((lanes,), reduced)


def _trace(kernel, parameters, constants):
    with TracingContext(kernel.name) as context:
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        context.func.constexprs.update(constants, STRICT_MATH=True)
        arguments = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        kernel.fn(*arguments, **constants)
    _mark_outputs(context.func)
    return lower(context.func)


@pytest.mark.parametrize(
    "dtype,operation,identity",
    [
        ("f16", "sum", "half(0.0h)"),
        ("f16", "max", "half((-INFINITY))"),
        ("f16", "min", "half(INFINITY)"),
        ("f32", "sum", "float(0.0f)"),
        ("f32", "max", "float((-INFINITY))"),
        ("f32", "min", "float(INFINITY)"),
        ("bf16", "max", "bfloat((-INFINITY))"),
        ("bf16", "min", "bfloat(INFINITY)"),
        ("i32", "sum", "int(0)"),
        ("i32", "max", "int(-2147483648)"),
        ("i32", "min", "int(2147483647)"),
        ("u32", "sum", "uint(0u)"),
        ("u32", "max", "uint(0u)"),
        ("u32", "min", "uint(4294967295u)"),
        ("u8", "max", "uchar(0)"),
        ("u8", "min", "uchar(255)"),
        ("bool", "max", "bool(0)"),
        ("bool", "min", "bool(1)"),
    ],
)
def test_inactive_simd_lanes_use_operation_and_dtype_identity(dtype, operation, identity):
    function = mir.MFunction("reduction_identity", threadgroup_size=(64, 1, 1))
    reduction = mir.MThreadgroupReduce(
        reduce_op=operation,
        operand=mir.MValue("input", ScalarType(dtype)),
        block_size=64,
        dtype=dtype,
    )
    function.add_op(reduction)
    lines = []
    _emit_threadgroup_reduce(reduction, lines, 1, function)
    assert f"? shared_reduce[lid] : {identity};" in "\n".join(lines)


@pytest.mark.parametrize("operation", ["sum", "max", "min"])
@pytest.mark.parametrize("threads", [32, 64, 256, 1024, 2048])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_generic_reduction_retires_shared_reads_before_return(operation, threads, dtype):
    function = mir.MFunction("reduction_scratch", threadgroup_size=(threads, 1, 1))
    reduction = mir.MThreadgroupReduce(
        reduce_op=operation,
        operand=mir.MValue("input", ScalarType(dtype)),
        block_size=threads,
        dtype=dtype,
    )
    function.add_op(reduction)
    lines = []
    _emit_threadgroup_reduce(reduction, lines, 1, function)
    source = "\n".join(lines)
    if threads == 32:
        assert "threadgroup_barrier(" not in source
        assert "shared_reduce[" not in source
    else:
        assert lines[-1].strip() == BARRIER
        assert source.count(BARRIER) == 3
        assert source.rindex(BARRIER) > source.rindex("= shared_reduce[0];")


@pytest.mark.parametrize("tile_size,threads", [(128, 32), (1024, 256)])
def test_replicated_partials_retire_reads_without_changing_single_simd_path(tile_size, threads):
    function = _trace(
        rmsnorm_register,
        [
            ("X", PtrType("f32")),
            ("W", PtrType("f32")),
            ("Out", PtrType("f32")),
            ("eps", ScalarType("f32")),
        ],
        {"N": tile_size - 1, "BLOCK": tile_size},
    )
    reduction = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupReduce)
    )
    assert reduction.replicate_partials
    assert reduction.block_size == threads
    lines = []
    _emit_threadgroup_reduce(reduction, lines, 1, function)
    source = "\n".join(lines)
    if threads == 32:
        assert BARRIER not in source
    else:
        assert lines[-1].strip() == BARRIER
        assert source.count(BARRIER) == 2
        assert source.rindex(BARRIER) > source.index("= simd_sum(_partial);")


@pytest.mark.parametrize("operation", ["sum", "max", "min"])
@pytest.mark.parametrize("vectorize", [False, True])
def test_optimized_reduction_loop_preserves_read_retirement(operation, vectorize):
    function = _trace(
        _repeated_reduction,
        [("Source", PtrType("f32")), ("Output", PtrType("f32")), ("ROWS", I32), ("COLUMNS", I32)],
        {"BLOCK": 256, "OPERATION": operation},
    )
    function = split_elementwise_loops(function)
    if vectorize:
        function = vectorize_elementwise(function, vec_size=4)
    function = fold_constants(function)
    source = emit(function)
    reads = re.findall(r"\w+ = shared_reduce_\d+\[0\];", source)
    retired = re.findall(
        r"\w+ = shared_reduce_\d+\[0\];\s*}\s*"
        r"threadgroup_barrier\(mem_flags::mem_threadgroup\);",
        source,
    )
    assert len(reads) == 1
    assert retired and len(retired) == len(reads)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.uint32])
@pytest.mark.parametrize("operation", ["max", "min"])
def test_gpu_extrema_identity_preserves_dtype_boundaries(dtype, operation):
    if dtype == np.float16:
        value = -3 if operation == "max" else 3
    elif dtype == np.float32:
        value = -np.inf if operation == "max" else np.inf
    else:
        limits = np.iinfo(dtype)
        value = limits.min if operation == "max" else limits.max
    values = np.full(64, value, dtype=dtype)
    output = np.zeros_like(values)
    _tile_extrema[(1,)](values, output, BLOCK=64, OPERATION=operation, STRICT_MATH=True)
    np.testing.assert_array_equal(output, values)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("operation", ["sum", "max", "min"])
@pytest.mark.parametrize("threads", [64, 256, 1024])
def test_gpu_repeated_reduction_keeps_every_simdgroup_result_identical(operation, threads):
    generator = np.random.default_rng(412)
    rows, columns = 2, threads * 1024 + 13
    values = generator.integers(-4, 5, size=(rows, columns)).astype(np.float32)
    values += ((np.arange(columns) // threads) % 17 - 8).astype(np.float32)
    source = metile.Buffer(data=values)
    output = metile.Buffer.empty((rows, threads))
    reduce = {"sum": np.sum, "max": np.max, "min": np.min}[operation]
    expected = np.zeros(rows, dtype=np.float32)
    for start in range(0, columns, threads):
        expected += reduce(values[:, start : start + threads], axis=1)
    expected = np.broadcast_to(expected[:, None], output.shape)
    dispatch = _repeated_reduction[(rows,)].prepare(
        source, output, rows, columns, BLOCK=threads, OPERATION=operation, STRICT_MATH=True
    )
    for _iteration in range(16):
        output.numpy().fill(np.nan)
        dispatch()
        np.testing.assert_array_equal(output.numpy(), expected)
