import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.execution_report import execution_report
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.compiler.ownership import validate_register_reductions
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType, TileType
from metile_kernels.rmsnorm import rmsnorm_register


def _trace(width=1009, block=1024, dtype="f32", layout=None):
    context = TracingContext("register_norm")
    with context:
        arguments = []
        for name, datatype in (
            ("X", PtrType(dtype)),
            ("W", PtrType(dtype)),
            ("Out", PtrType(dtype)),
            ("eps", ScalarType("f32")),
        ):
            context.func.params.append(tir.Param(name, datatype, is_output=name == "Out"))
            arguments.append(TracingProxy(tir.Value(name, datatype)))
        rmsnorm_register.fn(*arguments, N=width, BLOCK=block, LAYOUT=layout)
    return context.func


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_register_rmsnorm_reuses_four_values_across_one_fp32_reduction(dtype):
    function = lower(_trace(dtype=dtype))
    validate_register_reductions(function)
    loads = [operation for operation in function.ops if isinstance(operation, mir.DeviceLoad)]
    stores = [operation for operation in function.ops if isinstance(operation, mir.DeviceStore)]
    reductions = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupReduce)
    ]
    assert len(loads) == 8
    assert len([load for load in loads if load.ptr.name == "X"]) == 4
    assert len([load for load in loads if load.ptr.name == "W"]) == 4
    assert len(stores) == 4
    assert all(operation.mask is not None for operation in (*loads, *stores))
    assert not any(isinstance(operation, mir.MForLoop) for operation in function.ops)
    assert len(reductions) == 1
    assert reductions[0].dtype == "f32"
    assert reductions[0].block_size == 256
    assert function.threadgroup_size == (256, 1, 1)
    assert function.schedule_plan.tile_shape == (1024,)
    assert function.schedule_plan.staging == "threadgroup"
    report = execution_report(function, []).to_dict()
    assert report["register_reductions"][0]["elements_per_thread"] == 4
    assert all(value["elements_per_thread"] == 4 for value in report["value_layouts"])
    assert report["allocations"][0]["bytes"] == 32
    source = emit(function)
    assert source.count("threadgroup_barrier(") == 2
    assert "simd_sum(" in source


def test_one_simdgroup_register_sum_needs_no_shared_storage():
    function = lower(_trace(width=127, block=128))
    validate_register_reductions(function)
    assert function.threadgroup_size == (32, 1, 1)
    assert function.schedule_plan.staging == "device"
    assert function.register_reductions[0].scratch is None
    assert "threadgroup_barrier(" not in emit(function)


@pytest.mark.parametrize("width", [0, -1, 1025, 1.0, True])
def test_single_read_kernel_rejects_invalid_or_uncovered_widths(width):
    with pytest.raises(ValueError, match="0 < N <= BLOCK"):
        _trace(width=width)


@pytest.mark.parametrize(
    "corruption",
    [
        "dtype",
        "operation",
        "geometry",
        "scratch",
        "nested",
        "reuse",
        "broadcast",
        "alias",
        "orphan",
    ],
)
def test_post_pass_reduction_validation_rejects_contract_corruption(corruption):
    function = lower(_trace())
    reduction = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupReduce)
    )
    if corruption == "dtype":
        reduction.dtype = "f16"
    elif corruption == "operation":
        reduction.reduce_op = "max"
    elif corruption == "geometry":
        function.threadgroup_size = (128, 1, 1)
    elif corruption == "scratch":
        reduction.shared_name = "missing"
    elif corruption == "nested":
        function.ops[function.ops.index(reduction)] = mir.MForLoop(body=[reduction])
    elif corruption == "reuse":
        function.ops.append(mir.MThreadgroupStore(array_name=reduction.shared_name))
    elif corruption == "broadcast":
        reduction.replicate_partials = False
    elif corruption == "alias":
        function.ops.append(
            mir.DeviceLoad(ptr=mir.MValue(reduction.shared_name, PtrType("f32", "threadgroup")))
        )
    elif corruption == "orphan":
        function.register_reductions = ()
    with pytest.raises(LoweringError, match="register reduction"):
        validate_register_reductions(function)


@pytest.mark.parametrize("operation,dtype", [("max", "f32"), ("sum", "f16")])
def test_unsupported_reductions_reject_before_scalarization(operation, dtype):
    function = _trace()
    reduction = next(item for item in function.ops if isinstance(item, tir.Reduce))
    reduction.op = operation
    reduction.operand.type = TileType(
        (1024,), dtype, metile.ThreadLayout.identity(1024, elements_per_thread=4)
    )
    with pytest.raises(LoweringError, match="FP32 tile sum"):
        lower(function)


def test_device_only_schedule_cannot_hide_register_reduction_scratch():
    function = _trace()
    function.constexprs["SCHEDULE"] = metile.Schedule(staging="device")
    with pytest.raises(LoweringError, match="staging"):
        lower(function)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("rows", [1, 32, 256])
@pytest.mark.parametrize("width", [1009, 1024])
def test_gpu_register_rmsnorm_matches_full_fp32_reference(dtype, rows, width):
    generator = np.random.default_rng(rows + width)
    source = generator.standard_normal((rows, width)).astype(dtype)
    weights = generator.standard_normal(width).astype(dtype)
    output = np.full_like(source, np.nan)
    dispatch = rmsnorm_register[(rows,)].prepare(
        source, weights, output, 1e-5, N=width, BLOCK=1024, RELAXED_PRECISION=False
    )
    promoted = source.astype(np.float32)
    expected = (
        promoted
        / np.sqrt(np.mean(promoted * promoted, axis=1, keepdims=True) + 1e-5)
        * weights.astype(np.float32)
    ).astype(dtype)
    tolerance = 2e-3 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
    assert dispatch.execution_report.register_reductions


@metile.kernel
def register_copy(source, output, width, BLOCK: metile.constexpr, LAYOUT: metile.constexpr):
    inputs = metile.tensor(source, shape=(width,), access="read")
    outputs = metile.tensor(output, shape=(width,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK, layout=LAYOUT)
    values = inputs.load((positions,))
    outputs.store((positions,), values)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.uint32])
@pytest.mark.parametrize("block", [128, 1024, 4096])
def test_gpu_register_permutations_cover_each_logical_value_once(dtype, block):
    identity = metile.ThreadLayout.identity(block, elements_per_thread=4)
    layout = metile.ThreadLayout(
        tuple(reversed(identity.bit_order)), xor_mask=7, elements_per_thread=4
    )
    source = np.arange(block * 2 - 5, dtype=dtype)
    output = np.zeros_like(source)
    register_copy[(2,)].prepare(source, output, source.size, BLOCK=block, LAYOUT=layout)
    np.testing.assert_array_equal(output, source)


def test_register_pointer_origin_remains_uniform():
    function = _trace()
    arange = next(operation for operation in function.ops if isinstance(operation, tir.Arange))
    arange.start = tir.Value("uniform_origin", I32)
    function.params.append(tir.Param("uniform_origin", I32))
    lowered = lower(function)
    assert lowered.threadgroup_size == (256, 1, 1)


@metile.kernel
def register_sums(source, output, width, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(width,), access="read")
    outputs = metile.tensor(output, shape=(width,), access="write")
    positions = metile.arange(
        0, BLOCK, layout=metile.ThreadLayout.identity(BLOCK, elements_per_thread=4)
    )
    values = inputs.load((positions,))
    first = metile.sum(values)
    second = metile.sum(values * values)
    outputs.store((positions,), first + second + values)


@pytest.mark.parametrize("block", [128, 256, 512, 1024, 2048, 4096])
def test_gpu_each_simdgroup_receives_two_complete_sums_with_distinct_immutable_scratch(block):
    source = (np.arange(block - 3, dtype=np.float32) % 5) - 2
    output = np.full_like(source, np.nan)
    dispatch = register_sums[(1,)].prepare(source, output, source.size, BLOCK=block)
    np.testing.assert_array_equal(output, np.sum(source) + np.sum(source * source) + source)
    allocation_count = 0 if block == 128 else 2
    assert len(dispatch.execution_report.allocations) == allocation_count
    assert (
        len({allocation.name for allocation in dispatch.execution_report.allocations})
        == allocation_count
    )
