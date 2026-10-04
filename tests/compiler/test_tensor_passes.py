import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import fold_constants, split_elementwise_loops, vectorize_elementwise
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile_kernels.layernorm import layernorm
from metile_kernels.rmsnorm import rmsnorm


def _append_value(operations, operation, name):
    value = mir.MValue(name, operation.result_type(), operation)
    operation.result = value
    operations.append(operation)
    return value


def _masked_loop_function(
    *, stride=1, shift=0, extra_bound=False, bound_name="size", shared=False, hoisted_lane=False
):
    function = mir.MFunction("tensor_loop", kernel_type="row_parallel")
    function.threadgroup_size = (32, 1, 1)
    function.params = [
        mir.MParam("source", PtrType("f32")),
        mir.MParam("destination", PtrType("f32"), is_output=True),
        mir.MParam("size", I32, is_scalar=True),
    ]
    size = mir.MValue("size", I32)
    source = mir.MValue("source", PtrType("f32"))
    destination = mir.MValue("destination", PtrType("f32"))
    lane = function.add_op(mir.ThreadPositionInThreadgroup(), "lid")
    body = []
    signed_lane = _append_value(
        function.ops if hoisted_lane else body,
        mir.MCast(value=lane, target_dtype="i32"),
        "signed_lane",
    )
    zero = _append_value(body, mir.MConstant(value=0, dtype="i32"), "zero")
    lane_offset = _append_value(
        body, mir.MBinOp(op="add", lhs=signed_lane, rhs=zero), "lane_offset"
    )
    coordinate = _append_value(
        body, mir.MBinOp(op="add", lhs=mir.MValue("offset", I32), rhs=lane_offset), "coordinate"
    )
    if shift:
        shifted = _append_value(body, mir.MConstant(value=shift, dtype="i32"), "shift")
        coordinate = _append_value(
            body, mir.MBinOp(op="add", lhs=coordinate, rhs=shifted), "shifted_coordinate"
        )
    lower = _append_value(
        body, mir.MCompare(predicate="ge", lhs=coordinate, rhs=zero), "lower_bound"
    )
    upper = _append_value(
        body,
        mir.MCompare(predicate="lt", lhs=coordinate, rhs=mir.MValue(bound_name, I32)),
        "upper_bound",
    )
    mask = _append_value(body, mir.MBinOp(op="bitand", lhs=lower, rhs=upper), "mask")
    if extra_bound:
        row = _append_value(function.ops, mir.ThreadgroupPositionInGrid(), "tgp_id_x")
        row_bound = _append_value(
            body, mir.MCompare(predicate="lt", lhs=row, rhs=size), "row_bound"
        )
        mask = _append_value(body, mir.MBinOp(op="bitand", lhs=mask, rhs=row_bound), "full_mask")
    memory_stride = _append_value(body, mir.MConstant(value=stride, dtype="i32"), "stride")
    address = _append_value(
        body, mir.MBinOp(op="mul", lhs=coordinate, rhs=memory_stride), "address"
    )
    if shared:
        load = mir.MThreadgroupLoad(array_name="shared", index=address, mask=mask, other=zero)
    else:
        load = mir.DeviceLoad(ptr=source, index=address, mask=mask, other=zero)
    loaded = _append_value(body, load, "loaded")
    if shared:
        body.append(
            mir.MThreadgroupStore(array_name="shared", index=address, value=loaded, mask=mask)
        )
    else:
        body.append(mir.DeviceStore(ptr=destination, index=address, value=loaded, mask=mask))
    loop = mir.MForLoop(iv_name="offset", start=0, end=size, step=32, body=body)
    function.ops.append(loop)
    return function, loop


def _loops(function):
    return [operation for operation in function.ops if isinstance(operation, mir.MForLoop)]


def _accesses(loop):
    return [
        operation
        for operation in loop.body
        if isinstance(
            operation,
            (mir.DeviceLoad, mir.DeviceStore, mir.MThreadgroupLoad, mir.MThreadgroupStore),
        )
    ]


def test_rank_one_masks_split_and_vectorize_with_masked_tail():
    function, original = _masked_loop_function()
    original._num_stages = 2

    split_elementwise_loops(function)
    vectorize_elementwise(function)

    aligned, tail = _loops(function)
    assert aligned._ew_aligned
    assert aligned._vec_size == 4
    assert aligned._num_stages == 2
    assert tail._ew_tail and tail._vec_tail
    assert aligned._ew_id == tail._ew_id
    assert all(access.mask is None for access in _accesses(aligned))
    assert all(access.mask is not None for access in _accesses(tail))
    assert all(access.mask is not None for access in _accesses(original))
    source = emit(function)
    assert "float4 loaded" in source
    assert "static_cast<int>(lid) * 4" in source
    assert "? source[address]" in source


def test_loop_clones_preserve_parent_ssa_forwarding():
    function, loop = _masked_loop_function()
    size = mir.MValue("size", I32)
    parent_ops = []
    original = _append_value(parent_ops, mir.MBinOp(op="mul", lhs=size, rhs=size), "original")
    duplicate = _append_value(parent_ops, mir.MBinOp(op="mul", lhs=size, rhs=size), "duplicate")
    function.ops[:0] = parent_ops
    load, _ = _accesses(loop)
    offset = _append_value(
        loop.body, mir.MBinOp(op="add", lhs=duplicate, rhs=load.index), "shifted_address"
    )
    address_operation = loop.body.pop()
    load_position = next(index for index, operation in enumerate(loop.body) if operation is load)
    loop.body.insert(load_position, address_operation)
    load.index = offset

    split_elementwise_loops(function)
    fold_constants(function)

    aligned, tail = _loops(function)
    for clone in (aligned, tail):
        address = _accesses(clone)[0].index.defining_op
        assert address.lhs is duplicate
        assert mir.resolve(address.lhs) is original
    assert "duplicate" not in emit(function)


@pytest.mark.parametrize("shift", [-32, -1, 1, 32])
def test_shifted_coordinate_masks_are_not_removed(shift):
    function, original = _masked_loop_function(shift=shift)

    split_elementwise_loops(function)
    vectorize_elementwise(function)

    assert _loops(function) == [original]
    assert all(access.mask is not None for access in _accesses(original))
    assert not hasattr(original, "_vec_size")


@pytest.mark.parametrize("options", [{"extra_bound": True}, {"bound_name": "other_size"}])
def test_independent_and_multidimensional_bounds_are_preserved(options):
    function, original = _masked_loop_function(**options)

    split_elementwise_loops(function)

    assert _loops(function) == [original]
    assert all(access.mask is not None for access in _accesses(original))


@pytest.mark.parametrize("stride", [-1, 0, 2, 4])
def test_nonunit_lane_stride_stays_scalar_after_mask_proof(stride):
    function, _ = _masked_loop_function(stride=stride)

    split_elementwise_loops(function)
    vectorize_elementwise(function)

    aligned, tail = _loops(function)
    assert all(access.mask is None for access in _accesses(aligned))
    assert not hasattr(aligned, "_vec_size")
    assert not hasattr(tail, "_vec_tail")


@pytest.mark.parametrize("remaining_access", [0, 1])
def test_vectorization_rejects_any_surviving_access_mask(remaining_access):
    function, loop = _masked_loop_function()
    loop._ew_aligned = True
    _accesses(loop)[1 - remaining_access].mask = None

    vectorize_elementwise(function)

    assert not hasattr(loop, "_vec_size")


@pytest.mark.parametrize(
    ("start", "step", "threadgroup_size"),
    [(1, 32, (32, 1, 1)), (0, 16, (32, 1, 1)), (0, 0, (32, 1, 1)), (0, 32, (16, 2, 1))],
)
def test_noncanonical_loop_domains_are_not_split(start, step, threadgroup_size):
    function, original = _masked_loop_function()
    original.start = start
    original.step = step
    function.threadgroup_size = threadgroup_size

    split_elementwise_loops(function)

    assert _loops(function) == [original]


def test_shared_memory_masks_split_without_unsupported_vector_accesses():
    function, _ = _masked_loop_function(shared=True)

    split_elementwise_loops(function)
    vectorize_elementwise(function)

    aligned, tail = _loops(function)
    assert all(access.mask is None for access in _accesses(aligned))
    assert all(access.mask is not None for access in _accesses(tail))
    assert not hasattr(aligned, "_vec_size")


def test_hoisted_lane_expression_is_not_vectorized():
    function, _ = _masked_loop_function(hoisted_lane=True)

    split_elementwise_loops(function)
    vectorize_elementwise(function)

    aligned, _ = _loops(function)
    assert not hasattr(aligned, "_vec_size")


def test_masked_loop_keeps_unrelated_control_flow():
    function, original = _masked_loop_function()
    store = original.body.pop()
    guard = mir.MValue("enabled", I32)
    original.body.append(mir.IfBlock(condition=guard, body=[store]))

    split_elementwise_loops(function)

    assert _loops(function) == [original]
    assert isinstance(original.body[-1], mir.IfBlock)


def test_constant_loop_bound_is_matched_by_value():
    function, original = _masked_loop_function()
    original.end = 257
    upper = next(
        operation
        for operation in original.body
        if isinstance(operation, mir.MCompare) and operation.predicate == "lt"
    )
    upper.rhs = _append_value(original.body, mir.MConstant(value=257, dtype="i32"), "size")

    split_elementwise_loops(function)

    assert len(_loops(function)) == 2


@pytest.mark.parametrize("kernel", [rmsnorm, layernorm])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_descriptor_normalization_keeps_vectorized_interior(kernel, dtype):
    with TracingContext(kernel.name) as context:
        parameters = []
        arguments = []
        for name in kernel._sig.parameters:
            if name == "BLOCK":
                continue
            parameter_type = (
                I32 if name == "N" else ScalarType("f32") if name == "eps" else PtrType(dtype)
            )
            parameters.append(tir.Param(name, parameter_type, is_output=name == "Out"))
            arguments.append(TracingProxy(tir.Value(name, parameter_type)))
        context.func.params = parameters
        context.func.constexprs = {"BLOCK": 32}
        kernel.fn(*arguments, BLOCK=32)

    function = lower(context.func)
    split_elementwise_loops(function)
    vectorize_elementwise(function)

    loops = _loops(function)
    assert len(loops) == 4
    assert all(getattr(loop, "_vec_size", None) == 4 for loop in loops[::2])
    assert all(getattr(loop, "_vec_tail", False) for loop in loops[1::2])
    source = emit(function)
    assert f"device const {ScalarType(dtype).to_msl()}4*" in source
