from dataclasses import replace

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.codegen.msl_emitter.common import _val_name
from metile.compiler.execution_report import execution_report, validate_materialized_schedule
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.compiler.ownership import ValueOwnership
from metile.compiler.passes import fold_constants
from metile.compiler.register_memory import group_register_memory, validate_register_memory
from metile.compiler.scheduling import _operands, _register_cost, reorder_for_latency
from metile.ir import metal_ir as mir
from metile.ir.types import BOOL, I32, PtrType, ScalarType, VectorType
from tests.codegen.test_register_lowering_audit import _trace
from tests.codegen.test_register_tiling_audit import _layout


def _copy_function(elements=4, *, kind="blocked4", dtype="f32", stride=1, schedule=None):
    layout = _layout(elements, kind)

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), strides=(stride,), access="read")
        outputs = metile.tensor(output, shape=(width,), strides=(stride,), access="write")
        indices = metile.arange(0, 1024, layout=layout)
        values = inputs.load((indices,), other=-3)
        outputs.store((indices,), values)

    function = _trace(body, dtype=dtype)
    if schedule is not None:
        function.constexprs["SCHEDULE"] = schedule
    return lower(function)


def _vector_loads(function):
    return [operation for operation in function.ops if isinstance(operation, mir.MVectorLoad)]


def _vector_stores(function):
    return [operation for operation in function.ops if isinstance(operation, mir.MVectorStore)]


def _scalar_expansion(*, dtype="f32", stride=1):
    function = mir.MFunction("grouped_memory", kernel_type="row_parallel")
    function.threadgroup_size = (32, 1, 1)
    layout = metile.ThreadLayout((5, 6, 0, 1, 2, 3, 4), elements_per_thread=4)
    function.value_layouts = (ValueOwnership("manual", dtype, (128,), layout, 4),)
    function.params = [
        mir.MParam("source", PtrType(dtype)),
        mir.MParam("output", PtrType(dtype), is_output=True),
        mir.MParam("width", I32, is_scalar=True),
        mir.MParam("origin", I32, is_scalar=True),
    ]
    pointer = mir.MValue("source", PtrType(dtype))
    width = mir.MValue("width", I32)
    origin = mir.MValue("origin", I32)
    lane = function.add_op(mir.ThreadPositionInThreadgroup(), "lid")
    signed_lane = function.add_op(mir.MCast(value=lane, target_dtype="i32"), "signed_lane")
    four = function.add_op(mir.MConstant(value=4 * stride, dtype="i32"), "group_width")
    scaled = function.add_op(mir.MBinOp(op="mul", lhs=signed_lane, rhs=four), "scaled")
    base = function.add_op(mir.MBinOp(op="add", lhs=scaled, rhs=origin), "base")
    accesses = []
    for component in range(4):
        offset = function.add_op(
            mir.MConstant(value=component * stride, dtype="i32"), f"offset_{component}"
        )
        index = function.add_op(mir.MBinOp(op="add", lhs=base, rhs=offset), f"index_{component}")
        mask = function.add_op(
            mir.MCompare(predicate="lt", lhs=index, rhs=width), f"mask_{component}"
        )
        fill = function.add_op(
            mir.MConstant(value=component + 7.5, dtype="f32"), f"fill_{component}"
        )
        load = mir.DeviceLoad(ptr=pointer, index=index, dtype=dtype, mask=mask, other=fill)
        load.result = mir.MValue(f"loaded_{component}", ScalarType(dtype), load)
        accesses.append(load)
    return function, accesses


@pytest.mark.parametrize("elements", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("optimized", [False, True])
def test_blocked_register_groups_materialize_typed_vector_memory(elements, dtype, optimized):
    function = _copy_function(elements, dtype=dtype)
    if optimized:
        function = reorder_for_latency(fold_constants(function))
    validate_register_memory(function)
    validate_materialized_schedule(function)
    loads = _vector_loads(function)
    stores = _vector_stores(function)
    extracts = [
        operation for operation in function.ops if isinstance(operation, mir.MVectorExtract)
    ]
    assert len(loads) == len(stores) == elements // 4
    assert len(extracts) == elements
    assert not any(
        isinstance(operation, (mir.DeviceLoad, mir.DeviceStore)) for operation in function.ops
    )
    assert all(operation.result.type == VectorType(dtype, 4) for operation in loads)
    assert all(operation.result.type == ScalarType(dtype) for operation in extracts)
    assert all(
        len(operation.indices) == len(operation.masks) == 4 for operation in (*loads, *stores)
    )
    source = emit(function)
    scalar_type = ScalarType(dtype).to_msl()
    assert f"packed_{scalar_type}4" in source
    assert "long(" in source
    assert "else" in source
    report = execution_report(function, [])
    assert report.vectorized_loads == elements // 4
    assert report.grouped_loads == report.grouped_stores == elements // 4


@pytest.mark.parametrize("kind,stride", [("striped", 1), ("reversed_xor", 1), ("blocked4", 2)])
@pytest.mark.parametrize("elements", [4, 8, 16, 32])
def test_unproven_or_nonunit_register_addresses_stay_scalar(kind, stride, elements):
    function = _copy_function(elements, kind=kind, stride=stride)
    validate_register_memory(function)
    assert not _vector_loads(function)
    assert not _vector_stores(function)
    assert sum(isinstance(operation, mir.DeviceLoad) for operation in function.ops) == elements


@pytest.mark.parametrize("elements", [4, 8, 16, 32])
def test_scalar_schedule_prevents_register_memory_grouping(elements):
    function = _copy_function(elements, schedule=metile.Schedule(vector_width=1))
    validate_register_memory(function)
    validate_materialized_schedule(function)
    assert not _vector_loads(function)
    assert not _vector_stores(function)
    assert execution_report(function, []).vectorized_loads == 0


@pytest.mark.parametrize("elements", [2, 4, 8, 16, 32])
def test_thread_range_mask_is_an_identity_for_every_materialized_physical_lane(elements):
    function = _copy_function(elements)
    threads = 1024 // elements
    bounded = [
        operation
        for operation in function.ops
        if isinstance(operation, mir.MBinOp)
        and operation.result.name.startswith("_metile_bounded_thread")
    ]
    assert len(bounded) == 1
    bounded = bounded[0]
    assert bounded.op == "bitand"
    assert bounded.lhs.defining_op.target_dtype == "i32"
    assert isinstance(bounded.lhs.defining_op.value.defining_op, mir.ThreadPositionInThreadgroup)
    assert mir.resolve(bounded.rhs).defining_op.value == threads - 1
    assert all(thread & (threads - 1) == thread for thread in range(threads))
    assert function.threadgroup_size == function.schedule_plan.threadgroup_size == (threads, 1, 1)
    assert all(record.layout.thread_count == threads for record in function.value_layouts)
    mappings = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadIndexMap)
    ]
    assert len(mappings) == elements
    for register, mapping in enumerate(mappings):
        packed = mapping.thread.defining_op
        assert packed.op == "add"
        assert packed.lhs is bounded.result
        assert mir.resolve(packed.rhs).defining_op.value == register * threads
    source = emit(function)
    assert f"& {threads - 1};" in source


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_grouping_preserves_each_scalar_result_mask_index_and_fill(dtype):
    function, accesses = _scalar_expansion(dtype=dtype)
    original_results = tuple(operation.result for operation in accesses)
    original_indices = tuple(operation.index for operation in accesses)
    original_masks = tuple(operation.mask for operation in accesses)
    original_fills = tuple(operation.other for operation in accesses)
    grouped = group_register_memory(accesses, 32, lambda stem: stem + "_test")
    function.ops.extend(grouped)
    validate_register_memory(function)
    load = _vector_loads(function)[0]
    extracts = [operation for operation in grouped if isinstance(operation, mir.MVectorExtract)]
    assert tuple(load.indices) == original_indices
    assert tuple(load.masks) == original_masks
    assert tuple(load.others) == original_fills
    assert tuple(operation.result for operation in extracts) == original_results
    assert [operation.lane for operation in extracts] == [0, 1, 2, 3]
    assert all(operation.result.defining_op is operation for operation in extracts)
    assert all(operation.value is load.result for operation in extracts)
    source = emit(function)
    for mask, index, fill in zip(original_masks, original_indices, original_fills, strict=True):
        assert f"{_val_name(mask, function)} ? source[{_val_name(index, function)}]" in source
        assert _val_name(fill, function) in source


def test_grouping_does_not_cross_distinct_original_tensor_loads():
    layout = _layout(8, "blocked4")

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, 1024, layout=layout)
        left = inputs.load((indices - 1,), other=-7)
        right = inputs.load((indices + 3,), other=11)
        outputs.store((indices,), left + right)

    function = lower(_trace(body))
    validate_register_memory(function)
    loads = _vector_loads(function)
    assert len(loads) == 4
    fills = [tuple(mir.resolve(fill).defining_op.value for fill in load.others) for load in loads]
    assert fills.count((-7, -7, -7, -7)) == 2
    assert fills.count((11, 11, 11, 11)) == 2


def test_memory_groups_retain_tuple_operands_for_scheduling_and_liveness():
    function, accesses = _scalar_expansion()
    function.ops.extend(group_register_memory(accesses, 32, lambda stem: stem + "_test"))
    load = _vector_loads(function)[0]
    operands = _operands(load)
    assert all(any(candidate is value for candidate in operands) for value in load.indices)
    assert all(any(candidate is value for candidate in operands) for value in load.masks)
    assert all(any(candidate is value for candidate in operands) for value in load.others)
    assert _register_cost(load.result) == 4
    function = reorder_for_latency(fold_constants(function))
    validate_register_memory(function)
    emitted = emit(function)
    assert emitted.index(f"float4 {load.result.name}") < emitted.index("float loaded_0")


def test_runtime_group_guard_widens_indices_before_comparing_wrapped_offsets():
    function, accesses = _scalar_expansion()
    function.ops.extend(group_register_memory(accesses, 32, lambda stem: stem + "_test"))
    source = emit(function)
    guard = next(line for line in source.splitlines() if line.strip().startswith("if ("))
    for component in range(4):
        assert f"(mask_{component})" in guard
    for component in range(1, 4):
        assert f"(long(index_{component}) == long(index_0) + {component})" in guard
    assert source.index(guard) < source.index("reinterpret_cast")
    wrapped = (2**32 - 2, 2**32 - 1, 0, 1)
    assert not all(index == wrapped[0] + component for component, index in enumerate(wrapped))


@pytest.mark.parametrize("change", ["mixed_effect", "missing_lane", "other_pointer", "dependency"])
def test_grouping_keeps_unsafe_scalar_expansions_unchanged(change):
    _, accesses = _scalar_expansion()
    if change == "mixed_effect":
        accesses[2] = mir.MBarrier()
    elif change == "missing_lane":
        accesses.pop()
    elif change == "other_pointer":
        accesses[-1].ptr = mir.MValue("other_source", PtrType("f32"))
    else:
        accesses[-1].other = accesses[0].result
    results = tuple(operation.result for operation in accesses)
    grouped = group_register_memory(accesses, 32, lambda stem: stem + "_test")
    assert len(grouped) == len(accesses)
    assert all(first is second for first, second in zip(grouped, accesses, strict=True))
    assert all(
        result is None or result.defining_op is operation
        for result, operation in zip(results, accesses, strict=True)
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "indices_arity",
        "masks_arity",
        "fills_arity",
        "pointer_dtype",
        "pointer_space",
        "load_dtype",
        "load_result",
        "index_dtype",
        "mask_dtype",
        "extract_lane",
        "extract_result",
        "orphan_extract",
        "extract_before_load",
        "store_values_arity",
        "store_value_dtype",
        "store_pointer_dtype",
        "store_masks_arity",
        "ownership_count",
        "ownership_shape",
        "ownership_layout",
        "missing_layout",
        "invalid_pointer",
        "invalid_result",
        "invalid_extract_value",
        "empty_geometry",
        "missing_index_thread",
        "missing_index_layout",
    ],
)
def test_vector_memory_verifier_rejects_malformed_typed_operations(corruption):
    function = _copy_function()
    load = _vector_loads(function)[0]
    store = _vector_stores(function)[0]
    extract = next(
        operation for operation in function.ops if isinstance(operation, mir.MVectorExtract)
    )
    if corruption == "indices_arity":
        load.indices = load.indices[:3]
    elif corruption == "masks_arity":
        load.masks = load.masks[:3]
    elif corruption == "fills_arity":
        load.others = load.others[:3]
    elif corruption == "pointer_dtype":
        load.ptr = mir.MValue("other_pointer", PtrType("f16"))
    elif corruption == "pointer_space":
        load.ptr = mir.MValue("other_pointer", PtrType("f32", "threadgroup"))
    elif corruption == "load_dtype":
        load.dtype = "i32"
    elif corruption == "load_result":
        load.result.type = ScalarType("f32")
    elif corruption == "index_dtype":
        load.indices = (mir.MValue("floating_index", ScalarType("f32")), *load.indices[1:])
    elif corruption == "mask_dtype":
        load.masks = (mir.MValue("integer_mask", I32), *load.masks[1:])
    elif corruption == "extract_lane":
        extract.lane = 4
    elif corruption == "extract_result":
        extract.result.type = ScalarType("f16")
    elif corruption == "orphan_extract":
        extract.value = mir.MValue("missing_vector", VectorType("f32", 4))
    elif corruption == "extract_before_load":
        function.ops.remove(extract)
        function.ops.insert(function.ops.index(load), extract)
    elif corruption == "store_values_arity":
        store.values = store.values[:3]
    elif corruption == "store_value_dtype":
        store.values = (mir.MValue("pointer_value", PtrType("f32")), *store.values[1:])
    elif corruption == "store_pointer_dtype":
        store.ptr = mir.MValue("other_pointer", PtrType("f16"))
    elif corruption == "store_masks_arity":
        store.masks = (mir.MValue("valid_mask", BOOL),)
    elif corruption == "ownership_count":
        function.value_layouts = (
            replace(function.value_layouts[0], elements_per_thread=8),
            *function.value_layouts[1:],
        )
    elif corruption == "ownership_shape":
        function.value_layouts = (
            replace(function.value_layouts[0], shape=(2048,)),
            *function.value_layouts[1:],
        )
    elif corruption == "ownership_layout":
        function.value_layouts = (
            replace(
                function.value_layouts[0],
                layout=metile.ThreadLayout.identity(2048, elements_per_thread=8),
                elements_per_thread=8,
                shape=(2048,),
            ),
            *function.value_layouts[1:],
        )
    elif corruption == "missing_layout":
        function.value_layouts = (
            replace(function.value_layouts[0], layout=None),
            *function.value_layouts[1:],
        )
    elif corruption == "invalid_pointer":
        load.ptr = 7
    elif corruption == "invalid_result":
        load.result = 7
    elif corruption == "invalid_extract_value":
        extract.value = 7
    elif corruption == "empty_geometry":
        function.threadgroup_size = ()
    elif corruption == "missing_index_thread":
        mapping = next(
            operation for operation in function.ops if isinstance(operation, mir.MThreadIndexMap)
        )
        mapping.thread = None
    elif corruption == "missing_index_layout":
        mapping = next(
            operation for operation in function.ops if isinstance(operation, mir.MThreadIndexMap)
        )
        mapping.layout = None
    with pytest.raises(LoweringError):
        validate_register_memory(function)


@metile.kernel
def grouped_shifted_neighbors(
    source, output, width, BLOCK: metile.constexpr, LAYOUT: metile.constexpr
):
    row = metile.program_id(0)
    inputs = metile.tensor(source + row * width + 1, shape=(width,), access="read")
    outputs = metile.tensor(output + row * width + 1, shape=(width,), access="write")
    indices = metile.arange(0, BLOCK, layout=LAYOUT)
    previous = metile.cast(inputs.load((indices - 1,), other=-7.003), "f32")
    upcoming = metile.cast(inputs.load((indices + 3,), other=11.007), "f32")
    outputs.store((indices,), previous * 0.25 + upcoming * 2.0)


@pytest.mark.parametrize("elements", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("width", [1009, 1024])
def test_gpu_grouped_memory_preserves_shifted_pointers_ragged_bounds_and_distinct_fills(
    elements, dtype, width
):
    rows = 3
    source = ((np.arange(rows * width + 2) % 37) - 16).astype(dtype)
    output = np.full_like(source, -123)
    dispatch = grouped_shifted_neighbors[(rows,)].prepare(
        source, output, width, BLOCK=1024, LAYOUT=_layout(elements, "blocked4")
    )
    expected = np.full_like(source, -123)
    for row in range(rows):
        start = row * width + 1
        values = source[start : start + width].astype(np.float32)
        previous = np.r_[np.float32(dtype(-7.003)), values[:-1]]
        fill = np.float32(dtype(11.007))
        upcoming = np.r_[values[3:], fill, fill, fill]
        expected[start : start + width] = previous * 0.25 + upcoming * 2.0
    np.testing.assert_array_equal(output, expected)
    assert dispatch.execution_report.grouped_loads == 2 * (elements // 4)
    assert dispatch.execution_report.grouped_stores == elements // 4


@metile.kernel
def grouped_inplace_copy(
    source,
    output,
    width,
    BLOCK: metile.constexpr,
    LAYOUT: metile.constexpr,
    STRIDE: metile.constexpr,
):
    row = metile.program_id(0)
    inputs = metile.tensor(
        source + row * width * STRIDE + 1,
        shape=(width,),
        strides=(STRIDE,),
        access="read",
    )
    outputs = metile.tensor(
        output + row * width * STRIDE + 1,
        shape=(width,),
        strides=(STRIDE,),
        access="write",
    )
    indices = metile.arange(0, BLOCK, layout=LAYOUT)
    outputs.store((indices,), inputs.load((indices,)))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("scalar", [False, True])
def test_gpu_scheduled_inplace_copy_preserves_signed_zeros_and_scalar_fallback(
    dtype, stride, scalar, monkeypatch
):
    monkeypatch.setenv("METILE_SCHEDULE", "1")
    width = 1009
    rows = 3
    buffer = np.resize(
        np.asarray([0.0, -0.0, 1.0, -1.0, 0.5], dtype=dtype), rows * width * stride + 2
    )
    expected = buffer.copy()
    schedule = metile.Schedule(vector_width=1) if scalar else metile.Schedule()
    dispatch = grouped_inplace_copy[(rows,)].prepare(
        buffer,
        buffer,
        width,
        BLOCK=1024,
        LAYOUT=_layout(32, "blocked4"),
        STRIDE=stride,
        SCHEDULE=schedule,
    )
    bits = np.uint16 if dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(buffer.view(bits), expected.view(bits))
    grouped = 8 if stride == 1 and not scalar else 0
    assert dispatch.execution_report.grouped_loads == grouped
    assert dispatch.execution_report.grouped_stores == grouped
