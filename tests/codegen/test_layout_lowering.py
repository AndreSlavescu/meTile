import itertools

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.compiler.ownership import (
    conversion_map,
    conversion_mechanism,
    validate_layout_exchanges,
)
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType


@metile.kernel
def ownership_copy(
    source,
    destination,
    width,
    BLOCK: metile.constexpr,
    SOURCE: metile.constexpr,
    TARGET: metile.constexpr,
    ROUNDTRIP: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(width,), access="read")
    outputs = metile.tensor(destination, shape=(width,), access="write")
    origin = metile.program_id(0) * BLOCK
    positions = origin + metile.arange(0, BLOCK, layout=SOURCE)
    values = inputs.load((positions,))
    values = metile.convert_layout(values, TARGET)
    if ROUNDTRIP:
        values = values + 0.0
        values = metile.convert_layout(values, SOURCE)
        outputs.store((positions,), values)
    else:
        target_positions = origin + metile.arange(0, BLOCK, layout=TARGET)
        outputs.store((target_positions,), values)


def _trace(source, destination, roundtrip=False):
    context = TracingContext("owned_copy")
    with context:
        arguments = []
        for name, dtype in (
            ("source", PtrType("f32")),
            ("destination", PtrType("f32")),
            ("width", I32),
        ):
            context.func.params.append(tir.Param(name, dtype, is_output=name == "destination"))
            arguments.append(TracingProxy(tir.Value(name, dtype)))
        ownership_copy.fn(
            *arguments,
            BLOCK=source.size,
            SOURCE=source,
            TARGET=destination,
            ROUNDTRIP=roundtrip,
        )
    return context.func


def test_conversion_composition_matches_exact_owner_lookup():
    layouts = [
        metile.ThreadLayout(order, mask)
        for order in itertools.permutations(range(5))
        for mask in (0, 3, 31)
    ]
    for source, destination in zip(layouts, reversed(layouts), strict=True):
        mapping = conversion_map(source, destination)
        assert [mapping.logical_index(thread) for thread in range(32)] == [
            source.owner(destination.logical_index(thread)) for thread in range(32)
        ]


def test_conversion_mechanisms_follow_actual_ownership_not_names():
    source = metile.ThreadLayout.identity(64)
    within = metile.ThreadLayout((2, 3, 4, 0, 1, 5))
    across = metile.ThreadLayout((3, 4, 5, 0, 1, 2))
    assert conversion_mechanism(source, source) == "identity"
    assert conversion_mechanism(source, within) == "simd_shuffle"
    assert conversion_mechanism(source, across) == "threadgroup"
    assert conversion_mechanism(source, metile.ThreadLayout(source.bit_order, 32)) == "threadgroup"


def test_identity_conversion_emits_no_communication():
    layout = metile.ThreadLayout.identity(32)
    function = lower(_trace(layout, layout))
    source = emit(function)
    assert "simd_shuffle(" not in source
    assert "threadgroup_barrier(" not in source
    assert not function.layout_conversions
    assert function.layout_optimizations.identities_removed == 1
    assert function.value_layouts


def test_simd_conversion_uses_arbitrary_source_lane_shuffle():
    function = lower(_trace(metile.ThreadLayout.identity(32), metile.ThreadLayout((2, 3, 4, 0, 1))))
    source = emit(function)
    assert "simd_shuffle(" in source
    assert "simd_broadcast(" not in source
    assert "threadgroup_barrier(" not in source
    assert not any(isinstance(operation, mir.MThreadgroupAlloc) for operation in function.ops)


def test_cross_simdgroup_conversion_reuses_scratch_after_recycle_barrier():
    function = lower(
        _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)), True)
    )
    validate_layout_exchanges(function)
    allocations = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert len(allocations) == 1
    assert allocations[0].size == 64
    assert (
        len([operation for operation in function.ops if isinstance(operation, mir.MBarrier)]) == 4
    )
    assert function.layout_conversions[0].scratch == function.layout_conversions[1].scratch
    assert function.schedule_plan.staging == "threadgroup"


@pytest.mark.parametrize("barrier_index", [0, 1])
def test_exchange_validation_rejects_removed_publish_or_recycle(barrier_index):
    function = lower(
        _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)))
    )
    indices = [
        index for index, operation in enumerate(function.ops) if isinstance(operation, mir.MBarrier)
    ]
    del function.ops[indices[barrier_index]]
    with pytest.raises(LoweringError, match="publish and recycle"):
        validate_layout_exchanges(function)


def test_expert_device_staging_cannot_hide_cross_group_communication():
    function = _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)))
    function.constexprs["SCHEDULE"] = metile.Schedule(staging="device")
    with pytest.raises(LoweringError, match="staging"):
        lower(function)


@pytest.mark.parametrize("operation", [tir.Reduce(), tir.ForRange(), tir.ThreadId(), tir.Barrier()])
def test_unsupported_collective_contexts_fail_before_layout_materialization(operation):
    layout = metile.ThreadLayout.identity(32)
    function = _trace(layout, layout)
    function.ops.append(operation)
    with pytest.raises(LoweringError, match="straight-line"):
        lower(function)


def test_masked_raw_pointer_loads_are_not_accepted_as_convergent():
    layout = metile.ThreadLayout.identity(32)
    function = _trace(layout, layout)
    load = next(operation for operation in function.ops if isinstance(operation, tir.Load))
    load.tensor = None
    with pytest.raises(LoweringError, match="descriptor-based"):
        lower(function)


def test_tensor_base_pointer_cannot_hide_nonuniform_indexing():
    layout = metile.ThreadLayout.identity(32)
    function = _trace(layout, layout)
    load = next(operation for operation in function.ops if isinstance(operation, tir.Load))
    load.tensor.ptr = load.ptr
    with pytest.raises(LoweringError, match="base pointers must be uniform"):
        lower(function)


def test_scalar_typing_cannot_hide_per_thread_values():
    layout = metile.ThreadLayout.identity(32)
    function = _trace(layout, layout)
    positions = next(operation for operation in function.ops if isinstance(operation, tir.Arange))
    varying = tir.Unary(op="reverse_bits", operand=positions.result)
    varying.result = tir.Value("hidden_varying", ScalarType("u32"), defining_op=varying)
    function.ops.append(varying)
    with pytest.raises(LoweringError, match="nonuniform scalar"):
        lower(function)


@pytest.mark.parametrize("origin", ["tile", "float"])
def test_arange_origins_must_be_uniform_integer_scalars(origin):
    layout = metile.ThreadLayout.identity(32)
    function = _trace(layout, layout)
    ranges = [operation for operation in function.ops if isinstance(operation, tir.Arange)]
    ranges[-1].start = (
        ranges[0].result if origin == "tile" else tir.Value("floating_origin", ScalarType("f32"))
    )
    with pytest.raises(LoweringError, match="uniform integer scalars"):
        lower(function)


@pytest.mark.parametrize("layout", [None, metile.ThreadLayout.identity(32)])
@pytest.mark.parametrize("origin", [0.5, True])
def test_layout_entry_cannot_silently_drop_an_invalid_origin(layout, origin):
    with TracingContext("floating_origin"), pytest.raises(TypeError, match="integer scalars"):
        metile.arange(origin, 32, layout=layout)


def test_exchange_validation_rejects_owner_mapping_corruption():
    function = lower(
        _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)))
    )
    load = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupLoad)
    )
    load.index.defining_op.layout = metile.ThreadLayout.identity(64)
    with pytest.raises(LoweringError, match="owner mapping"):
        validate_layout_exchanges(function)


@pytest.mark.parametrize(
    "corruption",
    ["size", "dtype", "load_mask", "store_mask", "condition", "scope", "flags", "geometry"],
)
def test_exchange_validation_rejects_unsafe_materialized_protocols(corruption):
    function = lower(
        _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)))
    )
    allocation = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    )
    load = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupLoad)
    )
    store = next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupStore)
    )
    publish = next(operation for operation in function.ops if isinstance(operation, mir.MBarrier))
    if corruption == "size":
        allocation.size -= 1
    elif corruption == "dtype":
        allocation.elem_type = "half"
    elif corruption == "load_mask":
        load.mask = mir.MValue("mask", ScalarType("bool"))
    elif corruption == "store_mask":
        store.mask = mir.MValue("mask", ScalarType("bool"))
    elif corruption == "condition":
        publish.condition = "lid == 0"
    elif corruption == "scope":
        publish.kind = "simdgroup"
    elif corruption == "flags":
        publish.flags = "mem_none"
    elif corruption == "geometry":
        function.threadgroup_size = (32, 1, 1)
    with pytest.raises(LoweringError, match=r"violates its contract|publish and recycle"):
        validate_layout_exchanges(function)


def test_exchange_survives_constant_folding_and_instruction_scheduling():
    from metile.compiler.passes import fold_constants
    from metile.compiler.scheduling import reorder_for_latency

    function = lower(
        _trace(metile.ThreadLayout.identity(64), metile.ThreadLayout((3, 4, 5, 0, 1, 2)), True)
    )
    function = reorder_for_latency(fold_constants(function))
    validate_layout_exchanges(function)


@metile.kernel
def ownership_predicate(source, output, width, BLOCK: metile.constexpr, TARGET: metile.constexpr):
    inputs = metile.tensor(source, shape=(width,), access="read")
    outputs = metile.tensor(output, shape=(width,), access="write")
    positions = metile.arange(0, BLOCK)
    predicate = inputs.load((positions,)) > 0.25
    converted = metile.convert_layout(predicate, TARGET)
    target_positions = metile.arange(0, BLOCK, layout=TARGET)
    outputs.store((target_positions,), metile.where(converted, 1.0, -1.0))


@pytest.mark.parametrize("size", [32, 64])
def test_gpu_boolean_ownership_conversion_keeps_predication_convergent(size):
    source = np.linspace(-1, 1, size - 3, dtype=np.float32)
    output = np.zeros_like(source)
    layout = metile.ThreadLayout(tuple(reversed(metile.ThreadLayout.identity(size).bit_order)))
    dispatch = ownership_predicate[(1,)].prepare(
        source, output, source.size, BLOCK=size, TARGET=layout
    )
    np.testing.assert_array_equal(output, np.where(source > 0.25, 1.0, -1.0))
    if size == 64:
        assert dispatch.execution_report.allocations[0].bytes == 64


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.uint32])
@pytest.mark.parametrize("size", [32, 64, 256, 512, 1024])
def test_gpu_ownership_conversion_preserves_values_and_ragged_bounds(dtype, size):
    identity = metile.ThreadLayout.identity(size)
    target = metile.ThreadLayout(tuple(reversed(identity.bit_order)), xor_mask=3)
    values = np.arange(size * 3 - 7, dtype=dtype)
    output = np.zeros_like(values)
    dispatch = ownership_copy[(metile.cdiv(values.size, size),)].prepare(
        values, output, values.size, BLOCK=size, SOURCE=identity, TARGET=target, ROUNDTRIP=False
    )
    np.testing.assert_array_equal(output, values)
    report = dispatch.execution_report
    assert report.value_layouts
    assert report.layout_conversions[0].mechanism == (
        "simd_shuffle" if size == 32 else "threadgroup"
    )
    assert report.plan.threadgroup_size == (size, 1, 1)


def test_gpu_roundtrip_from_nonidentity_layout_reuses_storage_safely():
    source = metile.ThreadLayout((4, 0, 5, 2, 1, 3), xor_mask=17)
    target = metile.ThreadLayout.identity(64)
    values = np.arange(123, dtype=np.float32) / 7
    output = np.zeros_like(values)
    dispatch = ownership_copy[(2,)].prepare(
        values, output, values.size, BLOCK=64, SOURCE=source, TARGET=target, ROUNDTRIP=True
    )
    np.testing.assert_array_equal(output, values)
    assert len(dispatch.execution_report.allocations) == 1
    assert dispatch.execution_report.allocations[0].bytes == 256
