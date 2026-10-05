import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.execution_report import validate_materialized_schedule
from metile.compiler.lowering import LoweringError, lower
from metile.compiler.passes import fold_constants
from metile.compiler.scheduling import reorder_for_latency
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def _two_products(
    Left, Right, Third, Output, *, DTYPE: metile.constexpr, BLOCK: metile.constexpr = 64
):
    left_memory = metile.shared(256, dtype=DTYPE)
    right_memory = metile.shared(256, dtype=DTYPE)
    third_memory = metile.shared(256, dtype=DTYPE)
    middle_memory = metile.shared(256, dtype=DTYPE)
    left = metile.tensor(Left, shape=(16, 16), access="read")
    right = metile.tensor(Right, shape=(16, 16), access="read")
    third = metile.tensor(Third, shape=(16, 16), access="read")
    output = metile.tensor(Output, shape=(16, 16), access="write")
    left_values = metile.tensor(left_memory, shape=(16, 16))
    right_values = metile.tensor(right_memory, shape=(16, 16))
    third_values = metile.tensor(third_memory, shape=(16, 16))
    middle_values = metile.tensor(middle_memory, shape=(16, 16))
    left_matrix = metile.tensor(left_memory, shape=(16, 16), block_shape=(8, 8))
    right_matrix = metile.tensor(right_memory, shape=(16, 16), strides=(1, 16), block_shape=(8, 8))
    third_matrix = metile.tensor(third_memory, shape=(16, 16), block_shape=(8, 8))
    middle_matrix = metile.tensor(middle_memory, shape=(16, 16), block_shape=(8, 8))
    thread = metile.thread_id()
    row = (thread // 32) * 8
    for index in metile.tile_range(thread, 256, BLOCK):
        position = (index // 16, index % 16)
        left_values.store(position, left.load(position))
        right_values.store(position, right.load(position))
        third_values.store(position, third.load(position))
    metile.barrier()
    for column in range(0, 16, 8):
        accumulator = metile.loop_state(metile.zeros((8, 8), dtype="f32"))
        for feature in metile.tile_range(0, 16, 8):
            accumulator.update(
                metile.dot(
                    left_matrix.load((row, feature)),
                    right_matrix.load((feature, column)),
                    accumulator.value,
                )
            )
        middle_matrix.store((row, column), accumulator.value * 0.5 + 0.125)
    metile.barrier()
    for index in metile.tile_range(thread, 256, BLOCK):
        position = (index // 16, index % 16)
        middle_values.store(position, metile.tanh(metile.cast(middle_values.load(position), "f32")))
    metile.barrier()
    for column in range(0, 16, 8):
        accumulator = metile.loop_state(metile.zeros((8, 8), dtype="f32"))
        for feature in metile.tile_range(0, 16, 8):
            accumulator.update(
                metile.dot(
                    middle_matrix.load((row, feature)),
                    third_matrix.load((feature, column)),
                    accumulator.value,
                )
            )
        left_matrix.store((row, column), accumulator.value)
    metile.barrier()
    for index in metile.tile_range(thread, 256, BLOCK):
        position = (index // 16, index % 16)
        output.store(position, left_values.load(position))


@metile.kernel
def _contract_probe(Output, Limit, *, CASE: metile.constexpr, BLOCK: metile.constexpr = 64):
    allocation = metile.shared(256, dtype="f32")
    values = metile.tensor(allocation, shape=(16, 16))
    rows = 12 if CASE == "varying_start" else 24 if CASE == "allocation_span" else 16
    matrix = metile.tensor(
        Output if CASE == "device_tile" else allocation,
        shape=(rows, 16),
        strides=(2, 3) if CASE == "bad_stride" else None,
        block_shape=(4, 8) if CASE == "bad_shape" else (8, 8),
    )
    destination = metile.tensor(Output, shape=(16, 16), access="write")
    empty = metile.tensor(Output, shape=(0,), access="read")
    thread = metile.thread_id()
    for index in metile.tile_range(thread, 256, BLOCK):
        values.store((index // 16, index % 16), 1.0)
    if CASE not in {"missing_publish", "zero_publish", "scalar_matrix_write"}:
        metile.barrier()
    if CASE == "zero_publish":
        for _index in metile.tile_range(0, Limit, 1):
            metile.barrier()
    origin = thread if CASE == "divergent_origin" else (thread // 32) * 8
    if CASE == "tail":
        origin = 9
    if CASE == "overflow_origin":
        origin = ((thread // 32) * 1073741824) * 8 // 1073741824
    if CASE == "divergent_loop":
        for _index in metile.tile_range(thread, 64, 1):
            matrix.load((0, 0))
    elif CASE == "varying_start":
        for index in metile.tile_range(thread // 32, 8, 4):
            matrix.load((index, 0))
    elif CASE == "divergent_barrier":
        matrix.load((0, 0))
        for _index in metile.tile_range(thread // 32, 2, 1):
            metile.barrier()
    elif CASE == "divergent_fill":
        matrix.load((0, 0))
        limit = metile.cast(empty.load((0,), other=metile.simd_lane_id()), "i32")
        for _index in metile.tile_range(0, limit, 1):
            metile.barrier()
    elif CASE == "missing_consume":
        matrix.store((origin, 0), metile.zeros((8, 8), dtype="f32"))
        destination.store((0, 0), values.load((0, 0)))
    elif CASE == "zero_consume":
        matrix.store((origin, 0), metile.zeros((8, 8), dtype="f32"))
        for _index in metile.tile_range(0, Limit, 1):
            metile.barrier()
        destination.store((0, 0), values.load((0, 0)))
    elif CASE == "matrix_scalar_write":
        matrix.store((origin, 0), metile.zeros((8, 8), dtype="f32"))
        values.store((0, thread), 2.0)
    elif CASE == "scalar_matrix_write":
        matrix.store((origin, 0), metile.zeros((8, 8), dtype="f32"))
    elif CASE == "zero_recycle":
        matrix.load((origin, 0))
        for _index in metile.tile_range(0, Limit, 1):
            metile.barrier()
        values.store((0, thread), 2.0)
    elif CASE == "missing_recycle":
        matrix.load((origin, 0))
        values.store((0, thread), 2.0)
    elif CASE == "backedge":
        for _index in metile.tile_range(0, 2, 1):
            values.store((0, thread), 2.0)
            metile.barrier()
            matrix.load((origin, 0))
    else:
        fragment = matrix.load((origin, 0))
        if CASE == "varying_scalar":
            fragment = fragment * metile.cast(metile.simd_lane_id(), "f32")
        if CASE == "reduce":
            destination.store((0, 0), metile.sum(fragment))
        if CASE == "mixed_dot":
            metile.dot(fragment, metile.cast(fragment, "f16"), metile.zeros((8, 8)))
        if CASE == "half_accumulator":
            metile.dot(fragment, fragment, metile.zeros((8, 8), dtype="f16"))
        if CASE == "divergent_assignment":
            state = metile.loop_state(fragment)
            for _index in metile.tile_range(thread, 64, 1):
                state.update(fragment)


def _trace(kernel=_two_products, dtype="f32", **constants):
    parameters = (
        ["Left", "Right", "Third", "Output"] if kernel is _two_products else ["Output", "Limit"]
    )
    with TracingContext(kernel.name) as context:
        context.func.constexprs.update(
            SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
            BLOCK=64,
            STRICT_MATH=True,
            **constants,
        )
        context.func.params = [
            tir.Param(name, I32 if name == "Limit" else PtrType(dtype)) for name in parameters
        ]
        arguments = [
            TracingProxy(tir.Value(parameter.name, parameter.type))
            for parameter in context.func.params
        ]
        if kernel is _two_products:
            kernel.fn(*arguments, DTYPE=dtype, BLOCK=64)
        else:
            kernel.fn(*arguments, **constants, BLOCK=64)
    _mark_outputs(context.func)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_composable_products_keep_typed_fragments_shared_bridge_and_mutable_state(dtype):
    metal = lower(_trace(dtype=dtype))
    operations = tuple(_walk(metal.ops))
    assert metal.kernel_type == "simdgroup_inline"
    assert metal.threadgroup_size == (64, 1, 1)
    assert sum(isinstance(operation, mir.MFragmentDot) for operation in operations) == 4
    assert sum(isinstance(operation, mir.MFragmentStateAssign) for operation in operations) == 4
    assert any(
        isinstance(operation, mir.MFragmentLoad) and operation.transpose for operation in operations
    )
    assert all(
        operation.accumulator.type.dtype == "f32"
        for operation in operations
        if isinstance(operation, mir.MFragmentDot)
    )
    source = emit(metal)
    assert "#include <metal_simdgroup_matrix>" in source
    assert "max_total_threads_per_threadgroup(64)" in source
    assert "simdgroup_multiply_accumulate" in source
    assert "simdgroup_matrix<float, 8, 8>" in source
    assert ("half" in source) == (dtype == "f16")
    assert "tanh(" in source
    validate_materialized_schedule(metal)


@metile.kernel
def _fragment_cast_probe(
    Output,
    Limit,
    *,
    SOURCE_DTYPE: metile.constexpr,
    DESTINATION_DTYPE: metile.constexpr,
    BLOCK: metile.constexpr = 64,
):
    source_memory = metile.shared(64, dtype=SOURCE_DTYPE)
    destination_memory = metile.shared(64, dtype=DESTINATION_DTYPE)
    source = metile.tensor(source_memory, shape=(8, 8), block_shape=(8, 8))
    destination = metile.tensor(destination_memory, shape=(8, 8), block_shape=(8, 8))
    destination.store((0, 0), metile.cast(source.load((0, 0)), DESTINATION_DTYPE))


@pytest.mark.parametrize("source_dtype,destination_dtype", [("f16", "f16"), ("f32", "f32")])
@pytest.mark.parametrize("optimize", [False, True])
def test_inline_same_dtype_cast_forwards_fragment_without_elementwise_copy(
    source_dtype, destination_dtype, optimize
):
    metal = lower(
        _trace(
            _fragment_cast_probe,
            SOURCE_DTYPE=source_dtype,
            DESTINATION_DTYPE=destination_dtype,
        )
    )
    if optimize:
        fold_constants(metal)
    operations = tuple(_walk(metal.ops))
    loads = [operation for operation in operations if isinstance(operation, mir.MFragmentLoad)]
    stores = [operation for operation in operations if isinstance(operation, mir.MFragmentStore)]
    assert len(loads) == len(stores) == 1
    assert stores[0].value is loads[0].result
    assert stores[0].value.type.dtype == source_dtype
    assert not any(isinstance(operation, mir.MFragmentElementwise) for operation in operations)
    assert ".thread_elements()" not in emit(metal)


@pytest.mark.parametrize("source_dtype,destination_dtype", [("f16", "f32"), ("f32", "f16")])
@pytest.mark.parametrize("optimize", [False, True])
def test_inline_dtype_conversion_preserves_explicit_fragment_cast(
    source_dtype, destination_dtype, optimize
):
    metal = lower(
        _trace(
            _fragment_cast_probe,
            SOURCE_DTYPE=source_dtype,
            DESTINATION_DTYPE=destination_dtype,
        )
    )
    if optimize:
        fold_constants(metal)
    operations = tuple(_walk(metal.ops))
    casts = [
        operation for operation in operations if isinstance(operation, mir.MFragmentElementwise)
    ]
    loads = [operation for operation in operations if isinstance(operation, mir.MFragmentLoad)]
    stores = [operation for operation in operations if isinstance(operation, mir.MFragmentStore)]
    assert len(casts) == len(loads) == len(stores) == 1
    assert casts[0].operation == "cast"
    assert casts[0].operands[0] is loads[0].result
    assert casts[0].operands[0].type.dtype == source_dtype
    assert casts[0].result.type.dtype == destination_dtype
    assert stores[0].value is casts[0].result
    assert ".thread_elements()" in emit(metal)


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_publish", "barrier before matrix loads"),
        ("zero_publish", "barrier before matrix loads"),
        ("divergent_origin", "SIMD-uniform"),
        ("tail", "in-bounds"),
        ("varying_start", "in-bounds"),
        ("overflow_origin", "in-bounds"),
        ("allocation_span", "exceeds its shared allocation"),
        ("device_tile", "explicit shared scratch"),
        ("bad_stride", "row-major"),
        ("bad_shape", "8x8"),
        ("divergent_loop", "SIMD-uniform control"),
        ("divergent_barrier", "threadgroup-uniform"),
        ("divergent_fill", "threadgroup-uniform"),
        ("missing_consume", "barrier before scalar"),
        ("zero_consume", "barrier before scalar"),
        ("matrix_scalar_write", "barrier before scalar shared overwrite"),
        ("scalar_matrix_write", "barrier before matrix stores"),
        ("zero_recycle", "barrier before shared overwrite"),
        ("missing_recycle", "barrier before shared overwrite"),
        ("backedge", "barrier before shared overwrite"),
        ("varying_scalar", "SIMD-uniform"),
        ("divergent_assignment", "SIMD-uniform control"),
        ("mixed_dot", "matching operand dtypes"),
        ("half_accumulator", "FP32 accumulator"),
        ("reduce", "shared-memory scalar bridge"),
    ],
)
def test_inline_matrix_contract_rejects_unsafe_memory_or_control(case, match):
    with pytest.raises(LoweringError, match=match):
        lower(_trace(_contract_probe, CASE=case))


@metile.kernel
def _shared_alias_probe(
    Output,
    Limit,
    *,
    CASE: metile.constexpr,
    BASE: metile.constexpr = 64,
    STRIDE: metile.constexpr = 8,
    TRANSPOSE: metile.constexpr = False,
    DTYPE: metile.constexpr = "f32",
    BLOCK: metile.constexpr = 64,
):
    allocation = metile.shared(256, dtype=DTYPE)
    left = metile.tensor(
        allocation,
        shape=(8, 8),
        strides=(1, STRIDE) if TRANSPOSE else (STRIDE, 1),
        block_shape=(8, 8),
    )
    right_pointer = allocation + (BASE // 2) + (BASE - BASE // 2)
    right = metile.tensor(right_pointer, shape=(8, 8), block_shape=(8, 8))
    values = metile.tensor(right_pointer, shape=(64,))
    left_values = metile.tensor(allocation, shape=(256,))
    output = metile.tensor(Output, shape=(64,), access="write")
    thread = metile.thread_id()
    zero = metile.zeros((8, 8), dtype=DTYPE)
    if CASE == "scalar_then_load":
        values.store((thread,), 1.0)
        left.load((0, 0))
    elif CASE == "load_then_scalar":
        left.load((0, 0))
        values.store((thread,), 1.0)
    elif CASE == "load_then_matrix":
        right.store((0, 0), left.load((0, 0)))
    elif CASE == "matrix_then_scalar_load":
        left.store((0, 0), zero)
        output.store((thread,), values.load((thread,)))
    elif CASE == "matrix_then_scalar_store":
        left.store((0, 0), zero)
        values.store((thread,), 1.0)
    elif CASE == "matrix_then_load":
        left.store((0, 0), zero)
        right.load((0, 0))
    elif CASE == "scalar_then_matrix":
        values.store((thread,), 1.0)
        left.store((0, 0), zero)
    elif CASE == "matrix_does_not_hide_scalar":
        left.store((0, 0), zero)
        values.store((thread,), 1.0)
        right.store((0, 0), zero)
    elif CASE in {
        "unknown_pointer",
        "unknown_index",
        "overflow_pointer",
        "unknown_mask",
        "cast_overflow",
        "chained_overflow",
        "float_roundtrip",
        "mixed_integer_offsets",
    }:
        left.load((0, 0))
        if CASE == "unknown_pointer":
            metile.store(allocation + Limit + BASE, 1.0)
        elif CASE == "unknown_index":
            values.store((Limit,), 1.0)
        elif CASE == "overflow_pointer":
            metile.store(allocation + ((thread // 32) * 1073741824) * 8 + BASE, 1.0)
        elif CASE == "cast_overflow":
            metile.store(allocation + metile.cast(-1, "u32") + BASE, 1.0)
        elif CASE == "chained_overflow":
            metile.store(allocation + 2147483647 + thread + (BASE - 2147483647), 1.0)
        elif CASE == "float_roundtrip":
            rounded = metile.cast(metile.cast(16777217, "f32"), "i32")
            metile.store(allocation + (rounded - 16777216) * BASE, 1.0)
        elif CASE == "mixed_integer_offsets":
            metile.store(allocation + BASE + metile.cast(thread, "u32"), 1.0)
        else:
            metile.store(allocation + Limit + BASE, 1.0, mask=Limit < 0)
    elif CASE == "backedge_disjoint":
        for _iteration in metile.tile_range(0, 2, 1):
            left.load((0, 0))
            values.store((thread,), 1.0)
    elif CASE == "backedge_overlap":
        for _iteration in metile.tile_range(0, 2, 1):
            values.store((thread,), 1.0)
            metile.barrier()
            right.load((0, 0))
    elif CASE == "zero_trip":
        right.load((0, 0))
        for _iteration in metile.tile_range(0, Limit, 1):
            metile.barrier()
        values.store((thread,), 1.0)
    elif CASE == "same_descriptor_disjoint":
        left_values.store((BASE + thread,), 1.0)
        left.load((0, 0))


_ALIAS_ORDERS = [
    ("scalar_then_load", "barrier before matrix loads"),
    ("load_then_scalar", "barrier before shared overwrite"),
    ("load_then_matrix", "barrier before shared overwrite"),
    ("matrix_then_scalar_load", "barrier before scalar shared loads"),
    ("matrix_then_scalar_store", "barrier before scalar shared overwrite"),
    ("matrix_then_load", "barrier before matrix loads"),
    ("scalar_then_matrix", "barrier before matrix stores"),
]


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("case,_match", _ALIAS_ORDERS)
def test_inline_matrix_disjoint_shared_aliases_do_not_need_a_barrier(dtype, case, _match):
    metal = lower(_trace(_shared_alias_probe, CASE=case, DTYPE=dtype, dtype=dtype))
    assert sum(isinstance(operation, mir.MThreadgroupAlloc) for operation in metal.ops) == 1
    assert "threadgroup_barrier(" not in emit(metal)


@pytest.mark.parametrize("case,match", _ALIAS_ORDERS)
def test_inline_matrix_one_element_shared_overlap_still_requires_a_barrier(case, match):
    with pytest.raises(LoweringError, match=match):
        lower(_trace(_shared_alias_probe, CASE=case, BASE=63))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("base", [64, 91, 92])
def test_inline_matrix_alias_ranges_include_padding_and_transposed_strides(transpose, base):
    function = _trace(
        _shared_alias_probe, CASE="load_then_scalar", BASE=base, STRIDE=12, TRANSPOSE=transpose
    )
    if base < 92:
        with pytest.raises(LoweringError, match="barrier before shared overwrite"):
            lower(function)
    else:
        lower(function)


@pytest.mark.parametrize(
    "case", ["backedge_disjoint", "same_descriptor_disjoint", "mixed_integer_offsets"]
)
def test_inline_matrix_access_ranges_allow_disjoint_loop_and_descriptor_aliases(case):
    lower(_trace(_shared_alias_probe, CASE=case))


@pytest.mark.parametrize(
    "case,match",
    [
        ("unknown_pointer", "barrier before shared overwrite"),
        ("unknown_index", "barrier before shared overwrite"),
        ("overflow_pointer", "barrier before shared overwrite"),
        ("unknown_mask", "barrier before shared overwrite"),
        ("cast_overflow", "barrier before shared overwrite"),
        ("chained_overflow", "barrier before shared overwrite"),
        ("float_roundtrip", "barrier before shared overwrite"),
        ("backedge_overlap", "barrier before shared overwrite"),
        ("zero_trip", "barrier before shared overwrite"),
        ("matrix_does_not_hide_scalar", "barrier before matrix stores"),
    ],
)
def test_inline_matrix_alias_uncertainty_and_loop_hazards_fail_closed(case, match):
    with pytest.raises(LoweringError, match=match):
        lower(_trace(_shared_alias_probe, CASE=case))


@metile.kernel
def _selected_shared_alias_probe(
    Output,
    Limit,
    *,
    SCRATCH: metile.constexpr,
    WRITE: metile.constexpr,
    OFFSET: metile.constexpr,
    UNIFORM: metile.constexpr,
    BLOCK: metile.constexpr = 64,
):
    allocation = metile.shared(128, dtype="f32")
    alternate = metile.shared(128, dtype="f32")
    scratch = metile.tensor(allocation, shape=(BLOCK // 4, 8))
    matrix = metile.tensor(allocation, shape=(16, 8), block_shape=(8, 8))
    output = metile.tensor(Output, shape=(9, 8), block_shape=(8, 8), access="write")
    condition = Limit > 0 if UNIFORM else metile.thread_id() == 0
    selected = metile.where(condition, allocation, alternate)
    if OFFSET:
        selected = selected + 1
    scalar = metile.tensor(selected, shape=(127,))
    if WRITE:
        scalar.store((0,), 1.0)
    else:
        scalar.load((0,))
    row = metile.thread_id() // 32 * 8
    if SCRATCH:
        output.store((row, 0), metile.zeros((8, 8)), scratch=scratch)
    else:
        matrix.load((row, 0))


@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_selected_shared_aliases_cannot_bypass_scratch_or_matrix_hazard_proofs(
    scratch, write, offset, uniform
):
    with pytest.raises(LoweringError, match="shared allocation through pointer offsets only"):
        lower(
            _trace(
                _selected_shared_alias_probe,
                SCRATCH=scratch,
                WRITE=write,
                OFFSET=offset,
                UNIFORM=uniform,
            )
        )


def test_optimization_preserves_fragment_memory_and_state_effect_order():
    metal = lower(_trace())
    effect_types = (
        mir.MFragmentLoad,
        mir.MFragmentStore,
        mir.MFragmentDot,
        mir.MFragmentStateInit,
        mir.MFragmentStateRead,
        mir.MFragmentStateAssign,
        mir.MBarrier,
    )
    before = [operation for operation in _walk(metal.ops) if isinstance(operation, effect_types)]
    fold_constants(metal)
    reorder_for_latency(metal)
    after = [operation for operation in _walk(metal.ops) if isinstance(operation, effect_types)]
    assert [id(operation) for operation in before] == [id(operation) for operation in after]
    source = emit(metal)
    assert source.count("simdgroup_multiply_accumulate(") == 4
    assert source.count("threadgroup_barrier(") == 4


@pytest.mark.parametrize(
    "settings,match",
    [
        (
            {"SCHEDULE": metile.Schedule(backend="simdgroup_inline", vector_width=4)},
            "forced vectorized",
        ),
        (
            {"SCHEDULE": metile.Schedule(backend="simdgroup_inline", double_buffer=True)},
            "buffering",
        ),
        ({"SCHEDULE": metile.Schedule(backend="simdgroup_inline", staging="device")}, "staging"),
        (
            {"SCHEDULE": metile.Schedule(backend="simdgroup_inline", num_simdgroups=4)},
            "SIMDgroup count",
        ),
        ({"WM": 2}, "explicitly declared"),
        ({"NAX_FRAGMENTS": True}, "explicitly declared"),
        ({"BLOCK": 48}, "multiple of 32"),
    ],
)
def test_inline_matrix_schedule_requirements_fail_closed(settings, match):
    function = _trace()
    function.constexprs.update(settings)
    with pytest.raises(LoweringError, match=match):
        lower(function)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_two_matrix_products_with_scalar_bridge_and_transpose(dtype):
    generator = np.random.default_rng(639)
    left, right, third = [generator.normal(0, 0.3, (16, 16)).astype(dtype) for index in range(3)]
    actual = np.empty_like(left)
    _two_products[(1,)].prepare(
        left,
        right,
        third,
        actual,
        DTYPE="f16" if dtype == np.float16 else "f32",
        BLOCK=64,
        STRICT_MATH=True,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
    )
    middle = (left.astype(np.float64) @ right.astype(np.float64).T * 0.5 + 0.125).astype(dtype)
    middle = np.tanh(middle.astype(np.float64)).astype(dtype)
    expected = (middle.astype(np.float64) @ third.astype(np.float64)).astype(dtype)
    tolerance = 2e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
