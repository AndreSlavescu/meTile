import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import fold_constants
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType


@metile.kernel
def _loop_copy(
    Source,
    Control,
    Output,
    Counts,
    *,
    CASE: metile.constexpr = "eligible",
    DTYPE: metile.constexpr = "f32",
    STEP: metile.constexpr = 8,
    OFFSET: metile.constexpr = 0,
    BLOCK: metile.constexpr = 32,
):
    scratch = metile.tensor(metile.shared(BLOCK * 2, dtype=DTYPE), shape=(BLOCK // 4, 8))
    control = metile.tensor(Control, shape=(2,), access="read")
    limit = control.load((0,))
    end = metile.cast(limit, "u32") if CASE == "unsigned_end" else limit
    source_rows = control.load((1,)) if CASE == "unrelated_extent" else limit
    output_rows = (
        control.load((1,))
        if CASE == "unrelated_output_extent"
        else 8
        if CASE == "fragment_state"
        else limit
    )
    source = metile.tensor(
        Source + 8,
        shape=(source_rows, 8),
        strides=(8, 1),
        block_shape=(8, 8),
        access="read",
    )
    output = metile.tensor(
        Output + 8,
        shape=(output_rows, 8),
        strides=(8, 1),
        block_shape=(8, 8),
        access="write",
    )
    counts = metile.tensor(Counts, shape=(1,), access="write")
    count = metile.loop_state(metile.cast(0, "i32"))
    if CASE == "fragment_state":
        accumulated = metile.loop_state(metile.zeros((8, 8), dtype="f32"))
    start = 1 if CASE == "nonzero_start" else 0
    for index in metile.tile_range(start, end, STEP):
        if CASE == "negative_offset":
            row = index - 1
        elif CASE == "unsigned_origin":
            row = metile.cast(index, "u32")
        elif CASE == "cancellation_overflow":
            row = (index + 1073741824) - 1073741824
        elif CASE == "dynamic_offset":
            row = index + control.load((1,))
        elif CASE == "commuted_offset":
            row = OFFSET + index
        elif CASE == "chained_offset":
            row = index + OFFSET // 2 + (OFFSET - OFFSET // 2)
        else:
            row = index + OFFSET
        column = 1 if CASE == "incomplete_static_axis" else 0
        if CASE == "nested_loop":
            for _inner in metile.tile_range(0, 1, 1):
                fragment = source.load((row, column), scratch=scratch)
                output.store((row, column), fragment + 1.0, scratch=scratch)
        else:
            fragment = source.load((row, column), scratch=scratch)
            if CASE == "fragment_state":
                accumulated.update(accumulated.value + metile.cast(fragment, "f32"))
            else:
                output.store((row, column), fragment + 1.0, scratch=scratch)
        count.update(count.value + 1)
    if CASE == "fragment_state":
        output.store((0, 0), accumulated.value, scratch=scratch)
    counts.store((metile.thread_id(),), count.value)


def _trace(dtype="f32", **constants):
    with TracingContext(_loop_copy.name) as context:
        context.func.constexprs.update(
            SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
            STRICT_MATH=True,
            BLOCK=32,
            DTYPE=dtype,
            **constants,
        )
        context.func.params = [
            tir.Param(name, PtrType("i32" if name in {"Control", "Counts"} else dtype))
            for name in ("Source", "Control", "Output", "Counts")
        ]
        arguments = [
            TracingProxy(tir.Value(parameter.name, parameter.type))
            for parameter in context.func.params
        ]
        _loop_copy.fn(*arguments, DTYPE=dtype, BLOCK=32, **constants)
    _mark_outputs(context.func)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def _fragments(operations):
    return [
        operation
        for operation in _walk(operations)
        if isinstance(operation, (mir.MFragmentLoad, mir.MFragmentStore))
    ]


def _evaluate_bound(value, dynamic_end, limit):
    if value is dynamic_end:
        return limit
    operation = value.defining_op
    if isinstance(operation, mir.MConstant):
        return operation.value
    assert isinstance(operation, mir.MBinOp)
    left = _evaluate_bound(operation.lhs, dynamic_end, limit)
    right = _evaluate_bound(operation.rhs, dynamic_end, limit)
    if operation.op == "max":
        return max(left, right)
    if operation.op == "div":
        return left // right
    assert operation.op == "mul"
    return left * right


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize(
    "case,step,offset",
    [
        ("eligible", 8, 0),
        ("eligible", 16, 8),
        ("eligible", 32, 24),
        ("commuted_offset", 16, 8),
        ("chained_offset", 32, 24),
    ],
)
def test_proven_device_loops_have_unguarded_full_tiles_and_guarded_tail(dtype, case, step, offset):
    metal = lower(_trace(dtype, CASE=case, STEP=step, OFFSET=offset))
    loops = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    assert len(loops) == 2
    complete, tail = loops
    assert complete.iv_name == tail.iv_name
    assert complete.step == tail.step == step
    assert complete.end is tail.start
    assert len(_fragments(complete.body)) == len(_fragments(tail.body)) == 2
    assert all(operation.full_tile for operation in _fragments(complete.body))
    assert not any(operation.full_tile for operation in _fragments(tail.body))
    assert [type(operation) for operation in complete.body] == [
        type(operation) for operation in tail.body
    ]
    source = emit(metal)
    first_loop, tail_loop = source.split(f"for (int {complete.iv_name} =", 2)[1:]
    assert "tile_scratch" not in first_loop
    assert "simdgroup_barrier(" not in first_loop
    assert "tile_scratch" in tail_loop
    assert "simdgroup_barrier(" in tail_loop


@pytest.mark.parametrize("step", [8, 16, 32, 2**31 - 1])
def test_full_tile_bound_is_nonnegative_rounded_down_and_does_not_overflow(step):
    metal = lower(_trace(STEP=step))
    complete, tail = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    for limit in (-(2**31), -1, 0, 1, 7, 8, 9, 15, 16, 17, 33, 2**31 - 1):
        actual = _evaluate_bound(complete.end, tail.end, limit)
        assert actual == max(limit, 0) // step * step
        assert 0 <= actual <= 2**31 - 1
        if limit <= 33:
            assert [*range(0, actual, step), *range(actual, limit, step)] == list(
                range(0, limit, step)
            )


def test_loop_peeling_keeps_one_counter_state_across_full_iterations_and_tail():
    metal = lower(_trace())
    states = [operation for operation in metal.ops if isinstance(operation, mir.MVarDecl)]
    loops = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    assert len(states) == 1
    assignments = [
        [operation for operation in loop.body if isinstance(operation, mir.MVarAssign)]
        for loop in loops
    ]
    assert len(assignments) == 2 and all(len(operations) == 1 for operations in assignments)
    assert assignments[0][0] is not assignments[1][0]
    assert assignments[0][0].var_name == assignments[1][0].var_name == states[0].var_name
    assert not any(isinstance(operation, mir.MVarDecl) for loop in loops for operation in loop.body)


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("case", ["eligible", "fragment_state"])
def test_peeled_loop_ssa_definitions_remain_owned_after_folding(dtype, case):
    metal = lower(_trace(dtype, CASE=case))
    loops = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    assert len(loops) == 2
    loads = [
        next(operation for operation in loop.body if isinstance(operation, mir.MFragmentLoad))
        for loop in loops
    ]
    assert loads[0] is not loads[1]
    assert loads[0].result is not loads[1].result
    assert all(operation.result.defining_op is operation for operation in loads)
    fold_constants(metal)
    for loop, original_load in zip(loops, loads, strict=True):
        assert any(operation is original_load for operation in loop.body)
        assert original_load.result.defining_op is original_load
    assert loads[0].full_tile is True and loads[1].full_tile is False
    assert "simdgroup_load(" in emit(metal)


def test_peeled_matrix_state_is_shared_by_both_loops_without_being_reinitialized():
    metal = lower(_trace(CASE="fragment_state"))
    states = [operation for operation in metal.ops if isinstance(operation, mir.MFragmentStateInit)]
    complete, tail = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    assert len(states) == 1
    for loop in (complete, tail):
        assert not any(isinstance(operation, mir.MFragmentStateInit) for operation in loop.body)
        reads = [
            operation for operation in loop.body if isinstance(operation, mir.MFragmentStateRead)
        ]
        writes = [
            operation for operation in loop.body if isinstance(operation, mir.MFragmentStateAssign)
        ]
        assert len(reads) == len(writes) == 1
        assert reads[0].state.name == writes[0].state_name == states[0].state_name


@pytest.mark.parametrize(
    "constants",
    [
        {"CASE": "negative_offset"},
        {"CASE": "unrelated_extent"},
        {"CASE": "unrelated_output_extent"},
        {"CASE": "unsigned_origin"},
        {"CASE": "unsigned_end"},
        {"CASE": "cancellation_overflow"},
        {"CASE": "nested_loop"},
        {"CASE": "nonzero_start"},
        {"CASE": "dynamic_offset"},
        {"CASE": "incomplete_static_axis"},
        {"STEP": 4},
        {"STEP": 8, "OFFSET": 1},
        {"STEP": 16, "OFFSET": 9},
        {"STEP": 2**31},
    ],
)
def test_unproven_device_loops_retain_all_fragment_guards(constants):
    metal = lower(_trace(**constants))
    loops = [operation for operation in metal.ops if isinstance(operation, mir.MForLoop)]
    assert len(loops) == 1
    assert not any(operation.full_tile for operation in _fragments(metal.ops))
    assert "tile_scratch" in emit(metal)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("limit", [0, 1, 7, 8, 9, 15, 16, 17, 33])
def test_gpu_peeled_copy_keeps_tail_guards_nan_padding_and_iteration_count(dtype, limit):
    source = np.full((42, 8), np.nan, dtype=dtype)
    source[1 : limit + 1] = np.arange(limit * 8, dtype=dtype).reshape(limit, 8) * 0.125
    original = source.copy()
    output = np.full_like(source, -119.0)
    counts = np.array([-1], dtype=np.int32)
    _loop_copy[(1,)].prepare(
        source,
        np.array([limit, limit], dtype=np.int32),
        output,
        counts,
        DTYPE="f16" if dtype == np.float16 else "f32",
        BLOCK=32,
        STRICT_MATH=True,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
    )
    expected = np.full_like(source, -119.0)
    expected[1 : limit + 1] = source[1 : limit + 1] + dtype(1.0)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(counts, [(limit + 7) // 8])


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("limit", [0, 7, 8, 9, 17, 33])
def test_gpu_peeled_matrix_state_continues_into_partial_tail(limit):
    source = np.full((42, 8), np.nan, dtype=np.float32)
    source[1 : limit + 1] = np.arange(limit * 8, dtype=np.float32).reshape(limit, 8) * 0.125
    output = np.full_like(source, -119.0)
    counts = np.array([-1], dtype=np.int32)
    _loop_copy[(1,)].prepare(
        source,
        np.array([limit, limit], dtype=np.int32),
        output,
        counts,
        CASE="fragment_state",
        DTYPE="f32",
        BLOCK=32,
        STRICT_MATH=True,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
    )
    expected = np.full_like(source, -119.0)
    accumulated = np.zeros((8, 8), dtype=np.float32)
    for start in range(0, limit, 8):
        valid = min(limit - start, 8)
        accumulated[:valid] += source[1 + start : 1 + start + valid]
    expected[1:9] = accumulated
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(counts, [(limit + 7) // 8])
