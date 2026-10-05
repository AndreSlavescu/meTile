import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.passes import _stable_val_key, fold_constants
from metile.ir import metal_ir as mir
from metile.ir.types import BOOL, I32, U32, MatrixFragmentType, PtrType, ScalarType


def _constant(function, value, dtype="i32"):
    return function.add_op(mir.MConstant(value=value, dtype=dtype))


def _compare(function, source, bound, predicate="lt"):
    return function.add_op(mir.MCompare(predicate=predicate, lhs=source, rhs=bound))


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("dtype,value", [("i32", 16), ("u32", 16), ("u8", 16), ("bool", True)])
def test_repeated_integer_literals_share_typed_comparisons(dtype, value):
    function = mir.MFunction("literal_cse")
    source = mir.MValue("input", ScalarType(dtype))
    first = _compare(function, source, _constant(function, value, dtype))
    second = _compare(function, source, _constant(function, value, dtype))
    fold_constants(function)
    assert mir.resolve(second) is first
    assert sum(isinstance(operation, mir.MCompare) for operation in function.ops) == 1


@pytest.mark.parametrize("dtype,delta", [("f16", 2.0**-12), ("f32", 2.0**-25)])
def test_float_literal_keys_use_storage_precision(dtype, delta):
    function = mir.MFunction("float_literal_cse")
    source = mir.MValue("input", ScalarType(dtype))
    first = _compare(function, source, _constant(function, 1.0, dtype))
    second = _compare(function, source, _constant(function, 1.0 + delta, dtype))
    fold_constants(function)
    assert mir.resolve(second) is first


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_signed_zero_literals_remain_distinct_in_selects(dtype):
    function = mir.MFunction("zero_bits")
    condition = mir.MValue("condition", BOOL)
    otherwise = _constant(function, 2.0, dtype)
    results = [
        function.add_op(
            mir.MSelect(
                condition=condition,
                true_val=_constant(function, zero, dtype),
                false_val=otherwise,
            )
        )
        for zero in (0.0, -0.0)
    ]
    fold_constants(function)
    assert mir.resolve(results[0]) is not mir.resolve(results[1])
    source = emit(function)
    assert "-0.0" in source


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_adjacent_float_storage_values_are_not_coalesced(dtype):
    numpy_type = np.float16 if dtype == "f16" else np.float32
    function = mir.MFunction("distinct_float_bits")
    source = mir.MValue("input", ScalarType(dtype))
    first = _compare(function, source, _constant(function, 1.0, dtype))
    successor = float(np.nextafter(numpy_type(1), numpy_type(2)))
    second = _compare(function, source, _constant(function, successor, dtype))
    fold_constants(function)
    assert mir.resolve(second) is not first


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("f16", float("inf")),
        ("f32", float("-inf")),
        ("f32", float("nan")),
        ("f16", 100000.0),
        ("f32", 1.0e40),
        ("f16", 2.0**-24),
        ("f32", 2.0**-149),
        ("bf16", 1.0),
        ("i32", 2**31),
        ("u32", -1),
        ("u8", 256),
        ("bool", 2),
    ],
)
def test_unsupported_or_exceptional_literals_keep_distinct_identities(dtype, value):
    function = mir.MFunction("conservative_literals")
    source = mir.MValue("input", ScalarType(dtype))
    first = _compare(function, source, _constant(function, value, dtype))
    second = _compare(function, source, _constant(function, value, dtype))
    fold_constants(function)
    assert mir.resolve(second) is not first


def test_equal_numeric_literals_with_different_types_have_different_keys():
    function = mir.MFunction("typed_literals")
    keys = {
        _stable_val_key(_constant(function, 1, dtype))
        for dtype in ("bool", "i32", "u32", "u8", "f16", "f32")
    }
    assert len(keys) == 6


def test_parameter_keys_include_the_complete_pointer_address_space():
    device = mir.MValue("pointer", PtrType("f32", "device"))
    shared = mir.MValue("pointer", PtrType("f32", "threadgroup"))
    assert _stable_val_key(device) != _stable_val_key(shared)


def test_forwarded_arithmetic_and_casts_feed_later_comparison_and_select_cse():
    function = mir.MFunction("forwarded_values")
    source = mir.MValue("input", I32)
    results = []
    for _iteration in range(2):
        product = function.add_op(mir.MBinOp(op="mul", lhs=source, rhs=_constant(function, 16)))
        converted = function.add_op(mir.MCast(value=product, target_dtype="u32"))
        condition = _compare(function, converted, _constant(function, 128, "u32"))
        results.append(
            function.add_op(
                mir.MSelect(
                    condition=condition,
                    true_val=converted,
                    false_val=_constant(function, 512, "u32"),
                )
            )
        )
    fold_constants(function)
    assert mir.resolve(results[1]) is results[0]
    for operation_type in (mir.MBinOp, mir.MCast, mir.MCompare, mir.MSelect):
        assert sum(isinstance(operation, operation_type) for operation in function.ops) == 1
    source = emit(function)
    fold_constants(function)
    assert emit(function) == source


def test_comparison_predicates_operand_order_and_select_branch_order_stay_distinct():
    function = mir.MFunction("operand_order")
    left, right = mir.MValue("left", I32), mir.MValue("right", I32)
    conditions = [
        _compare(function, left, right, predicate) for predicate in ("lt", "le", "gt", "ge")
    ]
    conditions.append(_compare(function, right, left))
    selected = [
        function.add_op(mir.MSelect(condition=conditions[0], true_val=first, false_val=second))
        for first, second in ((left, right), (right, left))
    ]
    fold_constants(function)
    assert len({id(mir.resolve(value)) for value in conditions + selected}) == 7


@pytest.mark.parametrize("kind", ["device", "shared"])
def test_distinct_load_snapshots_survive_stores_and_barriers(kind):
    function = mir.MFunction("snapshot_identity")
    pointer = mir.MValue("data", PtrType("f32"))
    index = _constant(function, 0)
    bound = _constant(function, 2.0, "f32")
    results = []
    memory_operations = []
    for _iteration in range(2):
        load = (
            mir.DeviceLoad(ptr=pointer, index=index)
            if kind == "device"
            else mir.MThreadgroupLoad(array_name="scratch", index=index)
        )
        value = function.add_op(load)
        results.append(_compare(function, value, bound))
        store = (
            mir.DeviceStore(ptr=pointer, index=index, value=value)
            if kind == "device"
            else mir.MThreadgroupStore(array_name="scratch", index=index, value=value)
        )
        function.add_op(store)
        barrier = mir.MBarrier()
        function.add_op(barrier)
        memory_operations.extend((load, store, barrier))
    fold_constants(function)
    assert mir.resolve(results[0]) is not mir.resolve(results[1])
    assert [
        id(operation)
        for operation in function.ops
        if isinstance(
            operation,
            (
                mir.DeviceLoad,
                mir.DeviceStore,
                mir.MThreadgroupLoad,
                mir.MThreadgroupStore,
                mir.MBarrier,
            ),
        )
    ] == [id(operation) for operation in memory_operations]


@pytest.mark.parametrize("assignment", ["scalar", "fragment"])
def test_mutable_assignment_invalidates_expression_cse(assignment):
    function = mir.MFunction("mutable_state")
    state = mir.MValue("state", I32)
    first = _compare(function, state, _constant(function, 16))
    if assignment == "scalar":
        function.add_op(mir.MVarAssign(var_name=state.name, value=_constant(function, 2)))
    else:
        fragment = mir.MValue("fragment", MatrixFragmentType("f32"))
        function.add_op(mir.MFragmentStateAssign(state_name=fragment.name, value=fragment))
    second = _compare(function, state, _constant(function, 16))
    fold_constants(function)
    assert mir.resolve(second) is not first


@pytest.mark.parametrize("kind", ["for", "while", "if", "role"])
def test_control_flow_cse_respects_dominance_and_scope(kind):
    function = mir.MFunction("scopes")
    source = mir.MValue("input", I32)
    first = _compare(function, source, _constant(function, 16))
    nested = mir.MFunction("nested")
    inside = _compare(nested, source, _constant(nested, 16))
    extra = _compare(nested, source, _constant(nested, 32))
    for operation in nested.ops:
        if operation.result is not None:
            operation.result.name = f"nested_{operation.result.name}"
    if kind == "for":
        block = mir.MForLoop(iv_name="index", start=0, end=2, step=1, body=nested.ops)
    elif kind == "while":
        block = mir.MWhileTrue(body=nested.ops)
    elif kind == "if":
        block = mir.IfBlock(condition=first, body=nested.ops)
    else:
        block = mir.MSimdgroupRoleBlock(sgid=mir.MValue("sgid", U32), body=nested.ops)
    function.add_op(block)
    after = _compare(function, source, _constant(function, 16))
    after_extra = _compare(function, source, _constant(function, 32))
    fold_constants(function)
    assert (mir.resolve(inside) is first) == (kind in {"if", "role"})
    assert mir.resolve(after) is not first
    assert mir.resolve(after_extra) is not extra


def test_select_result_dtype_is_part_of_the_cse_key():
    function = mir.MFunction("typed_result")
    condition = mir.MValue("condition", BOOL)
    source = mir.MValue("source", I32)
    first = function.add_op(mir.MSelect(condition=condition, true_val=source, false_val=source))
    second = function.add_op(mir.MSelect(condition=condition, true_val=source, false_val=source))
    second.type = U32
    fold_constants(function)
    assert mir.resolve(second) is not first


def test_cse_does_not_coalesce_explicit_fma_or_collectives():
    function = mir.MFunction("explicit_math")
    source = mir.MValue("input", ScalarType("f32"))
    operations = []
    for _iteration in range(2):
        fused = mir.MFma(left=source, right=source, addend=source)
        collective = mir.MUnary(op="simd_sum", operand=source)
        function.add_op(fused)
        function.add_op(collective)
        operations.extend((fused, collective))
    fold_constants(function)
    assert [id(operation) for operation in function.ops] == [
        id(operation) for operation in operations
    ]


def test_production_attention_cse_reduces_source_without_changing_memory_or_matrix_work():
    from metile.compiler.lowering import lower
    from tests.kernels.test_qwen3_prefill_matrix_attention import _trace

    function = lower(
        _trace(
            CHUNK=256,
            QUERY_HEADS=16,
            KV_HEADS=8,
            LAYERS=28,
            MAX_CONTEXT=5119,
            QUERY_TILE=32,
            KEY_TILE=16,
            SHARED_PADDING=0,
            UNROLL_MMA=True,
            SOFTMAX_LANES=8,
            TRANSPOSE_KEYS=True,
            REGISTER_STATS=True,
            UNROLL_SOFTMAX=True,
            LOAD_VECTOR=16,
            SOFTMAX_BASE2=True,
        )
    )
    effect_types = (
        mir.DeviceLoad,
        mir.DeviceStore,
        mir.MThreadgroupLoad,
        mir.MThreadgroupStore,
        mir.MFragmentLoad,
        mir.MFragmentStore,
        mir.MFragmentDot,
        mir.MFragmentStateAssign,
        mir.MBarrier,
        mir.MVarAssign,
        mir.MFma,
    )
    before = [operation for operation in _walk(function.ops) if isinstance(operation, effect_types)]
    comparisons_before = sum(
        isinstance(operation, mir.MCompare) for operation in _walk(function.ops)
    )
    fold_constants(function)
    after = [operation for operation in _walk(function.ops) if isinstance(operation, effect_types)]
    comparisons_after = sum(
        isinstance(operation, mir.MCompare) for operation in _walk(function.ops)
    )
    assert [id(operation) for operation in before] == [id(operation) for operation in after]
    assert comparisons_after < comparisons_before * 0.6
    assert len(emit(function)) < 120000
