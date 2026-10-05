import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.passes import fold_constants
from metile.ir import metal_ir as mir
from metile.ir.types import PtrType


def _binary(function, operation, left, right, dtype):
    left_value = function.add_op(mir.MConstant(value=left, dtype=dtype))
    right_value = function.add_op(mir.MConstant(value=right, dtype=dtype))
    return function.add_op(mir.MBinOp(op=operation, lhs=left_value, rhs=right_value))


def _constant(value, dtype):
    operation = mir.resolve(value).defining_op
    assert isinstance(operation, mir.MConstant)
    assert operation.dtype == dtype
    return operation.value


@pytest.mark.parametrize("dtype,numpy_type", [("f16", np.float16), ("f32", np.float32)])
@pytest.mark.parametrize(
    "operation,left,right",
    [
        ("add", 0.1, 0.2),
        ("sub", 0.3, 0.1),
        ("mul", 0.1, 0.2),
        ("div", 0.1, 0.3),
        ("add", 1.0 + 2.0**-24, -1.0),
        ("sub", 1.0 + 2.0**-24, 1.0),
        ("mul", 1.0 + 2.0**-24, 1.0 + 2.0**-24),
        ("div", 1.0 + 2.0**-24, 0.75),
    ],
)
def test_float_binary_folding_rounds_operands_and_result_to_the_declared_type(
    dtype, numpy_type, operation, left, right
):
    function = mir.MFunction("typed_float_arithmetic")
    result = _binary(function, operation, left, right, dtype)
    reference_operation = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[
        operation
    ]
    expected = reference_operation(numpy_type(left), numpy_type(right), dtype=numpy_type)
    fold_constants(function)
    actual = _constant(result, dtype)
    assert actual == float(expected)
    assert numpy_type(actual).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "dtype,numpy_type,large", [("f16", np.float16, 2048.0), ("f32", np.float32, 16777216.0)]
)
def test_float_arithmetic_chains_round_each_intermediate(dtype, numpy_type, large):
    function = mir.MFunction("rounded_intermediate")
    large_value = function.add_op(mir.MConstant(value=large, dtype=dtype))
    one = function.add_op(mir.MConstant(value=1.0, dtype=dtype))
    intermediate = function.add_op(mir.MBinOp(op="add", lhs=large_value, rhs=one))
    result = function.add_op(mir.MBinOp(op="sub", lhs=intermediate, rhs=large_value))
    fold_constants(function)
    expected = numpy_type(numpy_type(numpy_type(large) + numpy_type(1.0)) - numpy_type(large))
    assert _constant(intermediate, dtype) == large
    assert _constant(result, dtype) == float(expected) == 0.0


@pytest.mark.parametrize(
    "source_value", [1.0003, 1.0 + 2.0**-11 + 2.0**-25, -1.0 - 2.0**-11 - 2.0**-25]
)
def test_chained_float_casts_round_source_and_each_destination(source_value):
    function = mir.MFunction("typed_cast_chain")
    source = function.add_op(mir.MConstant(value=source_value, dtype="f32"))
    narrow = function.add_op(mir.MCast(value=source, target_dtype="f16"))
    wide = function.add_op(mir.MCast(value=narrow, target_dtype="f32"))
    fold_constants(function)
    expected_half = np.float16(np.float32(source_value))
    expected_float = np.float32(expected_half)
    assert _constant(narrow, "f16") == float(expected_half)
    assert _constant(wide, "f32") == float(expected_float)
    assert np.float32(_constant(wide, "f32")).tobytes() == expected_float.tobytes()


@pytest.mark.parametrize("head_dimension", [96, 128, 160, 192])
def test_base_two_attention_scale_matches_native_float32_multiply_bitwise(head_dimension):
    function = mir.MFunction("attention_scale")
    inverse_root = head_dimension**-0.5
    log_two_e = float(np.log2(np.e))
    result = _binary(function, "mul", inverse_root, log_two_e, "f32")
    fold_constants(function)
    expected = np.float32(np.float32(inverse_root) * np.float32(log_two_e))
    actual = np.float32(_constant(result, "f32"))
    assert actual.tobytes() == expected.tobytes()
    if head_dimension in {160, 192}:
        assert np.float32(inverse_root * log_two_e).tobytes() != expected.tobytes()


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_fused_operation_is_not_replaced_with_separately_rounded_python_math(dtype):
    function = mir.MFunction("preserve_fma")
    delta = 2.0 ** (-6 if dtype == "f16" else -13)
    operands = [
        function.add_op(mir.MConstant(value=value, dtype=dtype))
        for value in (1.0 + delta, 1.0 - delta, -1.0)
    ]
    operation = mir.MFma(left=operands[0], right=operands[1], addend=operands[2])
    result = function.add_op(operation)
    fold_constants(function)
    assert mir.resolve(result).defining_op is operation
    assert operation in function.ops
    assert "fma(" in emit(function)


@pytest.mark.parametrize(
    "dtype,operation,left,right",
    [
        ("bf16", "add", 0.1, 0.2),
        ("f32", "unsupported", 0.1, 0.2),
        ("f32", "mul", 3.0e38, 2.0),
        ("f16", "mul", 40000.0, 2.0),
        ("f32", "add", 1.0e40, 2.0),
        ("f16", "add", 100000.0, 2.0),
        ("f32", "div", 1.0, 0.0),
        ("f16", "div", 0.0, 0.0),
        ("f32", "mul", float("inf"), 0.0),
        ("f32", "add", float("-inf"), 0.0),
        ("f16", "mul", float("nan"), 1.0),
    ],
)
def test_unsupported_overflow_and_nonfinite_float_arithmetic_is_left_for_metal(
    dtype, operation, left, right
):
    function = mir.MFunction("deferred_float_operation")
    result = _binary(function, operation, left, right, dtype)
    original = result.defining_op
    fold_constants(function)
    assert mir.resolve(result).defining_op is original
    assert original in function.ops


@pytest.mark.parametrize(
    "source_dtype,destination_dtype,value",
    [
        ("f32", "f16", 100000.0),
        ("f32", "f16", 1.0e40),
        ("f16", "f32", 100000.0),
        ("f32", "f16", float("inf")),
        ("f16", "f32", float("nan")),
        ("bf16", "f32", 0.1),
        ("f32", "bf16", 0.1),
    ],
)
def test_unsupported_overflow_and_nonfinite_float_casts_are_not_folded(
    source_dtype, destination_dtype, value
):
    function = mir.MFunction("deferred_float_cast")
    source = function.add_op(mir.MConstant(value=value, dtype=source_dtype))
    operation = mir.MCast(value=source, target_dtype=destination_dtype)
    result = function.add_op(operation)
    fold_constants(function)
    assert mir.resolve(result).defining_op is operation
    assert operation in function.ops


def test_emitter_does_not_bypass_a_deferred_overflowing_float_cast():
    function = mir.MFunction("cast_kept_in_shader")
    function.params.append(mir.MParam("output", PtrType("f16"), is_output=True))
    output = mir.MValue("output", PtrType("f16"))
    source = function.add_op(mir.MConstant(value=100000.0, dtype="f32"))
    converted = function.add_op(mir.MCast(value=source, target_dtype="f16"), "converted")
    index = function.add_op(mir.MConstant(value=0, dtype="u32"))
    function.add_op(mir.DeviceStore(ptr=output, index=index, value=converted))
    fold_constants(function)
    emitted = emit(function)
    assert "half converted = static_cast<half>(100000.0f);" in emitted
    assert "output[0u] = converted;" in emitted


@pytest.mark.parametrize(
    "operation,left,right,expected",
    [
        ("add", 17, 4, 21),
        ("sub", 17, 4, 13),
        ("mul", 17, 4, 68),
        ("div", 17, 4, 4),
        ("mod", 17, 4, 1),
        ("and", 17, 4, 0),
        ("or", 17, 4, 21),
        ("xor", 17, 4, 21),
        ("shl", 17, 2, 68),
        ("shr", 17, 2, 4),
    ],
)
@pytest.mark.parametrize("dtype", ["i32", "u32"])
def test_integer_binary_constant_folding_is_unchanged(dtype, operation, left, right, expected):
    function = mir.MFunction("integer_arithmetic")
    result = _binary(function, operation, left, right, dtype)
    fold_constants(function)
    assert _constant(result, dtype) == expected


def test_integer_constant_cast_is_unchanged():
    function = mir.MFunction("integer_cast")
    source = function.add_op(mir.MConstant(value=17, dtype="i32"))
    result = function.add_op(mir.MCast(value=source, target_dtype="u32"))
    fold_constants(function)
    assert _constant(result, "u32") == 17
