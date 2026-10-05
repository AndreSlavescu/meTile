import operator

import numpy as np
import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType, TileType


def _input(context, name, datatype):
    context.func.params.append(tir.Param(name, datatype))
    return TracingProxy(tir.Value(name, datatype))


def _evaluate(proxy, arguments):
    def evaluate(value):
        if value.name in arguments:
            return arguments[value.name]
        operation = value.defining_op
        if isinstance(operation, tir.Constant):
            return operation.value
        if isinstance(operation, tir.BinOp):
            return {"add": operator.add, "mul": operator.mul}[operation.op](
                evaluate(operation.lhs), evaluate(operation.rhs)
            )
        if isinstance(operation, tir.Fma):
            return evaluate(operation.left) * evaluate(operation.right) + evaluate(operation.addend)
        if isinstance(operation, tir.Compare) and operation.predicate == "eq":
            return evaluate(operation.lhs) == evaluate(operation.rhs)
        if isinstance(operation, tir.Select):
            return np.where(
                evaluate(operation.condition),
                evaluate(operation.true_val),
                evaluate(operation.false_val),
            )
        if isinstance(operation, tir.Reduce) and operation.op == "sum":
            return np.sum(evaluate(operation.operand))
        if isinstance(operation, tir.Cast):
            return np.asarray(
                evaluate(operation.value),
                dtype={"f16": np.float16, "f32": np.float32}[operation.dtype],
            )
        if isinstance(operation, tir.Unary) and operation.op == "fast_exp2":
            return np.exp2(evaluate(operation.operand))
        raise AssertionError(f"Unexpected operation {operation}")

    return evaluate(proxy._value)


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("shape", [(), (37,), (8, 8)])
def test_fma_traces_one_fused_operation_and_preserves_type(dtype, shape):
    datatype = TileType(shape, dtype) if shape else ScalarType(dtype)
    with TracingContext("fused") as context:
        operands = [_input(context, name, datatype) for name in ("left", "right", "addend")]
        result = metile.fma(*operands)
    assert "fma" in metile.__all__
    assert result._value.type == datatype
    assert len(context.func.ops) == 1
    operation = context.func.ops[0]
    assert isinstance(operation, tir.Fma)
    assert operation.left is operands[0]._value
    assert operation.right is operands[1]._value
    assert operation.addend is operands[2]._value


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("scalar_index", [0, 1, 2])
def test_fma_broadcasts_scalar_proxy_in_every_operand_position(dtype, scalar_index):
    tile_type = TileType((8, 8), dtype)
    with TracingContext("broadcast") as context:
        operands = [
            _input(context, name, ScalarType(dtype) if index == scalar_index else tile_type)
            for index, name in enumerate(("left", "right", "addend"))
        ]
        result = metile.fma(*operands)
    assert result._value.type == tile_type


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("proxy_index", [0, 1, 2])
def test_fma_python_literals_adopt_the_floating_proxy_dtype(dtype, proxy_index):
    datatype = TileType((32,), dtype)
    with TracingContext("literal_broadcast") as context:
        operands = [2, 0.25, -1]
        operands[proxy_index] = _input(context, "source", datatype)
        result = metile.fma(*operands)
    assert result._value.type == datatype
    constants = [operation for operation in context.func.ops if isinstance(operation, tir.Constant)]
    assert len(constants) == 2
    assert all(operation.dtype == dtype for operation in constants)


def test_fma_all_literal_inputs_default_to_f32_without_unfused_constant_folding():
    with TracingContext("constants") as context:
        output = metile.fma(2, 0.25, -1)
    assert output._value.type == ScalarType("f32")
    assert isinstance(context.func.ops[-1], tir.Fma)


@pytest.mark.parametrize("operand_index", [0, 1, 2])
@pytest.mark.parametrize(
    "datatype,error,message",
    [
        (ScalarType("f16"), TypeError, "matching floating dtypes"),
        (ScalarType("i32"), TypeError, "f16 or f32"),
        (ScalarType("bool"), TypeError, "f16 or f32"),
        (PtrType("f32"), TypeError, "f16 or f32"),
        (TileType((4, 8), "f32"), ValueError, "matching tile shapes"),
    ],
)
def test_fma_rejects_invalid_proxy_types_and_shapes(operand_index, datatype, error, message):
    with TracingContext("invalid") as context:
        valid = _input(context, "source", TileType((32,), "f32"))
        invalid = _input(context, "invalid", datatype)
        operands = [valid, valid, valid]
        operands[operand_index] = invalid
        with pytest.raises(error, match=message):
            metile.fma(*operands)


@pytest.mark.parametrize("invalid", [True, False, None, 1j, "2", [1.0]])
def test_fma_rejects_non_numeric_and_boolean_literals(invalid):
    with TracingContext("invalid_literal") as context:
        source = _input(context, "source", ScalarType("f32"))
        with pytest.raises(TypeError, match="floating proxies or real scalar literals"):
            metile.fma(source, invalid, 0.0)


def test_fma_rejects_values_from_another_trace():
    with TracingContext("first") as context:
        foreign = _input(context, "foreign", ScalarType("f32"))
    with TracingContext("second"), pytest.raises(ValueError, match="cross tracing contexts"):
        metile.fma(foreign, 1.0, 0.0)


@pytest.mark.parametrize("scalar_index", [None, 0, 1, 2])
def test_fma_vjp_matches_all_three_analytic_derivatives_and_scalar_broadcast(scalar_index):
    names = ("left", "right", "addend")
    arguments = {
        "left": np.array([0.25, -2.0, 0.5, 1.25]),
        "right": np.array([-0.75, 1.0, 2.0, 0.5]),
        "addend": np.array([1.0, 0.25, -0.5, 2.0]),
        "seed": np.array([0.5, 2.0, -1.0, 0.25]),
    }
    if scalar_index is not None:
        arguments[names[scalar_index]] = 0.75
    with TracingContext("fused_vjp") as context:
        operands = [
            _input(
                context, name, ScalarType("f32") if index == scalar_index else TileType((4,), "f32")
            )
            for index, name in enumerate(names)
        ]
        seed = _input(context, "seed", TileType((4,), "f32"))
        gradients = metile.vjp(metile.fma(*operands), tuple(operands), seed)
    expected = (
        arguments["seed"] * arguments["right"],
        arguments["seed"] * arguments["left"],
        arguments["seed"],
    )
    for index, (gradient, reference) in enumerate(zip(gradients, expected)):
        assert gradient._value.type == operands[index]._value.type
        if index == scalar_index:
            reference = np.sum(reference)
        np.testing.assert_allclose(_evaluate(gradient, arguments), reference, rtol=1e-7, atol=0)


def test_fma_vjp_accumulates_repeated_operand_derivatives():
    with TracingContext("repeated") as context:
        source = _input(context, "source", TileType((4,), "f32"))
        gradient = metile.vjp(metile.fma(source, source, source), source, 2.0)
    values = np.array([-1.0, 0.0, 0.5, 2.0])
    np.testing.assert_array_equal(
        _evaluate(gradient, {"source": values}), 2.0 * (2.0 * values + 1.0)
    )


def test_fast_exp2_vjp_uses_log_two_times_the_primal_without_recomputation():
    with TracingContext("fast_exp2_vjp") as context:
        source = _input(context, "source", TileType((4,), "f32"))
        gradient = metile.vjp(metile.fast_exp2(source), source, 2.0)
    values = np.array([-1.0, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(
        _evaluate(gradient, {"source": values}), 2.0 * np.log(2.0) * np.exp2(values)
    )
    assert (
        sum(
            isinstance(operation, tir.Unary) and operation.op == "fast_exp2"
            for operation in context.func.ops
        )
        == 1
    )


@pytest.mark.parametrize("datatype", [ScalarType("i32"), ScalarType("bool"), PtrType("f32")])
def test_fast_exp2_rejects_non_floating_operands(datatype):
    with TracingContext("invalid_fast_exp2") as context:
        source = _input(context, "source", datatype)
        with pytest.raises(TypeError, match="f16 or f32"):
            metile.fast_exp2(source)
