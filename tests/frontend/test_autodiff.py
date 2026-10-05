import operator

import numpy as np
import pytest

import metile
from metile.frontend import tracing
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType, TileType


def _input(context, name="source", shape=(4,), dtype="f32"):
    datatype = TileType(shape, dtype) if shape else ScalarType(dtype)
    context.func.params.append(tir.Param(name, datatype))
    return TracingProxy(tir.Value(name, datatype))


def _evaluate(proxy, arguments):
    memo = {}
    binary = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "div": operator.truediv,
        "max": np.maximum,
        "min": np.minimum,
    }
    unary = {
        "exp": np.exp,
        "exp2": np.exp2,
        "fast_cos": np.cos,
        "fast_exp": np.exp,
        "fast_sin": np.sin,
        "log": np.log,
        "sqrt": np.sqrt,
        "rsqrt": lambda values: 1 / np.sqrt(values),
        "tanh": np.tanh,
        "abs": np.abs,
        "neg": np.negative,
    }
    compare = {
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    }

    def evaluate(value):
        if id(value) in memo:
            return memo[id(value)]
        if value.name in arguments:
            result = arguments[value.name]
        else:
            operation = value.defining_op
            if isinstance(operation, tir.Constant):
                result = operation.value
            elif isinstance(operation, tir.BinOp):
                result = binary[operation.op](evaluate(operation.lhs), evaluate(operation.rhs))
            elif isinstance(operation, tir.Unary):
                result = unary[operation.op](evaluate(operation.operand))
            elif isinstance(operation, tir.Compare):
                result = compare[operation.predicate](
                    evaluate(operation.lhs), evaluate(operation.rhs)
                )
            elif isinstance(operation, tir.Cast):
                result = np.asarray(
                    evaluate(operation.value),
                    dtype={"f32": np.float32, "f16": np.float16}[operation.dtype],
                )
            elif isinstance(operation, tir.Select):
                result = np.where(
                    evaluate(operation.condition),
                    evaluate(operation.true_val),
                    evaluate(operation.false_val),
                )
            elif isinstance(operation, tir.Reduce):
                result = {"sum": np.sum, "max": np.max, "min": np.min}[operation.op](
                    evaluate(operation.operand)
                )
            else:
                raise AssertionError(f"Unexpected operation {operation}")
        memo[id(value)] = result
        return result

    return evaluate(proxy._value)


@pytest.mark.parametrize(
    ("expression", "derivative"),
    [
        (lambda source: source + 2.0, lambda source: np.ones_like(source)),
        (lambda source: source - 2.0, lambda source: np.ones_like(source)),
        (lambda source: 2.0 - source, lambda source: -np.ones_like(source)),
        (lambda source: source * source, lambda source: 2 * source),
        (lambda source: source / 2.0, lambda source: np.full_like(source, 0.5)),
        (lambda source: 2.0 / source, lambda source: -2 / source**2),
        (metile.exp, np.exp),
        (metile.exp2, lambda source: np.log(2.0) * np.exp2(source)),
        (metile.fast_cos, lambda source: -np.sin(source)),
        (metile.fast_exp, np.exp),
        (metile.fast_sin, np.cos),
        (metile.log, lambda source: 1 / source),
        (metile.sqrt, lambda source: 0.5 / np.sqrt(source)),
        (metile.rsqrt, lambda source: -0.5 / source**1.5),
        (metile.tanh, lambda source: 1 - np.tanh(source) ** 2),
        (metile.abs, np.sign),
        (lambda source: tracing._unary("neg", source), lambda source: -np.ones_like(source)),
    ],
)
def test_elementwise_rules_match_analytic_derivatives(expression, derivative):
    values = np.array([0.3, 0.7, 1.2, 2.0])
    seed = np.array([1.0, -0.5, 2.0, 0.3])
    with TracingContext("rules") as context:
        source = _input(context)
        cotangent = _input(context, "seed")
        gradient = metile.vjp(expression(source), source, cotangent)
    actual = _evaluate(gradient, {"source": values, "seed": seed})
    np.testing.assert_allclose(actual, seed * derivative(values), rtol=1e-6, atol=1e-6)


def test_abs_zero_and_negative_derivatives():
    values = np.array([-2.0, -0.0, 0.0, 3.0])
    with TracingContext("abs") as context:
        source = _input(context)
        gradient = metile.vjp(metile.abs(source), source, 2.0)
    np.testing.assert_array_equal(_evaluate(gradient, {"source": values}), [-2, 0, 0, 2])


def test_shared_subexpressions_accumulate_and_scalar_broadcasts_reduce():
    values = np.array([0.2, 0.4, -0.3, 1.0])
    seed = np.array([1.0, 0.5, 2.0, -0.2])
    with TracingContext("shared") as context:
        source = _input(context)
        scale = _input(context, "scale", shape=())
        cotangent = _input(context, "seed")
        shared = source * scale
        output = shared * shared + metile.exp(shared)
        source_gradient, scale_gradient = metile.vjp(output, (source, scale), cotangent)
    arguments = {"source": values, "scale": 0.75, "seed": seed}
    shared_gradient = seed * (2 * values * 0.75 + np.exp(values * 0.75))
    np.testing.assert_allclose(_evaluate(source_gradient, arguments), shared_gradient * 0.75)
    np.testing.assert_allclose(
        _evaluate(scale_gradient, arguments), np.sum(shared_gradient * values)
    )
    assert isinstance(scale_gradient._value.type, ScalarType)


def test_reduction_broadcast_adjoint_and_softmax():
    values = np.array([0.2, -0.4, 1.2, 0.7])
    seed = np.array([0.4, 1.1, -0.3, 0.8])
    with TracingContext("softmax") as context:
        source = _input(context)
        cotangent = _input(context, "seed")
        numerator = metile.exp(source - metile.max(source))
        probabilities = numerator / metile.sum(numerator)
        gradient = metile.vjp(probabilities, source, cotangent)
    expected = np.exp(values - np.max(values))
    expected /= np.sum(expected)
    expected *= seed - np.sum(seed * expected)
    np.testing.assert_allclose(_evaluate(gradient, {"source": values, "seed": seed}), expected)


@pytest.mark.parametrize("operation", [metile.max, metile.min])
def test_extremum_reductions_split_gradient_equally_between_ties(operation):
    values = np.array([2.0, 2.0, -1.0, -1.0])
    with TracingContext("ties") as context:
        source = _input(context)
        gradient = metile.vjp(operation(source), source, 6.0)
    expected = [3, 3, 0, 0] if operation is metile.max else [0, 0, 3, 3]
    np.testing.assert_array_equal(_evaluate(gradient, {"source": values}), expected)


@pytest.mark.parametrize("operation", [metile.maximum, metile.minimum])
def test_binary_extrema_split_ties_and_reduce_scalar_broadcast(operation):
    values = np.array([-1.0, 1.0, 1.0, 2.0])
    with TracingContext("binary_ties") as context:
        source = _input(context)
        threshold = _input(context, "threshold", shape=())
        gradients = metile.vjp(operation(source, threshold), (source, threshold), 2.0)
    expected = [0, 1, 1, 2] if operation is metile.maximum else [2, 1, 1, 0]
    arguments = {"source": values, "threshold": 1.0}
    np.testing.assert_array_equal(_evaluate(gradients[0], arguments), expected)
    assert _evaluate(gradients[1], arguments) == 4


def test_where_freezes_predicate_and_reduces_selected_scalar_branch():
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    with TracingContext("where") as context:
        source = _input(context)
        bias = _input(context, "bias", shape=())
        gradients = metile.vjp(
            metile.where(source > 0.0, source * source, bias), (source, bias), 1.0
        )
        predicate_only = metile.vjp(metile.where(source > 0.0, 2.0, 3.0), source, 1.0)
    arguments = {"source": values, "bias": 5.0}
    np.testing.assert_array_equal(_evaluate(gradients[0], arguments), [0, 0, 2, 4])
    assert _evaluate(gradients[1], arguments) == 2
    np.testing.assert_array_equal(_evaluate(predicate_only, arguments), np.zeros(4))


@pytest.mark.parametrize(("source_dtype", "target_dtype"), [("f16", "f32"), ("f32", "f16")])
def test_float_cast_rule_restores_input_gradient_dtype(source_dtype, target_dtype):
    with TracingContext("cast") as context:
        source = _input(context, dtype=source_dtype)
        gradient = metile.vjp(metile.cast(source, target_dtype), source, 0.75)
    assert gradient._value.type == source._value.type
    np.testing.assert_array_equal(_evaluate(gradient, {"source": np.arange(4)}), [0.75] * 4)


def test_requested_intermediates_are_independent_stop_leaves():
    with TracingContext("stops") as context:
        source = _input(context)
        shared = source * source
        source_gradient, shared_gradient = metile.vjp(shared * source, (source, shared), 1.0)
    values = np.arange(4, dtype=float)
    np.testing.assert_array_equal(_evaluate(source_gradient, {"source": values}), values**2)
    np.testing.assert_array_equal(_evaluate(shared_gradient, {"source": values}), values)


def test_disconnected_zero_and_scalar_seed_broadcast_do_not_multiply_by_nan():
    with TracingContext("zeros") as context:
        source = _input(context)
        other = _input(context, "other")
        gradient = metile.vjp(source, other, 1.0)
        identity = metile.vjp(source, (source,), 3.0)
    arguments = {"source": np.array([0, np.inf, np.nan, -np.inf]), "other": np.full(4, np.nan)}
    np.testing.assert_array_equal(_evaluate(gradient, arguments), np.zeros(4))
    assert isinstance(identity, tuple)
    np.testing.assert_array_equal(_evaluate(identity[0], arguments), [3.0] * 4)


def test_loaded_values_are_stops_and_unrequested_loads_are_frozen_coefficients():
    with TracingContext("loads") as context:
        pointer = TracingProxy(tir.Value("pointer", PtrType("f32")))
        context.func.params.append(tir.Param("pointer", pointer._value.type))
        positions = metile.arange(0, 4)
        source = metile.load(pointer + positions)
        coefficient = metile.load(pointer + positions + 4)
        gradient = metile.vjp(source * coefficient, source, 2.0)
    arguments = {source._value.name: np.arange(4), coefficient._value.name: np.array([1, 2, 3, 4])}
    np.testing.assert_array_equal(_evaluate(gradient, arguments), [2, 4, 6, 8])
    assert sum(isinstance(operation, tir.Load) for operation in context.func.ops) == 2


@pytest.mark.parametrize(
    "expression",
    [
        lambda source: metile.simd_sum(source),
        lambda source: metile.simd_max(source),
        lambda source: metile.simd_shuffle_xor(source, 1),
        lambda source: metile.simd_broadcast(source, 0),
        lambda source: source % 2.0,
        lambda source: source & source,
        lambda source: metile.cast(metile.cast(source, "i32"), "f32"),
        lambda source: metile.dot(source, source, source),
    ],
)
def test_unsupported_differentiated_operations_fail_before_emitting_gradients(expression):
    with TracingContext("unsupported") as context:
        source = _input(context)
        output = expression(source)
        before = tuple(id(operation) for operation in context.func.ops)
        with pytest.raises((NotImplementedError, TypeError), match="vjp"):
            metile.vjp(output, source, 1.0)
        assert tuple(id(operation) for operation in context.func.ops) == before


def test_unsupported_unrelated_subgraph_is_a_frozen_coefficient():
    with TracingContext("unrelated") as context:
        source = _input(context)
        other = _input(context, "other")
        frozen = metile.simd_sum(other)
        gradient = metile.vjp(source * frozen, source, 1.0)
    np.testing.assert_array_equal(
        _evaluate(gradient, {"source": np.arange(4), frozen._value.name: np.full(4, 7.0)}),
        np.full(4, 7.0),
    )


def test_runtime_loop_result_cannot_be_differentiated_as_a_straight_line_expression():
    with TracingContext("loop") as context:
        source = _input(context)
        for _ in metile.tile_range(0, 4):
            output = source * source
        with pytest.raises(ValueError, match="runtime loop"):
            metile.vjp(output, source, 1.0)


@pytest.mark.parametrize("inputs", [(), [], [1.0], 1.0, (1.0,)])
def test_input_container_validation(inputs):
    with TracingContext("validation") as context:
        source = _input(context)
        with pytest.raises(TypeError, match="inputs"):
            metile.vjp(source, inputs, 1.0)


def test_active_trace_float_types_distinct_inputs_and_cotangent_shapes_are_required():
    with pytest.raises(RuntimeError, match="inside"):
        metile.vjp(None, None, 1.0)
    with TracingContext("validation") as context:
        source = _input(context)
        integer = _input(context, "integer", dtype="i32")
        matrix = _input(context, "matrix", shape=(2, 2))
        mismatched = _input(context, "mismatched", shape=(8,))
        with pytest.raises(TypeError, match="floating"):
            metile.vjp(source, integer, 1.0)
        with pytest.raises(ValueError, match="distinct"):
            metile.vjp(source, (source, source), 1.0)
        with pytest.raises(ValueError, match="one-dimensional"):
            metile.vjp(matrix, matrix, 1.0)
        with pytest.raises(ValueError, match="cotangent"):
            metile.vjp(source, source, mismatched)
        with pytest.raises(TypeError, match="cotangent"):
            metile.vjp(source, source, True)
        with pytest.raises(TypeError, match="floating"):
            metile.vjp(source, source, integer)


def test_values_from_another_trace_and_cycles_are_rejected():
    with TracingContext("first"):
        foreign = metile.scalar(2.0)
    with TracingContext("second") as context:
        source = _input(context)
        with pytest.raises(ValueError, match="active tracing region"):
            metile.vjp(foreign, source, 1.0)
        output = source * 2.0
        output._value.defining_op.lhs = output._value
        with pytest.raises(ValueError, match="acyclic"):
            metile.vjp(output, source, 1.0)


@pytest.mark.parametrize("role", ["input", "output", "cotangent", "coefficient"])
def test_foreign_scalar_parameter_with_identical_name_and_type_is_rejected(role):
    with TracingContext("first") as first:
        foreign = _input(first, "source", shape=())
    with TracingContext("second") as second:
        source = _input(second, "source", shape=())
        output = source * source
        arguments = (output, source, 1.0)
        if role == "input":
            arguments = (output, foreign, 1.0)
        elif role == "output":
            arguments = (foreign, source, 1.0)
        elif role == "cotangent":
            arguments = (output, source, foreign)
        else:
            arguments = (source * foreign, source, 1.0)
        with pytest.raises(ValueError, match="active tracing region"):
            metile.vjp(*arguments)


@pytest.mark.parametrize(
    "role", ["predicate", "nested_predicate", "cotangent", "cotangent_predicate"]
)
def test_foreign_parameter_in_frozen_ancestry_is_rejected_before_emitting_adjoints(role):
    with TracingContext("first") as first:
        foreign = _input(first, "source", shape=())
    with TracingContext("second") as second:
        source = _input(second, "source", shape=())
        output = source * source
        seed = 1.0
        if role == "predicate":
            output = metile.where(foreign > 0.0, source * 2.0, source * 3.0)
        elif role == "nested_predicate":
            coefficient = metile.where(foreign > 0.0, 2.0, 3.0)
            output = metile.where(coefficient > 2.5, source * 2.0, source * 3.0)
        elif role == "cotangent":
            seed = metile.exp(foreign * 0.5)
        else:
            seed = metile.where(foreign > 0.0, 2.0, 3.0)
        operation_count = len(second.func.ops)
        with pytest.raises(ValueError, match="active tracing region"):
            metile.vjp(output, source, seed)
        assert len(second.func.ops) == operation_count


def test_current_predicate_and_expression_cotangent_remain_frozen():
    with TracingContext("frozen_ancestry") as context:
        source = _input(context, "source", shape=())
        coefficient = _input(context, "coefficient", shape=())
        output = metile.where(coefficient > 0.0, source * 2.0, source * 3.0)
        gradient = metile.vjp(output, source, metile.exp(coefficient))
    np.testing.assert_allclose(
        _evaluate(gradient, {"source": 4.0, "coefficient": 0.5}), 2.0 * np.exp(0.5)
    )


def test_parameter_proxies_created_before_entering_context_are_supported():
    context = TracingContext("prepared_parameters")
    source = _input(context, "source", shape=())
    disconnected = _input(context, "disconnected", shape=())
    cotangent = _input(context, "cotangent", shape=())
    with context:
        gradient, zero = metile.vjp(source * source, (source, disconnected), cotangent)
    arguments = {"source": 2.0, "disconnected": 4.0, "cotangent": 3.0}
    assert _evaluate(gradient, arguments) == 12.0
    assert _evaluate(zero, arguments) == 0.0


@pytest.mark.parametrize("initial_depends_on_source", [False, True])
@pytest.mark.parametrize("role", ["output", "coefficient", "unrelated"])
def test_mutable_state_cannot_hide_dependencies_from_vjp(initial_depends_on_source, role):
    with TracingContext("mutable_state_dependency") as context:
        source = _input(context, "source", shape=())
        unrelated = _input(context, "unrelated", shape=())
        state = metile.loop_state(source if initial_depends_on_source else 0.0)
        state.update(source * 2.0)
        snapshot = state.value
        output = source * snapshot if role == "coefficient" else snapshot
        requested = unrelated if role == "unrelated" else source
        before = tuple(id(operation) for operation in context.func.ops)
        with pytest.raises(NotImplementedError, match="mutable loop state"):
            metile.vjp(output, requested, 1.0)
        assert tuple(id(operation) for operation in context.func.ops) == before


def test_mutable_state_declaration_is_also_rejected_before_dependency_pruning():
    with TracingContext("mutable_state_declaration") as context:
        source = _input(context, "source", shape=())
        state = metile.loop_state(0.0)
        with pytest.raises(NotImplementedError, match="mutable loop state"):
            metile.vjp(TracingProxy(state._state), source, 1.0)


def test_mutable_state_snapshot_is_allowed_as_an_explicit_differentiation_leaf():
    with TracingContext("snapshot_leaf") as context:
        source = _input(context, "source", shape=())
        state = metile.loop_state(0.0)
        state.update(source * 2.0)
        snapshot = state.value
        gradient = metile.vjp(snapshot * snapshot, snapshot, 3.0)
    assert _evaluate(gradient, {snapshot._value.name: 4.0}) == 24.0
