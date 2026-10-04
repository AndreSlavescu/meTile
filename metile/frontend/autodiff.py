"""Explicit reverse-mode differentiation of bounded, traced tile expressions."""

from dataclasses import fields
from numbers import Real

from metile.frontend import tracing
from metile.ir import tile_ir as tir
from metile.ir.types import ScalarType, TileType, merge_tile_layouts

_FLOATS = {"f16", "f32"}
_BINARY = {"add", "sub", "mul", "div", "max", "min"}
_UNARY = {"exp", "fast_exp", "log", "sqrt", "tanh", "abs", "neg"}


def _floating(value, label):
    if not isinstance(value.type, (ScalarType, TileType)) or value.type.dtype not in _FLOATS:
        raise TypeError(f"vjp {label} must have f16 or f32 floating type")
    if isinstance(value.type, TileType) and (
        len(value.type.shape) != 1 or value.type.shape[0] <= 0
    ):
        raise ValueError(f"vjp {label} must be a scalar or nonempty one-dimensional tile")


def _operands(operation):
    if operation is None or isinstance(operation, (tir.Constant, tir.Load, tir.TileLoad)):
        return ()
    if isinstance(operation, tir.Select):
        return operation.true_val, operation.false_val
    return tuple(
        operand
        for field in fields(operation)
        if field.name != "result"
        and isinstance(operand := getattr(operation, field.name), tir.Value)
    )


def _check_rule(value, operands):
    operation = value.defining_op
    supported = (
        (isinstance(operation, tir.BinOp) and operation.op in _BINARY)
        or (isinstance(operation, tir.Unary) and operation.op in _UNARY)
        or isinstance(operation, tir.Select)
        or (
            isinstance(operation, tir.Cast)
            and operation.value.type.dtype in _FLOATS
            and operation.dtype in _FLOATS
        )
        or (
            isinstance(operation, tir.Reduce)
            and operation.op in {"sum", "max", "min"}
            and isinstance(operation.operand.type, TileType)
        )
    )
    if not supported:
        label = type(operation).__name__
        if hasattr(operation, "op"):
            label += f"({operation.op})"
        raise NotImplementedError(f"vjp does not support differentiating {label}")
    _floating(value, "differentiated result")
    if isinstance(operation, tir.Select):
        operands = (*operands, operation.condition)
    tile_types = [
        operand.type for operand in (*operands, value) if isinstance(operand.type, TileType)
    ]
    if len({datatype.shape for datatype in tile_types}) > 1:
        raise ValueError("vjp only supports scalar broadcasting and matching tile shapes")
    merge_tile_layouts(*tile_types)


def _cast(value, dtype):
    return value if value._value.type.dtype == dtype else tracing.cast(value, dtype)


def _broadcast(value, template):
    if isinstance(template.type, TileType) and isinstance(value._value.type, ScalarType):
        proxy = tracing.TracingProxy(template)
        return tracing.where(proxy == proxy, value, value)
    return value


def _match(value, template):
    if isinstance(template.type, ScalarType) and isinstance(value._value.type, TileType):
        value = tracing.sum(value)
    return _cast(_broadcast(value, template), template.type.dtype)


def _pullback(value, gradient, depends):
    operation = value.defining_op
    proxy = tracing.TracingProxy
    result = proxy(value)
    if isinstance(operation, tir.BinOp):
        left, right = proxy(operation.lhs), proxy(operation.rhs)
        for operand, first in ((operation.lhs, True), (operation.rhs, False)):
            if not depends[id(operand)]:
                continue
            if operation.op == "add":
                contribution = gradient
            elif operation.op == "sub":
                contribution = gradient if first else 0.0 - gradient
            elif operation.op == "mul":
                contribution = gradient * (right if first else left)
            elif operation.op == "div":
                contribution = gradient / right if first else (0.0 - gradient * result) / right
            else:
                selected, other = (left, right) if first else (right, left)
                wins = selected > other if operation.op == "max" else selected < other
                contribution = tracing.where(
                    wins, gradient, tracing.where(selected == other, gradient * 0.5, 0.0)
                )
            yield operand, contribution
    elif isinstance(operation, tir.Unary):
        operand = proxy(operation.operand)
        if operation.op in {"exp", "fast_exp"}:
            contribution = gradient * result
        elif operation.op == "log":
            contribution = gradient / operand
        elif operation.op == "sqrt":
            contribution = gradient * 0.5 / result
        elif operation.op == "tanh":
            contribution = gradient * (1.0 - result * result)
        elif operation.op == "abs":
            contribution = tracing.where(
                operand > 0.0, gradient, tracing.where(operand < 0.0, 0.0 - gradient, 0.0)
            )
        else:
            contribution = 0.0 - gradient
        yield operation.operand, contribution
    elif isinstance(operation, tir.Cast):
        yield operation.value, _cast(gradient, operation.value.type.dtype)
    elif isinstance(operation, tir.Select):
        condition = proxy(operation.condition)
        if depends[id(operation.true_val)]:
            yield operation.true_val, tracing.where(condition, gradient, 0.0)
        if depends[id(operation.false_val)]:
            yield operation.false_val, tracing.where(condition, 0.0, gradient)
    elif isinstance(operation, tir.Reduce):
        if operation.op == "sum":
            contribution = gradient
        else:
            winners = proxy(operation.operand) == result
            count = tracing.sum(tracing.where(winners, 1.0, 0.0))
            contribution = tracing.where(winners, gradient / count, 0.0)
        yield operation.operand, contribution


def vjp(
    output: tracing.TracingProxy,
    inputs: tracing.TracingProxy | tuple[tracing.TracingProxy, ...],
    cotangent: tracing.TracingProxy | Real,
) -> tracing.TracingProxy | tuple[tracing.TracingProxy, ...]:
    """Trace a vector-Jacobian product for an explicit pure expression.

    ``inputs`` is a floating proxy or a nonempty tuple of distinct floating
    proxies; the return has the same container form. ``cotangent`` is a
    floating proxy or real scalar; a scalar seed broadcasts over a tile output.
    Supported values are f16/f32 scalars and one-dimensional tiles. Scalar
    broadcast inputs receive the sum of their elementwise contributions.

    Requested inputs are stop leaves, even when they are intermediates. Other
    memory loads are frozen coefficients: no memory/address derivatives,
    stores, atomics, whole-kernel backward pass, or runtime-loop differentiation
    are implied. Unsupported operations on a path to an input raise rather
    than silently producing zero. A disconnected input receives zero.

    Select predicates are nondifferentiable. Max/min split cotangents equally
    between ties (including reduction ties); abs has derivative zero at zero.
    Floating casts differentiate their real-valued conversion, ignoring
    rounding, and cast the adjoint back. Fast exp uses its computed value as
    its derivative. These are finite-real, first-order rules, not derivatives
    of floating-point rounding, NaN behavior, or approximate instruction bits.
    """
    context = tracing._get_ctx()
    single = isinstance(inputs, tracing.TracingProxy)
    requested = (inputs,) if single else inputs
    if not isinstance(output, tracing.TracingProxy):
        raise TypeError("vjp output must be a TracingProxy")
    if (
        not isinstance(requested, tuple)
        or not requested
        or any(not isinstance(value, tracing.TracingProxy) for value in requested)
    ):
        raise TypeError("vjp inputs must be a proxy or a nonempty tuple of proxies")
    targets = {id(value._value) for value in requested}
    if len(targets) != len(requested):
        raise ValueError("vjp inputs must be distinct values")
    region = {id(operation) for operation in context.func.ops}
    parameters = {parameter.name: parameter.type for parameter in context.func.params}

    def check_scope(value):
        if value.defining_op is None:
            if (
                parameters.get(value.name) != value.type
                or context._parameter_values.get(id(value)) is not value
            ):
                raise ValueError("vjp values must belong to the active tracing region")
        elif id(value.defining_op) not in region or value.defining_op.result is not value:
            raise ValueError(
                "vjp values must belong to the active tracing region; runtime loop results "
                "and values from other traces are unsupported"
            )

    scoped = set()
    scope_visiting = set()

    def check_ancestry(value):
        identity = id(value)
        if identity in scoped:
            return
        if identity in scope_visiting:
            raise ValueError("vjp requires an acyclic expression")
        check_scope(value)
        scope_visiting.add(identity)
        operands = _operands(value.defining_op)
        if isinstance(value.defining_op, tir.Select):
            operands = (value.defining_op.condition, *operands)
        for operand in operands:
            check_ancestry(operand)
        scope_visiting.remove(identity)
        scoped.add(identity)

    _floating(output._value, "output")
    check_ancestry(tracing._to_value(output))
    for value in requested:
        _floating(value._value, "input")
        check_ancestry(tracing._to_value(value))
    if isinstance(cotangent, tracing.TracingProxy):
        _floating(cotangent._value, "cotangent")
        check_ancestry(tracing._to_value(cotangent))
        if isinstance(cotangent._value.type, TileType):
            if not isinstance(output._value.type, TileType) or (
                cotangent._value.type.shape != output._value.type.shape
            ):
                raise ValueError("vjp cotangent must match the output shape or be scalar")
            merge_tile_layouts(cotangent._value.type, output._value.type)
    elif not isinstance(cotangent, Real) or isinstance(cotangent, bool):
        raise TypeError("vjp cotangent must be a floating proxy or real scalar")

    order = []
    depends = {}
    visiting = set()

    def visit(value):
        identity = id(value)
        if identity in depends:
            return depends[identity]
        if identity in visiting:
            raise ValueError("vjp requires an acyclic expression")
        check_scope(value)
        if identity in targets:
            depends[identity] = True
            return True
        if isinstance(value.defining_op, (tir.ReadLoopState, tir.LoopState)):
            raise NotImplementedError(
                "vjp cannot traverse mutable loop state; request a snapshot as an explicit input"
            )
        visiting.add(identity)
        operands = _operands(value.defining_op)
        dependencies = [visit(operand) for operand in operands]
        relevant = any(dependencies)
        if relevant:
            _check_rule(value, operands)
            if isinstance(value.defining_op, tir.Select):
                check_scope(value.defining_op.condition)
            order.append(value)
        visiting.remove(identity)
        depends[identity] = relevant
        return relevant

    visit(output._value)
    seed = (
        cotangent
        if isinstance(cotangent, tracing.TracingProxy)
        else tracing.scalar(float(cotangent))
    )
    gradients = {id(output._value): _match(seed, output._value)}
    for value in reversed(order):
        for operand, contribution in _pullback(value, gradients[id(value)], depends):
            contribution = _match(contribution, operand)
            previous = gradients.get(id(operand))
            gradients[id(operand)] = contribution if previous is None else previous + contribution
    result = tuple(
        gradients[id(value._value)]
        if id(value._value) in gradients
        else _match(tracing.scalar(0.0), value._value)
        for value in requested
    )
    return result[0] if single else result


__all__ = ["vjp"]
