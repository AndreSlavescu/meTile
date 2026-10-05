from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from metile.ir import tile_ir as tir
from metile.ir.types import ScalarType, TileType


class EpilogueError(ValueError):
    """A GEMM output expression cannot be fused without changing its meaning."""


@dataclass(frozen=True)
class EpilogueInstruction:
    name: str
    kind: Literal[
        "accumulator", "constant", "parameter", "unary", "binary", "compare", "select", "cast"
    ]
    type: ScalarType
    operands: tuple[str, ...] = ()
    operation: str = ""
    value: int | float | None = None
    parameter: str = ""

    @property
    def capture_name(self) -> str:
        return f"{self.name}_capture"


@dataclass(frozen=True)
class EpilogueProgram:
    """An immutable scalar SSA program applied independently to each f32 accumulator."""

    instructions: tuple[EpilogueInstruction, ...]
    result: str


def _unsupported(reason: str):
    raise EpilogueError(f"GEMM descriptor epilogue is not supported: {reason}")


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def build_epilogue(function: tir.Function) -> EpilogueProgram | None:
    """Extract the tensor store's pure dependency DAG, rooted at the completed dot.

    Independent scalar coefficients retain their declared types and receive
    hygienic entry-scope captures. Tile arithmetic remains f32, and the backend
    owns the final conversion to the output dtype.
    Dimension parameters are excluded until all backends preserve their bindings
    through dimension aliasing and static specialization.
    """
    operations = tuple(_walk(function.ops))
    dots = [operation for operation in operations if isinstance(operation, tir.Dot)]
    stores = [operation for operation in operations if isinstance(operation, tir.TileStore)]
    if len(dots) != 1 or len(stores) != 1:
        _unsupported("expected one dot recurrence and one tensor store")
    accumulator = dots[0].result
    output = stores[0]
    if (
        accumulator is None
        or not isinstance(accumulator.type, TileType)
        or accumulator.type.dtype != "f32"
    ):
        _unsupported("the accumulator must be an f32 tile")
    positions = {id(operation): index for index, operation in enumerate(function.ops)}
    loops = [
        operation
        for operation in function.ops
        if isinstance(operation, tir.ForRange)
        and any(nested is dots[0] for nested in operation.body)
    ]
    if (
        len(loops) != 1
        or id(output) not in positions
        or positions[id(output)] <= positions[id(loops[0])]
    ):
        _unsupported("the tensor store must follow one top-level dot reduction")
    if output.value is accumulator:
        return None
    if not isinstance(output.value.type, TileType) or output.value.type != accumulator.type:
        _unsupported("the stored expression must preserve the f32 accumulator tile type")
    parameters = {parameter.name: parameter for parameter in function.params}
    dimension_parameters = {"M", "N", "K"}
    for operation in operations:
        memory = getattr(operation, "tensor", None)
        if memory is not None:
            dimension_parameters.update(
                dimension.name for dimension in memory.shape if dimension.defining_op is None
            )
    prefix = "_metile_epilogue_"
    while any(name.startswith(prefix) for name in parameters):
        prefix += "_"
    instructions = []
    translated = {}
    visiting = set()
    allowed_scalars = {"bool", "i32", "u32", "f16", "f32"}
    numeric_binary = {"add", "sub", "mul", "div", "mod", "max", "min"}
    integer_binary = {"bitand", "bitor", "bitxor", "shl", "shr"}
    unary_operations = {
        "exp",
        "exp2",
        "fast_cos",
        "fast_exp",
        "fast_exp2",
        "fast_sin",
        "log",
        "sqrt",
        "rsqrt",
        "abs",
        "neg",
        "tanh",
    }

    def value_type(value):
        if isinstance(value.type, TileType):
            if value.type.shape != accumulator.type.shape or value.type.dtype not in {
                "f32",
                "bool",
            }:
                _unsupported(
                    "tile expressions must preserve the accumulator shape and f32 precision"
                )
            return ScalarType(value.type.dtype)
        if not isinstance(value.type, ScalarType) or value.type.dtype not in allowed_scalars:
            _unsupported("only numeric or boolean scalar values may enter an epilogue")
        return value.type

    def visit(value):
        identity = id(value)
        if identity in translated:
            return translated[identity]
        if identity in visiting:
            _unsupported("cyclic value dependencies")
        visiting.add(identity)
        result_type = value_type(value)
        operation = value.defining_op
        operands = ()
        opcode = ""
        literal = None
        parameter_name = ""
        if value is accumulator:
            kind = "accumulator"
        elif operation is None:
            parameter = parameters.get(value.name)
            if parameter is None or parameter.type != value.type:
                _unsupported(f"unbound scalar value %{value.name}")
            if value.name in dimension_parameters:
                _unsupported(
                    f"dimension parameter %{value.name} requires a preserved scalar binding"
                )
            kind = "parameter"
            parameter_name = value.name
        elif isinstance(operation, tir.Constant):
            if not isinstance(operation.value, (int, float)):
                _unsupported("constants must be numeric")
            kind = "constant"
            literal = operation.value
            if result_type.dtype in {"i32", "u32"}:
                minimum = -(1 << 31) if result_type.dtype == "i32" else 0
                maximum = (1 << 31) - 1 if result_type.dtype == "i32" else (1 << 32) - 1
                if not minimum <= literal <= maximum:
                    _unsupported("integer constants must fit their declared scalar type")
        else:
            if id(operation) not in positions or positions[id(operation)] >= positions[id(output)]:
                _unsupported("epilogue dependencies must precede the store outside control flow")
            if isinstance(operation, tir.Unary):
                if operation.op not in unary_operations or result_type.dtype == "bool":
                    _unsupported(f"unary operation {operation.op}")
                if operation.op not in {"abs", "neg"} and result_type.dtype not in {"f16", "f32"}:
                    _unsupported("transcendental unary operations require floating-point values")
                if operation.op == "abs" and result_type.dtype == "u32":
                    _unsupported("unsigned absolute value is not a supported intrinsic")
                kind = "unary"
                opcode = operation.op
                operands = (visit(operation.operand),)
            elif isinstance(operation, tir.BinOp):
                if operation.op in numeric_binary:
                    if result_type.dtype == "bool":
                        _unsupported("numeric arithmetic on boolean values")
                elif operation.op in integer_binary:
                    if any(
                        datatype.dtype not in {"bool", "i32", "u32"}
                        for datatype in (
                            result_type,
                            value_type(operation.lhs),
                            value_type(operation.rhs),
                        )
                    ):
                        _unsupported("bit operations require integer or boolean values")
                else:
                    _unsupported(f"binary operation {operation.op}")
                kind = "binary"
                opcode = operation.op
                operands = (visit(operation.lhs), visit(operation.rhs))
            elif isinstance(operation, tir.Fma):
                operation.result_type()
                kind = "fma"
                operands = tuple(
                    visit(operand)
                    for operand in (operation.left, operation.right, operation.addend)
                )
            elif isinstance(operation, tir.Compare):
                if operation.predicate not in {"lt", "le", "gt", "ge", "eq", "ne"}:
                    _unsupported(f"comparison {operation.predicate}")
                kind = "compare"
                opcode = operation.predicate
                operands = (visit(operation.lhs), visit(operation.rhs))
            elif isinstance(operation, tir.Select):
                if value_type(operation.condition).dtype != "bool":
                    _unsupported("where conditions must be boolean")
                kind = "select"
                operands = (
                    visit(operation.condition),
                    visit(operation.true_val),
                    visit(operation.false_val),
                )
            elif isinstance(operation, tir.Cast):
                if operation.dtype != "f32":
                    _unsupported("casts inside the epilogue may only convert to f32")
                kind = "cast"
                operands = (visit(operation.value),)
            else:
                _unsupported(f"effectful or non-elementwise operation {type(operation).__name__}")
        name = f"{prefix}{len(instructions)}"
        instruction = EpilogueInstruction(
            name=name,
            kind=kind,
            type=result_type,
            operands=operands,
            operation=opcode,
            value=literal,
            parameter=parameter_name,
        )
        instructions.append(instruction)
        translated[identity] = name
        visiting.remove(identity)
        return name

    result = visit(output.value)
    if not any(instruction.kind == "accumulator" for instruction in instructions):
        _unsupported("the stored expression must depend on the completed dot accumulator")
    return EpilogueProgram(tuple(instructions), result)
