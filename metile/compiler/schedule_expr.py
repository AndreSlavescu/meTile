from __future__ import annotations

import zlib
from dataclasses import dataclass

_BINARY_SYMBOLS = {
    "add": "+",
    "and": "&",
    "div": "/",
    "mod": "%",
    "mul": "*",
    "shl": "<<",
    "shr": ">>",
}

_TARGET_COST = {
    "add": 1,
    "and": 1,
    "cast_u32": 0,
    "div": 8,
    "mod": 8,
    "mul": 2,
    "shl": 1,
    "shr": 1,
    "symbol": 0,
    "u32": 0,
    "u64": 0,
}


@dataclass(frozen=True)
class ScheduleExpr:
    """A scalar expression in the composable tile-schedule IR."""

    op: str
    args: tuple[object, ...]

    def __post_init__(self):
        if self.op not in {*_BINARY_SYMBOLS, "cast_u32", "symbol", "u32", "u64"}:
            raise ValueError(f"unknown schedule expression operation: {self.op}")

    @classmethod
    def symbol(cls, name: str) -> ScheduleExpr:
        return cls("symbol", (name,))

    @classmethod
    def u32(cls, value: int) -> ScheduleExpr:
        return cls("u32", (value,))

    @classmethod
    def u64(cls, value: int) -> ScheduleExpr:
        return cls("u64", (value,))

    def evaluate(self, environment: dict[str, int]) -> int:
        if self.op == "symbol":
            return environment[str(self.args[0])]
        if self.op in {"u32", "u64"}:
            return int(self.args[0])
        if self.op == "cast_u32":
            return _expr(self.args[0]).evaluate(environment) & 0xFFFFFFFF

        lhs = _expr(self.args[0]).evaluate(environment)
        rhs = _expr(self.args[1]).evaluate(environment)
        if self.op == "add":
            return lhs + rhs
        if self.op == "and":
            return lhs & rhs
        if self.op == "div":
            return lhs // rhs
        if self.op == "mod":
            return lhs % rhs
        if self.op == "mul":
            return lhs * rhs
        if self.op == "shl":
            return lhs << rhs
        if self.op == "shr":
            return lhs >> rhs
        raise AssertionError(self.op)

    def emit(self) -> str:
        if self.op == "symbol":
            return str(self.args[0])
        if self.op == "u32":
            return f"{int(self.args[0])}u"
        if self.op == "u64":
            return f"0x{int(self.args[0]):X}ul"
        if self.op == "cast_u32":
            return f"uint({_expr(self.args[0]).emit()})"
        lhs = _expr(self.args[0]).emit()
        rhs = _expr(self.args[1]).emit()
        return f"({lhs} {_BINARY_SYMBOLS[self.op]} {rhs})"

    def encode(self) -> str:
        if self.op == "symbol":
            return f"s:{self.args[0]}"
        if self.op in {"u32", "u64"}:
            return f"{self.op}:{int(self.args[0])}"
        return f"{self.op}({','.join(_expr(arg).encode() for arg in self.args)})"

    @property
    def target_cost(self) -> int:
        local_cost = _TARGET_COST[self.op]
        if self.op in {"div", "mod", "mul"}:
            rhs = _expr(self.args[1])
            if rhs.op == "u32" and _is_power_of_two(int(rhs.args[0])):
                local_cost = 1
        return local_cost + sum(
            _expr(arg).target_cost for arg in self.args if isinstance(arg, ScheduleExpr)
        )


@dataclass(frozen=True)
class ScheduleAssignment:
    name: str
    expression: ScheduleExpr
    dtype: str = "uint"
    constexpr: bool = False

    def emit(self) -> str:
        qualifier = "constexpr" if self.constexpr else "const"
        return f"{qualifier} {self.dtype} {self.name} = {self.expression.emit()};"

    def encode(self) -> str:
        qualifier = "k" if self.constexpr else "c"
        return f"{qualifier}:{self.dtype}:{self.name}={self.expression.encode()}"


@dataclass(frozen=True)
class ScheduleProgram:
    """One exact decoder representation for a tile traversal."""

    pattern: str
    encoding: str
    assignments: tuple[ScheduleAssignment, ...]

    def evaluate(self, tgp_x: int, tgp_y: int) -> tuple[int, int]:
        environment = {"tgp_id.x": tgp_x, "tgp_id.y": tgp_y}
        for assignment in self.assignments:
            environment[assignment.name] = assignment.expression.evaluate(environment)
        return environment["pid_m"], environment["pid_n"]

    def emit_lines(self) -> tuple[str, ...]:
        return tuple(assignment.emit() for assignment in self.assignments)

    @property
    def encoded(self) -> str:
        return "|".join(assignment.encode() for assignment in self.assignments)

    @property
    def description_bits(self) -> int:
        """Computable upper bound in a fixed schedule-description language."""
        return len(zlib.compress(self.encoded.encode(), level=9)) * 8

    @property
    def target_cost(self) -> int:
        return sum(assignment.expression.target_cost for assignment in self.assignments)


def schedule_programs(pattern: str, grid_m: int, grid_n: int) -> tuple[ScheduleProgram, ...]:
    """Build exact, equivalent decoder programs for one resolved schedule."""
    programs = [
        _build_schedule_program(pattern, grid_m, grid_n, encoding)
        for encoding in ("arithmetic", "bitwise")
    ]
    unique = {}
    for program in programs:
        semantic_encoding = tuple(
            assignment.expression.encode() for assignment in program.assignments
        )
        unique.setdefault(semantic_encoding, program)
    return tuple(unique.values())


def select_schedule_program(
    pattern: str,
    grid_m: int,
    grid_n: int,
    encoding: str = "auto",
) -> ScheduleProgram:
    """Extract the cheapest legal decoder, using MDL to break target-cost ties."""
    if encoding != "auto":
        if encoding not in {"arithmetic", "bitwise"}:
            raise ValueError(f"unknown schedule encoding '{encoding}'")
        return _build_schedule_program(pattern, grid_m, grid_n, encoding)
    programs = schedule_programs(pattern, grid_m, grid_n)
    return min(
        programs,
        key=lambda program: (program.target_cost, program.description_bits, program.encoded),
    )


def _build_schedule_program(
    pattern: str,
    grid_m: int,
    grid_n: int,
    encoding: str,
) -> ScheduleProgram:
    if grid_m <= 0 or grid_n <= 0:
        raise ValueError("schedule grids must have positive dimensions")

    x = ScheduleExpr.symbol("tgp_id.x")
    y = ScheduleExpr.symbol("tgp_id.y")
    assignments: list[ScheduleAssignment] = []

    def assign(name: str, expression: ScheduleExpr, dtype: str = "uint", constexpr: bool = False):
        assignments.append(ScheduleAssignment(name, expression, dtype, constexpr))
        return ScheduleExpr.symbol(name)

    if pattern == "linear":
        assign("pid_m", x)
        assign("pid_n", y)
    elif pattern == "diagonal":
        assign("pid_m", x)
        assign("pid_n", _remainder(_add(y, x), grid_n, encoding))
    elif pattern == "morton":
        linear_id = assign("linear_id", _add(_mul(x, grid_n, encoding), y))
        panel_id = assign("panel_id", _quotient(linear_id, 4, encoding))
        within = assign("within", _remainder(linear_id, 4, encoding))
        panels_n = grid_n // 2
        assign(
            "pid_m",
            _add(
                _mul(_quotient(panel_id, panels_n, encoding), 2, encoding),
                _quotient(within, 2, encoding),
            ),
        )
        assign(
            "pid_n",
            _add(
                _mul(_remainder(panel_id, panels_n, encoding), 2, encoding),
                _remainder(within, 2, encoding),
            ),
        )
    elif pattern.startswith("grouped"):
        group = int(pattern.removeprefix("grouped"))
        linear_id = assign("linear_id", _add(_mul(y, grid_m, encoding), x))
        virtual_width = grid_n * group
        virtual_x = assign("virtual_x", _remainder(linear_id, virtual_width, encoding))
        virtual_y = assign("virtual_y", _quotient(linear_id, virtual_width, encoding))
        assign(
            "pid_m",
            _add(
                _mul(virtual_y, group, encoding),
                _remainder(virtual_x, group, encoding),
            ),
        )
        assign("pid_n", _quotient(virtual_x, group, encoding))
    elif pattern == "hilbert":
        linear_id = assign("linear_id", _add(_mul(x, grid_n, encoding), y))
        hilbert_m = assign("hilbert_m", ScheduleExpr.u64(0xEBFA5014), dtype="ulong", constexpr=True)
        hilbert_n = assign("hilbert_n", ScheduleExpr.u64(0x05BEBE50), dtype="ulong", constexpr=True)
        panel_id = assign("panel_id", _quotient(linear_id, 16, encoding))
        within = assign("within", _remainder(linear_id, 16, encoding))
        bit_offset = _mul(within, 2, encoding)
        local_m = ScheduleExpr(
            "cast_u32",
            (
                ScheduleExpr(
                    "and",
                    (ScheduleExpr("shr", (hilbert_m, bit_offset)), ScheduleExpr.u64(3)),
                ),
            ),
        )
        local_n = ScheduleExpr(
            "cast_u32",
            (
                ScheduleExpr(
                    "and",
                    (ScheduleExpr("shr", (hilbert_n, bit_offset)), ScheduleExpr.u64(3)),
                ),
            ),
        )
        panels_n = grid_n // 4
        assign(
            "pid_m",
            _add(_mul(_quotient(panel_id, panels_n, encoding), 4, encoding), local_m),
        )
        assign(
            "pid_n",
            _add(_mul(_remainder(panel_id, panels_n, encoding), 4, encoding), local_n),
        )
    else:
        raise ValueError(f"cannot build an expression program for schedule '{pattern}'")

    return ScheduleProgram(pattern, encoding, tuple(assignments))


def _expr(value: object) -> ScheduleExpr:
    if not isinstance(value, ScheduleExpr):
        raise TypeError(f"expected ScheduleExpr, got {type(value).__name__}")
    return value


def _u32(value: int) -> ScheduleExpr:
    return ScheduleExpr.u32(value)


def _add(lhs: ScheduleExpr, rhs: ScheduleExpr) -> ScheduleExpr:
    return ScheduleExpr("add", (lhs, rhs))


def _mul(value: ScheduleExpr, factor: int, encoding: str) -> ScheduleExpr:
    if factor == 1:
        return value
    if encoding == "bitwise" and _is_power_of_two(factor):
        return ScheduleExpr("shl", (value, _u32(_power_of_two_exponent(factor))))
    return ScheduleExpr("mul", (value, _u32(factor)))


def _quotient(value: ScheduleExpr, divisor: int, encoding: str) -> ScheduleExpr:
    if divisor == 1:
        return value
    if encoding == "bitwise" and _is_power_of_two(divisor):
        return ScheduleExpr("shr", (value, _u32(_power_of_two_exponent(divisor))))
    return ScheduleExpr("div", (value, _u32(divisor)))


def _remainder(value: ScheduleExpr, divisor: int, encoding: str) -> ScheduleExpr:
    if divisor == 1:
        return _u32(0)
    if encoding == "bitwise" and _is_power_of_two(divisor):
        return ScheduleExpr("and", (value, _u32(divisor - 1)))
    return ScheduleExpr("mod", (value, _u32(divisor)))


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _power_of_two_exponent(value: int) -> int:
    return value.bit_length() - 1
