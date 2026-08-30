"""Shared MSL emission helpers: value naming and literal formatting."""

from __future__ import annotations

from metile.ir import metal_ir as mir

_BINOP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "bitand": "&",
    "bitor": "|",
    "bitxor": "^",
    "shl": "<<",
    "shr": ">>",
}
_CMP_SYMBOLS = {
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}
_UNARY_MSL = {
    "exp": "exp",
    "fast_exp": "fast::exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    "neg": "-",
    "tanh": "tanh",
    "reverse_bits": "reverse_bits",
    "simd_sum": "simd_sum",
    "simd_max": "simd_max",
}
_BINOP_SYMBOLS_EPILOGUE = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}
# The IR owns this rule, because passes other than codegen depend on agreeing with it.
_resolve = mir.resolve


def _format_float_literal(v: float) -> str:
    """Format a float constant for MSL."""
    s = f"{v}f"
    if "." not in s and "e" not in s.lower():
        s = f"{v}.0f"
    return s


def _uses_thread_position(ops: list[mir.MOp]) -> bool:
    for op in ops:
        if isinstance(op, mir.ThreadPositionInGrid):
            return True
        if isinstance(op, mir.IfBlock) and _uses_thread_position(op.body):
            return True
        if isinstance(op, mir.MForLoop) and _uses_thread_position(op.body):
            return True
        if isinstance(op, mir.MSimdgroupRoleBlock) and _uses_thread_position(op.body):
            return True
    return False


def _uses_op_type(ops: list[mir.MOp], op_type) -> bool:
    for op in ops:
        if isinstance(op, op_type):
            return True
        if isinstance(op, mir.MForLoop) and _uses_op_type(op.body, op_type):
            return True
        if isinstance(op, mir.IfBlock) and _uses_op_type(op.body, op_type):
            return True
        if isinstance(op, mir.MWhileTrue) and _uses_op_type(op.body, op_type):
            return True
        if isinstance(op, mir.MSimdgroupRoleBlock) and _uses_op_type(op.body, op_type):
            return True
    return False


def _emit_tensor_view_decl(op, lines, indent, func):
    """Emit Metal tensor view declarations."""
    pad = "    " * indent
    a_name = _val_name_gemm(op.ptr_a, func)
    b_name = _val_name_gemm(op.ptr_b, func)
    c_name = _val_name_gemm(op.ptr_c, func)
    in_type = op.in_type
    out_type = op.out_type
    lines.append(f"{pad}auto tA = tensor<device {in_type}, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {a_name}, dextents<int32_t, 2>(K, M));")
    lines.append(f"{pad}auto tB = tensor<device {in_type}, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {b_name}, dextents<int32_t, 2>(N, K));")
    lines.append(f"{pad}auto tC = tensor<device {out_type}, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {c_name}, dextents<int32_t, 2>(N, M));")
    lines.append("")


def _fold_vector_lanes(operand, vector_type, reduce_op, msl_type, name, lines, pad):
    """Reduce a per-thread vector to a scalar with the same operator.

    A threadgroup reduction combines one value per thread, but inside a vectorized loop
    each thread holds `width` elements. Folding the lanes first keeps the reduction
    associative-equivalent to the scalar loop and matches what the accumulator path
    already emits for `scalar_acc OP vec4`.
    """
    width = int(vector_type[-1]) if vector_type[-1].isdigit() else 4
    lanes = ("x", "y", "z", "w")[:width]
    if len(lanes) < 2:
        return operand
    if reduce_op == "sum":
        expression = " + ".join(f"{operand}.{lane}" for lane in lanes)
    elif reduce_op in ("max", "min"):
        expression = f"{operand}.{lanes[0]}"
        for lane in lanes[1:]:
            expression = f"{reduce_op}({expression}, {operand}.{lane})"
    else:
        return operand
    folded = f"_lane_{name}"
    lines.append(f"{pad}{msl_type} {folded} = {expression};")
    return folded


def _val_name(val: mir.MValue, func: mir.MFunction) -> str:
    """Get the MSL variable name for a Metal IR value (element-wise)."""
    val = _resolve(val)
    if val.defining_op:
        # Constant folding: inline constants directly as literals
        if isinstance(val.defining_op, mir.MConstant):
            return _format_literal(val.defining_op.value, val.defining_op.dtype)
        # Constant folding: inline cast of constant as literal in target type
        if isinstance(val.defining_op, mir.MCast):
            inner = val.defining_op.value
            if inner.defining_op and isinstance(inner.defining_op, mir.MConstant):
                return _format_literal(inner.defining_op.value, val.defining_op.target_dtype)
        if isinstance(val.defining_op, mir.ThreadPositionInGrid):
            return "tid"
        if isinstance(val.defining_op, mir.ThreadgroupPositionInGrid):
            return "tgp_id_x"
        if isinstance(val.defining_op, mir.ThreadPositionInThreadgroup):
            return "lid"
        if isinstance(val.defining_op, mir.MSimdgroupId):
            return "sgid"
        if isinstance(val.defining_op, mir.MThreadInSimdgroup):
            return "slid"
    for p in func.params:
        if p.name == val.name:
            return p.name
    return val.name


def _val_name_gemm(val: mir.MValue, func: mir.MFunction) -> str:
    """Get the MSL variable name for a Metal IR value (GEMM)."""
    val = _resolve(val)
    if val.defining_op:
        # Constant folding: inline constants directly as literals
        if isinstance(val.defining_op, mir.MConstant):
            return _format_literal(val.defining_op.value, val.defining_op.dtype)
        # Constant folding: inline cast of constant as literal in target type
        if isinstance(val.defining_op, mir.MCast):
            inner = val.defining_op.value
            if inner.defining_op and isinstance(inner.defining_op, mir.MConstant):
                return _format_literal(inner.defining_op.value, val.defining_op.target_dtype)
        if isinstance(val.defining_op, mir.MSimdgroupId):
            return "sgid"
        if isinstance(val.defining_op, mir.MThreadInSimdgroup):
            return "slid"
        if isinstance(val.defining_op, mir.ThreadgroupPositionInGrid):
            axis_map = {0: "tgp_id.x", 1: "tgp_id.y", 2: "tgp_id.z"}
            return axis_map.get(val.defining_op.axis, "tgp_id.x")
    for p in func.params:
        if p.name == val.name:
            return p.name
    return val.name


def _format_literal(value, dtype: str) -> str:
    if dtype in ("f32", "f16", "bf16"):
        suffix = "f" if dtype == "f32" else "h"
        fval = float(value)
        if fval == float("inf"):
            return "INFINITY"
        if fval == float("-inf"):
            return "(-INFINITY)"
        if fval != fval:  # NaN
            return "NAN"
        return f"{fval!r}{suffix}"
    if dtype == "u32":
        return f"{int(value)}u"
    return str(int(value))
