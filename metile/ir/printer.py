from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir


def print_tile_ir(func: tir.Function) -> str:
    """Pretty-print a Tile IR function."""
    lines = []
    params_str = ", ".join(f"%{p.name}: {p.type}" for p in func.params)
    lines.append(f"func @{func.name}({params_str}) {{")
    if func.constexprs:
        ce = ", ".join(f"{k}={v}" for k, v in func.constexprs.items())
        lines.append(f"  // constexprs: {ce}")

    for op in func.ops:
        lines.append("  " + _format_tile_op(op))

    lines.append("}")
    return "\n".join(lines)


def _format_tile_op(op: tir.Op) -> str:
    result = op.result
    prefix = f"%{result.name} = " if result else ""
    suffix = f" : {result.type}" if result else ""

    if isinstance(op, tir.ProgramId):
        return f"{prefix}program_id(axis={op.axis}){suffix}"
    elif isinstance(op, tir.Constant):
        return f"{prefix}constant({op.value}){suffix}"
    elif isinstance(op, tir.Arange):
        start = f"%{op.start.name}" if op.start else "0"
        return f"{prefix}arange({start}, {start}+{op.size}){suffix}"
    elif isinstance(op, tir.BinOp):
        return f"{prefix}{op.op}(%{op.lhs.name}, %{op.rhs.name}){suffix}"
    elif isinstance(op, tir.Unary):
        return f"{prefix}{op.op}(%{op.operand.name}){suffix}"
    elif isinstance(op, tir.Select):
        return f"{prefix}select(%{op.condition.name}, %{op.true_val.name}, %{op.false_val.name}){suffix}"
    elif isinstance(op, tir.Compare):
        return f"{prefix}cmp_{op.predicate}(%{op.lhs.name}, %{op.rhs.name}){suffix}"
    elif isinstance(op, tir.Load):
        mask = f", mask=%{op.mask.name}" if op.mask else ""
        return f"{prefix}load(%{op.ptr.name}, %{op.offsets.name}{mask}){suffix}"
    elif isinstance(op, tir.Store):
        mask = f", mask=%{op.mask.name}" if op.mask else ""
        return f"store(%{op.ptr.name}, %{op.offsets.name}, %{op.value.name}{mask})"
    elif isinstance(op, tir.PtrOffset):
        return f"{prefix}ptr_offset(%{op.ptr.name}, %{op.offsets.name}){suffix}"
    elif isinstance(op, tir.Zeros):
        return f"{prefix}zeros(shape={op.shape}, dtype={op.dtype}){suffix}"
    elif isinstance(op, tir.Dot):
        return f"{prefix}dot(%{op.a.name}, %{op.b.name}, %{op.acc.name}){suffix}"
    elif isinstance(op, tir.TileLoad):
        return f"{prefix}tile_load(%{op.ptr.name}, %{op.row_offset.name}, %{op.col_offset.name}, stride=%{op.stride.name}, shape={op.tile_shape}){suffix}"
    elif isinstance(op, tir.TileStore):
        return f"tile_store(%{op.ptr.name}, %{op.row_offset.name}, %{op.col_offset.name}, stride=%{op.stride.name}, %{op.value.name}, shape={op.tile_shape})"
    elif isinstance(op, tir.ForRange):
        return f"for %{op.iv.name} in range(%{op.start.name}, %{op.end.name}, step={op.step}) {{ ... {len(op.body)} ops }}"
    else:
        return f"<unknown op: {type(op).__name__}>"


def print_metal_ir(func: mir.MFunction) -> str:
    """Pretty-print a Metal IR function."""
    lines = []
    params_str = ", ".join(f"%{p.name}: {p.type}" for p in func.params)
    lines.append(f"metal_func @{func.name}({params_str}) {{")
    lines.append(f"  // grid={func.grid}, threadgroup={func.threadgroup_size}")

    for op in func.ops:
        _format_metal_op(op, lines, indent=1)

    lines.append("}")
    return "\n".join(lines)


def _format_metal_op(op: mir.MOp, lines: list[str], indent: int = 1):
    pad = "  " * indent
    result = op.result
    prefix = f"%{result.name} = " if result else ""
    suffix = f" : {result.type}" if result else ""

    if isinstance(op, mir.ThreadPositionInGrid):
        lines.append(f"{pad}{prefix}thread_position_in_grid.{'xyz'[op.axis]}{suffix}")
    elif isinstance(op, mir.ThreadgroupPositionInGrid):
        lines.append(f"{pad}{prefix}threadgroup_position_in_grid.{'xyz'[op.axis]}{suffix}")
    elif isinstance(op, mir.MConstant):
        lines.append(f"{pad}{prefix}constant({op.value}){suffix}")
    elif isinstance(op, mir.MBinOp):
        lines.append(f"{pad}{prefix}{op.op}(%{op.lhs.name}, %{op.rhs.name}){suffix}")
    elif isinstance(op, mir.MCast):
        lines.append(f"{pad}{prefix}cast(%{op.value.name}, {op.target_dtype}){suffix}")
    elif isinstance(op, mir.MUnary):
        lines.append(f"{pad}{prefix}{op.op}(%{op.operand.name}){suffix}")
    elif isinstance(op, mir.MSelect):
        lines.append(
            f"{pad}{prefix}select(%{op.condition.name}, %{op.true_val.name}, %{op.false_val.name}){suffix}"
        )
    elif isinstance(op, mir.MCompare):
        lines.append(f"{pad}{prefix}cmp_{op.predicate}(%{op.lhs.name}, %{op.rhs.name}){suffix}")
    elif isinstance(op, mir.DeviceLoad):
        lines.append(f"{pad}{prefix}device_load(%{op.ptr.name}, %{op.index.name}){suffix}")
    elif isinstance(op, mir.DeviceStore):
        lines.append(f"{pad}device_store(%{op.ptr.name}, %{op.index.name}, %{op.value.name})")
    elif isinstance(op, mir.IfBlock):
        lines.append(f"{pad}if %{op.condition.name} {{")
        for body_op in op.body:
            _format_metal_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    else:
        lines.append(f"{pad}<unknown: {type(op).__name__}>")
