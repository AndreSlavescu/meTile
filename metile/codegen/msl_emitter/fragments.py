"""Emit generic opaque matrix fragments without prescribing an attention algorithm."""

from metile.codegen.msl_emitter.common import (
    _BINOP_SYMBOLS,
    _UNARY_MSL,
    _val_name,
)
from metile.ir import metal_ir as mir
from metile.ir.types import MatrixFragmentType, ScalarType

FRAGMENT_OPS = (
    mir.MFragmentInit,
    mir.MFragmentLoad,
    mir.MFragmentStore,
    mir.MFragmentDot,
    mir.MFragmentElementwise,
    mir.MFragmentStateInit,
    mir.MFragmentStateRead,
    mir.MFragmentStateAssign,
)


def _emit_device_fragment(operation, lines, indent, function):
    pad = "    " * indent

    def value(operand):
        return _val_name(operand, function)

    row, column = value(operation.row), value(operation.column)
    rows, columns = (value(extent) for extent in operation.shape)
    base = (
        value(operation.base_offset)
        if isinstance(operation.base_offset, mir.MValue)
        else str(operation.base_offset)
    )
    pointer = value(operation.ptr)
    address = (
        f"{pointer} + long({base}) + long({row}) * {operation.row_stride} "
        f"+ long({column}) * {operation.column_stride}"
    )
    leading = operation.column_stride if operation.transpose else operation.row_stride
    transpose = ", ulong2(0), true" if operation.transpose else ""
    loading = isinstance(operation, mir.MFragmentLoad)
    fragment = operation.result.name if loading else value(operation.value)
    if loading:
        lines.append(f"{pad}{operation.result.type.to_msl()} {fragment};")
    intrinsic = "simdgroup_load" if loading else "simdgroup_store"
    if operation.full_tile:
        lines.append(f"{pad}{intrinsic}({fragment}, {address}, {leading}{transpose});")
        return
    complete = (
        f"long({row}) >= 0 && long({column}) >= 0 && "
        f"long({row}) + 8 <= long({rows}) && long({column}) + 8 <= long({columns})"
    )
    lines.append(f"{pad}if ({complete}) {{")
    lines.append(f"{pad}    {intrinsic}({fragment}, {address}, {leading}{transpose});")
    lines.append(f"{pad}}} else {{")
    scalar = ScalarType(operation.ptr.type.dtype).to_msl()
    lines.append(
        f"{pad}    threadgroup {scalar}* tile_scratch = {value(operation.scratch)} + sgid * 64;"
    )
    if not loading:
        lines.append(f"{pad}    simdgroup_store({fragment}, tile_scratch, 8);")
        lines.append(f"{pad}    simdgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append(f"{pad}    for (uint element = slid; element < 64; element += 32) {{")
    lines.append(f"{pad}        long tile_row = long({row}) + long(element / 8);")
    lines.append(f"{pad}        long tile_column = long({column}) + long(element % 8);")
    valid = (
        f"tile_row >= 0 && tile_row < long({rows}) && "
        f"tile_column >= 0 && tile_column < long({columns})"
    )
    offset = (
        f"long({base}) + tile_row * {operation.row_stride} "
        f"+ tile_column * {operation.column_stride}"
    )
    if loading:
        lines.append(
            f"{pad}        tile_scratch[element] = ({valid}) ? {pointer}[{offset}] : {scalar}(0);"
        )
    else:
        lines.append(f"{pad}        if ({valid}) {{")
        lines.append(f"{pad}            {pointer}[{offset}] = tile_scratch[element];")
        lines.append(f"{pad}        }}")
    lines.append(f"{pad}    }}")
    lines.append(f"{pad}    simdgroup_barrier(mem_flags::mem_threadgroup);")
    if loading:
        lines.append(f"{pad}    simdgroup_load({fragment}, tile_scratch, 8);")
        lines.append(f"{pad}    simdgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append(f"{pad}}}")


def emit_fragment(operation, lines, indent, function):
    pad = "    " * indent

    def value(operand):
        return _val_name(operand, function)

    if isinstance(operation, mir.MFragmentInit):
        scalar = ScalarType(operation.dtype).to_msl()
        lines.append(
            f"{pad}{operation.result.type.to_msl()} {operation.result.name} = "
            f"make_filled_simdgroup_matrix<{scalar}, 8, 8>(0.0f);"
        )
    elif isinstance(operation, (mir.MFragmentLoad, mir.MFragmentStore)):
        if operation.shape is not None:
            _emit_device_fragment(operation, lines, indent, function)
            return
        address = (
            f"{value(operation.ptr)} + {operation.base_offset} + "
            f"({value(operation.row)}) * {operation.row_stride} + "
            f"({value(operation.column)}) * {operation.column_stride}"
        )
        leading = operation.column_stride if operation.transpose else operation.row_stride
        transpose = ", ulong2(0), true" if operation.transpose else ""
        if isinstance(operation, mir.MFragmentLoad):
            lines.append(f"{pad}{operation.result.type.to_msl()} {operation.result.name};")
            lines.append(
                f"{pad}simdgroup_load({operation.result.name}, {address}, {leading}{transpose});"
            )
        else:
            lines.append(
                f"{pad}simdgroup_store({value(operation.value)}, {address}, {leading}{transpose});"
            )
    elif isinstance(operation, mir.MFragmentDot):
        lines.append(f"{pad}{operation.result.type.to_msl()} {operation.result.name};")
        lines.append(
            f"{pad}simdgroup_multiply_accumulate({operation.result.name}, "
            f"{value(operation.left)}, {value(operation.right)}, {value(operation.accumulator)});"
        )
    elif isinstance(operation, mir.MFragmentElementwise):
        scalar = ScalarType(operation.dtype).to_msl()
        lines.append(f"{pad}{operation.result.type.to_msl()} {operation.result.name};")
        for element in range(2):
            operands = [
                f"{value(operand)}.thread_elements()[{element}]"
                if isinstance(operand.type, MatrixFragmentType)
                else f"{scalar}({value(operand)})"
                for operand in operation.operands
            ]
            if operation.operation == "cast":
                expression = f"{scalar}({operands[0]})"
            elif operation.operation == "fma":
                expression = f"fma({', '.join(operands)})"
            elif operation.operation in {"min", "max"}:
                expression = f"{operation.operation}({operands[0]}, {operands[1]})"
            elif len(operands) == 2:
                expression = f"{operands[0]} {_BINOP_SYMBOLS[operation.operation]} {operands[1]}"
            elif operation.operation == "neg":
                expression = f"-{operands[0]}"
            else:
                expression = f"{_UNARY_MSL[operation.operation]}({operands[0]})"
            lines.append(
                f"{pad}{operation.result.name}.thread_elements()[{element}] = {expression};"
            )
    elif isinstance(operation, mir.MFragmentStateInit):
        lines.append(
            f"{pad}{operation.value.type.to_msl()} {operation.state_name} = {value(operation.value)};"
        )
    elif isinstance(operation, mir.MFragmentStateRead):
        lines.append(
            f"{pad}{operation.result.type.to_msl()} {operation.result.name} = {value(operation.state)};"
        )
    elif isinstance(operation, mir.MFragmentStateAssign):
        lines.append(f"{pad}{operation.state_name} = {value(operation.value)};")
    else:
        raise ValueError("unsupported matrix fragment operation")
