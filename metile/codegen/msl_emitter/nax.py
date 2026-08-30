"""MSL emission for NAX fragment operations."""

from __future__ import annotations

from metile.codegen.msl_emitter.common import (
    _val_name_gemm,
)
from metile.ir.types import PtrType, ScalarType


def _emit_nax_vector(
    lines,
    pad,
    name,
    row0,
    row1,
    ptr,
    stride,
    element_type,
    condition0=None,
    condition1=None,
):
    zero = f"{element_type}4(0)"
    load0 = f"*((device const {element_type}4*)(&{ptr}[{row0} * {stride}]))"
    load1 = f"*((device const {element_type}4*)(&{ptr}[{row1} * {stride}]))"
    if condition0:
        load0 = f"({condition0}) ? {load0} : {zero}"
    if condition1:
        load1 = f"({condition1}) ? {load1} : {zero}"
    lines.append(f"{pad}const {element_type}4 {name}0 = {load0};")
    lines.append(f"{pad}const {element_type}4 {name}1 = {load1};")
    lines.append(
        f"{pad}const metal::vec<{element_type}, 8> {name} = metal::vec<{element_type}, 8>("
    )
    lines.append(
        f"{pad}    {name}0.x, {name}0.y, {name}0.z, {name}0.w, "
        f"{name}1.x, {name}1.y, {name}1.z, {name}1.w);"
    )


def _emit_nax_quantized_vector(
    lines, pad, name, row, col, values, scales, bits, decoded_scale=None
):
    lines.append(f"{pad}const uint {name}_element = ({row}) * N + ({col});")
    if decoded_scale is None:
        lines.append(
            f"{pad}const uchar4 {name}_scales = "
            f"*((device const uchar4*)(&{scales}[(({row}) >> 5u) * N + ({col})]));"
        )
        decoded_scale = f"mtile_decode_e8m0({name}_scales)"
    if bits == 4:
        lines.append(
            f"{pad}const ushort {name}_packed = "
            f"*((device const ushort*)(&{values}[{name}_element >> 1u]));"
        )
        lines.append(
            f"{pad}const uchar4 {name}_quantized = uchar4("
            + ", ".join(f"uchar(({name}_packed >> {shift}u) & 15u)" for shift in (0, 4, 8, 12))
            + ");"
        )
        decoder = "mtile_decode_e2m1"
    else:
        lines.append(
            f"{pad}const uchar4 {name}_packed = "
            f"*((device const uchar4*)(&{values}[{name}_element]));"
        )
        lines.append(f"{pad}const uchar4 {name}_quantized = {name}_packed;")
        decoder = "mtile_decode_e4m3"
    lines.append(f"{pad}const float4 {name} = {decoded_scale} * {decoder}({name}_quantized);")


def _emit_nax_tile_layout(op, lines, indent):
    pad = "    " * indent
    if op.m and op.n and op.k:
        lines.append(f"{pad}constexpr uint M = {op.m}u;")
        lines.append(f"{pad}constexpr uint N = {op.n}u;")
        lines.append(f"{pad}constexpr uint K = {op.k}u;")
    lines.append(f"{pad}const uint sg_row = sgid / {op.wn}u;")
    lines.append(f"{pad}const uint sg_col = sgid % {op.wn}u;")
    lines.append(f"{pad}const uint tile_row = pid_m * {op.block_m}u + sg_row * 32u;")
    lines.append(f"{pad}const uint tile_col = pid_n * {op.block_n}u + sg_col * 32u;")
    lines.append(f"{pad}const uint qid = slid >> 2u;")
    lines.append(f"{pad}const uint frag_m = ((qid & 4u) | ((slid >> 1u) & 3u));")
    lines.append(f"{pad}const uint frag_n = ((qid & 2u) | (slid & 1u)) * 4u;")


def _emit_nax_accumulator_init(op, lines, indent):
    pad = "    " * indent
    for name in op.names:
        lines.append(f"{pad}metal::vec<float, 8> {name} = metal::vec<float, 8>(0.0f);")


def _emit_nax_accumulator_reset(op, lines, indent):
    pad = "    " * indent
    for name in op.names:
        lines.append(f"{pad}{name} = metal::vec<float, 8>(0.0f);")


def _emit_nax_matmul2d_decl(op, lines, indent):
    pad = "    " * indent
    lines.append(f"{pad}constexpr auto nax_desc = matmul2d_descriptor(")
    lines.append(f"{pad}    {op.m}, {op.n}, {op.k}, false, false, true,")
    lines.append(f"{pad}    matmul2d_descriptor::mode::multiply_accumulate);")
    lines.append(f"{pad}matmul2d<nax_desc, execution_simdgroup> nax_mma;")
    lines.append(
        f"{pad}auto nax_a = nax_mma.get_left_input_cooperative_tensor<"
        f"{op.left_type}, {op.right_type}, {op.accumulator_type}>();"
    )
    lines.append(
        f"{pad}auto nax_b = nax_mma.get_right_input_cooperative_tensor<"
        f"{op.left_type}, {op.right_type}, {op.accumulator_type}>();"
    )
    lines.append(f"{pad}auto nax_c = nax_mma.get_destination_cooperative_tensor<")
    lines.append(f"{pad}    decltype(nax_a), decltype(nax_b), {op.accumulator_type}>();")
    lines.append("")


def _emit_nax_load_fragment(op, lines, indent, func):
    pad = "    " * indent
    ptr = _val_name_gemm(op.ptr, func)
    dtype = op.ptr.type.dtype if isinstance(op.ptr.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    k = "uint(k)" if not op.k_offset else f"uint(k) + {op.k_offset}u"
    if op.operand == "left":
        row = "tile_row + frag_m"
        if op.row_offset:
            row = f"tile_row + {op.row_offset}u + frag_m"
        col = f"K + {k} + frag_n"
        if op.col_offset:
            col += f" + {op.col_offset}u"
    elif op.operand == "right":
        row = f"{k} + frag_m"
        if op.row_offset:
            row += f" + {op.row_offset}u"
        col = "N + tile_col + frag_n"
        if op.col_offset:
            col = f"N + tile_col + {op.col_offset}u + frag_n"
    else:
        raise ValueError(f"unknown NAX fragment operand: {op.operand}")
    condition0 = condition1 = None
    if op.operand == "left" and op.row_bound:
        condition0 = f"({row}) < {op.row_bound}u"
        condition1 = f"({row} + 8u) < {op.row_bound}u"
    _emit_nax_vector(
        lines,
        pad,
        op.name,
        f"({row})",
        f"({row} + 8u)",
        ptr,
        col,
        element_type,
        condition0,
        condition1,
    )


def _emit_nax_load_block_scaled_fragment(op, lines, indent, func):
    pad = "    " * indent
    values = _val_name_gemm(op.ptr_values, func)
    k = "uint(k)" if not op.k_offset else f"uint(k) + {op.k_offset}u"
    col = "tile_col + frag_n"
    if op.col_offset:
        col = f"tile_col + {op.col_offset}u + frag_n"
    _emit_nax_quantized_vector(
        lines,
        pad,
        f"{op.name}0",
        f"{k} + frag_m",
        col,
        values,
        None,
        op.bits,
        op.scale,
    )
    _emit_nax_quantized_vector(
        lines,
        pad,
        f"{op.name}1",
        f"{k} + frag_m + 8u",
        col,
        values,
        None,
        op.bits,
        op.scale,
    )
    lines.append(
        f"{pad}const metal::vec<{op.fragment_type}, 8> {op.name} = "
        f"metal::vec<{op.fragment_type}, 8>("
    )
    cast = "" if op.fragment_type == "float" else "bfloat"
    components = []
    for vector in (f"{op.name}0", f"{op.name}1"):
        for field in "xyzw":
            component = f"{vector}.{field}"
            components.append(component if not cast else f"{cast}({component})")
    lines.append(f"{pad}    {', '.join(components[:4])}, {', '.join(components[4:])});")


def _emit_nax_load_block_scale(op, lines, indent, func):
    pad = "    " * indent
    scales = _val_name_gemm(op.ptr_scales, func)
    k = "uint(k)" if not op.k_offset else f"uint(k) + {op.k_offset}u"
    col = "tile_col + frag_n"
    if op.col_offset:
        col = f"tile_col + {op.col_offset}u + frag_n"
    lines.append(
        f"{pad}const uchar4 {op.name}_bits = "
        f"*((device const uchar4*)(&{scales}[(({k} + frag_m) >> 5u) * N + ({col})]));"
    )
    lines.append(f"{pad}const float4 {op.name} = mtile_decode_e8m0({op.name}_bits);")


def _emit_nax_load_affine_parameters(op, lines, indent, func):
    pad = "    " * indent
    scales = _val_name_gemm(op.ptr_scales, func)
    biases = _val_name_gemm(op.ptr_biases, func)
    k = "uint(k)" if not op.k_offset else f"uint(k) + {op.k_offset}u"
    col = "tile_col + frag_n"
    if op.col_offset:
        col = f"tile_col + {op.col_offset}u + frag_n"
    index = f"(({k} + frag_m) / {op.group_size}u) * N + ({col})"
    lines.append(
        f"{pad}const half4 {op.scale_name} = *((device const half4*)(&{scales}[{index}]));"
    )
    lines.append(f"{pad}const half4 {op.bias_name} = *((device const half4*)(&{biases}[{index}]));")


def _emit_nax_load_affine_fragment(op, lines, indent, func):
    pad = "    " * indent
    values = _val_name_gemm(op.ptr_values, func)
    k = "uint(k)" if not op.k_offset else f"uint(k) + {op.k_offset}u"
    col = "tile_col + frag_n"
    if op.col_offset:
        col = f"tile_col + {op.col_offset}u + frag_n"

    vectors = []
    for suffix, row in (("0", f"{k} + frag_m"), ("1", f"{k} + frag_m + 8u")):
        name = f"{op.name}{suffix}"
        lines.append(f"{pad}const uint {name}_element = ({row}) * N + ({col});")
        lines.append(
            f"{pad}const ushort {name}_packed = "
            f"*((device const ushort*)(&{values}[{name}_element >> 1u]));"
        )
        lines.append(
            f"{pad}const half4 {name}_quantized = half4("
            + ", ".join(f"half(({name}_packed >> {shift}u) & 15u)" for shift in (0, 4, 8, 12))
            + ");"
        )
        lines.append(f"{pad}const half4 {name} = {name}_quantized * {op.scale} + {op.bias};")
        vectors.append(name)

    lines.append(
        f"{pad}const metal::vec<{op.fragment_type}, 8> {op.name} = "
        f"metal::vec<{op.fragment_type}, 8>("
    )
    components = []
    for vector in vectors:
        components.extend(f"{op.fragment_type}({vector}.{field})" for field in "xyzw")
    lines.append(f"{pad}    {', '.join(components[:4])}, {', '.join(components[4:])});")


def _emit_nax_pack_right(op, lines, indent):
    pad = "    " * indent
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    nax_b[i] = {op.low}[i];")
    lines.append(f"{pad}    nax_b[8 + i] = {op.high}[i];")
    lines.append(f"{pad}}}")


def _emit_nax_fma_fragment(op, lines, indent):
    pad = "    " * indent
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    nax_a[i] = {op.left}[i];")
    lines.append(f"{pad}    nax_c[i] = {op.destination_low}[i];")
    lines.append(f"{pad}    nax_c[8 + i] = {op.destination_high}[i];")
    lines.append(f"{pad}}}")
    lines.append(f"{pad}nax_mma.run(nax_a, nax_b, nax_c);")
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    {op.destination_low}[i] = nax_c[i];")
    lines.append(f"{pad}    {op.destination_high}[i] = nax_c[8 + i];")
    lines.append(f"{pad}}}")


def _emit_nax_binary_fragment(op, lines, indent):
    pad = "    " * indent
    destination = op.destination or op.left
    raw_left = f"{op.left}[i]"
    raw_right = f"{op.right}[i]"
    left = raw_left
    right = raw_right
    if op.round_inputs:
        left = f"float({op.round_inputs}({left}))"
        right = f"float({op.round_inputs}({right}))"
    if op.operation == "add":
        expression = f"{left} + {right}"
    elif op.operation == "multiply":
        expression = f"{left} * {right}"
    elif op.operation == "swiglu":
        exponential = "fast::exp" if op.fast_math else "exp"
        if op.round_intermediates:
            low_type = op.round_intermediates
            lines.append(f"{pad}#pragma clang loop unroll(full)")
            lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
            lines.append(f"{pad}    const {low_type} swiglu_left = {low_type}({raw_left});")
            lines.append(f"{pad}    const {low_type} swiglu_right = {low_type}({raw_right});")
            lines.append(
                f"{pad}    const auto swiglu_y = 1 / (1 + {exponential}(abs(swiglu_left)));"
            )
            lines.append(
                f"{pad}    const {low_type} swiglu_sigmoid = "
                f"(swiglu_left < {low_type}(0)) ? swiglu_y : 1 - swiglu_y;"
            )
            lines.append(
                f"{pad}    const {low_type} swiglu_activation = swiglu_left * swiglu_sigmoid;"
            )
            lines.append(
                f"{pad}    {destination}[i] = float({low_type}(swiglu_activation * swiglu_right));"
            )
            lines.append(f"{pad}}}")
            return
        activation = f"({left} / (1.0f + {exponential}(-{left})))"
        expression = f"{activation} * {right}"
    else:
        raise ValueError(f"unknown NAX binary fragment operation: {op.operation}")
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    {destination}[i] = {expression};")
    lines.append(f"{pad}}}")


def _emit_nax_scratch_index(op):
    return f"((sgid * {op.slots_per_simdgroup}u + {op.slot}u) * 256u + uint(i) * 32u + slid)"


def _emit_nax_spill_fragment(op, lines, indent):
    pad = "    " * indent
    index = _emit_nax_scratch_index(op)
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    {op.scratch_name}[{index}] = {op.source}[i];")
    lines.append(f"{pad}}}")


def _emit_nax_reload_fragment(op, lines, indent):
    pad = "    " * indent
    index = _emit_nax_scratch_index(op)
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    lines.append(f"{pad}    {op.destination}[i] = {op.scratch_name}[{index}];")
    lines.append(f"{pad}}}")


def _emit_nax_store_fragment(op, lines, indent, func):
    pad = "    " * indent
    ptr_c = _val_name_gemm(op.ptr_c, func)
    dtype = op.ptr_c.type.dtype if isinstance(op.ptr_c.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    for component_offset, source_offset in ((op.row_offset, 0), (op.row_offset + 8, 4)):
        row = f"tile_row + {component_offset}u + frag_m"
        components = [f"{op.source}[{source_offset + index}]" for index in range(4)]
        if dtype == "bf16":
            components = [f"bfloat({component})" for component in components]
        statement = (
            f"*((device {element_type}4*)(&{ptr_c}[({row}) * N + tile_col "
            f"+ {op.col_offset}u + frag_n])) = {element_type}4("
            f"{', '.join(components)});"
        )
        if op.row_bound:
            lines.append(f"{pad}if (({row}) < {op.row_bound}u) {{ {statement} }}")
        else:
            lines.append(f"{pad}{statement}")
