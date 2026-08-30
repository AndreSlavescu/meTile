"""MSL emission for SIMD-group and cooperative-tensor operations."""

from __future__ import annotations

from metile.codegen.msl_emitter.common import (
    _val_name_gemm,
)
from metile.ir import metal_ir as mir
from metile.ir.types import PtrType, ScalarType


def _emit_cooperative_load(
    op: mir.MCooperativeLoad, lines: list[str], indent: int, func: mir.MFunction
):
    """Emit cooperative tile loading code.

    When op.load_layout is set, tile dimensions and smem stride are derived
    from the layout algebra rather than the legacy tile_rows/tile_cols/dst_stride
    fields. This is the migration path toward layout-driven codegen.
    """
    pad = "    " * indent
    ptr = _val_name_gemm(op.device_ptr, func)
    linear_tid = _val_name_gemm(op.linear_tid, func)
    kb_fallback = op.kb_expr if op.kb_expr is not None else "kb"
    row_off = _val_name_gemm(op.row_offset, func) if op.row_offset else kb_fallback
    col_off = _val_name_gemm(op.col_offset, func) if op.col_offset else kb_fallback
    stride = _val_name_gemm(op.src_stride, func)
    tg = op.tg_array

    # Prefer layout-derived dimensions when available
    if op.load_layout is not None:
        TR = op.load_layout.tile.rows
        TC = op.load_layout.tile.cols
        DS = op.load_layout.tile.smem_stride
        TG_SIZE = op.load_layout.num_threads
    else:
        TR, TC = op.tile_rows, op.tile_cols
        DS = op.dst_stride
        TG_SIZE = op.tg_size

    if op.vec_size == 4 and not op.bounds_check:
        # Vectorized load (no bounds checking)
        vec_type = "float4" if op.elem_type == "float" else "half4"
        cols_v = TC // 4
        total_v = TR * cols_v
        # Use bitwise ops when cols_v is power of 2
        is_pow2 = cols_v > 0 and (cols_v & (cols_v - 1)) == 0
        cols_v_shift = cols_v.bit_length() - 1 if is_pow2 else 0
        lines.append(f"{pad}{{")
        lines.append(f"{pad}    constexpr uint _cols_v = {cols_v};")
        lines.append(f"{pad}    for (uint _i = {linear_tid}; _i < {total_v}u; _i += {TG_SIZE}u) {{")
        if is_pow2:
            lines.append(f"{pad}        uint _r = _i >> {cols_v_shift};")
            lines.append(f"{pad}        uint _c = (_i & {cols_v - 1}u) * 4;")
        else:
            lines.append(f"{pad}        uint _r = _i / _cols_v;")
            lines.append(f"{pad}        uint _c = (_i % _cols_v) * 4;")
        # Row offset for A-style (row is block_row + _r), for B-style (row is kb + _r)
        if op.row_offset is not None:
            lines.append(f"{pad}        uint _dev = ({row_off} + _r) * {stride} + {col_off} + _c;")
        else:
            # B-style: row is loop IV, col is block_col
            lines.append(
                f"{pad}        uint _dev = ({kb_fallback} + _r) * {stride} + {col_off} + _c;"
            )
        lines.append(f"{pad}        {vec_type} _v = *((const device {vec_type}*)(&{ptr}[_dev]));")
        lines.append(f"{pad}        uint _base = _r * {DS} + _c;")
        if op.swizzle_bits > 0:
            # XOR swizzle: scalar writes with permuted addresses
            sw_mask = (1 << op.swizzle_bits) - 1
            for j in range(4):
                lines.append(
                    f"{pad}        {{ uint _off = _base + {j}u; "
                    f"{tg}[_off ^ ((_off >> {op.swizzle_shift}u) & {sw_mask}u)] = _v[{j}]; }}"
                )
        # Use vectorized threadgroup write when stride is 4-aligned
        elif DS % 4 == 0:
            tg_vec = "float4" if op.elem_type == "float" else "half4"
            lines.append(f"{pad}        *((threadgroup {tg_vec}*)(&{tg}[_base])) = _v;")
        else:
            lines.append(f"{pad}        {tg}[_base] = _v[0]; {tg}[_base+1] = _v[1];")
            lines.append(f"{pad}        {tg}[_base+2] = _v[2]; {tg}[_base+3] = _v[3];")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")

    elif not op.bounds_check:
        # Scalar load without bounds checking
        total = TR * TC
        lines.append(f"{pad}for (uint _i = {linear_tid}; _i < {total}u; _i += {TG_SIZE}u) {{")
        lines.append(f"{pad}    uint _r = _i / {TC}u, _c = _i % {TC}u;")
        if op.row_offset is not None:
            dev_expr = f"{ptr}[({row_off} + _r) * {stride} + {col_off} + _c]"
        else:
            dev_expr = f"{ptr}[({kb_fallback} + _r) * {stride} + {col_off} + _c]"
        if op.swizzle_bits > 0:
            sw_mask = (1 << op.swizzle_bits) - 1
            lines.append(f"{pad}    uint _off = _r * {DS}u + _c;")
            lines.append(
                f"{pad}    {tg}[_off ^ ((_off >> {op.swizzle_shift}u) & {sw_mask}u)] = {dev_expr};"
            )
        else:
            lines.append(f"{pad}    {tg}[_r * {DS} + _c] = {dev_expr};")
        lines.append(f"{pad}}}")

    else:
        # Scalar load with bounds checking
        total = TR * TC
        row_bound = _val_name_gemm(op.row_bound, func) if op.row_bound else "M"
        col_bound = _val_name_gemm(op.col_bound, func) if op.col_bound else "K"
        lines.append(f"{pad}for (uint _i = {linear_tid}; _i < {total}u; _i += {TG_SIZE}u) {{")
        lines.append(f"{pad}    uint _r = _i / {TC}u, _c = _i % {TC}u;")
        if op.row_offset is not None:
            lines.append(f"{pad}    uint _gr = {row_off} + _r, _gc = {kb_fallback} + _c;")
        else:
            lines.append(f"{pad}    uint _gr = {kb_fallback} + _r, _gc = {col_off} + _c;")
        dev_val = (
            f"(_gr < uint({row_bound}) && _gc < uint({col_bound})) "
            f"? {ptr}[_gr * {stride} + _gc] : {op.elem_type}(0)"
        )
        if op.swizzle_bits > 0:
            sw_mask = (1 << op.swizzle_bits) - 1
            lines.append(f"{pad}    uint _off = _r * {DS}u + _c;")
            lines.append(
                f"{pad}    {tg}[_off ^ ((_off >> {op.swizzle_shift}u) & {sw_mask}u)] = {dev_val};"
            )
        else:
            lines.append(f"{pad}    {tg}[_r * {DS} + _c] = {dev_val};")
        lines.append(f"{pad}}}")


def _emit_sg_load_swizzle(
    lines: list[str],
    pad: str,
    mat_var: str,
    shared_array: str,
    row_base: str,
    col_base: str,
    stride: int,
    swizzle_bits: int,
    swizzle_shift: int,
):
    """Emit manual thread_elements() load with XOR swizzle.

    Replaces simdgroup_load when shared memory uses XOR address permutation.
    Empirically verified Apple GPU simdgroup_matrix<float,8,8> mapping:
      _row = ((slid & 7u) >> 1u) + ((slid >> 4u) << 2u)
      _col = ((slid & 1u) << 1u) | ((slid & 8u) >> 1u)
      element 0: (row_base + _row, col_base + _col)
      element 1: (row_base + _row, col_base + _col + 1)
    """
    mask = (1 << swizzle_bits) - 1
    lines.append(f"{pad}{{")
    lines.append(f"{pad}    uint _row = ((slid & 7u) >> 1u) + ((slid >> 4u) << 2u);")
    lines.append(f"{pad}    uint _col = ((slid & 1u) << 1u) | ((slid & 8u) >> 1u);")
    lines.append(
        f"{pad}    uint _off0 = (uint({row_base}) + _row) * {stride}u + uint({col_base}) + _col;"
    )
    lines.append(f"{pad}    uint _off1 = _off0 + 1u;")
    lines.append(
        f"{pad}    {mat_var}.thread_elements()[0] = "
        f"{shared_array}[_off0 ^ ((_off0 >> {swizzle_shift}u) & {mask}u)];"
    )
    lines.append(
        f"{pad}    {mat_var}.thread_elements()[1] = "
        f"{shared_array}[_off1 ^ ((_off1 >> {swizzle_shift}u) & {mask}u)];"
    )
    lines.append(f"{pad}}}")


def _emit_simdgroup_acc_decl(op, lines, indent):
    """Emit accumulator array + temp tile declarations + zero-init."""
    pad = "    " * indent
    acc_t = getattr(op, "acc_type", op.in_type)
    lines.append(f"{pad}simdgroup_matrix<{acc_t}, 8, 8> {op.acc_name}[{op.num_8m}][{op.num_8n}];")
    for mi in range(op.num_8m):
        for ni in range(op.num_8n):
            lines.append(
                f"{pad}{op.acc_name}[{mi}][{ni}] = make_filled_simdgroup_matrix<{acc_t}, 8, 8>(0.0f);"
            )
    # Temp tile arrays for loads
    lines.append(f"{pad}simdgroup_matrix<{op.in_type}, 8, 8> a_tile[{op.num_8m}];")
    lines.append(f"{pad}simdgroup_matrix<{op.in_type}, 8, 8> b_tile[{op.num_8n}];")


def _emit_simdgroup_load(op, lines, indent, func):
    """Emit simdgroup_load or swizzled manual load for one 8x8 tile."""
    pad = "    " * indent
    sg_off = _val_name_gemm(op.sg_offset, func)

    if op.swizzle_bits > 0:
        # Manual thread_elements() load with XOR swizzle
        if op.is_b:
            row_base = op.kk_var
            col_base = f"{sg_off} + {op.tile_offset}"
        else:
            row_base = f"{sg_off} + {op.tile_offset}"
            col_base = op.kk_var
        _emit_sg_load_swizzle(
            lines,
            pad,
            f"{op.tile_name}[{op.tile_idx}]",
            op.src_array,
            row_base,
            col_base,
            op.stride,
            op.swizzle_bits,
            op.swizzle_shift,
        )
    else:
        # Standard simdgroup_load
        if op.is_b:
            offset = f"{op.kk_var} * {op.stride} + ({sg_off} + {op.tile_offset})"
        else:
            offset = f"({sg_off} + {op.tile_offset}) * {op.stride} + {op.kk_var}"
        lines.append(
            f"{pad}simdgroup_load({op.tile_name}[{op.tile_idx}], "
            f"{op.src_array} + {offset}, {op.stride});"
        )


def _emit_simdgroup_mma(op, lines, indent):
    """Emit simdgroup_multiply_accumulate for one (mi, ni) pair."""
    pad = "    " * indent
    lines.append(
        f"{pad}simdgroup_multiply_accumulate({op.acc_name}[{op.mi}][{op.ni}], "
        f"{op.a_tile}[{op.mi}], {op.b_tile}[{op.ni}], {op.acc_name}[{op.mi}][{op.ni}]);"
    )


def _emit_simdgroup_store(op, lines, indent, func):
    """Emit bounds-checked simdgroup_store for one 8x8 accumulator tile."""
    pad = "    " * indent
    ptr = _val_name_gemm(op.device_ptr, func)
    br = _val_name_gemm(op.block_row, func)
    bc = _val_name_gemm(op.block_col, func)
    sr = _val_name_gemm(op.sg_row, func)
    sc = _val_name_gemm(op.sg_col, func)
    stride = _val_name_gemm(op.stride, func)
    M = _val_name_gemm(op.m_bound, func)
    N = _val_name_gemm(op.n_bound, func)

    lines.append(f"{pad}{{")
    lines.append(f"{pad}    uint _or = {br} + {sr} + {op.mi_offset};")
    lines.append(f"{pad}    uint _oc = {bc} + {sc} + {op.ni_offset};")
    lines.append(f"{pad}    if (_or + 8 <= uint({M}) && _oc + 8 <= uint({N})) {{")
    if op.out_type == "half" and op.acc_type == "float":
        lines.append(f"{pad}        simdgroup_matrix<half, 8, 8> _out;")
        lines.append(
            f"{pad}        _out.thread_elements()[0] = half({op.acc_name}[{op.mi}][{op.ni}].thread_elements()[0]);"
        )
        lines.append(
            f"{pad}        _out.thread_elements()[1] = half({op.acc_name}[{op.mi}][{op.ni}].thread_elements()[1]);"
        )
        lines.append(f"{pad}        simdgroup_store(_out, {ptr} + _or * {stride} + _oc, {stride});")
    else:
        lines.append(
            f"{pad}        simdgroup_store({op.acc_name}[{op.mi}][{op.ni}], {ptr} + _or * {stride} + _oc, {stride});"
        )
    lines.append(f"{pad}    }}")
    lines.append(f"{pad}}}")


def _emit_simdgroup_qmv_layout(op, lines, indent):
    pad = "    " * indent
    stride = op.outputs_per_simdgroup * op.simdgroups_per_threadgroup
    lines.append(
        f"{pad}const uint qmv_output_base = tgp_id.x * {stride}u "
        f"+ sgid * {op.outputs_per_simdgroup}u;"
    )


def _emit_dot_accumulator_init(op, lines, indent):
    pad = "    " * indent
    for row in range(op.rows):
        for output in range(op.outputs_per_simdgroup):
            lines.append(f"{pad}float qmv_dot_{row}_{output} = 0.0f;")


def _emit_dot_accumulate(op, lines, indent, func):
    if op.elements_per_lane != 4:
        raise ValueError("SIMDgroup dot accumulation requires four elements per lane")
    pad = "    " * indent
    input_ptr = _val_name_gemm(op.ptr_input, func)
    weight_ptr = _val_name_gemm(op.ptr_weight, func)
    dtype = op.ptr_input.type.dtype if isinstance(op.ptr_input.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    lines.append(f"{pad}const uint qmv_k = uint(k) + slid * 4u;")
    for row in range(op.rows):
        offset = "qmv_k" if row == 0 else f"qmv_k + {row * op.input_features}u"
        lines.append(
            f"{pad}const {element_type}4 qmv_input_{row} = "
            f"*((device const {element_type}4*)(&{input_ptr}[{offset}]));"
        )
    for output in range(op.outputs_per_simdgroup):
        values = f"qmv_weight_values_{output}"
        weight_row = f"qmv_output_base + {output}u"
        # Loaded once per output and reused by every activation row below.
        lines.append(
            f"{pad}const {element_type}4 {values} = *((device const {element_type}4*)"
            f"(&{weight_ptr}[({weight_row}) * {op.input_features}u + qmv_k]));"
        )
        for row in range(op.rows):
            for component in "xyzw":
                lines.append(
                    f"{pad}qmv_dot_{row}_{output} += "
                    f"float({values}.{component}) * float(qmv_input_{row}.{component});"
                )


def _emit_dot_residual_store(op, lines, indent, func):
    pad = "    " * indent
    residual_ptr = _val_name_gemm(op.ptr_residual, func)
    output_ptr = _val_name_gemm(op.ptr_output, func)
    dtype = op.ptr_output.type.dtype if isinstance(op.ptr_output.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    low_type = op.round_intermediates or element_type
    for row in range(op.rows):
        row_base = f"{row * op.output_features}u + " if row else ""
        for output in range(op.outputs_per_simdgroup):
            accumulator = f"qmv_dot_{row}_{output}"
            index = f"{row_base}qmv_output_base + {output}u"
            lines.append(f"{pad}#pragma clang loop unroll(full)")
            lines.append(f"{pad}for (ushort offset = 16; offset >= 1; offset >>= 1) {{")
            lines.append(f"{pad}    {accumulator} += simd_shuffle_down({accumulator}, offset);")
            lines.append(f"{pad}}}")
            lines.append(f"{pad}if (slid == 0u) {{")
            lines.append(f"{pad}    const {low_type} qmv_projected = {low_type}({accumulator});")
            lines.append(
                f"{pad}    {output_ptr}[{index}] = "
                f"{element_type}(qmv_projected + {residual_ptr}[{index}]);"
            )
            lines.append(f"{pad}}}")


def _emit_paired_dot_accumulator_init(op, lines, indent):
    pad = "    " * indent
    for row in range(op.rows):
        for output in range(op.outputs_per_simdgroup):
            lines.append(f"{pad}float qmv_left_{row}_{output} = 0.0f;")
            lines.append(f"{pad}float qmv_right_{row}_{output} = 0.0f;")


def _emit_paired_dot_accumulate(op, lines, indent, func):
    if op.elements_per_lane != 4:
        raise ValueError("paired SIMDgroup dot accumulation requires four elements per lane")
    pad = "    " * indent
    input_ptr = _val_name_gemm(op.ptr_input, func)
    left_ptr = _val_name_gemm(op.ptr_left, func) if op.ptr_left is not None else None
    right_ptr = _val_name_gemm(op.ptr_right, func) if op.ptr_right is not None else None
    interleaved_ptr = (
        _val_name_gemm(op.ptr_interleaved, func) if op.ptr_interleaved is not None else None
    )
    dtype = op.ptr_input.type.dtype if isinstance(op.ptr_input.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    suffix = f"_{op.k_offset}" if op.k_offset else ""
    qmv_k = f"qmv_k{suffix}"
    qmv_input = f"qmv_input{suffix}"
    offset = f" + {op.k_offset}u" if op.k_offset else ""
    lines.append(f"{pad}const uint {qmv_k} = uint(k){offset} + slid * 4u;")
    for activation_row in range(op.rows):
        row_offset = "" if activation_row == 0 else f" + {activation_row * op.input_features}u"
        lines.append(
            f"{pad}const {element_type}4 {qmv_input}_{activation_row} = "
            f"*((device const {element_type}4*)(&{input_ptr}[{qmv_k}{row_offset}]));"
        )
    for output in range(op.outputs_per_simdgroup):
        row = f"qmv_output_base + {output}u"
        if interleaved_ptr is not None:
            base = f"qmv_weight_base_{output}{suffix}"
            low = f"qmv_paired_low_{output}{suffix}"
            high = f"qmv_paired_high_{output}{suffix}"
            lines.append(
                f"{pad}const uint {base} = (({row}) * {op.input_features}u + {qmv_k}) * 2u;"
            )
            lines.append(
                f"{pad}const {element_type}4 {low} = *((device const {element_type}4*)"
                f"(&{interleaved_ptr}[{base}]));"
            )
            lines.append(
                f"{pad}const {element_type}4 {high} = *((device const {element_type}4*)"
                f"(&{interleaved_ptr}[{base} + 4u]));"
            )
            for activation_row in range(op.rows):
                for input_component, source, component in zip(
                    "xyzw",
                    (low, low, high, high),
                    "xzxz",
                    strict=True,
                ):
                    lines.append(
                        f"{pad}qmv_left_{activation_row}_{output} += "
                        f"float({source}.{component}) "
                        f"* float({qmv_input}_{activation_row}.{input_component});"
                    )
                for input_component, source, component in zip(
                    "xyzw",
                    (low, low, high, high),
                    "ywyw",
                    strict=True,
                ):
                    lines.append(
                        f"{pad}qmv_right_{activation_row}_{output} += "
                        f"float({source}.{component}) "
                        f"* float({qmv_input}_{activation_row}.{input_component});"
                    )
        else:
            left = f"qmv_left_values_{output}{suffix}"
            right = f"qmv_right_values_{output}{suffix}"
            lines.append(
                f"{pad}const {element_type}4 {left} = *((device const {element_type}4*)"
                f"(&{left_ptr}[({row}) * {op.input_features}u + {qmv_k}]));"
            )
            lines.append(
                f"{pad}const {element_type}4 {right} = *((device const {element_type}4*)"
                f"(&{right_ptr}[({row}) * {op.input_features}u + {qmv_k}]));"
            )
            for activation_row in range(op.rows):
                for component in "xyzw":
                    lines.append(
                        f"{pad}qmv_left_{activation_row}_{output} += float({left}.{component}) "
                        f"* float({qmv_input}_{activation_row}.{component});"
                    )
                for component in "xyzw":
                    lines.append(
                        f"{pad}qmv_right_{activation_row}_{output} += float({right}.{component}) "
                        f"* float({qmv_input}_{activation_row}.{component});"
                    )


def _emit_paired_dot_swiglu_store(op, lines, indent, func):
    pad = "    " * indent
    output_ptr = _val_name_gemm(op.ptr_output, func)
    dtype = op.ptr_output.type.dtype if isinstance(op.ptr_output.type, PtrType) else "f32"
    element_type = ScalarType(dtype).to_msl()
    exponential = "fast::exp" if op.fast_math else "exp"
    low_type = op.round_intermediates or element_type
    for row in range(op.rows):
        row_base = f"{row * op.output_features}u + " if row else ""
        for output in range(op.outputs_per_simdgroup):
            left = f"qmv_left_{row}_{output}"
            right = f"qmv_right_{row}_{output}"
            lines.append(f"{pad}#pragma clang loop unroll(full)")
            lines.append(f"{pad}for (ushort offset = 16; offset >= 1; offset >>= 1) {{")
            lines.append(f"{pad}    {left} += simd_shuffle_down({left}, offset);")
            lines.append(f"{pad}    {right} += simd_shuffle_down({right}, offset);")
            lines.append(f"{pad}}}")
            lines.append(f"{pad}if (slid == 0u) {{")
            lines.append(f"{pad}    const {low_type} swiglu_left = {low_type}({left});")
            lines.append(f"{pad}    const {low_type} swiglu_right = {low_type}({right});")
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
                f"{pad}    {output_ptr}[{row_base}qmv_output_base + {output}u] = "
                f"{element_type}(swiglu_activation * swiglu_right);"
            )
            lines.append(f"{pad}}}")


def _emit_coop_tensor_init(op, lines, indent):
    """Emit cooperative_tensor declaration + zero-init."""
    pad = "    " * indent
    in_type = op.in_type
    left_type = op.left_type or in_type
    right_type = op.right_type or in_type
    acc_type = op.acc_type

    if op.use_separated:
        lines.append(
            f"{pad}auto ct_a = op.get_left_input_cooperative_tensor<{left_type}, {right_type}, {acc_type}>();"
        )
        lines.append(
            f"{pad}auto ct_b = op.get_right_input_cooperative_tensor<{left_type}, {right_type}, {acc_type}>();"
        )
        lines.append(f"{pad}auto {op.ct_name} = op.get_destination_cooperative_tensor<")
        lines.append(f"{pad}    decltype(ct_a), decltype(ct_b), {acc_type}>();")
    else:
        lines.append(f"{pad}auto {op.ct_name} = op.get_destination_cooperative_tensor<")
        lines.append(
            f"{pad}    tensor<{op.left_address_space} {left_type}, dextents<int32_t, 2>, tensor_inline>,"
        )
        lines.append(
            f"{pad}    tensor<{op.right_address_space} {right_type}, dextents<int32_t, 2>, tensor_inline>, {acc_type}>();"
        )

    # Zero-init
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (uint16_t i = 0; i < {op.ct_name}.get_capacity(); ++i) {{")
    lines.append(f"{pad}    if ({op.ct_name}.is_valid_element(i)) {op.ct_name}[i] = {acc_type}(0);")
    lines.append(f"{pad}}}")
    lines.append("")


def _emit_coop_tensor_load(op, lines, indent):
    """Emit cooperative_tensor load from tensor view slice."""
    pad = "    " * indent
    lines.append(
        f"{pad}{op.ct_name}.load({op.tensor_name}.template slice<{op.slice_d0}, {op.slice_d1}>({op.offset_0}, {op.offset_1}));"
    )


def _emit_coop_tensor_store(op, lines, indent):
    """Emit cooperative_tensor store to output slice."""
    pad = "    " * indent
    if getattr(op, "_needs_bounds_guard", False):
        lines.append(f"{pad}if (_valid_tile) {op.ct_name}.store({op.output_slice});")
    else:
        lines.append(f"{pad}{op.ct_name}.store({op.output_slice});")
