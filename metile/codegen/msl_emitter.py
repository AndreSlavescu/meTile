from __future__ import annotations

from metile.compiler.schedule_expr import select_schedule_program
from metile.ir import metal_ir as mir
from metile.ir.types import PtrType, ScalarType

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


def _format_float_literal(v: float) -> str:
    """Format a float constant for MSL."""
    s = f"{v}f"
    if "." not in s and "e" not in s.lower():
        s = f"{v}.0f"
    return s


def _emit_epilogue_chain(operations: list, elem_expr: str, lines: list, pad: str):
    """Emit a chain of element-wise epilogue ops on a single element.

    Handles both simple (relu, unary, scale) and compound (binop with
    constants, binop referencing original accumulator) epilogue patterns.
    Operates on elem_expr (e.g. "acc[0][0].thread_elements()[0]" or "ct[i]").
    """
    # Check if the chain needs save_orig / binop_orig
    has_chain = any(e[0] in ("save_orig", "binop", "binop_orig") for e in operations)

    if has_chain:
        # Use temporaries for the chain
        lines.append(f"{pad}{{")
        lines.append(f"{pad}    float _v = {elem_expr};")
        has_orig = any(e[0] == "save_orig" for e in operations)
        if has_orig:
            lines.append(f"{pad}    float _orig = _v;")
        for epi in operations:
            if epi[0] == "save_orig":
                continue
            elif epi[0] == "relu":
                lines.append(f"{pad}    _v = max(_v, 0.0f);")
            elif epi[0] == "unary":
                fn = _UNARY_MSL.get(epi[1], epi[1])
                lines.append(f"{pad}    _v = {fn}(_v);")
            elif epi[0] == "scale":
                lines.append(f"{pad}    _v *= _scale;")
            elif epi[0] == "binop":
                _, op_name, const_side, const_val = epi
                lit = _format_float_literal(const_val)
                if op_name in _BINOP_SYMBOLS_EPILOGUE:
                    sym = _BINOP_SYMBOLS_EPILOGUE[op_name]
                    if const_side == "lhs":
                        lines.append(f"{pad}    _v = {lit} {sym} _v;")
                    else:
                        lines.append(f"{pad}    _v = _v {sym} {lit};")
                elif op_name in ("max", "min"):
                    lines.append(f"{pad}    _v = {op_name}(_v, {lit});")
            elif epi[0] == "binop_orig":
                _, op_name, orig_side = epi
                if op_name in _BINOP_SYMBOLS_EPILOGUE:
                    sym = _BINOP_SYMBOLS_EPILOGUE[op_name]
                    if orig_side == "lhs":
                        lines.append(f"{pad}    _v = _orig {sym} _v;")
                    else:
                        lines.append(f"{pad}    _v = _v {sym} _orig;")
                elif op_name in ("max", "min"):
                    lines.append(f"{pad}    _v = {op_name}(_v, _orig);")
        lines.append(f"{pad}    {elem_expr} = _v;")
        lines.append(f"{pad}}}")
    else:
        # Simple ops — apply directly (backward compatible)
        for epi in operations:
            if epi[0] == "relu":
                lines.append(f"{pad}{elem_expr} = max({elem_expr}, 0.0f);")
            elif epi[0] == "unary":
                fn = _UNARY_MSL.get(epi[1], epi[1])
                lines.append(f"{pad}{elem_expr} = {fn}({elem_expr});")
            elif epi[0] == "scale":
                lines.append(f"{pad}{elem_expr} *= _scale;")


def emit(func: mir.MFunction) -> str:
    """Generate MSL source code from a Metal IR function."""
    if func.kernel_type == "tensor_ops_gemm":
        return _emit_tensor_ops_kernel(func)
    if func.kernel_type in ("gemm", "persistent_gemm", "specialized_gemm"):
        return _emit_gemm(func)
    return _emit_elementwise(func)


def _emit_tensor_ops_kernel(func: mir.MFunction) -> str:
    """Generate MSL for tensor_ops kernels by walking decomposed ops."""
    # Find setup op to determine if we need sgid
    need_sgid = False
    for op in func.ops:
        if isinstance(op, (mir.MNaxTileLayout, mir.MSimdgroupQMVLayout)) or (
            isinstance(op, mir.MMatmul2dSetup) and not op.cooperative
        ):
            need_sgid = True
            break

    lines = [
        "#include <metal_stdlib>",
        "#include <metal_tensor>",
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
        "using namespace metal;",
        "using namespace mpp::tensor_ops;",
        "",
    ]
    if any(isinstance(op, mir.MBlockScaledTensorViewDecl) for op in func.ops) or _uses_op_type(
        func.ops, mir.MNaxLoadBlockScaledFragment
    ):
        lines.extend(_block_scaled_helpers())

    # Function signature
    params = []
    for buffer_idx, p in enumerate(func.params):
        if isinstance(p.type, PtrType):
            msl_t = ScalarType(p.type.dtype).to_msl()
            params.append(f"    device {msl_t}* {p.name} [[buffer({buffer_idx})]]")
        elif p.is_scalar:
            msl_t = (
                ScalarType(p.type.dtype).to_msl()
                if isinstance(p.type, ScalarType)
                else p.type.to_msl()
            )
            params.append(f"    constant {msl_t}& {p.name} [[buffer({buffer_idx})]]")

    params.append("    uint3 tgp_id [[threadgroup_position_in_grid]]")
    if _uses_op_type(func.ops, mir.ThreadPositionInThreadgroup):
        params.append("    uint lid [[thread_index_in_threadgroup]]")
    if need_sgid:
        params.append("    uint sgid [[simdgroup_index_in_threadgroup]]")
    if _uses_op_type(func.ops, mir.MThreadInSimdgroup):
        params.append("    uint slid [[thread_index_in_simdgroup]]")

    params_str = ",\n".join(params)
    max_threads = func.threadgroup_size[0] * func.threadgroup_size[1] * func.threadgroup_size[2]
    lines.append(f"[[kernel, max_total_threads_per_threadgroup({max_threads})]] void {func.name}(")
    lines.append(params_str)
    lines.append(") {")

    # Check if preemptive mode (needs bounds guards for OOB simdgroups)
    _preemptive = any(isinstance(op, mir.MMatmul2dSetup) and not op.cooperative for op in func.ops)

    # Emit body by walking ops
    for op in func.ops:
        if _preemptive and isinstance(op, mir.MCoopTensorStore):
            op._needs_bounds_guard = True
        if _preemptive and isinstance(op, mir.MCoopTensorEpilogue):
            op._needs_bounds_guard = True
        _emit_gemm_op(op, lines, indent=1, func=func, _tensor_ops_preemptive=_preemptive)

    lines.append("}")
    return "\n".join(lines)


def _emit_elementwise(func: mir.MFunction) -> str:
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
    ]

    # Function signature
    params = []

    for buffer_idx, p in enumerate(func.params):
        if isinstance(p.type, PtrType):
            if p.is_output:
                type_str = f"device {ScalarType(p.type.dtype).to_msl()}*"
            else:
                type_str = f"device const {ScalarType(p.type.dtype).to_msl()}*"
            params.append(f"    {type_str} {p.name} [[buffer({buffer_idx})]]")
        elif p.is_scalar:
            msl_type = (
                ScalarType(p.type.dtype).to_msl()
                if isinstance(p.type, ScalarType)
                else p.type.to_msl()
            )
            params.append(f"    constant {msl_type}& {p.name} [[buffer({buffer_idx})]]")
        else:
            msl_type = p.type.to_msl() if hasattr(p.type, "to_msl") else str(p.type)
            params.append(f"    {msl_type} {p.name} [[buffer({buffer_idx})]]")

    # Thread position attributes
    if _uses_thread_position(func.ops):
        params.append("    uint tid [[thread_position_in_grid]]")
    if _uses_op_type(func.ops, mir.ThreadgroupPositionInGrid):
        params.append("    uint tgp_id_x [[threadgroup_position_in_grid]]")
    if _uses_op_type(func.ops, mir.ThreadPositionInThreadgroup):
        params.append("    uint lid [[thread_position_in_threadgroup]]")
    if _uses_op_type(func.ops, mir.MSimdgroupId):
        params.append("    uint sgid [[simdgroup_index_in_threadgroup]]")
    if (
        _uses_op_type(func.ops, mir.MThreadInSimdgroup)
        or _uses_op_type(func.ops, mir.MSimdShuffleXor)
        or _uses_op_type(func.ops, mir.MSimdBroadcast)
    ):
        params.append("    uint slid [[thread_index_in_simdgroup]]")

    params_str = ",\n".join(params)
    lines.append(f"[[kernel]] void {func.name}(")
    lines.append(params_str)
    lines.append(") {")

    # Emit body
    for op in func.ops:
        _emit_op(op, lines, indent=1, func=func)

    lines.append("}")
    return "\n".join(lines)


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


def _emit_gemm(func: mir.MFunction) -> str:
    lines = [
        "#include <metal_stdlib>",
        "#include <metal_simdgroup_matrix>",
        "using namespace metal;",
        "",
    ]

    # Function signature with GEMM-specific attributes
    params = []

    for buffer_idx, p in enumerate(func.params):
        if isinstance(p.type, PtrType):
            if p.is_atomic:
                type_str = "device atomic_uint*"
            elif p.is_output:
                type_str = f"device {ScalarType(p.type.dtype).to_msl()}*"
            else:
                type_str = f"device const {ScalarType(p.type.dtype).to_msl()}*"
            params.append(f"    {type_str} {p.name} [[buffer({buffer_idx})]]")
        elif p.is_scalar:
            msl_type = (
                ScalarType(p.type.dtype).to_msl()
                if isinstance(p.type, ScalarType)
                else p.type.to_msl()
            )
            params.append(f"    constant {msl_type}& {p.name} [[buffer({buffer_idx})]]")

    # GEMM thread attributes
    if _uses_op_type(func.ops, mir.ThreadgroupPositionInGrid):
        params.append("    uint3 tgp_id [[threadgroup_position_in_grid]]")
    if _uses_op_type(func.ops, mir.MSimdgroupId):
        params.append("    uint sgid [[simdgroup_index_in_threadgroup]]")
    if (
        _uses_op_type(func.ops, mir.MThreadInSimdgroup)
        or _uses_op_type(func.ops, mir.MSimdShuffleXor)
        or _uses_op_type(func.ops, mir.MSimdBroadcast)
    ):
        params.append("    uint slid [[thread_index_in_simdgroup]]")

    params_str = ",\n".join(params)
    lines.append(f"[[kernel]] void {func.name}(")
    lines.append(params_str)
    lines.append(") {")

    # Check for swizzle
    has_swizzle = getattr(func, "_swizzle", False)

    # Emit body
    for op in func.ops:
        _emit_gemm_op(op, lines, indent=1, func=func, has_swizzle=has_swizzle)

    lines.append("}")
    return "\n".join(lines)


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


def _emit_gemm_op(
    op: mir.MOp,
    lines: list[str],
    indent: int,
    func: mir.MFunction,
    has_swizzle: bool = False,
    _tensor_ops_preemptive: bool = False,
):
    # Skip ops folded to constants by the fold pass
    if (
        hasattr(op, "result")
        and op.result is not None
        and op.result.defining_op is not op
        and isinstance(op.result.defining_op, mir.MConstant)
    ):
        return
    # Skip standalone MConstant declarations — values are always inlined by _val_name
    if isinstance(op, mir.MConstant):
        return

    pad = "    " * indent

    if isinstance(op, mir.MSimdgroupId):
        pass  # provided as function parameter 'sgid'

    elif isinstance(op, mir.ThreadPositionInThreadgroup):
        pass  # provided as function parameter 'lid'

    elif isinstance(op, mir.MThreadInSimdgroup):
        pass  # provided as function parameter 'slid'

    elif isinstance(op, mir.ThreadgroupPositionInGrid):
        # tgp_id is a uint3 parameter, axes accessed as tgp_id.x, tgp_id.y
        pass

    elif isinstance(op, mir.MConstant):
        msl_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        # Use constexpr for integer constants to enable better Metal compiler optimization
        qualifier = "constexpr" if op.dtype in ("u32", "i32") else "const"
        lines.append(f"{pad}{qualifier} {msl_type} {name} = {_format_literal(op.value, op.dtype)};")

    elif isinstance(op, mir.MCast):
        target_type = ScalarType(op.target_dtype).to_msl()
        src = _val_name_gemm(op.value, func)
        name = op.result.name
        lines.append(f"{pad}{target_type} {name} = static_cast<{target_type}>({src});")

    elif isinstance(op, mir.MBinOp):
        lhs = _val_name_gemm(op.lhs, func)
        rhs = _val_name_gemm(op.rhs, func)
        result_type = op.result.type.to_msl()
        name = op.result.name

        if op.op in ("max", "min"):
            lines.append(f"{pad}const {result_type} {name} = {op.op}({lhs}, {rhs});")
        elif has_swizzle and name == "block_col":
            sym = _BINOP_SYMBOLS[op.op]
            lines.append(f"{pad}const {result_type} {name}_raw = {lhs} {sym} {rhs};")
            lines.append(
                f"{pad}const uint grid_n_sw = (uint(N) + {_val_name_gemm(op.rhs, func)} - 1) / {_val_name_gemm(op.rhs, func)};"
            )
            lines.append(
                f"{pad}const {result_type} {name} = ((tgp_id.y + tgp_id.x) % grid_n_sw) * {_val_name_gemm(op.rhs, func)};"
            )
        else:
            sym = _BINOP_SYMBOLS[op.op]
            lines.append(f"{pad}const {result_type} {name} = {lhs} {sym} {rhs};")

    elif isinstance(op, mir.MThreadgroupAlloc):
        lines.append(f"{pad}threadgroup {op.elem_type} {op.alloc_name}[{op.size}];")

    elif isinstance(op, mir.MPointerOffset):
        ptr_type = op.result.type.to_msl()
        ptr = _val_name_gemm(op.ptr, func)
        lines.append(f"{pad}{ptr_type} {op.result.name} = {ptr} + {op.offset};")

    # --- New decomposed simdgroup primitive handlers ---
    elif isinstance(op, mir.MSimdgroupAccDecl):
        _emit_simdgroup_acc_decl(op, lines, indent)

    elif isinstance(op, mir.MSimdgroupLoad):
        _emit_simdgroup_load(op, lines, indent, func)

    elif isinstance(op, mir.MSimdgroupMMA):
        _emit_simdgroup_mma(op, lines, indent)

    elif isinstance(op, mir.MSimdgroupStore):
        _emit_simdgroup_store(op, lines, indent, func)

    elif isinstance(op, mir.MAccElemApply):
        _emit_acc_elem_apply(op, lines, indent, func)

    # --- New decomposed tensor_ops primitive handlers ---
    elif isinstance(op, mir.MTensorViewDecl):
        _emit_tensor_view_decl(op, lines, indent, func)

    elif isinstance(op, mir.MTileSchedule):
        _emit_tile_schedule(op, lines, indent)

    elif isinstance(op, mir.MBlockScaledTensorViewDecl):
        _emit_block_scaled_tensor_views(op, lines, indent, func)

    elif isinstance(op, mir.MBlockScaledTileLoad):
        _emit_block_scaled_tile_load(op, lines, indent, func)

    elif isinstance(
        op,
        (
            mir.MNaxGemmSetup,
            mir.MNaxGemmRun,
            mir.MNaxBlockScaledRun,
            mir.MNaxAffineRun,
            mir.MNaxGemmEpilogue,
            mir.MNaxGemmStore,
        ),
    ):
        raise ValueError("fused NAX operations must run through decompose_nax_fragments")

    elif isinstance(op, mir.MNaxTileLayout):
        _emit_nax_tile_layout(op, lines, indent)

    elif isinstance(op, mir.MSimdgroupQMVLayout):
        _emit_simdgroup_qmv_layout(op, lines, indent)

    elif isinstance(op, mir.MDotAccumulatorInit):
        _emit_dot_accumulator_init(op, lines, indent)

    elif isinstance(op, mir.MDotAccumulate):
        _emit_dot_accumulate(op, lines, indent, func)

    elif isinstance(op, mir.MDotResidualStore):
        _emit_dot_residual_store(op, lines, indent, func)

    elif isinstance(op, mir.MPairedDotAccumulatorInit):
        _emit_paired_dot_accumulator_init(op, lines, indent)

    elif isinstance(op, mir.MPairedDotAccumulate):
        _emit_paired_dot_accumulate(op, lines, indent, func)

    elif isinstance(op, mir.MPairedDotSwiGLUStore):
        _emit_paired_dot_swiglu_store(op, lines, indent, func)

    elif isinstance(op, mir.MNaxAccumulatorInit):
        _emit_nax_accumulator_init(op, lines, indent)

    elif isinstance(op, mir.MNaxAccumulatorReset):
        _emit_nax_accumulator_reset(op, lines, indent)

    elif isinstance(op, mir.MNaxMatmul2dDecl):
        _emit_nax_matmul2d_decl(op, lines, indent)

    elif isinstance(op, mir.MNaxLoadFragment):
        _emit_nax_load_fragment(op, lines, indent, func)

    elif isinstance(op, mir.MNaxLoadBlockScale):
        _emit_nax_load_block_scale(op, lines, indent, func)

    elif isinstance(op, mir.MNaxLoadBlockScaledFragment):
        _emit_nax_load_block_scaled_fragment(op, lines, indent, func)

    elif isinstance(op, mir.MNaxLoadAffineParameters):
        _emit_nax_load_affine_parameters(op, lines, indent, func)

    elif isinstance(op, mir.MNaxLoadAffineFragment):
        _emit_nax_load_affine_fragment(op, lines, indent, func)

    elif isinstance(op, mir.MNaxPackRight):
        _emit_nax_pack_right(op, lines, indent)

    elif isinstance(op, mir.MNaxFmaFragment):
        _emit_nax_fma_fragment(op, lines, indent)

    elif isinstance(op, mir.MNaxApplyFragment):
        _emit_nax_apply_fragment(op, lines, indent)

    elif isinstance(op, mir.MNaxBinaryFragment):
        _emit_nax_binary_fragment(op, lines, indent)

    elif isinstance(op, mir.MNaxSpillFragment):
        _emit_nax_spill_fragment(op, lines, indent)

    elif isinstance(op, mir.MNaxReloadFragment):
        _emit_nax_reload_fragment(op, lines, indent)

    elif isinstance(op, mir.MNaxStoreFragment):
        _emit_nax_store_fragment(op, lines, indent, func)

    elif isinstance(op, mir.MMatmul2dSetup):
        _emit_matmul2d_setup(op, lines, indent, func)

    elif isinstance(op, mir.MCoopTensorInit):
        _emit_coop_tensor_init(op, lines, indent)

    elif isinstance(op, mir.MCoopTensorLoad):
        _emit_coop_tensor_load(op, lines, indent)

    elif isinstance(op, mir.MMatmul2dRun):
        _emit_matmul2d_run(op, lines, indent)

    elif isinstance(op, mir.MCoopTensorEpilogue):
        _emit_coop_tensor_epilogue(op, lines, indent)

    elif isinstance(op, mir.MCoopTensorStore):
        _emit_coop_tensor_store(op, lines, indent)

    elif isinstance(op, mir.MBarrier):
        barrier = (
            f"threadgroup_barrier(mem_flags::{op.flags});"
            if op.kind == "threadgroup"
            else f"simdgroup_barrier(mem_flags::{op.flags});"
        )
        if op.condition:
            lines.append(f"{pad}if ({op.condition}) {{ {barrier} }}")
        else:
            lines.append(f"{pad}{barrier}")

    elif isinstance(op, mir.MSimdShuffleXor):
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        val = _val_name_gemm(op.value, func)
        mask = _val_name_gemm(op.mask, func)
        lines.append(f"{pad}{result_type} {name} = simd_shuffle_xor({val}, {mask});")

    elif isinstance(op, mir.MSimdBroadcast):
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        val = _val_name_gemm(op.value, func)
        lane = _val_name_gemm(op.lane, func)
        lines.append(f"{pad}{result_type} {name} = simd_broadcast({val}, {lane});")

    elif isinstance(op, mir.MCooperativeLoad):
        _emit_cooperative_load(op, lines, indent, func)

    elif isinstance(op, mir.MForLoop):
        if _tensor_ops_preemptive and op.iv_name in ("k", "k0"):
            _emit_for_loop_guarded(op, lines, indent, func)
        else:
            _emit_for_loop(op, lines, indent, func, has_swizzle)

    elif isinstance(op, mir.MSimdgroupRoleBlock):
        sgid_name = _val_name_gemm(op.sgid, func)
        end_sg = op.first_sg + op.num_sgs
        if op.num_sgs == 1:
            lines.append(f"{pad}if ({sgid_name} == {op.first_sg}u) {{")
        else:
            lines.append(f"{pad}if ({sgid_name} >= {op.first_sg}u && {sgid_name} < {end_sg}u) {{")
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        lines.append(f"{pad}}}")

    elif isinstance(op, mir.IfBlock):
        cond = _val_name_gemm(op.condition, func)
        lines.append(f"{pad}if ({cond}) {{")
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        lines.append(f"{pad}}}")

    elif isinstance(op, mir.MWhileTrue):
        lines.append(f"{pad}while (true) {{")
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        lines.append(f"{pad}}}")

    elif isinstance(op, mir.MPersistentGrab):
        _emit_persistent_grab(op, lines, indent, func)


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


def _emit_acc_elem_apply(op, lines, indent, func):
    """Emit element-wise epilogue on accumulators via thread_elements()."""
    pad = "    " * indent
    acc = op.acc_name
    lines.append(
        f"{pad}// Fused epilogue: scalar element-wise ops on register-resident accumulators"
    )
    for mi in range(op.num_8m):
        for ni in range(op.num_8n):
            for e in (0, 1):
                elem = f"{acc}[{mi}][{ni}].thread_elements()[{e}]"
                _emit_epilogue_chain(op.operations, elem, lines, pad)


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


def _block_scaled_helpers():
    return [
        "inline float mtile_decode_e2m1(uchar bits) {",
        "    const ushort raw = (ushort(bits & 7u) << 9u) | (ushort(bits & 8u) << 12u);",
        "    return float(as_type<half>(raw)) * 16384.0f;",
        "}",
        "",
        "inline float4 mtile_decode_e2m1(uchar4 bits) {",
        "    const ushort4 raw = (ushort4(bits & 7u) << 9u) | (ushort4(bits & 8u) << 12u);",
        "    return float4(as_type<half4>(raw)) * 16384.0f;",
        "}",
        "",
        "inline float mtile_decode_e4m3(uchar bits) {",
        "    const ushort raw = (ushort(bits & 127u) << 7u) | (ushort(bits & 128u) << 8u);",
        "    return float(as_type<half>(raw)) * 256.0f;",
        "}",
        "",
        "inline float4 mtile_decode_e4m3(uchar4 bits) {",
        "    const ushort4 raw = (ushort4(bits & 127u) << 7u) | (ushort4(bits & 128u) << 8u);",
        "    return float4(as_type<half4>(raw)) * 256.0f;",
        "}",
        "",
        "inline float mtile_decode_e8m0(uchar bits) {",
        "    const uint raw = bits == 0u ? 0x00400000u : uint(bits) << 23u;",
        "    return as_type<float>(raw);",
        "}",
        "",
        "inline float4 mtile_decode_e8m0(uchar4 bits) {",
        "    const uint4 raw = select(uint4(bits) << 23u, uint4(0x00400000u), bits == 0u);",
        "    return as_type<float4>(raw);",
        "}",
        "",
    ]


def _emit_block_scaled_tensor_views(op, lines, indent, func):
    pad = "    " * indent
    a_name = _val_name_gemm(op.ptr_a, func)
    c_name = _val_name_gemm(op.ptr_c, func)
    lines.append(f"{pad}constexpr uint M = {op.m}u;")
    lines.append(f"{pad}constexpr uint N = {op.n}u;")
    lines.append(f"{pad}constexpr uint K = {op.k}u;")
    lines.append(f"{pad}auto tA = tensor<device float, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {a_name}, dextents<int32_t, 2>(K, M));")
    lines.append(f"{pad}auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {c_name}, dextents<int32_t, 2>(N, M));")
    lines.append(f"{pad}threadgroup {op.stage_type} b_tile[{op.block_k * op.block_n}];")
    lines.append(
        f"{pad}auto tB = tensor<threadgroup {op.stage_type}, dextents<int32_t, 2>, tensor_inline>("
    )
    lines.append(f"{pad}    b_tile, dextents<int32_t, 2>({op.block_n}, {op.block_k}));")
    lines.append("")


def _emit_block_scaled_tile_load(op, lines, indent, func):
    pad = "    " * indent
    values = _val_name_gemm(op.ptr_values, func)
    scales = _val_name_gemm(op.ptr_scales, func)
    total = op.block_k * op.block_n
    scale_groups = op.block_k // 32
    lines.append(f"{pad}const uint scale_n = lid % {op.block_n}u;")
    lines.append(f"{pad}float block_scales[{scale_groups}];")
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (uint group = 0u; group < {scale_groups}u; ++group) {{")
    lines.append(
        f"{pad}    block_scales[group] = mtile_decode_e8m0({scales}[(uint(k) / 32u + group) * {op.matrix_n}u + pid_n * {op.block_n}u + scale_n]);"
    )
    lines.append(f"{pad}}}")
    lines.append(f"{pad}for (uint index = lid; index < {total}u; index += {op.num_threads}u) {{")
    lines.append(f"{pad}    const uint local_k = index / {op.block_n}u;")
    lines.append(f"{pad}    const uint local_n = index % {op.block_n}u;")
    lines.append(f"{pad}    const uint global_k = uint(k) + local_k;")
    lines.append(f"{pad}    const uint global_n = pid_n * {op.block_n}u + local_n;")
    lines.append(f"{pad}    const uint element = global_k * {op.matrix_n}u + global_n;")
    if op.bits == 4:
        lines.append(f"{pad}    const uchar byte = {values}[element >> 1u];")
        lines.append(
            f"{pad}    const uchar quantized = (element & 1u) ? (byte >> 4u) : (byte & 15u);"
        )
        decode = "mtile_decode_e2m1(quantized)"
    else:
        decode = f"mtile_decode_e4m3({values}[element])"
    lines.append(
        f"{pad}    b_tile[index] = {op.stage_type}(block_scales[local_k >> 5u] * {decode});"
    )
    lines.append(f"{pad}}}")


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


def _emit_nax_apply_fragment(op, lines, indent):
    pad = "    " * indent
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    _emit_epilogue_chain(op.operations, f"{op.source}[i]", lines, f"{pad}    ")
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


def _emit_tile_schedule(op, lines, indent):
    """Emit a bijective cache-local tile schedule for every grid shape."""
    pad = "    " * indent
    BM, BN = op.block_m, op.block_n
    if op.is_static:
        lines.append(f"{pad}constexpr uint grid_m = {op.grid_m}u;")
        lines.append(f"{pad}constexpr uint grid_n = {op.grid_n}u;")
        _emit_static_tile_schedule(op, lines, indent)
        lines.append("")
        return
    lines.append(f"{pad}const uint grid_m = (uint(M) + {BM}u - 1u) / {BM}u;")
    lines.append(f"{pad}const uint grid_n = (uint(N) + {BN}u - 1u) / {BN}u;")
    if op.pattern.startswith("grouped"):
        group = int(op.pattern.removeprefix("grouped"))
        lines.append(f"{pad}uint pid_m, pid_n;")
        lines.append(f"{pad}if ((grid_m % {group}u) == 0u) {{")
        lines.append(f"{pad}    const uint linear_id = tgp_id.y * grid_m + tgp_id.x;")
        lines.append(f"{pad}    const uint virtual_x = linear_id % (grid_n * {group}u);")
        lines.append(f"{pad}    const uint virtual_y = linear_id / (grid_n * {group}u);")
        lines.append(f"{pad}    pid_m = virtual_y * {group}u + virtual_x % {group}u;")
        lines.append(f"{pad}    pid_n = virtual_x / {group}u;")
        lines.append(f"{pad}}} else {{")
        lines.append(f"{pad}    pid_m = tgp_id.x;")
        lines.append(f"{pad}    pid_n = (tgp_id.y + tgp_id.x) % grid_n;")
        lines.append(f"{pad}}}")
        lines.append("")
        return
    if op.pattern == "linear":
        lines.append(f"{pad}const uint pid_m = tgp_id.x;")
        lines.append(f"{pad}const uint pid_n = tgp_id.y;")
        lines.append("")
        return

    lines.append(f"{pad}const uint linear_id = tgp_id.x * grid_n + tgp_id.y;")
    lines.append(f"{pad}uint pid_m, pid_n;")

    if op.pattern in {"auto", "hilbert"}:
        lines.append(f"{pad}if ((grid_m & 3u) == 0u && (grid_n & 3u) == 0u) {{")
        lines.append(f"{pad}    constexpr ulong hilbert_m = 0xEBFA5014ul;")
        lines.append(f"{pad}    constexpr ulong hilbert_n = 0x05BEBE50ul;")
        lines.append(f"{pad}    const uint panel_id = linear_id >> 4u;")
        lines.append(f"{pad}    const uint within = linear_id & 15u;")
        lines.append(f"{pad}    const uint panels_n = grid_n >> 2u;")
        lines.append(f"{pad}    pid_m = (panel_id / panels_n) * 4u")
        lines.append(f"{pad}        + uint((hilbert_m >> (within * 2u)) & 3ul);")
        lines.append(f"{pad}    pid_n = (panel_id % panels_n) * 4u")
        lines.append(f"{pad}        + uint((hilbert_n >> (within * 2u)) & 3ul);")
        lines.append(f"{pad}}} else if ((grid_m & 1u) == 0u && (grid_n & 1u) == 0u) {{")
        branch_indent = "    "
    elif op.pattern == "morton":
        lines.append(f"{pad}if ((grid_m & 1u) == 0u && (grid_n & 1u) == 0u) {{")
        branch_indent = "    "
    else:
        branch_indent = ""

    if op.pattern in {"auto", "hilbert", "morton"}:
        inner = pad + branch_indent
        lines.append(f"{inner}const uint panel_id = linear_id >> 2u;")
        lines.append(f"{inner}const uint within = linear_id & 3u;")
        lines.append(f"{inner}const uint panels_n = grid_n >> 1u;")
        lines.append(f"{inner}pid_m = (panel_id / panels_n) * 2u + within / 2u;")
        lines.append(f"{inner}pid_n = (panel_id % panels_n) * 2u + within % 2u;")
        lines.append(f"{pad}}} else {{")
        lines.append(f"{pad}    pid_m = tgp_id.x;")
        lines.append(f"{pad}    pid_n = (tgp_id.y + tgp_id.x) % grid_n;")
        lines.append(f"{pad}}}")
    else:
        lines.append(f"{pad}pid_m = tgp_id.x;")
        lines.append(f"{pad}pid_n = (tgp_id.y + tgp_id.x) % grid_n;")
    lines.append("")


def _emit_static_tile_schedule(op, lines, indent):
    """Emit the branch-free expression program extracted by schedule search."""
    pad = "    " * indent
    program = select_schedule_program(op.pattern, op.grid_m, op.grid_n, op.encoding)
    lines.extend(f"{pad}{line}" for line in program.emit_lines())


def _emit_matmul2d_setup(op, lines, indent, func):
    """Emit matmul2d descriptor, operator, SG assignment, output slice."""
    pad = "    " * indent
    SM, SN, BK = op.sm, op.sn, op.bk
    BM, BN = op.block_m, op.block_n
    WM, WN = op.wm, op.wn
    relaxed = "true" if op.relaxed else "false"

    if not op.cooperative:
        # Preemptive: per-SG tile assignment
        lines.append(
            f"{pad}// {op.num_sg} preemptive simdgroups, {WM}x{WN} layout, each handles {SM}x{SN}"
        )
        lines.append(f"{pad}const uint sg_row = sgid / {WN}u;")
        lines.append(f"{pad}const uint sg_col = sgid % {WN}u;")
        lines.append(f"{pad}const uint tile_row = pid_m * {BM}u + sg_row * {SM}u;")
        lines.append(f"{pad}const uint tile_col = pid_n * {BN}u + sg_col * {SN}u;")
        # Guard: skip OOB simdgroups when M or N < BLOCK_M or BLOCK_N
        lines.append(f"{pad}const bool _valid_tile = (tile_row < uint(M)) && (tile_col < uint(N));")
        lines.append("")

        desc_bk = min(32, BK) if op.use_separated else BK
        lines.append(f"{pad}constexpr auto desc = matmul2d_descriptor(")
        lines.append(f"{pad}    {SM}, {SN}, {desc_bk},")
        lines.append(f"{pad}    false, false, {relaxed},")
        lines.append(f"{pad}    matmul2d_descriptor::mode::multiply_accumulate);")
        lines.append(f"{pad}matmul2d<desc, execution_simdgroup> op;")
        lines.append("")
        lines.append(f"{pad}auto mC = tC.template slice<{SN}, {SM}>(tile_col, tile_row);")
    else:
        # Cooperative: full tile
        lines.append(f"{pad}constexpr auto desc = matmul2d_descriptor(")
        lines.append(f"{pad}    {BM}, {BN}, {BK},")
        lines.append(f"{pad}    false, false, {relaxed},")
        lines.append(f"{pad}    matmul2d_descriptor::mode::multiply_accumulate);")
        lines.append(f"{pad}matmul2d<desc, execution_simdgroups<{op.num_sg}>> op;")
        lines.append("")
        row_expr = "pid_m"
        col_expr = "pid_n"
        lines.append(
            f"{pad}auto mC = tC.template slice<{BN}, {BM}>({col_expr} * {BN}u, {row_expr} * {BM}u);"
        )
    lines.append("")


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


def _emit_matmul2d_run(op, lines, indent):
    """Emit matmul2d op.run()."""
    pad = "    " * indent
    if op.use_tensor_view:
        lines.append(
            f"{pad}auto mA = {op.a_tensor}.template slice<{op.a_slice_d0}, {op.a_slice_d1}>({op.a_offset_0}, {op.a_offset_1});"
        )
        lines.append(
            f"{pad}auto mB = {op.b_tensor}.template slice<{op.b_slice_d0}, {op.b_slice_d1}>({op.b_offset_0}, {op.b_offset_1});"
        )
        lines.append(f"{pad}op.run(mA, mB, {op.ct_out});")
    else:
        lines.append(f"{pad}op.run({op.ct_a}, {op.ct_b}, {op.ct_out});")


def _emit_coop_tensor_epilogue(op, lines, indent):
    """Emit element-wise epilogue on cooperative_tensor."""
    pad = "    " * indent
    ct = op.ct_name
    needs_guard = getattr(op, "_needs_bounds_guard", False)
    if needs_guard:
        lines.append(f"{pad}if (_valid_tile) {{")
        indent += 1
        pad = "    " * indent
    lines.append(f"{pad}// Fused epilogue on cooperative_tensor registers")
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (uint16_t i = 0; i < {ct}.get_capacity(); ++i) {{")
    lines.append(f"{pad}    if ({ct}.is_valid_element(i)) {{")
    _emit_epilogue_chain(op.operations, f"{ct}[i]", lines, f"{pad}        ")
    lines.append(f"{pad}    }}")
    lines.append(f"{pad}}}")
    if needs_guard:
        lines.append(f"{'    ' * (indent - 1)}}}")
    lines.append("")


def _emit_coop_tensor_store(op, lines, indent):
    """Emit cooperative_tensor store to output slice."""
    pad = "    " * indent
    if getattr(op, "_needs_bounds_guard", False):
        lines.append(f"{pad}if (_valid_tile) {op.ct_name}.store({op.output_slice});")
    else:
        lines.append(f"{pad}{op.ct_name}.store({op.output_slice});")


def _emit_persistent_grab(
    op: mir.MPersistentGrab, lines: list[str], indent: int, func: mir.MFunction
):
    """Emit the atomic tile grab + broadcast + break-if-done pattern."""
    pad = "    " * indent
    linear_tid = _val_name_gemm(op.linear_tid, func)
    counter = _val_name_gemm(op.counter_ptr, func)
    name = op.result.name

    # Thread 0 grabs next tile via atomic fetch-add
    lines.append(f"{pad}if ({linear_tid} == 0u) {{")
    lines.append(
        f"{pad}    {op.shared_name}[0] = atomic_fetch_add_explicit({counter}, 1u, memory_order_relaxed);"
    )
    lines.append(f"{pad}}}")
    # Broadcast to all threads via threadgroup memory
    lines.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append(f"{pad}uint {name} = {op.shared_name}[0];")
    # Break if all tiles processed
    lines.append(f"{pad}if ({name} >= {op.total_tiles}u) break;")


def _emit_double_buffered_k_loop(
    op: mir.MForLoop, lines: list[str], indent: int, func: mir.MFunction, has_swizzle: bool = False
):
    """Emit a double-buffered K-loop with software pipelining.

    Structure: prologue (load first tile) → main loop (prefetch next +
    compute current) → epilogue (compute last tile).
    """
    pad = "    " * indent
    end = _val_name_gemm(op.end, func)
    step = op.step

    # Extract cooperative loads and compute ops from loop body
    loads = [o for o in op.body if isinstance(o, mir.MCooperativeLoad)]
    kk_loops = [o for o in op.body if isinstance(o, mir.MForLoop) and getattr(o, "_unroll", False)]
    if not loads or not kk_loops:
        # Fallback to regular emission
        _emit_for_loop_regular(op, lines, indent, func, has_swizzle)
        return

    elem_type = loads[0].elem_type

    # Declare pointer-swap variables
    lines.append(f"{pad}// Double-buffered K-loop: prefetch next tile while computing current")
    lines.append(f"{pad}threadgroup {elem_type}* sa_curr = shared_a_0;")
    lines.append(f"{pad}threadgroup {elem_type}* sa_next = shared_a_1;")
    lines.append(f"{pad}threadgroup {elem_type}* sb_curr = shared_b_0;")
    lines.append(f"{pad}threadgroup {elem_type}* sb_next = shared_b_1;")
    lines.append("")

    # Prologue: load first tile into buffer 0
    lines.append(f"{pad}// Prologue: load first tile")
    for ld in loads:
        old_tg = ld.tg_array
        old_kb = ld.kb_expr
        ld.tg_array = (
            f"{old_tg.replace('shared_a', 'shared_a_0').replace('shared_b', 'shared_b_0')}"
        )
        ld.kb_expr = "0"
        _emit_cooperative_load(ld, lines, indent, func)
        ld.tg_array = old_tg
        ld.kb_expr = old_kb
    lines.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
    lines.append("")

    # Main loop: prefetch next + compute current
    lines.append(f"{pad}for (int kb = 0; kb < {end} - {step}; kb += {step}) {{")

    # Prefetch next tile into sa_next/sb_next
    for ld in loads:
        old_tg = ld.tg_array
        old_kb = ld.kb_expr
        ld.tg_array = f"{'sa_next' if 'shared_a' in old_tg else 'sb_next'}"
        ld.kb_expr = f"kb + {step}"
        _emit_cooperative_load(ld, lines, indent + 1, func)
        ld.tg_array = old_tg
        ld.kb_expr = old_kb

    # Compute on current tile from sa_curr/sb_curr
    kk = kk_loops[0]
    _emit_kk_with_buffer(kk, "sa_curr", "sb_curr", lines, indent + 1, func, has_swizzle)

    # Barrier: wait for both prefetch and compute
    p1 = "    " * (indent + 1)
    lines.append(f"{p1}threadgroup_barrier(mem_flags::mem_threadgroup);")

    # Swap buffer pointers
    lines.append(
        f"{p1}{{ threadgroup {elem_type}* _t = sa_curr; sa_curr = sa_next; sa_next = _t; }}"
    )
    lines.append(
        f"{p1}{{ threadgroup {elem_type}* _t = sb_curr; sb_curr = sb_next; sb_next = _t; }}"
    )

    lines.append(f"{pad}}}")
    lines.append("")

    # Epilogue: compute last tile (now in sa_curr after final swap)
    lines.append(f"{pad}// Epilogue: compute last tile")
    kk = kk_loops[0]
    _emit_kk_with_buffer(kk, "sa_curr", "sb_curr", lines, indent, func, has_swizzle)


def _emit_kk_with_buffer(
    kk_loop: mir.MForLoop,
    sa_name: str,
    sb_name: str,
    lines: list[str],
    indent: int,
    func: mir.MFunction,
    has_swizzle: bool = False,
):
    """Emit a kk inner loop with patched shared memory array names.

    Used by double-buffered K-loop to redirect MSimdgroupLoad ops
    to sa_curr/sb_curr or sa_next/sb_next pointer variables.
    """
    # Temporarily patch src_array on all MSimdgroupLoad ops
    originals = []
    for op in kk_loop.body:
        if isinstance(op, mir.MSimdgroupLoad):
            originals.append((op, op.src_array))
            if "shared_a" in op.src_array:
                op.src_array = sa_name
            elif "shared_b" in op.src_array:
                op.src_array = sb_name
    # Emit the loop
    _emit_for_loop_regular(kk_loop, lines, indent, func, has_swizzle)
    # Restore
    for op, orig in originals:
        op.src_array = orig


def _emit_for_loop_regular(
    op: mir.MForLoop, lines: list[str], indent: int, func: mir.MFunction, has_swizzle: bool = False
):
    """Emit a regular for loop (no special markers)."""
    pad = "    " * indent
    end = _val_name_gemm(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)
    start = _val_name_gemm(op.start, func) if isinstance(op.start, mir.MValue) else str(op.start)
    # Check for unroll pragma
    if getattr(op, "_unroll", False):
        lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(
        f"{pad}for (int {op.iv_name} = {start}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
    )
    if op.index_alias and op.index_expression:
        lines.append(f"{pad}    const int {op.index_alias} = {op.index_expression};")
    for body_op in op.body:
        _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
    if getattr(op, "_tg_barrier", False):
        lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_none);")
    lines.append(f"{pad}}}")


def _emit_specialized_db_k_loop(
    op: mir.MForLoop, lines: list[str], indent: int, func: mir.MFunction, has_swizzle: bool = False
):
    """Emit double-buffered K-loop with producer/consumer simdgroup specialization.

    Structure:
        // Prologue already emitted before this loop
        // Pointer init: curr = buf0, next = buf1
        for (kb = 0; kb < K - BK; kb += BK) {
            if (producer) { prefetch into next buffer }
            if (consumer) { MMA from curr buffer }
            barrier;
            swap curr/next;
        }
        // Epilogue (MMA on last tile) emitted after this loop
    """
    pad = "    " * indent
    bk = getattr(op, "_bk", 32)
    end = _val_name_gemm(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)

    # Emit pointer swap variables for double-buffering (before loop for epilogue access)
    lines.append(f"{pad}threadgroup float* sa_curr = shared_a_0;")
    lines.append(f"{pad}threadgroup float* sa_next = shared_a_1;")
    lines.append(f"{pad}threadgroup float* sb_curr = shared_b_0;")
    lines.append(f"{pad}threadgroup float* sb_next = shared_b_1;")

    # Main loop: iterate K - BK steps (last tile handled by epilogue)
    lines.append(f"{pad}for (int kb = 0; kb < {end} - {bk}; kb += {bk}) {{")

    # Emit body ops, replacing shared_a/shared_b references
    for body_op in op.body:
        if isinstance(body_op, mir.MSimdgroupRoleBlock):
            # Check if this is producer (role 0) or consumer (role 1)
            is_producer = body_op.first_sg == 0
            if is_producer:
                # Producers: prefetch into next buffer
                # Emit the role block, but patch cooperative loads to use sa_next/sb_next
                sgid_name = _val_name_gemm(body_op.sgid, func)
                end_sg = body_op.first_sg + body_op.num_sgs
                lines.append(f"{pad}    if ({sgid_name} < {end_sg}u) {{")
                for inner_op in body_op.body:
                    if isinstance(inner_op, mir.MCooperativeLoad):
                        # Replace tg_array name: shared_a -> sa_next, shared_b -> sb_next
                        orig_tg = inner_op.tg_array
                        if "shared_a" in orig_tg:
                            inner_op.tg_array = "sa_next"
                        elif "shared_b" in orig_tg:
                            inner_op.tg_array = "sb_next"
                        _emit_gemm_op(inner_op, lines, indent + 2, func, has_swizzle)
                        inner_op.tg_array = orig_tg  # restore
                    else:
                        _emit_gemm_op(inner_op, lines, indent + 2, func, has_swizzle)
                lines.append(f"{pad}    }}")
            else:
                # Consumers: MMA from current buffer
                sgid_name = _val_name_gemm(body_op.sgid, func)
                first = body_op.first_sg
                end_sg = first + body_op.num_sgs
                lines.append(f"{pad}    if ({sgid_name} >= {first}u && {sgid_name} < {end_sg}u) {{")
                for inner_op in body_op.body:
                    if isinstance(inner_op, mir.MForLoop) and getattr(inner_op, "_unroll", False):
                        # Decomposed: patch MSimdgroupLoad src_array
                        _emit_kk_with_buffer(
                            inner_op, "sa_curr", "sb_curr", lines, indent + 2, func, has_swizzle
                        )
                    else:
                        _emit_gemm_op(inner_op, lines, indent + 2, func, has_swizzle)
                lines.append(f"{pad}    }}")
        else:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)

    # Buffer swap after barrier
    lines.append(
        f"{pad}    {{ threadgroup float* _t = sa_curr; sa_curr = sa_next; sa_next = _t; }}"
    )
    lines.append(
        f"{pad}    {{ threadgroup float* _t = sb_curr; sb_curr = sb_next; sb_next = _t; }}"
    )
    lines.append(f"{pad}}}")


def _emit_for_loop_guarded(op: mir.MForLoop, lines: list[str], indent: int, func: mir.MFunction):
    """Emit a tensor_ops K-loop with _valid_tile bounds guard.

    The _valid_tile guard is hoisted outside the loop since a simdgroup's
    validity is invariant across K iterations. Invalid simdgroups skip
    the entire loop, avoiding both barrier stalls and wasted iterations.

    Barriers inside the loop body are only emitted when threadgroup
    synchronization is genuinely needed (i.e., when barrier ops exist in IR).
    For preemptive tensor_ops with no shared memory, no barriers are emitted.
    """
    pad = "    " * indent
    end = _val_name_gemm(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)

    # Check if any barriers exist (tensor_ops preemptive loops typically have none)
    has_barriers = any(isinstance(b, mir.MBarrier) for b in op.body)
    compute_ops = [b for b in op.body if not isinstance(b, mir.MBarrier)]

    if has_barriers:
        # Barriers require all threads to participate — guard only compute ops
        lines.append(
            f"{pad}for (int {op.iv_name} = {op.start}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
        )
        for b_op in op.body:
            if isinstance(b_op, mir.MBarrier):
                _emit_gemm_op(b_op, lines, indent + 1, func)
            else:
                lines.append(f"{pad}    if (_valid_tile) {{")
                _emit_gemm_op(b_op, lines, indent + 2, func, _tensor_ops_preemptive=True)
                lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")
    else:
        # No barriers — hoist _valid_tile guard outside the entire loop.
        # Invalid simdgroups skip all K iterations with zero overhead.
        lines.append(f"{pad}if (_valid_tile) {{")
        lines.append(
            f"{pad}    for (int {op.iv_name} = {op.start}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
        )
        for body_op in compute_ops:
            _emit_gemm_op(body_op, lines, indent + 2, func, _tensor_ops_preemptive=True)
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")


def _emit_for_loop(
    op: mir.MForLoop, lines: list[str], indent: int, func: mir.MFunction, has_swizzle: bool = False
):
    """Emit a for loop, handling aligned/tail/double-buffered split."""
    pad = "    " * indent

    is_specialized_db = getattr(op, "_specialized_db", False)
    if is_specialized_db:
        _emit_specialized_db_k_loop(op, lines, indent, func, has_swizzle)
        return

    is_double_buffered = getattr(op, "_double_buffered", False)
    if is_double_buffered:
        _emit_double_buffered_k_loop(op, lines, indent, func, has_swizzle)
        return

    is_aligned = getattr(op, "_aligned", False)
    is_tail = getattr(op, "_is_tail", False)

    if is_aligned:
        # Aligned loop: iterate up to k_aligned = (K / step) * step
        end = _val_name_gemm(op.end, func)
        lines.append(f"{pad}const int k_aligned = ({end} / {op.step}) * {op.step};")
        # Check for unroll pragma
        if getattr(op, "_unroll", False):
            lines.append(f"{pad}#pragma clang loop unroll(full)")
        lines.append(
            f"{pad}for (int {op.iv_name} = 0; {op.iv_name} < k_aligned; {op.iv_name} += {op.step}) {{"
        )
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        if getattr(op, "_tg_barrier", False):
            lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_none);")
        lines.append(f"{pad}}}")

    elif is_tail:
        # Tail block: single iteration for remainder
        end = _val_name_gemm(op.end, func)
        lines.append(f"{pad}if (k_aligned < {end}) {{")
        lines.append(f"{pad}    const int kb = k_aligned;")
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        lines.append(f"{pad}}}")

    else:
        # Regular for loop (no split)
        end = _val_name_gemm(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)
        start = (
            _val_name_gemm(op.start, func) if isinstance(op.start, mir.MValue) else str(op.start)
        )
        # Check for unroll pragma
        if getattr(op, "_unroll", False):
            lines.append(f"{pad}#pragma clang loop unroll(full)")
        lines.append(
            f"{pad}for (int {op.iv_name} = {start}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
        )
        if op.index_alias and op.index_expression:
            lines.append(f"{pad}    const int {op.index_alias} = {op.index_expression};")
        for body_op in op.body:
            _emit_gemm_op(body_op, lines, indent + 1, func, has_swizzle)
        if getattr(op, "_tg_barrier", False):
            lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_none);")
        lines.append(f"{pad}}}")


def _emit_op(op: mir.MOp, lines: list[str], indent: int, func: mir.MFunction):
    """Emit a single Metal IR op (element-wise path)."""
    # Skip ops folded to constants by the fold pass
    if (
        hasattr(op, "result")
        and op.result is not None
        and op.result.defining_op is not op
        and isinstance(op.result.defining_op, mir.MConstant)
    ):
        return
    # Skip standalone MConstant declarations — values are always inlined by _val_name
    if isinstance(op, mir.MConstant):
        return

    pad = "    " * indent

    if isinstance(op, mir.ThreadPositionInGrid):
        pass  # provided as function parameter 'tid'

    elif isinstance(op, mir.ThreadgroupPositionInGrid):
        pass  # provided as function parameter 'tgp_id_x'

    elif isinstance(op, mir.ThreadPositionInThreadgroup):
        pass  # provided as function parameter 'lid'

    elif isinstance(op, mir.MSimdgroupId):
        pass  # provided as function parameter 'sgid'

    elif isinstance(op, mir.MThreadInSimdgroup):
        pass  # provided as function parameter 'slid'

    elif isinstance(op, mir.MThreadgroupAlloc):
        lines.append(f"{pad}threadgroup {op.elem_type} {op.alloc_name}[{op.size}];")

    elif isinstance(op, mir.MBarrier):
        barrier = (
            f"threadgroup_barrier(mem_flags::{op.flags});"
            if op.kind == "threadgroup"
            else f"simdgroup_barrier(mem_flags::{op.flags});"
        )
        if op.condition:
            lines.append(f"{pad}if ({op.condition}) {{ {barrier} }}")
        else:
            lines.append(f"{pad}{barrier}")

    elif isinstance(op, mir.MThreadgroupReduce):
        _emit_threadgroup_reduce(op, lines, indent, func)

    elif isinstance(op, mir.MConstant):
        msl_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        lines.append(f"{pad}{msl_type} {name} = {_format_literal(op.value, op.dtype)};")

    elif isinstance(op, mir.MCast):
        target_type = ScalarType(op.target_dtype).to_msl()
        src = _val_name(op.value, func)
        name = op.result.name
        lines.append(f"{pad}{target_type} {name} = static_cast<{target_type}>({src});")

    elif isinstance(op, mir.MBinOp):
        lhs = _val_name(op.lhs, func)
        rhs = _val_name(op.rhs, func)
        result_type = op.result.type.to_msl()
        name = op.result.name
        if op.op in ("max", "min"):
            lines.append(f"{pad}{result_type} {name} = {op.op}({lhs}, {rhs});")
        else:
            sym = _BINOP_SYMBOLS[op.op]
            lines.append(f"{pad}{result_type} {name} = {lhs} {sym} {rhs};")

    elif isinstance(op, mir.MUnary):
        msl_fn = _UNARY_MSL[op.op]
        src = _val_name(op.operand, func)
        result_type = op.result.type.to_msl()
        name = op.result.name
        if op.op == "neg":
            lines.append(f"{pad}{result_type} {name} = -{src};")
        else:
            lines.append(f"{pad}{result_type} {name} = {msl_fn}({src});")

    elif isinstance(op, mir.MSelect):
        cond = _val_name(op.condition, func)
        tv = _val_name(op.true_val, func)
        fv = _val_name(op.false_val, func)
        result_type = op.result.type.to_msl()
        name = op.result.name
        lines.append(f"{pad}{result_type} {name} = {cond} ? {tv} : {fv};")

    elif isinstance(op, mir.MCompare):
        sym = _CMP_SYMBOLS[op.predicate]
        lhs = _val_name(op.lhs, func)
        rhs = _val_name(op.rhs, func)
        name = op.result.name
        lines.append(f"{pad}bool {name} = {lhs} {sym} {rhs};")

    elif isinstance(op, mir.MSimdShuffleXor):
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        val = _val_name(op.value, func)
        mask = _val_name(op.mask, func)
        lines.append(f"{pad}{result_type} {name} = simd_shuffle_xor({val}, {mask});")

    elif isinstance(op, mir.MSimdBroadcast):
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        val = _val_name(op.value, func)
        lane = _val_name(op.lane, func)
        lines.append(f"{pad}{result_type} {name} = simd_broadcast({val}, {lane});")

    elif isinstance(op, mir.DeviceLoad):
        ptr = _val_name(op.ptr, func)
        idx = _val_name(op.index, func)
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        lines.append(f"{pad}{result_type} {name} = {ptr}[{idx}];")

    elif isinstance(op, mir.DeviceStore):
        ptr = _val_name(op.ptr, func)
        idx = _val_name(op.index, func)
        val = _val_name(op.value, func)
        lines.append(f"{pad}{ptr}[{idx}] = {val};")

    elif isinstance(op, mir.MThreadgroupLoad):
        result_type = ScalarType(op.dtype).to_msl()
        name = op.result.name
        idx = _val_name(op.index, func)
        lines.append(f"{pad}{result_type} {name} = {op.array_name}[{idx}];")

    elif isinstance(op, mir.MThreadgroupStore):
        idx = _val_name(op.index, func)
        val = _val_name(op.value, func)
        lines.append(f"{pad}{op.array_name}[{idx}] = {val};")

    elif isinstance(op, mir.MVarDecl):
        msl_type = ScalarType(op.dtype).to_msl()
        init = _val_name(op.init_value, func)
        lines.append(f"{pad}{msl_type} {op.var_name} = {init};")

    elif isinstance(op, mir.MVarAssign):
        val = _val_name(op.value, func)
        lines.append(f"{pad}{op.var_name} = {val};")

    elif isinstance(op, mir.MForLoop):
        end = _val_name(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)
        start = _val_name(op.start, func) if isinstance(op.start, mir.MValue) else str(op.start)

        vec_size = getattr(op, "_vec_size", 0)

        if getattr(op, "_ew_aligned", False) and vec_size > 1:
            # Vec4-aligned loop: wider step, float4 loads/stores
            _emit_vec4_for_loop(op, lines, indent, func, vec_size)

        elif getattr(op, "_ew_aligned", False):
            # Scalar aligned loop: iterate up to aligned_end, no bounds check
            ew_id = getattr(op, "_ew_id", 0)
            var = f"_ew_end_{ew_id}"
            lines.append(f"{pad}const int {var} = ({end} / {op.step}) * {op.step};")
            lines.append(
                f"{pad}for (int {op.iv_name} = {start}; {op.iv_name} < {var}; {op.iv_name} += {op.step}) {{"
            )
            for body_op in op.body:
                _emit_op(body_op, lines, indent + 1, func)
            lines.append(f"{pad}}}")

        elif getattr(op, "_ew_tail", False) and getattr(op, "_vec_tail", False):
            # Tail loop after vec4 aligned — need full loop, not single iteration
            ew_id = getattr(op, "_ew_id", 0)
            var = f"_ew_end_{ew_id}"
            lines.append(
                f"{pad}for (int {op.iv_name} = {var}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
            )
            for body_op in op.body:
                _emit_op(body_op, lines, indent + 1, func)
            lines.append(f"{pad}}}")

        elif getattr(op, "_ew_tail", False):
            # Tail: single iteration for remainder
            ew_id = getattr(op, "_ew_id", 0)
            var = f"_ew_end_{ew_id}"
            lines.append(f"{pad}if ({var} < {end}) {{")
            lines.append(f"{pad}    const int {op.iv_name} = {var};")
            for body_op in op.body:
                _emit_op(body_op, lines, indent + 2, func)
            lines.append(f"{pad}}}")

        else:
            lines.append(
                f"{pad}for (int {op.iv_name} = {start}; {op.iv_name} < {end}; {op.iv_name} += {op.step}) {{"
            )
            for body_op in op.body:
                _emit_op(body_op, lines, indent + 1, func)
            lines.append(f"{pad}}}")

    elif isinstance(op, mir.MSimdgroupRoleBlock):
        sgid = _val_name(op.sgid, func)
        end_sg = op.first_sg + op.num_sgs
        if op.num_sgs == 1:
            lines.append(f"{pad}if ({sgid} == {op.first_sg}u) {{")
        else:
            lines.append(f"{pad}if ({sgid} >= {op.first_sg}u && {sgid} < {end_sg}u) {{")
        for body_op in op.body:
            _emit_op(body_op, lines, indent + 1, func)
        lines.append(f"{pad}}}")

    elif isinstance(op, mir.IfBlock):
        cond = _val_name(op.condition, func)
        lines.append(f"{pad}if ({cond}) {{")
        for body_op in op.body:
            _emit_op(body_op, lines, indent + 1, func)
        lines.append(f"{pad}}}")


def _emit_vec4_for_loop(
    op: mir.MForLoop,
    lines: list[str],
    indent: int,
    func: mir.MFunction,
    vec_size: int,
):
    """Emit a vec4-optimized aligned element-wise loop.

    Each thread processes vec_size consecutive elements using float4,
    reducing instruction count and enabling wider memory transactions.

    When num_stages > 1, the loop is software-pipelined: each iteration
    contains num_stages copies of the body with staggered IV offsets,
    each in its own scope. This lets the GPU compiler interleave loads
    from the next stage with compute from the current stage.
    """
    pad = "    " * indent
    end = _val_name(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)
    start = _val_name(op.start, func) if isinstance(op.start, mir.MValue) else str(op.start)
    ew_id = getattr(op, "_ew_id", 0)
    var = f"_ew_end_{ew_id}"
    num_stages = getattr(op, "_num_stages", 1)
    vstep = op.step * vec_size  # per-stage step (e.g., 256 threads * 4 = 1024 elements)
    total_step = vstep * num_stages  # full step per loop iteration

    lines.append(f"{pad}const int {var} = ({end} / {total_step}) * {total_step};")

    if num_stages <= 1:
        # Simple vec4 loop — no pipelining
        lines.append(
            f"{pad}for (int {op.iv_name} = {start}; {op.iv_name} < {var}; "
            f"{op.iv_name} += {vstep}) {{"
        )
        vec4_vals: dict[str, str] = {}
        for body_op in op.body:
            _emit_vec4_op(body_op, lines, indent + 1, func, vec4_vals, vec_size)
        lines.append(f"{pad}}}")
    else:
        # Software-pipelined: unroll num_stages copies per iteration,
        # each in its own scope so variable names don't clash.
        # The accumulator (_acc_N) lives in the outer scope and chains
        # across stages.
        pipe_iv = f"_pipe_{op.iv_name}"
        lines.append(
            f"{pad}for (int {pipe_iv} = {start}; {pipe_iv} < {var}; {pipe_iv} += {total_step}) {{"
        )
        for stage in range(num_stages):
            offset = stage * vstep
            lines.append(f"{pad}    {{ // stage {stage}")
            lines.append(f"{pad}        const int {op.iv_name} = {pipe_iv} + {offset};")
            vec4_vals_stage: dict[str, str] = {}
            for body_op in op.body:
                _emit_vec4_op(body_op, lines, indent + 2, func, vec4_vals_stage, vec_size)
            lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")


def _emit_vec4_op(
    op: mir.MOp,
    lines: list[str],
    indent: int,
    func: mir.MFunction,
    vec4_vals: dict[str, str],
    vec_size: int,
):
    """Emit one vectorized op while preserving each value's element type."""
    pad = "    " * indent

    if isinstance(op, mir.MCast):
        src = _val_name(op.value, func)
        target_type = ScalarType(op.target_dtype).to_msl()
        name = op.result.name
        if src in vec4_vals:
            vector_type = f"{target_type}4"
            vec4_vals[name] = vector_type
            lines.append(f"{pad}{vector_type} {name} = {vector_type}({src});")
        elif src == "lid":
            lines.append(
                f"{pad}{target_type} {name} = static_cast<{target_type}>(lid) * {vec_size};"
            )
        else:
            lines.append(f"{pad}{target_type} {name} = static_cast<{target_type}>({src});")

    elif isinstance(op, mir.DeviceLoad):
        ptr = _val_name(op.ptr, func)
        idx = _val_name(op.index, func)
        name = op.result.name
        vector_type = f"{ScalarType(op.dtype).to_msl()}4"
        vec4_vals[name] = vector_type
        lines.append(f"{pad}{vector_type} {name} = *(device const {vector_type}*)({ptr} + {idx});")

    elif isinstance(op, mir.DeviceStore):
        ptr = _val_name(op.ptr, func)
        idx = _val_name(op.index, func)
        val = _val_name(op.value, func)
        if val in vec4_vals:
            pointer_dtype = op.ptr.type.dtype if isinstance(op.ptr.type, PtrType) else "f32"
            vector_type = f"{ScalarType(pointer_dtype).to_msl()}4"
            stored_value = val if vec4_vals[val] == vector_type else f"{vector_type}({val})"
            lines.append(f"{pad}*(device {vector_type}*)({ptr} + {idx}) = {stored_value};")
        else:
            lines.append(f"{pad}{ptr}[{idx}] = {val};")

    elif isinstance(op, mir.MBinOp):
        lhs = _val_name(op.lhs, func)
        rhs = _val_name(op.rhs, func)
        name = op.result.name
        lhs_v = lhs in vec4_vals
        rhs_v = rhs in vec4_vals

        if lhs_v or rhs_v:
            # --- Accumulator reductions: scalar_acc OP vec4 → reduce then combine ---
            is_acc_lhs = not lhs_v and rhs_v and lhs.startswith("_acc_")
            is_acc_rhs = lhs_v and not rhs_v and rhs.startswith("_acc_")

            if is_acc_lhs and op.op == "add":
                lines.append(
                    f"{pad}float {name} = {lhs} + ({rhs}.x + {rhs}.y + {rhs}.z + {rhs}.w);"
                )
                return
            if is_acc_rhs and op.op == "add":
                lines.append(
                    f"{pad}float {name} = {rhs} + ({lhs}.x + {lhs}.y + {lhs}.z + {lhs}.w);"
                )
                return
            if is_acc_lhs and op.op == "max":
                lines.append(
                    f"{pad}float {name} = max({lhs}, "
                    f"max(max({rhs}.x, {rhs}.y), max({rhs}.z, {rhs}.w)));"
                )
                return
            if is_acc_rhs and op.op == "max":
                lines.append(
                    f"{pad}float {name} = max({rhs}, "
                    f"max(max({lhs}.x, {lhs}.y), max({lhs}.z, {lhs}.w)));"
                )
                return
            if is_acc_lhs and op.op == "min":
                lines.append(
                    f"{pad}float {name} = min({lhs}, "
                    f"min(min({rhs}.x, {rhs}.y), min({rhs}.z, {rhs}.w)));"
                )
                return
            if is_acc_rhs and op.op == "min":
                lines.append(
                    f"{pad}float {name} = min({rhs}, "
                    f"min(min({lhs}.x, {lhs}.y), min({lhs}.z, {lhs}.w)));"
                )
                return

            # --- Vec4 x Vec4, or Vec4 x scalar (broadcast) ---
            vector_type = f"{op.result.type.to_msl()}4"
            vec4_vals[name] = vector_type
            if op.op in ("max", "min"):
                lines.append(f"{pad}{vector_type} {name} = {op.op}({lhs}, {rhs});")
            elif op.op in _BINOP_SYMBOLS:
                sym = _BINOP_SYMBOLS[op.op]
                lines.append(f"{pad}{vector_type} {name} = {lhs} {sym} {rhs};")
            else:
                lines.append(f"{pad}{vector_type} {name} = {op.op}({lhs}, {rhs});")
        else:
            # Scalar-only op (index math, etc.)
            result_type = op.result.type.to_msl()
            if op.op in ("max", "min"):
                lines.append(f"{pad}{result_type} {name} = {op.op}({lhs}, {rhs});")
            else:
                sym = _BINOP_SYMBOLS.get(op.op, "+")
                lines.append(f"{pad}{result_type} {name} = {lhs} {sym} {rhs};")

    elif isinstance(op, mir.MUnary):
        src = _val_name(op.operand, func)
        name = op.result.name
        msl_fn = _UNARY_MSL.get(op.op, op.op)
        if src in vec4_vals:
            vector_type = f"{op.result.type.to_msl()}4"
            vec4_vals[name] = vector_type
            if op.op == "neg":
                lines.append(f"{pad}{vector_type} {name} = -{src};")
            else:
                lines.append(f"{pad}{vector_type} {name} = {msl_fn}({src});")
        else:
            result_type = op.result.type.to_msl()
            if op.op == "neg":
                lines.append(f"{pad}{result_type} {name} = -{src};")
            else:
                lines.append(f"{pad}{result_type} {name} = {msl_fn}({src});")

    elif isinstance(op, mir.MSelect):
        cond = _val_name(op.condition, func)
        tv = _val_name(op.true_val, func)
        fv = _val_name(op.false_val, func)
        name = op.result.name
        if tv in vec4_vals or fv in vec4_vals:
            vector_type = vec4_vals.get(tv, vec4_vals.get(fv))
            vec4_vals[name] = vector_type
            lines.append(f"{pad}{vector_type} {name} = select({fv}, {tv}, {cond});")
        else:
            result_type = op.result.type.to_msl()
            lines.append(f"{pad}{result_type} {name} = {cond} ? {tv} : {fv};")

    elif isinstance(op, mir.MCompare):
        # Dead in aligned loops, but emit for correctness
        sym = _CMP_SYMBOLS[op.predicate]
        lhs = _val_name(op.lhs, func)
        rhs = _val_name(op.rhs, func)
        lines.append(f"{pad}bool {op.result.name} = {lhs} {sym} {rhs};")

    elif isinstance(op, mir.MVarAssign):
        val = _val_name(op.value, func)
        lines.append(f"{pad}{op.var_name} = {val};")

    elif isinstance(op, mir.MConstant):
        msl_type = ScalarType(op.dtype).to_msl()
        lines.append(f"{pad}{msl_type} {op.result.name} = {_format_literal(op.value, op.dtype)};")

    else:
        # Fallback: emit with normal path
        _emit_op(op, lines, indent, func)


def _emit_threadgroup_reduce(
    op: mir.MThreadgroupReduce, lines: list[str], indent: int, func: mir.MFunction
):
    """Emit threadgroup reduction: simd_sum + shared memory tree + broadcast."""
    pad = "    " * indent
    operand = _val_name(op.operand, func)
    name = op.result.name
    num_sg = op.block_size // 32
    msl_type = ScalarType(op.dtype).to_msl()

    _SIMD_REDUCE = {"sum": "simd_sum", "max": "simd_max", "min": "simd_min"}
    simd_fn = _SIMD_REDUCE.get(op.reduce_op, "simd_sum")

    if num_sg <= 1:
        lines.append(f"{pad}{msl_type} {name} = {simd_fn}({operand});")
    elif num_sg <= 32:
        # Two-level reduction: simd_sum within each simdgroup, then simd_sum
        # across partial sums — avoids serial loop for large simdgroup counts.
        lines.append(f"{pad}{msl_type} {name};")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}    {msl_type} _simd_val = {simd_fn}({operand});")
        lines.append(f"{pad}    if (slid == 0u) {op.shared_name}[sgid] = _simd_val;")
        lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(
            f"{pad}    {msl_type} _partial = (lid < {num_sg}u) ? {op.shared_name}[lid] : 0.0f;"
        )
        lines.append(f"{pad}    {msl_type} _result = {simd_fn}(_partial);")
        lines.append(f"{pad}    if (lid == 0u) {op.shared_name}[0] = _result;")
        lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"{pad}    {name} = {op.shared_name}[0];")
        lines.append(f"{pad}}}")
    else:
        # Fallback for > 32 simdgroups (unlikely)
        _COMBINE = {
            "sum": "_total += {v};",
            "max": "_total = max(_total, {v});",
            "min": "_total = min(_total, {v});",
        }
        combine_fmt = _COMBINE.get(op.reduce_op, "_total += {v};")
        lines.append(f"{pad}{msl_type} {name};")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}    {msl_type} _simd_val = {simd_fn}({operand});")
        lines.append(f"{pad}    if (slid == 0u) {op.shared_name}[sgid] = _simd_val;")
        lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"{pad}    if (lid == 0u) {{")
        lines.append(f"{pad}        {msl_type} _total = {op.shared_name}[0];")
        for j in range(1, num_sg):
            lines.append(f"{pad}        {combine_fmt.format(v=f'{op.shared_name}[{j}]')}")
        lines.append(f"{pad}        {op.shared_name}[0] = _total;")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);")
        lines.append(f"{pad}    {name} = {op.shared_name}[0];")
        lines.append(f"{pad}}}")


def _resolve(val: mir.MValue) -> mir.MValue:
    """Follow CSE forwarding chain to canonical value."""
    while (
        val.defining_op and val.defining_op.result is not None and val.defining_op.result is not val
    ):
        val = val.defining_op.result
    return val


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
