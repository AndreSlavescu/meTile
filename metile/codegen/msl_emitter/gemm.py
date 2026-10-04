"""MSL emission for the GEMM kernel families and their K-loops."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

from metile.codegen.msl_emitter.block_scaled import (
    _block_scaled_helpers,
    _emit_block_scaled_tensor_views,
    _emit_block_scaled_tile_load,
)
from metile.codegen.msl_emitter.common import (
    _BINOP_SYMBOLS,
    _emit_dimension_bindings,
    _emit_tensor_view_decl,
    _format_literal,
    _uses_op_type,
    _val_name_gemm,
)
from metile.codegen.msl_emitter.elementwise import (
    _emit_acc_elem_apply,
    _emit_coop_tensor_epilogue,
    _emit_epilogue_captures,
    _emit_nax_apply_fragment,
)
from metile.codegen.msl_emitter.nax import (
    _emit_nax_accumulator_init,
    _emit_nax_accumulator_reset,
    _emit_nax_binary_fragment,
    _emit_nax_fma_fragment,
    _emit_nax_load_affine_fragment,
    _emit_nax_load_affine_parameters,
    _emit_nax_load_block_scale,
    _emit_nax_load_block_scaled_fragment,
    _emit_nax_load_fragment,
    _emit_nax_matmul2d_decl,
    _emit_nax_pack_right,
    _emit_nax_reload_fragment,
    _emit_nax_spill_fragment,
    _emit_nax_store_fragment,
    _emit_nax_tile_layout,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_coop_tensor_init,
    _emit_coop_tensor_load,
    _emit_coop_tensor_store,
    _emit_cooperative_load,
    _emit_dot_accumulate,
    _emit_dot_accumulator_init,
    _emit_dot_residual_store,
    _emit_paired_dot_accumulate,
    _emit_paired_dot_accumulator_init,
    _emit_paired_dot_swiglu_store,
    _emit_simdgroup_acc_decl,
    _emit_simdgroup_load,
    _emit_simdgroup_mma,
    _emit_simdgroup_qmv_layout,
    _emit_simdgroup_store,
)
from metile.compiler.schedule_expr import select_schedule_program
from metile.ir import metal_ir as mir
from metile.ir.types import PtrType, ScalarType


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
    scalar_captures = _emit_epilogue_captures(func, lines)
    dimension_scope = _emit_dimension_bindings(func, lines)
    if scalar_captures and not dimension_scope:
        lines.append("    {")
        dimension_scope = True

    # Check if preemptive mode (needs bounds guards for OOB simdgroups)
    _preemptive = any(isinstance(op, mir.MMatmul2dSetup) and not op.cooperative for op in func.ops)

    # Emit body by walking ops
    for op in func.ops:
        if _preemptive and isinstance(op, mir.MCoopTensorStore):
            op._needs_bounds_guard = True
        if _preemptive and isinstance(op, mir.MCoopTensorEpilogue):
            op._needs_bounds_guard = True
        _emit_gemm_op(
            op,
            lines,
            indent=2 if dimension_scope else 1,
            func=func,
            _tensor_ops_preemptive=_preemptive,
        )

    if dimension_scope:
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


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
    scalar_captures = _emit_epilogue_captures(func, lines)
    dimension_scope = _emit_dimension_bindings(func, lines)
    if scalar_captures and not dimension_scope:
        lines.append("    {")
        dimension_scope = True

    # Check for swizzle
    has_swizzle = getattr(func, "_swizzle", False)

    # Emit body
    for op in func.ops:
        _emit_gemm_op(
            op, lines, indent=2 if dimension_scope else 1, func=func, has_swizzle=has_swizzle
        )

    if dimension_scope:
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


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
    """Emit the verified software copy/compute/publication/recycling phase program."""
    from metile.compiler.staging import StagingError, validate_staging

    if op.staging is None:
        raise StagingError("double-buffered emission requires a verified staging contract")
    validate_staging(func)
    pipeline = op.staging
    buffers = {buffer.logical_name: buffer for buffer in pipeline.buffers}
    loads = {load.tg_array: load for load in op.body[:2]}
    compute = op.body[3]
    pad = "    " * indent
    end = _val_name_gemm(op.end, func) if isinstance(op.end, mir.MValue) else str(op.end)
    lines.append(f"{pad}if ({end} > 0) {{")
    for buffer in pipeline.buffers:
        lines.append(
            f"{pad}    threadgroup {buffer.element_type}* {buffer.current_pointer} = {buffer.slots[0]};"
        )
        lines.append(
            f"{pad}    threadgroup {buffer.element_type}* {buffer.next_pointer} = {buffer.slots[1]};"
        )

    def emit_region(region, level):
        region_pad = "    " * level
        for phase in pipeline.phases:
            if phase.region != region:
                continue
            if phase.kind == "copy":
                for access in phase.writes:
                    buffer = buffers[access.buffer]
                    target = (
                        buffer.current_pointer if access.slot == "current" else buffer.next_pointer
                    )
                    reduction = "0" if region == "prologue" else f"{op.iv_name} + {op.step}"
                    load = replace(loads[access.buffer], tg_array=target, kb_expr=reduction)
                    _emit_cooperative_load(load, lines, level, func)
            elif phase.kind == "compute":
                aliases = {
                    access.buffer: buffers[access.buffer].current_pointer for access in phase.reads
                }
                current_compute = copy(compute)
                current_compute.body = [
                    replace(operation, src_array=aliases[operation.src_array])
                    if isinstance(operation, mir.MSimdgroupLoad)
                    else operation
                    for operation in compute.body
                ]
                _emit_for_loop_regular(current_compute, lines, level, func, has_swizzle)
            elif phase.kind == "barrier":
                lines.append(
                    f"{region_pad}{phase.barrier_scope}_barrier(mem_flags::{phase.barrier_flags});"
                )
            elif phase.kind == "rotate":
                for name in phase.rotates:
                    buffer = buffers[name]
                    current, upcoming = buffer.current_pointer, buffer.next_pointer
                    lines.append(
                        f"{region_pad}{{ threadgroup {buffer.element_type}* _stage_swap = {current}; "
                        f"{current} = {upcoming}; {upcoming} = _stage_swap; }}"
                    )

    emit_region("prologue", indent + 1)
    lines.append(
        f"{pad}    for (int {op.iv_name} = 0; {op.iv_name} < {end} - {op.step}; {op.iv_name} += {op.step}) {{"
    )
    emit_region("steady", indent + 2)
    lines.append(f"{pad}    }}")
    emit_region("drain", indent + 1)
    lines.append(f"{pad}}}")


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
    aliases = {"shared_a": sa_name, "shared_b": sb_name}
    current_loop = copy(kk_loop)
    current_loop.body = [
        replace(operation, src_array=aliases[operation.src_array])
        if isinstance(operation, mir.MSimdgroupLoad) and operation.src_array in aliases
        else operation
        for operation in kk_loop.body
    ]
    _emit_for_loop_regular(current_loop, lines, indent, func, has_swizzle)


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
    staged_names = {"shared_a_0", "shared_a_1", "shared_b_0", "shared_b_1"}
    allocations = {
        allocation.alloc_name: allocation.elem_type
        for allocation in func.ops
        if isinstance(allocation, mir.MThreadgroupAlloc) and allocation.alloc_name in staged_names
    }
    element_types = set(allocations.values())
    if set(allocations) != staged_names or len(element_types) != 1:
        raise ValueError("specialized staging requires matching typed operand allocations")
    element_type = element_types.pop()

    # Emit pointer swap variables for double-buffering (before loop for epilogue access)
    lines.append(f"{pad}threadgroup {element_type}* sa_curr = shared_a_0;")
    lines.append(f"{pad}threadgroup {element_type}* sa_next = shared_a_1;")
    lines.append(f"{pad}threadgroup {element_type}* sb_curr = shared_b_0;")
    lines.append(f"{pad}threadgroup {element_type}* sb_next = shared_b_1;")

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
                        aliases = {"shared_a": "sa_next", "shared_b": "sb_next"}
                        staged_load = replace(
                            inner_op, tg_array=aliases.get(inner_op.tg_array, inner_op.tg_array)
                        )
                        _emit_gemm_op(staged_load, lines, indent + 2, func, has_swizzle)
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
        f"{pad}    {{ threadgroup {element_type}* _t = sa_curr; sa_curr = sa_next; sa_next = _t; }}"
    )
    lines.append(
        f"{pad}    {{ threadgroup {element_type}* _t = sb_curr; sb_curr = sb_next; sb_next = _t; }}"
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

    is_double_buffered = op.staging is not None or getattr(op, "_double_buffered", False)
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
