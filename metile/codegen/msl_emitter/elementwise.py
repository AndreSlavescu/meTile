"""MSL emission for element-wise kernels and their epilogues."""

from __future__ import annotations

from metile.codegen.msl_emitter.common import (
    _BINOP_SYMBOLS,
    _BINOP_SYMBOLS_EPILOGUE,
    _CMP_SYMBOLS,
    _UNARY_MSL,
    _fold_vector_lanes,
    _format_float_literal,
    _format_literal,
    _uses_op_type,
    _uses_thread_position,
    _val_name,
)
from metile.ir import metal_ir as mir
from metile.ir.types import PtrType, ScalarType


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


def _emit_nax_apply_fragment(op, lines, indent):
    pad = "    " * indent
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (ushort i = 0; i < 8; ++i) {{")
    _emit_epilogue_chain(op.operations, f"{op.source}[i]", lines, f"{pad}    ")
    lines.append(f"{pad}}}")


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

    elif isinstance(op, mir.MVarDecl) and op.tile_valued:
        # Stands in for a tile element, so it has to be as wide as the lane group and
        # the identity has to reach every lane.
        msl_type = ScalarType(op.dtype).to_msl()
        vector_type = f"{msl_type}{vec_size}"
        vec4_vals[op.var_name] = vector_type
        init = _val_name(op.init_value, func)
        lines.append(f"{pad}{vector_type} {op.var_name} = {vector_type}({init});")

    elif isinstance(op, mir.MVarAssign):
        val = _val_name(op.value, func)
        lines.append(f"{pad}{op.var_name} = {val};")

    elif isinstance(op, mir.MConstant):
        msl_type = ScalarType(op.dtype).to_msl()
        lines.append(f"{pad}{msl_type} {op.result.name} = {_format_literal(op.value, op.dtype)};")

    elif isinstance(op, mir.MThreadgroupReduce):
        # Reducing inside a vectorized loop: the operand holds one vector per thread, so
        # the lanes have to be folded before the cross-thread reduction.
        _emit_threadgroup_reduce(op, lines, indent, func, vec4_vals)

    else:
        # Fallback: emit with normal path
        _emit_op(op, lines, indent, func)


def _emit_threadgroup_reduce(
    op: mir.MThreadgroupReduce,
    lines: list[str],
    indent: int,
    func: mir.MFunction,
    vec4_vals: dict[str, str] | None = None,
):
    """Emit threadgroup reduction: simd_sum + shared memory tree + broadcast."""
    pad = "    " * indent
    operand = _val_name(op.operand, func)
    name = op.result.name
    num_sg = op.block_size // 32
    msl_type = ScalarType(op.dtype).to_msl()

    vector_type = (vec4_vals or {}).get(operand)
    if vector_type is not None:
        operand = _fold_vector_lanes(operand, vector_type, op.reduce_op, msl_type, name, lines, pad)

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
