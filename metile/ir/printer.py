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
        _format_tile_op(op, lines, indent=1)

    lines.append("}")
    return "\n".join(lines)


def _format_tile_op(op: tir.Op, lines: list[str], indent: int = 1):
    pad = "  " * indent
    result = op.result
    prefix = f"%{result.name} = " if result else ""
    suffix = f" : {result.type}" if result else ""

    if isinstance(op, tir.ProgramId):
        lines.append(f"{pad}{prefix}program_id(axis={op.axis}){suffix}")
    elif isinstance(op, tir.ThreadId):
        lines.append(f"{pad}{prefix}thread_id(){suffix}")
    elif isinstance(op, tir.Constant):
        lines.append(f"{pad}{prefix}constant({op.value}){suffix}")
    elif isinstance(op, tir.Arange):
        start = f"%{op.start.name}" if op.start else "0"
        lines.append(f"{pad}{prefix}arange({start}, {start}+{op.size}){suffix}")
    elif isinstance(op, tir.BinOp):
        lines.append(f"{pad}{prefix}{op.op}(%{op.lhs.name}, %{op.rhs.name}){suffix}")
    elif isinstance(op, tir.Unary):
        lines.append(f"{pad}{prefix}{op.op}(%{op.operand.name}){suffix}")
    elif isinstance(op, tir.Reduce):
        lines.append(f"{pad}{prefix}reduce_{op.op}(%{op.operand.name}){suffix}")
    elif isinstance(op, tir.Select):
        lines.append(
            f"{pad}{prefix}select(%{op.condition.name}, %{op.true_val.name}, %{op.false_val.name}){suffix}"
        )
    elif isinstance(op, tir.Compare):
        lines.append(f"{pad}{prefix}cmp_{op.predicate}(%{op.lhs.name}, %{op.rhs.name}){suffix}")
    elif isinstance(op, tir.Load):
        mask = f", mask=%{op.mask.name}" if op.mask else ""
        lines.append(f"{pad}{prefix}load(%{op.ptr.name}, %{op.offsets.name}{mask}){suffix}")
    elif isinstance(op, tir.Store):
        mask = f", mask=%{op.mask.name}" if op.mask else ""
        lines.append(f"{pad}store(%{op.ptr.name}, %{op.offsets.name}, %{op.value.name}{mask})")
    elif isinstance(op, tir.PtrOffset):
        lines.append(f"{pad}{prefix}ptr_offset(%{op.ptr.name}, %{op.offsets.name}){suffix}")
    elif isinstance(op, tir.Zeros):
        lines.append(f"{pad}{prefix}zeros(shape={op.shape}, dtype={op.dtype}){suffix}")
    elif isinstance(op, tir.Dot):
        lines.append(f"{pad}{prefix}dot(%{op.a.name}, %{op.b.name}, %{op.acc.name}){suffix}")
    elif isinstance(op, tir.TileLoad):
        lines.append(
            f"{pad}{prefix}tile_load(%{op.ptr.name}, %{op.row_offset.name}, "
            f"%{op.col_offset.name}, stride=%{op.stride.name}, shape={op.tile_shape}){suffix}"
        )
    elif isinstance(op, tir.TileStore):
        lines.append(
            f"{pad}tile_store(%{op.ptr.name}, %{op.row_offset.name}, "
            f"%{op.col_offset.name}, stride=%{op.stride.name}, %{op.value.name}, "
            f"shape={op.tile_shape})"
        )
    elif isinstance(op, tir.SharedAlloc):
        lines.append(f"{pad}{prefix}shared_alloc(size={op.size}, dtype={op.dtype}){suffix}")
    elif isinstance(op, tir.Barrier):
        lines.append(f"{pad}barrier()")
    elif isinstance(op, tir.SimdShuffleXor):
        lines.append(f"{pad}{prefix}simd_shuffle_xor(%{op.value.name}, %{op.mask.name}){suffix}")
    elif isinstance(op, tir.SimdBroadcast):
        lines.append(f"{pad}{prefix}simd_broadcast(%{op.value.name}, %{op.lane.name}){suffix}")
    elif isinstance(op, tir.SimdLaneId):
        lines.append(f"{pad}{prefix}simd_lane_id(){suffix}")
    elif isinstance(op, tir.ForRange):
        start = f"%{op.start.name}" if hasattr(op.start, "name") else str(op.start)
        end = f"%{op.end.name}" if hasattr(op.end, "name") else str(op.end)
        lines.append(f"{pad}for %{op.iv.name} in range({start}, {end}, step={op.step}) {{")
        for body_op in op.body:
            _format_tile_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, tir.PersistentRange):
        lines.append(f"{pad}persistent_range(total={op.total}) {{")
        for body_op in op.body:
            _format_tile_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, tir.SimdgroupRole):
        lines.append(f"{pad}simdgroup_role(role={op.role}/{op.num_roles}) {{")
        for body_op in op.body:
            _format_tile_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    else:
        lines.append(f"{pad}{prefix}{type(op).__name__}(...){suffix}")


def print_metal_ir(func: mir.MFunction) -> str:
    """Pretty-print a Metal IR function."""
    lines = []
    params_str = ", ".join(f"%{p.name}: {p.type}" for p in func.params)
    lines.append(
        f"metal_func @{func.name}({params_str}) "
        f"[kernel_type={func.kernel_type}, tg={func.threadgroup_size}] {{"
    )

    for op in func.ops:
        _format_metal_op(op, lines, indent=1)

    lines.append("}")
    return "\n".join(lines)


def _val(v):
    """Format an MValue reference."""
    if v is None:
        return "?"
    return f"%{v.name}"


def _format_metal_op(op: mir.MOp, lines: list[str], indent: int = 1):
    pad = "  " * indent
    result = getattr(op, "result", None)
    prefix = f"%{result.name} = " if result else ""

    # Thread position ops
    if isinstance(op, mir.ThreadPositionInGrid):
        lines.append(f"{pad}{prefix}thread_position_in_grid.{'xyz'[op.axis]}")
    elif isinstance(op, mir.ThreadgroupPositionInGrid):
        lines.append(f"{pad}{prefix}threadgroup_position_in_grid.{'xyz'[op.axis]}")
    elif isinstance(op, mir.ThreadPositionInThreadgroup):
        lines.append(f"{pad}{prefix}thread_position_in_threadgroup.{'xyz'[op.axis]}")
    elif isinstance(op, mir.MSimdgroupId):
        lines.append(f"{pad}{prefix}simdgroup_index_in_threadgroup")
    elif isinstance(op, mir.MThreadInSimdgroup):
        lines.append(f"{pad}{prefix}thread_index_in_simdgroup")

    # Constants and arithmetic
    elif isinstance(op, mir.MConstant):
        lines.append(f"{pad}{prefix}constant({op.value}, {op.dtype})")
    elif isinstance(op, mir.MBinOp):
        lines.append(f"{pad}{prefix}{op.op}({_val(op.lhs)}, {_val(op.rhs)})")
    elif isinstance(op, mir.MCast):
        lines.append(f"{pad}{prefix}cast({_val(op.value)}, {op.target_dtype})")
    elif isinstance(op, mir.MUnary):
        lines.append(f"{pad}{prefix}{op.op}({_val(op.operand)})")
    elif isinstance(op, mir.MSelect):
        lines.append(
            f"{pad}{prefix}select({_val(op.condition)}, {_val(op.true_val)}, {_val(op.false_val)})"
        )
    elif isinstance(op, mir.MCompare):
        lines.append(f"{pad}{prefix}cmp_{op.predicate}({_val(op.lhs)}, {_val(op.rhs)})")

    # Memory ops
    elif isinstance(op, mir.DeviceLoad):
        lines.append(f"{pad}{prefix}device_load({_val(op.ptr)}, {_val(op.index)})")
    elif isinstance(op, mir.DeviceStore):
        lines.append(f"{pad}device_store({_val(op.ptr)}, {_val(op.index)}, {_val(op.value)})")
    elif isinstance(op, mir.MThreadgroupAlloc):
        lines.append(f"{pad}threadgroup_alloc({op.alloc_name}, {op.elem_type}, size={op.size})")
    elif isinstance(op, mir.MThreadgroupLoad):
        lines.append(f"{pad}{prefix}threadgroup_load({op.array_name}, {_val(op.index)})")
    elif isinstance(op, mir.MThreadgroupStore):
        lines.append(f"{pad}threadgroup_store({op.array_name}, {_val(op.index)}, {_val(op.value)})")
    elif isinstance(op, mir.MCooperativeLoad):
        bounds = "bounds_check" if op.bounds_check else "no_bounds"
        swizzle = f", swizzle={op.swizzle_bits}b" if op.swizzle_bits > 0 else ""
        lines.append(
            f"{pad}cooperative_load({_val(op.device_ptr)} -> {op.tg_array}, "
            f"{op.tile_rows}x{op.tile_cols}, stride={op.dst_stride}, "
            f"vec={op.vec_size}, {bounds}{swizzle})"
        )

    # Barriers
    elif isinstance(op, mir.MBarrier):
        lines.append(f"{pad}barrier({op.kind}, {op.flags})")

    # Simdgroup ops
    elif isinstance(op, mir.MSimdgroupAccDecl):
        lines.append(
            f"{pad}simdgroup_acc_decl({op.acc_name}, {op.num_8m}x{op.num_8n}, {op.in_type})"
        )
    elif isinstance(op, mir.MSimdgroupLoad):
        src = "B" if op.is_b else "A"
        lines.append(
            f"{pad}simdgroup_load({op.tile_name}[{op.tile_idx}], {op.src_array}, "
            f"type={src}, stride={op.stride})"
        )
    elif isinstance(op, mir.MSimdgroupMMA):
        lines.append(
            f"{pad}simdgroup_mma({op.acc_name}[{op.mi}][{op.ni}], {op.a_tile}[{op.mi}], {op.b_tile}[{op.ni}])"
        )
    elif isinstance(op, mir.MSimdgroupBarrierOp):
        lines.append(f"{pad}simdgroup_barrier()")
    elif isinstance(op, mir.MSimdgroupStore):
        lines.append(
            f"{pad}simdgroup_store({op.acc_name}[{op.mi}][{op.ni}], {_val(op.device_ptr)})"
        )
    elif isinstance(op, mir.MAccElemApply):
        lines.append(f"{pad}acc_elem_apply({op.acc_name}, ops={op.operations})")
    elif isinstance(op, mir.MSimdShuffleXor):
        lines.append(f"{pad}{prefix}simd_shuffle_xor({_val(op.value)}, {_val(op.mask)})")
    elif isinstance(op, mir.MSimdBroadcast):
        lines.append(f"{pad}{prefix}simd_broadcast({_val(op.value)}, {_val(op.lane)})")

    # Tensor ops
    elif isinstance(op, mir.MTensorViewDecl):
        lines.append(
            f"{pad}tensor_view_decl("
            f"A={_val(op.ptr_a)}, B={_val(op.ptr_b)}, C={_val(op.ptr_c)}, "
            f"in={op.in_type}, out={op.out_type})"
        )
    elif isinstance(op, mir.MTileSchedule):
        lines.append(f"{pad}tile_schedule(pattern={op.pattern}, block={op.block_m}x{op.block_n})")
    elif isinstance(op, mir.MMatmul2dSetup):
        mode = "cooperative" if op.cooperative else "preemptive"
        relaxed = "relaxed" if op.relaxed else "strict"
        separated = ", separated_loads" if op.use_separated else ""
        lines.append(
            f"{pad}matmul2d_setup(SM={op.sm}, SN={op.sn}, BK={op.bk}, "
            f"block={op.block_m}x{op.block_n}, WM={op.wm}xWN={op.wn}, "
            f"{mode}, {relaxed}, {op.num_sg} SGs{separated})"
        )
    elif isinstance(op, mir.MCoopTensorInit):
        separated = "separated" if op.use_separated else "direct"
        lines.append(
            f"{pad}coop_tensor_init({op.ct_name}, acc={op.acc_type}, in={op.in_type}, {separated})"
        )
    elif isinstance(op, mir.MCoopTensorLoad):
        lines.append(
            f"{pad}coop_tensor_load({op.ct_name}, {op.tensor_name}"
            f".slice<{op.slice_d0}, {op.slice_d1}>({op.offset_0}, {op.offset_1}))"
        )
    elif isinstance(op, mir.MMatmul2dRun):
        if op.use_tensor_view:
            lines.append(
                f"{pad}matmul2d_run("
                f"{op.a_tensor}.slice<{op.a_slice_d0},{op.a_slice_d1}>({op.a_offset_0},{op.a_offset_1}), "
                f"{op.b_tensor}.slice<{op.b_slice_d0},{op.b_slice_d1}>({op.b_offset_0},{op.b_offset_1}) "
                f"-> {op.ct_out})"
            )
        else:
            lines.append(f"{pad}matmul2d_run({op.ct_a}, {op.ct_b} -> {op.ct_out})")
    elif isinstance(op, mir.MCoopTensorEpilogue):
        lines.append(f"{pad}coop_tensor_epilogue({op.ct_name}, ops={op.operations})")
    elif isinstance(op, mir.MCoopTensorStore):
        lines.append(f"{pad}coop_tensor_store({op.ct_name} -> {op.output_slice})")

    # Reductions
    elif isinstance(op, mir.MThreadgroupReduce):
        lines.append(f"{pad}{prefix}threadgroup_reduce_{op.reduce_op}({_val(op.operand)})")

    # Variables
    elif isinstance(op, mir.MVarDecl):
        lines.append(f"{pad}var {op.var_name} = {_val(op.init_value)} : {op.dtype}")
    elif isinstance(op, mir.MVarAssign):
        lines.append(f"{pad}{op.var_name} = {_val(op.value)}")

    # Control flow
    elif isinstance(op, mir.MForLoop):
        end = _val(op.end) if isinstance(op.end, mir.MValue) else str(op.end)
        start = _val(op.start) if isinstance(op.start, mir.MValue) else str(op.start)
        markers = []
        if getattr(op, "_unroll", False):
            markers.append("unroll")
        if getattr(op, "_aligned", False):
            markers.append("aligned")
        if getattr(op, "_is_tail", False):
            markers.append("tail")
        if getattr(op, "_double_buffered", False):
            markers.append("double_buf")
        tag = f" [{', '.join(markers)}]" if markers else ""
        lines.append(f"{pad}for {op.iv_name} in [{start}, {end}, step={op.step}]{tag} {{")
        for body_op in op.body:
            _format_metal_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, mir.IfBlock):
        lines.append(f"{pad}if {_val(op.condition)} {{")
        for body_op in op.body:
            _format_metal_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, mir.MWhileTrue):
        lines.append(f"{pad}while true {{")
        for body_op in op.body:
            _format_metal_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, mir.MSimdgroupRoleBlock):
        lines.append(f"{pad}simdgroup_role(sg={op.first_sg}..{op.first_sg + op.num_sgs}) {{")
        for body_op in op.body:
            _format_metal_op(body_op, lines, indent + 1)
        lines.append(f"{pad}}}")
    elif isinstance(op, mir.MPersistentGrab):
        lines.append(f"{pad}{prefix}persistent_grab(total={op.total_tiles})")
    elif isinstance(op, mir.MBreak):
        lines.append(f"{pad}break")

    else:
        lines.append(f"{pad}{prefix}{type(op).__name__}(...)")
