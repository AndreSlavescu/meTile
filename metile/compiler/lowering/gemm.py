"""Lowering for the GEMM kernel families."""

from __future__ import annotations

from metile.compiler.lowering.common import (
    LoweringError,
    _build_acc_stores,
    _build_kk_loop,
    _check_tg_memory,
    _compute_coop_load_layout,
    _compute_simdgroup_layout,
    _detect_dtype,
    _detect_epilogue,
    _extract_gemm_params,
    _has_gemm_ops,
    _lower_params,
    _select_num_sg,
)
from metile.compiler.schedules import validate_schedule
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import U32, PtrType, ScalarType


def _lower_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a GEMM Tile IR function to structured Metal IR.

    Produces Metal IR with MCooperativeLoad, MSimdgroupLoad/MMA, etc.
    These ops are then progressively optimized by passes.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="gemm")

    _, msl_type = _detect_dtype(func)
    param_values = _lower_params(func, mfunc)

    # Extract tile shapes from constexprs and ops
    constexprs = func.constexprs
    BM = constexprs.get("BLOCK_M", 32)
    BN = constexprs.get("BLOCK_N", 32)
    BK = constexprs.get("BLOCK_K", 16)

    # Simdgroup layout: derived from layout algebra
    NUM_SG = constexprs.get("NUM_SG", _select_num_sg(BM, BN))
    sg_layout = _compute_simdgroup_layout(BM, BN, NUM_SG)
    SG_COLS = sg_layout.sg_cols
    SG_M = sg_layout.sg_m
    SG_N = sg_layout.sg_n
    TG_SIZE = NUM_SG * 32

    # Cooperative load layouts: derived from layout algebra
    a_load_layout = _compute_coop_load_layout(BM, BK, TG_SIZE)
    b_load_layout = _compute_coop_load_layout(BK, BN, TG_SIZE)

    # Threadgroup memory strides (no padding initially - passes add it)
    A_STRIDE = BK
    B_STRIDE = BN

    # Validate threadgroup memory fits
    _check_tg_memory({"shared_a": BM * A_STRIDE, "shared_b": BK * B_STRIDE}, msl_type, "GEMM")

    mfunc.threadgroup_size = (TG_SIZE, 1, 1)

    # --- Emit thread indexing ---
    sgid = mfunc.add_op(mir.MSimdgroupId(), "sgid")
    slid = mfunc.add_op(mir.MThreadInSimdgroup(), "slid")

    # linear_tid = sgid * 32 + slid
    c32 = mfunc.add_op(mir.MConstant(value=32, dtype="u32"), "c32")
    sgid_x_32 = mfunc.add_op(mir.MBinOp(op="mul", lhs=sgid, rhs=c32), "sgid_x_32")
    linear_tid = mfunc.add_op(mir.MBinOp(op="add", lhs=sgid_x_32, rhs=slid), "linear_tid")

    # Block coordinates
    tgp_x = mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=0), "tgp_x")
    tgp_y = mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=1), "tgp_y")
    c_bm = mfunc.add_op(mir.MConstant(value=BM, dtype="u32"), "c_bm")
    c_bn = mfunc.add_op(mir.MConstant(value=BN, dtype="u32"), "c_bn")
    block_row = mfunc.add_op(mir.MBinOp(op="mul", lhs=tgp_x, rhs=c_bm), "block_row")
    block_col = mfunc.add_op(mir.MBinOp(op="mul", lhs=tgp_y, rhs=c_bn), "block_col")

    # Simdgroup coordinates
    c_sg_cols = mfunc.add_op(mir.MConstant(value=SG_COLS, dtype="u32"), "c_sg_cols")
    c_sg_m = mfunc.add_op(mir.MConstant(value=SG_M, dtype="u32"), "c_sg_m")
    c_sg_n = mfunc.add_op(mir.MConstant(value=SG_N, dtype="u32"), "c_sg_n")
    sg_row_idx = mfunc.add_op(mir.MBinOp(op="div", lhs=sgid, rhs=c_sg_cols), "sg_row_idx")
    sg_col_idx = mfunc.add_op(mir.MBinOp(op="mod", lhs=sgid, rhs=c_sg_cols), "sg_col_idx")
    sg_row = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_row_idx, rhs=c_sg_m), "sg_row")
    sg_col = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_col_idx, rhs=c_sg_n), "sg_col")

    # --- Threadgroup memory allocation ---
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_a", elem_type=msl_type, size=BM * A_STRIDE)
    )
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_b", elem_type=msl_type, size=BK * B_STRIDE)
    )

    # --- Accumulator init ---
    NUM_8M = SG_M // 8
    NUM_8N = SG_N // 8
    mfunc.add_op(
        mir.MSimdgroupAccDecl(
            acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type, acc_type="float"
        )
    )

    # --- Find the K-param, A/B/C pointers from the traced IR ---
    ptr_A, ptr_B, ptr_C, M_val, N_val, K_val = _extract_gemm_params(func, param_values)

    # --- K-loop ---
    loop_body: list[mir.MOp] = []

    # Cooperative load A tile
    loop_body.append(
        mir.MCooperativeLoad(
            device_ptr=ptr_A,
            tg_array="shared_a",
            row_offset=block_row,
            col_offset=None,  # col_offset is loop iv
            src_stride=K_val,
            tile_rows=BM,
            tile_cols=BK,
            dst_stride=A_STRIDE,
            tg_size=TG_SIZE,
            linear_tid=linear_tid,
            bounds_check=True,
            row_bound=M_val,
            col_bound=K_val,
            vec_size=1,
            elem_type=msl_type,
            load_layout=a_load_layout,
        )
    )

    # Cooperative load B tile
    loop_body.append(
        mir.MCooperativeLoad(
            device_ptr=ptr_B,
            tg_array="shared_b",
            row_offset=None,  # row_offset is loop iv
            col_offset=block_col,
            src_stride=N_val,
            tile_rows=BK,
            tile_cols=BN,
            dst_stride=B_STRIDE,
            tg_size=TG_SIZE,
            linear_tid=linear_tid,
            bounds_check=True,
            row_bound=K_val,
            col_bound=N_val,
            vec_size=1,
            elem_type=msl_type,
            load_layout=b_load_layout,
        )
    )

    # Barrier
    loop_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    # MMA inner loop
    loop_body.append(
        _build_kk_loop(
            NUM_8M, NUM_8N, sg_row, sg_col, "shared_a", "shared_b", A_STRIDE, B_STRIDE, msl_type, BK
        )
    )

    # Barrier
    loop_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    # Emit for loop
    mfunc.add_op(
        mir.MForLoop(
            iv_name="kb",
            start=0,
            end=K_val,
            step=BK,
            body=loop_body,
        )
    )

    # --- Detect and emit epilogue (fused element-wise ops on accumulators) ---
    epilogue = _detect_epilogue(func.ops)
    if epilogue:
        mfunc.add_op(
            mir.MAccElemApply(
                acc_name="acc",
                num_8m=NUM_8M,
                num_8n=NUM_8N,
                operations=epilogue,
            )
        )

    # --- Store accumulators ---
    for store_op in _build_acc_stores(
        NUM_8M, NUM_8N, ptr_C, block_row, block_col, sg_row, sg_col, N_val, M_val, msl_type
    ):
        mfunc.add_op(store_op)

    return mfunc


def _lower_specialized_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a GEMM with explicit simdgroup_role producer/consumer blocks.

    Detects SimdgroupRole blocks inside the K-loop ForRange:
    - Role containing tile_load → producer (cooperative loads to shared memory)
    - Role containing dot → consumer (MMA from shared memory)

    Double-buffered: producers prefetch tile k+1 while consumers compute tile k.
    Barriers are placed OUTSIDE role blocks so all SGs participate.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="specialized_gemm")

    _, msl_type = _detect_dtype(func)
    param_values = _lower_params(func, mfunc)

    # Find the K-loop ForRange and extract producer/consumer roles
    k_loop = None
    for op in func.ops:
        if isinstance(op, tir.ForRange):
            roles = [b for b in op.body if isinstance(b, tir.SimdgroupRole)]
            if roles and _has_gemm_ops(op.body):
                k_loop = op
                break
    if k_loop is None:
        raise LoweringError("Specialized GEMM: no ForRange with SimdgroupRole found")

    producer_role = consumer_role = None
    for role_op in (b for b in k_loop.body if isinstance(b, tir.SimdgroupRole)):
        has_loads = any(isinstance(b, tir.TileLoad) for b in role_op.body)
        has_dot = any(isinstance(b, tir.Dot) for b in role_op.body)
        if has_loads:
            producer_role = role_op
        if has_dot:
            consumer_role = role_op

    if producer_role is None or consumer_role is None:
        raise LoweringError(
            "Specialized GEMM: need one role with tile_load (producer) and one with dot (consumer)"
        )

    # Extract tile shapes
    constexprs = func.constexprs
    BM = constexprs.get("BLOCK_M", 64)
    BN = constexprs.get("BLOCK_N", 64)
    BK = constexprs.get("BLOCK_K", 32)
    PRODUCER_SGS = producer_role.num_sgs or 2
    CONSUMER_SGS = consumer_role.num_sgs or 4
    TOTAL_SGS = PRODUCER_SGS + CONSUMER_SGS
    TG_SIZE = TOTAL_SGS * 32
    PRODUCER_THREADS = PRODUCER_SGS * 32

    # Consumer SG layout (only consumer SGs compute MMA)
    sg_layout = _compute_simdgroup_layout(BM, BN, CONSUMER_SGS)
    SG_COLS = sg_layout.sg_cols
    SG_M = sg_layout.sg_m
    SG_N = sg_layout.sg_n
    NUM_8M = SG_M // 8
    NUM_8N = SG_N // 8

    # Use shared padding logic from passes module
    from metile.compiler.passes import _optimal_pad

    A_PAD = _optimal_pad(BK)
    B_PAD = _optimal_pad(BN)
    A_STRIDE = BK + A_PAD
    B_STRIDE = BN + B_PAD

    mfunc.threadgroup_size = (TG_SIZE, 1, 1)

    # --- Thread indexing ---
    sgid = mfunc.add_op(mir.MSimdgroupId(), "sgid")
    slid = mfunc.add_op(mir.MThreadInSimdgroup(), "slid")

    # Producer linear_tid = sgid * 32 + slid (for cooperative loads)
    c32 = mfunc.add_op(mir.MConstant(value=32, dtype="u32"), "c32")
    sgid_x_32 = mfunc.add_op(mir.MBinOp(op="mul", lhs=sgid, rhs=c32), "sgid_x_32")
    linear_tid = mfunc.add_op(mir.MBinOp(op="add", lhs=sgid_x_32, rhs=slid), "linear_tid")

    # Block coordinates
    tgp_x = mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=0), "tgp_x")
    tgp_y = mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=1), "tgp_y")
    c_bm = mfunc.add_op(mir.MConstant(value=BM, dtype="u32"), "c_bm")
    c_bn = mfunc.add_op(mir.MConstant(value=BN, dtype="u32"), "c_bn")
    block_row = mfunc.add_op(mir.MBinOp(op="mul", lhs=tgp_x, rhs=c_bm), "block_row")
    block_col = mfunc.add_op(mir.MBinOp(op="mul", lhs=tgp_y, rhs=c_bn), "block_col")

    # Consumer SG coordinates: offset sgid by PRODUCER_SGS, then compute grid pos
    c_prod_sgs = mfunc.add_op(mir.MConstant(value=PRODUCER_SGS, dtype="u32"), "c_prod_sgs")
    consumer_sgid = mfunc.add_op(mir.MBinOp(op="sub", lhs=sgid, rhs=c_prod_sgs), "consumer_sgid")
    c_sg_cols = mfunc.add_op(mir.MConstant(value=SG_COLS, dtype="u32"), "c_sg_cols")
    c_sg_m = mfunc.add_op(mir.MConstant(value=SG_M, dtype="u32"), "c_sg_m")
    c_sg_n = mfunc.add_op(mir.MConstant(value=SG_N, dtype="u32"), "c_sg_n")
    sg_row_idx = mfunc.add_op(mir.MBinOp(op="div", lhs=consumer_sgid, rhs=c_sg_cols), "sg_row_idx")
    sg_col_idx = mfunc.add_op(mir.MBinOp(op="mod", lhs=consumer_sgid, rhs=c_sg_cols), "sg_col_idx")
    sg_row = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_row_idx, rhs=c_sg_m), "sg_row")
    sg_col = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_col_idx, rhs=c_sg_n), "sg_col")

    # --- Double-buffered threadgroup memory ---
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_a_0", elem_type=msl_type, size=BM * A_STRIDE)
    )
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_a_1", elem_type=msl_type, size=BM * A_STRIDE)
    )
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_b_0", elem_type=msl_type, size=BK * B_STRIDE)
    )
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_b_1", elem_type=msl_type, size=BK * B_STRIDE)
    )

    # Check double-buffer fits in 32KB
    db_bytes = 2 * (BM * A_STRIDE + BK * B_STRIDE) * 4  # float = 4 bytes
    if db_bytes > 32768:
        raise LoweringError(
            f"Specialized GEMM double-buffering requires {db_bytes} bytes "
            f"threadgroup memory (limit 32KB). Reduce tile sizes."
        )

    # --- Accumulator init (consumers only, but OK for all — producers just waste registers) ---
    mfunc.add_op(
        mir.MSimdgroupAccDecl(
            acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type, acc_type="float"
        )
    )

    # --- Extract A, B, C, M, N, K ---
    ptr_A, ptr_B, ptr_C, M_val, N_val, K_val = _extract_gemm_params(func, param_values)

    # Helper to create a cooperative load op for a given buffer
    def _make_coop_load(
        ptr,
        tg_array,
        row_off,
        col_off,
        src_stride,
        tile_rows,
        tile_cols,
        dst_stride,
        row_bound,
        col_bound,
    ):
        return mir.MCooperativeLoad(
            device_ptr=ptr,
            tg_array=tg_array,
            row_offset=row_off,
            col_offset=col_off,
            src_stride=src_stride,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            dst_stride=dst_stride,
            tg_size=PRODUCER_THREADS,
            linear_tid=linear_tid,
            bounds_check=True,
            row_bound=row_bound,
            col_bound=col_bound,
            vec_size=1,
            elem_type=msl_type,
        )

    # --- Prologue: producers load first tile into buffer 0 ---
    prologue_loads = [
        _make_coop_load(
            ptr_A, "shared_a_0", block_row, None, K_val, BM, BK, A_STRIDE, M_val, K_val
        ),
        _make_coop_load(
            ptr_B, "shared_b_0", None, block_col, N_val, BK, BN, B_STRIDE, K_val, N_val
        ),
    ]
    # Mark prologue loads with kb_expr="0" so emitter uses 0 instead of loop IV
    prologue_loads[0].kb_expr = "0"
    prologue_loads[1].kb_expr = "0"

    mfunc.add_op(
        mir.MSimdgroupRoleBlock(
            role=0,
            num_roles=2,
            first_sg=0,
            num_sgs=PRODUCER_SGS,
            sgid=sgid,
            body=prologue_loads,
        )
    )
    mfunc.add_op(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    # --- Main K-loop: producers prefetch k+BK while consumers compute k ---
    # Mark loop as double-buffered for the emitter
    loop_body: list[mir.MOp] = []

    # Producers: prefetch next tile into alternate buffer
    prefetch_loads = [
        _make_coop_load(ptr_A, "shared_a", block_row, None, K_val, BM, BK, A_STRIDE, M_val, K_val),
        _make_coop_load(ptr_B, "shared_b", None, block_col, N_val, BK, BN, B_STRIDE, K_val, N_val),
    ]
    # kb_expr will be set by emitter to "kb + BK" for prefetch
    prefetch_loads[0].kb_expr = f"kb + {BK}"
    prefetch_loads[1].kb_expr = f"kb + {BK}"

    loop_body.append(
        mir.MSimdgroupRoleBlock(
            role=0,
            num_roles=2,
            first_sg=0,
            num_sgs=PRODUCER_SGS,
            sgid=sgid,
            body=prefetch_loads,
        )
    )

    # Consumers: compute MMA from current buffer
    kk_loop_main = _build_kk_loop(
        NUM_8M,
        NUM_8N,
        sg_row,
        sg_col,
        "shared_a",
        "shared_b",
        A_STRIDE,
        B_STRIDE,
        msl_type,
        BK,
    )
    mma_ops = [kk_loop_main]
    loop_body.append(
        mir.MSimdgroupRoleBlock(
            role=1,
            num_roles=2,
            first_sg=PRODUCER_SGS,
            num_sgs=CONSUMER_SGS,
            sgid=sgid,
            body=mma_ops,
        )
    )

    # Barrier: producers done writing next tile, consumers done reading current tile
    loop_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    k_loop = mir.MForLoop(
        iv_name="kb",
        start=0,
        end=K_val,
        step=BK,
        body=loop_body,
    )
    # Mark as specialized double-buffered for the emitter
    k_loop._specialized_db = True
    k_loop._bk = BK
    k_loop._producer_sgs = PRODUCER_SGS
    k_loop._consumer_sgs = CONSUMER_SGS
    mfunc.add_op(k_loop)

    # --- Epilogue: consumers compute last tile from current buffer ---
    kk_loop_epilogue = _build_kk_loop(
        NUM_8M,
        NUM_8N,
        sg_row,
        sg_col,
        "sa_curr",
        "sb_curr",
        A_STRIDE,
        B_STRIDE,
        msl_type,
        BK,
    )
    mfunc.add_op(
        mir.MSimdgroupRoleBlock(
            role=1,
            num_roles=2,
            first_sg=PRODUCER_SGS,
            num_sgs=CONSUMER_SGS,
            sgid=sgid,
            body=[kk_loop_epilogue],
        )
    )

    # --- Detect and emit fused epilogue ---
    epilogue = _detect_epilogue(func.ops)
    if epilogue:
        mfunc.add_op(
            mir.MAccElemApply(
                acc_name="acc",
                num_8m=NUM_8M,
                num_8n=NUM_8N,
                operations=epilogue,
            )
        )

    # --- Store accumulators (consumers only) ---
    store_ops = _build_acc_stores(
        NUM_8M, NUM_8N, ptr_C, block_row, block_col, sg_row, sg_col, N_val, M_val, msl_type
    )
    mfunc.add_op(
        mir.MSimdgroupRoleBlock(
            role=1,
            num_roles=2,
            first_sg=PRODUCER_SGS,
            num_sgs=CONSUMER_SGS,
            sgid=sgid,
            body=store_ops,
        )
    )

    return mfunc


def _lower_tensor_ops_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a GEMM to Metal 4 tensor_ops matmul2d.

    Uses preemptive execution: each simdgroup independently runs matmul2d
    on its own subtile. Manual K-tiling with cooperative_tensor accumulation
    in registers. Static template slice<> for optimal codegen.

    Auto-selected when device supports Metal 4 tensor_ops.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="tensor_ops_gemm")

    _, msl_type = _detect_dtype(func)
    param_values = _lower_params(func, mfunc)

    # Extract tile shapes — defaults tuned from benchmarks
    constexprs = func.constexprs
    BM = constexprs.get("BLOCK_M", 128)
    BN = constexprs.get("BLOCK_N", 64)
    BK = constexprs.get("BLOCK_K", 32)
    WM = constexprs.get("WM", 2)
    WN = constexprs.get("WN", 2)
    relaxed = constexprs.get("RELAXED_PRECISION", True)
    cooperative = constexprs.get("COOPERATIVE", False)
    # User-specified tile_swizzle() takes priority, then constexpr, then compiler default
    if func.swizzle_pattern is not None:
        swizzle = func.swizzle_pattern
    else:
        swizzle = constexprs.get("SWIZZLE", "auto")
    swizzle = validate_schedule(swizzle)
    swizzle_block_size = func.swizzle_block_size if func.swizzle_pattern is not None else 4
    k_unroll = constexprs.get("K_UNROLL", 1)
    num_stages = constexprs.get("num_stages", 1)

    NUM_SG = WM * WN
    mfunc.threadgroup_size = (NUM_SG * 32, 1, 1)

    ptr_A, ptr_B, ptr_C, _M_val, _N_val, K_val = _extract_gemm_params(func, param_values)

    # Detect epilogue ops
    epilogue = _detect_epilogue(func.ops)

    # Derived constants
    SM = BM // WM
    SN = BN // WN
    acc_type = "float"
    out_type = msl_type

    if constexprs.get("NAX_FRAGMENTS", False):
        if msl_type not in {"float", "half"} or cooperative or SM != 32 or SN != 32 or BK != 16:
            raise ValueError(
                "NAX fragments require f16/f32, 32x32 per-simdgroup tiles, and BLOCK_K=16"
            )
        if not all(constexprs.get(f"_ALIGNED_{axis}", False) for axis in ("N", "K")):
            raise ValueError("NAX fragments currently require aligned N and K")
        outer_k = int(constexprs.get("NAX_OUTER_K", 0))
        nax_k_unroll = int(constexprs.get("NAX_K_UNROLL", 1))
        if nax_k_unroll not in {1, 2}:
            raise ValueError("NAX_K_UNROLL must be 1 or 2")
        reduction_step = 16 * nax_k_unroll
        static_shape = tuple(constexprs.get(f"_STATIC_{axis}") for axis in ("M", "N", "K"))
        if all(dimension is not None for dimension in static_shape):
            static_m, static_n, static_k = (int(dimension) for dimension in static_shape)
            mfunc.params = [param for param in mfunc.params if param.name not in {"M", "N", "K"}]
            K_val = static_k
        else:
            static_m = static_n = static_k = 0
        row_bound = static_m if not constexprs.get("_ALIGNED_M", False) else 0
        if outer_k and (
            outer_k % reduction_step or not constexprs.get("_ALIGNED_NAX_OUTER_K", False)
        ):
            raise ValueError("NAX_OUTER_K must align to the reduction step and evenly divide K")
        mfunc.add_op(mir.MThreadInSimdgroup())
        mfunc.add_op(
            mir.MTileSchedule(
                pattern=swizzle,
                block_m=BM,
                block_n=BN,
                block_size=swizzle_block_size,
                grid_m=constexprs.get("_GRID_M"),
                grid_n=constexprs.get("_GRID_N"),
                encoding=constexprs.get("SCHEDULE_ENCODING", "auto"),
            )
        )
        mfunc.add_op(
            mir.MNaxGemmSetup(
                block_m=BM,
                block_n=BN,
                wm=WM,
                wn=WN,
                m=static_m,
                n=static_n,
                k=static_k,
                left_type=msl_type,
                right_type=msl_type,
            )
        )
        if outer_k:
            epoch_a_op = mir.MPointerOffset(ptr=ptr_A, offset="k0")
            epoch_a = mir.MValue("nax_epoch_a", epoch_a_op.result_type(), epoch_a_op)
            epoch_a_op.result = epoch_a
            epoch_b_op = mir.MPointerOffset(ptr=ptr_B, offset="k0 * N")
            epoch_b = mir.MValue("nax_epoch_b", epoch_b_op.result_type(), epoch_b_op)
            epoch_b_op.result = epoch_b
            inner_loop = mir.MForLoop(
                iv_name="k1",
                start=0,
                end=outer_k,
                step=reduction_step,
                body=[
                    mir.MNaxGemmRun(
                        ptr_a=epoch_a,
                        ptr_b=epoch_b,
                        k_offset=offset,
                        row_bound=row_bound,
                    )
                    for offset in range(0, reduction_step, 16)
                ],
                index_alias="k",
                index_expression="k1",
            )
            if constexprs.get("NAX_TRAILING_EPOCH_BARRIER", False):
                epoch_body = [
                    epoch_a_op,
                    epoch_b_op,
                    inner_loop,
                    mir.MBarrier(
                        kind="threadgroup",
                        flags="mem_none",
                        condition=f"k0 + {outer_k}u < K",
                    ),
                ]
            else:
                epoch_body = [
                    mir.MBarrier(
                        kind="threadgroup",
                        flags="mem_none",
                        condition=(
                            "k0 != 0u"
                            if constexprs.get("NAX_SKIP_FIRST_EPOCH_BARRIER", False)
                            else None
                        ),
                    ),
                    epoch_a_op,
                    epoch_b_op,
                    inner_loop,
                ]
            mfunc.add_op(
                mir.MForLoop(
                    iv_name="k0",
                    start=0,
                    end=K_val,
                    step=outer_k,
                    body=epoch_body,
                )
            )
        else:
            mfunc.add_op(
                mir.MForLoop(
                    iv_name="k",
                    start=0,
                    end=K_val,
                    step=reduction_step,
                    body=[
                        mir.MNaxGemmRun(
                            ptr_a=ptr_A,
                            ptr_b=ptr_B,
                            k_offset=offset,
                            row_bound=row_bound,
                        )
                        for offset in range(0, reduction_step, 16)
                    ],
                )
            )
        if epilogue:
            mfunc.add_op(mir.MNaxGemmEpilogue(operations=epilogue))
        mfunc.add_op(mir.MNaxGemmStore(ptr_c=ptr_C, row_bound=row_bound))
        return mfunc

    # Use separated loads when descriptor dimensions allow cooperative_tensor inputs
    separated_default = not cooperative and SM <= 32 and SN <= 32
    use_separated = constexprs.get("SEPARATED", separated_default) and not cooperative
    if use_separated and (SM > 32 or SN > 32):
        raise ValueError("separated tensor inputs require per-simdgroup M/N tiles <= 32")
    bk_inner = min(32, BK) if use_separated else BK

    # --- Emit decomposed tensor ops ---

    # 1. Tensor view declarations
    mfunc.add_op(
        mir.MTensorViewDecl(
            ptr_a=ptr_A,
            ptr_b=ptr_B,
            ptr_c=ptr_C,
            in_type=msl_type,
            out_type=out_type,
        )
    )

    # 2. Tile scheduling
    mfunc.add_op(
        mir.MTileSchedule(
            pattern=swizzle,
            block_m=BM,
            block_n=BN,
            block_size=swizzle_block_size,
            grid_m=constexprs.get("_GRID_M"),
            grid_n=constexprs.get("_GRID_N"),
            encoding=constexprs.get("SCHEDULE_ENCODING", "auto"),
        )
    )

    # 3. Matmul2d descriptor + operator setup
    mfunc.add_op(
        mir.MMatmul2dSetup(
            sm=SM,
            sn=SN,
            bk=bk_inner if use_separated else BK,
            block_m=BM,
            block_n=BN,
            wm=WM,
            wn=WN,
            relaxed=relaxed,
            cooperative=cooperative,
            num_sg=NUM_SG,
            in_type=msl_type,
            acc_type=acc_type,
            out_type=out_type,
            use_separated=use_separated,
        )
    )

    # 4. Cooperative tensor init (output accumulator)
    mfunc.add_op(
        mir.MCoopTensorInit(
            ct_name="cT",
            acc_type=acc_type,
            in_type=msl_type,
            use_separated=use_separated,
        )
    )

    # 5. K-loop with loads and compute
    if use_separated:
        # Separated mode: cooperative_tensor loads + op.run
        # No barriers needed; preemptive simdgroups are independent, each
        # loading directly from device memory into register-resident
        # cooperative_tensors with no shared threadgroup memory.
        #
        # Always use 2x when BK >= 2*bk_inner. User K_UNROLL can override.
        effective_unroll = 2 if 2 * bk_inner <= BK else 1
        effective_unroll = max(effective_unroll, k_unroll)

        k_body = []
        for u in range(effective_unroll):
            k_offset = f"k + {bk_inner * u}" if u > 0 else "k"
            k_body.append(
                mir.MCoopTensorLoad(
                    ct_name="ct_a",
                    tensor_name="tA",
                    slice_d0=bk_inner,
                    slice_d1=SM,
                    offset_0=k_offset,
                    offset_1="tile_row",
                )
            )
            k_body.append(
                mir.MCoopTensorLoad(
                    ct_name="ct_b",
                    tensor_name="tB",
                    slice_d0=SN,
                    slice_d1=bk_inner,
                    offset_0="tile_col",
                    offset_1=k_offset,
                )
            )
            k_body.append(
                mir.MMatmul2dRun(
                    ct_a="ct_a",
                    ct_b="ct_b",
                    ct_out="cT",
                    use_tensor_view=False,
                )
            )

        k_step = bk_inner * effective_unroll
        k_loop = mir.MForLoop(iv_name="k", start=0, end=K_val, step=k_step, body=k_body)
        if num_stages >= 2:
            k_loop._num_stages = num_stages
        mfunc.add_op(k_loop)
    else:
        # Direct tensor view mode: pass slices directly to op.run
        k_body = []
        for u in range(k_unroll):
            k_body.append(
                mir.MMatmul2dRun(
                    ct_a="ct_a",
                    ct_b="ct_b",
                    ct_out="cT",
                    use_tensor_view=True,
                    a_tensor="tA",
                    b_tensor="tB",
                    a_slice_d0=BK,
                    a_slice_d1=SM if not cooperative else BM,
                    b_slice_d0=SN if not cooperative else BN,
                    b_slice_d1=BK,
                    a_offset_0=f"k + {BK * u}" if u > 0 else "k",
                    a_offset_1="tile_row" if not cooperative else f"pid_m * {BM}u",
                    b_offset_0="tile_col" if not cooperative else f"pid_n * {BN}u",
                    b_offset_1=f"k + {BK * u}" if u > 0 else "k",
                )
            )
        k_step = BK * k_unroll
        k_loop = mir.MForLoop(iv_name="k", start=0, end=K_val, step=k_step, body=k_body)
        mfunc.add_op(k_loop)

    # 6. Epilogue (element-wise ops on cooperative_tensor)
    if epilogue:
        mfunc.add_op(
            mir.MCoopTensorEpilogue(
                ct_name="cT",
                operations=epilogue,
            )
        )

    # 7. Store output
    mfunc.add_op(
        mir.MCoopTensorStore(
            ct_name="cT",
            output_slice="mC",
        )
    )

    return mfunc


def _lower_persistent_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a persistent GEMM kernel.

    Wraps the standard GEMM body in a while(true) loop with atomic
    tile index grabbing. Thread 0 atomically increments a device-memory
    counter, broadcasts via threadgroup memory, and all threads break
    when tiles are exhausted.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="persistent_gemm")

    # Find the PersistentRange op
    persistent_op = None
    for op in func.ops:
        if isinstance(op, tir.PersistentRange):
            persistent_op = op
            break
    assert persistent_op is not None

    total_tiles = persistent_op.total

    _, msl_type = _detect_dtype(func)

    # Lower params — identify A, B, C, counter, M, N, K
    param_values: dict[str, mir.MValue] = {}
    ptr_params = []
    scalar_params = []
    counter_param_name = None

    # The counter pointer is the one referenced by the PersistentRange
    counter_ref_name = persistent_op.counter.name

    for p in func.params:
        if isinstance(p.type, PtrType):
            is_counter = p.name == counter_ref_name
            mp = mir.MParam(
                name=p.name,
                type=p.type,
                is_output=p.is_output,
                is_scalar=False,
                is_atomic=is_counter,
            )
            mfunc.params.append(mp)
            param_values[p.name] = mir.MValue(p.name, p.type)
            if is_counter:
                counter_param_name = p.name
            else:
                ptr_params.append(p.name)
        elif isinstance(p.type, ScalarType):
            mp = mir.MParam(name=p.name, type=p.type, is_scalar=True)
            mfunc.params.append(mp)
            param_values[p.name] = mir.MValue(p.name, p.type)
            scalar_params.append(p.name)

    # Assign A, B, C from non-counter pointer params (positional)
    assert len(ptr_params) >= 3, f"Need at least 3 pointer params (A, B, C), got {len(ptr_params)}"
    ptr_A = param_values[ptr_params[0]]
    ptr_B = param_values[ptr_params[1]]
    ptr_C = param_values[ptr_params[2]]
    counter_ptr = param_values[counter_param_name]

    # Find M, N, K
    M_val = N_val = K_val = None
    for p in func.params:
        if isinstance(p.type, ScalarType):
            pv = param_values[p.name]
            if p.name == "M":
                M_val = pv
            elif p.name == "N":
                N_val = pv
            elif p.name == "K":
                K_val = pv

    if K_val is None or M_val is None or N_val is None:
        scalars = [param_values[n] for n in scalar_params]
        if len(scalars) >= 3:
            M_val, N_val, K_val = scalars[0], scalars[1], scalars[2]
        else:
            raise LoweringError("Cannot determine M, N, K parameters")

    # Extract tile shapes
    constexprs = func.constexprs
    BM = constexprs.get("BLOCK_M", 64)
    BN = constexprs.get("BLOCK_N", 64)
    BK = constexprs.get("BLOCK_K", 16)

    NUM_SG = constexprs.get("NUM_SG", _select_num_sg(BM, BN))
    sg_layout = _compute_simdgroup_layout(BM, BN, NUM_SG)
    SG_COLS = sg_layout.sg_cols
    SG_M = sg_layout.sg_m
    SG_N = sg_layout.sg_n
    TG_SIZE = NUM_SG * 32

    a_load_layout = _compute_coop_load_layout(BM, BK, TG_SIZE)
    b_load_layout = _compute_coop_load_layout(BK, BN, TG_SIZE)

    A_STRIDE = BK
    B_STRIDE = BN
    NUM_8M = SG_M // 8
    NUM_8N = SG_N // 8

    mfunc.threadgroup_size = (TG_SIZE, 1, 1)

    # --- Thread indexing (outside while loop) ---
    sgid = mfunc.add_op(mir.MSimdgroupId(), "sgid")
    slid = mfunc.add_op(mir.MThreadInSimdgroup(), "slid")
    c32 = mfunc.add_op(mir.MConstant(value=32, dtype="u32"), "c32")
    sgid_x_32 = mfunc.add_op(mir.MBinOp(op="mul", lhs=sgid, rhs=c32), "sgid_x_32")
    linear_tid = mfunc.add_op(mir.MBinOp(op="add", lhs=sgid_x_32, rhs=slid), "linear_tid")

    # Simdgroup coordinates
    c_sg_cols = mfunc.add_op(mir.MConstant(value=SG_COLS, dtype="u32"), "c_sg_cols")
    c_sg_m = mfunc.add_op(mir.MConstant(value=SG_M, dtype="u32"), "c_sg_m")
    c_sg_n = mfunc.add_op(mir.MConstant(value=SG_N, dtype="u32"), "c_sg_n")
    sg_row_idx = mfunc.add_op(mir.MBinOp(op="div", lhs=sgid, rhs=c_sg_cols), "sg_row_idx")
    sg_col_idx = mfunc.add_op(mir.MBinOp(op="mod", lhs=sgid, rhs=c_sg_cols), "sg_col_idx")
    sg_row = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_row_idx, rhs=c_sg_m), "sg_row")
    sg_col = mfunc.add_op(mir.MBinOp(op="mul", lhs=sg_col_idx, rhs=c_sg_n), "sg_col")

    # --- Threadgroup memory ---
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_a", elem_type=msl_type, size=BM * A_STRIDE)
    )
    mfunc.add_op(
        mir.MThreadgroupAlloc(alloc_name="shared_b", elem_type=msl_type, size=BK * B_STRIDE)
    )
    mfunc.add_op(mir.MThreadgroupAlloc(alloc_name="shared_tile_idx", elem_type="uint", size=1))

    # --- Build while(true) body ---
    while_body: list[mir.MOp] = []

    # 1. Persistent tile grab (atomic + broadcast + break)
    grab_op = mir.MPersistentGrab(
        counter_ptr=counter_ptr,
        linear_tid=linear_tid,
        total_tiles=total_tiles,
        shared_name="shared_tile_idx",
        tile_idx_name="tile_idx",
    )
    grab_val = mir.MValue("tile_idx", U32, grab_op)
    grab_op.result = grab_val
    while_body.append(grab_op)

    # 2. Decompose tile_idx → block_row, block_col
    # grid_n = cdiv(N, BN)
    c_bn_val = mir.MConstant(value=BN, dtype="u32")
    c_bn_mv = mir.MValue("c_bn_persistent", U32, c_bn_val)
    c_bn_val.result = c_bn_mv
    while_body.append(c_bn_val)

    c_bn_m1 = mir.MConstant(value=BN - 1, dtype="u32")
    c_bn_m1_mv = mir.MValue("c_bn_m1", U32, c_bn_m1)
    c_bn_m1.result = c_bn_m1_mv
    while_body.append(c_bn_m1)

    n_plus = mir.MBinOp(op="add", lhs=N_val, rhs=c_bn_m1_mv)
    n_plus_mv = mir.MValue("n_plus_bnm1", U32, n_plus)
    n_plus.result = n_plus_mv
    while_body.append(n_plus)

    grid_n = mir.MBinOp(op="div", lhs=n_plus_mv, rhs=c_bn_mv)
    grid_n_mv = mir.MValue("grid_n", U32, grid_n)
    grid_n.result = grid_n_mv
    while_body.append(grid_n)

    # tile_m = tile_idx / grid_n, tile_n = tile_idx % grid_n
    tile_m = mir.MBinOp(op="div", lhs=grab_val, rhs=grid_n_mv)
    tile_m_mv = mir.MValue("tile_m", U32, tile_m)
    tile_m.result = tile_m_mv
    while_body.append(tile_m)

    tile_n = mir.MBinOp(op="mod", lhs=grab_val, rhs=grid_n_mv)
    tile_n_mv = mir.MValue("tile_n", U32, tile_n)
    tile_n.result = tile_n_mv
    while_body.append(tile_n)

    # block_row = tile_m * BM, block_col = tile_n * BN
    c_bm_val = mir.MConstant(value=BM, dtype="u32")
    c_bm_mv = mir.MValue("c_bm_persistent", U32, c_bm_val)
    c_bm_val.result = c_bm_mv
    while_body.append(c_bm_val)

    block_row = mir.MBinOp(op="mul", lhs=tile_m_mv, rhs=c_bm_mv)
    block_row_mv = mir.MValue("block_row", U32, block_row)
    block_row.result = block_row_mv
    while_body.append(block_row)

    block_col = mir.MBinOp(op="mul", lhs=tile_n_mv, rhs=c_bn_mv)
    block_col_mv = mir.MValue("block_col", U32, block_col)
    block_col.result = block_col_mv
    while_body.append(block_col)

    # 3. Accumulator init
    while_body.append(
        mir.MSimdgroupAccDecl(
            acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type, acc_type="float"
        )
    )

    # 4. K-loop (same structure as regular GEMM)
    loop_body: list[mir.MOp] = []

    loop_body.append(
        mir.MCooperativeLoad(
            device_ptr=ptr_A,
            tg_array="shared_a",
            row_offset=block_row_mv,
            col_offset=None,
            src_stride=K_val,
            tile_rows=BM,
            tile_cols=BK,
            dst_stride=A_STRIDE,
            tg_size=TG_SIZE,
            linear_tid=linear_tid,
            bounds_check=True,
            row_bound=M_val,
            col_bound=K_val,
            vec_size=1,
            elem_type=msl_type,
            load_layout=a_load_layout,
        )
    )

    loop_body.append(
        mir.MCooperativeLoad(
            device_ptr=ptr_B,
            tg_array="shared_b",
            row_offset=None,
            col_offset=block_col_mv,
            src_stride=N_val,
            tile_rows=BK,
            tile_cols=BN,
            dst_stride=B_STRIDE,
            tg_size=TG_SIZE,
            linear_tid=linear_tid,
            bounds_check=True,
            row_bound=K_val,
            col_bound=N_val,
            vec_size=1,
            elem_type=msl_type,
            load_layout=b_load_layout,
        )
    )

    loop_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    loop_body.append(
        _build_kk_loop(
            NUM_8M, NUM_8N, sg_row, sg_col, "shared_a", "shared_b", A_STRIDE, B_STRIDE, msl_type, BK
        )
    )

    loop_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    k_loop = mir.MForLoop(
        iv_name="kb",
        start=0,
        end=K_val,
        step=BK,
        body=loop_body,
    )
    while_body.append(k_loop)

    # 5. Store accumulators
    for store_op in _build_acc_stores(
        NUM_8M, NUM_8N, ptr_C, block_row_mv, block_col_mv, sg_row, sg_col, N_val, M_val, msl_type
    ):
        while_body.append(store_op)

    # 6. Barrier before next iteration (protect shared memory)
    while_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    # Wrap in while(true)
    mfunc.add_op(mir.MWhileTrue(body=while_body))

    return mfunc
