from __future__ import annotations

from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.layout import Layout, row_major
from metile.ir.types import I32, U32, PtrType, ScalarType, TileType


class LoweringError(Exception):
    pass


def lower(func: tir.Function) -> mir.MFunction:
    """Lower a Tile IR function to Metal IR."""
    if _is_persistent_gemm(func):
        return _lower_persistent_gemm(func)
    if _is_specialized_gemm(func):
        return _lower_specialized_gemm(func)
    if _is_gemm(func):
        from metile.runtime.metal_device import MetalDevice

        if MetalDevice.get().supports_tensor_ops:
            return _lower_tensor_ops_gemm(func)
        return _lower_gemm(func)
    ctx = _ElementwiseLoweringContext(func)
    return ctx.lower()


def _is_gemm(func: tir.Function) -> bool:
    """Check if the function contains GEMM tile ops."""
    return _has_gemm_ops(func.ops)


def _is_persistent_gemm(func: tir.Function) -> bool:
    """Check if the function contains a PersistentRange wrapping GEMM ops."""
    return any(isinstance(op, tir.PersistentRange) and _has_gemm_ops(op.body) for op in func.ops)


def _is_specialized_gemm(func: tir.Function) -> bool:
    """Check if this GEMM has explicit simdgroup_role blocks wrapping tile_load/dot."""
    for op in func.ops:
        if isinstance(op, tir.ForRange):
            has_role = any(isinstance(b, tir.SimdgroupRole) for b in op.body)
            if has_role and _has_gemm_ops(op.body):
                return True
    return False


def _has_gemm_ops(ops: list) -> bool:
    for op in ops:
        if isinstance(op, (tir.Dot, tir.TileLoad, tir.TileStore)):
            return True
        if isinstance(op, tir.ForRange) and _has_gemm_ops(op.body):
            return True
        if isinstance(op, tir.PersistentRange) and _has_gemm_ops(op.body):
            return True
        if isinstance(op, tir.SimdgroupRole) and _has_gemm_ops(op.body):
            return True
    return False


_MSL_TYPES = {"f32": "float", "f16": "half", "i32": "int", "u32": "uint"}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_simdgroup_layout(BM: int, BN: int, NUM_SG: int) -> mir.SimdgroupLayout:
    """Derive simdgroup tiling from layout algebra.

    The MMA accumulator grid is (BM/8) x (BN/8) tiles of 8x8 simdgroup_matrix.
    We partition this grid across NUM_SG simdgroups using logical_divide.
    The factorization (sg_rows x sg_cols) is chosen to keep acc_per_sg <= 16.
    """
    mma_m, mma_n = BM // 8, BN // 8

    # Layout of MMA tiles in the accumulator grid
    acc_grid = row_major(mma_m, mma_n)

    # Find best factorization of NUM_SG into (sg_rows, sg_cols)
    # such that the per-SG subtile dimensions are 8-aligned and acc_per_sg <= 16.
    # Prefer balanced factorizations (sg_rows ≈ sg_cols) for square-ish subtiles.
    candidates = []
    for sg_rows in range(1, NUM_SG + 1):
        if NUM_SG % sg_rows != 0:
            continue
        sg_cols = NUM_SG // sg_rows
        if mma_m % sg_rows != 0 or mma_n % sg_cols != 0:
            continue
        per_sg_m = mma_m // sg_rows
        per_sg_n = mma_n // sg_cols
        acc_count = per_sg_m * per_sg_n
        if acc_count > 16:
            continue

        # Use logical_divide to verify the partition is clean
        sg_tiler = Layout((sg_rows, sg_cols))
        divided = acc_grid.logical_divide(sg_tiler)
        if divided.size == acc_grid.size:
            # Score: prefer balanced (minimize |sg_rows - sg_cols|)
            balance = abs(sg_rows - sg_cols)
            candidates.append((balance, sg_rows, sg_cols, per_sg_m * 8, per_sg_n * 8))

    if candidates:
        candidates.sort()
        _, sg_rows, sg_cols, sg_m, sg_n = candidates[0]
        best = (sg_rows, sg_cols, sg_m, sg_n)
    else:
        best = None

    if best is None:
        # Fallback to legacy heuristic
        sg_rows = {4: 2, 8: 4, 16: 4}.get(NUM_SG, 2)
        sg_cols = NUM_SG // sg_rows
        best = (sg_rows, sg_cols, BM // sg_rows, BN // sg_cols)

    sg_rows, sg_cols, sg_m, sg_n = best
    return mir.SimdgroupLayout(
        num_sg=NUM_SG,
        sg_rows=sg_rows,
        sg_cols=sg_cols,
        sg_m=sg_m,
        sg_n=sg_n,
    )


def _compute_coop_load_layout(
    tile_rows: int, tile_cols: int, num_threads: int
) -> mir.CooperativeLoadLayout:
    """Derive thread-to-element mapping for cooperative tile loads.

    Uses logical_divide to partition a row-major tile across threads.
    Each thread handles ceil(rows * cols / num_threads) elements.
    """
    total = tile_rows * tile_cols
    elems_per_thread = _ceil_div(total, num_threads)

    tile = mir.TileLayout(
        rows=tile_rows,
        cols=tile_cols,
        smem_stride=tile_cols,
    )
    return mir.CooperativeLoadLayout(
        tile=tile,
        num_threads=num_threads,
        elems_per_thread=elems_per_thread,
    )


def _select_num_sg(BM: int, BN: int) -> int:
    """Auto-select number of simdgroups based on tile sizes.

    Targets 8-16 accumulators per simdgroup for good register occupancy.
    """
    # Try NUM_SG=4 first (simpler, fewer threads)
    for num_sg in [4, 8]:
        sg_rows = {4: 2, 8: 4}.get(num_sg, 2)
        sg_cols = num_sg // sg_rows
        sg_m = BM // sg_rows
        sg_n = BN // sg_cols
        if sg_m % 8 != 0 or sg_n % 8 != 0:
            continue
        acc_per_sg = (sg_m // 8) * (sg_n // 8)
        if acc_per_sg <= 16:
            return num_sg
    return 4


def _lower_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a GEMM Tile IR function to structured Metal IR.

    Produces Metal IR with MCooperativeLoad, MSimdgroupLoad/MMA, etc.
    These ops are then progressively optimized by passes.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="gemm")

    # Detect dtype from first pointer param
    dtype = "f32"
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            break
    msl_type = _MSL_TYPES.get(dtype, "float")

    # Lower params
    param_values: dict[str, mir.MValue] = {}
    for p in func.params:
        if isinstance(p.type, PtrType):
            mp = mir.MParam(name=p.name, type=p.type, is_output=p.is_output, is_scalar=False)
        elif isinstance(p.type, ScalarType):
            mp = mir.MParam(name=p.name, type=p.type, is_scalar=True)
        else:
            raise LoweringError(f"Unsupported param type: {p.type}")
        mfunc.params.append(mp)
        param_values[p.name] = mir.MValue(p.name, p.type)

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
        mir.MSimdgroupAccDecl(acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type)
    )

    # --- Find the K-param, A/B/C pointers from the traced IR ---
    # Walk the Tile IR to find ForRange and extract pointers + stride refs
    ptr_A = ptr_B = ptr_C = M_val = N_val = K_val = None
    for p in func.params:
        pv = param_values[p.name]
        if isinstance(p.type, PtrType) and ptr_A is None:
            ptr_A = pv
        elif isinstance(p.type, PtrType) and ptr_B is None:
            ptr_B = pv
        elif isinstance(p.type, PtrType) and ptr_C is None:
            ptr_C = pv
        elif isinstance(p.type, ScalarType) and p.name == "M":
            M_val = pv
        elif isinstance(p.type, ScalarType) and p.name == "N":
            N_val = pv
        elif isinstance(p.type, ScalarType) and p.name == "K":
            K_val = pv

    if K_val is None or M_val is None or N_val is None:
        # Try positional: params after the 3 pointers are M, N, K
        scalar_params = [p for p in func.params if isinstance(p.type, ScalarType)]
        if len(scalar_params) >= 3:
            M_val = param_values[scalar_params[0].name]
            N_val = param_values[scalar_params[1].name]
            K_val = param_values[scalar_params[2].name]
        else:
            raise LoweringError("Cannot determine M, N, K parameters")

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

    # MMA inner loop: explicit kk loop with individual load/MMA ops
    kk_body = []
    for mi in range(NUM_8M):
        kk_body.append(
            mir.MSimdgroupLoad(
                tile_name="a_tile",
                tile_idx=mi,
                src_array="shared_a",
                sg_offset=sg_row,
                tile_offset=mi * 8,
                kk_var="kk",
                stride=A_STRIDE,
                is_b=False,
                in_type=msl_type,
            )
        )
        for ni in range(NUM_8N):
            kk_body.append(
                mir.MSimdgroupLoad(
                    tile_name="b_tile",
                    tile_idx=ni,
                    src_array="shared_b",
                    sg_offset=sg_col,
                    tile_offset=ni * 8,
                    kk_var="kk",
                    stride=B_STRIDE,
                    is_b=True,
                    in_type=msl_type,
                )
            )
            kk_body.append(
                mir.MSimdgroupMMA(
                    acc_name="acc",
                    a_tile="a_tile",
                    b_tile="b_tile",
                    mi=mi,
                    ni=ni,
                )
            )

    kk_loop = mir.MForLoop(iv_name="kk", start=0, end=BK, step=8, body=kk_body)
    kk_loop._unroll = True  # mark for #pragma clang loop unroll(full)
    loop_body.append(kk_loop)

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
    out_type = msl_type
    for mi in range(NUM_8M):
        for ni in range(NUM_8N):
            mfunc.add_op(
                mir.MSimdgroupStore(
                    acc_name="acc",
                    mi=mi,
                    ni=ni,
                    device_ptr=ptr_C,
                    block_row=block_row,
                    block_col=block_col,
                    sg_row=sg_row,
                    sg_col=sg_col,
                    mi_offset=mi * 8,
                    ni_offset=ni * 8,
                    stride=N_val,
                    m_bound=M_val,
                    n_bound=N_val,
                    out_type=out_type,
                    acc_type="float",
                )
            )

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

    # Detect dtype
    dtype = "f32"
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            break
    msl_type = _MSL_TYPES.get(dtype, "float")

    # Lower params
    param_values: dict[str, mir.MValue] = {}
    for p in func.params:
        if isinstance(p.type, PtrType):
            mp = mir.MParam(name=p.name, type=p.type, is_output=p.is_output, is_scalar=False)
        elif isinstance(p.type, ScalarType):
            mp = mir.MParam(name=p.name, type=p.type, is_scalar=True)
        else:
            raise LoweringError(f"Unsupported param type: {p.type}")
        mfunc.params.append(mp)
        param_values[p.name] = mir.MValue(p.name, p.type)

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

    # Inline padding to avoid bank conflicts (same logic as passes._optimal_pad)
    from math import gcd as _gcd

    def _opt_pad(stride):
        for p in range(1, 5):
            g = _gcd(stride + p, 32)
            if g & (g - 1) != 0 or g == 1:
                return p
        return 2

    A_PAD = _opt_pad(BK)
    B_PAD = _opt_pad(BN)
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
        mir.MSimdgroupAccDecl(acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type)
    )

    # --- Extract A, B, C, M, N, K ---
    ptr_A = ptr_B = ptr_C = M_val = N_val = K_val = None
    for p in func.params:
        pv = param_values[p.name]
        if isinstance(p.type, PtrType) and ptr_A is None:
            ptr_A = pv
        elif isinstance(p.type, PtrType) and ptr_B is None:
            ptr_B = pv
        elif isinstance(p.type, PtrType) and ptr_C is None:
            ptr_C = pv
        elif isinstance(p.type, ScalarType) and p.name == "M":
            M_val = pv
        elif isinstance(p.type, ScalarType) and p.name == "N":
            N_val = pv
        elif isinstance(p.type, ScalarType) and p.name == "K":
            K_val = pv

    if K_val is None or M_val is None or N_val is None:
        scalar_params = [p for p in func.params if isinstance(p.type, ScalarType)]
        if len(scalar_params) >= 3:
            M_val = param_values[scalar_params[0].name]
            N_val = param_values[scalar_params[1].name]
            K_val = param_values[scalar_params[2].name]
        else:
            raise LoweringError("Cannot determine M, N, K parameters")

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
    kk_body_main = []
    for mi in range(NUM_8M):
        kk_body_main.append(
            mir.MSimdgroupLoad(
                tile_name="a_tile",
                tile_idx=mi,
                src_array="shared_a",
                sg_offset=sg_row,
                tile_offset=mi * 8,
                kk_var="kk",
                stride=A_STRIDE,
                is_b=False,
                in_type=msl_type,
            )
        )
        for ni in range(NUM_8N):
            kk_body_main.append(
                mir.MSimdgroupLoad(
                    tile_name="b_tile",
                    tile_idx=ni,
                    src_array="shared_b",
                    sg_offset=sg_col,
                    tile_offset=ni * 8,
                    kk_var="kk",
                    stride=B_STRIDE,
                    is_b=True,
                    in_type=msl_type,
                )
            )
            kk_body_main.append(
                mir.MSimdgroupMMA(
                    acc_name="acc",
                    a_tile="a_tile",
                    b_tile="b_tile",
                    mi=mi,
                    ni=ni,
                )
            )

    kk_loop_main = mir.MForLoop(iv_name="kk", start=0, end=BK, step=8, body=kk_body_main)
    kk_loop_main._unroll = True
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
    kk_body_epilogue = []
    for mi in range(NUM_8M):
        kk_body_epilogue.append(
            mir.MSimdgroupLoad(
                tile_name="a_tile",
                tile_idx=mi,
                src_array="sa_curr",
                sg_offset=sg_row,
                tile_offset=mi * 8,
                kk_var="kk",
                stride=A_STRIDE,
                is_b=False,
                in_type=msl_type,
            )
        )
        for ni in range(NUM_8N):
            kk_body_epilogue.append(
                mir.MSimdgroupLoad(
                    tile_name="b_tile",
                    tile_idx=ni,
                    src_array="sb_curr",
                    sg_offset=sg_col,
                    tile_offset=ni * 8,
                    kk_var="kk",
                    stride=B_STRIDE,
                    is_b=True,
                    in_type=msl_type,
                )
            )
            kk_body_epilogue.append(
                mir.MSimdgroupMMA(
                    acc_name="acc",
                    a_tile="a_tile",
                    b_tile="b_tile",
                    mi=mi,
                    ni=ni,
                )
            )

    kk_loop_epilogue = mir.MForLoop(iv_name="kk", start=0, end=BK, step=8, body=kk_body_epilogue)
    kk_loop_epilogue._unroll = True
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
    store_ops = []
    for mi in range(NUM_8M):
        for ni in range(NUM_8N):
            store_ops.append(
                mir.MSimdgroupStore(
                    acc_name="acc",
                    mi=mi,
                    ni=ni,
                    device_ptr=ptr_C,
                    block_row=block_row,
                    block_col=block_col,
                    sg_row=sg_row,
                    sg_col=sg_col,
                    mi_offset=mi * 8,
                    ni_offset=ni * 8,
                    stride=N_val,
                    m_bound=M_val,
                    n_bound=N_val,
                    out_type=msl_type,
                    acc_type="float",
                )
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


def _detect_epilogue(ops: list) -> list[tuple]:
    """Detect element-wise epilogue ops between GEMM dot loop and tile store.

    Finds ops that produce TileType results (operating on the accumulator)
    between the ForRange containing Dot and the TileStore. Distinguishes
    these from offset computations (which produce ScalarType/I32).

    Supported patterns:
      - Select with Compare(gt, acc, 0) → ReLU: ("relu",)
      - Unary(fn, acc) → element-wise math: ("unary", fn_name)
      - BinOp(mul, acc, scalar) → scale: ("scale",)
    """
    for_idx = store_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, tir.ForRange) and _has_gemm_ops(op.body):
            for_idx = i
        if isinstance(op, tir.TileStore):
            store_idx = i

    if for_idx is None or store_idx is None or store_idx <= for_idx + 1:
        return []

    epilogue = []
    for op in ops[for_idx + 1 : store_idx]:
        if not hasattr(op, "result") or op.result is None:
            continue
        rt = op.result.type
        if not isinstance(rt, TileType):
            continue
        # This op produces a TileType result → epilogue op
        if isinstance(op, tir.Select):
            # ReLU: where(acc > 0, acc, 0)
            cond_op = op.condition.defining_op
            if cond_op and isinstance(cond_op, tir.Compare) and cond_op.predicate == "gt":
                epilogue.append(("relu",))
            else:
                # Generic clamp/select — treat as relu-like for now
                epilogue.append(("relu",))
        elif isinstance(op, tir.Unary):
            epilogue.append(("unary", op.op))
        elif isinstance(op, tir.BinOp) and op.op == "mul":
            epilogue.append(("scale",))

    return epilogue


def _lower_tensor_ops_gemm(func: tir.Function) -> mir.MFunction:
    """Lower a GEMM to Metal 4 tensor_ops matmul2d.

    Uses preemptive execution: each simdgroup independently runs matmul2d
    on its own subtile. Manual K-tiling with cooperative_tensor accumulation
    in registers. Static template slice<> for optimal codegen.

    Auto-selected when device supports Metal 4 tensor_ops.
    """
    mfunc = mir.MFunction(name=f"mtile_{func.name}", kernel_type="tensor_ops_gemm")

    # Detect dtype from first pointer param
    dtype = "f32"
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            break
    msl_type = _MSL_TYPES.get(dtype, "float")

    # Lower params
    param_values: dict[str, mir.MValue] = {}
    for p in func.params:
        if isinstance(p.type, PtrType):
            mp = mir.MParam(name=p.name, type=p.type, is_output=p.is_output, is_scalar=False)
        elif isinstance(p.type, ScalarType):
            mp = mir.MParam(name=p.name, type=p.type, is_scalar=True)
        else:
            raise LoweringError(f"Unsupported param type: {p.type}")
        mfunc.params.append(mp)
        param_values[p.name] = mir.MValue(p.name, p.type)

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
        swizzle = constexprs.get("SWIZZLE", "morton")
    k_unroll = constexprs.get("K_UNROLL", 1)
    num_stages = constexprs.get("num_stages", 1)

    NUM_SG = WM * WN
    mfunc.threadgroup_size = (NUM_SG * 32, 1, 1)

    # Find A, B, C pointers and M, N, K scalars
    ptr_A = ptr_B = ptr_C = M_val = N_val = K_val = None
    for p in func.params:
        pv = param_values[p.name]
        if isinstance(p.type, PtrType):
            if ptr_A is None:
                ptr_A = pv
            elif ptr_B is None:
                ptr_B = pv
            elif ptr_C is None:
                ptr_C = pv
        elif isinstance(p.type, ScalarType):
            if p.name == "M":
                M_val = pv
            elif p.name == "N":
                N_val = pv
            elif p.name == "K":
                K_val = pv

    if K_val is None or M_val is None or N_val is None:
        scalar_params = [p for p in func.params if isinstance(p.type, ScalarType)]
        if len(scalar_params) >= 3:
            M_val = param_values[scalar_params[0].name]
            N_val = param_values[scalar_params[1].name]
            K_val = param_values[scalar_params[2].name]
        else:
            raise LoweringError("Cannot determine M, N, K parameters")

    # Detect epilogue ops
    epilogue = _detect_epilogue(func.ops)

    # Derived constants
    SM = BM // WM
    SN = BN // WN
    acc_type = msl_type if msl_type == "half" else "float"
    out_type = msl_type

    # Use separated loads when descriptor dimensions allow cooperative_tensor inputs
    use_separated = SM <= 32 and SN <= 32
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
        if num_stages >= 2:
            # Pipelined (double-buffered) separated K-loop
            # This is complex enough that we keep it as a single MForLoop
            # with MCoopTensorLoad + MMatmul2dRun inside
            k_body = []
            k_body.append(
                mir.MCoopTensorLoad(
                    ct_name="ct_a",
                    tensor_name="tA",
                    slice_d0=bk_inner,
                    slice_d1=SM,
                    offset_0="k",
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
                    offset_1="k",
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
            k_loop = mir.MForLoop(iv_name="k", start=0, end=K_val, step=bk_inner, body=k_body)
            if num_stages >= 2:
                k_loop._num_stages = num_stages
            mfunc.add_op(k_loop)
        else:
            # Simple separated K-loop
            # Two-level loop when BK > bk_inner: outer steps by BK (with barrier),
            # inner steps by bk_inner (load+run). Barrier only at outer boundary
            # to avoid restricting the Metal compiler's optimization window.
            if bk_inner < BK:
                # Inner loop body: load + run (no barrier)
                inner_body = []
                inner_body.append(
                    mir.MCoopTensorLoad(
                        ct_name="ct_a",
                        tensor_name="tA",
                        slice_d0=bk_inner,
                        slice_d1=SM,
                        offset_0="k",
                        offset_1="tile_row",
                    )
                )
                inner_body.append(
                    mir.MCoopTensorLoad(
                        ct_name="ct_b",
                        tensor_name="tB",
                        slice_d0=SN,
                        slice_d1=bk_inner,
                        offset_0="tile_col",
                        offset_1="k",
                    )
                )
                inner_body.append(
                    mir.MMatmul2dRun(
                        ct_a="ct_a",
                        ct_b="ct_b",
                        ct_out="cT",
                        use_tensor_view=False,
                    )
                )
                # Inner loop iterates bk_inner steps within each BK block
                inner_loop = mir.MForLoop(
                    iv_name="k1",
                    start=0,
                    end=BK,
                    step=bk_inner,
                    body=inner_body,
                )
                # Mark inner loop so emitter uses "k0 + k1" as the k variable
                inner_loop._inner_k = True

                # Outer loop with barrier at each BK step
                outer_body = [
                    mir.MBarrier(kind="threadgroup", flags="mem_none"),
                    inner_loop,
                ]
                outer_loop = mir.MForLoop(
                    iv_name="k0",
                    start=0,
                    end=K_val,
                    step=BK,
                    body=outer_body,
                )
                mfunc.add_op(outer_loop)
            else:
                k_body = []
                k_body.append(mir.MBarrier(kind="threadgroup", flags="mem_none"))
                k_body.append(
                    mir.MCoopTensorLoad(
                        ct_name="ct_a",
                        tensor_name="tA",
                        slice_d0=bk_inner,
                        slice_d1=SM,
                        offset_0="k",
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
                        offset_1="k",
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
                k_loop = mir.MForLoop(iv_name="k", start=0, end=K_val, step=bk_inner, body=k_body)
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
                    a_offset_1="tile_row" if not cooperative else "row_expr",
                    b_offset_0="tile_col" if not cooperative else "col_expr",
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

    # Detect dtype from first pointer param
    dtype = "f32"
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            break
    msl_type = _MSL_TYPES.get(dtype, "float")

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
        mir.MSimdgroupAccDecl(acc_name="acc", num_8m=NUM_8M, num_8n=NUM_8N, in_type=msl_type)
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

    kk_body_persistent = []
    for mi in range(NUM_8M):
        kk_body_persistent.append(
            mir.MSimdgroupLoad(
                tile_name="a_tile",
                tile_idx=mi,
                src_array="shared_a",
                sg_offset=sg_row,
                tile_offset=mi * 8,
                kk_var="kk",
                stride=A_STRIDE,
                is_b=False,
                in_type=msl_type,
            )
        )
        for ni in range(NUM_8N):
            kk_body_persistent.append(
                mir.MSimdgroupLoad(
                    tile_name="b_tile",
                    tile_idx=ni,
                    src_array="shared_b",
                    sg_offset=sg_col,
                    tile_offset=ni * 8,
                    kk_var="kk",
                    stride=B_STRIDE,
                    is_b=True,
                    in_type=msl_type,
                )
            )
            kk_body_persistent.append(
                mir.MSimdgroupMMA(
                    acc_name="acc",
                    a_tile="a_tile",
                    b_tile="b_tile",
                    mi=mi,
                    ni=ni,
                )
            )

    kk_loop_persistent = mir.MForLoop(
        iv_name="kk", start=0, end=BK, step=8, body=kk_body_persistent
    )
    kk_loop_persistent._unroll = True
    loop_body.append(kk_loop_persistent)

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
    out_type = msl_type
    for mi in range(NUM_8M):
        for ni in range(NUM_8N):
            while_body.append(
                mir.MSimdgroupStore(
                    acc_name="acc",
                    mi=mi,
                    ni=ni,
                    device_ptr=ptr_C,
                    block_row=block_row_mv,
                    block_col=block_col_mv,
                    sg_row=sg_row,
                    sg_col=sg_col,
                    mi_offset=mi * 8,
                    ni_offset=ni * 8,
                    stride=N_val,
                    m_bound=M_val,
                    n_bound=N_val,
                    out_type=out_type,
                    acc_type="float",
                )
            )

    # 6. Barrier before next iteration (protect shared memory)
    while_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))

    # Wrap in while(true)
    mfunc.add_op(mir.MWhileTrue(body=while_body))

    return mfunc


class _ElementwiseLoweringContext:
    def __init__(self, func: tir.Function):
        self.func = func
        self.mfunc = mir.MFunction(name=f"mtile_{func.name}")
        # Map from Tile IR Value -> Metal IR MValue
        self.value_map: dict[str, mir.MValue] = {}
        # Track which values are tile-expanded (use thread_position_in_grid)
        self.tid_value: mir.MValue | None = None
        # Track the block size for grid computation
        self.block_size: int | None = None
        # Track scalar param values by name -> MValue
        self.param_values: dict[str, mir.MValue] = {}
        # Row-parallel mode values
        self.tgp_id: mir.MValue | None = None
        self.lid_value: mir.MValue | None = None
        self.sgid: mir.MValue | None = None
        self.slid: mir.MValue | None = None
        # Lowering mode: "elementwise", "row_wise", "row_parallel"
        self._mode: str = "elementwise"
        self._reduce_counter: int = 0
        self._acc_counter: int = 0
        self._next_sg: int = 0  # tracks next available simdgroup index for role assignment
        # Track shared memory allocations: tile IR value name -> threadgroup array name
        self._shared_allocs: dict[str, str] = {}

    def lower(self) -> mir.MFunction:
        self._lower_params()

        if self._has_reduce() or self._has_shared():
            self._mode = "row_parallel"
            self._setup_row_parallel()
        elif self._has_arange():
            self._mode = "elementwise"
            self._setup_elementwise()
        else:
            self._mode = "row_wise"
            self._setup_row_wise()

        self._lower_ops()
        self._set_grid()
        return self.mfunc

    def _setup_elementwise(self):
        """Standard element-wise: tid = pid * BLOCK + arange."""
        tid_op = mir.ThreadPositionInGrid(axis=0)
        self.tid_value = self.mfunc.add_op(tid_op, "tid")

    def _setup_row_wise(self):
        """Row-wise: one thread per row, no arange."""
        tid_op = mir.ThreadPositionInGrid(axis=0)
        self.tid_value = self.mfunc.add_op(tid_op, "tid")

    def _setup_row_parallel(self):
        """Row-parallel: multiple threads per row with reduction."""
        self.tgp_id = self.mfunc.add_op(mir.ThreadgroupPositionInGrid(axis=0), "tgp_id_x")
        self.lid_value = self.mfunc.add_op(mir.ThreadPositionInThreadgroup(axis=0), "lid")
        self.sgid = self.mfunc.add_op(mir.MSimdgroupId(), "sgid")
        self.slid = self.mfunc.add_op(mir.MThreadInSimdgroup(), "slid")
        self.tid_value = None  # not used in row-parallel

    def _lower_params(self):
        for p in self.func.params:
            if isinstance(p.type, PtrType):
                mp = mir.MParam(
                    name=p.name,
                    type=p.type,
                    is_output=p.is_output,
                    is_scalar=False,
                )
            elif isinstance(p.type, ScalarType):
                mp = mir.MParam(
                    name=p.name,
                    type=p.type,
                    is_scalar=True,
                )
            else:
                raise LoweringError(f"Unsupported param type: {p.type}")
            self.mfunc.params.append(mp)
            # Create a sentinel MValue for param references
            mv = mir.MValue(p.name, p.type)
            self.param_values[p.name] = mv
            self.value_map[p.name] = mv

    def _lower_ops(self):
        # Collect all ops that are inside if-blocks (stores with masks)
        # First pass: identify mask values and the if-block pattern
        mask_value = None

        # Analyze: find the mask and split ops
        for op in self.func.ops:
            if isinstance(op, tir.Store) and op.mask is not None:
                # This store is masked - everything from the mask def onward
                # goes inside an if-block
                mask_value = op.mask
                break

        # Second pass: lower ops, grouping masked stores into if-block
        body_ops = []
        for op in self.func.ops:
            lowered = self._lower_op(op)
            if lowered is not None:
                for m_op in lowered:
                    body_ops.append(m_op)

        # If we have a mask, wrap the relevant ops in an if-block
        if mask_value and mask_value.name in self.value_map:
            mask_mv = self.value_map[mask_value.name]
            # Find the compare op that produces the mask
            # Everything after (and including loads) goes inside the if
            compare_idx = None
            for i, m_op in enumerate(body_ops):
                if hasattr(m_op, "result") and m_op.result and m_op.result is mask_mv:
                    compare_idx = i
                    break

            if compare_idx is not None:
                pre_ops = body_ops[: compare_idx + 1]  # include the compare
                post_ops = body_ops[compare_idx + 1 :]  # loads, compute, stores
                if_block = mir.IfBlock(condition=mask_mv, body=post_ops)
                for m_op in pre_ops:
                    self.mfunc.ops.append(m_op)
                self.mfunc.ops.append(if_block)
            else:
                for m_op in body_ops:
                    self.mfunc.ops.append(m_op)
        else:
            for m_op in body_ops:
                self.mfunc.ops.append(m_op)

    def _lower_op(self, op: tir.Op) -> list[mir.MOp] | None:
        """Lower a single Tile IR op to Metal IR op(s). Returns list or None."""
        if isinstance(op, tir.ProgramId):
            if self._mode == "row_parallel":
                self.value_map[op.result.name] = self.tgp_id
            elif self._mode == "elementwise":
                self.value_map[op.result.name] = None  # placeholder for pid*BLOCK+arange
            else:  # row_wise
                self.value_map[op.result.name] = self.tid_value
            return None

        elif isinstance(op, tir.Constant):
            m_op = mir.MConstant(value=op.value, dtype=op.dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.Arange):
            self.block_size = op.size
            if self._mode == "row_parallel":
                self.value_map[op.result.name] = self.lid_value
            else:
                self.value_map[op.result.name] = self.tid_value
            return None

        elif isinstance(op, tir.Reduce):
            return self._lower_reduce(op)

        elif isinstance(op, tir.BinOp):
            return self._lower_binop(op)

        elif isinstance(op, tir.Compare):
            return self._lower_compare(op)

        elif isinstance(op, tir.Load):
            return self._lower_load(op)

        elif isinstance(op, tir.Store):
            return self._lower_store(op)

        elif isinstance(op, tir.Unary):
            return self._lower_unary(op)

        elif isinstance(op, tir.Select):
            return self._lower_select(op)

        elif isinstance(op, tir.SharedAlloc):
            msl_type = _MSL_TYPES.get(op.dtype, "float")
            alloc_name = op.result.name  # e.g., "shared_5"
            alloc_op = mir.MThreadgroupAlloc(
                alloc_name=alloc_name, elem_type=msl_type, size=op.size
            )
            self._shared_allocs[op.result.name] = alloc_name
            # Create a sentinel MValue for the shared memory pointer
            val = mir.MValue(alloc_name, PtrType(op.dtype, "threadgroup"))
            self.value_map[op.result.name] = val
            return [alloc_op]

        elif isinstance(op, tir.Barrier):
            return [mir.MBarrier(kind="threadgroup", flags="mem_threadgroup")]

        elif isinstance(op, tir.ThreadId):
            # Ensure lid is available
            if self.lid_value is None:
                self.lid_value = self.mfunc.add_op(mir.ThreadPositionInThreadgroup(axis=0), "lid")
            val = self.lid_value
            self.value_map[op.result.name] = val
            return None

        elif isinstance(op, tir.PtrOffset):
            # ptr + offsets: track as (base_ptr, index) tuple
            ptr_val = self.value_map.get(op.ptr.name)
            offset_val = self.value_map.get(op.offsets.name, self.tid_value)

            if isinstance(ptr_val, tuple) and len(ptr_val) == 2:
                # Chained PtrOffset (e.g. X + a + b): combine offsets
                base, old_offset = ptr_val
                if isinstance(old_offset, mir.MValue) and isinstance(offset_val, mir.MValue):
                    ops = []
                    rhs = offset_val
                    if old_offset.type != offset_val.type:
                        cast_op = mir.MCast(value=offset_val, target_dtype=old_offset.type.dtype)
                        cast_v = mir.MValue(
                            f"cast_off_{op.result.name}", cast_op.result_type(), cast_op
                        )
                        cast_op.result = cast_v
                        ops.append(cast_op)
                        rhs = cast_v
                    combined_op = mir.MBinOp(op="add", lhs=old_offset, rhs=rhs)
                    combined_mv = mir.MValue(
                        f"_off_{op.result.name}", combined_op.result_type(), combined_op
                    )
                    combined_op.result = combined_mv
                    ops.append(combined_op)
                    self.value_map[op.result.name] = (base, combined_mv)
                    return ops
                else:
                    self.value_map[op.result.name] = (base, offset_val)
                    return None
            else:
                self.value_map[op.result.name] = (ptr_val, offset_val)
                return None

        elif isinstance(op, tir.ForRange):
            return self._lower_for_range(op)

        elif isinstance(op, tir.SimdShuffleXor):
            val = self._resolve(op.value)
            mask = self._resolve(op.mask)
            dtype = "f32"
            if isinstance(op.value.type, ScalarType):
                dtype = op.value.type.dtype
            m_op = mir.MSimdShuffleXor(value=val, mask=mask, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdBroadcast):
            val = self._resolve(op.value)
            lane = self._resolve(op.lane)
            dtype = "f32"
            if isinstance(op.value.type, ScalarType):
                dtype = op.value.type.dtype
            m_op = mir.MSimdBroadcast(value=val, lane=lane, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdLaneId):
            m_op = mir.MThreadInSimdgroup()
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        elif isinstance(op, tir.SimdgroupRole):
            return self._lower_simdgroup_role(op)

        else:
            raise LoweringError(f"Unsupported Tile IR op: {type(op).__name__}")

    def _lower_binop(self, op: tir.BinOp) -> list[mir.MOp] | None:
        lhs = self.value_map.get(op.lhs.name)
        rhs = self.value_map.get(op.rhs.name)

        # Check if this is the pid * BLOCK + arange pattern
        # pid * BLOCK: lhs is None (program_id placeholder), rhs is BLOCK constant
        if lhs is None and isinstance(op.lhs.defining_op, tir.ProgramId):
            # pid * BLOCK -> we still map to tid when combined with arange
            # For now, store a marker
            self.value_map[op.result.name] = ("pid_times_block",)
            return None

        # pid * BLOCK + arange -> thread_position_in_grid
        if (
            isinstance(lhs, tuple)
            and len(lhs) == 1
            and lhs[0] == "pid_times_block"
            and rhs is self.tid_value
        ):
            self.value_map[op.result.name] = self.tid_value
            return None

        # Regular scalar binary op
        if isinstance(lhs, mir.MValue) and isinstance(rhs, mir.MValue):
            # May need a cast if types differ (e.g., uint vs int)
            ops = []
            if lhs.type != rhs.type:
                cast_op = mir.MCast(value=rhs, target_dtype=lhs.type.dtype)
                cast_v = mir.MValue(f"cast_{op.result.name}", cast_op.result_type(), cast_op)
                cast_op.result = cast_v
                ops.append(cast_op)
                rhs = cast_v

            m_op = mir.MBinOp(op=op.op, lhs=lhs, rhs=rhs)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            ops.append(m_op)
            return ops

        # Tile-level binop where one side is tid and other is a scalar/tile
        if lhs is self.tid_value or rhs is self.tid_value:
            actual_lhs = lhs if isinstance(lhs, mir.MValue) else self.tid_value
            actual_rhs = rhs if isinstance(rhs, mir.MValue) else self.tid_value
            m_op = mir.MBinOp(op=op.op, lhs=actual_lhs, rhs=actual_rhs)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        raise LoweringError(f"Cannot lower BinOp: lhs={lhs}, rhs={rhs}, op={op.op}")

    def _lower_compare(self, op: tir.Compare) -> list[mir.MOp]:
        lhs = self._resolve(op.lhs)
        rhs = self._resolve(op.rhs)
        ops = []

        # Cast if types differ
        if lhs.type != rhs.type:
            cast_op = mir.MCast(value=rhs, target_dtype=lhs.type.dtype)
            cast_v = mir.MValue(f"cast_{op.result.name}", cast_op.result_type(), cast_op)
            cast_op.result = cast_v
            ops.append(cast_op)
            rhs = cast_v

        m_op = mir.MCompare(predicate=op.predicate, lhs=lhs, rhs=rhs)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        ops.append(m_op)
        return ops

    def _is_shared_ptr(self, base: mir.MValue) -> bool:
        """Check if a base MValue corresponds to a shared memory allocation."""
        return isinstance(base, mir.MValue) and base.name in self._shared_allocs

    def _lower_load(self, op: tir.Load) -> list[mir.MOp]:
        ptr_info = self.value_map.get(op.ptr.name)
        # ptr_info could be a tuple (base_ptr, index) from PtrOffset
        if isinstance(ptr_info, tuple) and len(ptr_info) == 2:
            base, index = ptr_info
        else:
            base = self._resolve_ptr(op.ptr)
            index = self._resolve(op.offsets)

        dtype = op.ptr.type.dtype if isinstance(op.ptr.type, PtrType) else "f32"

        # Shared memory load
        if self._is_shared_ptr(base):
            array_name = self._shared_allocs[base.name]
            m_op = mir.MThreadgroupLoad(array_name=array_name, index=index, dtype=dtype)
            mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
            m_op.result = mv
            self.value_map[op.result.name] = mv
            return [m_op]

        m_op = mir.DeviceLoad(ptr=base, index=index, dtype=dtype)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _lower_store(self, op: tir.Store) -> list[mir.MOp]:
        ptr_info = self.value_map.get(op.ptr.name)
        if isinstance(ptr_info, tuple) and len(ptr_info) == 2:
            base, index = ptr_info
        else:
            base = self._resolve_ptr(op.ptr)
            index = self._resolve(op.offsets)

        value = self._resolve(op.value)

        # Shared memory store
        if self._is_shared_ptr(base):
            array_name = self._shared_allocs[base.name]
            m_op = mir.MThreadgroupStore(array_name=array_name, index=index, value=value)
            return [m_op]

        m_op = mir.DeviceStore(ptr=base, index=index, value=value)
        return [m_op]

    def _lower_unary(self, op: tir.Unary) -> list[mir.MOp]:
        operand = self._resolve(op.operand)
        m_op = mir.MUnary(op=op.op, operand=operand)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _lower_select(self, op: tir.Select) -> list[mir.MOp]:
        cond = self._resolve(op.condition)
        true_v = self._resolve(op.true_val)
        false_v = self._resolve(op.false_val)
        m_op = mir.MSelect(condition=cond, true_val=true_v, false_val=false_v)
        mv = mir.MValue(op.result.name, m_op.result_type(), m_op)
        m_op.result = mv
        self.value_map[op.result.name] = mv
        return [m_op]

    def _resolve(self, val: tir.Value) -> mir.MValue:
        """Resolve a Tile IR value to its Metal IR equivalent."""
        mv = self.value_map.get(val.name)
        if isinstance(mv, mir.MValue):
            return mv
        if mv is self.tid_value:
            return self.tid_value
        # Could be a param
        if val.name in self.param_values:
            return self.param_values[val.name]
        raise LoweringError(f"Cannot resolve value: %{val.name}")

    def _resolve_ptr(self, val: tir.Value) -> mir.MValue:
        """Resolve a pointer value."""
        mv = self.value_map.get(val.name)
        if isinstance(mv, mir.MValue):
            return mv
        if val.name in self.param_values:
            return self.param_values[val.name]
        raise LoweringError(f"Cannot resolve pointer: %{val.name}")

    def _lower_simdgroup_role(self, op: tir.SimdgroupRole) -> list[mir.MOp]:
        """Lower a SimdgroupRole to MSimdgroupRoleBlock.

        Ensures sgid is available, computes simdgroup range for this role,
        and lowers the body ops into the role block.
        """
        # Ensure we have sgid available
        result_ops = []
        if self.sgid is None:
            self.sgid = self.mfunc.add_op(mir.MSimdgroupId(), "sgid")

        # Compute SG assignment using cumulative tracking
        tg_size = self.mfunc.threadgroup_size[0]
        total_sgs = tg_size // 32

        if op.num_sgs > 0:
            role_sgs = op.num_sgs
        else:
            role_sgs = total_sgs // op.num_roles

        first_sg = self._next_sg
        self._next_sg += role_sgs

        # Lower body ops
        body_metal_ops = []
        for body_op in op.body:
            lowered = self._lower_op(body_op)
            if lowered is not None:
                body_metal_ops.extend(lowered)

        role_block = mir.MSimdgroupRoleBlock(
            role=op.role,
            num_roles=op.num_roles,
            first_sg=first_sg,
            num_sgs=role_sgs,
            sgid=self.sgid,
            body=body_metal_ops,
        )
        result_ops.append(role_block)
        return result_ops

    def _lower_for_range(self, op: tir.ForRange) -> list[mir.MOp]:
        """Lower a ForRange (tile_range) to an MForLoop in element-wise context.

        Detects accumulation patterns (values defined inside the loop that are
        used after it) and emits MVarDecl/MVarAssign for loop-carried deps.
        In row-parallel mode, also detects masks and wraps body in IfBlock.
        """
        start = self._resolve(op.start)
        end = self._resolve(op.end)

        iv_mv = mir.MValue(op.iv.name, I32)
        self.value_map[op.iv.name] = iv_mv

        # Pre-pass: detect accumulation patterns in the Tile IR body
        acc_infos = self._detect_accumulation(op)

        # Pre-pass: detect mask in body (for row-parallel mode)
        body_mask_name = None
        if self._mode == "row_parallel":
            for body_op in op.body:
                if isinstance(body_op, tir.Load) and getattr(body_op, "mask", None) is not None:
                    body_mask_name = body_op.mask.name
                    break
                if isinstance(body_op, tir.Store) and getattr(body_op, "mask", None) is not None:
                    body_mask_name = body_op.mask.name
                    break

        # Lower body ops
        body_metal_ops = []
        for body_op in op.body:
            lowered = self._lower_op(body_op)
            if lowered is not None:
                body_metal_ops.extend(lowered)

        result_ops = []

        if acc_infos:
            for init_tile_name, init_const_value, final_tile_name in acc_infos:
                init_mv = self.value_map.get(init_tile_name)
                final_mv = self.value_map.get(final_tile_name)

                if isinstance(init_mv, mir.MValue) and isinstance(final_mv, mir.MValue):
                    acc_id = self._acc_counter
                    self._acc_counter += 1
                    var_name = f"_acc_{acc_id}"
                    dtype = init_mv.type.dtype if hasattr(init_mv.type, "dtype") else "f32"

                    # Create a new MConstant for the init value (before the loop)
                    init_const_op = mir.MConstant(value=init_const_value, dtype=dtype)
                    init_const_mv = mir.MValue(f"_acc_init_{acc_id}", ScalarType(dtype))
                    init_const_op.result = init_const_mv
                    init_const_mv.defining_op = init_const_op
                    result_ops.append(init_const_op)

                    # Declare the mutable accumulator variable
                    result_ops.append(
                        mir.MVarDecl(var_name=var_name, init_value=init_const_mv, dtype=dtype)
                    )

                    # Remove the MConstant from body_metal_ops (it's now before the loop)
                    body_metal_ops = [
                        m
                        for m in body_metal_ops
                        if not (isinstance(m, mir.MConstant) and m.result is init_mv)
                    ]

                    # Replace references to init_mv in body with the accumulator var
                    var_mv = mir.MValue(var_name, init_mv.type)
                    self._replace_mvalue_refs(body_metal_ops, init_mv, var_mv)

                    # Assign final value back to accumulator at end of body
                    body_metal_ops.append(mir.MVarAssign(var_name=var_name, value=final_mv))

                    # Map final value to accumulator for post-loop use
                    self.value_map[final_tile_name] = var_mv

        # Handle masking: wrap post-mask body ops in IfBlock
        if body_mask_name and body_mask_name in self.value_map:
            mask_mv = self.value_map[body_mask_name]
            if isinstance(mask_mv, mir.MValue):
                compare_idx = None
                for i, m_op in enumerate(body_metal_ops):
                    if isinstance(m_op, mir.MCompare) and m_op.result is mask_mv:
                        compare_idx = i
                        break
                if compare_idx is not None:
                    pre_ops = body_metal_ops[: compare_idx + 1]
                    if_body = body_metal_ops[compare_idx + 1 :]
                    body_metal_ops = [*pre_ops, mir.IfBlock(condition=mask_mv, body=if_body)]

        loop = mir.MForLoop(
            iv_name=op.iv.name,
            start=start,
            end=end,
            step=op.step,
            body=body_metal_ops,
        )
        if op.num_stages > 1:
            loop._num_stages = op.num_stages
        result_ops.append(loop)
        return result_ops

    def _detect_accumulation(self, for_range_op: tir.ForRange):
        """Detect accumulation patterns in a ForRange body.

        Looks for Constants that feed into BinOp chains, where the
        final values are used after the ForRange. Returns a list of
        (init_name, init_value, final_name) tuples, or None if empty.
        """
        body = for_range_op.body

        # Map body value names to their defining ops
        body_defs = {}
        for body_op in body:
            if hasattr(body_op, "result") and body_op.result:
                body_defs[body_op.result.name] = body_op

        # Find values used after the ForRange
        post_refs = set()
        found = False
        for parent_op in self.func.ops:
            if parent_op is for_range_op:
                found = True
                continue
            if found:
                self._collect_tile_ir_refs(parent_op, post_refs)

        # Find escaped values (defined in body, used after)
        escaped = set(body_defs.keys()) & post_refs

        results = []
        used_inits = set()
        for esc_name in escaped:
            chain = self._walk_acc_chain(esc_name, body_defs)
            if chain and chain[0] not in used_inits:
                results.append(chain)
                used_inits.add(chain[0])
        return results or None

    def _walk_acc_chain(self, start_name, body_defs):
        """Walk backward through BinOp chain to find the Constant root.

        Returns (init_name, init_value, final_name) or None.
        """
        current_name = start_name
        visited = set()

        while current_name in body_defs and current_name not in visited:
            visited.add(current_name)
            op = body_defs[current_name]

            if isinstance(op, tir.BinOp):
                lhs_name = op.lhs.name
                rhs_name = op.rhs.name

                # Check if either side is a Constant (the accumulation root)
                if lhs_name in body_defs and isinstance(body_defs[lhs_name], tir.Constant):
                    return (lhs_name, body_defs[lhs_name].value, start_name)
                if rhs_name in body_defs and isinstance(body_defs[rhs_name], tir.Constant):
                    return (rhs_name, body_defs[rhs_name].value, start_name)

                # Follow the accumulation chain (the side that's a BinOp)
                if lhs_name in body_defs and isinstance(body_defs[lhs_name], tir.BinOp):
                    current_name = lhs_name
                elif rhs_name in body_defs and isinstance(body_defs[rhs_name], tir.BinOp):
                    current_name = rhs_name
                else:
                    break
            else:
                break
        return None

    def _collect_tile_ir_refs(self, op, refs: set):
        """Collect value names referenced by a Tile IR op."""
        for attr in (
            "lhs",
            "rhs",
            "operand",
            "condition",
            "true_val",
            "false_val",
            "ptr",
            "offsets",
            "value",
        ):
            val = getattr(op, attr, None)
            if isinstance(val, tir.Value):
                refs.add(val.name)
        if isinstance(op, tir.Store) and getattr(op, "mask", None) is not None:
            refs.add(op.mask.name)

    def _replace_mvalue_refs(self, ops: list, old_mv: mir.MValue, new_mv: mir.MValue):
        """Replace all references to old_mv with new_mv in Metal IR ops."""
        for m_op in ops:
            for attr in (
                "lhs",
                "rhs",
                "operand",
                "condition",
                "true_val",
                "false_val",
                "ptr",
                "index",
                "value",
            ):
                if getattr(m_op, attr, None) is old_mv:
                    setattr(m_op, attr, new_mv)

    def _lower_reduce(self, op: tir.Reduce) -> list[mir.MOp]:
        """Lower a Reduce op to MThreadgroupReduce."""
        operand = self._resolve(op.operand)
        assert self.block_size is not None, "Reduce requires a block size (from arange)"
        num_sg = self.block_size // 32
        dtype = operand.type.dtype if hasattr(operand.type, "dtype") else "f32"

        shared_name = f"shared_reduce_{self._reduce_counter}"
        self._reduce_counter += 1

        result_ops = []
        if num_sg > 1:
            result_ops.append(
                mir.MThreadgroupAlloc(
                    alloc_name=shared_name, elem_type=_MSL_TYPES.get(dtype, "float"), size=num_sg
                )
            )

        reduce_op = mir.MThreadgroupReduce(
            reduce_op=op.op,
            operand=operand,
            shared_name=shared_name,
            block_size=self.block_size,
            sgid=self.sgid,
            slid=self.slid,
            dtype=dtype,
        )
        mv = mir.MValue(op.result.name, reduce_op.result_type(), reduce_op)
        reduce_op.result = mv
        self.value_map[op.result.name] = mv
        result_ops.append(reduce_op)
        return result_ops

    def _has_reduce(self) -> bool:
        """Check if the function uses Reduce (row-parallel pattern)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, tir.Reduce):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _has_shared(self) -> bool:
        """Check if the function uses SharedAlloc (needs threadgroup indexing)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, (tir.SharedAlloc, tir.ThreadId)):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _has_arange(self) -> bool:
        """Check if the function uses Arange (standard element-wise pattern)."""

        def _check(ops):
            for op in ops:
                if isinstance(op, tir.Arange):
                    return True
                if isinstance(op, tir.ForRange) and _check(op.body):
                    return True
            return False

        return _check(self.func.ops)

    def _set_grid(self):
        """Set grid and threadgroup dimensions based on the kernel pattern."""
        block = self.block_size or self.func.constexprs.get("BLOCK", 0)
        if block:
            self.mfunc.threadgroup_size = (block, 1, 1)
        else:
            # Row-wise kernel: one thread per program, no arange
            self.mfunc.threadgroup_size = (1, 1, 1)
        # Grid is set at dispatch time based on input size
        self.mfunc.grid = (0, 1, 1)  # 0 = determined at dispatch time
