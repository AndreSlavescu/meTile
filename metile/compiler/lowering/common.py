"""Shared lowering helpers: errors, kernel-shape detection, and layout maths."""

from __future__ import annotations

from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.layout import Layout, _ceil_div, row_major
from metile.ir.types import PtrType, ScalarType, TileType

_MSL_TYPES = {"f32": "float", "f16": "half", "i32": "int", "u32": "uint"}
# Maximum threadgroup memory in bytes (Apple GPU limit)
_MAX_TG_BYTES = 32768


class LoweringError(Exception):
    pass


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


def _detect_dtype(func: tir.Function) -> tuple[str, str]:
    """Detect dtype and MSL type from the first pointer parameter.

    Returns (dtype, msl_type) e.g. ("f32", "float").
    Raises LoweringError if no pointer parameters exist.
    """
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            return dtype, _MSL_TYPES.get(dtype, "float")
    raise LoweringError("No pointer parameters found — cannot detect dtype")


def _extract_gemm_params(
    func: tir.Function, param_values: dict[str, mir.MValue]
) -> tuple[mir.MValue, mir.MValue, mir.MValue, mir.MValue, mir.MValue, mir.MValue]:
    """Extract A, B, C pointers and M, N, K scalars from function params.

    Tries named lookup first (M, N, K), then falls back to positional.
    Returns (ptr_A, ptr_B, ptr_C, M_val, N_val, K_val).
    """
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

    return ptr_A, ptr_B, ptr_C, M_val, N_val, K_val


def _lower_params(func: tir.Function, mfunc: mir.MFunction) -> dict[str, mir.MValue]:
    """Lower Tile IR params to Metal IR params. Returns param value map."""
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
    return param_values


def _build_kk_loop(
    NUM_8M: int,
    NUM_8N: int,
    sg_row: mir.MValue,
    sg_col: mir.MValue,
    src_a: str,
    src_b: str,
    A_STRIDE: int,
    B_STRIDE: int,
    msl_type: str,
    BK: int,
) -> mir.MForLoop:
    """Build the MMA inner loop (kk) with simdgroup loads and MMA ops.

    Constructs the standard pattern: for each mi, load A tile, then for each ni
    load B tile and issue MMA. Returns an MForLoop marked for unrolling.
    """
    kk_body = []
    for mi in range(NUM_8M):
        kk_body.append(
            mir.MSimdgroupLoad(
                tile_name="a_tile",
                tile_idx=mi,
                src_array=src_a,
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
                    src_array=src_b,
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
    return kk_loop


def _build_acc_stores(
    NUM_8M: int,
    NUM_8N: int,
    ptr_C: mir.MValue,
    block_row: mir.MValue,
    block_col: mir.MValue,
    sg_row: mir.MValue,
    sg_col: mir.MValue,
    N_val: mir.MValue,
    M_val: mir.MValue,
    out_type: str,
) -> list[mir.MSimdgroupStore]:
    """Build accumulator store ops for all (mi, ni) tiles."""
    stores = []
    for mi in range(NUM_8M):
        for ni in range(NUM_8N):
            stores.append(
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
    return stores


def _check_tg_memory(allocs: dict[str, int], elem_type: str, label: str):
    """Validate that threadgroup memory allocations fit within hardware limits.

    allocs: mapping of alloc_name -> num_elements
    elem_type: MSL element type ("float", "half", etc.)
    label: description for error message (e.g. "GEMM")
    """
    elem_sizes = {"float": 4, "half": 2, "int": 4, "uint": 4}
    elem_sz = elem_sizes.get(elem_type, 4)
    total_bytes = sum(sz * elem_sz for sz in allocs.values())
    if total_bytes > _MAX_TG_BYTES:
        raise LoweringError(
            f"{label} requires {total_bytes} bytes threadgroup memory "
            f"(limit {_MAX_TG_BYTES}). Reduce tile sizes."
        )


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


def _detect_epilogue(ops: list) -> list[tuple]:
    """Detect element-wise epilogue ops between GEMM dot loop and tile store.

    Traces the chain of element-wise ops applied to the accumulator after
    the GEMM loop and before the tile store. Handles arbitrary compositions
    of unary, binary-with-constant, and binary-with-original-accumulator ops.

    Returns a list of epilogue tuples:
      - ("relu",)                         — max(val, 0)
      - ("unary", fn_name)                — fn(val)
      - ("scale",)                        — val *= _scale (non-constant scalar)
      - ("binop", op, "lhs"|"rhs", float) — binary op with a constant
            "lhs"/"rhs" indicates which side the CONSTANT is on
      - ("binop_orig", op, "lhs"|"rhs")   — binary op referencing original acc
            "lhs"/"rhs" indicates which side the ORIGINAL acc is on
      - ("save_orig",)                    — prepended when binop_orig is used
    """
    for_idx = store_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, tir.ForRange) and _has_gemm_ops(op.body):
            for_idx = i
        if isinstance(op, tir.TileStore):
            store_idx = i

    if for_idx is None or store_idx is None or store_idx <= for_idx + 1:
        return []

    # Find the accumulator value name (last Dot result in the loop body)
    acc_name = _find_dot_result_name(ops[for_idx].body)
    if acc_name is None:
        return []

    epilogue = []
    chain_name = acc_name
    needs_orig = False

    for op in ops[for_idx + 1 : store_idx]:
        if not hasattr(op, "result") or op.result is None:
            continue
        rt = op.result.type
        if not isinstance(rt, TileType):
            continue

        if isinstance(op, tir.Select):
            cond_op = op.condition.defining_op
            if cond_op and isinstance(cond_op, tir.Compare) and cond_op.predicate == "gt":
                epilogue.append(("relu",))
            else:
                # Non-gt Select patterns (e.g. clamp, abs-via-select) are not
                # currently fusible — bail out of epilogue detection.
                return []
            chain_name = op.result.name

        elif isinstance(op, tir.Unary):
            epilogue.append(("unary", op.op))
            chain_name = op.result.name

        elif isinstance(op, tir.BinOp):
            lhs_tile = isinstance(op.lhs.type, TileType)
            rhs_tile = isinstance(op.rhs.type, TileType)

            if lhs_tile and not rhs_tile:
                # chain OP scalar_const
                const_val = _extract_constant(op.rhs)
                if const_val is not None:
                    epilogue.append(("binop", op.op, "rhs", const_val))
                elif op.op == "mul":
                    epilogue.append(("scale",))
                else:
                    return []  # non-constant scalar for non-mul op, can't fuse
            elif rhs_tile and not lhs_tile:
                # scalar_const OP chain
                const_val = _extract_constant(op.lhs)
                if const_val is not None:
                    epilogue.append(("binop", op.op, "lhs", const_val))
                else:
                    return []  # non-constant scalar on lhs, can't fuse
            elif lhs_tile and rhs_tile:
                # Both TileType: one must be original acc, other is chain
                lhs_is_orig = op.lhs.name == acc_name and op.lhs.name != chain_name
                rhs_is_orig = op.rhs.name == acc_name and op.rhs.name != chain_name
                if lhs_is_orig:
                    epilogue.append(("binop_orig", op.op, "lhs"))
                    needs_orig = True
                elif rhs_is_orig:
                    epilogue.append(("binop_orig", op.op, "rhs"))
                    needs_orig = True
                else:
                    return []  # can't fuse: two non-acc tile operands
            else:
                return []
            chain_name = op.result.name

    if needs_orig:
        epilogue.insert(0, ("save_orig",))

    return epilogue


def _find_dot_result_name(body_ops: list) -> str | None:
    """Find the name of the last Dot result in a loop body."""
    name = None
    for op in body_ops:
        if isinstance(op, tir.Dot) and op.result:
            name = op.result.name
    return name


def _extract_constant(val) -> float | None:
    """Extract a numeric literal from a Value, or None."""
    if val.defining_op and isinstance(val.defining_op, tir.Constant):
        return float(val.defining_op.value)
    return None
