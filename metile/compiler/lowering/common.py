"""Shared lowering helpers: errors, kernel-shape detection, and layout maths."""

from __future__ import annotations

from dataclasses import dataclass

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
    if _is_gemm(func):
        binding = _analyze_gemm(func)
        dtype = binding.left.ptr.type.dtype
        return dtype, _MSL_TYPES[dtype]
    for p in func.params:
        if isinstance(p.type, PtrType):
            dtype = p.type.dtype
            return dtype, _MSL_TYPES.get(dtype, "float")
    raise LoweringError("No pointer parameters found — cannot detect dtype")


@dataclass(frozen=True)
class _GemmBinding:
    left: tir.TileLoad
    right: tir.TileLoad
    output: tir.TileStore
    dimensions: tuple[tir.Value, tir.Value, tir.Value]
    tiles: tuple[int, int, int]
    uses_descriptors: bool


def _walk_tile_ops(ops):
    for operation in ops:
        yield operation
        body = getattr(operation, "body", None)
        if body is not None:
            yield from _walk_tile_ops(body)


def _value_key(value):
    operation = value.defining_op
    if isinstance(operation, tir.Constant):
        return ("constant", operation.value)
    if isinstance(operation, tir.ProgramId):
        return ("program", operation.axis)
    if isinstance(operation, tir.BinOp):
        operands = (_value_key(operation.lhs), _value_key(operation.rhs))
        if operation.op in {"add", "mul"}:
            operands = tuple(sorted(operands, key=repr))
        return (operation.op, *operands)
    return ("value", value.name)


def _same_value(left, right):
    return _value_key(left) == _value_key(right)


def _is_constant(value, expected):
    return isinstance(value.defining_op, tir.Constant) and value.defining_op.value == expected


def _is_tile_origin(value, axis, block):
    operation = value.defining_op
    if not isinstance(operation, tir.BinOp) or operation.op != "mul":
        return False
    return any(
        isinstance(index.defining_op, tir.ProgramId)
        and index.defining_op.axis == axis
        and _is_constant(scale, block)
        for index, scale in ((operation.lhs, operation.rhs), (operation.rhs, operation.lhs))
    )


def _analyze_gemm(func: tir.Function) -> _GemmBinding:
    operations = list(_walk_tile_ops(func.ops))
    dots = [operation for operation in operations if isinstance(operation, tir.Dot)]
    stores = [operation for operation in operations if isinstance(operation, tir.TileStore)]
    if len(dots) != 1 or len(stores) != 1:
        raise LoweringError("GEMM requires one dot recurrence and one tensor store")
    dot = dots[0]
    left, right = dot.a.defining_op, dot.b.defining_op
    output = stores[0]
    if not isinstance(left, tir.TileLoad) or not isinstance(right, tir.TileLoad):
        raise LoweringError("GEMM dot operands must be direct tensor tile loads")
    parameters = {parameter.name: parameter for parameter in func.params}
    for memory_op in (left, right, output):
        pointer = memory_op.ptr
        if pointer.defining_op is not None or pointer.name not in parameters:
            raise LoweringError("GEMM tensor pointers must reference kernel parameters directly")
        if not isinstance(pointer.type, PtrType) or pointer.type.address_space != "device":
            raise LoweringError("GEMM tensors currently require device memory")
    dtypes = {operation.ptr.type.dtype for operation in (left, right, output)}
    if len(dtypes) != 1 or not dtypes.issubset({"f16", "f32"}):
        raise LoweringError("GEMM currently requires matching f16 or f32 input/output dtypes")
    block_m, block_k = left.tile_shape
    right_k, block_n = right.tile_shape
    if right_k != block_k or output.tile_shape != (block_m, block_n):
        raise LoweringError("GEMM load and store tile shapes are inconsistent")
    if not isinstance(dot.acc.type, TileType) or dot.acc.type.shape != (block_m, block_n):
        raise LoweringError("GEMM accumulator shape must match the output tile")
    if dot.acc.type.dtype != "f32" or not isinstance(dot.acc.defining_op, tir.Zeros):
        raise LoweringError("GEMM currently requires a zero-initialized f32 accumulator")
    tiles = (block_m, block_n, block_k)
    for axis, size in zip(("M", "N", "K"), tiles, strict=True):
        configured = func.constexprs.get(f"BLOCK_{axis}", size)
        if configured != size or size < 8 or size % 8:
            raise LoweringError("GEMM tiles must match BLOCK_* and be positive multiples of 8")
    tensors = [getattr(operation, "tensor", None) for operation in (left, right, output)]
    uses_descriptors = any(tensor is not None for tensor in tensors)
    if uses_descriptors:
        if any(tensor is None for tensor in tensors):
            raise LoweringError("GEMM requires descriptors on all input and output tensors")
        for tensor in tensors:
            if len(tensor.shape) != 2 or len(tensor.strides) != 2:
                raise LoweringError("GEMM tensor descriptors must have rank two")
            if tensor.address_space != "device":
                raise LoweringError("GEMM tensor descriptors require device memory")
            if not _is_constant(tensor.strides[1], 1) or not _same_value(
                tensor.strides[0], tensor.shape[1]
            ):
                raise LoweringError(
                    "GEMM currently supports contiguous row-major tensor descriptors"
                )
        left_tensor, right_tensor, output_tensor = tensors
        dimensions = (left_tensor.shape[0], right_tensor.shape[1], left_tensor.shape[1])
        rows, columns, reduction = dimensions
        if not all(
            _same_value(actual, expected)
            for actual, expected in (
                (right_tensor.shape[0], reduction),
                (output_tensor.shape[0], rows),
                (output_tensor.shape[1], columns),
            )
        ):
            raise LoweringError("GEMM tensor descriptor dimensions do not agree")
        if left_tensor.access == "write" or right_tensor.access == "write":
            raise LoweringError("GEMM inputs must be readable")
        if output_tensor.access == "read":
            raise LoweringError("GEMM output must be writable")
        if output.ptr.name in {left.ptr.name, right.ptr.name}:
            raise LoweringError("GEMM output cannot reuse an input tensor pointer")
        if sum(isinstance(operation, tir.TileLoad) for operation in operations) != 2 or any(
            isinstance(operation, (tir.Load, tir.Store, tir.SharedAlloc, tir.Barrier))
            for operation in operations
        ):
            raise LoweringError(
                "GEMM descriptor kernels cannot contain additional memory operations"
            )
        for dimension in dimensions:
            if not isinstance(dimension.type, ScalarType) or dimension.type.dtype not in {
                "i32",
                "u32",
            }:
                raise LoweringError("GEMM dimensions must be integer scalars")
            if isinstance(dimension.defining_op, tir.Constant):
                if dimension.defining_op.value <= 0:
                    raise LoweringError("GEMM dimensions must be positive")
            elif dimension.defining_op is not None or dimension.name not in parameters:
                raise LoweringError("GEMM dimensions must be scalar parameters or constants")
        loops = [
            operation
            for operation in operations
            if isinstance(operation, tir.ForRange)
            and any(nested is dot for nested in _walk_tile_ops(operation.body))
        ]
        if len(loops) != 1:
            raise LoweringError("GEMM requires one reduction loop")
        loop = loops[0]
        if not any(operation is loop for operation in func.ops) or any(
            isinstance(operation, (tir.ForRange, tir.PersistentRange, tir.SimdgroupRole))
            and operation is not loop
            for operation in operations
        ):
            raise LoweringError("GEMM descriptor kernels require one top-level reduction loop")
        if any(
            not isinstance(operation, (tir.Constant, tir.BinOp, tir.TileLoad, tir.Dot))
            or (isinstance(operation, tir.BinOp) and isinstance(operation.result.type, TileType))
            for operation in loop.body
        ):
            raise LoweringError(
                "GEMM reduction loops support only tensor loads and the dot recurrence"
            )
        if not (
            _is_constant(loop.start, 0)
            and _same_value(loop.end, reduction)
            and loop.step == block_k
            and _same_value(left.col_offset, loop.iv)
            and _same_value(right.row_offset, loop.iv)
            and _is_tile_origin(left.row_offset, 0, block_m)
            and _is_tile_origin(right.col_offset, 1, block_n)
            and _same_value(output.row_offset, left.row_offset)
            and _same_value(output.col_offset, right.col_offset)
        ):
            raise LoweringError("GEMM descriptor offsets must describe the canonical tiled product")
        if any(operation.ptr.name in {"M", "N", "K"} for operation in (left, right, output)):
            raise LoweringError("GEMM tensor pointers cannot use reserved dimension names M/N/K")
        _detect_epilogue(func.ops, func=func)
    else:
        rows_param = parameters.get("M")
        if rows_param is None or not isinstance(rows_param.type, ScalarType):
            raise LoweringError("Legacy GEMM requires an M dimension; use tensor descriptors")
        dimensions = (tir.Value("M", rows_param.type), right.stride, left.stride)
        if not _same_value(output.stride, right.stride):
            raise LoweringError("Legacy GEMM output stride must match the right input stride")
    return _GemmBinding(left, right, output, dimensions, tiles, uses_descriptors)


def _gemm_tiles(func: tir.Function) -> tuple[int, int, int]:
    return _analyze_gemm(func).tiles


def _tensor_ops_aligned(func: tir.Function) -> bool:
    alignment = dict(func.constexprs.get("_SCALAR_ALIGNMENT_32", ()))
    dimensions = _analyze_gemm(func).dimensions
    if not all(
        dimension.defining_op.value % 32 == 0
        if isinstance(dimension.defining_op, tir.Constant)
        else alignment.get(dimension.name) == 0
        for dimension in dimensions
    ):
        return False
    step = _tensor_ops_k_step(func)
    reduction = dimensions[2]
    if isinstance(reduction.defining_op, tir.Constant):
        return reduction.defining_op.value % step == 0
    runtime_scalars = dict(func.constexprs.get("_RUNTIME_SCALARS", ()))
    if reduction.name in runtime_scalars:
        return runtime_scalars[reduction.name] % step == 0
    return 32 % step == 0


def _tensor_ops_k_step(func: tir.Function) -> int:
    block_m, block_n, block_k = _gemm_tiles(func)
    constexprs = func.constexprs
    warp_rows, warp_columns = constexprs.get("WM", 2), constexprs.get("WN", 2)
    if any(not isinstance(count, int) or count < 1 for count in (warp_rows, warp_columns)):
        raise LoweringError("WM and WN must be positive integers")
    simdgroup_m = block_m // warp_rows
    simdgroup_n = block_n // warp_columns
    cooperative = constexprs.get("COOPERATIVE", False)
    separated_default = (
        not cooperative
        and simdgroup_m <= 32
        and simdgroup_n <= 32
        and 32 in (simdgroup_m, simdgroup_n, min(32, block_k))
    )
    separated = constexprs.get("SEPARATED", separated_default) and not cooperative
    unroll = constexprs.get("K_UNROLL", 1)
    if not isinstance(unroll, int) or unroll < 1:
        raise LoweringError("K_UNROLL must be a positive integer")
    if separated:
        inner = min(32, block_k)
        return inner * max(2 if 2 * inner <= block_k else 1, unroll)
    return block_k * unroll


def _extract_gemm_params(
    func: tir.Function, param_values: dict[str, mir.MValue], mfunc: mir.MFunction
) -> tuple[mir.MValue, mir.MValue, mir.MValue, mir.MValue, mir.MValue, mir.MValue]:
    """Bind matrix operands and dimensions from the traced tensor operations."""
    binding = _analyze_gemm(func)
    for axis, dimension in zip(("M", "N", "K"), binding.dimensions, strict=True):
        if isinstance(dimension.defining_op, tir.Constant):
            mfunc.dimension_bindings[axis] = int(dimension.defining_op.value)
        elif dimension.name in param_values:
            mfunc.dimension_bindings[axis] = param_values[dimension.name]
        else:
            raise LoweringError("GEMM dimensions must reference scalar parameters or constants")
    pointers = tuple(
        param_values[operation.ptr.name]
        for operation in (binding.left, binding.right, binding.output)
    )
    dimensions = tuple(mir.MValue(axis, ScalarType("i32")) for axis in ("M", "N", "K"))
    return (*pointers, *dimensions)


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


def _compute_simdgroup_layout(
    BM: int,
    BN: int,
    NUM_SG: int,
    *,
    simdgroup_grid: tuple[int, int] | None = None,
) -> mir.SimdgroupLayout:
    """Derive simdgroup tiling from layout algebra.

    The MMA accumulator grid is (BM/8) x (BN/8) tiles of 8x8 simdgroup_matrix.
    We partition this grid across NUM_SG simdgroups using logical_divide.
    The factorization (sg_rows x sg_cols) is chosen to keep acc_per_sg <= 16.
    """
    if type(NUM_SG) is not int or not 1 <= NUM_SG <= 32:
        raise LoweringError("SIMDgroup count must be an integer between 1 and 32")
    if BM < 8 or BN < 8 or BM % 8 or BN % 8:
        raise LoweringError("SIMDgroup tile dimensions must be positive multiples of 8")
    if simdgroup_grid is not None and (
        len(simdgroup_grid) != 2
        or any(type(size) is not int or size < 1 for size in simdgroup_grid)
        or simdgroup_grid[0] * simdgroup_grid[1] != NUM_SG
    ):
        raise LoweringError("SIMDgroup grid must match the requested group count")
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
        if simdgroup_grid is not None and (sg_rows, sg_cols) != simdgroup_grid:
            continue
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
        raise LoweringError("GEMM tile cannot be partitioned across the requested SIMDgroups")

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
    for num_sg in (2, 1):
        try:
            _compute_simdgroup_layout(BM, BN, num_sg)
        except LoweringError:
            continue
        return num_sg
    raise LoweringError("GEMM tile exceeds the supported SIMDgroup accumulator capacity")


def _detect_epilogue(ops: list, *, func: tir.Function | None = None) -> list:
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
    if func is not None and func.tensors:
        from metile.compiler.epilogue import EpilogueError, build_epilogue

        try:
            program = build_epilogue(func)
        except EpilogueError as error:
            raise LoweringError(str(error)) from error
        return [program] if program is not None else []

    strict = any(
        isinstance(operation, tir.TileStore) and getattr(operation, "tensor", None) is not None
        for operation in _walk_tile_ops(ops)
    )

    def unsupported():
        if strict:
            raise LoweringError("GEMM descriptor epilogue is not supported by this lowering")
        return []

    for_idx = store_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, tir.ForRange) and _has_gemm_ops(op.body):
            for_idx = i
        if isinstance(op, tir.TileStore):
            store_idx = i

    if for_idx is None or store_idx is None:
        return unsupported()

    # Find the accumulator value name (last Dot result in the loop body)
    acc_name = _find_dot_result_name(ops[for_idx].body)
    if acc_name is None:
        return unsupported()
    if store_idx <= for_idx + 1:
        if strict and ops[store_idx].value.name != acc_name:
            return unsupported()
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
                if strict and not (
                    cond_op.lhs.name == chain_name
                    and _is_constant(cond_op.rhs, 0)
                    and op.true_val.name == chain_name
                    and _is_constant(op.false_val, 0)
                ):
                    return unsupported()
                epilogue.append(("relu",))
            else:
                # Non-gt Select patterns (e.g. clamp, abs-via-select) are not
                # currently fusible — bail out of epilogue detection.
                return unsupported()
            chain_name = op.result.name

        elif isinstance(op, tir.Unary):
            if strict and (
                op.operand.name != chain_name
                or op.op
                not in {
                    "exp",
                    "exp2",
                    "fast_cos",
                    "fast_exp",
                    "fast_exp2",
                    "fast_sin",
                    "log",
                    "sqrt",
                    "rsqrt",
                    "abs",
                    "neg",
                    "tanh",
                }
            ):
                return unsupported()
            epilogue.append(("unary", op.op))
            chain_name = op.result.name

        elif isinstance(op, tir.BinOp):
            if strict and op.op not in {"add", "sub", "mul", "div"}:
                return unsupported()
            lhs_tile = isinstance(op.lhs.type, TileType)
            rhs_tile = isinstance(op.rhs.type, TileType)

            if lhs_tile and not rhs_tile:
                if strict and op.lhs.name != chain_name:
                    return unsupported()
                # chain OP scalar_const
                const_val = _extract_constant(op.rhs)
                if const_val is not None:
                    epilogue.append(("binop", op.op, "rhs", const_val))
                elif op.op == "mul":
                    if strict:
                        return unsupported()
                    epilogue.append(("scale",))
                else:
                    return unsupported()
            elif rhs_tile and not lhs_tile:
                if strict and op.rhs.name != chain_name:
                    return unsupported()
                # scalar_const OP chain
                const_val = _extract_constant(op.lhs)
                if const_val is not None:
                    epilogue.append(("binop", op.op, "lhs", const_val))
                else:
                    return unsupported()
            elif lhs_tile and rhs_tile:
                # Both TileType: one must be original acc, other is chain
                lhs_is_orig = op.lhs.name == acc_name and op.lhs.name != chain_name
                rhs_is_orig = op.rhs.name == acc_name and op.rhs.name != chain_name
                if strict and not (
                    (lhs_is_orig and op.rhs.name == chain_name)
                    or (rhs_is_orig and op.lhs.name == chain_name)
                ):
                    return unsupported()
                if lhs_is_orig:
                    epilogue.append(("binop_orig", op.op, "lhs"))
                    needs_orig = True
                elif rhs_is_orig:
                    epilogue.append(("binop_orig", op.op, "rhs"))
                    needs_orig = True
                else:
                    return unsupported()
            else:
                return unsupported()
            chain_name = op.result.name
        elif strict and not isinstance(op, tir.Compare):
            return unsupported()

    if strict and ops[store_idx].value.name != chain_name:
        return unsupported()

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
