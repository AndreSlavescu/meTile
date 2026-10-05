from __future__ import annotations

import copy

import numpy as np

from metile.ir import metal_ir as mir


def _optimal_pad(stride: int, num_banks: int = 32) -> int:
    """Compute the minimum pad that breaks bank-conflict alignment.

    Apple GPUs have 32 threadgroup memory banks (4 bytes each).
    Bank conflicts occur when gcd(stride, num_banks) is a power of 2
    greater than 1 — meaning many threads map to the same few banks.

    We want the smallest pad (1-4) such that gcd(stride + pad, num_banks)
    is either 1 or an odd number > 1 (i.e., NOT a power of 2 > 1).
    """
    from math import gcd

    for p in range(1, 5):
        g = gcd(stride + p, num_banks)
        # Good if gcd is 1, or has an odd factor (not a pure power of 2)
        if g & (g - 1) != 0 or g == 1:
            return p
    return 2  # fallback


def pad_shared_memory(func: mir.MFunction, pad: int | None = None) -> mir.MFunction:
    """Add padding to threadgroup memory strides to avoid bank conflicts.

    On Apple GPUs, threadgroup memory has 32 banks. If the stride equals
    or is a multiple of the bank count, all threads in a column access
    the same bank, causing serialization. Padding the stride breaks this
    alignment.

    If pad is None (default), computes the minimum pad per tile dimension
    that breaks bank alignment. Pass an explicit pad to override.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    # Collect cooperative loads to determine per-array optimal padding
    loads = {}  # tg_array -> tile_cols

    def _collect(op):
        if isinstance(op, mir.MCooperativeLoad) and op.tg_array not in loads:
            loads[op.tg_array] = op.tile_cols

    _walk_ops(func.ops, _collect)

    # Compute per-array pad: either explicit or auto-computed
    array_pads = {}
    for name, cols in loads.items():
        array_pads[name] = pad if pad is not None else _optimal_pad(cols)

    # Track what we've already padded to avoid double-padding
    padded = set()

    def apply_pad(op):
        op_id = id(op)
        if op_id in padded:
            return
        padded.add(op_id)
        if isinstance(op, mir.MCooperativeLoad):
            p = array_pads.get(op.tg_array, pad or 2)
            op.dst_stride = op.tile_cols + p
            # Keep layout info in sync
            if op.load_layout is not None:
                op.load_layout.tile.smem_stride = op.tile_cols + p
        elif isinstance(op, mir.MSimdgroupLoad):
            p = array_pads.get(op.src_array, pad or 2)
            op.stride += p

    _walk_ops(func.ops, apply_pad)

    # Update threadgroup alloc sizes
    for op in func.ops:
        if isinstance(op, mir.MThreadgroupAlloc):
            p = array_pads.get(op.alloc_name, pad or 2)
            _update_alloc_size(op, func.ops, p)

    return func


def split_k_loop(func: mir.MFunction) -> mir.MFunction:
    """Split K-loop into aligned interior (no bounds checks) + tail.

    The aligned loop processes K in multiples of BK without bounds checking.
    The tail loop handles remaining elements with bounds checking.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    func.ops = _split_k_recursive(func.ops)
    return func


def _split_k_recursive(ops: list[mir.MOp]) -> list[mir.MOp]:
    """Recursively find and split kb for-loops, including inside MWhileTrue."""
    new_ops = []
    for op in ops:
        if isinstance(op, mir.MForLoop) and op.iv_name == "kb":
            new_ops.extend(_split_k_for_loop(op))
        elif isinstance(op, mir.MWhileTrue):
            op.body = _split_k_recursive(op.body)
            new_ops.append(op)
        else:
            new_ops.append(op)
    return new_ops


def split_elementwise_loops(func: mir.MFunction) -> mir.MFunction:
    """Split element-wise ForRange loops into aligned interior + tail.

    The aligned loop runs without bounds checks (IfBlock removed).
    The tail handles remaining elements with bounds checks.
    Eliminates per-iteration branches for the common case where N >> BLOCK.
    """
    if func.kernel_type not in ("elementwise", "row_parallel"):
        return func

    counter = [0]  # per-function counter, avoids global state leaking across calls
    func.ops = _split_ew_recursive(func.ops, counter, func.threadgroup_size)
    return func


def _split_ew_recursive(
    ops: list[mir.MOp], counter: list[int], threadgroup_size: tuple[int, int, int]
) -> list[mir.MOp]:
    new_ops = []
    for op in ops:
        if isinstance(op, mir.MForLoop) and _masked_memory_operations(op.body):
            split = _split_masked_ew_for_loop(op, counter[0], threadgroup_size)
            new_ops.extend(split)
            counter[0] += len(split) == 2
        elif isinstance(op, mir.MForLoop) and _has_ifblock(op.body):
            new_ops.extend(_split_ew_for_loop(op, counter[0]))
            counter[0] += 1
        else:
            new_ops.append(op)
    return new_ops


_ELEMENTWISE_MEMORY_OPS = (
    mir.DeviceLoad,
    mir.DeviceStore,
    mir.MThreadgroupLoad,
    mir.MThreadgroupStore,
)


def _masked_memory_operations(ops: list[mir.MOp]) -> list[mir.MOp]:
    masked = []
    for op in ops:
        if isinstance(op, _ELEMENTWISE_MEMORY_OPS) and op.mask is not None:
            masked.append(op)
        if hasattr(op, "body"):
            masked.extend(_masked_memory_operations(op.body))
    return masked


def _integer_constant(value: mir.MValue | int) -> int | None:
    if isinstance(value, int):
        return value
    value = mir.resolve(value)
    if (
        isinstance(value.defining_op, mir.MConstant)
        and value.type.dtype in ("i32", "u32")
        and isinstance(value.defining_op.value, int)
    ):
        return value.defining_op.value
    return None


def _loop_index_affine(value: mir.MValue, iv_name: str) -> tuple[int, int, int] | None:
    """Prove an integer index is affine in the loop index and thread index."""
    value = mir.resolve(value)
    if value.type.dtype not in ("i32", "u32"):
        return None
    operation = value.defining_op
    if operation is None:
        return (1, 0, 0) if value.name == iv_name else None
    if isinstance(operation, mir.ThreadPositionInThreadgroup) and operation.axis == 0:
        return (0, 1, 0)
    constant = _integer_constant(value)
    if constant is not None:
        return (0, 0, constant)
    if isinstance(operation, mir.MCast) and operation.target_dtype in ("i32", "u32"):
        return _loop_index_affine(operation.value, iv_name)
    if isinstance(operation, mir.MBinOp) and operation.op in ("add", "sub", "mul"):
        left = _loop_index_affine(operation.lhs, iv_name)
        right = _loop_index_affine(operation.rhs, iv_name)
        if left is None or right is None:
            return None
        if operation.op == "add":
            return tuple(first + second for first, second in zip(left, right))
        if operation.op == "sub":
            return tuple(first - second for first, second in zip(left, right))
        if left[:2] == (0, 0):
            return tuple(left[2] * coefficient for coefficient in right)
        if right[:2] == (0, 0):
            return tuple(right[2] * coefficient for coefficient in left)
    return None


def _same_loop_bound(value: mir.MValue, bound: mir.MValue | int) -> bool:
    constant = _integer_constant(value)
    bound_constant = _integer_constant(bound)
    if constant is not None or bound_constant is not None:
        return constant is not None and constant == bound_constant
    value = mir.resolve(value)
    bound = mir.resolve(bound)
    return value is bound or (
        value.defining_op is None
        and bound.defining_op is None
        and value.name == bound.name
        and value.type == bound.type
    )


def _mask_true_in_aligned_loop(mask: mir.MValue, loop: mir.MForLoop) -> bool:
    mask = mir.resolve(mask)
    operation = mask.defining_op
    if isinstance(operation, mir.MConstant):
        return mask.type.dtype == "bool" and operation.value in (True, 1)
    if (
        isinstance(operation, mir.MBinOp)
        and operation.op in ("and", "bitand")
        and operation.lhs.type.dtype == operation.rhs.type.dtype == "bool"
    ):
        return _mask_true_in_aligned_loop(operation.lhs, loop) and _mask_true_in_aligned_loop(
            operation.rhs, loop
        )
    if not isinstance(operation, mir.MCompare):
        return False
    if _loop_index_affine(operation.lhs, loop.iv_name) != (1, 1, 0):
        return False
    if operation.predicate == "ge":
        return _integer_constant(operation.rhs) == 0
    return operation.predicate == "lt" and _same_loop_bound(operation.rhs, loop.end)


def _split_masked_ew_for_loop(
    loop: mir.MForLoop, loop_id: int, threadgroup_size: tuple[int, int, int]
) -> list[mir.MOp]:
    if (
        _integer_constant(loop.start) != 0
        or loop.step <= 0
        or threadgroup_size != (loop.step, 1, 1)
        or any(hasattr(operation, "body") for operation in loop.body)
        or not all(
            _mask_true_in_aligned_loop(operation.mask, loop)
            for operation in _masked_memory_operations(loop.body)
        )
    ):
        return [loop]
    aligned, tail = _split_ew_for_loop(loop, loop_id)
    for operation in _masked_memory_operations(aligned.body):
        operation.mask = None
    return [aligned, tail]


def _has_ifblock(body: list[mir.MOp]) -> bool:
    return any(isinstance(op, mir.IfBlock) for op in body)


def _split_ew_for_loop(loop: mir.MForLoop, loop_id: int) -> list[mir.MOp]:
    """Split an element-wise ForRange into aligned + tail."""
    local_operations = []
    _walk_ops([loop], local_operations.append)
    local_ids = {id(operation) for operation in local_operations}
    external_values = {}

    def preserve_external(value):
        if isinstance(value, mir.MValue):
            if id(value.defining_op) not in local_ids:
                external_values[id(value)] = value
        elif isinstance(value, (list, tuple)):
            for member in value:
                preserve_external(member)
        elif isinstance(value, dict):
            for member in value.values():
                preserve_external(member)

    for operation in local_operations:
        for value in vars(operation).values():
            preserve_external(value)

    aligned = copy.deepcopy(loop, dict(external_values))
    # Aligned loop: body without IfBlock wrapper (ops inlined)
    aligned_body = []
    for op in aligned.body:
        if isinstance(op, mir.IfBlock):
            aligned_body.extend(op.body)
        else:
            aligned_body.append(op)

    aligned.body = aligned_body
    aligned._ew_aligned = True
    aligned._ew_id = loop_id
    # Propagate num_stages from lowering
    if hasattr(loop, "_num_stages"):
        aligned._num_stages = loop._num_stages

    # Tail: single iteration with original body (IfBlock intact)
    tail = copy.deepcopy(loop, dict(external_values))
    tail._ew_tail = True
    tail._ew_id = loop_id

    return [aligned, tail]


def vectorize_elementwise(func: mir.MFunction, vec_size: int = 4) -> mir.MFunction:
    """Mark aligned element-wise loops for vec4 emission.

    Each thread loads/stores vec_size consecutive elements per iteration,
    reducing loop overhead and enabling wider memory transactions.
    The tail loop becomes a scalar loop to handle remainders.
    """
    if func.kernel_type not in ("elementwise", "row_parallel"):
        return func

    ops = func.ops
    i = 0
    while i < len(ops):
        op = ops[i]
        if (
            isinstance(op, mir.MForLoop)
            and getattr(op, "_ew_aligned", False)
            and _elementwise_loop_supports_vectorization(op.body)
        ):
            op._vec_size = vec_size
            # Mark the paired tail as a loop (not single-iteration)
            if i + 1 < len(ops):
                nxt = ops[i + 1]
                if (
                    isinstance(nxt, mir.MForLoop)
                    and getattr(nxt, "_ew_tail", False)
                    and getattr(nxt, "_ew_id", -1) == getattr(op, "_ew_id", -2)
                ):
                    nxt._vec_tail = True
        i += 1
    return func


def _elementwise_loop_supports_vectorization(
    ops: list[mir.MOp], local_operations: set[int] | None = None
) -> bool:
    """Reject scalar/subgroup semantics that vec4 emission cannot preserve."""
    if local_operations is None:
        local_operations = set()
        _walk_ops(ops, lambda operation: local_operations.add(id(operation)))
    for op in ops:
        if isinstance(op, (mir.MSimdBroadcast, mir.MSimdShuffleXor)):
            return False
        if isinstance(op, (mir.MThreadgroupLoad, mir.MThreadgroupStore)):
            return False
        if isinstance(op, (mir.DeviceLoad, mir.DeviceStore)) and (
            op.mask is not None
            or _elementwise_lane_stride(op.index) != 1
            or not _lane_coordinates_are_local(op.index, local_operations)
        ):
            return False
        if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue)) and not (
            _elementwise_loop_supports_vectorization(op.body, local_operations)
        ):
            return False
    return True


def _lane_coordinates_are_local(value: mir.MValue, local_operations: set[int]) -> bool:
    """Reject hoisted lane expressions that the vector emitter cannot rescale."""
    value = mir.resolve(value)
    operation = value.defining_op
    if _elementwise_lane_stride(value) == 0 or isinstance(
        operation, mir.ThreadPositionInThreadgroup
    ):
        return True
    if id(operation) not in local_operations:
        return False
    if isinstance(operation, mir.MCast):
        return _lane_coordinates_are_local(operation.value, local_operations)
    if isinstance(operation, mir.MBinOp):
        return _lane_coordinates_are_local(
            operation.lhs, local_operations
        ) and _lane_coordinates_are_local(operation.rhs, local_operations)
    return False


def _elementwise_lane_stride(value: mir.MValue) -> int | None:
    """Determine the constant address increment between adjacent thread lanes."""
    value = mir.resolve(value)
    if value.type.dtype not in ("i32", "u32"):
        return None
    operation = value.defining_op
    if operation is None or isinstance(operation, (mir.MConstant, mir.ThreadgroupPositionInGrid)):
        return 0
    if isinstance(operation, mir.ThreadPositionInThreadgroup) and operation.axis == 0:
        return 1
    if isinstance(operation, mir.MCast) and operation.target_dtype in ("i32", "u32"):
        return _elementwise_lane_stride(operation.value)
    if isinstance(operation, mir.MBinOp):
        left = _elementwise_lane_stride(operation.lhs)
        right = _elementwise_lane_stride(operation.rhs)
        if left is None or right is None:
            return None
        if operation.op == "add":
            return left + right
        if operation.op == "sub":
            return left - right
        if left == right == 0:
            return 0
        if operation.op == "mul":
            left_constant = _integer_constant(operation.lhs)
            right_constant = _integer_constant(operation.rhs)
            if left_constant is not None:
                return left_constant * right
            if right_constant is not None:
                return right_constant * left
    return None


def vectorize_loads(func: mir.MFunction, vec_size: int = 4) -> mir.MFunction:
    """Transform cooperative loads to use vec4 device reads.

    Changes MCooperativeLoad.vec_size from 1 to vec_size.
    Only applies to loads without bounds checking (in aligned K-loop).
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _vectorize(op):
        if isinstance(op, mir.MCooperativeLoad) and not op.bounds_check:
            op.vec_size = vec_size

    _walk_ops(func.ops, _vectorize)
    return func


def serpentine_mma(func: mir.MFunction) -> mir.MFunction:
    """Enable serpentine (zigzag) N-traversal in MMA inner loops.

    On even M rows, N iterates 0,1,2,...; on odd rows, N iterates
    ...,2,1,0. This keeps recently-used B fragments alive in registers
    across consecutive M iterations, improving data reuse.

    Reorders MSimdgroupLoad/MMA ops within kk ForLoops so that odd M rows
    iterate N in reverse.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _transform(op):
        if isinstance(op, mir.MForLoop) and getattr(op, "_unroll", False):
            _reorder_serpentine(op)

    _walk_ops(func.ops, _transform)
    return func


def _reorder_serpentine(kk_loop: mir.MForLoop):
    """Reorder loads and MMA ops in a kk inner loop for serpentine traversal.

    Groups ops by mi value. For odd mi values, reverses the ni ordering
    of B loads and MMA ops.
    """
    body = kk_loop.body

    # Collect into structured groups
    # The default ordering from lowering is:
    # For each mi:
    #   A_load(mi)
    #   For each ni:
    #     B_load(ni)
    #     MMA(mi, ni)
    groups = {}  # mi -> {'a_load': op, 'b_mma_pairs': [(b_load, mma), ...]}
    current_mi = None

    for op in body:
        if isinstance(op, mir.MSimdgroupLoad) and not op.is_b:
            current_mi = op.tile_idx
            if current_mi not in groups:
                groups[current_mi] = {"a_load": op, "b_mma_pairs": []}
            else:
                groups[current_mi]["a_load"] = op
        elif isinstance(op, mir.MSimdgroupLoad) and op.is_b:
            # This B load belongs to the current mi group
            if current_mi is not None and current_mi in groups:
                groups[current_mi]["b_mma_pairs"].append([op])
        elif isinstance(op, mir.MSimdgroupMMA):
            if current_mi is not None and current_mi in groups:
                pairs = groups[current_mi]["b_mma_pairs"]
                if pairs and len(pairs[-1]) == 1:
                    pairs[-1].append(op)

    if not groups:
        return

    # Rebuild body with serpentine ordering
    new_body = []
    for mi in sorted(groups.keys()):
        g = groups[mi]
        new_body.append(g["a_load"])
        pairs = g["b_mma_pairs"]
        if mi % 2 == 1:
            # Odd mi: reverse ni order
            pairs = list(reversed(pairs))
        for pair in pairs:
            new_body.extend(pair)

    kk_loop.body = new_body


def preload_mma_tiles(func: mir.MFunction) -> mir.MFunction:
    """Preload all A and B simdgroup tiles before computing.

    Separates simdgroup loads from MMA compute for better instruction-level
    parallelism. Instead of interleaving load-A/load-B/MMA, the restructured
    loop does: load all A tiles -> load all B tiles -> all MMA operations.

    Reorders ops in kk ForLoop bodies.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _enable_preload(op):
        if isinstance(op, mir.MForLoop) and getattr(op, "_unroll", False):
            _reorder_preload(op)

    _walk_ops(func.ops, _enable_preload)
    return func


def decompose_nax_fragments(func: mir.MFunction) -> mir.MFunction:
    """Lower fused NAX operations into composable register-fragment primitives.

    The high-level operations remain useful as a compact target for GEMM and
    block-scaled lowering. This pass exposes the individual layout, load,
    packing, native MMA, and store operations so later passes can reorder or
    replace them without owning an entire kernel template.
    """
    if func.kernel_type != "tensor_ops_gemm":
        return func
    func.ops = _decompose_nax_ops(func.ops)
    return func


def _decompose_nax_ops(ops: list[mir.MOp]) -> list[mir.MOp]:
    decomposed = []
    index = 0
    while index < len(ops):
        op = ops[index]
        if isinstance(op, mir.MNaxGemmSetup):
            decomposed.extend(
                (
                    mir.MNaxTileLayout(
                        block_m=op.block_m,
                        block_n=op.block_n,
                        wn=op.wn,
                        m=op.m,
                        n=op.n,
                        k=op.k,
                    ),
                    mir.MNaxAccumulatorInit(),
                    mir.MNaxMatmul2dDecl(
                        left_type=op.left_type,
                        right_type=op.right_type,
                        relaxed=op.relaxed,
                    ),
                )
            )
        elif isinstance(op, mir.MNaxGemmRun):
            runs = []
            while index < len(ops) and isinstance(ops[index], mir.MNaxGemmRun):
                runs.append(ops[index])
                index += 1
            decomposed.extend(_dense_nax_steps(runs))
            continue
        elif isinstance(op, mir.MNaxBlockScaledRun):
            runs = []
            while index < len(ops) and isinstance(ops[index], mir.MNaxBlockScaledRun):
                runs.append(ops[index])
                index += 1
            decomposed.extend(_block_scaled_nax_steps(runs))
            continue
        elif isinstance(op, mir.MNaxAffineRun):
            runs = []
            while index < len(ops) and isinstance(ops[index], mir.MNaxAffineRun):
                runs.append(ops[index])
                index += 1
            decomposed.extend(_affine_nax_steps(runs))
            continue
        elif isinstance(op, mir.MNaxGemmEpilogue):
            decomposed.extend(
                mir.MNaxApplyFragment(source=source, operations=list(op.operations))
                for source in ("d00", "d01", "d10", "d11")
            )
        elif isinstance(op, mir.MNaxGemmStore):
            for source, row_offset, col_offset in (
                ("d00", 0, 0),
                ("d01", 0, 16),
                ("d10", 16, 0),
                ("d11", 16, 16),
            ):
                decomposed.append(
                    mir.MNaxStoreFragment(
                        ptr_c=op.ptr_c,
                        source=source,
                        row_offset=row_offset,
                        col_offset=col_offset,
                        row_bound=op.row_bound,
                    )
                )
        else:
            if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
                op.body = _decompose_nax_ops(op.body)
            decomposed.append(op)
        index += 1
    return decomposed


def _dense_nax_steps(runs: list[mir.MNaxGemmRun]) -> list[mir.MOp]:
    operations = []
    fragments = []
    for run_index, run in enumerate(runs):
        suffix = "" if len(runs) == 1 else f"_{run_index}"
        b0 = f"b0{suffix}"
        b1 = f"b1{suffix}"
        a0 = f"a0{suffix}"
        a1 = f"a1{suffix}"
        operations.extend(
            (
                mir.MNaxLoadFragment(
                    ptr=run.ptr_b,
                    name=b0,
                    operand="right",
                    k_offset=run.k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=run.ptr_b,
                    name=b1,
                    operand="right",
                    col_offset=16,
                    k_offset=run.k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=run.ptr_a,
                    name=a0,
                    operand="left",
                    k_offset=run.k_offset,
                    row_bound=run.row_bound,
                ),
                mir.MNaxLoadFragment(
                    ptr=run.ptr_a,
                    name=a1,
                    operand="left",
                    row_offset=16,
                    k_offset=run.k_offset,
                    row_bound=run.row_bound,
                ),
            )
        )
        fragments.append((b0, b1, a0, a1))
    for b0, b1, a0, a1 in fragments:
        operations.extend(
            (
                mir.MNaxPackRight(low=b0, high=b1),
                mir.MNaxFmaFragment(left=a0),
                mir.MNaxFmaFragment(
                    left=a1,
                    destination_low="d10",
                    destination_high="d11",
                ),
            )
        )
    return operations


def _block_scaled_nax_steps(runs: list[mir.MNaxBlockScaledRun]) -> list[mir.MOp]:
    operations = []
    grouped_runs = []
    for run in runs:
        scale_group = run.k_offset // 32
        if not grouped_runs or grouped_runs[-1][0] != scale_group:
            grouped_runs.append((scale_group, []))
        grouped_runs[-1][1].append(run)

    run_index = 0
    for group_index, (_, scale_runs) in enumerate(grouped_runs):
        scale_suffix = "" if len(runs) == 1 else f"_g{group_index}"
        scale_low = f"b0_scale{scale_suffix}"
        scale_high = f"b1_scale{scale_suffix}"
        first_run = scale_runs[0]
        operations.extend(
            (
                mir.MNaxLoadBlockScale(
                    ptr_scales=first_run.ptr_scales,
                    name=scale_low,
                    k_offset=first_run.k_offset,
                ),
                mir.MNaxLoadBlockScale(
                    ptr_scales=first_run.ptr_scales,
                    name=scale_high,
                    col_offset=16,
                    k_offset=first_run.k_offset,
                ),
            )
        )
        for run in scale_runs:
            suffix = "" if len(runs) == 1 else f"_{run_index}"
            operations.extend(_block_scaled_nax_step(run, scale_low, scale_high, suffix))
            run_index += 1
    return operations


def _block_scaled_nax_step(
    run: mir.MNaxBlockScaledRun,
    scale_low: str,
    scale_high: str,
    suffix: str,
) -> list[mir.MOp]:
    b0 = f"b0{suffix}"
    b1 = f"b1{suffix}"
    a0 = f"a0{suffix}"
    a1 = f"a1{suffix}"
    return [
        mir.MNaxLoadBlockScaledFragment(
            ptr_values=run.ptr_values,
            name=b0,
            scale=scale_low,
            bits=run.bits,
            k_offset=run.k_offset,
            fragment_type=run.fragment_type,
        ),
        mir.MNaxLoadBlockScaledFragment(
            ptr_values=run.ptr_values,
            name=b1,
            scale=scale_high,
            bits=run.bits,
            col_offset=16,
            k_offset=run.k_offset,
            fragment_type=run.fragment_type,
        ),
        mir.MNaxPackRight(low=b0, high=b1),
        mir.MNaxLoadFragment(
            ptr=run.ptr_a,
            name=a0,
            operand="left",
            k_offset=run.k_offset,
            row_bound=run.row_bound,
        ),
        mir.MNaxFmaFragment(left=a0),
        mir.MNaxLoadFragment(
            ptr=run.ptr_a,
            name=a1,
            operand="left",
            row_offset=16,
            k_offset=run.k_offset,
            row_bound=run.row_bound,
        ),
        mir.MNaxFmaFragment(
            left=a1,
            destination_low="d10",
            destination_high="d11",
        ),
    ]


def _affine_nax_steps(runs: list[mir.MNaxAffineRun]) -> list[mir.MOp]:
    operations = []
    grouped_runs = []
    for run in runs:
        parameter_group = run.k_offset // run.group_size
        if not grouped_runs or grouped_runs[-1][0] != parameter_group:
            grouped_runs.append((parameter_group, []))
        grouped_runs[-1][1].append(run)

    run_index = 0
    for group_index, (_, parameter_runs) in enumerate(grouped_runs):
        suffix = "" if len(grouped_runs) == 1 else f"_g{group_index}"
        low_scale = f"b0_scale{suffix}"
        low_bias = f"b0_bias{suffix}"
        high_scale = f"b1_scale{suffix}"
        high_bias = f"b1_bias{suffix}"
        first_run = parameter_runs[0]
        operations.extend(
            (
                mir.MNaxLoadAffineParameters(
                    ptr_scales=first_run.ptr_scales,
                    ptr_biases=first_run.ptr_biases,
                    scale_name=low_scale,
                    bias_name=low_bias,
                    group_size=first_run.group_size,
                    k_offset=first_run.k_offset,
                ),
                mir.MNaxLoadAffineParameters(
                    ptr_scales=first_run.ptr_scales,
                    ptr_biases=first_run.ptr_biases,
                    scale_name=high_scale,
                    bias_name=high_bias,
                    group_size=first_run.group_size,
                    col_offset=16,
                    k_offset=first_run.k_offset,
                ),
            )
        )
        for run in parameter_runs:
            run_suffix = "" if len(runs) == 1 else f"_{run_index}"
            operations.extend(
                _affine_nax_step(
                    run,
                    low_scale,
                    low_bias,
                    high_scale,
                    high_bias,
                    run_suffix,
                )
            )
            run_index += 1
    return operations


def _affine_nax_step(
    run: mir.MNaxAffineRun,
    low_scale: str,
    low_bias: str,
    high_scale: str,
    high_bias: str,
    suffix: str,
) -> list[mir.MOp]:
    b0 = f"b0{suffix}"
    b1 = f"b1{suffix}"
    a0 = f"a0{suffix}"
    a1 = f"a1{suffix}"
    return [
        mir.MNaxLoadAffineFragment(
            ptr_values=run.ptr_values,
            name=b0,
            scale=low_scale,
            bias=low_bias,
            k_offset=run.k_offset,
            fragment_type=run.fragment_type,
        ),
        mir.MNaxLoadAffineFragment(
            ptr_values=run.ptr_values,
            name=b1,
            scale=high_scale,
            bias=high_bias,
            col_offset=16,
            k_offset=run.k_offset,
            fragment_type=run.fragment_type,
        ),
        mir.MNaxPackRight(low=b0, high=b1),
        mir.MNaxLoadFragment(
            ptr=run.ptr_a,
            name=a0,
            operand="left",
            k_offset=run.k_offset,
            row_bound=run.row_bound,
        ),
        mir.MNaxFmaFragment(left=a0),
        mir.MNaxLoadFragment(
            ptr=run.ptr_a,
            name=a1,
            operand="left",
            row_offset=16,
            k_offset=run.k_offset,
            row_bound=run.row_bound,
        ),
        mir.MNaxFmaFragment(
            left=a1,
            destination_low="d10",
            destination_high="d11",
        ),
    ]


def _reorder_preload(kk_loop: mir.MForLoop):
    """Reorder ops in kk body: all A loads → all B loads → all MMA."""
    a_loads = []
    b_loads = []
    mma_ops = []
    other = []
    for op in kk_loop.body:
        if isinstance(op, mir.MSimdgroupLoad) and not op.is_b:
            a_loads.append(op)
        elif isinstance(op, mir.MSimdgroupLoad) and op.is_b:
            b_loads.append(op)
        elif isinstance(op, mir.MSimdgroupMMA):
            mma_ops.append(op)
        else:
            other.append(op)
    if a_loads and b_loads and mma_ops:
        kk_loop.body = other + a_loads + b_loads + mma_ops


def block_swizzle(func: mir.MFunction) -> mir.MFunction:
    """Add block coordinate swizzling for better L2 cache locality.

    Rotates the column assignment by the row index so that adjacent
    threadgroup rows access different column blocks, improving reuse
    of both A-row and B-column data in the L2 cache.

    Inserts a swizzle op after the block_col computation.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    # Find the block_col value and insert swizzle
    # The swizzle is: by = (tgp_id.y + tgp_id.x) % grid_n
    # We implement this by finding the tgp_y multiplication op and modifying it
    # For now, mark the function for swizzle in emission
    # (actual swizzle is applied in the emitter based on presence of this marker)
    func._swizzle = True
    return func


def swizzle_shared_memory(func: mir.MFunction) -> mir.MFunction:
    """Use XOR swizzle for bank-conflict-free shared memory access.

    Alternative to pad_shared_memory. Applies CuTe Swizzle<B,0,S> where
    S = log2(tile_cols) and B = min(S, 5). Requires power-of-2 tile_cols.

    On the write path (cooperative loads), threadgroup writes use XOR'd
    addresses. On the read path (MMA inner loop), simdgroup_load is replaced
    by manual per-thread loading via thread_elements() with XOR addressing.

    Mutually exclusive with pad_shared_memory — run one or the other.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    # Collect tile_cols per shared array
    loads: dict[str, int] = {}

    def _collect(op):
        if isinstance(op, mir.MCooperativeLoad) and op.tg_array not in loads:
            loads[op.tg_array] = op.tile_cols

    _walk_ops(func.ops, _collect)

    if not loads:
        return func

    # Validate: all must be power of 2 for XOR swizzle
    for cols in loads.values():
        if cols <= 1 or (cols & (cols - 1)) != 0:
            return func

    # Compute swizzle params per array
    from metile.ir.layout import make_swizzle

    array_swizzle: dict[str, tuple[int, int]] = {}
    for name, cols in loads.items():
        sw = make_swizzle(cols)
        if sw is None:
            return func
        array_swizzle[name] = (sw.bits, sw.shift)

    # Apply to cooperative loads and MMA inner loops
    def _apply(op):
        if isinstance(op, mir.MCooperativeLoad):
            sw = array_swizzle.get(op.tg_array)
            if sw:
                op.swizzle_bits, op.swizzle_shift = sw
                # No padding — stride stays at tile_cols
                op.dst_stride = op.tile_cols
                if op.load_layout is not None:
                    op.load_layout.tile.smem_stride = op.tile_cols
        elif isinstance(op, mir.MSimdgroupLoad):
            sw = array_swizzle.get(op.src_array)
            if sw:
                op.swizzle_bits, op.swizzle_shift = sw

    _walk_ops(func.ops, _apply)
    return func


def double_buffer_k_loop(func: mir.MFunction, max_tg_bytes: int = 30720):
    """Materialize a verified two-stage software pipeline within its memory budget."""
    from metile.compiler.staging import materialize_double_buffer

    return func, materialize_double_buffer(func, max_tg_bytes)


def _walk_ops(ops: list[mir.MOp], fn):
    """Walk all ops recursively, applying fn to each."""
    for op in ops:
        fn(op)
        if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
            _walk_ops(op.body, fn)


def _update_alloc_size(alloc: mir.MThreadgroupAlloc, ops: list[mir.MOp], pad: int):
    """Update threadgroup alloc size based on padded strides."""
    for op in ops:
        if isinstance(op, mir.MCooperativeLoad) and op.tg_array == alloc.alloc_name:
            alloc.size = op.tile_rows * (op.tile_cols + pad)
            return
        # Recurse into nested blocks but continue searching siblings
        if isinstance(op, (mir.MForLoop, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
            _update_alloc_size(alloc, op.body, pad)


def _split_k_for_loop(loop: mir.MForLoop) -> list[mir.MOp]:
    # Skip loops already transformed by double_buffer_k_loop
    if loop.staging is not None or getattr(loop, "_double_buffered", False):
        return [loop]

    """Split a K-dimension for loop into aligned + tail."""
    step = loop.step

    # Create aligned loop (no bounds checking)
    aligned_body = copy.deepcopy(loop.body)
    for op in aligned_body:
        if isinstance(op, mir.MCooperativeLoad):
            op.bounds_check = False

    aligned_loop = mir.MForLoop(
        iv_name=loop.iv_name,
        start=0,
        end=loop.end,  # will be k_aligned in emission
        step=step,
        body=aligned_body,
    )
    aligned_loop._aligned = True  # marker for emitter

    # Create tail block (bounds checking, single iteration)
    tail_body = copy.deepcopy(loop.body)
    for op in tail_body:
        if isinstance(op, mir.MCooperativeLoad):
            op.bounds_check = True

    tail_block = mir.MForLoop(
        iv_name=f"{loop.iv_name}_tail",
        start=0,
        end=loop.end,
        step=step,
        body=tail_body,
    )
    tail_block._is_tail = True  # marker for emitter

    return [aligned_loop, tail_block]


# ---------------------------------------------------------------------------
# Constant folding pass
# ---------------------------------------------------------------------------

# Binary ops that Python can evaluate at compile time
_FOLDABLE_BINOPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) and b != 0 else a / b,
    "mod": lambda a, b: a % b if b != 0 else a,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "shl": lambda a, b: a << b,
    "shr": lambda a, b: a >> b,
}

_FLOAT_DTYPES = {"f16": np.float16, "f32": np.float32}
_FLOAT_BINOPS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}


def _rounded_float(value, dtype):
    with np.errstate(all="raise"):
        result = _FLOAT_DTYPES[dtype](value)
    if not np.isfinite(result) or 0 < np.abs(result) < np.finfo(result.dtype).tiny:
        raise ValueError("nonfinite and subnormal constants retain device arithmetic")
    return result


def _is_constant_val(val: mir.MValue, target: int | float) -> bool:
    """Check if a value is a constant with the given numeric value."""
    if val.defining_op and isinstance(val.defining_op, mir.MConstant):
        return val.defining_op.value == target
    return False


def fold_constants(func: mir.MFunction) -> mir.MFunction:
    """Constant folding, identity elimination, CSE, and DCE on Metal IR.

    Optimizations:
    1. Fold MBinOp(MConstant(a), MConstant(b)) -> MConstant(result)
    2. Fold MCast(MConstant(v, src), target) -> MConstant(v, target)
    3. Eliminate identity ops: x + 0, x * 1, x - 0, x | 0, x ^ 0
    4. CSE: deduplicate identical arithmetic, casts, comparisons, and selects
    5. DCE: remove ops whose results are never referenced
    """
    _fold_constants_recursive(func.ops)
    _cse_recursive(func.ops, {})
    func.ops = _dce_constants(func.ops)
    return func


def _fold_constants_recursive(ops: list[mir.MOp]):
    """Walk ops recursively and apply constant folding."""
    for op in ops:
        _try_fold(op)
        # Recurse into nested bodies
        if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
            _fold_constants_recursive(op.body)


def _try_fold(op: mir.MOp):
    """Attempt to fold a single op in place via value forwarding."""
    if isinstance(op, mir.MBinOp) and op.result is not None:
        lhs_op = op.lhs.defining_op if op.lhs else None
        rhs_op = op.rhs.defining_op if op.rhs else None

        # Case 1: Both operands are constants -> fold to single constant
        if (
            lhs_op
            and isinstance(lhs_op, mir.MConstant)
            and rhs_op
            and isinstance(rhs_op, mir.MConstant)
        ):
            fold_fn = _FOLDABLE_BINOPS.get(op.op)
            if fold_fn is not None:
                try:
                    if any(
                        dtype in {"f16", "f32", "bf16"}
                        for dtype in (lhs_op.dtype, rhs_op.dtype, op.result.type.dtype)
                    ):
                        dtype = op.result.type.dtype
                        if (
                            dtype not in _FLOAT_DTYPES
                            or lhs_op.dtype != dtype
                            or rhs_op.dtype != dtype
                            or op.op not in _FLOAT_BINOPS
                        ):
                            return
                        left = _rounded_float(lhs_op.value, lhs_op.dtype)
                        right = _rounded_float(rhs_op.value, rhs_op.dtype)
                        with np.errstate(all="raise"):
                            result = _FLOAT_BINOPS[op.op](left, right, dtype=_FLOAT_DTYPES[dtype])
                        result_val = float(_rounded_float(result, dtype))
                    else:
                        result_val = fold_fn(lhs_op.value, rhs_op.value)
                    # Forward: make this value look like a constant
                    folded = mir.MConstant(value=result_val, dtype=lhs_op.dtype)
                    folded.result = op.result
                    op.result.defining_op = folded
                    return
                except (ArithmeticError, OverflowError, ValueError):
                    if any(
                        dtype in {"f16", "f32", "bf16"} for dtype in (lhs_op.dtype, rhs_op.dtype)
                    ):
                        return

        # Case 3: Identity elimination
        # x + 0 -> x, x - 0 -> x
        if op.op in ("add", "sub") and _is_constant_val(op.rhs, 0) and op.lhs.defining_op:
            op.result.defining_op = op.lhs.defining_op
            return
        # 0 + x -> x
        if op.op == "add" and _is_constant_val(op.lhs, 0) and op.rhs.defining_op:
            op.result.defining_op = op.rhs.defining_op
            return
        # x * 1 -> x
        if op.op == "mul" and _is_constant_val(op.rhs, 1) and op.lhs.defining_op:
            op.result.defining_op = op.lhs.defining_op
            return
        # 1 * x -> x
        if op.op == "mul" and _is_constant_val(op.lhs, 1) and op.rhs.defining_op:
            op.result.defining_op = op.rhs.defining_op
            return
        # x | 0 -> x, x ^ 0 -> x
        if op.op in ("or", "xor") and _is_constant_val(op.rhs, 0) and op.lhs.defining_op:
            op.result.defining_op = op.lhs.defining_op
            return
        # 0 | x -> x, 0 ^ x -> x
        if op.op in ("or", "xor") and _is_constant_val(op.lhs, 0) and op.rhs.defining_op:
            op.result.defining_op = op.rhs.defining_op
            return

        # Case 4: Absorbing element elimination
        # x * 0 -> 0, 0 * x -> 0
        if op.op == "mul" and (_is_constant_val(op.rhs, 0) or _is_constant_val(op.lhs, 0)):
            zero_side = op.rhs if _is_constant_val(op.rhs, 0) else op.lhs
            op.result.defining_op = zero_side.defining_op
            return
        # x & 0 -> 0, 0 & x -> 0
        if op.op == "and" and (_is_constant_val(op.rhs, 0) or _is_constant_val(op.lhs, 0)):
            zero_side = op.rhs if _is_constant_val(op.rhs, 0) else op.lhs
            op.result.defining_op = zero_side.defining_op
            return

    elif isinstance(op, mir.MCast) and op.result is not None:
        # Case 2: Cast of constant -> constant in target type
        inner = op.value
        if inner.defining_op and isinstance(inner.defining_op, mir.MConstant):
            value = inner.defining_op.value
            source_dtype = inner.defining_op.dtype
            if source_dtype in {"f16", "f32", "bf16"} or op.target_dtype in {
                "f16",
                "f32",
                "bf16",
            }:
                if op.target_dtype not in _FLOAT_DTYPES or source_dtype not in {
                    "f16",
                    "f32",
                    "i32",
                    "u32",
                }:
                    return
                try:
                    if source_dtype in _FLOAT_DTYPES:
                        value = _rounded_float(value, source_dtype)
                    else:
                        value = int(value)
                        minimum, maximum = (
                            (0, 2**32 - 1) if source_dtype == "u32" else (-(2**31), 2**31 - 1)
                        )
                        if not minimum <= value <= maximum:
                            return
                    value = float(_rounded_float(value, op.target_dtype))
                except (ArithmeticError, OverflowError, ValueError):
                    return
            folded = mir.MConstant(
                value=value,
                dtype=op.target_dtype,
            )
            folded.result = op.result
            op.result.defining_op = folded


def _stable_val_key(val: mir.MValue | None):
    """Key emitted values by typed identity, or finite constants by exact storage bits."""
    if val is None:
        return None
    val = mir.resolve(val)
    operation = val.defining_op
    if isinstance(operation, mir.MConstant) and operation.dtype == val.type.dtype:
        if operation.dtype in _FLOAT_DTYPES:
            try:
                bits = _rounded_float(operation.value, operation.dtype).tobytes()
                return "constant", val.type, bits
            except (ArithmeticError, OverflowError, ValueError):
                pass
        elif operation.dtype in {"bool", "i32", "u32", "u8"}:
            limits = {
                "bool": (0, 1),
                "i32": (-(2**31), 2**31 - 1),
                "u32": (0, 2**32 - 1),
                "u8": (0, 255),
            }
            lower, upper = limits[operation.dtype]
            if isinstance(operation.value, (int, bool)) and lower <= operation.value <= upper:
                return "constant", val.type, int(operation.value)
    return "value", val.type, val.name


def _cse_key(op: mir.MOp):
    """Generate a hashable key for an op, or None if not eligible for CSE."""
    if op.result is None:
        return None
    if isinstance(op, mir.MBinOp):
        lhs_key = _stable_val_key(op.lhs)
        rhs_key = _stable_val_key(op.rhs)
        return ("binop", op.op, lhs_key, rhs_key, op.result.type)
    if isinstance(op, mir.MCast):
        val_key = _stable_val_key(op.value)
        return ("cast", val_key, op.target_dtype, op.result.type)
    if isinstance(op, mir.MCompare):
        return (
            "compare",
            op.predicate,
            _stable_val_key(op.lhs),
            _stable_val_key(op.rhs),
            op.result.type,
        )
    if isinstance(op, mir.MSelect):
        return (
            "select",
            _stable_val_key(op.condition),
            _stable_val_key(op.true_val),
            _stable_val_key(op.false_val),
            op.result.type,
        )
    return None


def _cse_recursive(ops: list[mir.MOp], seen: dict):
    """Common Subexpression Elimination: deduplicate identical ops.

    When two ops compute the same thing (same op type, same operands),
    the second one's result is forwarded to the first.

    Scope rules:
    - Loop bodies (MForLoop, MWhileTrue) get a FRESH seen dict because the
      same op text represents different values across iterations (the loop
      variable changes).
    - If/role blocks get a COPY of the parent seen dict so they can dedup
      with outer-scope values, but inner discoveries don't leak outward.
    """
    for op in ops:
        if isinstance(op, (mir.MVarAssign, mir.MFragmentStateAssign)):
            seen.clear()
        key = _cse_key(op)
        if key is not None:
            if key in seen:
                # Forward this result to the existing one
                existing = seen[key]
                if op.result is not None and existing.result is not None:
                    op.result.defining_op = existing.result.defining_op
            else:
                seen[key] = op
        # For nested bodies: scope-aware recursion
        if isinstance(op, (mir.MForLoop, mir.MWhileTrue)):
            _cse_recursive(op.body, {})  # fresh scope for loops
        elif isinstance(op, (mir.IfBlock, mir.MSimdgroupRoleBlock)):
            _cse_recursive(op.body, dict(seen))  # copy for if/role blocks
        if hasattr(op, "body"):
            seen.clear()


def _dce_constants(ops: list[mir.MOp]) -> list[mir.MOp]:
    """Remove MConstant ops (always inlined by _val_name) and
    ops whose results were forwarded by fold/CSE (defining_op changed)."""

    def _should_remove(op):
        # MConstants are always inlined as literals
        if isinstance(op, mir.MConstant):
            return True
        # Ops whose results were forwarded to a different op by fold/CSE
        return hasattr(op, "result") and op.result is not None and op.result.defining_op is not op

    def _filter(ops_list):
        result = []
        for op in ops_list:
            if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
                op.body = _filter(op.body)
            if not _should_remove(op):
                result.append(op)
        return result

    return _filter(ops)


# ---------------------------------------------------------------------------
# Pass ordering validation
# ---------------------------------------------------------------------------

# Known ordering constraints between passes. Each entry is (before, after).
_PASS_ORDER_CONSTRAINTS = [
    # double_buffer_k_loop is attempted first and reports whether it applied; split_k_loop is
    # the fallback for when it declines the K-loop (doubling the threadgroup allocation would
    # exceed max_tg_bytes). Running the fallback first would split the kb loop that
    # double_buffer_k_loop then looks for, so the attempt has to come first.
    ("double_buffer_k_loop", "split_k_loop"),
    # split_k_loop rewrites the kb loop into aligned interior + tail; vectorize_loads must see
    # that final loop structure to widen the right loads.
    ("split_k_loop", "vectorize_loads"),
]

# Mutually exclusive passes — running both is an error.
_MUTUALLY_EXCLUSIVE = [
    ("pad_shared_memory", "swizzle_shared_memory"),
]


class PassOrderError(Exception):
    pass


def validate_pass_order(pass_names: list[str]) -> None:
    """Validate that a list of pass names respects ordering constraints.

    Raises PassOrderError if:
    - A required-before pass appears after its dependent.
    - Two mutually exclusive passes are both present.
    """
    index_of = {name: i for i, name in enumerate(pass_names)}

    for before, after in _PASS_ORDER_CONSTRAINTS:
        if before in index_of and after in index_of and index_of[before] > index_of[after]:
            raise PassOrderError(
                f"Pass '{before}' must run before '{after}', "
                f"but '{after}' appears first in the pass list."
            )

    for a, b in _MUTUALLY_EXCLUSIVE:
        if a in index_of and b in index_of:
            raise PassOrderError(
                f"Passes '{a}' and '{b}' are mutually exclusive — run one or the other."
            )
