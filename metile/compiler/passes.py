from __future__ import annotations

import copy

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
        elif isinstance(op, mir.MMAInnerLoop):
            op.a_stride += array_pads.get(op.shared_a, pad or 2)
            op.b_stride += array_pads.get(op.shared_b, pad or 2)
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

    func.ops = _split_ew_recursive(func.ops)
    return func


_ew_split_counter = 0


def _split_ew_recursive(ops: list[mir.MOp]) -> list[mir.MOp]:
    global _ew_split_counter
    new_ops = []
    for op in ops:
        if isinstance(op, mir.MForLoop) and _has_ifblock(op.body):
            new_ops.extend(_split_ew_for_loop(op, _ew_split_counter))
            _ew_split_counter += 1
        else:
            new_ops.append(op)
    return new_ops


def _has_ifblock(body: list[mir.MOp]) -> bool:
    return any(isinstance(op, mir.IfBlock) for op in body)


def _split_ew_for_loop(loop: mir.MForLoop, loop_id: int) -> list[mir.MOp]:
    """Split an element-wise ForRange into aligned + tail."""
    # Aligned loop: body without IfBlock wrapper (ops inlined)
    aligned_body = []
    for op in loop.body:
        if isinstance(op, mir.IfBlock):
            aligned_body.extend(op.body)
        else:
            aligned_body.append(op)

    aligned = copy.deepcopy(loop)
    aligned.body = copy.deepcopy(aligned_body)
    aligned._ew_aligned = True
    aligned._ew_id = loop_id
    # Propagate num_stages from lowering
    if hasattr(loop, "_num_stages"):
        aligned._num_stages = loop._num_stages

    # Tail: single iteration with original body (IfBlock intact)
    tail = copy.deepcopy(loop)
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
        if isinstance(op, mir.MForLoop) and getattr(op, "_ew_aligned", False):
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


def add_simdgroup_barriers(func: mir.MFunction) -> mir.MFunction:
    """Add simdgroup_barrier hints in MMA inner loops.

    These zero-cost barriers help the Metal compiler schedule
    simdgroup loads and MMA operations more efficiently.

    For decomposed IR: inserts MSimdgroupBarrierOp between each
    load and MMA op in kk ForLoops.
    For legacy IR: sets the use_simdgroup_barrier flag on MMAInnerLoop.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _add_barriers(op):
        if isinstance(op, mir.MMAInnerLoop):
            op.use_simdgroup_barrier = True
        elif isinstance(op, mir.MForLoop) and getattr(op, "_unroll", False):
            _insert_barriers(op)

    _walk_ops(func.ops, _add_barriers)
    return func


def _insert_barriers(kk_loop: mir.MForLoop):
    """Insert MSimdgroupBarrierOp between loads and MMA ops in kk body."""
    old_body = kk_loop.body
    new_body = []
    for op in old_body:
        if isinstance(op, (mir.MSimdgroupLoad, mir.MSimdgroupMMA)):
            new_body.append(mir.MSimdgroupBarrierOp())
        new_body.append(op)
    kk_loop.body = new_body


def serpentine_mma(func: mir.MFunction) -> mir.MFunction:
    """Enable serpentine (zigzag) N-traversal in MMA inner loops.

    On even M rows, N iterates 0,1,2,...; on odd rows, N iterates
    ...,2,1,0. This keeps recently-used B fragments alive in registers
    across consecutive M iterations, improving data reuse.

    For decomposed IR: reorders MSimdgroupLoad/MMA ops within kk ForLoops
    so that odd M rows iterate N in reverse.
    For legacy IR: sets the serpentine flag on MMAInnerLoop.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _transform(op):
        # Legacy path
        if isinstance(op, mir.MMAInnerLoop):
            op.serpentine = True
        # New decomposed path: find kk ForLoops and reorder
        elif isinstance(op, mir.MForLoop) and getattr(op, "_unroll", False):
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
    loop does: load all A tiles → load all B tiles → all MMA operations.

    For decomposed IR: reorders ops in kk ForLoop bodies.
    For legacy IR: sets the preload_tiles flag on MMAInnerLoop.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func

    def _enable_preload(op):
        if isinstance(op, mir.MMAInnerLoop):
            op.preload_tiles = True
        elif isinstance(op, mir.MForLoop) and getattr(op, "_unroll", False):
            _reorder_preload(op)

    _walk_ops(func.ops, _enable_preload)
    return func


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
        elif isinstance(op, mir.MSimdgroupBarrierOp):
            pass  # drop existing barriers, we'll re-insert if needed
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
        elif isinstance(op, mir.MMAInnerLoop):
            sw_a = array_swizzle.get(op.shared_a)
            sw_b = array_swizzle.get(op.shared_b)
            if sw_a:
                op.a_swizzle_bits, op.a_swizzle_shift = sw_a
            if sw_b:
                op.b_swizzle_bits, op.b_swizzle_shift = sw_b
        elif isinstance(op, mir.MSimdgroupLoad):
            sw = array_swizzle.get(op.src_array)
            if sw:
                op.swizzle_bits, op.swizzle_shift = sw

    _walk_ops(func.ops, _apply)
    return func


def double_buffer_k_loop(func: mir.MFunction, max_tg_bytes: int = 30720):
    """Double-buffer the K-loop for software-pipelined prefetching.

    Overlaps loading of the next K-block with computing the current one.
    Doubles threadgroup memory allocations (shared_a_0/1, shared_b_0/1).

    Returns (func, did_apply) — skips if doubled memory exceeds max_tg_bytes.
    """
    if func.kernel_type not in ("gemm", "persistent_gemm", "specialized_gemm"):
        return func, False

    # Check memory budget: sum all threadgroup allocs
    elem_sizes = {"float": 4, "half": 2}
    total_bytes = 0
    alloc_names = []
    for op in func.ops:
        if isinstance(op, mir.MThreadgroupAlloc):
            total_bytes += op.size * elem_sizes.get(op.elem_type, 4)
            alloc_names.append(op.alloc_name)

    if 2 * total_bytes > max_tg_bytes:
        return func, False

    # Double the threadgroup allocations
    new_ops = []
    for op in func.ops:
        if isinstance(op, mir.MThreadgroupAlloc) and op.alloc_name in ("shared_a", "shared_b"):
            # Replace with two buffers
            new_ops.append(
                mir.MThreadgroupAlloc(
                    alloc_name=f"{op.alloc_name}_0", elem_type=op.elem_type, size=op.size
                )
            )
            new_ops.append(
                mir.MThreadgroupAlloc(
                    alloc_name=f"{op.alloc_name}_1", elem_type=op.elem_type, size=op.size
                )
            )
        else:
            new_ops.append(op)
    func.ops = new_ops

    # Mark the K-loop
    _mark_double_buffer_recursive(func.ops)

    return func, True


def _mark_double_buffer_recursive(ops: list[mir.MOp]):
    """Find and mark kb for-loops for double buffering."""
    for op in ops:
        if isinstance(op, mir.MForLoop) and op.iv_name == "kb":
            op._double_buffered = True
            op._db_step = op.step
        elif isinstance(op, mir.MWhileTrue):
            _mark_double_buffer_recursive(op.body)


def _walk_ops(ops: list[mir.MOp], fn):
    """Walk all ops recursively, applying fn to each."""
    for op in ops:
        fn(op)
        if isinstance(op, (mir.MForLoop, mir.IfBlock, mir.MWhileTrue, mir.MSimdgroupRoleBlock)):
            _walk_ops(op.body, fn)


def _apply_pad(op: mir.MOp, pad: int):
    """Apply padding to cooperative loads and MMA loops."""
    if isinstance(op, mir.MCooperativeLoad):
        op.dst_stride = op.tile_cols + pad
    elif isinstance(op, mir.MMAInnerLoop):
        op.a_stride += pad
        op.b_stride += pad


def _update_alloc_size(alloc: mir.MThreadgroupAlloc, ops: list[mir.MOp], pad: int):
    """Update threadgroup alloc size based on padded strides."""
    for op in ops:
        if isinstance(op, mir.MCooperativeLoad) and op.tg_array == alloc.alloc_name:
            alloc.size = op.tile_rows * (op.tile_cols + pad)
            return
        if isinstance(op, mir.MForLoop):
            _update_alloc_size(alloc, op.body, pad)
            return
        if isinstance(op, mir.MWhileTrue):
            _update_alloc_size(alloc, op.body, pad)
            return
        if isinstance(op, mir.MSimdgroupRoleBlock):
            _update_alloc_size(alloc, op.body, pad)


def _split_k_for_loop(loop: mir.MForLoop) -> list[mir.MOp]:
    # Skip loops already transformed by double_buffer_k_loop
    if getattr(loop, "_double_buffered", False):
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
