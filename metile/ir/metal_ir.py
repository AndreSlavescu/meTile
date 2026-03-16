from __future__ import annotations

from dataclasses import dataclass, field

from metile.ir.types import BOOL, U32, PtrType, ScalarType


@dataclass
class TileLayout:
    """Precomputed layout-derived constants for a 2D tile.

    Computed from metile.ir.layout during lowering, consumed during MSL emission.
    All fields are plain ints — no Layout objects cross the IR boundary.

    The logical shape is (rows, cols). The smem_stride is the leading dimension
    of the shared memory allocation (cols + padding). For row-major tiles:
        smem[r * smem_stride + c] = device[row_offset + r][col_offset + c]
    """

    rows: int = 0
    cols: int = 0
    smem_stride: int = 0  # leading dimension in threadgroup memory (cols + pad)


@dataclass
class CooperativeLoadLayout:
    """Thread-to-element mapping for cooperative tile loading.

    Describes how num_threads threads cooperatively load a (tile_rows x tile_cols)
    tile from device memory into threadgroup memory. Derived from layout algebra's
    logical_divide: data_layout / thread_tiler.

    The mapping is: thread tid loads elements at linear indices
        tid, tid + num_threads, tid + 2*num_threads, ...
    up to tile_rows * tile_cols total elements. Each linear index maps to a 2D
    coordinate via the tile layout.
    """

    tile: TileLayout = field(default_factory=TileLayout)
    num_threads: int = 128
    elems_per_thread: int = 1  # ceil(rows * cols / num_threads)


@dataclass
class SimdgroupLayout:
    """Simdgroup-to-accumulator tiling derived from layout algebra.

    Describes how num_sg simdgroups are arranged in a (sg_rows x sg_cols) grid,
    each handling an (sg_m x sg_n) subtile of the (BM x BN) output tile.
    The accumulator count per SG is (sg_m/8) * (sg_n/8) simdgroup_matrix tiles.
    """

    num_sg: int = 4
    sg_rows: int = 2  # simdgroup grid rows
    sg_cols: int = 2  # simdgroup grid cols
    sg_m: int = 16  # per-SG subtile rows
    sg_n: int = 16  # per-SG subtile cols


@dataclass
class MValue:
    """An SSA value in Metal IR."""

    name: str
    type: ScalarType | PtrType
    defining_op: MOp | None = field(default=None, repr=False)


@dataclass
class MOp:
    """Base class for Metal IR operations."""

    result: MValue | None = field(default=None, init=False)


@dataclass
class ThreadPositionInGrid(MOp):
    """uint tid [[thread_position_in_grid]]"""

    axis: int = 0

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class ThreadgroupPositionInGrid(MOp):
    """uint3 tgp_id [[threadgroup_position_in_grid]]"""

    axis: int = 0

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class ThreadPositionInThreadgroup(MOp):
    """uint lid [[thread_position_in_threadgroup]]"""

    axis: int = 0

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class MSimdgroupId(MOp):
    """uint sgid [[simdgroup_index_in_threadgroup]]"""

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class MThreadInSimdgroup(MOp):
    """uint slid [[thread_index_in_simdgroup]]"""

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class MConstant(MOp):
    """A constant value."""

    value: int | float = 0
    dtype: str = "u32"

    def result_type(self) -> ScalarType:
        return ScalarType(self.dtype)


@dataclass
class MBinOp(MOp):
    """Binary operation at scalar level."""

    op: str = ""
    lhs: MValue = None
    rhs: MValue = None

    def result_type(self) -> ScalarType:
        return self.lhs.type


@dataclass
class MCast(MOp):
    """Type cast."""

    value: MValue = None
    target_dtype: str = "u32"

    def result_type(self) -> ScalarType:
        return ScalarType(self.target_dtype)


@dataclass
class MUnary(MOp):
    """Unary math operation (exp, log, sqrt, abs, neg)."""

    op: str = ""
    operand: MValue = None

    def result_type(self) -> ScalarType:
        return self.operand.type


@dataclass
class MSelect(MOp):
    """Ternary select: cond ? true_val : false_val."""

    condition: MValue = None
    true_val: MValue = None
    false_val: MValue = None

    def result_type(self) -> ScalarType:
        return self.true_val.type


@dataclass
class MCompare(MOp):
    """Comparison."""

    predicate: str = ""
    lhs: MValue = None
    rhs: MValue = None

    def result_type(self) -> ScalarType:
        return BOOL


@dataclass
class DeviceLoad(MOp):
    """Load from device memory: ptr[index]"""

    ptr: MValue = None
    index: MValue = None
    dtype: str = "f32"

    def result_type(self) -> ScalarType:
        return ScalarType(self.dtype)


@dataclass
class DeviceStore(MOp):
    """Store to device memory: ptr[index] = value"""

    ptr: MValue = None
    index: MValue = None
    value: MValue = None

    def result_type(self):
        return None


@dataclass
class MThreadgroupAlloc(MOp):
    """Declare a threadgroup memory array."""

    alloc_name: str = ""
    elem_type: str = "float"
    size: int = 0

    def result_type(self):
        return None


@dataclass
class MCooperativeLoad(MOp):
    """Cooperative tile load: threads cooperate to load a 2D tile.

    Progressively optimized by passes:
    - Base: scalar loads with bounds checking
    - vectorize pass: vec_size 1 → 4
    - split_k pass: bounds_check True → False for aligned iterations
    - pad pass: dst_stride adjusted for bank conflict avoidance
    - swizzle pass: XOR-based address permutation (alternative to pad)
    """

    device_ptr: MValue = None
    tg_array: str = ""
    row_offset: MValue = None
    col_offset: MValue = None
    src_stride: MValue = None
    tile_rows: int = 0
    tile_cols: int = 0
    dst_stride: int = 0
    tg_size: int = 128
    linear_tid: MValue = None
    bounds_check: bool = True
    row_bound: MValue | None = None
    col_bound: MValue | None = None
    vec_size: int = 1
    elem_type: str = "float"
    kb_expr: str | None = (
        None  # Override loop IV expr (e.g., "0" for prologue, "kb + 32" for prefetch)
    )
    load_layout: CooperativeLoadLayout | None = None  # layout-derived mapping
    swizzle_bits: int = 0  # >0 enables XOR swizzle: addr ^ ((addr >> shift) & mask)
    swizzle_shift: int = 0  # log2(tile_cols) — shift for XOR source bits

    def result_type(self):
        return None


@dataclass
class MBarrier(MOp):
    """Memory barrier."""

    kind: str = "threadgroup"
    flags: str = "mem_threadgroup"

    def result_type(self):
        return None


@dataclass
class MThreadgroupReduce(MOp):
    """Full threadgroup reduction: simd_sum + shared memory tree.

    For block_size <= 32: just simd_sum.
    For block_size > 32: simd_sum, write to shared, barrier, reduce across simdgroups, broadcast.
    """

    reduce_op: str = "sum"
    operand: MValue = None
    shared_name: str = "shared_reduce"
    block_size: int = 256
    sgid: MValue = None
    slid: MValue = None
    dtype: str = "f32"

    def result_type(self) -> ScalarType:
        return ScalarType(self.dtype)


@dataclass
class MAccumulatorInit(MOp):
    """Declare and zero-initialize simdgroup_matrix accumulator array."""

    acc_name: str = "acc"
    num_8m: int = 4
    num_8n: int = 4

    def result_type(self):
        return None


@dataclass
class MMAInnerLoop(MOp):
    """Inner loop of simdgroup MMA computation.

    Emits: for kk in 0..bk step 8: load A/B tiles, MMA
    Progressively optimized:
    - simdgroup_barrier pass: adds scheduling hints
    """

    shared_a: str = ""
    shared_b: str = ""
    acc_name: str = "acc"
    a_stride: int = 0
    b_stride: int = 0
    sg_row: MValue = None
    sg_col: MValue = None
    num_8m: int = 4
    num_8n: int = 4
    bk: int = 32
    in_type: str = "float"
    use_simdgroup_barrier: bool = False
    preload_tiles: bool = False
    serpentine: bool = False
    sg_layout: SimdgroupLayout | None = None  # layout-derived SG tiling
    a_swizzle_bits: int = 0  # >0 enables XOR swizzle for A loads
    a_swizzle_shift: int = 0
    b_swizzle_bits: int = 0  # >0 enables XOR swizzle for B loads
    b_swizzle_shift: int = 0

    def result_type(self):
        return None


@dataclass
class MAccumulatorStore(MOp):
    """Store simdgroup_matrix accumulators to device memory."""

    acc_name: str = "acc"
    device_ptr: MValue = None
    block_row: MValue = None
    block_col: MValue = None
    sg_row: MValue = None
    sg_col: MValue = None
    num_8m: int = 4
    num_8n: int = 4
    stride: MValue = None
    m_bound: MValue = None
    n_bound: MValue = None
    out_type: str = "float"
    acc_type: str = "float"

    def result_type(self):
        return None


@dataclass
class MAccumulatorEpilogue(MOp):
    """Apply element-wise operations to simdgroup_matrix accumulators.

    Uses thread_elements() to access individual elements owned by each thread,
    fusing scalar element-wise work with vector GEMM computation in registers.
    No memory traffic between GEMM and epilogue — data stays in registers.

    Each operation is a tuple:
      ("relu",)              — max(x, 0)
      ("unary", fn_name)     — exp/log/sqrt/abs/neg
      ("scale", factor_name) — x * scalar (param name)
    """

    acc_name: str = "acc"
    num_8m: int = 4
    num_8n: int = 4
    operations: list = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MTensorOpsGemm(MOp):
    """Metal 4 matmul2d via MetalPerformancePrimitives tensor_ops.

    Preemptive execution: each simdgroup independently runs matmul2d on
    its own subtile via execution_simdgroup. Manual K-tiling with
    cooperative_tensor accumulation in registers. Static template slice<>
    for optimal codegen.

    Layout: NUM_SG preemptive simdgroups arranged as WM x WN grid.
    Each SG handles a (SM x SN) subtile where SM = block_m/wm, SN = block_n/wn.
    matmul2d descriptor uses (SM, SN, bk) with multiply_accumulate mode.
    """

    ptr_a: MValue = None
    ptr_b: MValue = None
    ptr_c: MValue = None
    m_val: MValue = None
    n_val: MValue = None
    k_val: MValue = None
    block_m: int = 128
    block_n: int = 64
    bk: int = 32
    wm: int = 2
    wn: int = 2
    relaxed_precision: bool = True
    cooperative: bool = False  # True = execution_simdgroups<N>, False = execution_simdgroup
    in_type: str = "float"
    out_type: str = "float"
    acc_type: str = "float"
    epilogue_ops: list = field(default_factory=list)
    swizzle: str = "morton"  # Tile scheduling: "morton" (2x2 blocks), "diagonal", or "" (none)
    k_unroll: int = 1  # K-loop unroll factor (1 = no unrolling)
    num_stages: int = 1  # Software pipeline stages (1 = no pipelining, 2 = double buffer)
    frag_m: int | None = None  # Fragment M dim (None = use SM, i.e. ct.load path)
    frag_n: int | None = None  # Fragment N dim (None = use SN)
    frag_k: int | None = None  # Fragment K dim (None = use bk_inner=32)

    @property
    def num_sg(self) -> int:
        return self.wm * self.wn

    @property
    def sm(self) -> int:
        return self.block_m // self.wm

    @property
    def sn(self) -> int:
        return self.block_n // self.wn

    @property
    def use_manual_load(self) -> bool:
        """True when fragment dims differ from subtile dims, requiring manual element loading."""
        if self.frag_m is None:
            return False
        return self.frag_m != self.sm or self.frag_n != self.sn or self.frag_k != min(32, self.bk)

    def result_type(self):
        return None


@dataclass
class MSimdgroupRoleBlock(MOp):
    """Conditional block that executes only for simdgroups with a specific role.

    Partitions simdgroups into roles. Each role gets a contiguous range
    [first_sg, first_sg + num_sgs). The body runs under:
        if (sgid >= first_sg && sgid < first_sg + num_sgs) { ... }

    Enables simdgroup specialization: producers load tiles, consumers compute.
    """

    role: int = 0
    num_roles: int = 2
    first_sg: int = 0  # first simdgroup index for this role
    num_sgs: int = 1  # number of simdgroups assigned to this role
    sgid: MValue = None  # simdgroup_index_in_threadgroup value
    body: list[MOp] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MVarDecl(MOp):
    """Declare a mutable local variable with an initial value."""

    var_name: str = ""
    init_value: MValue = None
    dtype: str = "f32"

    def result_type(self):
        return None


@dataclass
class MVarAssign(MOp):
    """Assign a new value to a previously declared mutable variable."""

    var_name: str = ""
    value: MValue = None

    def result_type(self):
        return None


@dataclass
class IfBlock(MOp):
    """Conditional block."""

    condition: MValue = None
    body: list[MOp] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MForLoop(MOp):
    """For loop."""

    iv_name: str = "i"
    start: MValue | int = 0
    end: MValue | int = 0
    step: int = 1
    body: list[MOp] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MParam:
    """Metal function parameter."""

    name: str
    type: ScalarType | PtrType
    is_output: bool = False
    is_scalar: bool = False  # passed as constant T& vs device T*
    is_atomic: bool = False  # device atomic_uint* parameter


@dataclass
class MWhileTrue(MOp):
    """while(true) { body } — persistent kernel outer loop."""

    body: list[MOp] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MPersistentGrab(MOp):
    """Grab next tile index via atomic counter, broadcast, break if done.

    Thread 0 atomically increments the counter, stores result to threadgroup
    memory, all threads read it after barrier, and break if >= total.
    """

    counter_ptr: MValue = None
    linear_tid: MValue = None
    total_tiles: int = 0
    shared_name: str = "shared_tile_idx"
    tile_idx_name: str = "tile_idx"

    def result_type(self) -> ScalarType:
        return U32


@dataclass
class MBreak(MOp):
    """break statement."""

    def result_type(self):
        return None


# ---------------------------------------------------------------------------
# Decomposed simdgroup primitives
# ---------------------------------------------------------------------------


@dataclass
class MSimdgroupAccDecl(MOp):
    """Declare simdgroup_matrix accumulator array + temp tile arrays, zero-init.

    Generates:
        simdgroup_matrix<in_type, 8, 8> acc[num_8m][num_8n];
        // zero-init all accumulators
        simdgroup_matrix<in_type, 8, 8> a_tile[num_8m];
        simdgroup_matrix<in_type, 8, 8> b_tile[num_8n];
    """

    acc_name: str = "acc"
    num_8m: int = 4
    num_8n: int = 4
    in_type: str = "float"

    def result_type(self):
        return None


@dataclass
class MSimdgroupLoad(MOp):
    """Load one 8x8 tile from threadgroup memory into simdgroup_matrix.

    Two addressing modes:
      A tiles: offset = (sg_offset + tile_offset) * stride + kk_var
      B tiles: offset = kk_var * stride + sg_offset + tile_offset

    When swizzle_bits > 0, uses manual thread_elements() with XOR addressing
    instead of simdgroup_load.
    """

    tile_name: str = ""  # "a_tile" or "b_tile"
    tile_idx: int = 0  # index into the tile array
    src_array: str = ""  # threadgroup array name
    sg_offset: MValue = None  # sg_row (for A) or sg_col (for B)
    tile_offset: int = 0  # mi * 8 (for A) or ni * 8 (for B)
    kk_var: str = "kk"  # inner loop IV name
    stride: int = 0  # threadgroup memory stride
    is_b: bool = False  # True for B tiles (transposed addressing)
    in_type: str = "float"  # element type
    swizzle_bits: int = 0  # >0 enables XOR swizzle
    swizzle_shift: int = 0  # log2(tile_cols) for swizzle

    def result_type(self):
        return None


@dataclass
class MSimdgroupMMA(MOp):
    """simdgroup_multiply_accumulate on 8x8 tiles.

    Generates: simdgroup_multiply_accumulate(acc[mi][ni], a_tile[mi], b_tile[ni], acc[mi][ni])
    """

    acc_name: str = "acc"
    a_tile: str = "a_tile"
    b_tile: str = "b_tile"
    mi: int = 0
    ni: int = 0

    def result_type(self):
        return None


@dataclass
class MSimdgroupBarrierOp(MOp):
    """simdgroup_barrier(mem_flags::mem_none) scheduling hint."""

    def result_type(self):
        return None


@dataclass
class MSimdgroupStore(MOp):
    """Store one 8x8 simdgroup_matrix accumulator to device memory.

    Includes bounds checking: only stores if the 8x8 block is within (M, N).
    """

    acc_name: str = "acc"
    mi: int = 0
    ni: int = 0
    device_ptr: MValue = None
    block_row: MValue = None
    block_col: MValue = None
    sg_row: MValue = None
    sg_col: MValue = None
    mi_offset: int = 0  # mi * 8
    ni_offset: int = 0  # ni * 8
    stride: MValue = None  # N (output leading dimension)
    m_bound: MValue = None  # M
    n_bound: MValue = None  # N
    out_type: str = "float"
    acc_type: str = "float"

    def result_type(self):
        return None


@dataclass
class MAccElemApply(MOp):
    """Element-wise operations on accumulator via thread_elements().

    Same operation format as the old MAccumulatorEpilogue.
    """

    acc_name: str = "acc"
    num_8m: int = 4
    num_8n: int = 4
    operations: list = field(default_factory=list)

    def result_type(self):
        return None


# ---------------------------------------------------------------------------
# Decomposed tensor ops primitives
# ---------------------------------------------------------------------------


@dataclass
class MTensorViewDecl(MOp):
    """Declare Metal tensor views for A, B, C matrices."""

    ptr_a: MValue = None
    ptr_b: MValue = None
    ptr_c: MValue = None
    in_type: str = "float"
    out_type: str = "float"

    def result_type(self):
        return None


@dataclass
class MTileSchedule(MOp):
    """Compute tile coordinates with optional cache-friendly scheduling.

    Patterns: "morton" (2x2 Z-curve), "diagonal", "" (linear).
    """

    pattern: str = "morton"
    block_m: int = 128
    block_n: int = 64

    def result_type(self):
        return None


@dataclass
class MMatmul2dSetup(MOp):
    """Declare matmul2d descriptor, operator, SG assignment, and output slice.

    Covers: descriptor creation, operator instantiation, per-SG tile_row/tile_col,
    and output slice declaration.
    """

    sm: int = 32
    sn: int = 32
    bk: int = 32
    block_m: int = 128
    block_n: int = 64
    wm: int = 2
    wn: int = 2
    relaxed: bool = True
    cooperative: bool = False
    num_sg: int = 4
    in_type: str = "float"
    acc_type: str = "float"
    out_type: str = "float"
    use_separated: bool = False

    def result_type(self):
        return None


@dataclass
class MCoopTensorInit(MOp):
    """Declare and zero-initialize cooperative_tensor output."""

    ct_name: str = "cT"
    acc_type: str = "float"
    in_type: str = "float"
    use_separated: bool = False

    def result_type(self):
        return None


@dataclass
class MCoopTensorLoad(MOp):
    """Load from tensor view slice into cooperative_tensor input."""

    ct_name: str = ""  # "ct_a" or "ct_b"
    tensor_name: str = ""  # "tA" or "tB"
    slice_d0: int = 0  # first template dim
    slice_d1: int = 0  # second template dim
    offset_0: str = ""  # first offset expression
    offset_1: str = ""  # second offset expression

    def result_type(self):
        return None


@dataclass
class MMatmul2dRun(MOp):
    """Execute matmul2d: op.run(inputs, output).

    Two modes:
    - Separated (default): op.run(ct_a, ct_b, ct_out) — uses cooperative_tensor inputs
    - Direct tensor view: op.run(mA, mB, ct_out) — passes tensor slices directly
    """

    ct_a: str = "ct_a"
    ct_b: str = "ct_b"
    ct_out: str = "cT"
    use_tensor_view: bool = False
    # For tensor view mode:
    a_tensor: str = "tA"
    b_tensor: str = "tB"
    a_slice_d0: int = 0
    a_slice_d1: int = 0
    b_slice_d0: int = 0
    b_slice_d1: int = 0
    a_offset_0: str = "k"
    a_offset_1: str = "tile_row"
    b_offset_0: str = "tile_col"
    b_offset_1: str = "k"

    def result_type(self):
        return None


@dataclass
class MCoopTensorEpilogue(MOp):
    """Element-wise operations on cooperative_tensor registers."""

    ct_name: str = "cT"
    operations: list = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class MCoopTensorStore(MOp):
    """Store cooperative_tensor to output tensor view slice."""

    ct_name: str = "cT"
    output_slice: str = "mC"  # name of the output tensor slice

    def result_type(self):
        return None


@dataclass
class MFunction:
    """A Metal IR function."""

    name: str
    params: list[MParam] = field(default_factory=list)
    ops: list[MOp] = field(default_factory=list)
    grid: tuple[int, int, int] = (1, 1, 1)
    threadgroup_size: tuple[int, int, int] = (256, 1, 1)
    kernel_type: str = "elementwise"  # "elementwise" or "gemm"

    def add_op(self, op: MOp, name: str | None = None) -> MValue | None:
        rt = op.result_type() if hasattr(op, "result_type") else None
        if rt is not None:
            val_name = name or f"m{len(self.ops)}"
            val = MValue(val_name, rt, op)
            op.result = val
            self.ops.append(op)
            return val
        else:
            self.ops.append(op)
            return None
