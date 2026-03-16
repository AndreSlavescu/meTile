from __future__ import annotations

from dataclasses import dataclass, field

from metile.ir.types import BOOL, I32, PtrType, ScalarType, TileType


@dataclass
class Value:
    """An SSA value in the Tile IR."""

    name: str
    type: ScalarType | TileType | PtrType
    defining_op: Op | None = field(default=None, repr=False)

    def __repr__(self):
        return f"%{self.name}: {self.type}"


@dataclass
class Op:
    """Base class for Tile IR operations."""

    result: Value | None = field(default=None, init=False)


@dataclass
class ProgramId(Op):
    """Get the program (block) index along an axis."""

    axis: int = 0

    def result_type(self) -> ScalarType:
        return I32


@dataclass
class TileSwizzle(Op):
    """Reorder tile dispatch for cache locality.

    Takes two program_id values (M and N axes) and a scheduling pattern.
    The compiler applies the swizzle during codegen — the returned
    coordinates replace the raw program_ids in the kernel body.

    Patterns:
        "morton"   — 2D Morton (Z-curve) ordering with configurable block_size.
                     Groups nearby tiles into block_size x block_size blocks,
                     maximizing L2 cache reuse of shared A-rows and B-columns.
        "diagonal" — pid_n = (pid_m + pid_n) % grid_n.
        ""         — No swizzle (linear dispatch order).
    """

    pid_m: Value = None
    pid_n: Value = None
    pattern: str = "morton"
    block_size: int = 2


@dataclass
class Constant(Op):
    """A compile-time constant value."""

    value: int | float = 0
    dtype: str = "i32"

    def result_type(self) -> ScalarType:
        return ScalarType(self.dtype)


@dataclass
class Arange(Op):
    """Create a 1D tile of sequential integers [start, start+size)."""

    start: Value = None
    size: int = 0  # compile-time constant (determines tile shape)

    def result_type(self) -> TileType:
        return TileType((self.size,), "i32")


@dataclass
class BinOp(Op):
    """Binary operation on scalars or tiles."""

    op: str = ""  # "add", "mul", "sub", "div", "mod"
    lhs: Value = None
    rhs: Value = None

    def result_type(self):
        # If either operand is a tile, result is a tile
        if isinstance(self.lhs.type, TileType):
            return self.lhs.type
        if isinstance(self.rhs.type, TileType):
            return self.rhs.type
        return self.lhs.type


@dataclass
class Unary(Op):
    """Unary operation on scalars or tiles."""

    op: str = ""  # "exp", "log", "sqrt", "abs", "neg"
    operand: Value = None

    def result_type(self):
        return self.operand.type


@dataclass
class Reduce(Op):
    """Reduce a value across threads in a threadgroup (e.g., sum)."""

    op: str = "sum"  # "sum", "max", "min"
    operand: Value = None

    def result_type(self):
        if isinstance(self.operand.type, TileType):
            return ScalarType(self.operand.type.dtype)
        return self.operand.type


@dataclass
class Select(Op):
    """Ternary select: result = cond ? true_val : false_val."""

    condition: Value = None
    true_val: Value = None
    false_val: Value = None

    def result_type(self):
        return self.true_val.type


@dataclass
class Compare(Op):
    """Comparison operation."""

    predicate: str = ""  # "lt", "le", "gt", "ge", "eq", "ne"
    lhs: Value = None
    rhs: Value = None

    def result_type(self):
        if isinstance(self.lhs.type, TileType):
            return TileType(self.lhs.type.shape, "bool")
        if isinstance(self.rhs.type, TileType):
            return TileType(self.rhs.type.shape, "bool")
        return BOOL


@dataclass
class Load(Op):
    """Load a tile from memory."""

    ptr: Value = None
    offsets: Value = None
    mask: Value | None = None

    def result_type(self):
        assert isinstance(self.ptr.type, PtrType)
        if isinstance(self.offsets.type, TileType):
            return TileType(self.offsets.type.shape, self.ptr.type.dtype)
        # Scalar offset → scalar load
        return ScalarType(self.ptr.type.dtype)


@dataclass
class Store(Op):
    """Store a tile to memory. No result value."""

    ptr: Value = None
    offsets: Value = None
    value: Value = None
    mask: Value | None = None

    def result_type(self):
        return None


@dataclass
class PtrOffset(Op):
    """Pointer arithmetic: ptr + offsets."""

    ptr: Value = None
    offsets: Value = None

    def result_type(self) -> PtrType:
        assert isinstance(self.ptr.type, PtrType)
        return self.ptr.type


@dataclass
class Zeros(Op):
    """Create a tile filled with zeros (accumulator init)."""

    shape: tuple[int, int] = (32, 32)
    dtype: str = "f32"

    def result_type(self) -> TileType:
        return TileType(self.shape, self.dtype)


@dataclass
class Dot(Op):
    """Tile-level matrix multiply-accumulate: result = a @ b + acc."""

    a: Value = None  # tile<BM x BK>
    b: Value = None  # tile<BK x BN>
    acc: Value = None  # tile<BM x BN>

    def result_type(self) -> TileType:
        assert isinstance(self.acc.type, TileType)
        return self.acc.type


@dataclass
class TileLoad(Op):
    """Load a 2D tile from a matrix in device memory."""

    ptr: Value = None
    row_offset: Value = None
    col_offset: Value = None
    stride: Value = None  # leading dimension
    tile_shape: tuple[int, int] = (32, 32)

    def result_type(self) -> TileType:
        assert isinstance(self.ptr.type, PtrType)
        return TileType(self.tile_shape, self.ptr.type.dtype)


@dataclass
class TileStore(Op):
    """Store a 2D tile to a matrix in device memory."""

    ptr: Value = None
    row_offset: Value = None
    col_offset: Value = None
    stride: Value = None
    value: Value = None
    tile_shape: tuple[int, int] = (32, 32)

    def result_type(self):
        return None


@dataclass
class ForRange(Op):
    """A counted for loop with body captured during tracing."""

    start: Value = None
    end: Value = None
    step: int = 1
    iv: Value = None  # induction variable
    body: list[Op] = field(default_factory=list)
    num_stages: int = 1  # software pipeline stages (1 = no pipelining)

    def result_type(self):
        return None


@dataclass
class PersistentRange(Op):
    """Work-stealing loop: each threadgroup atomically grabs tile indices.

    The counter pointer is a device-memory atomic_uint. Each iteration,
    thread 0 does atomic_fetch_add(counter, 1), broadcasts the result
    to the threadgroup, and the body executes if tile_idx < total.
    """

    counter: Value = None  # pointer to atomic uint counter in device memory
    total: int = 0  # total number of work items
    iv: Value = None  # induction variable (tile index per iteration)
    body: list[Op] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class SimdgroupRole(Op):
    """Assign different roles to simdgroups within a threadgroup.

    Partitions the simdgroup grid into roles. Each role gets a contiguous
    range of simdgroup indices. The body executes only for simdgroups
    matching this role.

    This enables simdgroup specialization: e.g., producer SGs load tiles
    while consumer SGs compute MMA, overlapping memory and compute.
    """

    role: int = 0  # which role this block represents
    num_roles: int = 2  # total number of roles
    num_sgs: int = 0  # how many simdgroups assigned to this role (0 = auto)
    body: list[Op] = field(default_factory=list)

    def result_type(self):
        return None


@dataclass
class Param:
    """A function parameter."""

    name: str
    type: ScalarType | TileType | PtrType
    is_output: bool = False


@dataclass
class Function:
    """A Tile IR function (one kernel)."""

    name: str
    params: list[Param] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    constexprs: dict[str, int] = field(default_factory=dict)
    swizzle_pattern: str | None = None  # Set by tile_swizzle(); None = compiler infers
    swizzle_block_size: int = 2

    def add_op(self, op: Op, name: str | None = None) -> Value | None:
        rt = op.result_type() if hasattr(op, "result_type") else None
        if rt is not None:
            val_name = name or f"v{len(self.ops)}"
            val = Value(val_name, rt, op)
            op.result = val
            self.ops.append(op)
            return val
        else:
            self.ops.append(op)
            return None
