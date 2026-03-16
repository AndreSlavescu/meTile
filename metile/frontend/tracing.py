from __future__ import annotations

import threading
from contextlib import contextmanager

from metile.ir import tile_ir as tir
from metile.ir.types import PtrType


class constexpr:
    """Compile-time constant with full expression folding.

    Used both as a type annotation marker (x: constexpr) and as a value
    wrapper that supports arithmetic. Operations between constexprs produce
    constexprs; the folded value is available at trace time.

    Examples:
        BLOCK_M: constexpr = 64
        BLOCK_N: constexpr = 64
        num_stages: constexpr = 3
        tiles = cdiv(M, BLOCK_M)  # also a constexpr if M is constexpr
    """

    def __init__(self, value=None):
        self._value = value

    @property
    def value(self):
        return self._value

    def __repr__(self):
        if self._value is None:
            return "constexpr"
        return f"constexpr({self._value})"

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __bool__(self):
        return bool(self._value)

    def __index__(self):
        return int(self._value)

    # Arithmetic — constexpr op constexpr -> constexpr
    def __add__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value + other._value)
        return self._value + other

    def __radd__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value + self._value)
        return other + self._value

    def __sub__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value - other._value)
        return self._value - other

    def __rsub__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value - self._value)
        return other - self._value

    def __mul__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value * other._value)
        return self._value * other

    def __rmul__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value * self._value)
        return other * self._value

    def __truediv__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value / other._value)
        return self._value / other

    def __rtruediv__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value / self._value)
        return other / self._value

    def __floordiv__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value // other._value)
        return self._value // other

    def __rfloordiv__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value // self._value)
        return other // self._value

    def __mod__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value % other._value)
        return self._value % other

    def __rmod__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value % self._value)
        return other % self._value

    def __pow__(self, other):
        if isinstance(other, constexpr):
            return constexpr(self._value**other._value)
        return self._value**other

    def __rpow__(self, other):
        if isinstance(other, constexpr):
            return constexpr(other._value**self._value)
        return other**self._value

    def __neg__(self):
        return constexpr(-self._value)

    def __pos__(self):
        return constexpr(+self._value)

    def __abs__(self):
        return constexpr(abs(self._value))

    # Bitwise
    def __lshift__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(self._value << v)

    def __rlshift__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(v << self._value)

    def __rshift__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(self._value >> v)

    def __rrshift__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(v >> self._value)

    def __and__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(self._value & v)

    def __rand__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(v & self._value)

    def __or__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(self._value | v)

    def __ror__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(v | self._value)

    def __xor__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(self._value ^ v)

    def __rxor__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return constexpr(v ^ self._value)

    def __invert__(self):
        return constexpr(~self._value)

    # Comparison — returns plain bool for use in Python control flow
    def __lt__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return self._value < v

    def __le__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return self._value <= v

    def __gt__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return self._value > v

    def __ge__(self, other):
        v = other._value if isinstance(other, constexpr) else other
        return self._value >= v

    def __eq__(self, other):
        if other is constexpr or (isinstance(other, type) and issubclass(other, constexpr)):
            # Support `param.annotation is constexpr` check
            return False
        v = other._value if isinstance(other, constexpr) else other
        return self._value == v

    def __ne__(self, other):
        if other is constexpr or (isinstance(other, type) and issubclass(other, constexpr)):
            return True
        v = other._value if isinstance(other, constexpr) else other
        return self._value != v

    def __hash__(self):
        return hash(self._value)


class TensorDescriptor:
    """Tensor descriptor for matmul2d — defines the hardware MMA atom shape.

    Like CuTe's MMA_Atom: specifies the M x N x K dimensions of each
    matmul2d instruction. The compiler tiles the full simdgroup subtile
    across these fragments and generates optimal loading code.

    When fragment dims match the simdgroup subtile (SM, SN), the compiler
    uses ct.load() for bulk loading. When fragments are smaller, it
    generates manual per-element loading with computed thread-to-element
    mappings for optimal memory coalescing.

    Examples:
        TensorDescriptor(32, 32, 32)  — full subtile, ct.load() path
        TensorDescriptor(16, 32, 16)  — MLX-style manual element loading
    """

    def __init__(self, M: int, N: int, K: int):
        self.M = M
        self.N = N
        self.K = K

    def __repr__(self):
        return f"TensorDescriptor({self.M}, {self.N}, {self.K})"

    def __eq__(self, other):
        if isinstance(other, TensorDescriptor):
            return self.M == other.M and self.N == other.N and self.K == other.K
        return NotImplemented

    def __hash__(self):
        return hash((self.M, self.N, self.K))


def cdiv(a, b):
    """Ceiling division: cdiv(a, b) = (a + b - 1) // b.

    Preserves constexpr if both inputs are constexpr.
    """
    a_v = a._value if isinstance(a, constexpr) else a
    b_v = b._value if isinstance(b, constexpr) else b
    result = (a_v + b_v - 1) // b_v
    if isinstance(a, constexpr) and isinstance(b, constexpr):
        return constexpr(result)
    return result


def next_power_of_2(n):
    """Return the smallest power of 2 >= n.

    Preserves constexpr if input is constexpr.
    """
    n_v = n._value if isinstance(n, constexpr) else n
    if n_v <= 0:
        result = 1
    else:
        result = 1
        while result < n_v:
            result <<= 1
    if isinstance(n, constexpr):
        return constexpr(result)
    return result


# Thread-local active tracing context
_active_ctx = threading.local()


def _get_ctx() -> TracingContext:
    ctx = getattr(_active_ctx, "ctx", None)
    if ctx is None:
        raise RuntimeError("This function must be called inside a @metile.kernel")
    return ctx


class TracingContext:
    """Captures operations during kernel tracing into Tile IR."""

    def __init__(self, func_name: str):
        self.func = tir.Function(name=func_name)
        self._counter = 0

    def _next_name(self) -> str:
        name = f"v{self._counter}"
        self._counter += 1
        return name

    def add_op(self, op: tir.Op) -> tir.Value | None:
        return self.func.add_op(op, self._next_name())

    def __enter__(self):
        _active_ctx.ctx = self
        return self

    def __exit__(self, *args):
        _active_ctx.ctx = None


class TracingProxy:
    """Proxy object that records operations into the active TracingContext."""

    def __init__(self, value: tir.Value):
        self._value = value

    # Arithmetic
    def __add__(self, other):
        # Pointer arithmetic: ptr + offsets -> PtrOffset
        if isinstance(self._value.type, PtrType):
            return _ptr_offset(self, other)
        return _binop("add", self, other)

    def __radd__(self, other):
        if isinstance(other, TracingProxy) and isinstance(other._value.type, PtrType):
            return _ptr_offset(other, self)
        return _binop("add", other, self)

    def __sub__(self, other):
        return _binop("sub", self, other)

    def __rsub__(self, other):
        return _binop("sub", other, self)

    def __mul__(self, other):
        return _binop("mul", self, other)

    def __rmul__(self, other):
        return _binop("mul", other, self)

    def __truediv__(self, other):
        return _binop("div", self, other)

    def __rtruediv__(self, other):
        return _binop("div", other, self)

    def __floordiv__(self, other):
        return _binop("div", self, other)

    def __rfloordiv__(self, other):
        return _binop("div", other, self)

    def __mod__(self, other):
        return _binop("mod", self, other)

    def __and__(self, other):
        return _binop("bitand", self, other)

    def __rand__(self, other):
        return _binop("bitand", other, self)

    def __or__(self, other):
        return _binop("bitor", self, other)

    def __ror__(self, other):
        return _binop("bitor", other, self)

    def __xor__(self, other):
        return _binop("bitxor", self, other)

    def __rxor__(self, other):
        return _binop("bitxor", other, self)

    def __lshift__(self, other):
        return _binop("shl", self, other)

    def __rlshift__(self, other):
        return _binop("shl", other, self)

    def __rshift__(self, other):
        return _binop("shr", self, other)

    def __rrshift__(self, other):
        return _binop("shr", other, self)

    # Comparison
    def __lt__(self, other):
        return _compare("lt", self, other)

    def __le__(self, other):
        return _compare("le", self, other)

    def __gt__(self, other):
        return _compare("gt", self, other)

    def __ge__(self, other):
        return _compare("ge", self, other)

    def __eq__(self, other):
        return _compare("eq", self, other)

    def __ne__(self, other):
        return _compare("ne", self, other)

    def __hash__(self):
        return id(self)


def _to_value(x) -> tir.Value:
    """Convert a Python value or TracingProxy to a Tile IR Value."""
    if isinstance(x, TracingProxy):
        return x._value
    if isinstance(x, (int, float)):
        ctx = _get_ctx()
        dtype = "f32" if isinstance(x, float) else "i32"
        op = tir.Constant(value=x, dtype=dtype)
        return ctx.add_op(op)
    raise TypeError(f"Cannot convert {type(x)} to Tile IR value")


def _ptr_offset(ptr_proxy: TracingProxy, offsets_proxy) -> TracingProxy:
    """Create a PtrOffset IR op for pointer arithmetic."""
    ctx = _get_ctx()
    ptr_val = ptr_proxy._value
    offsets_val = _to_value(offsets_proxy)
    op = tir.PtrOffset(ptr=ptr_val, offsets=offsets_val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def _binop(op_name: str, lhs, rhs) -> TracingProxy:
    ctx = _get_ctx()
    lhs_val = _to_value(lhs)
    rhs_val = _to_value(rhs)
    op = tir.BinOp(op=op_name, lhs=lhs_val, rhs=rhs_val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def _compare(pred: str, lhs, rhs) -> TracingProxy:
    ctx = _get_ctx()
    lhs_val = _to_value(lhs)
    rhs_val = _to_value(rhs)
    op = tir.Compare(predicate=pred, lhs=lhs_val, rhs=rhs_val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def _unary(op_name: str, operand) -> TracingProxy:
    ctx = _get_ctx()
    val = _to_value(operand)
    op = tir.Unary(op=op_name, operand=val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def exp(x) -> TracingProxy:
    """Element-wise exponential."""
    return _unary("exp", x)


def log(x) -> TracingProxy:
    """Element-wise natural logarithm."""
    return _unary("log", x)


def sqrt(x) -> TracingProxy:
    """Element-wise square root."""
    return _unary("sqrt", x)


def abs(x) -> TracingProxy:
    """Element-wise absolute value."""
    return _unary("abs", x)


def tanh(x) -> TracingProxy:
    """Element-wise tanh."""
    return _unary("tanh", x)


def maximum(a, b) -> TracingProxy:
    """Element-wise maximum: max(a, b)."""
    return _binop("max", a, b)


def minimum(a, b) -> TracingProxy:
    """Element-wise minimum: min(a, b)."""
    return _binop("min", a, b)


def sum(x) -> TracingProxy:
    """Sum-reduce across threads in a threadgroup."""
    ctx = _get_ctx()
    val = _to_value(x)
    op = tir.Reduce(op="sum", operand=val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def max(x) -> TracingProxy:
    """Max-reduce across threads in a threadgroup."""
    ctx = _get_ctx()
    val = _to_value(x)
    op = tir.Reduce(op="max", operand=val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def min(x) -> TracingProxy:
    """Min-reduce across threads in a threadgroup."""
    ctx = _get_ctx()
    val = _to_value(x)
    op = tir.Reduce(op="min", operand=val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def where(condition, true_val, false_val) -> TracingProxy:
    """Element-wise ternary select: where(cond, x, y) = cond ? x : y."""
    ctx = _get_ctx()
    cond_val = _to_value(condition)
    true_v = _to_value(true_val)
    false_v = _to_value(false_val)
    op = tir.Select(condition=cond_val, true_val=true_v, false_val=false_v)
    result = ctx.add_op(op)
    return TracingProxy(result)


def program_id(axis: int = 0) -> TracingProxy:
    """Get the program (block) index along an axis."""
    ctx = _get_ctx()
    op = tir.ProgramId(axis=axis)
    result = ctx.add_op(op)
    return TracingProxy(result)


def tile_swizzle(
    pid_m: TracingProxy,
    pid_n: TracingProxy,
    pattern: str = "morton",
    block_size: int = 2,
) -> tuple[TracingProxy, TracingProxy]:
    """Reorder tile dispatch for cache-friendly scheduling.

    Takes two program_id values and returns swizzled block coordinates.
    The compiler applies the reordering during codegen.

    If not called, the compiler infers the best pattern automatically
    (morton for GEMM, none for element-wise).

    Args:
        pid_m: program_id for the M (row) axis.
        pid_n: program_id for the N (column) axis.
        pattern: "morton" (2x2 Z-curve blocks), "diagonal", or "" (none).
        block_size: Block size for Morton ordering (default 2).

    Returns:
        (pid_m, pid_n) — same proxies, swizzle is applied in codegen.

    Example::

        pid_m, pid_n = metile.tile_swizzle(
            metile.program_id(0), metile.program_id(1),
            pattern="morton", block_size=2,
        )
    """
    ctx = _get_ctx()
    # Record the swizzle op in IR for visibility
    op = tir.TileSwizzle(
        pid_m=pid_m._value,
        pid_n=pid_n._value,
        pattern=pattern,
        block_size=block_size,
    )
    ctx.func.ops.append(op)
    # Store on function for lowering to read
    ctx.func.swizzle_pattern = pattern
    ctx.func.swizzle_block_size = block_size
    return pid_m, pid_n


def arange(start, end) -> TracingProxy:
    """Create a tile of sequential indices [start, start+size).

    `end` must be a compile-time constant int (the tile size).
    `start` can be a TracingProxy or int.
    """
    ctx = _get_ctx()
    if isinstance(end, int) and isinstance(start, int):
        size = end - start
        start_val = _to_value(start) if start != 0 else None
    elif isinstance(end, int):
        # start is a proxy, end is the total size
        size = end
        start_val = _to_value(start) if not isinstance(start, int) or start != 0 else None
    else:
        raise TypeError("arange end must be a compile-time constant int")

    if isinstance(start, TracingProxy):
        start_val = start._value
    elif isinstance(start, int) and start != 0:
        start_val = _to_value(start)
    else:
        start_val = None

    op = tir.Arange(start=start_val, size=size)
    result = ctx.add_op(op)
    return TracingProxy(result)


def load(ptr, mask=None) -> TracingProxy:
    """Load a tile from memory.

    ptr should be a pointer proxy (result of ptr + offsets).
    """
    ctx = _get_ctx()
    ptr_val = _to_value(ptr)
    mask_val = _to_value(mask) if mask is not None else None

    # The ptr_val should be from a PtrOffset op which carries the offsets
    if ptr_val.defining_op and isinstance(ptr_val.defining_op, tir.PtrOffset):
        offsets_val = ptr_val.defining_op.offsets
    else:
        raise TypeError("load() expects a pointer with offsets (ptr + offsets)")

    op = tir.Load(ptr=ptr_val, offsets=offsets_val, mask=mask_val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def store(ptr, value, mask=None):
    """Store a tile to memory.

    ptr should be a pointer proxy (result of ptr + offsets).
    """
    ctx = _get_ctx()
    ptr_val = _to_value(ptr)
    val = _to_value(value)
    mask_val = _to_value(mask) if mask is not None else None

    if ptr_val.defining_op and isinstance(ptr_val.defining_op, tir.PtrOffset):
        offsets_val = ptr_val.defining_op.offsets
    else:
        raise TypeError("store() expects a pointer with offsets (ptr + offsets)")

    op = tir.Store(ptr=ptr_val, offsets=offsets_val, value=val, mask=mask_val)
    ctx.add_op(op)


def zeros(shape: tuple[int, int], dtype: str = "f32") -> TracingProxy:
    """Create a 2D tile filled with zeros (accumulator init)."""
    ctx = _get_ctx()
    op = tir.Zeros(shape=shape, dtype=dtype)
    result = ctx.add_op(op)
    return TracingProxy(result)


def dot(a, b, acc) -> TracingProxy:
    """Tile-level matrix multiply accumulate: result = a @ b + acc."""
    ctx = _get_ctx()
    a_val = _to_value(a)
    b_val = _to_value(b)
    acc_val = _to_value(acc)
    op = tir.Dot(a=a_val, b=b_val, acc=acc_val)
    result = ctx.add_op(op)
    return TracingProxy(result)


def tile_load(ptr, row_offset, col_offset, stride, shape: tuple[int, int]) -> TracingProxy:
    """Load a 2D tile from a matrix."""
    ctx = _get_ctx()
    ptr_val = _to_value(ptr)
    row_val = _to_value(row_offset)
    col_val = _to_value(col_offset)
    stride_val = _to_value(stride)
    op = tir.TileLoad(
        ptr=ptr_val, row_offset=row_val, col_offset=col_val, stride=stride_val, tile_shape=shape
    )
    result = ctx.add_op(op)
    return TracingProxy(result)


def tile_store(ptr, row_offset, col_offset, stride, value, shape: tuple[int, int]):
    """Store a 2D tile to a matrix."""
    ctx = _get_ctx()
    ptr_val = _to_value(ptr)
    row_val = _to_value(row_offset)
    col_val = _to_value(col_offset)
    stride_val = _to_value(stride)
    val = _to_value(value)
    op = tir.TileStore(
        ptr=ptr_val,
        row_offset=row_val,
        col_offset=col_val,
        stride=stride_val,
        value=val,
        tile_shape=shape,
    )
    ctx.add_op(op)


def shared(size, dtype="f32") -> TracingProxy:
    """Allocate threadgroup (shared) memory.

    Args:
        size: number of elements (must be a compile-time constant)
        dtype: element type (default "f32")

    Returns:
        A proxy representing a pointer to the shared memory allocation.
        Use with load/store just like device pointers.
    """
    ctx = _get_ctx()
    if isinstance(size, constexpr):
        size = size._value
    if not isinstance(size, int):
        raise TypeError("shared() size must be a compile-time constant")
    op = tir.SharedAlloc(size=size, dtype=dtype)
    val = ctx.func.add_op(op, f"shared_{len(ctx.func.ops)}")
    return TracingProxy(val)


def barrier():
    """Threadgroup memory barrier.

    Synchronizes all threads in the threadgroup and ensures all
    threadgroup memory writes are visible to all threads.
    """
    ctx = _get_ctx()
    ctx.func.add_op(tir.Barrier())


def thread_id() -> TracingProxy:
    """Thread position within threadgroup.

    Returns the thread's index within its threadgroup (0 to BLOCK-1).
    This is the Metal `thread_position_in_threadgroup` attribute.
    """
    ctx = _get_ctx()
    op = tir.ThreadId()
    val = ctx.func.add_op(op, f"tid_{len(ctx.func.ops)}")
    return TracingProxy(val)


def simd_shuffle_xor(value, mask) -> TracingProxy:
    """Exchange a value with another thread in the simdgroup.

    Returns the value held by thread (simd_lane_id XOR mask).
    Both threads must call this — it's a symmetric exchange.
    """
    ctx = _get_ctx()
    value_val = _to_value(value) if not isinstance(value, TracingProxy) else value._value
    mask_val = _to_value(mask) if not isinstance(mask, TracingProxy) else mask._value
    op = tir.SimdShuffleXor(value=value_val, mask=mask_val)
    val = ctx.func.add_op(op, f"shfl_{len(ctx.func.ops)}")
    return TracingProxy(val)


def simd_broadcast(value, lane) -> TracingProxy:
    """Broadcast a value from a specific lane to all lanes in the simdgroup."""
    ctx = _get_ctx()
    value_val = _to_value(value) if not isinstance(value, TracingProxy) else value._value
    lane_val = _to_value(lane) if not isinstance(lane, TracingProxy) else lane._value
    op = tir.SimdBroadcast(value=value_val, lane=lane_val)
    val = ctx.func.add_op(op, f"bcast_{len(ctx.func.ops)}")
    return TracingProxy(val)


def simd_lane_id() -> TracingProxy:
    """Thread's lane index within its simdgroup (0-31).

    This is the Metal `thread_index_in_simdgroup` attribute.
    """
    ctx = _get_ctx()
    op = tir.SimdLaneId()
    val = ctx.func.add_op(op, f"slid_{len(ctx.func.ops)}")
    return TracingProxy(val)


def tile_range(start, end, step=1, num_stages=1):
    """Traced for loop. The loop body executes once during tracing.

    Usage:
        for k in metile.tile_range(0, K, BK):
            ...
        for k in metile.tile_range(0, K, BK, num_stages=2):
            ...  # software-pipelined: prefetch next iteration while computing current
    """
    ctx = _get_ctx()
    start_val = _to_value(start)
    end_val = _to_value(end)

    iv_val = tir.Value(ctx._next_name(), tir.I32)

    # Save current ops and redirect to capture list
    saved_ops = ctx.func.ops
    body_ops: list[tir.Op] = []
    ctx.func.ops = body_ops

    yield TracingProxy(iv_val)

    # Restore and wrap captured body
    ctx.func.ops = saved_ops
    for_op = tir.ForRange(
        start=start_val,
        end=end_val,
        step=step,
        iv=iv_val,
        body=body_ops,
        num_stages=num_stages,
    )
    ctx.func.ops.append(for_op)


@contextmanager
def simdgroup_role(role: int, num_roles: int = 2, num_sgs: int = 0):
    """Assign a role to simdgroups for simdgroup specialization.

    Different simdgroups within a threadgroup execute different code paths.

    Args:
        role: Which role this block represents (0, 1, ..., num_roles-1).
        num_roles: Total number of distinct roles.
        num_sgs: How many simdgroups to assign to this role.
                 0 = auto (divide evenly among roles).

    Usage::

        with metile.simdgroup_role(role=0, num_roles=2):
            # Producer: load tiles into shared memory
            ...
        with metile.simdgroup_role(role=1, num_roles=2):
            # Consumer: compute MMA from shared memory
            ...
    """
    ctx = _get_ctx()

    # Save current ops and redirect to capture body
    saved_ops = ctx.func.ops
    body_ops: list[tir.Op] = []
    ctx.func.ops = body_ops

    yield  # body executes here during tracing

    # Restore and wrap captured body
    ctx.func.ops = saved_ops
    role_op = tir.SimdgroupRole(
        role=role,
        num_roles=num_roles,
        num_sgs=num_sgs,
        body=body_ops,
    )
    ctx.func.ops.append(role_op)


def persistent_range(counter_ptr, total):
    """Work-stealing loop for persistent kernels.

    Each threadgroup atomically grabs tile indices from a shared counter.
    The body executes once per tile until all tiles are processed.

    Args:
        counter_ptr: pointer to atomic uint counter in device memory
        total: total number of work items (constexpr int)

    Usage:
        for tile_idx in metile.persistent_range(tile_counter, NUM_TILES):
            ...
    """
    ctx = _get_ctx()
    counter_val = _to_value(counter_ptr)
    total_int = total._value if isinstance(total, constexpr) else int(total)

    iv_val = tir.Value(ctx._next_name(), tir.I32)

    saved_ops = ctx.func.ops
    body_ops: list[tir.Op] = []
    ctx.func.ops = body_ops

    yield TracingProxy(iv_val)

    ctx.func.ops = saved_ops
    persistent_op = tir.PersistentRange(
        counter=counter_val,
        total=total_int,
        iv=iv_val,
        body=body_ops,
    )
    ctx.func.ops.append(persistent_op)
