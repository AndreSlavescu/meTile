from __future__ import annotations

from math import prod


def _is_int(x) -> bool:
    return isinstance(x, int)


def _is_tuple(x) -> bool:
    return isinstance(x, tuple)


def _flatten(x) -> tuple[int, ...]:
    """Flatten an HTuple to a flat tuple of ints."""
    if _is_int(x):
        return (x,)
    return sum((_flatten(xi) for xi in x), ())


def _size(shape) -> int:
    """Product of all elements in an HTuple."""
    if _is_int(shape):
        return shape
    return prod(_size(s) for s in shape)


def _rank(x) -> int:
    """Top-level rank of an HTuple."""
    if _is_int(x):
        return 0
    return len(x)


def _depth(x) -> int:
    """Nesting depth of an HTuple."""
    if _is_int(x):
        return 0
    if len(x) == 0:
        return 1
    return 1 + max(_depth(xi) for xi in x)


def _congruent(a, b) -> bool:
    """Check if two HTuples have the same hierarchical structure."""
    if _is_int(a) and _is_int(b):
        return True
    if _is_tuple(a) and _is_tuple(b):
        if len(a) != len(b):
            return False
        return all(_congruent(ai, bi) for ai, bi in zip(a, b))
    return False


def _compatible(p, s) -> bool:
    """Check if shape p is compatible with shape s (same total size, refinable)."""
    if _is_int(p) and _is_int(s):
        return p == s
    if _is_int(p):
        return p == _size(s)
    if _is_int(s):
        return _size(p) == s
    if len(p) != len(s):
        return False
    return all(_compatible(pi, si) for pi, si in zip(p, s))


def idx2crd(idx: int, shape) -> int | tuple:
    """Convert 1D index to natural coordinate in given shape (colexicographic)."""
    if _is_int(shape):
        return idx % shape
    result = []
    remaining = idx
    for s in shape:
        sz = _size(s)
        sub_idx = remaining % sz
        result.append(idx2crd(sub_idx, s))
        remaining //= sz
    return tuple(result)


def crd2idx(coord, shape) -> int:
    """Convert natural coordinate to 1D index (colexicographic)."""
    if _is_int(coord) and _is_int(shape):
        return coord
    if _is_int(coord):
        return crd2idx(idx2crd(coord, shape), shape)
    result = 0
    stride = 1
    for c, s in zip(coord, shape):
        result += crd2idx(c, s) * stride
        stride *= _size(s)
    return result


def _inner_product(coord, stride) -> int:
    """Compute inner product of coordinate and stride (both HTuples)."""
    if _is_int(coord) and _is_int(stride):
        return coord * stride
    if _is_int(coord):
        # Broadcast scalar coord across tuple stride via idx2crd
        raise ValueError("coord must be at least as structured as stride")
    if _is_int(stride):
        # Collapse coord to scalar
        return (
            crd2idx(coord, tuple(_size(s) for s in coord) if _is_tuple(coord) else coord) * stride
        )
    return sum(_inner_product(c, d) for c, d in zip(coord, stride))


def _prefix_products(shape) -> list[int]:
    """Exclusive prefix product of flattened shape: [1, S0, S0*S1, ...]."""
    flat = _flatten(shape) if _is_tuple(shape) else (shape,)
    result = [1]
    for s in flat:
        result.append(result[-1] * s)
    return result


class Layout:
    """A function from coordinates to offsets: L = Shape : Stride.

    The shape defines the domain (a set of valid coordinates), and the stride
    defines the linear map from natural coordinates to the codomain (offsets).

    Usage:
        L = Layout((4, 8), (1, 4))    # col-major 4x8
        L = Layout((4, 8), (8, 1))    # row-major 4x8
        L = Layout((4, 8))            # default: compact col-major
        L = Layout(32)                 # 1D, stride 1

    Evaluation:
        L(0)       -> 0
        L(5)       -> L(idx2crd(5, shape))
        L((1, 2))  -> inner_product((1, 2), stride) = 1*1 + 2*4 = 9
    """

    __slots__ = ("shape", "stride")

    def __init__(self, shape, stride=None):
        if isinstance(shape, Layout):
            self.shape = shape.shape
            self.stride = shape.stride
            return

        self.shape = shape

        if stride is None:
            # Default: compact column-major (contiguous)
            self.stride = _make_compact_stride(shape)
        else:
            if not _congruent(shape, stride):
                raise ValueError(
                    f"Shape {shape} and stride {stride} must be congruent "
                    f"(same hierarchical structure)"
                )
            self.stride = stride

    # -- Evaluation --

    def __call__(self, coord) -> int:
        """Evaluate layout: coord -> offset."""
        if _is_int(coord):
            if _is_int(self.shape):
                return coord * self.stride
            # Convert 1D index to natural coordinate
            nat = idx2crd(coord, self.shape)
            return _inner_product(nat, self.stride)
        # Multi-dimensional coordinate
        return _inner_product(coord, self.stride)

    # -- Properties --

    @property
    def size(self) -> int:
        """Total number of elements in the domain."""
        return _size(self.shape)

    @property
    def rank(self) -> int:
        """Number of modes (top-level dimensions)."""
        return _rank(self.shape)

    @property
    def depth(self) -> int:
        """Nesting depth of the layout."""
        return _depth(self.shape)

    def sublayout(self, i: int) -> Layout:
        """Get the i-th sublayout (mode)."""
        if _is_int(self.shape):
            raise IndexError("Cannot index a rank-0 layout")
        return Layout(self.shape[i], self.stride[i])

    def __getitem__(self, i) -> Layout:
        return self.sublayout(i)

    # -- Tabulation (useful for debugging) --

    def table(self) -> list[int]:
        """Evaluate the layout at every integer index 0..size-1."""
        return [self(i) for i in range(self.size)]

    def is_injective(self) -> bool:
        """Check if the layout maps distinct coords to distinct offsets."""
        t = self.table()
        return len(set(t)) == len(t)

    def is_compact(self) -> bool:
        """Check if the layout is a bijection onto 0..size-1."""
        t = self.table()
        return sorted(t) == list(range(self.size))

    def cobound(self) -> int:
        """Smallest integer larger than any offset in the image."""
        return max(self.table()) + 1 if self.size > 0 else 0

    # -- Algebra --

    def coalesce(self) -> Layout:
        """Simplify to minimal representation (flatten + merge adjacent modes)."""
        return _coalesce(self)

    def compose(self, other: Layout) -> Layout:
        """Layout composition: self o other. other defines the new domain."""
        return _compose(self, Layout(other) if not isinstance(other, Layout) else other)

    def complement(self, cosize: int | None = None) -> Layout:
        """Complement: layout of elements in codomain not in self's image."""
        if cosize is None:
            cosize = self.cobound()
        return _complement(self.coalesce(), cosize)

    def logical_divide(self, tiler) -> Layout:
        """Split into tile + grid-of-tiles: L / T = L o (T, T*_{|L|})."""
        return _logical_divide(self, tiler)

    def logical_product(self, other: Layout) -> Layout:
        """Product: (self, self* o other)."""
        return _logical_product(self, Layout(other) if not isinstance(other, Layout) else other)

    def right_inverse(self) -> Layout:
        """Right pseudo-inverse: offsets -> coordinates."""
        return _right_inverse(self.coalesce())

    # -- Pretty printing --

    def __repr__(self):
        return f"Layout({self.shape}:{self.stride})"

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return NotImplemented
        return self.shape == other.shape and self.stride == other.stride

    def __hash__(self):
        return hash((self.shape, self.stride))


def _make_compact_stride(shape) -> int | tuple:
    """Generate column-major compact stride for a shape."""
    if _is_int(shape):
        return 1
    # Compute strides for each mode
    strides = []
    cumulative = 1
    for s in shape:
        strides.append(_make_compact_stride_with_base(s, cumulative))
        cumulative *= _size(s)
    return tuple(strides)


def _make_compact_stride_with_base(shape, base: int) -> int | tuple:
    """Generate compact stride starting from a base stride."""
    if _is_int(shape):
        return base
    strides = []
    cumulative = base
    for s in shape:
        strides.append(_make_compact_stride_with_base(s, cumulative))
        cumulative *= _size(s)
    return tuple(strides)


def _coalesce(layout: Layout) -> Layout:
    """Flatten and merge adjacent modes where possible."""
    # Flatten to rank-1 pairs of (size, stride)
    flat_shape = _flatten(layout.shape)
    flat_stride = _flatten(layout.stride)
    pairs = list(zip(flat_shape, flat_stride))

    # Remove size-1 modes
    pairs = [(s, d) for s, d in pairs if s != 1]
    if not pairs:
        return Layout(1, 0)

    # Sort by stride (ascending) for merging
    pairs.sort(key=lambda p: p[1])

    # Merge adjacent modes where stride[i+1] == stride[i] * shape[i]
    merged = [pairs[0]]
    for s, d in pairs[1:]:
        prev_s, prev_d = merged[-1]
        if d == prev_d * prev_s:
            merged[-1] = (prev_s * s, prev_d)
        else:
            merged.append((s, d))

    if len(merged) == 1:
        return Layout(merged[0][0], merged[0][1])
    return Layout(tuple(s for s, _ in merged), tuple(d for _, d in merged))


def _compose(a: Layout, b: Layout) -> Layout:
    """Compose layouts: R = A o B, where R(c) = A(B(c)) for all c in Z(B)."""
    # Distributive: composition distributes over concatenation of B
    if _is_tuple(b.shape) and _rank(b.shape) > 0:
        # Apply composition mode-by-mode
        results = []
        for i in range(len(b.shape)):
            bi = Layout(b.shape[i], b.stride[i])
            results.append(_compose(a, bi))
        return Layout(
            tuple(r.shape for r in results),
            tuple(r.stride for r in results),
        )

    # Base case: b is rank-0 (scalar shape : scalar stride)
    b_shape = b.shape if _is_int(b.shape) else b.shape[0]
    b_stride = b.stride if _is_int(b.stride) else b.stride[0]

    a_coal = a.coalesce()
    return _compose_base(a_coal, b_shape, b_stride)


def _compose_base(a: Layout, s: int, d: int) -> Layout:
    """Base case composition: A o (s : d)."""
    if _is_int(a.shape):
        # Both rank-0
        a_s, a_d = a.shape, a.stride
        # Result shape: s elements starting at offset d, stepping by d through A
        # New stride = a_d * d, but we need to account for wrapping
        new_s = min(s, _ceil_div(a_s, max(d, 1)))
        new_d = a_d * d
        if s > new_s:
            # Need to wrap around — not representable as single mode
            # Fall back to tabulation
            return _compose_fallback_scalar(a, s, d)
        return Layout(new_s, new_d)

    # A is coalesced with rank >= 1
    flat_s = _flatten(a.shape)
    flat_d = _flatten(a.stride)
    prefix = _prefix_products(a.shape)

    result_shapes = []
    result_strides = []

    for r in range(len(flat_s)):
        s_r = flat_s[r]
        d_r = flat_d[r]
        bar_r = prefix[r]

        # Check stride divisibility: bar_r | d or d | bar_r
        if d == 0:
            delta_r = 1
        elif bar_r != 0 and d % bar_r == 0:
            delta_r = d // bar_r
        elif d != 0 and bar_r % d == 0:
            delta_r = 1
        else:
            delta_r = _ceil_div(d, bar_r) if bar_r > 0 else 1

        new_s_r = _ceil_div(s_r, delta_r)
        new_d_r = d_r * delta_r

        if new_s_r > 0:
            result_shapes.append(new_s_r)
            result_strides.append(new_d_r)

    # Trim to match target size s
    total = prod(result_shapes) if result_shapes else 1
    if total > s and result_shapes:
        # Truncate last mode
        result_shapes[-1] = max(1, s // (total // result_shapes[-1]))

    # Pad if needed
    total = prod(result_shapes) if result_shapes else 1
    if total < s and result_shapes:
        ratio = _ceil_div(s, total)
        result_shapes.append(ratio)
        result_strides.append(flat_d[-1] * flat_s[-1] if flat_d else 0)

    if len(result_shapes) == 1:
        return Layout(result_shapes[0], result_strides[0])
    return Layout(tuple(result_shapes), tuple(result_strides))


def _compose_fallback_scalar(a: Layout, s: int, d: int) -> Layout:
    """Fallback: evaluate composition by tabulation for small layouts."""
    offsets = [a(i * d) for i in range(s)]
    # Try to detect a single stride pattern
    if s <= 1:
        return Layout(s, 0)
    stride = offsets[1] - offsets[0] if s > 1 else 0
    if all(offsets[i] == offsets[0] + i * stride for i in range(s)):
        return Layout(s, stride)
    # Cannot represent as a single Layout — return as tabulated
    # For now, return best-effort
    return Layout(s, stride)


def _complement(layout: Layout, cosize: int) -> Layout:
    """Compute complement layout: elements in [0, cosize) not in image of layout.

    The complement L* satisfies:
      - For all b in Z(L), a in Z(L*)/{0}: L(b) != L*(a)
      - L*(a-1) < L*(a) (ordered)
    """
    if layout.size == 0:
        return Layout(cosize, 1)

    coal = layout.coalesce()
    flat_s = _flatten(coal.shape)
    flat_d = _flatten(coal.stride)

    # Sort by stride
    pairs = sorted(zip(flat_d, flat_s), key=lambda p: abs(p[0]))

    result_shapes = []
    result_strides = []
    current = 1  # tracks the "covered" region

    for d, s in pairs:
        if d == 0:
            continue
        abs_d = abs(d)
        if abs_d > current:
            # Gap between current coverage and this stride
            gap = abs_d // current
            result_shapes.append(gap)
            result_strides.append(current)
            current = abs_d
        current *= s

    # Final complement to fill up to cosize
    if current < cosize:
        remaining = _ceil_div(cosize, current)
        result_shapes.append(remaining)
        result_strides.append(current)

    if not result_shapes:
        return Layout(1, 0)
    if len(result_shapes) == 1:
        return Layout(result_shapes[0], result_strides[0])
    return Layout(tuple(result_shapes), tuple(result_strides))


def _logical_divide(layout: Layout, tiler) -> Layout:
    """Logical divide: L / T = L o (T, T*_{|L|}).

    Splits layout into:
      - Mode 0: the "tile" (what T hits)
      - Mode 1: the "rest" / grid of tiles (what T misses)
    """
    if isinstance(tiler, int) or not isinstance(tiler, Layout):
        tiler = Layout(tiler)

    comp = tiler.complement(layout.size)
    # Compose layout with the concatenation (tiler, complement)
    tile_part = layout.compose(tiler)
    rest_part = layout.compose(comp)
    return Layout((tile_part.shape, rest_part.shape), (tile_part.stride, rest_part.stride))


def _logical_product(a: Layout, b: Layout) -> Layout:
    """Logical product: (A, A* o B).

    Mode 0 is the original layout A, mode 1 is A's complement composed with B.
    """
    comp = a.complement()
    rest = comp.compose(b) if comp.size > 1 else b
    return Layout((a.shape, rest.shape), (a.stride, rest.stride))


def _right_inverse(layout: Layout) -> Layout:
    """Right pseudo-inverse: offsets -> coordinates.

    For injective layout L, L^dag satisfies L(L^dag(k)) = k for k in image(L).
    """
    coal = layout.coalesce()
    flat_s = _flatten(coal.shape)
    flat_d = _flatten(coal.stride)

    # For each mode with stride d_i, the inverse maps offset -> coordinate
    # by dividing by d_i and taking mod s_i
    # Build inverse: for offset o, coordinate_i = (o // d_i) % s_i
    # This is itself a Layout with shape = cobound, stride = inverse mapping

    # Simple case: build by sorting modes by stride and constructing inverse
    pairs = sorted(zip(flat_d, flat_s), key=lambda p: abs(p[0]))

    inv_shapes = []
    inv_strides = []
    current = 1

    for d, s in pairs:
        if d == 0:
            continue
        abs_d = abs(d)
        if abs_d > current:
            # Gap: elements mapped to same coordinate (broadcast)
            gap = abs_d // current
            inv_shapes.append(gap)
            inv_strides.append(0)  # stride 0 = broadcast
            current = abs_d
        inv_shapes.append(s)
        inv_strides.append(current)
        current *= s

    if not inv_shapes:
        return Layout(1, 0)
    if len(inv_shapes) == 1:
        return Layout(inv_shapes[0], inv_strides[0])
    return Layout(tuple(inv_shapes), tuple(inv_strides))


def make_layout(shape, stride=None) -> Layout:
    """Create a layout from shape and optional stride."""
    return Layout(shape, stride)


def col_major(M: int, N: int) -> Layout:
    """Column-major layout for M x N matrix."""
    return Layout((M, N), (1, M))


def row_major(M: int, N: int) -> Layout:
    """Row-major layout for M x N matrix."""
    return Layout((M, N), (N, 1))


def make_identity(shape) -> Layout:
    """Identity layout: compact column-major."""
    return Layout(shape)


def simdgroup_layout_8x8() -> Layout:
    """Layout for Apple's 8x8 simdgroup_matrix.

    32 threads in a simdgroup, each holding 2 elements of an 8x8 matrix.
    Thread-value layout: ((4, 8), 2) : ((16, 1), 8)
    - Thread mode: 32 threads mapped as (4 rows, 8 cols)
    - Value mode: 2 elements per thread (strided by 8)

    This matches Apple's simdgroup_matrix<float, 8, 8> hardware layout.
    """
    return Layout(((4, 8), 2), ((16, 1), 8))


def threadgroup_tile(BM: int, BN: int, num_threads: int = 256) -> Layout:
    """Layout for cooperative loading of a BM x BN tile by num_threads.

    Distributes elements across threads in a row-major pattern for
    coalesced memory access.
    """
    elems_per_thread = (BM * BN + num_threads - 1) // num_threads
    return Layout(
        (num_threads, elems_per_thread),
        (elems_per_thread, 1),
    )


class Swizzle:
    """XOR-based address permutation for bank-conflict-free shared memory.

    CuTe's Swizzle<B, M, S> XORs B bits starting at position M with B bits
    starting at position M+S:

        swizzle(x) = x ^ ((x >> S) & mask)
        where mask = ((1 << B) - 1) << M

    Self-inverse: swizzle(swizzle(x)) = x, since XOR is its own inverse.

    Parameters:
        bits (B): number of bits to swizzle
        base (M): starting bit position (usually 0 for element-level)
        shift (S): distance between the two bit groups being XORed

    For shared memory with 32 banks (5-bit bank index), stride = 2^S:
        Swizzle(3, 0, 3) — 8-element stride, XOR 3 bits
        Swizzle(5, 0, 5) — 32-element stride, XOR 5 bits (full bank coverage)

    Apple GPU note: This permutation cannot be used with simdgroup_load(),
    which takes a uniform (base, stride) and computes per-thread addresses
    internally. Swizzle requires per-element address control (like CUDA's
    ldmatrix). On Apple GPU, shared memory bank conflicts are resolved via
    stride padding instead. The Swizzle class is provided for layout algebra
    completeness, bank conflict analysis, and future per-thread load paths.
    """

    __slots__ = ("_mask", "base", "bits", "shift")

    def __init__(self, bits: int, base: int, shift: int):
        if bits < 0 or base < 0 or shift < 0:
            raise ValueError("Swizzle parameters must be non-negative")
        if bits == 0:
            raise ValueError("Swizzle with 0 bits is identity — use no swizzle")
        self.bits = bits
        self.base = base
        self.shift = shift
        self._mask = ((1 << bits) - 1) << base

    def __call__(self, offset: int) -> int:
        """Apply swizzle: offset -> XOR-permuted offset."""
        return offset ^ ((offset >> self.shift) & self._mask)

    @property
    def mask(self) -> int:
        """Bitmask for the swizzled bit positions."""
        return self._mask

    def __repr__(self):
        return f"Swizzle<{self.bits},{self.base},{self.shift}>"

    def __eq__(self, other):
        if not isinstance(other, Swizzle):
            return NotImplemented
        return self.bits == other.bits and self.base == other.base and self.shift == other.shift

    def __hash__(self):
        return hash((self.bits, self.base, self.shift))


def make_swizzle(tile_cols: int, num_banks: int = 32) -> Swizzle | None:
    """Compute optimal swizzle for a shared memory tile with given column count.

    The swizzle XORs column-address bits with row-address bits to distribute
    accesses across banks. Requires tile_cols to be a power of 2 (the stride
    must align with bit boundaries for XOR to work cleanly).

    The constraint B <= S (= log2(tile_cols)) ensures the source and
    destination bit groups don't overlap, which is required for the XOR
    to be self-inverse. When tile_cols < num_banks, B < log2(num_banks)
    and the swizzle reduces but cannot fully eliminate bank conflicts.

    Examples:
        tile_cols=32, num_banks=32: Swizzle(5, 0, 5) — conflict-free
        tile_cols=16, num_banks=32: Swizzle(4, 0, 4) — reduces to 2-way
        tile_cols=8,  num_banks=32: Swizzle(3, 0, 3) — reduces to 4-way

    Returns None if tile_cols is not a power of 2.
    """
    if tile_cols <= 1 or (tile_cols & (tile_cols - 1)) != 0:
        return None  # not power of 2

    col_bits = tile_cols.bit_length() - 1  # log2(tile_cols)
    bank_bits = num_banks.bit_length() - 1 if num_banks > 1 else 0
    # B <= S ensures non-overlapping groups (self-inverse property)
    B = min(col_bits, bank_bits)

    return Swizzle(B, 0, col_bits)


def bank_conflicts(
    layout: Layout,
    num_banks: int = 32,
    group_size: int = 32,
    swizzle: Swizzle | None = None,
) -> dict:
    """Analyze bank conflicts for a shared memory access pattern.

    Simulates group_size threads (e.g. 32 = one simdgroup) accessing
    consecutive elements of the layout. Computes which bank each access
    hits and counts conflicts (multiple threads hitting the same bank
    in the same cycle).

    Args:
        layout: The shared memory layout being accessed.
        num_banks: Number of memory banks (32 on Apple GPU, 4 bytes each).
        group_size: Threads accessing simultaneously (32 = simdgroup width).
        swizzle: Optional swizzle permutation applied to offsets.

    Returns:
        Dictionary with:
        - total_conflicts: sum of (max_per_bank - 1) across all groups
        - max_way: worst-case N-way conflict (1 = no conflict)
        - groups: number of access groups analyzed
        - conflict_free: True if no bank conflicts at all
    """
    table = layout.table()
    if swizzle is not None:
        table = [swizzle(offset) for offset in table]

    total_conflicts = 0
    max_way = 1

    for start in range(0, len(table), group_size):
        group = table[start : start + group_size]
        bank_counts: dict[int, int] = {}
        for offset in group:
            bank = offset % num_banks
            bank_counts[bank] = bank_counts.get(bank, 0) + 1
        worst = max(bank_counts.values()) if bank_counts else 1
        total_conflicts += worst - 1
        max_way = max(max_way, worst)

    num_groups = _ceil_div(len(table), group_size)
    return {
        "total_conflicts": total_conflicts,
        "max_way": max_way,
        "groups": num_groups,
        "conflict_free": total_conflicts == 0,
    }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b if b > 0 else 0


def print_layout(layout: Layout, name: str = ""):
    """Pretty-print a layout with its evaluation table."""
    prefix = f"{name} = " if name else ""
    print(f"{prefix}{layout}")
    print(f"  size={layout.size}, rank={layout.rank}, depth={layout.depth}")
    if layout.size <= 64:
        print(f"  table={layout.table()}")
    if layout.rank > 0 and _is_tuple(layout.shape) and len(layout.shape) == 2:
        s0 = _size(layout.shape[0])
        s1 = _size(layout.shape[1])
        if s0 * s1 <= 64:
            print(f"  2D view ({s0} x {s1}):")
            for r in range(s0):
                row = [layout((r, c)) for c in range(s1)]
                print(f"    {row}")
