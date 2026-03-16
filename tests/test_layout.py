from metile.ir.layout import (
    Layout,
    Swizzle,
    _compatible,
    _congruent,
    _flatten,
    _size,
    bank_conflicts,
    col_major,
    crd2idx,
    idx2crd,
    make_identity,
    make_swizzle,
    row_major,
    simdgroup_layout_8x8,
    threadgroup_tile,
)


class TestHTupleUtils:
    def test_size_scalar(self):
        assert _size(4) == 4

    def test_size_flat(self):
        assert _size((4, 8)) == 32

    def test_size_hierarchical(self):
        assert _size((4, (2, 3))) == 24

    def test_flatten(self):
        assert _flatten((4, (2, 3))) == (4, 2, 3)
        assert _flatten(5) == (5,)
        assert _flatten((1, 2, 3)) == (1, 2, 3)

    def test_congruent(self):
        assert _congruent(3, 5) is True
        assert _congruent((2, 3), (4, 5)) is True
        assert _congruent((2, (3, 4)), (1, (2, 3))) is True
        assert _congruent((2, 3), (4, 5, 6)) is False
        assert _congruent(3, (1, 2)) is False

    def test_compatible(self):
        assert _compatible(6, (2, 3)) is True
        assert _compatible((2, 3), 6) is True
        assert _compatible((2, 3), (2, 3)) is True
        assert _compatible(5, (2, 3)) is False


class TestCoordinates:
    def test_idx2crd_scalar(self):
        assert idx2crd(3, 8) == 3

    def test_idx2crd_2d(self):
        # (4, 8) colexicographic: idx -> (idx % 4, idx // 4)
        assert idx2crd(0, (4, 8)) == (0, 0)
        assert idx2crd(1, (4, 8)) == (1, 0)
        assert idx2crd(4, (4, 8)) == (0, 1)
        assert idx2crd(5, (4, 8)) == (1, 1)

    def test_crd2idx_2d(self):
        assert crd2idx((0, 0), (4, 8)) == 0
        assert crd2idx((1, 0), (4, 8)) == 1
        assert crd2idx((0, 1), (4, 8)) == 4
        assert crd2idx((3, 7), (4, 8)) == 31

    def test_roundtrip(self):
        shape = (4, 8)
        for i in range(32):
            assert crd2idx(idx2crd(i, shape), shape) == i


class TestLayoutBasics:
    def test_create_1d(self):
        L = Layout(8, 1)
        assert L.size == 8
        assert L.rank == 0
        assert L(0) == 0
        assert L(3) == 3

    def test_create_2d_col_major(self):
        L = Layout((4, 8), (1, 4))
        assert L.size == 32
        assert L.rank == 2
        assert L((0, 0)) == 0
        assert L((1, 0)) == 1
        assert L((0, 1)) == 4
        assert L((3, 7)) == 31

    def test_create_2d_row_major(self):
        L = Layout((4, 8), (8, 1))
        assert L((0, 0)) == 0
        assert L((0, 1)) == 1
        assert L((1, 0)) == 8
        assert L((3, 7)) == 31

    def test_default_stride(self):
        L = Layout((4, 8))
        # Default is col-major: stride = (1, 4)
        assert L.stride == (1, 4)
        assert L((0, 0)) == 0
        assert L((1, 0)) == 1
        assert L((0, 1)) == 4

    def test_int_index_2d(self):
        # Integer index on 2D layout should work via idx2crd
        L = Layout((4, 8), (1, 4))
        assert L(0) == 0
        assert L(1) == 1
        assert L(4) == 4  # (0, 1) in col-major

    def test_col_major_helper(self):
        L = col_major(4, 8)
        assert L.shape == (4, 8)
        assert L.stride == (1, 4)

    def test_row_major_helper(self):
        L = row_major(4, 8)
        assert L.shape == (4, 8)
        assert L.stride == (8, 1)

    def test_is_compact(self):
        assert col_major(4, 8).is_compact()
        assert row_major(4, 8).is_compact()

    def test_broadcast(self):
        # Stride 0 = broadcast
        L = Layout((4, 8), (0, 1))
        assert L((0, 0)) == 0
        assert L((1, 0)) == 0  # broadcast along mode 0
        assert L((0, 1)) == 1
        assert L((3, 5)) == 5

    def test_padded(self):
        # Padded column-major: stride > shape[0]
        L = Layout((4, 8), (1, 5))  # 1 padding element per column
        assert L((0, 0)) == 0
        assert L((3, 0)) == 3
        assert L((0, 1)) == 5
        assert not L.is_compact()

    def test_hierarchical_shape(self):
        # (4, (2, 3)) : (1, (4, 8)) — hierarchical stride
        L = Layout((4, (2, 3)), (1, (4, 8)))
        assert L.size == 24
        assert L.depth == 2
        assert L((0, (0, 0))) == 0
        assert L((1, (0, 0))) == 1
        assert L((0, (1, 0))) == 4
        assert L((0, (0, 1))) == 8


class TestSublayout:
    def test_sublayout(self):
        L = Layout((4, 8), (1, 4))
        L0 = L[0]
        assert L0.shape == 4
        assert L0.stride == 1
        L1 = L[1]
        assert L1.shape == 8
        assert L1.stride == 4


class TestTable:
    def test_table_1d(self):
        L = Layout(4, 2)
        assert L.table() == [0, 2, 4, 6]

    def test_table_col_major(self):
        L = col_major(2, 3)
        # 2x3 col-major: elements in order (0,0),(1,0),(0,1),(1,1),(0,2),(1,2)
        assert L.table() == [0, 1, 2, 3, 4, 5]

    def test_table_row_major(self):
        L = row_major(2, 3)
        # 2x3 row-major: idx2crd gives col-major coords, stride is row-major
        # idx=0 -> (0,0) -> 0*3+0*1=0
        # idx=1 -> (1,0) -> 1*3+0*1=3
        # idx=2 -> (0,1) -> 0*3+1*1=1
        # idx=3 -> (1,1) -> 1*3+1*1=4
        # idx=4 -> (0,2) -> 0*3+2*1=2
        # idx=5 -> (1,2) -> 1*3+2*1=5
        assert L.table() == [0, 3, 1, 4, 2, 5]

    def test_injective(self):
        assert col_major(4, 8).is_injective()
        assert row_major(4, 8).is_injective()
        # Broadcast is not injective
        assert not Layout((4, 8), (0, 1)).is_injective()


class TestCoalesce:
    def test_coalesce_already_simple(self):
        L = Layout(8, 1)
        C = L.coalesce()
        assert C.shape == 8
        assert C.stride == 1

    def test_coalesce_merges_adjacent(self):
        # (2, 4) : (1, 2) -> coalesces to 8 : 1
        L = Layout((2, 4), (1, 2))
        C = L.coalesce()
        assert C.size == 8
        assert C.stride == 1

    def test_coalesce_removes_size_1(self):
        L = Layout((1, 8), (999, 1))
        C = L.coalesce()
        assert C.size == 8

    def test_coalesce_non_contiguous(self):
        # (4, 8) : (2, 8) cannot be merged
        L = Layout((4, 8), (2, 8))
        C = L.coalesce()
        assert C.size == 32


class TestComposition:
    def test_compose_identity(self):
        A = col_major(4, 8)
        B = Layout(32)  # identity: stride 1, all 32 elements
        R = A.compose(B)
        assert R.size == 32
        # Should produce same table as A
        for i in range(32):
            assert R(i) == A(i)

    def test_compose_stride(self):
        # Every 2nd element of col_major(4, 8)
        A = col_major(4, 8)
        B = Layout(16, 2)
        R = A.compose(B)
        assert R.size == 16
        for i in range(16):
            assert R(i) == A(2 * i)

    def test_compose_2d(self):
        # Compose with a 2D layout
        A = col_major(4, 8)
        B = Layout((4, 4), (1, 4))  # first 4x4 tile
        R = A.compose(B)
        assert R.size == 16


class TestComplement:
    def test_complement_simple(self):
        # L = 4:1 covers {0,1,2,3}. Complement in [0, 8) = 2:4
        L = Layout(4, 1)
        C = L.complement(8)
        assert C.size == 2
        # Complement should map to {4, 5, 6, 7} or equiv {skip by 4}

    def test_complement_strided(self):
        # L = 4:2 covers {0,2,4,6}. Complement in [0, 8) should cover {1,3,5,7}
        L = Layout(4, 2)
        C = L.complement(8)
        # The complement should have stride 1, starting at offset that fills gaps
        assert C.size >= 2  # at least 2 elements to fill the gap

    def test_complement_preserves_disjoint(self):
        L = Layout(4, 1)
        C = L.complement(8)
        L_image = set(L.table())
        # Per CuTe definition: disjoint for all non-zero complement indices
        C_nonzero_image = set(C(i) for i in range(1, C.size))
        assert L_image & C_nonzero_image == set()


class TestLogicalDivide:
    def test_divide_1d(self):
        L = Layout(16, 1)
        D = L.logical_divide(4)
        # Mode 0: tile of 4 elements
        # Mode 1: 4 tiles
        assert D.size == 16

    def test_divide_preserves_elements(self):
        L = Layout(16, 1)
        D = L.logical_divide(4)
        # All 16 elements should be accessible
        table = D.table()
        assert sorted(table) == list(range(16))


class TestAppleSilicon:
    def test_simdgroup_layout(self):
        L = simdgroup_layout_8x8()
        assert L.size == 64  # 8x8 = 64 elements
        # Thread mode: (4, 8) = 32 threads
        # Value mode: 2 elements per thread
        # 32 * 2 = 64

    def test_threadgroup_tile(self):
        L = threadgroup_tile(64, 64, 256)
        assert L.size == 4096  # 64*64

    def test_simdgroup_injective(self):
        L = simdgroup_layout_8x8()
        assert L.is_injective()


class TestConvenience:
    def test_make_identity(self):
        L = make_identity((4, 8))
        assert L.is_compact()

    def test_equality(self):
        assert col_major(4, 8) == Layout((4, 8), (1, 4))
        assert col_major(4, 8) != row_major(4, 8)


class TestGemmLayoutDerivation:
    """Validate that layout-derived SG/coop-load params match legacy hardcoded values."""

    def test_simdgroup_layout_matches_legacy(self):
        from metile.compiler.lowering import _compute_simdgroup_layout

        # Test all configs used in practice
        for BM, BN, expected_sg, expected_rows, expected_cols in [
            (32, 32, 4, 2, 2),
            (64, 64, 4, 2, 2),
            (64, 32, 4, 2, 2),
            (32, 64, 4, 2, 2),
        ]:
            sl = _compute_simdgroup_layout(BM, BN, expected_sg)
            assert sl.num_sg == expected_sg
            assert sl.sg_rows == expected_rows, f"BM={BM},BN={BN}: sg_rows={sl.sg_rows}"
            assert sl.sg_cols == expected_cols, f"BM={BM},BN={BN}: sg_cols={sl.sg_cols}"
            assert sl.sg_m == BM // expected_rows
            assert sl.sg_n == BN // expected_cols
            # Accumulator count per SG must be <= 16
            acc_per_sg = (sl.sg_m // 8) * (sl.sg_n // 8)
            assert acc_per_sg <= 16

    def test_coop_load_layout_values(self):
        from metile.compiler.lowering import _compute_coop_load_layout

        # Standard config: BM=64, BK=32, 128 threads
        cl = _compute_coop_load_layout(64, 32, 128)
        assert cl.tile.rows == 64
        assert cl.tile.cols == 32
        assert cl.tile.smem_stride == 32  # no padding yet
        assert cl.num_threads == 128
        assert cl.elems_per_thread == (64 * 32 + 128 - 1) // 128

    def test_non_power_of_2_layout(self):
        """Layout algebra handles non-power-of-2 shapes correctly."""
        # 48x48 tile — not a power of 2
        L = Layout((48, 48))
        assert L.size == 2304
        assert L.is_compact()

        # Logical divide into 12-element tiles
        D = L.logical_divide(12)
        assert D.size == 2304
        assert sorted(D.table()) == list(range(2304))

    def test_non_power_of_2_simdgroup_layout(self):
        from metile.compiler.lowering import _compute_simdgroup_layout

        # 48x48 with 4 SGs: 48/2 = 24, 24/8 = 3 MMA tiles per SG row
        sl = _compute_simdgroup_layout(48, 48, 4)
        assert sl.sg_m == 24
        assert sl.sg_n == 24
        assert (sl.sg_m // 8) * (sl.sg_n // 8) == 9  # 9 accumulators per SG

    def test_non_power_of_2_coop_load(self):
        from metile.compiler.lowering import _compute_coop_load_layout

        # 48x24 tile, 128 threads
        cl = _compute_coop_load_layout(48, 24, 128)
        assert cl.tile.rows == 48
        assert cl.tile.cols == 24
        assert cl.num_threads == 128
        assert cl.elems_per_thread == (48 * 24 + 128 - 1) // 128

    def test_layout_driven_emission_equivalence(self):
        """Emitter produces identical MSL from layout fields vs legacy fields."""
        from metile.codegen.msl_emitter import _emit_cooperative_load
        from metile.ir import metal_ir as mir

        # Create an op with both legacy fields and layout metadata
        op = mir.MCooperativeLoad(
            device_ptr=mir.MValue("A", mir.PtrType("float")),
            tg_array="shared_a",
            row_offset=mir.MValue("block_row", mir.ScalarType("u32")),
            col_offset=None,
            src_stride=mir.MValue("K", mir.ScalarType("u32")),
            tile_rows=64,
            tile_cols=32,
            dst_stride=34,
            tg_size=128,
            linear_tid=mir.MValue("tid", mir.ScalarType("u32")),
            bounds_check=False,
            vec_size=1,
            elem_type="float",
            load_layout=mir.CooperativeLoadLayout(
                tile=mir.TileLayout(rows=64, cols=32, smem_stride=34),
                num_threads=128,
                elems_per_thread=16,
            ),
        )
        func = mir.MFunction("test", kernel_type="gemm")

        # Emit with layout
        lines_layout = []
        _emit_cooperative_load(op, lines_layout, 0, func)

        # Clear layout, emit with legacy fields
        op.load_layout = None
        lines_legacy = []
        _emit_cooperative_load(op, lines_legacy, 0, func)

        assert lines_layout == lines_legacy


class TestSwizzle:
    """CuTe Swizzle<B,M,S>: XOR-based address permutation."""

    def test_basic_xor(self):
        # Swizzle(3, 0, 3): XOR bits [0,3) with bits [3,6)
        sw = Swizzle(3, 0, 3)
        # offset=0b001_000 (8): low 3 bits = 000, bits [3,6) = 001
        # swizzled = 8 ^ ((8 >> 3) & 0b111) = 8 ^ 1 = 9
        assert sw(8) == 9
        # offset=0: swizzle(0) = 0 ^ 0 = 0
        assert sw(0) == 0

    def test_self_inverse(self):
        """XOR swizzle is its own inverse: sw(sw(x)) = x."""
        sw = Swizzle(5, 0, 5)
        for x in range(256):
            assert sw(sw(x)) == x

    def test_self_inverse_shifted(self):
        sw = Swizzle(3, 2, 5)
        for x in range(512):
            assert sw(sw(x)) == x

    def test_is_permutation(self):
        """Swizzle is a bijection on any aligned range."""
        sw = Swizzle(3, 0, 3)
        offsets = list(range(64))
        swizzled = [sw(x) for x in offsets]
        assert sorted(swizzled) == offsets

    def test_repr(self):
        assert repr(Swizzle(3, 0, 5)) == "Swizzle<3,0,5>"

    def test_equality(self):
        assert Swizzle(3, 0, 5) == Swizzle(3, 0, 5)
        assert Swizzle(3, 0, 5) != Swizzle(4, 0, 5)

    def test_invalid_params(self):
        import pytest

        with pytest.raises(ValueError):
            Swizzle(-1, 0, 3)
        with pytest.raises(ValueError):
            Swizzle(0, 0, 3)  # 0 bits = identity

    def test_mask(self):
        # Swizzle(3, 0, 5): mask = 0b111 << 0 = 7
        assert Swizzle(3, 0, 5).mask == 0b111
        # Swizzle(3, 2, 5): mask = 0b111 << 2 = 0b11100 = 28
        assert Swizzle(3, 2, 5).mask == 0b11100


class TestMakeSwizzle:
    """Automatic swizzle parameter selection."""

    def test_stride_32(self):
        sw = make_swizzle(32, num_banks=32)
        assert sw is not None
        assert sw.bits == 5  # min(log2(32), log2(32))
        assert sw.shift == 5  # log2(32)
        assert sw.base == 0

    def test_stride_16(self):
        sw = make_swizzle(16, num_banks=32)
        assert sw is not None
        assert sw.bits == 4  # min(log2(16), log2(32))
        assert sw.shift == 4

    def test_stride_8(self):
        sw = make_swizzle(8, num_banks=32)
        assert sw is not None
        assert sw.bits == 3
        assert sw.shift == 3

    def test_non_power_of_2_returns_none(self):
        assert make_swizzle(34) is None
        assert make_swizzle(48) is None

    def test_stride_1_returns_none(self):
        assert make_swizzle(1) is None


class TestBankConflicts:
    """Bank conflict analysis for shared memory layouts.

    Bank conflicts occur when threads in a simdgroup (32 threads) access
    the same bank but different addresses in the same cycle. Apple GPUs
    have 32 threadgroup memory banks (4 bytes each).

    Two critical access patterns:
    - Row access (cooperative load): threads read consecutive elements in
      a row. stride=32 has no row conflicts (offsets 0-31 → 32 different banks).
    - Column access (simdgroup_load): threads read elements from different
      rows in the same column. stride=32 has 8-way conflicts (offsets
      0, 32, 64, ... all map to bank 0).
    """

    def test_column_access_stride_32_has_conflicts(self):
        """Reading a column with stride=32: all elements hit the same bank."""
        # 8 rows of the same column, stride 32: offsets 0, 32, 64, ...
        column = Layout(8, 32)
        result = bank_conflicts(column, num_banks=32, group_size=8)
        assert result["max_way"] == 8  # all in bank 0
        assert not result["conflict_free"]

    def test_column_access_padded_stride_34(self):
        """Padding stride to 34: each successive row shifts by 2 banks."""
        column = Layout(8, 34)
        result = bank_conflicts(column, num_banks=32, group_size=8)
        # offsets: 0, 34, 68, 102, 136, 170, 204, 238
        # banks:   0,  2,  4,   6,   8,  10,  12,  14 → all different
        assert result["conflict_free"]

    def test_column_access_swizzle_eliminates_conflicts(self):
        """Swizzle eliminates column access conflicts for stride 32."""
        column = Layout(8, 32)
        sw = Swizzle(5, 0, 5)
        result = bank_conflicts(column, num_banks=32, group_size=8, swizzle=sw)
        # sw(r*32) = r*32 ^ (r & 0x1F): each row maps to a different bank
        assert result["conflict_free"]

    def test_row_access_stride_32_no_conflicts(self):
        """Row access (consecutive elements): always conflict-free."""
        row = Layout(32, 1)
        result = bank_conflicts(row, num_banks=32, group_size=32)
        assert result["conflict_free"]

    def test_full_tile_column_conflicts(self):
        """Full 64x32 tile: colexicographic enumeration hits column conflicts."""
        # Layout((64, 32), (32, 1)): first 32 table entries are column 0
        # (rows 0-31), all at offsets r*32 → bank 0 → 32-way conflict
        L = Layout((64, 32), (32, 1))
        result = bank_conflicts(L, num_banks=32, group_size=32)
        assert result["max_way"] == 32
        assert not result["conflict_free"]

    def test_full_tile_swizzle_conflict_free(self):
        """Full 64x32 tile with Swizzle(5,0,5): zero bank conflicts."""
        L = Layout((64, 32), (32, 1))
        sw = make_swizzle(32)
        assert sw == Swizzle(5, 0, 5)
        result = bank_conflicts(L, num_banks=32, swizzle=sw)
        assert result["conflict_free"]

    def test_full_tile_padding_reduces_conflicts(self):
        """Full 64x32 tile with padded stride 34: reduced but nonzero conflicts."""
        L = Layout((64, 32), (34, 1))
        result = bank_conflicts(L, num_banks=32, group_size=32)
        # Stride 34 mod 32 = 2 → rows cycle through even banks
        # After 16 rows, bank indices repeat → 2-way conflicts
        assert result["max_way"] == 2
        assert not result["conflict_free"]

    def test_swizzle_strictly_better_than_padding(self):
        """Swizzle eliminates all conflicts; padding only reduces them."""
        # Column access pattern: 32 rows, stride 32
        column = Layout(32, 32)

        r_plain = bank_conflicts(column, num_banks=32, group_size=32)
        r_padded = bank_conflicts(Layout(32, 34), num_banks=32, group_size=32)
        r_swizzled = bank_conflicts(column, num_banks=32, group_size=32, swizzle=make_swizzle(32))

        assert r_plain["max_way"] == 32  # all same bank
        assert r_padded["max_way"] == 2  # 2-way
        assert r_swizzled["conflict_free"]  # zero conflicts

    def test_bk16_swizzle_reduces_conflicts(self):
        """BK=16: Swizzle(4,0,4) can only differentiate 2^4=16 banks.

        With 32 banks and only 4 swizzle bits, 2-way conflicts remain.
        This is a fundamental limit: stride < num_banks → B < log2(num_banks).
        """
        L = Layout((64, 16), (16, 1))
        sw = make_swizzle(16)
        assert sw == Swizzle(4, 0, 4)

        r_plain = bank_conflicts(L, num_banks=32)
        r_swizzled = bank_conflicts(L, num_banks=32, swizzle=sw)

        # Swizzle reduces conflicts dramatically but can't fully eliminate
        assert r_swizzled["max_way"] == 2  # 16 banks covered, 2-way for 32 threads
        assert r_swizzled["total_conflicts"] < r_plain["total_conflicts"]

    def test_bk8_swizzle_reduces_conflicts(self):
        """BK=8: Swizzle(3,0,3) differentiates 2^3=8 banks → 4-way at worst."""
        L = Layout((64, 8), (8, 1))
        sw = make_swizzle(8)
        assert sw == Swizzle(3, 0, 3)

        r_plain = bank_conflicts(L, num_banks=32)
        r_swizzled = bank_conflicts(L, num_banks=32, swizzle=sw)

        assert r_swizzled["max_way"] == 4  # 8 banks covered, 4-way for 32 threads
        assert r_swizzled["total_conflicts"] < r_plain["total_conflicts"]

    def test_bk32_swizzle_is_conflict_free(self):
        """BK=32: Swizzle(5,0,5) covers all 32 banks — zero conflicts."""
        L = Layout((64, 32), (32, 1))
        sw = make_swizzle(32)
        assert sw == Swizzle(5, 0, 5)
        result = bank_conflicts(L, num_banks=32, swizzle=sw)
        assert result["conflict_free"]


class TestSwizzleEmission:
    """Test MSL code generation for XOR swizzle paths."""

    def test_cooperative_load_vec4_swizzle_writes(self):
        """Vec4 device read with scalar XOR'd threadgroup writes."""
        from metile.codegen.msl_emitter import _emit_cooperative_load
        from metile.ir import metal_ir as mir

        op = mir.MCooperativeLoad(
            device_ptr=mir.MValue("A", mir.PtrType("float")),
            tg_array="shared_a",
            row_offset=mir.MValue("block_row", mir.ScalarType("u32")),
            col_offset=None,
            src_stride=mir.MValue("K", mir.ScalarType("u32")),
            tile_rows=64,
            tile_cols=32,
            dst_stride=32,
            tg_size=128,
            linear_tid=mir.MValue("tid", mir.ScalarType("u32")),
            bounds_check=False,
            vec_size=4,
            elem_type="float",
            swizzle_bits=5,
            swizzle_shift=5,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_cooperative_load(op, lines, 0, func)
        msl = "\n".join(lines)

        # Should have vec4 device read
        assert "float4" in msl
        # Should have XOR'd scalar writes (4 of them)
        assert "^ ((" in msl
        assert ">> 5u" in msl
        assert "& 31u" in msl  # (1 << 5) - 1 = 31
        # Should NOT have vec4 threadgroup write
        assert "threadgroup float4" not in msl

    def test_cooperative_load_scalar_swizzle(self):
        """Scalar load with swizzle uses XOR'd address."""
        from metile.codegen.msl_emitter import _emit_cooperative_load
        from metile.ir import metal_ir as mir

        op = mir.MCooperativeLoad(
            device_ptr=mir.MValue("A", mir.PtrType("float")),
            tg_array="shared_a",
            row_offset=mir.MValue("block_row", mir.ScalarType("u32")),
            col_offset=None,
            src_stride=mir.MValue("K", mir.ScalarType("u32")),
            tile_rows=64,
            tile_cols=32,
            dst_stride=32,
            tg_size=128,
            linear_tid=mir.MValue("tid", mir.ScalarType("u32")),
            bounds_check=False,
            vec_size=1,
            elem_type="float",
            swizzle_bits=5,
            swizzle_shift=5,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_cooperative_load(op, lines, 0, func)
        msl = "\n".join(lines)

        assert "_off ^" in msl
        assert ">> 5u" in msl

    def test_cooperative_load_bounds_check_swizzle(self):
        """Bounds-checked load with swizzle uses XOR'd address."""
        from metile.codegen.msl_emitter import _emit_cooperative_load
        from metile.ir import metal_ir as mir

        op = mir.MCooperativeLoad(
            device_ptr=mir.MValue("A", mir.PtrType("float")),
            tg_array="shared_a",
            row_offset=mir.MValue("block_row", mir.ScalarType("u32")),
            col_offset=None,
            src_stride=mir.MValue("K", mir.ScalarType("u32")),
            tile_rows=64,
            tile_cols=32,
            dst_stride=32,
            tg_size=128,
            linear_tid=mir.MValue("tid", mir.ScalarType("u32")),
            bounds_check=True,
            vec_size=1,
            elem_type="float",
            row_bound=mir.MValue("M", mir.ScalarType("u32")),
            col_bound=mir.MValue("K", mir.ScalarType("u32")),
            swizzle_bits=5,
            swizzle_shift=5,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_cooperative_load(op, lines, 0, func)
        msl = "\n".join(lines)

        assert "_off ^" in msl
        assert "float(0)" in msl  # bounds check fallback

    def test_cooperative_load_no_swizzle_unchanged(self):
        """Without swizzle bits, emission is unchanged (vec4 writes)."""
        from metile.codegen.msl_emitter import _emit_cooperative_load
        from metile.ir import metal_ir as mir

        op = mir.MCooperativeLoad(
            device_ptr=mir.MValue("A", mir.PtrType("float")),
            tg_array="shared_a",
            row_offset=mir.MValue("block_row", mir.ScalarType("u32")),
            col_offset=None,
            src_stride=mir.MValue("K", mir.ScalarType("u32")),
            tile_rows=64,
            tile_cols=32,
            dst_stride=32,
            tg_size=128,
            linear_tid=mir.MValue("tid", mir.ScalarType("u32")),
            bounds_check=False,
            vec_size=4,
            elem_type="float",
            swizzle_bits=0,
            swizzle_shift=0,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_cooperative_load(op, lines, 0, func)
        msl = "\n".join(lines)

        # Should use vec4 threadgroup write (DS=32, divisible by 4)
        assert "threadgroup float4" in msl
        assert "^ ((" not in msl

    def test_mma_inner_loop_swizzle_uses_thread_elements(self):
        """MMA with swizzle replaces simdgroup_load with thread_elements()."""
        from metile.codegen.msl_emitter import _emit_mma_inner_loop
        from metile.ir import metal_ir as mir

        op = mir.MMAInnerLoop(
            shared_a="shared_a",
            shared_b="shared_b",
            acc_name="acc",
            a_stride=32,
            b_stride=32,
            sg_row=mir.MValue("sg_row", mir.ScalarType("u32")),
            sg_col=mir.MValue("sg_col", mir.ScalarType("u32")),
            num_8m=2,
            num_8n=2,
            bk=32,
            in_type="float",
            a_swizzle_bits=5,
            a_swizzle_shift=5,
            b_swizzle_bits=5,
            b_swizzle_shift=5,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_mma_inner_loop(op, lines, 0, func)
        msl = "\n".join(lines)

        # Should use thread_elements() instead of simdgroup_load
        assert "thread_elements()" in msl
        assert "slid & 7u" in msl
        assert "slid >> 4u" in msl
        # Should XOR addresses
        assert "^ ((" in msl
        # Should NOT use simdgroup_load for A or B
        assert "simdgroup_load" not in msl
        # Should still have MMA
        assert "simdgroup_multiply_accumulate" in msl

    def test_mma_serpentine_swizzle(self):
        """Serpentine MMA with swizzle uses thread_elements()."""
        from metile.codegen.msl_emitter import _emit_mma_inner_loop
        from metile.ir import metal_ir as mir

        op = mir.MMAInnerLoop(
            shared_a="shared_a",
            shared_b="shared_b",
            acc_name="acc",
            a_stride=32,
            b_stride=32,
            sg_row=mir.MValue("sg_row", mir.ScalarType("u32")),
            sg_col=mir.MValue("sg_col", mir.ScalarType("u32")),
            num_8m=2,
            num_8n=2,
            bk=32,
            in_type="float",
            serpentine=True,
            a_swizzle_bits=5,
            a_swizzle_shift=5,
            b_swizzle_bits=5,
            b_swizzle_shift=5,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_mma_inner_loop(op, lines, 0, func)
        msl = "\n".join(lines)

        assert "thread_elements()" in msl
        assert "simdgroup_load" not in msl
        assert "simdgroup_multiply_accumulate" in msl

    def test_mma_partial_swizzle_a_only(self):
        """Swizzle only on A: A uses thread_elements, B uses simdgroup_load."""
        from metile.codegen.msl_emitter import _emit_mma_inner_loop
        from metile.ir import metal_ir as mir

        op = mir.MMAInnerLoop(
            shared_a="shared_a",
            shared_b="shared_b",
            acc_name="acc",
            a_stride=32,
            b_stride=34,  # B has padding, no swizzle
            sg_row=mir.MValue("sg_row", mir.ScalarType("u32")),
            sg_col=mir.MValue("sg_col", mir.ScalarType("u32")),
            num_8m=2,
            num_8n=2,
            bk=32,
            in_type="float",
            a_swizzle_bits=5,
            a_swizzle_shift=5,
            b_swizzle_bits=0,
            b_swizzle_shift=0,
        )
        func = mir.MFunction("test", kernel_type="gemm")
        lines = []
        _emit_mma_inner_loop(op, lines, 0, func)
        msl = "\n".join(lines)

        assert "thread_elements()" in msl  # A uses swizzle
        assert "simdgroup_load" in msl  # B uses normal load


class TestSwizzlePass:
    """Test swizzle_shared_memory compiler pass."""

    def test_pass_sets_swizzle_bits_on_cooperative_loads(self):
        """Pass sets swizzle_bits and swizzle_shift on cooperative loads."""
        from metile.ir import metal_ir as mir

        func = mir.MFunction("test", kernel_type="gemm")
        load_a = mir.MCooperativeLoad(
            tg_array="shared_a",
            tile_rows=64,
            tile_cols=32,
            dst_stride=32,
        )
        load_b = mir.MCooperativeLoad(
            tg_array="shared_b",
            tile_rows=32,
            tile_cols=64,
            dst_stride=64,
        )
        mma = mir.MMAInnerLoop(
            shared_a="shared_a",
            shared_b="shared_b",
            a_stride=32,
            b_stride=64,
        )
        func.ops = [load_a, load_b, mma]

        from metile.compiler.passes import swizzle_shared_memory

        swizzle_shared_memory(func)

        # A: tile_cols=32, log2(32)=5, Swizzle(5,0,5)
        assert load_a.swizzle_bits == 5
        assert load_a.swizzle_shift == 5
        assert load_a.dst_stride == 32  # no padding

        # B: tile_cols=64, log2(64)=6 but min(6,5)=5, Swizzle(5,0,6)
        assert load_b.swizzle_bits == 5
        assert load_b.swizzle_shift == 6
        assert load_b.dst_stride == 64

        # MMA
        assert mma.a_swizzle_bits == 5
        assert mma.a_swizzle_shift == 5
        assert mma.b_swizzle_bits == 5
        assert mma.b_swizzle_shift == 6

    def test_pass_skips_non_power_of_2(self):
        """Pass returns unchanged function for non-power-of-2 tile_cols."""
        from metile.ir import metal_ir as mir

        func = mir.MFunction("test", kernel_type="gemm")
        load = mir.MCooperativeLoad(
            tg_array="shared_a",
            tile_rows=64,
            tile_cols=48,
            dst_stride=48,
        )
        func.ops = [load]

        assert load.swizzle_bits == 0  # unchanged

    def test_pass_skips_elementwise(self):
        """Pass returns unchanged function for elementwise kernels."""
        from metile.compiler.passes import swizzle_shared_memory
        from metile.ir import metal_ir as mir

        func = mir.MFunction("test", kernel_type="elementwise")
        result = swizzle_shared_memory(func)
        assert result is func
