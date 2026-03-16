import numpy as np
import pytest

from metile.ir.layout import col_major, row_major
from metile.runtime.address_space import (
    GlobalAddressSpace,
    KernelPipeline,
)


class TestGlobalAddressSpace:
    def test_create(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        assert space.capacity == 1024 * 1024
        assert space.used == 0

    def test_alloc_tensor(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((64, 64), dtype=np.float32)
        assert A.shape == (64, 64)
        assert A.numel == 4096
        assert A.nbytes == 4096 * 4
        assert space.used > 0

    def test_alloc_with_layout(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((4, 8), dtype=np.float32, layout=row_major(4, 8))
        assert A.layout == row_major(4, 8)

    def test_numpy_view_zero_copy(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((4, 4), dtype=np.float32)
        np_view = A.numpy()
        assert np_view.shape == (4, 4)

        # Write via numpy
        np_view[:] = 42.0
        # Read back — same memory
        assert np.all(A.numpy() == 42.0)

    def test_multiple_tensors(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((64, 64), dtype=np.float32, name="A")
        B = space.tensor((64, 64), dtype=np.float32, name="B")
        C = space.tensor((64, 64), dtype=np.float32, name="C")

        # Different byte offsets (aligned to 256)
        assert A.byte_offset != B.byte_offset
        assert B.byte_offset != C.byte_offset

        # Independent memory
        A.fill(1.0)
        B.fill(2.0)
        C.fill(3.0)
        assert np.all(A.numpy() == 1.0)
        assert np.all(B.numpy() == 2.0)
        assert np.all(C.numpy() == 3.0)

    def test_alignment(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((1,), dtype=np.float32)  # 4 bytes
        B = space.tensor((1,), dtype=np.float32)
        # Both should be 256-byte aligned
        assert A.byte_offset % 256 == 0
        assert B.byte_offset % 256 == 0

    def test_reset(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        space.tensor((256, 256), dtype=np.float32)
        assert space.used > 0
        space.reset()
        assert space.used == 0

    def test_capacity_overflow(self):
        space = GlobalAddressSpace(capacity=1024)  # tiny
        with pytest.raises(MemoryError):
            space.tensor((1024, 1024), dtype=np.float32)

    def test_copy_from(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((4, 4), dtype=np.float32)
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        A.copy_from(data)
        assert np.allclose(A.numpy(), data)

    def test_same_metal_buffer(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((64,), dtype=np.float32)
        B = space.tensor((64,), dtype=np.float32)
        # All views share the same Metal buffer (different offsets)
        assert A.metal_buffer == B.metal_buffer


class TestTensorView:
    def test_with_layout(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((4, 8), dtype=np.float32, layout=col_major(4, 8))
        B = A.with_layout(row_major(4, 8))
        assert B.layout == row_major(4, 8)
        assert B.byte_offset == A.byte_offset  # same memory

    def test_subdivide(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((16, 16), dtype=np.float32)
        tiled = A.subdivide((4, 4))
        assert tiled.tile_shape == (4, 4)
        assert tiled.grid == (4, 4)
        assert tiled.num_tiles == 16


class TestTiledView:
    def test_tile_byte_offset(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((8, 8), dtype=np.float32, layout=col_major(8, 8))
        tiled = A.subdivide((4, 4))
        # Tile (0,0) starts at base offset
        assert tiled.tile_byte_offset(0, 0) == A.byte_offset
        # Tile (1,0) starts 4 elements down (stride 1, 4 bytes each)
        assert tiled.tile_byte_offset(1, 0) == A.byte_offset + 4 * 4
        # Tile (0,1) starts 4 columns over (stride 8, 4 elements = 32 floats)
        assert tiled.tile_byte_offset(0, 1) == A.byte_offset + 4 * 8 * 4

    def test_repr(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((16, 16), dtype=np.float32)
        tiled = A.subdivide((4, 4))
        r = repr(tiled)
        assert "tile=(4, 4)" in r
        assert "grid=(4, 4)" in r


class TestKernelPipeline:
    def test_create_pipeline(self):
        space = GlobalAddressSpace(capacity=1024 * 1024)
        A = space.tensor((64,), dtype=np.float32, name="A")
        B = space.tensor((64,), dtype=np.float32, name="B")

        pipeline = KernelPipeline(space)
        pipeline.add_stage("produce", lambda: None, (1,), inputs=[], outputs=[A])
        pipeline.add_stage("consume", lambda: None, (1,), inputs=[A], outputs=[B])

        assert len(pipeline.stages) == 2
        assert "produce -> consume" in repr(pipeline)
