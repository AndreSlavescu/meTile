from __future__ import annotations

import ctypes

import numpy as np

from metile.ir.layout import Layout, col_major
from metile.runtime.metal_device import MetalDevice


class GlobalAddressSpace:
    """A large unified memory arena for zero-copy GPU/CPU access.

    Manages a single Metal buffer in shared address space. Tensors are
    allocated as sub-regions with layout metadata. The arena acts as
    a last-level cache — both CPU and GPU operate on the same physical
    memory with no copies.

    Multiple arenas can coexist for different lifetime scopes (e.g.,
    one for model weights, one for activations, one for scratch).
    """

    def __init__(self, capacity: int = 256 * 1024 * 1024):
        """Create an address space with given capacity in bytes."""
        self._dev = MetalDevice.get()
        self._capacity = capacity
        self._metal_buffer = self._dev.new_empty_buffer(capacity)
        self._ptr = self._dev.buffer_contents(self._metal_buffer)

        # Create numpy view of the entire arena
        arr_type = ctypes.c_byte * capacity
        buf_array = arr_type.from_address(self._ptr)
        self._raw = np.frombuffer(buf_array, dtype=np.uint8)

        # Bump allocator state
        self._offset = 0
        self._views: list[TensorView] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        return self._offset

    @property
    def metal_buffer(self):
        return self._metal_buffer

    def tensor(
        self, shape, dtype=np.float32, layout: Layout | None = None, name: str = ""
    ) -> TensorView:
        """Allocate a tensor in the address space and return a view.

        Args:
            shape: tuple of ints (e.g., (M, N))
            dtype: numpy dtype
            layout: optional Layout for non-default memory ordering
            name: optional name for debugging

        Returns:
            TensorView bound to this region of the address space
        """
        dtype = np.dtype(dtype)
        if isinstance(shape, int):
            shape = (shape,)
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * dtype.itemsize

        # Align to 256 bytes (Metal buffer offset alignment)
        aligned_offset = (self._offset + 255) & ~255
        if aligned_offset + nbytes > self._capacity:
            raise MemoryError(
                f"Address space exhausted: need {aligned_offset + nbytes} bytes, "
                f"capacity is {self._capacity}"
            )

        # Default layout: compact column-major
        if layout is None:
            if len(shape) == 2:
                layout = col_major(shape[0], shape[1])
            elif len(shape) == 1:
                layout = Layout(shape[0])
            else:
                layout = Layout(shape)

        # Create the view
        view = TensorView(
            address_space=self,
            byte_offset=aligned_offset,
            shape=shape,
            dtype=dtype,
            layout=layout,
            name=name,
        )

        self._offset = aligned_offset + nbytes
        self._views.append(view)
        return view

    def reset(self):
        """Reset the allocator (free all tensors). Views become invalid."""
        self._offset = 0
        self._views.clear()

    def numpy_view(self, byte_offset: int, shape, dtype) -> np.ndarray:
        """Get a numpy view into the arena at a given byte offset."""
        dtype = np.dtype(dtype)
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * dtype.itemsize
        return np.frombuffer(
            self._raw[byte_offset : byte_offset + nbytes].data,
            dtype=dtype,
        ).reshape(shape)

    def __repr__(self):
        used_mb = self._offset / (1024 * 1024)
        cap_mb = self._capacity / (1024 * 1024)
        return f"GlobalAddressSpace({used_mb:.1f}/{cap_mb:.1f} MB, {len(self._views)} tensors)"


class TensorView:
    """A view into the global address space with layout metadata.

    Binds a Layout (coordinate -> offset mapping) to a region of unified
    memory. Views are lightweight — they don't own memory, just reference it.

    Views can be:
      - Passed directly to kernels (via .metal_buffer and .byte_offset)
      - Subdivided into tiles (via .subdivide)
      - Sliced (via .slice)
      - Read/written by CPU via .numpy()
    """

    def __init__(
        self,
        address_space: GlobalAddressSpace,
        byte_offset: int,
        shape: tuple,
        dtype: np.dtype,
        layout: Layout,
        name: str = "",
    ):
        self._space = address_space
        self._byte_offset = byte_offset
        self._shape = shape
        self._dtype = dtype
        self._layout = layout
        self._name = name

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def byte_offset(self) -> int:
        return self._byte_offset

    @property
    def metal_buffer(self):
        """The Metal buffer object (for kernel dispatch)."""
        return self._space.metal_buffer

    @property
    def numel(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * self._dtype.itemsize

    def numpy(self) -> np.ndarray:
        """Get a numpy view into this tensor's memory. Zero-copy.

        Writes to this array are immediately visible to the GPU,
        and GPU writes are immediately visible here.
        """
        return self._space.numpy_view(self._byte_offset, self._shape, self._dtype)

    def fill(self, value=0):
        """Fill the tensor with a value (CPU-side)."""
        self.numpy()[:] = value
        return self

    def copy_from(self, data: np.ndarray):
        """Copy data into this tensor's memory."""
        np.copyto(self.numpy(), data.reshape(self._shape))
        return self

    def subdivide(self, tile_shape) -> TiledView:
        """Subdivide this tensor into tiles using logical divide.

        Args:
            tile_shape: tuple of tile dimensions, e.g., (BM, BN)

        Returns:
            TiledView that provides tile access patterns
        """
        if isinstance(tile_shape, int):
            tile_shape = (tile_shape,)
        tiler = Layout(tile_shape)
        divided = self._layout.logical_divide(tiler)
        return TiledView(self, divided, tile_shape)

    def as_buffer(self):
        """Get an MtileBuffer aliasing this view's memory in the address space.

        The returned buffer points directly into the address space's Metal
        buffer — no copies. Writes by GPU kernels are visible in the address
        space immediately.
        """
        from metile.runtime.buffer import MtileBuffer

        buf = MtileBuffer.__new__(MtileBuffer)
        buf.shape = self._shape
        buf.dtype = self._dtype
        buf.nbytes = self.nbytes
        buf._source_array = None
        buf._metal_buffer = self._space.metal_buffer
        buf._ptr = self._space._ptr + self._byte_offset
        # Create numpy view into the arena at our offset
        import ctypes

        arr_type = ctypes.c_byte * self.nbytes
        buf_array = arr_type.from_address(buf._ptr)
        import numpy as np

        buf._np_view = np.frombuffer(buf_array, dtype=self._dtype).reshape(self._shape)
        # Override metal_buffer property to include offset info
        buf._byte_offset = self._byte_offset
        return buf

    def with_layout(self, layout: Layout) -> TensorView:
        """Create a new view with a different layout over the same memory."""
        return TensorView(
            address_space=self._space,
            byte_offset=self._byte_offset,
            shape=self._shape,
            dtype=self._dtype,
            layout=layout,
            name=self._name,
        )

    def __repr__(self):
        name = f"'{self._name}' " if self._name else ""
        return (
            f"TensorView({name}{self._shape}, {self._dtype}, "
            f"layout={self._layout}, offset={self._byte_offset})"
        )


class TiledView:
    """A tensor view subdivided into tiles.

    Provides structured access to tiles within a tensor, including:
    - Number of tiles in each dimension
    - Byte offsets for each tile (for kernel dispatch)
    - Layout of elements within each tile
    """

    def __init__(self, base: TensorView, divided_layout: Layout, tile_shape: tuple):
        self._base = base
        self._divided = divided_layout
        self._tile_shape = tile_shape

        # Compute grid dimensions
        self._grid = tuple((s + t - 1) // t for s, t in zip(base.shape, tile_shape))

    @property
    def tile_shape(self) -> tuple:
        return self._tile_shape

    @property
    def grid(self) -> tuple:
        """Number of tiles in each dimension."""
        return self._grid

    @property
    def num_tiles(self) -> int:
        n = 1
        for g in self._grid:
            n *= g
        return n

    @property
    def divided_layout(self) -> Layout:
        return self._divided

    @property
    def base(self) -> TensorView:
        return self._base

    def tile_byte_offset(self, *tile_idx) -> int:
        """Get byte offset for a specific tile.

        Args:
            tile_idx: tile coordinates (e.g., tile_byte_offset(2, 3))

        Returns:
            byte offset from start of the address space buffer
        """
        # Compute element offset of tile origin
        if len(tile_idx) == 1 and isinstance(tile_idx[0], tuple):
            tile_idx = tile_idx[0]
        elem_offset = 0
        stride = 1
        for i, (ti, ts) in enumerate(zip(tile_idx, self._tile_shape)):
            # Element offset of tile start in the original layout
            if len(self._base.shape) > 1:
                # Use the base layout stride for this dimension
                base_stride = self._base.layout.stride
                if isinstance(base_stride, tuple) and i < len(base_stride):
                    elem_offset += ti * ts * base_stride[i]
                else:
                    elem_offset += ti * ts * stride
            else:
                elem_offset += ti * ts
            stride *= self._base.shape[i] if i < len(self._base.shape) else 1
        return self._base.byte_offset + elem_offset * self._base.dtype.itemsize

    def __repr__(self):
        return f"TiledView(tile={self._tile_shape}, grid={self._grid}, base={self._base.shape})"


class KernelPipeline:
    """Chain producer and consumer kernels over shared address space views.

    Expresses multi-kernel dataflow where one kernel's output is another's
    input, all through the same unified memory — zero copies.

        pipeline = KernelPipeline(space)
        pipeline.add_stage("transform", transform_kernel, grid, inputs=[A], outputs=[B])
        pipeline.add_stage("gemm", gemm_kernel, grid, inputs=[B, C], outputs=[D])
        pipeline.run()  # executes stages in order, views are reusable
    """

    def __init__(self, address_space: GlobalAddressSpace):
        self._space = address_space
        self._stages: list[dict] = []

    def add_stage(
        self,
        name: str,
        kernel_fn,
        grid: tuple,
        inputs: list[TensorView],
        outputs: list[TensorView],
        **kwargs,
    ):
        """Add a pipeline stage."""
        self._stages.append(
            {
                "name": name,
                "kernel": kernel_fn,
                "grid": grid,
                "inputs": inputs,
                "outputs": outputs,
                "kwargs": kwargs,
            }
        )
        return self

    @property
    def stages(self) -> list[dict]:
        return self._stages

    def dispatch_stage(self, stage_idx: int):
        """Dispatch a single pipeline stage using address space offsets.

        Uses the address space's single Metal buffer with per-tensor byte
        offsets — true zero-copy producer/consumer.
        """
        stage = self._stages[stage_idx]
        views = stage["inputs"] + stage["outputs"]
        metal_dev = MetalDevice.get()

        compiled = stage["compiled"]
        buffers = []
        offsets = []
        for v in views:
            buffers.append(v.metal_buffer)
            offsets.append(v.byte_offset)
        # Add scalar args
        for buf in stage.get("scalar_bufs", []):
            buffers.append(buf)
            offsets.append(0)

        tg = compiled.threadgroup_size
        grid = stage["grid"]
        if compiled.is_gemm:
            grid_tg = (grid[0], grid[1] if len(grid) > 1 else 1, 1)
            metal_dev.dispatch_threadgroups(compiled.pipeline, buffers, grid_tg, tg, offsets)
        else:
            total_threads = (grid[0] * tg[0], 1, 1)
            metal_dev.dispatch_kernel(compiled.pipeline, buffers, total_threads, tg, offsets)

    def __repr__(self):
        names = [s["name"] for s in self._stages]
        return f"KernelPipeline({' -> '.join(names)})"
