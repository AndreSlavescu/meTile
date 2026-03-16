from __future__ import annotations

import contextlib
import ctypes
import weakref

import numpy as _np

from metile.runtime.metal_device import MetalDevice

# Buffer cache: maps (array_data_ptr, nbytes) -> MtileBuffer
# Uses weak references so buffers are freed when the numpy array is GC'd
_buffer_cache: dict[int, MtileBuffer] = {}


class MtileBuffer:
    """A GPU buffer backed by unified memory, accessible as a numpy array.

    The buffer lives in Metal's shared address space — both CPU and GPU
    can read/write it directly with no copies.

    Numpy arrays are automatically converted to MtileBuffer when passed
    to kernels — no manual buffer management needed:

        # Just pass numpy arrays directly:
        a = np.random.randn(1024, 1024).astype(np.float32)
        b = np.random.randn(1024, 1024).astype(np.float32)
        c = np.zeros((1024, 1024), dtype=np.float32)
        matmul[(grid_m, grid_n)](a, b, c, M, N, K, ...)
        # c now contains the result — automatically synced back

    Or use explicit buffers for persistent GPU-resident data:

        a = metile.Buffer(data=np.random.randn(1024, 1024).astype(np.float32))
        kernel[grid](a, ...)
        result = a.numpy()  # direct view, zero-copy
    """

    def __init__(self, shape=None, dtype=_np.float32, data: _np.ndarray | None = None):
        if data is not None:
            shape = data.shape
            dtype = data.dtype.type

        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = _np.dtype(dtype)
        self.nbytes = int(_np.prod(self.shape)) * self.dtype.itemsize
        # Track the source numpy array for sync-back (implicit conversion)
        self._source_array = None

        dev = MetalDevice.get()

        if data is not None:
            data = _np.ascontiguousarray(data)
            self._metal_buffer = dev.new_buffer(data.tobytes(), self.nbytes)
        else:
            self._metal_buffer = dev.new_empty_buffer(self.nbytes)

        # Get raw pointer and create numpy view into unified memory
        self._ptr = dev.buffer_contents(self._metal_buffer)
        arr_type = ctypes.c_byte * self.nbytes
        buf_array = arr_type.from_address(self._ptr)
        self._np_view = _np.frombuffer(buf_array, dtype=self.dtype).reshape(self.shape)

    def numpy(self) -> _np.ndarray:
        """Numpy view of the unified memory buffer. Reads and writes are direct.

        Waits for any pending GPU work to complete before returning,
        ensuring all writes are visible to the CPU.
        """
        MetalDevice.get().sync()
        return self._np_view

    @property
    def metal_buffer(self) -> ctypes.c_void_p:
        """The underlying Metal buffer object."""
        return self._metal_buffer

    def sync_to_source(self):
        """Copy buffer contents back to the source numpy array (if any)."""
        if self._source_array is not None:
            src = self._source_array
            if src is not None:
                ctypes.memmove(src.ctypes.data, self._ptr, self.nbytes)

    def sync_from_source(self):
        """Copy source numpy array contents into the buffer."""
        if self._source_array is not None:
            src = self._source_array
            if src is not None:
                data = _np.ascontiguousarray(src)
                ctypes.memmove(self._ptr, data.ctypes.data, self.nbytes)

    def __repr__(self):
        return f"MtileBuffer(shape={self.shape}, dtype={self.dtype})"

    @classmethod
    def from_numpy(cls, arr: _np.ndarray) -> MtileBuffer:
        """Create a buffer initialized from a numpy array."""
        return cls(data=_np.ascontiguousarray(arr))

    @classmethod
    def zeros(cls, shape, dtype=_np.float32) -> MtileBuffer:
        """Create a zero-initialized buffer."""
        buf = cls(shape, dtype)
        buf._np_view[:] = 0
        return buf

    @classmethod
    def empty(cls, shape, dtype=_np.float32) -> MtileBuffer:
        """Create an uninitialized buffer."""
        return cls(shape, dtype)

    @classmethod
    def _from_numpy_implicit(cls, arr: _np.ndarray) -> MtileBuffer:
        """Create or retrieve a cached buffer for implicit numpy conversion.

        Syncs data from the numpy array into the buffer before each kernel
        launch, and syncs results back after. The buffer is cached by the
        array's identity so repeated calls reuse the same Metal buffer.
        """
        arr = _np.ascontiguousarray(arr)
        cache_key = id(arr)

        cached = _buffer_cache.get(cache_key)
        if cached is not None and cached.nbytes == arr.nbytes and cached.dtype == arr.dtype:
            # Sync latest numpy data into the buffer
            cached.sync_from_source()
            return cached

        # Create new buffer and cache it
        buf = cls(data=arr)
        buf._source_array = arr
        _buffer_cache[cache_key] = buf

        # Clean up cache entry when the numpy array is garbage collected
        with contextlib.suppress(TypeError):
            weakref.finalize(arr, _buffer_cache.pop, cache_key, None)

        return buf
