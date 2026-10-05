from __future__ import annotations

import ctypes
import weakref

import numpy as _np

from metile.runtime.metal_device import MetalDevice

_buffer_cache: dict[int, MtileBuffer] = {}


def _owned_byte_array(device, metal_buffer, length):
    """Keep a Metal allocation alive through every escaped NumPy view."""
    try:
        array_type = ctypes.c_byte * length
        array = array_type.from_address(device.buffer_contents(metal_buffer))
    except BaseException:
        device.release_buffer(metal_buffer)
        raise
    finalizer = weakref.finalize(array, device.release_buffer, metal_buffer)
    finalizer.atexit = False
    return array


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
        self._source_ref = None

        dev = MetalDevice.get()

        if data is not None:
            data = _np.ascontiguousarray(data)
            self._metal_buffer = dev.new_buffer(data.tobytes(), self.nbytes)
        else:
            self._metal_buffer = dev.new_empty_buffer(self.nbytes)

        # Get raw pointer and create numpy view into unified memory
        buf_array = _owned_byte_array(dev, self._metal_buffer, self.nbytes)
        self._ptr = ctypes.addressof(buf_array)
        self._np_view = _np.frombuffer(buf_array, dtype=self.dtype).reshape(self.shape)

    @property
    def _source_array(self):
        return self._source_ref() if self._source_ref is not None else None

    @_source_array.setter
    def _source_array(self, array):
        self._source_ref = weakref.ref(array) if array is not None else None

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
        source = self._source_array
        if source is not None:
            _np.copyto(source, self._np_view)

    def sync_from_source(self):
        """Copy source numpy array contents into the buffer."""
        source = self._source_array
        if source is not None:
            data = _np.ascontiguousarray(source)
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
        cache_key = id(arr)

        cached = _buffer_cache.get(cache_key)
        if (
            cached is not None
            and cached.shape == arr.shape
            and cached.nbytes == arr.nbytes
            and cached.dtype == arr.dtype
            and cached._source_array is arr
        ):
            # Sync latest numpy data into the buffer
            cached.sync_from_source()
            return cached

        # Create new buffer and cache it
        buf = cls(data=arr)
        buf._source_array = arr
        _buffer_cache[cache_key] = buf

        # Clean up cache entry when the numpy array is garbage collected
        weakref.finalize(arr, _buffer_cache.pop, cache_key, None)

        return buf
