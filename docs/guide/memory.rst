Memory Model
============

Apple Silicon has a **unified memory architecture** where the CPU and GPU share the same physical
memory. meTile exposes this directly through ``metile.Buffer``.

.. image:: /_static/unified-memory.svg
   :alt: Unified memory: CPU and GPU both access the same physical memory through metile.Buffer
   :width: 100%


Buffers
-------

.. code-block:: python

   import numpy as np
   import metile

   # Create from numpy (zero-copy, the GPU reads the same memory)
   x = metile.Buffer(data=np.random.randn(1024).astype(np.float32))

   # Allocate zeroed
   out = metile.Buffer.zeros((1024,))

   # Allocate from existing numpy array
   arr = np.zeros(1024, dtype=np.float32)
   buf = metile.Buffer.from_numpy(arr)

   # Read results back to numpy (also zero-copy)
   result = out.numpy()

There is **no explicit host-to-device copy**. When you create a ``metile.Buffer``, the data lives
in unified memory accessible to both CPU and GPU. After a kernel writes to a buffer, call
``sync()`` to ensure the GPU has finished, then read the buffer directly:

.. code-block:: python

   from metile.runtime.metal_device import MetalDevice

   kernel[grid](x, out, N, BLOCK=256)
   MetalDevice.get().sync()   # wait for GPU to finish
   print(out.numpy())         # read results


Inside Kernels
--------------

Inside ``@metile.kernel`` functions, buffer parameters become device pointers. You access memory
through ``metile.load`` and ``metile.store``:

.. code-block:: python

   # Element-wise access with pointer arithmetic
   offs = pid * BLOCK + metile.arange(0, BLOCK)
   x = metile.load(X + offs, mask=offs < N)
   metile.store(Out + offs, x * 2.0, mask=offs < N)

   # 2D tile access for matrix operations
   a = metile.tile_load(A, row, col, stride, (ROWS, COLS))
   metile.tile_store(C, row, col, stride, result, (ROWS, COLS))


Masking
-------

When the data size is not a multiple of the block size, use masks to prevent out-of-bounds
memory access:

.. code-block:: python

   offs = pid * BLOCK + metile.arange(0, BLOCK)
   mask = offs < N    # boolean mask: True for valid elements

   x = metile.load(X + offs, mask=mask)       # masked-off lanes read 0
   metile.store(Out + offs, x, mask=mask)      # masked-off lanes are skipped

.. code-block:: text

   N = 10, BLOCK = 4, pid = 2 (last instance)

   offs = [8, 9, 10, 11]
   mask = [T, T, F, F] # values 10 and 11 are out of bounds

   load:  reads x[8], x[9], returns 0 for indices 10, 11
   store: writes out[8], out[9], skips indices 10, 11

Masking is essential for correctness. Without it, the last program instance would read/write
past the end of the array.


Shared (Threadgroup) Memory
---------------------------

For kernels that need inter-thread communication within a threadgroup, use shared memory:

.. code-block:: python

   buf = metile.shared(size=256, dtype="f32")
   metile.barrier()   # synchronize all threads in the threadgroup

Shared memory is threadgroup-local and not visible to other threadgroups. Use
``metile.barrier()`` to synchronize access within a threadgroup.
