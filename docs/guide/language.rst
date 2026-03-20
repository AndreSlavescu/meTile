Language Reference
==================

meTile provides a Python eDSL (embedded domain-specific language) for writing GPU kernels. Functions
decorated with ``@metile.kernel`` are traced and compiled to Metal shaders. They are not executed
as regular Python.

This page documents every construct available inside a ``@metile.kernel`` function.


Kernel Definition
-----------------

.. code-block:: python

   @metile.kernel
   def my_kernel(ptr_a, ptr_b, N, BLOCK: metile.constexpr):
       ...

Parameters are either:

- **Pointers**: numpy arrays or ``metile.Buffer`` objects become ``device float*`` in Metal
- **Scalars**: Python ints/floats become ``constant int&`` or ``constant float&``
- **Constexprs**: annotated with ``metile.constexpr``, baked into the shader at compile time

Constexprs are passed as keyword arguments at launch:

.. code-block:: python

   my_kernel[grid](a, b, N, BLOCK=256)


Launching Kernels
-----------------

.. code-block:: python

   kernel[grid](*args, **constexprs)

``grid`` is a tuple of 1, 2, or 3 integers specifying the number of program instances
(threadgroups) along each axis.

.. code-block:: python

   kernel[(N,)](...)           # 1D grid
   kernel[(M, N)](...)         # 2D grid
   kernel[(X, Y, Z)](...)     # 3D grid


Program Identity
----------------

.. function:: metile.program_id(axis)

   Returns the index of the current program instance along the given axis.

   .. code-block:: python

      pid_x = metile.program_id(0)   # threadgroup X index
      pid_y = metile.program_id(1)   # threadgroup Y index


Index Generation
----------------

.. function:: metile.arange(start, size)

   Creates a tile of ``size`` consecutive integers starting at ``start``.

   .. code-block:: python

      idx = metile.arange(0, 256)   # [0, 1, 2, ..., 255]

.. function:: metile.cdiv(a, b)

   Ceiling division. Useful for computing grid sizes.

   .. code-block:: python

      grid_size = metile.cdiv(N, BLOCK)   # ceil(N / BLOCK)


Element-wise Memory Access
--------------------------

For element-wise kernels (softmax, activations, reductions), use pointer arithmetic
with ``load`` and ``store``:

.. function:: metile.load(ptr, mask=None)

   Load elements from memory. Masked-off elements read zero.

   .. code-block:: python

      offs = pid * BLOCK + metile.arange(0, BLOCK)
      mask = offs < N
      x = metile.load(X + offs, mask=mask)

.. function:: metile.store(ptr, value, mask=None)

   Store elements to memory. Masked-off elements are skipped.

   .. code-block:: python

      metile.store(Out + offs, result, mask=mask)


Tile Memory Access
------------------

For matrix operations (GEMM), use tile-level loads and stores that map to simdgroup
or tensor_ops hardware:

.. function:: metile.tile_load(ptr, row_offset, col_offset, stride, shape)

   Load a 2D tile from row-major memory.

   :param ptr: base pointer to the matrix
   :param row_offset: row index of tile's top-left corner
   :param col_offset: column index of tile's top-left corner
   :param stride: leading dimension (number of columns in the full matrix)
   :param shape: ``(rows, cols)`` of the tile to load

   .. code-block:: python

      # Load a 128x32 tile of A starting at (pid_m * 128, k)
      a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))

.. function:: metile.tile_store(ptr, row_offset, col_offset, stride, value, shape)

   Store a 2D tile to row-major memory.

   .. code-block:: python

      metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))

.. function:: metile.zeros(shape, dtype="f32")

   Create a zero-initialized tile. Used to initialize accumulators.

   .. code-block:: python

      acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")


Matrix Multiply
---------------

.. function:: metile.dot(a, b, acc)

   Tile-level matrix multiply-accumulate: ``acc += a @ b``.

   The compiler maps this to ``simdgroup_multiply_accumulate`` (M1-M3) or
   ``matmul2d`` tensor_ops (M4+) depending on hardware.

   .. code-block:: python

      acc = metile.zeros((128, 128), dtype="f32")
      for k in metile.tile_range(0, K, BLOCK_K):
          a = metile.tile_load(A, pid_m * 128, k, K, (128, BLOCK_K))
          b = metile.tile_load(B, k, pid_n * 128, N, (BLOCK_K, 128))
          acc = metile.dot(a, b, acc)


Control Flow
------------

.. function:: metile.tile_range(start, end, step)

   A tiling loop. Equivalent to ``range(start, end, step)`` but tells the compiler this
   is a tile-level iteration (e.g., the K-loop in GEMM).

   .. code-block:: python

      for k in metile.tile_range(0, K, BLOCK_K):
          ...


Math Operations
---------------

All math ops are element-wise and work on both scalars and tiles:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``metile.exp(x)``
     - Exponential
   * - ``metile.log(x)``
     - Natural logarithm
   * - ``metile.sqrt(x)``
     - Square root
   * - ``metile.abs(x)``
     - Absolute value
   * - ``metile.tanh(x)``
     - Hyperbolic tangent
   * - ``metile.where(cond, x, y)``
     - Select ``x`` where ``cond`` is true, else ``y``
   * - ``metile.maximum(a, b)``
     - Element-wise maximum
   * - ``metile.minimum(a, b)``
     - Element-wise minimum

Standard Python arithmetic works inside kernels: ``+``, ``-``, ``*``, ``/``, ``<``, ``>``, etc.


Reductions
----------

.. function:: metile.sum(x)

   Sum-reduce a tile to a scalar.

.. function:: metile.max(x)

   Max-reduce a tile to a scalar.

.. function:: metile.min(x)

   Min-reduce a tile to a scalar.

These compile to simdgroup shuffle reductions on the GPU.

.. code-block:: python

   # Two-pass softmax: find max, then compute normalized exponentials
   m = -1e38
   for i in metile.tile_range(0, N, BLOCK):
       cols = i + metile.arange(0, BLOCK)
       x = metile.load(X + row * N + cols, mask=cols < N)
       m = metile.maximum(m, x)
   m = metile.max(m)   # reduce across the tile


Advanced: Simdgroup Operations
------------------------------

For low-level control over Apple GPU simdgroups:

.. function:: metile.simdgroup_role(role, num_roles, body, num_sgs=0)

   Execute different code on different simdgroup subsets within a threadgroup.
   Enables producer/consumer patterns.

   .. code-block:: python

      with metile.simdgroup_role(role=0, num_roles=2):
          # Only the first half of simdgroups run this
          ...
      with metile.simdgroup_role(role=1, num_roles=2):
          # Only the second half run this
          ...

.. function:: metile.simd_shuffle_xor(value, mask)

   Exchange data between lanes within a simdgroup using XOR addressing.

.. function:: metile.simd_broadcast(value, lane)

   Broadcast a value from one lane to all lanes in a simdgroup.

.. function:: metile.simd_lane_id()

   Returns the current thread's lane index within its simdgroup (0-31).

.. function:: metile.thread_id()

   Returns the thread's position within the threadgroup.

.. function:: metile.barrier()

   Threadgroup memory barrier. Forces all threads to reach this point before proceeding.

.. function:: metile.shared(size, dtype="f32")

   Allocate threadgroup (shared) memory.


Tile Scheduling
---------------

.. function:: metile.tile_swizzle(pid_m, pid_n, pattern="morton", block_size=2)

   Apply a tile scheduling pattern for better cache locality in 2D grids.
   Supported patterns: ``"morton"`` (Z-order), ``"diagonal"``.

   .. code-block:: python

      pid_m, pid_n = metile.tile_swizzle(
          metile.program_id(0), metile.program_id(1),
          pattern="morton", block_size=2,
      )
