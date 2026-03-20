Tile Operations & Hardware Mapping
===================================

meTile's tile operations are the bridge between your Python code and Apple GPU hardware.
This page explains how ``dot``, ``tile_load``, and ``tile_store`` map to the actual
hardware instructions.


The Two Backends
----------------

meTile automatically selects the best backend for your hardware when compiling GEMM kernels:

**Simdgroup Matrix (M1/M2/M3)**
   Uses ``simdgroup_matrix<float, 8, 8>``, Apple's 8x8 matrix multiply-accumulate
   primitive. Each simdgroup (32 threads) collaboratively computes an 8x8 tile.
   The compiler tiles the output across multiple simdgroups and uses threadgroup
   (shared) memory to stage data.

**Metal 4 Tensor Ops (M4+)**
   Uses ``matmul2d`` with ``cooperative_tensor``, Metal 4's hardware matrix multiply
   descriptors. Each simdgroup independently loads data from device memory into
   register-resident cooperative tensors and runs the MMA. No threadgroup memory needed.

You write the same kernel code for both. The ``lower()`` function in the compiler inspects
your hardware and chooses the right path.


How Tiling Works
----------------

A GEMM kernel tiles the computation into blocks. Each program instance computes
one output tile, iterating over K to accumulate partial products:

.. image:: /_static/tiling-overview.svg
   :alt: Output matrix tiled into blocks, with K-loop detail showing tile_load and dot accumulation
   :width: 100%

.. code-block:: python

   acc = metile.zeros((BLOCK_M, BLOCK_N))
   for k in metile.tile_range(0, K, BLOCK_K):
       a = metile.tile_load(A, row, k, K, (BLOCK_M, BLOCK_K))
       b = metile.tile_load(B, k, col, N, (BLOCK_K, BLOCK_N))
       acc = metile.dot(a, b, acc)    # acc += a @ b


Compiler Constexprs
-------------------

The tile sizes are compile-time constants that control how the hardware is used:

.. list-table::
   :header-rows: 1

   * - Constexpr
     - Meaning
     - Typical Values
   * - ``BLOCK_M``
     - Output tile rows
     - 64, 128
   * - ``BLOCK_N``
     - Output tile columns
     - 64, 128
   * - ``BLOCK_K``
     - K-loop step size
     - 32, 64, 128
   * - ``WM``
     - Simdgroup grid rows (tensor_ops only)
     - 2, 4
   * - ``WN``
     - Simdgroup grid cols (tensor_ops only)
     - 2, 4

``WM`` and ``WN`` control how many simdgroups tile the output block. With ``WM=4, WN=4``,
16 simdgroups each handle a ``(BLOCK_M/WM) x (BLOCK_N/WN)`` = 32x32 subtile:

.. image:: /_static/simdgroup-layout.svg
   :alt: 4x4 simdgroup grid layout, 16 simdgroups each handling a 32x32 subtile
   :width: 100%


Fused Epilogues
---------------

The compiler detects element-wise operations applied to the accumulator after the GEMM loop
and fuses them into the kernel. No extra memory traffic:

.. code-block:: python

   acc = metile.dot(a, b, acc)

   # These are fused into the GEMM, no global memory round-trip
   acc = metile.where(acc > 0, acc, 0)      # ReLU
   acc = acc * scale                          # scale
   acc = metile.exp(acc)                      # unary

Supported epilogues: ``where`` (ReLU), ``exp``, ``log``, ``sqrt``, ``abs``, ``tanh``,
scalar multiply, scalar add.


Tile Scheduling
---------------

For 2D grids, the order in which tiles are assigned to threadgroups affects L2 cache locality.
meTile supports several scheduling patterns:

.. image:: /_static/morton-swizzle.svg
   :alt: Morton Z-order vs linear tile scheduling, showing how 2x2 blocks share L2 cache
   :width: 100%

**Diagonal**:
   Column assignment is rotated by the row index. Distributes memory traffic.

**Linear**:
   Simple row-major assignment. No locality optimization.

The compiler applies Morton scheduling by default. You can override it:

.. code-block:: python

   pid_m, pid_n = metile.tile_swizzle(
       metile.program_id(0), metile.program_id(1),
       pattern="morton", block_size=2,
   )
