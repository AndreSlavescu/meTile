Matrix Multiply (GEMM)
======================

Tiled matrix multiplication using meTile's ``dot`` and ``tile_load`` operations.


Basic GEMM
----------

.. code-block:: python

   import metile

   @metile.kernel
   def matmul(A, B, C, M, N, K,
              BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
              BLOCK_K: metile.constexpr):
       pid_m = metile.program_id(0)
       pid_n = metile.program_id(1)

       acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
       for k in metile.tile_range(0, K, BLOCK_K):
           a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
           b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
           acc = metile.dot(a, b, acc)

       metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


Launching
---------

The grid is 2D, one program instance per output tile:

.. code-block:: python

   import numpy as np

   M, N, K = 1024, 1024, 1024
   A = metile.Buffer(data=np.random.randn(M, K).astype(np.float32))
   B = metile.Buffer(data=np.random.randn(K, N).astype(np.float32))
   C = metile.Buffer.zeros((M * N,))

   grid = (metile.cdiv(M, 128), metile.cdiv(N, 128))
   matmul[grid](A, B, C, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)


How It Works
------------

1. Each program instance owns a ``BLOCK_M x BLOCK_N`` tile of the output matrix C.
2. It initializes a register-resident accumulator with ``metile.zeros``.
3. The K-loop iterates in steps of ``BLOCK_K``, loading a tile of A and B each step.
4. ``metile.dot(a, b, acc)`` computes ``acc += a @ b`` using hardware matrix multiply.
5. After the loop, the accumulated result is written to C.

The compiler maps ``dot`` to the appropriate hardware:

- **M1/M2/M3**: ``simdgroup_matrix<float, 8, 8>`` with cooperative loads through
  threadgroup memory
- **M4+**: ``matmul2d`` tensor_ops with register-resident ``cooperative_tensor``


Fused GEMM + ReLU
------------------

Element-wise operations after the GEMM loop are fused into the kernel's epilogue.
They run on register-resident data with zero extra memory traffic:

.. code-block:: python

   @metile.kernel
   def matmul_relu(A, B, C, M, N, K,
                   BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
                   BLOCK_K: metile.constexpr):
       pid_m = metile.program_id(0)
       pid_n = metile.program_id(1)
       acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
       for k in metile.tile_range(0, K, BLOCK_K):
           a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
           b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
           acc = metile.dot(a, b, acc)
       acc = metile.where(acc > 0, acc, 0)   # fused ReLU, no global memory round-trip
       metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


Tile Swizzle for Cache Locality
--------------------------------

For large matrices, the order in which tiles are processed affects L2 cache hit rates.
Use ``tile_swizzle`` to apply Morton (Z-order) scheduling:

.. code-block:: python

   @metile.kernel
   def matmul_swizzled(A, B, C, M, N, K,
                       BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
                       BLOCK_K: metile.constexpr):
       pid_m, pid_n = metile.tile_swizzle(
           metile.program_id(0), metile.program_id(1),
           pattern="morton", block_size=2,
       )
       acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
       for k in metile.tile_range(0, K, BLOCK_K):
           a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
           b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
           acc = metile.dot(a, b, acc)
       metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


Autotuning
----------

Different matrix sizes benefit from different block sizes. Use the autotuner to search:

.. code-block:: python

   autotuned_matmul = metile.autotune(
       configs=[
           metile.Config(BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=32, WM=2, WN=2),
           metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, WM=4, WN=4),
       ],
       key=["M", "N", "K"],
   )(matmul)

   grid = lambda cfg, M=M, N=N: (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(N, cfg["BLOCK_N"]))
   autotuned_matmul[grid](A, B, C, M, N, K)

See :doc:`/guide/autotuning` for the full autotuning guide.


Concepts Introduced
-------------------

- ``metile.zeros``: register-resident accumulator initialization
- ``metile.dot``: tile-level matrix multiply-accumulate
- ``metile.tile_load`` / ``metile.tile_store``: 2D strided memory access
- 2D grids: ``kernel[(grid_m, grid_n)]``
- Fused epilogues: element-wise ops after GEMM are free
- Tile swizzle: cache-friendly scheduling patterns
