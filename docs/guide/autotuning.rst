Autotuning
==========

Different problem sizes benefit from different tile configurations. meTile's autotuner
benchmarks a set of configurations and caches the fastest one per problem shape.


Basic Usage
-----------

.. code-block:: python

   import metile
   from kernels.gemm import matmul

   autotuned_matmul = metile.autotune(
       configs=[
           metile.Config(BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=32,  WM=2, WN=2),
           metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,  WM=4, WN=4),
           metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, WM=4, WN=4),
       ],
       key=["M", "N", "K"],
   )(matmul)

``configs``
   A list of ``metile.Config`` objects. Each config is a set of constexpr values to try.

``key``
   The argument names that determine when to re-tune. When any key value changes,
   the autotuner re-benchmarks all configs.

Launching
---------

The grid must be a callable that computes the grid shape from the config:

.. code-block:: python

   grid = lambda cfg, M=M, N=N: (
       metile.cdiv(M, cfg["BLOCK_M"]),
       metile.cdiv(N, cfg["BLOCK_N"]),
   )

   autotuned_matmul[grid](A, B, C, M, N, K)

On the first call with new key values, the autotuner:

1. Benchmarks every config (warmup + timed iterations)
2. Selects the fastest one
3. Caches the result keyed by ``(kernel_name, key_values)``
4. Dispatches with the winning config

Subsequent calls with the same key values use the cached winner with zero overhead.

.. code-block:: text

   First call (M=1024, N=1024, K=1024):
   +--------------------------------------------------+
   |  Config(BM=64,  BN=64,  BK=32):   1.26ms         |
   |  Config(BM=128, BN=128, BK=64):   0.62ms  <--    |  winner cached
   |  Config(BM=128, BN=128, BK=128):  0.91ms         |
   +--------------------------------------------------+

   Subsequent calls (same M, N, K):
   +--------------------------------------------------+
   |  cached -> Config(BM=128, BN=128, BK=64)         |  no re-tuning
   +--------------------------------------------------+


Config Object
-------------

.. code-block:: python

   cfg = metile.Config(
       BLOCK_M=128,
       BLOCK_N=128,
       BLOCK_K=64,
       WM=4,
       WN=4,
       K_UNROLL=1,
   )

Any keyword arguments become constexprs passed to the kernel. Parameters not in the
kernel's signature are stored in ``func.constexprs`` and available to the compiler
(e.g., ``WM``, ``WN`` control the tensor_ops simdgroup layout).


Verbose Output
--------------

With ``verbose=True`` (the default), the autotuner prints results:

.. code-block:: text

   autotune matmul [M=1024, N=1024, K=1024]: Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, ...)
     Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, ...): 1.26ms
     Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, ...): 0.62ms <--
     Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, ...): 0.91ms

The ``<--`` marks the selected winner.

If a config fails (e.g., exceeds threadgroup memory limits), the error reason is shown:

.. code-block:: text

     Config(...): FAILED (LoweringError: GEMM requires 49152 bytes threadgroup memory ...)


Tuning Parameters
-----------------

.. code-block:: python

   metile.autotune(
       configs=[...],
       key=["M", "N", "K"],
       warmup=5,      # warmup iterations per config (default: 5)
       rep=20,         # timed iterations per config (default: 20)
       verbose=True,   # print results (default: True)
   )


Prepared Dispatch
-----------------

For latency-sensitive inference, use ``.prepare()`` to autotune once and get a
fast dispatcher that skips all Python overhead on subsequent calls:

.. code-block:: python

   dispatch = autotuned_matmul[grid].prepare(A, B, C, M, N, K)

   # hot path with minimal python overhead
   for _ in range(1000):
       dispatch()
