Autotuning
==========

Different problem sizes benefit from different tile configurations. meTile's autotuner
benchmarks GPU timestamps and caches the fastest one per problem shape. Winners persist
across processes and are invalidated when the device, compiler toolchain, kernel source,
or candidate family changes.


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

1. Compiles every valid config
2. Benchmarks candidates in rotated, alternating round-robin order
3. Selects the fastest one, using generated-code size only for a sub-percent tie
4. Caches the result with the device and toolchain identity
5. Dispatches with the winning config

Subsequent calls with the same key values reuse the winner without re-tuning.

The cache defaults to ``~/Library/Caches/metile`` on macOS. Set
``METILE_CACHE_DIR`` to relocate it, or ``METILE_DISABLE_DISK_CACHE=1`` to disable
persistent autotune choices while debugging.

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
       SWIZZLE="hilbert",
   )

Any keyword arguments become constexprs passed to the kernel. Parameters not in the
kernel's signature are stored in ``func.constexprs`` and available to the compiler
(e.g., ``WM``, ``WN`` control the tensor_ops simdgroup layout).
Schedules can be searched alongside tile shapes with ``SWIZZLE="linear"``,
``"grouped2"``, ``"grouped4"``, ``"grouped8"``, ``"diagonal"``,
``"morton"``, ``"hilbert"``, or ``"auto"``.


Schedule Algebra and MDL
------------------------

Schedule selection is a composable Metal IR pass, not a whole-kernel template.
Each traversal is represented as a finite permutation of the launch grid. Square
grids use the eight-element dihedral group ``D4``; rectangular grids use the four
shape-preserving reflections. The pass canonicalizes candidates under these group
actions and searches one representative per orbit.

This is a finite symmetry group and fundamental-domain construction, not a
topological fundamental group. Likewise, exact Kolmogorov complexity is
uncomputable. meTile uses DEFLATE-compressed generated MSL length as a reproducible
minimum-description-length upper bound. Measured latency is always primary: MDL can
only choose a smaller representation when it is within 0.25% of the fastest result.


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

   from metile.runtime.metal_device import MetalDevice

   dispatch = autotuned_matmul[grid].prepare(A, B, C, M, N, K)

   # compatible calls batch until sync(), numpy(), or an ordinary launch flushes them
   for _ in range(1000):
       dispatch()

   MetalDevice.get().sync()

Prepared GEMMs use an ordered encoder. Independent element-wise kernels can use a
concurrent encoder; the runtime tracks input/output buffer hazards and inserts Metal
buffer barriers between dependent dispatches. Optional selectors are capability
checked, and bound buffers remain alive through completion.
