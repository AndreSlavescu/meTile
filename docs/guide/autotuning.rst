Autotuning
==========

Tile configurations perform differently as problem sizes change.
``metile.autotune`` compiles and times a candidate list, then caches its
selection. The cache includes the device, toolchain, kernel and compiler
sources, candidate list, launch grid, operand type, dtype, shape, strides and
relevant compile-policy overrides. Matching the explicit tuning key alone
isn't enough to reuse a cached result.

The general autotuner only measures latency; it doesn't check outputs against a
reference. Validate every configuration before tuning, especially if precision
or reduction order changes. The optional MLX backends have their own numerical
checks.


Basic Usage
-----------

.. code-block:: python

   import metile
   from metile_kernels.gemm import matmul

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
   Argument names that identify the workload. A new key value triggers tuning
   unless a matching cached selection exists.

Launching
---------

When tile sizes vary, provide a callable to compute the grid for each configuration.
Here ``M``, ``N``, and ``K`` are the matrix dimensions, and ``A``, ``B``, and
``C`` are their input and output buffers:

.. code-block:: python

   grid = lambda cfg, M=M, N=N: (
       metile.cdiv(M, cfg["BLOCK_M"]),
       metile.cdiv(N, cfg["BLOCK_N"]),
   )

   autotuned_matmul[grid](A, B, C, M, N, K)

On the first call with new key values, the autotuner:

1. Attempts to compile every configuration, recording failures
2. Benchmarks candidates in rotated, alternating round-robin order, recording both
   GPU timestamps and synchronized launch-to-completion latency
3. Re-benchmarks up to eight candidates within 8% of the provisional winner in a
   30-round rotating finalist tournament
4. Selects the lowest measured latency, allowing compressed generated-code size
   to break a tie within 0.25%
5. Caches the result and measured latency with the device and toolchain identity
6. Dispatches with the winning config

Later calls with the same key values and compilation contract use the cached
winner without re-tuning.

For kernels measured at one millisecond or less, selection uses synchronized
end-to-end latency, including command encoding and completion handling. These
take a significant share of that time. Longer throughput kernels use GPU
timestamps to avoid host-side scheduling noise. The winner's raw GPU latency
is saved separately and determines its prepared-dispatch completion policy.

The cache defaults to ``~/Library/Caches/metile`` on macOS. Set
``METILE_CACHE_DIR`` to relocate it, or ``METILE_DISABLE_DISK_CACHE=1`` to disable
persistent autotune choices while debugging. The following timings illustrate
cache behavior; they are not benchmark results:

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

Keyword arguments become kernel constexprs. Parameters outside the kernel's
signature are stored in ``func.constexprs`` for the compiler; for example,
``WM`` and ``WN`` control the tensor_ops SIMDgroup layout.
The reserved ``num_simdgroups`` constructor argument defaults to ``None``
(automatic). Setting it adds a checked ``NUM_SG`` requirement. Caller launch
overrides take precedence over candidate values for both compilation and grid
evaluation. Configurations can include ``SCHEDULE=metile.Schedule(...)``;
see :doc:`execution-schedules`.
Schedules can be searched alongside tile shapes with ``SWIZZLE="linear"``,
``"grouped2"``, ``"grouped4"``, ``"grouped8"``, ``"diagonal"``,
``"morton"``, ``"hilbert"``, or ``"auto"``.
``SCHEDULE_ENCODING="arithmetic"`` or ``"bitwise"`` exposes equivalent decoder
representations as ordinary autotune candidates; the default ``"auto"`` uses the
compiler's target-cost and MDL extractor.
On the aligned M5 NAX path, ``NAX_OUTER_K`` controls the reduction epoch and
``NAX_K_UNROLL=2`` preloads two 16-wide K fragments before issuing their native MMAs.
Tune these parameters for the target shape: increasing the amount of live data
can outweigh the benefit of fewer loads or fences.
``NAX_SKIP_FIRST_EPOCH_BARRIER`` retains every inter-epoch scheduling fence but skips
the redundant fence before the first epoch. It is searched as a separate representation
because the uniform predicate helps medium reductions but the unconditional form can
remain faster for long reductions.
``NAX_TRAILING_EPOCH_BARRIER`` moves the same inter-epoch fence to the end of each
non-final epoch. The tuner measures this placement as another candidate.
The block-scaled runtime also measures a paired reduction representation that reuses
one E8M0 scale load across the two 16-wide steps in each 32-value quantization group.
It executes the decoded weight fragments sequentially, avoiding the register-pressure
cost of dense-style fragment preloading. Small aligned shapes additionally search a
``32x64`` two-SIMDgroup tile, which provides finer occupancy than the conventional
four-SIMDgroup ``64x64`` tile on the base M5.


Schedule Algebra and MDL
------------------------

Schedule selection runs as a Metal IR pass rather than choosing whole-kernel
templates. It represents each traversal as a finite permutation of the launch
grid. Starting from a small set of reflections and axis exchanges, it derives
the exact shape-preserving action: ``D4`` for interchangeable square grid and
tile axes, ``D2`` for ordinary rectangles or anisotropic square tiles, ``C2``
for degenerate one-axis grids, and the trivial group for a single tile.
It verifies that action by constructing orbits and stabilizers, canonicalizes
the traversals and searches one representative per orbit.

Each static traversal lowers to a scalar schedule-expression program. Exact
rewrites can replace power-of-two multiplication, division and remainder with
shifts and masks. The extractor first minimizes a Metal operation-cost model,
then breaks ties deterministically using the DEFLATE-compressed canonical
expression encoding as a minimum-description-length metric. Code generation
uses the selected expression tree directly; adding a decoder representation
doesn't require another whole-kernel template.

For cross-kernel autotuning, meTile uses DEFLATE-compressed generated MSL
length as a reproducible description-length metric. It is an approximation,
not an evaluation of Kolmogorov complexity. Measured latency is primary: MDL
can only select a smaller representation within 0.25% of the fastest result.

The same approach extends beyond GEMM traversal. The FFT candidate family uses
one kernel written with ordinary eDSL operations. It searches threadgroup
width, the number of register-local radix-2 stages, bit-reversed gather versus
shared scatter, and global versus threadgroup twiddle placement. Native
``reverse_bits`` lowering makes the permutation decoder branch-free without
a host-generated index table.

Decode attention applies the same policy across algorithms. Short and highly parallel
shapes search one online-softmax threadgroup per head. Long contexts additionally search
a multi-threadgroup partial pass followed by an online merge pass. Both kernels remain
ordinary eDSL programs, and the persisted winner is keyed by head grid, context length,
head dimension, device, toolchain, source, and candidate family.

The optional MLX backend also includes native MLX as a candidate. Attention and
RMSNorm require 5% headroom before selecting generated Metal; other families
have their own margins. A custom primitive can change graph scheduling even
when isolated timings are close. See :doc:`mlx-backend` for those policies.


Verbose Output
--------------

With ``verbose=True`` (the default), the autotuner prints results in this form.
The values here are illustrative:

.. code-block:: text

   autotune matmul [M=1024, N=1024, K=1024]: Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, ...)
     Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, ...): 1.26ms
     Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, ...): 0.62ms <--
     Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, ...): 0.91ms

The ``<--`` indicates which configuration was selected.

If a config fails (e.g., exceeds threadgroup memory limits), its failure reason is printed:

.. code-block:: text

     Config(...): FAILED (LoweringError: GEMM requires 49152 bytes threadgroup memory ...)


Tuning Parameters
-----------------

.. code-block:: python

   metile.autotune(
       configs=[...],
       key=["M", "N", "K"],
       warmup=5,
       rep=20,
       verbose=True,
   )

These are the defaults: five warmup rounds and twenty timed rounds per
configuration, followed by finalist remeasurement when needed.


Prepared Dispatch
-----------------

For repeated inference, use ``.prepare()`` to autotune once and bind a
dispatcher. Later calls skip tracing, lowering, compilation and argument
conversion:

.. code-block:: python

   from metile.runtime.metal_device import MetalDevice

   dispatch = autotuned_matmul[grid].prepare(A, B, C, M, N, K)

   dispatch.repeat(1000)

   MetalDevice.get().sync()

``repeat`` encodes repeated work under one runtime lock. Compatible calls batch
until ``sync()``, ``numpy()``, or an ordinary launch flushes them. Each repetition
uses the same bound buffers; account for any in-place updates in the kernel.

Prepared GEMMs use an ordered encoder. Independent element-wise kernels can
use a concurrent encoder; the runtime tracks input/output buffer hazards and
inserts Metal buffer barriers between dependent dispatches. Multi-buffer
kernels use one cached ``setBuffers:offsets:withRange:`` call instead of
repeated Objective-C bindings.

Compatible dispatches reuse unchanged pipeline and buffer state within the
shared encoder. ``repeat(count)`` also avoids repeated Python lock transitions
when encoding the same operation many times. The runtime checks support for
optional selectors and keeps bound buffers alive until completion.

The autotuner stores the selected kernel's measured GPU latency with its device-
and toolchain-specific configuration. Prepared kernels measured at one millisecond or
less receive a completion-poll budget derived from that latency: three times the GPU
time plus 300 microseconds, bounded between 900 and 1500 microseconds. Longer kernels
keep the blocking ``waitUntilCompleted`` path. A command buffer containing an
unclassified dispatch also blocks, and batches beyond eight dispatches never spin.

This measured policy covers short reductions and rectangular GEMMs without relying on
a square-shape heuristic. Directly configured small static GEMMs retain a conservative
900-microsecond fallback. Set ``METILE_LOW_LATENCY_SPIN_US=0`` to disable active
waiting, or provide a non-negative microsecond cap for application-specific
latency/power tradeoffs; the default cap is 1500 microseconds.
