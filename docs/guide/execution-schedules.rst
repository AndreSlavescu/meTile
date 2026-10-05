Execution Schedules and Fusion
==============================

Use ``metile.Schedule`` to specify execution requirements such as the backend,
thread count and buffering. Tensor declarations still describe the data.
The compiler picks the remaining settings and reports an error if it cannot
meet your requirements.

Compilation has three scheduling steps:

1. ``plan_schedule`` selects a legal backend, SIMD-group geometry and staging
   policy from Tile IR and target capabilities.
2. Lowering implements the plan; reusable passes choose buffering,
   vectorization and traversal within its constraints.
3. Validation checks the materialized requirements, then an execution report
   records the resulting geometry, loops, allocations and passes.

See :doc:`thread-layouts` for checked ownership of individual scalar values.
That subset is not a general distributed-layout language or a learned cost
model. Backend defaults use heuristics; autotuning compares supported
configurations by measuring them.

Checked expert controls
-----------------------

.. code-block:: python

   import metile
   from metile_kernels.gemm import matmul

   schedule = metile.Schedule(
       backend="simdgroup",
       num_simdgroups=4,
       staging="threadgroup",
       vector_width=4,
       double_buffer=False,
   )
   dispatch = matmul[grid].prepare(
       left, right, output, rows, columns, inner,
       BLOCK_M=64, BLOCK_N=64, BLOCK_K=16,
       RELAXED_PRECISION=False,
       SCHEDULE=schedule,
   )
   print(dispatch.explain())
   dispatch()

Here ``grid`` covers the output tiles, and the buffers contain row-major
matrices with dimensions ``rows``, ``columns``, and ``inner``.
The vector-width requirement in this example needs provably aligned
interior loads; arbitrary ragged shapes may reject it. Omitting ``SCHEDULE``
retains ordinary automatic selection. ``Schedule()`` also leaves every policy
automatic, but, like any explicit schedule, specializes integer scalar values
to support exact shape proofs in the compilation cache.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Control
     - Contract
   * - ``backend``
     - ``auto``, ``simdgroup``, ``simdgroup_inline``, ``tensor_ops``, ``nax`` or ``elementwise``.
       Explicit choices must support the kernel, hardware and precision policy.
   * - ``num_simdgroups``
     - An integer from 1 through 32. Matrix tiling must admit that factorization;
       elementwise kernels must already have a matching traced lane geometry.
   * - ``staging``
     - ``auto``, ``device`` or ``threadgroup``. Controls supported matrix
       temporaries; it never casts or moves a declared tensor view.
   * - ``vector_width``
     - ``None`` selects automatically, ``1`` disables compiler vectorization,
       and ``4`` requires vectorized aligned interiors. Masked tails may remain
       scalar. Opaque MPP/NAX accesses cannot satisfy an explicit width.
   * - ``double_buffer``
     - ``None`` selects automatically. ``True`` requires a materialized software
       double buffer; ``False`` prohibits it. Unsupported combinations reject.

With ``backend="auto"``, an explicit vector width or required double buffering
selects the controllable SIMD-group path instead of the opaque tensor backend.
Currently width four and required double buffering cannot be combined.
Threadgroup-memory capacity and shape checks still apply. Legacy ``NUM_SG``
and ``WM``/``WN`` settings are checked against the selected geometry and any
``Schedule`` requirement. ``Config(num_simdgroups=...)`` now sets ``NUM_SG``;
its default is automatic.

Automatic GEMM staging chooses between the supported MPP/device and
SIMD-group/shared paths; it does not optimize arbitrary allocation placement.
Elementwise staging must match the kernel's memory operations. Specialized
producer/consumer and persistent kernels keep their restricted schedules and
reject controls they cannot honor.

Composable matrix fragments
---------------------------

``Schedule(backend="simdgroup_inline")`` lets an ordinary DSL kernel contain
multiple matrix products, scalar work and explicit shared-memory exchanges.
It does not recognize an attention model or replace the kernel with a template.
Each SIMD-group owns an opaque 8-by-8 register fragment; ``dot`` lowers to a
SIMD-group matrix multiply with an FP32 accumulator.

For example, this single-group kernel computes ``max(A @ B, 0) @ B``:

.. code-block:: python

   @metile.kernel
   def two_products(A, B, Output, *, BLOCK: metile.constexpr = 32):
       left_memory = metile.shared(64, dtype="f32")
       right_memory = metile.shared(64, dtype="f32")
       result_memory = metile.shared(64, dtype="f32")
       left = metile.tensor(A, shape=(8, 8), access="read")
       right = metile.tensor(B, shape=(8, 8), access="read")
       output = metile.tensor(Output, shape=(8, 8), access="write")
       left_values = metile.tensor(left_memory, shape=(8, 8))
       right_values = metile.tensor(right_memory, shape=(8, 8))
       result_values = metile.tensor(result_memory, shape=(8, 8))
       left_matrix = metile.tensor(left_memory, shape=(8, 8), block_shape=(8, 8))
       right_matrix = metile.tensor(right_memory, shape=(8, 8), block_shape=(8, 8))
       result_matrix = metile.tensor(result_memory, shape=(8, 8), block_shape=(8, 8))
       for index in metile.tile_range(metile.thread_id(), 64, BLOCK):
           position = (index // 8, index % 8)
           left_values.store(position, left.load(position))
           right_values.store(position, right.load(position))
       metile.barrier()
       left_fragment = left_matrix.load((0, 0))
       right_fragment = right_matrix.load((0, 0))
       first = metile.dot(left_fragment, right_fragment, metile.zeros((8, 8)))
       second = metile.dot(metile.maximum(first, 0.0), right_fragment, metile.zeros((8, 8)))
       result_matrix.store((0, 0), second)
       metile.barrier()
       for index in metile.tile_range(metile.thread_id(), 64, BLOCK):
           position = (index // 8, index % 8)
           output.store(position, result_values.load(position))

   two_products[(1,)](
       left_buffer, right_buffer, output_buffer,
       STRICT_MATH=True,
       SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
   )

All three buffers in this example hold 8-by-8 FP32 matrices. Matrix operands
may instead use FP16 shared storage, but both inputs to each ``dot`` must have
the same dtype. Accumulators remain FP32; casting a fragment or storing it to
FP16 shared memory is an explicit storage conversion, not an automatic
reduced-precision optimization.

The shared-memory contract deliberately stays narrow:

* Matrix loads and stores use complete, provably in-bounds 8-by-8 shared tiles.
  Scalar DSL loads can fill padding before the matrix operation. Row-major,
  padded-row-major and transposed views are supported; the declared view must
  fit its allocation. Device tiles use the separate scratch contract below.
* Matrix origins and surrounding loops must be uniform within each SIMD-group.
  A threadgroup barrier additionally requires uniform control flow across the
  entire threadgroup. Shared allocations belong at the top of the kernel.
  Scalar shared-memory pointers must resolve to those allocations through
  pointer offsets only; conditional shared-pointer selection is unsupported.
* Explicit barriers publish scalar writes before matrix reads, publish matrix
  stores before scalar reads, and finish matrix reads before shared storage is
  overwritten. The compiler checks these transitions, including loop backedges.
  Provably disjoint regions of one shared allocation do not need a barrier
  between them. Unknown offsets and lossy index conversions conservatively
  count as overlapping the allocation. A loaded fragment retains its values
  when the shared source is reused after the required barrier.
  The kernel remains responsible for initializing the data it reads and keeping
  different groups' output regions disjoint.
* ``metile.loop_state`` can retain a fragment across iterations. Pointwise
  arithmetic, floating casts and supported unary math operate on its elements
  without exposing their lane mapping. Scalar operands must be SIMD-uniform.
  Row reductions currently use a shared-memory bridge to ordinary scalar/SIMD
  operations. Expression ``vjp`` does not differentiate matrix products or a
  complete mutable attention pipeline.

Set ``BLOCK`` to a multiple of 32. The kernel assigns work to SIMD-groups;
whole-GEMM ``WM``/``WN`` placement, forced vectorization and automatic double
buffering are unsupported here. The original ``simdgroup`` and ``tensor_ops``
GEMM backends are unchanged.

Direct device tiles and masked tails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An 8-by-8 device tile can load directly into a register fragment or store a
fragment back to device memory. Pass an explicit shared tensor as ``scratch``
for incomplete tiles. Complete tiles use Metal's matrix load/store operations;
tails use bounded scalar copies and SIMD-group barriers through the scratch
allocation. Neither path assumes which matrix elements belong to each lane.

This kernel adds one to the valid rows of an FP32 matrix with eight columns:

.. code-block:: python

   @metile.kernel
   def masked_tiles(Source, Control, Output, *, BLOCK: metile.constexpr = 32):
       scratch = metile.tensor(
           metile.shared(BLOCK * 2, dtype="f32"),
           shape=(BLOCK // 4, 8),
       )
       control = metile.tensor(Control, shape=(1,), access="read")
       rows = control.load((0,))
       source = metile.tensor(
           Source, shape=(rows, 8), strides=(8, 1),
           block_shape=(8, 8), access="read",
       )
       output = metile.tensor(
           Output, shape=(rows, 8), strides=(8, 1),
           block_shape=(8, 8), access="write",
       )
       for row in metile.tile_range(0, rows, 8):
           fragment = source.load((row, 0), scratch=scratch)
           output.store((row, 0), fragment + 1.0, scratch=scratch)

   masked_tiles[(1,)](
       source_buffer, control_buffer, output_buffer,
       STRICT_MATH=True,
       SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
   )

``control_buffer`` contains one nonnegative int32 row count. The input and
output buffers hold at least that many rows and must be disjoint. Invalid
load coordinates become zero; invalid stores leave the allocation untouched.
For example, a row count of nine processes one complete tile and one partial
tile without reading or writing any later rows.

The scratch tensor must reference a top-level shared allocation directly,
with shape ``(BLOCK // 4, 8)``, strides ``(8, 1)``, read/write access and the
same storage dtype as the device tensor. This reserves 64 elements for each
SIMD-group. Reuse it across consecutive tile accesses, but do not access that
allocation through scalar operations, another matrix view or an offset alias.
The compiler owns its publication and reuse barriers. FP16 and FP32 device
storage are supported; each ``dot`` still accumulates in FP32.

Device views require rank two and constant, positive row-major or
column-major strides. Their bases, logical extents and tile origins may be
runtime values, but must be uniform within each SIMD-group. Declare the
**logical valid extent**, not the allocation capacity. In attention, masking
a probability to zero does not make an invalid value safe: ``0 * NaN`` is
still NaN, so a KV view must exclude uninitialized cache positions before
the matrix load.

A device allocation used by matrix operations cannot be both read and written
within the same kernel, including through scalar aliases. Compilation rejects
that case even with ``metile.barrier()``, which only publishes threadgroup
memory. Distinct pointer parameters must also reference disjoint input/output
storage at launch; parameter names cannot prove that buffers do not overlap.

For eligible zero-start signed loops, the compiler separates provably complete
iterations from a guarded remainder. Every device tile access must satisfy
the proof; unrelated extents, uncertain offsets and nested loops keep their
guards. Scalar and fragment loop state continues across both portions.

Inspect the materialized kernel
-------------------------------

Both compiled kernels and prepared dispatchers expose:

* ``schedule_plan``: immutable pre-lowering decisions and selection reasons.
* ``execution_report``: the plan plus actual passes, loop widths, threadgroup
  allocations and byte counts, buffering and fused epilogue regions.
* ``explain()``: the execution report as formatted JSON.

Set ``METILE_DEBUG=schedule`` to print plans and reports and save JSON under
``debug_output/schedule``. Existing MSL and ISA inspection modes remain separate.
Loop widths describe emitted Metal operations, not a promise about final GPU
instructions. In loop reports, ``elements_per_lane`` is the emitter's loop
vectorization factor, not ownership of a matrix accumulator fragment;
``lanes`` counts launched x-dimension threads, not the active subset within a
specialized producer/consumer role. The SDK does not expose the native tensor
backend's exact lane/register mapping, so the report marks it as opaque.
It does not measure register pressure or occupancy.

Bounds are part of the schedule contract. In particular, removing a GEMM K-loop
mask must not discard row or column tail checks. Unproven outer tiles retain
guarded loads even when K alone is aligned. Tests cover ragged matrices with
and without software double buffering and independent elementwise bounds.

Pointwise GEMM epilogues
------------------------

Descriptor-based GEMM builds the epilogue from the stored value's pure SSA
dependencies, including expressions used more than once. For example,
after a canonical ``metile.dot`` reduction:

.. code-block:: python

   scaled = accumulator * alpha + beta
   activated = metile.where(scaled > 0, scaled, scaled * 0.1)
   output.store((row_origin, column_origin), activated)

``alpha`` and ``beta`` may be independent runtime scalar parameters. Shared
subexpressions, comparisons, selections and supported arithmetic/unary math
lower directly into the accumulator epilogue, avoiding an intermediate
device tensor.
Runtime coefficients are captured at kernel entry under compiler-generated
names, before backend locals or dimension aliases can shadow their bindings.
The same scalar program serves SIMD-group, MPP, and NAX emitters.

The accumulator and its tile arithmetic remain FP32 until the final output
conversion. Only supported pure operations enter the program. Additional memory
reads, reductions, control flow, incompatible casts, multiple matrix recurrences
and multiple output stores are not general fusion support. Dimension parameters,
including canonical ``M``/``N``/``K``, currently cannot serve as epilogue scalar
leaves because static specialization and dimension aliasing do not yet preserve
their bindings across every backend. Unsupported descriptor programs fail
compilation rather than losing operations.

This supports pointwise epilogues, not general matmul-to-matmul fusion. Scalar
layout conversions and canonical two-stage software buffering have checked
contracts; see :doc:`thread-layouts`. General matrix-fragment conversions,
asynchronous copies and cooperative-tensor chains remain future work. Before
exposing these controls, their dependencies and layouts need explicit contracts.

Tuning and measurement
----------------------

The tuning cache identifies operands by type, dtype, shape and strides. It
also includes caller compile-policy overrides, configurations, grid,
source/compiler fingerprints, relevant environment and target toolchain. A
winner measured for another dtype or numerical policy is not reused. Caller
overrides apply consistently to preparation, launch and grid evaluation.
These cache checks don't replace numerical validation of each candidate.

Reproduce the strict FP32 affine/leaky-selection fusion comparison with:

.. code-block:: bash

   MLX_ENABLE_TF32=0 python3 -m benchmarks.compiler.schedule_fusion --sizes 64 256 1024 --output /tmp/fusion.json
   python3 -m benchmarks.regression.paired_regression --baseline-root /path/to/pre-change/tree --rounds-per-sample 5

The fusion benchmark requires ``MLX_ENABLE_TF32=0`` before process startup,
following MLX's `numerical precision guidance
<https://ml-explore.github.io/mlx/build/html/usage/precision.html>`_. FP32 storage
alone does not establish full-precision matrix arithmetic. An initial run with
MLX's default policy failed the strict tolerance gate and was not timed; the
benchmark does not widen tolerances to hide that mismatch.

The fusion benchmark checks NumPy and MLX agreement before timing. It compares
one fused dispatch, the same meTile matmul followed by a separate pointwise
dispatch, and ``mx.compile``. All variants reuse resident inputs; execution
order alternates while measuring synchronized wall latency. meTile preallocates
outputs; MLX manages its own output storage. The comparison covers the full
operation, not isolated GPU instruction throughput, and does not claim bitwise equality.
Report regressions as well as improvements.

Recorded results
----------------

The two October 2, 2026 M5 runs recorded about 1.02–1.07x synchronized wall
speedup over separate meTile launches and about 1.03–1.08x over strict-FP32 MLX
on the three square sizes. Every timed variant passed the unchanged
``rtol=3e-4, atol=3e-5`` check. These are repeated point estimates for one
operation boundary, not confidence intervals or a comparison with MLX's
reduced-precision default.

The reports are in ``benchmarks/results/m5-schedule-fusion.json`` and
``m5-schedule-fusion-repeat.json``. Each reported ratio uses its own
order-alternated pair; absolute latencies from different pairs can reflect
different device states. The smaller margins do not consistently clear the
5% primitive switching threshold.

The scheduling regression artifact also retains two failed initial comparisons
and the probes used to investigate them. The flagged 256-square GEMM emitted
byte-identical MSL on both revisions. Interleaving both dispatch classes on the
same pipeline found similar wall time despite large shifts in absolute latency.
A five-round-per-sample ABBA rerun passed the unchanged 15% regression threshold
for all ten cases, with changes from 9.4% faster to 5.0% slower. This clears the
recorded regression check without attributing timing noise to compiler changes.

See :doc:`benchmarks` for the result tables, raw artifacts, and reproduction
details. ``m5-schedule-regression.json`` preserves the failed runs as well as the
expanded comparison.
