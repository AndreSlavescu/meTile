Execution Schedules and Fusion
==============================

Use ``metile.Schedule`` to constrain how a kernel executes. Tensor declarations
still describe the data; the schedule specifies requirements such as backend,
thread count, and buffering. The compiler chooses unspecified decisions and
reports an error when it cannot satisfy a requirement.

Compilation has three scheduling steps:

1. ``plan_schedule`` selects a legal backend, SIMD-group geometry and staging
   policy from Tile IR and target capabilities.
2. Lowering materializes that plan; reusable passes choose buffering,
   vectorization and traversal within its constraints.
3. Validation checks the materialized requirements, then an execution report
   records the resulting geometry, loops, allocations and passes.

The checked scalar per-value ownership subset is documented in
:doc:`thread-layouts`; it is not yet a general distributed-layout language or a
learned cost model. Backend defaults remain heuristics; autotuning compares
supported configurations with measured timings.

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
     - ``auto``, ``simdgroup``, ``tensor_ops``, ``nax`` or ``elementwise``.
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

Automatic GEMM staging uses the existing legal MPP/device or SIMD-group/shared
paths, not a general allocation-placement optimizer. Elementwise staging must
match its actual memory operations. Specialized producer/consumer and persistent
kernels retain their restricted schedules and reject controls they cannot honor.

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

Descriptor-based GEMM extracts a pure SSA dependency graph from the stored
value. Shared expressions can become part of the same epilogue.
For example, after a canonical ``metile.dot`` reduction:

.. code-block:: python

   scaled = accumulator * alpha + beta
   activated = metile.where(scaled > 0, scaled, scaled * 0.1)
   output.store((row_origin, column_origin), activated)

``alpha`` and ``beta`` may be independent runtime scalar parameters. Shared
subexpressions, comparisons, selections and supported arithmetic/unary math
lower into the accumulator epilogue without an intermediate device tensor.
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

This is pointwise epilogue fusion, not general matmul-to-matmul fusion. Checked
scalar layout conversions and canonical two-stage software buffer lifetimes now
have explicit contracts; see :doc:`thread-layouts`. General matrix-fragment
conversions, asynchronous copies and cooperative-tensor chains remain future
work. Their dependency and layout contracts must precede expert controls.

Tuning and measurement
----------------------

Tuning selection identity includes operand representation (type, dtype,
shape and strides), caller compile-policy overrides, configurations and grid,
source/compiler implementation fingerprints, relevant environment and target
toolchain. A winner measured for another dtype or numerical policy is not reused.
Caller overrides are applied consistently during preparation, launch and grid
evaluation. This cache safety does not replace numerical validation of a candidate.

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

The fusion benchmark validates against NumPy and MLX before timing. It compares
one fused dispatch with the same meTile matmul plus a separate pointwise dispatch,
and with ``mx.compile``. All use resident inputs and alternating synchronized
wall latency. meTile uses preallocated outputs while MLX manages its output
storage. This is an operation-boundary comparison, not isolated GPU instruction
throughput or a bitwise-equivalence claim. Report losses as well as wins.

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
