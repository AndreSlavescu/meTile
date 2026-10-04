Thread Ownership and Software Staging
======================================

Tensor declarations describe logical memory; thread layouts assign each logical
value to a GPU thread. Keeping these separate lets the compiler arrange
communication without requiring pointer arithmetic, scratch arrays and barriers
in every kernel.

Explicit layouts support 1, 2, 4, 8, 16, or 32 scalar values per thread in
straight-line programs with tensor descriptors. Multi-register paths also
support FP32 sum reductions. These controls cover a limited subset; existing
kernels keep their automatic execution path.

Declare memory first, then ownership
-------------------------------------

This batched transpose reads contiguous input elements, redistributes ownership,
and writes contiguous output elements:

.. code-block:: python

   import metile

   @metile.kernel
   def transpose8(source, destination, batches, OUTPUT_LAYOUT: metile.constexpr):
       inputs = metile.tensor(source, shape=(batches, 8, 8), access="read")
       outputs = metile.tensor(destination, shape=(batches, 8, 8), access="write")
       batch = metile.program_id(0)
       positions = metile.arange(0, 64)
       values = inputs.load((batch, positions // 8, positions % 8))
       converted = metile.convert_layout(values, OUTPUT_LAYOUT)
       owned = metile.arange(0, 64, layout=OUTPUT_LAYOUT)
       outputs.store((batch, owned % 8, owned // 8), converted)

   transpose_layout = metile.ThreadLayout((3, 4, 5, 0, 1, 2))
   dispatch = transpose8[(batches,)].prepare(
       source, destination, batches, OUTPUT_LAYOUT=transpose_layout,
   )
   print(dispatch.explain())

``convert_layout`` preserves the logical values; the changed output coordinates
perform the transpose. It does not change tensor shape, strides or address space.
The compiler inserts the required temporary storage and barriers.

``ThreadLayout(bit_order, xor_mask=0, elements_per_thread=1)`` is an immutable
bijection. Physical coordinates pack thread bits first, followed by register-slot
bits. Entry ``bit_order[logical_bit]`` selects the packed physical bit placed at
that logical bit position; ``xor_mask`` is then applied to the logical index. For example,
``ThreadLayout((1, 0, 2, 3, 4))`` exchanges the first two index bits within a
32-thread SIMD group. ``ThreadLayout.identity(size)`` constructs identity
ownership. ``logical_index(thread, register=0)``, ``owner(index)`` and
``register(index)`` expose the forward and inverse maps. ``size`` counts logical
elements; ``thread_count`` counts physical threads. The representation supports
bit permutations plus an XOR constant, rather than arbitrary binary linear
transformations.

Arithmetic, comparisons, selections, casts and descriptor loads preserve
ownership. Uniform scalars broadcast. To combine tiles with different owners,
use an explicit ``convert_layout``; an implicit layout means identity when
combined with an explicit one. Legacy operations that report a nonuniform
scalar lose per-thread ownership. The compiler rejects those values rather
than treating them as uniform broadcasts.

Compiler-selected communication
--------------------------------

The source-to-destination owner map determines the implementation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Ownership change
     - Lowering
   * - None
     - Forward the existing SSA value; no communication.
   * - Within each SIMD group
     - ``simd_shuffle`` with the exact source lane, not a uniform broadcast.
   * - Across SIMD groups
     - Store to typed threadgroup scratch, publish with a threadgroup barrier,
       load from the source owner, then release with a second barrier.

Every launched thread takes part in an exchange, whether or not its logical
device load or store is in bounds. Later conversions of the same dtype can
reuse scratch only after the release barrier. After the pass, a verifier checks
the allocation, owner map, geometry, unconditional barriers and access sequence.
``execution_report.value_layouts`` records per-value ownership;
``layout_conversions`` records each chosen mechanism and scratch identity.
``allocations`` includes the actual byte count.

The supported geometry is a power-of-two, one-dimensional tile on 32 through
1024 physical threads. Each thread owns a power-of-two number of scalar values
from 1 through 32: a tile contains ``thread_count * elements_per_thread`` logical
elements, up to 32768. Tile sizes and register counts must agree within a
kernel. Origins must be uniform integer scalars and tensor base pointers must
be uniform. Converted types are ``f16``, ``f32``, ``i32``, ``u32``
and ``bool``. Device-backed tensor views are required. Loops, matrix
operations, raw lane collectives and shared tensor views are not supported in
this subset. Multi-register tiles must carry explicit ownership; an unannotated
tile is not silently reinterpreted as a multi-register tile. Only identity
conversions are currently supported for multi-register layouts. Explicit vector
widths other than one reject;
``Schedule(staging="device")`` cannot conceal a required cross-group exchange.

The 1024-thread ceiling is an IR limit, not a guarantee that every compiled
kernel can launch that many threads. meTile checks the compiled pipeline's
``maxTotalThreadsPerThreadgroup`` before caching or dispatching it and raises
``OutOfResources`` when the requested geometry exceeds that limit. Reduce the
tile's physical thread count or register pressure explicitly; ownership is
never silently remapped. See Apple's `pipeline threadgroup limit
<https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup>`_.

Register-tiled reductions
-------------------------

``ThreadLayout.identity(1024, elements_per_thread=4)`` launches 256 threads.
Thread ``thread`` owns logical indices ``thread + register * 256`` for register
slots zero through three. Moving the two register bits to the low logical
positions, with ``ThreadLayout((8, 9, 0, 1, 2, 3, 4, 5, 6, 7),
elements_per_thread=4)``, instead gives each thread four adjacent elements.
This is logical register ownership, not a promise of specific hardware register
numbers, vector memory instructions, occupancy or spill-free machine code.

The compiler lowers ordinary pointwise operations to one SSA value per register
slot, preserving each slot's address, bounds mask, fill and dtype. An FP32
``sum`` combines local values in a balanced tree, then uses the SIMD/threadgroup
reduction primitive. Every thread participates, including those with masked
device accesses. One SIMD group needs no shared storage; larger groups allocate
one FP32 partial per SIMD group. The uniform result can broadcast into later
tile arithmetic. Other reductions and FP16 accumulation are rejected; cast to
FP32 before summing.

Lowering exposes the verified physical-thread range to Apple's compiler by
masking the thread index to ``thread_count - 1`` before adding register offsets.
This leaves every legal thread index unchanged while exposing zero bits that
simplify ownership permutations and eliminate redundant bounds checks. The
compiler checks the launch geometry; it does not assume these bounds for an
arbitrary caller-supplied index.

For private reduction scratch, every SIMD group reads the published partials
and independently computes the final sum. This removes the second barrier and
shared-result broadcast from the legacy reduction. The initial threadgroup
publication barrier remains mandatory. The verifier requires a dedicated,
immutable-after-publication allocation, rejecting later writers, pointer aliases,
other users and unrecorded requests for this mode. Each sequential reduction
gets separate scratch, so no release barrier is needed for buffer reuse.

The opt-in ``rmsnorm_register`` kernel retains its input values across the
reduction instead of reloading them for the normalization pass. Its default
remains four values per thread; an explicit ``LAYOUT`` selects another legal
register/thread partition:

.. code-block:: python

   from metile_kernels.rmsnorm import rmsnorm_register

   dispatch = rmsnorm_register[(rows,)].prepare(
       inputs, weights, output, 1e-5,
       N=1009, BLOCK=1024, RELAXED_PRECISION=False,
   )

For example, ``LAYOUT=metile.ThreadLayout.identity(1024, elements_per_thread=16)``
uses 64 threads and sixteen values per thread. With 32 values per thread,
the same logical tile uses one SIMD group and no reduction scratch or
threadgroup barrier. This trades parallelism for per-thread work and live
values; fewer threads are not inherently faster. The compiler still owns
scalarization, reduction trees, scratch allocation and barrier placement.

``N`` and ``BLOCK`` are compile-time values; ``0 < N <= BLOCK`` is required.
The supplied buffers must contain ``rows * N`` input/output elements and ``N``
weights. Tensor views remain declared at kernel entry, with uniform row offsets.
The kernel uses FP32 reduction and epilogue arithmetic, rounding only on the
output store. Existing ``rmsnorm`` retains its runtime-width, loop-based path.
The same register lowering handles pointwise programs and sequential FP32 sums.

``execution_report.register_reductions`` records the operation, dtype, physical
thread count, register count and scratch identity. Post-pass validation checks
uniform placement, FP32 semantics, geometry and storage. Per-value ownership
remains available in ``value_layouts``.

Contiguous register memory
--------------------------

Within a single tensor load or store, the compiler can group four FP16 or FP32
accesses when their final indices are contiguous for every physical thread.
The proof includes the register/thread permutation, uniform pointer offsets
and arithmetic strides. Striped ownership, nonunit strides or unsupported
index expressions retain scalar accesses. No access moves across another
tensor memory operation, including operations on pointers that might alias.

Grouped loads produce typed vector values and explicit scalar extracts, keeping
the original scalar SSA results. Stores preserve each value's storage cast.
The emitted packed vector path requires all four masks to hold and checks
consecutivity using widened indices; the latter also protects against wrapping
32-bit runtime origins. Otherwise each original scalar mask, index and fill is
used independently. This does not speculatively read beyond a ragged tensor.

``packed_float4`` and ``packed_half4`` need only scalar alignment (four and two
bytes respectively), unlike native ``float4`` and ``half4``. This permits
shifted scalar pointers and ragged row origins without an undocumented
vector-alignment promise. See Apple's `Metal Shading Language specification,
sections 2.2.3 and 2.5
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_.

``execution_report.grouped_loads`` and ``grouped_stores`` count guarded emitted
memory groups. These are not counts of final GPU vector instructions: Apple's
compiler may scalarize packed accesses. ``Schedule(vector_width=1)`` disables
grouping while preserving ownership and range propagation. Forced width four
still rejects for explicit ownership because grouping is conditional, not a
guarantee that every access can be vectorized. A late verifier checks types,
geometry, contiguity and vector extraction dominance after optimization.

Eliminating redundant conversions
---------------------------------

Before planning, the compiler validates ownership and simplifies adjacent pure
layout conversions. Identity conversions disappear, inverse pairs cancel, and
compatible chains compose into a single conversion. An intermediate conversion
with other users remains live. SSA references, including tensor metadata, are
rewritten on a cloned IR; unchanged programs are returned without cloning.
``execution_report.layout_optimizations`` records the proofs and counts.

The pass removes a barrier only when it proves the communication is redundant.
It does not remove other release barriers, move communication across memory
effects, support nonidentity multi-register conversions or relax staging
lifetime checks.

Verified two-stage matrix pipelines
-----------------------------------

The SIMD-group GEMM path represents software double buffering in typed IR.
Request it through the schedule interface:

.. code-block:: python

   schedule = metile.Schedule(
       backend="simdgroup", staging="threadgroup", double_buffer=True,
   )

The supported canonical matrix K-loop has two independent operand buffers and
two slots per buffer. Its ``SoftwarePipeline`` records typed storage and the
following dependency sequence:

1. Copy the first tiles and publish them with a threadgroup barrier.
2. Copy the next tiles into free slots while the current slots remain readable.
3. Consume the current tiles, then publish the next slots and release the current
   slots with a barrier before rotating the slot identities.
4. Consume and release the final tiles. An empty K dimension skips the pipeline.

Validation checks copy ownership, fragment bounds, uniform control flow,
publication, slot reuse, types and memory capacity. An unsupported loop is not
partially rewritten. Materialization is idempotent; only the operand allocations
are duplicated, not unrelated scratch. The emitter consumes verified phases
without mutating the original loop. ``execution_report.pipelines`` exposes this
contract and the allocation report exposes its storage cost.

This is synchronous software staging, not hardware-asynchronous copying or a
guarantee of load/compute overlap. Where automatic selection already chooses
native MPP/device staging, it still does; shared-memory staging is not assumed
faster. Specialized producer/consumer kernels retain their separate, restricted
implementation and are not covered by this proof.

Research and remaining work
----------------------------

`Triton's Gluon layout tutorial
<https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html>`_
separates logical tensors from distributed ownership. Its broader
`LinearLayout representation
<https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h>`_
and the `Linear Layouts paper <https://arxiv.org/abs/2505.23819>`_ motivate an
explicit algebra for register, lane and group ownership. meTile currently models
only a bijective register/thread-bit subset; it does not yet model replicated
values, arbitrary register counts or arbitrary linear maps.

Apple's `Metal Performance Primitives programming guide
<https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>`_
describes device-memory tensor operations and the tradeoffs between reuse,
threadgroup storage and occupancy. This is why native tensor operations and
explicit scalar communication remain separate choices rather than forcing every
kernel through shared-memory templates. The research-only AIR and native AGX
routes remain separate; see :doc:`compiler-bypasses`.

Further work includes composing more reductions and layouts, redistributing
multi-register tiles, choosing conversions by cost and chaining native tensors.
Each needs correctness and performance evidence before it can change defaults.

Measurements and reproduction
-----------------------------

:doc:`benchmarks` collects charts, raw artifacts, and detailed methodology.
These experiments measure different changes and use different baselines.

Ownership conversion and software staging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m benchmarks.compiler.thread_layouts --output-json /tmp/thread-layouts.json
   python3 -m benchmarks.compiler.thread_layouts --scatter-only --metile-root /path/to/pre-change/tree --output-json /tmp/thread-layouts-baseline.json
   python3 -m benchmarks.compiler.staged_gemm --baseline-root /path/to/pre-change/tree --output /tmp/staging.json

The ownership benchmark materializes batched transposes with ownership
conversion, scalar scatter, and MLX contiguous transpose. It checks bitwise
correctness, including signed zeros, before alternating synchronized wall-time
measurements. Inputs are resident; meTile preallocates outputs while MLX manages
its storage. The 16-case M5 manifest covers FP16/FP32, four tile shapes, and two
batch sizes. Ownership conversion has geometric-mean ratios of 0.992x scalar
scatter and 0.985x MLX, so it showed no speedup on this manifest. The path remains
opt-in. The separate-process baseline samples are not a controlled interleaved
compiler comparison.

The eight-case staging comparison requests the SIMD-group backend and double
buffering on both revisions, with identical 64-by-64-by-16 tiles and relaxed
precision disabled. All 32 executions pass the stated NumPy-reference
tolerances. In ``m5-verified-staging.json``, baseline/current GPU ratios range
from 0.960x to 1.017x and wall ratios from 0.974x to 1.002x. The checked lifetime
implementation incurs up to about 4.2% GPU-time overhead in this manifest.
These differences do not isolate the cost of any individual barrier.

Register RMSNorm: loop-based baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m benchmarks.compiler.register_rmsnorm --baseline-root /path/to/pre-register-tree --rounds 3 --warmup-ms 100 --rep-ms 250 --output /tmp/register-rmsnorm.json

This experiment compares the static-width register kernel with the earlier
runtime-width loop kernel. The driver exports frozen baseline MSL and its ABI,
recompiles it through the candidate's Metal toolchain, and alternates resident
dispatches in one process. Width specialization and input retention both differ
between the arms, so a gain cannot be attributed solely to register retention.

``m5-register-rmsnorm-final.json`` is the corrected record: all 36 shader
compilations use the same offline path and ``-O2 -ffast-math`` flags. Ragged
throughput cases improve, but three aligned cases miss the required 1.10x GPU
speedup against both controls. The promotion gate fails and the kernel remains
opt-in.

The earlier ``m5-register-rmsnorm.json`` and
``m5-register-rmsnorm-broadcast.json`` mix a JIT-compiled frozen baseline with
offline-compiled candidates. They remain available with caveats, but their
frozen-baseline ratios are **invalid as controlled compiler speedups**. The
corrected driver rejects mixed compilation paths.

The primary MLX comparator is a compiled FP32 normalization graph with one
final cast. The separate ``mx.fast.rms_norm`` arm differs for FP16: MLX 0.32.0
rounds normalized values before the weight multiply, while this kernel keeps
both operations in FP32 until storage. Each arm is checked against its own
reference. See the `versioned MLX RMSNorm implementation
<https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/rms_norm.metal>`_.
MLX ratios use synchronized wall time; meTile command-buffer timestamps are
not an MLX GPU-time measurement.

Bounded tiling: prior register baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m benchmarks.compiler.rmsnorm_tiling export --root /path/to/pre-tiling-tree --output /tmp/prior-register4.json
   python3 -m benchmarks.compiler.rmsnorm_tiling tune --baseline-json /tmp/prior-register4.json --manifest /tmp/tiling-selection.json --output /tmp/tiling-tuning.json
   python3 -m benchmarks.compiler.rmsnorm_tiling validate --baseline-json /tmp/prior-register4.json --manifest /tmp/tiling-selection.json --output /tmp/tiling-heldout.json
   python3 -m benchmarks.compiler.rmsnorm_tiling validate --baseline-json /tmp/prior-register4.json --manifest /tmp/tiling-selection.json --output /tmp/tiling-heldout-repeat.json

This later experiment freezes the previous static-width, four-register kernel.
Both arms retain inputs and use the same offline toolchain and flags. A tuning
run chooses one policy for all twelve dtype/width/batch cases; two fresh-process
validation runs use a different input seed and reject changed source or tuning
evidence.

The nine candidates vary striped ownership over 2/4/8/16/32 values per thread
and four-adjacent-element ownership over 4/8/16/32. The selected policy remains
striped ownership with four values per thread and scalar memory accesses.
Larger ownership and packed-memory candidates did not replace it.

On aligned width 1024, the two held-out runs record **1.070–1.079x FP16 GPU
speedup** over the prior register compiler. FP32 stays near parity at
1.000–1.024x. Both runs pass the 3% wall and ragged-GPU regression guards, but
all four aligned throughput cases miss the 10% improvement requirement.
**The promotion gate fails.** Four-value ownership remains the register
kernel's default; larger layouts stay explicit experiments, and the loop-based
``rmsnorm`` remains unchanged.

The AIR inspection records fewer conditional branches after exposing the
physical-thread range: 14 to 2 for width 1024 and 14 to 5 for width 1009.
These are intermediate-IR counts. They neither count final AGX instructions
nor isolate the cause of every timing change.

The artifacts are ``m5-rmsnorm-tiling-selection.json``,
``m5-rmsnorm-tiling-heldout.json``, ``m5-rmsnorm-tiling-heldout-repeat.json``,
and ``m5-rmsnorm-tiling-air.json`` in ``benchmarks/results/``. Across both
held-out runs, matched-policy compiled-MLX wall ratios range from 0.992x to
1.144x. Fast-MLX ratios range from 0.980x to 1.021x with geometric means below
one and the FP16 rounding caveat above. These repeated point estimates do not
establish a general MLX win; compilation latency is excluded.
