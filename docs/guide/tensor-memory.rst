Tensor and Memory Contracts
===========================

``metile.tensor`` declares a buffer's shape, strides, access permissions and
address space. The compiler checks accesses against those declarations when
lowering supported computations. To control execution separately, use
:doc:`execution-schedules` for schedule requirements and :doc:`thread-layouts`
for thread and register ownership.

Declare data before computation
-------------------------------

Declare a ``metile.tensor`` near the start of the kernel to describe each buffer:

.. code-block:: python

   source = metile.tensor(
       Input, shape=(rows, columns), access="read"
   )
   destination = metile.tensor(
       Output, shape=(rows, columns), access="write"
   )

   row = metile.program_id(0)
   column = metile.arange(0, BLOCK)
   values = source.load((row, column))
   destination.store((row, column), values * 2.0)

Without explicit strides, the view uses contiguous row-major storage. Strides
are measured in elements, not bytes, and both shape and stride expressions
can use kernel scalar arguments. A descriptor exists only as tracing metadata:
it neither allocates a buffer nor creates an object for every element.

The indexing path uses signed 32-bit coordinates and element offsets.
Shapes, strides, and intermediate offset calculations must fit that arithmetic
and describe an existing allocation. Statically known overflowing dimensions,
strides, offsets, and reachable address spans are rejected; dynamic arithmetic
remains a caller contract. Bounds checks enforce the declared logical shape;
they cannot establish the capacity of an arbitrary pointer's allocation.

``load(indices, other=0)`` and ``store(indices, value)`` accept one index per
dimension. Scalar coordinates can be combined with equally shaped vector
coordinates. Loads mask negative and upper-bound coordinates and fill them
with the supplied scalar; stores mask invalid coordinates. This is indexing
with matching coordinate shapes, not NumPy broadcasting or a general tensor
slice syntax.

Use ``access="read"``, ``"write"``, or ``"readwrite"`` to declare allowed
operations. The address space is inferred from the pointer; an explicit
``address_space`` must match it. A declaration does not move storage between
device and threadgroup memory. The access mode also does not imply that two
different pointers cannot alias.

For matrix operations, an optional ``block_shape=(tile_rows, tile_columns)``
selects tiled loads and stores:

.. code-block:: python

   left = metile.tensor(
       Left, shape=(rows, inner), access="read",
       block_shape=(BLOCK_M, BLOCK_K),
   )
   right = metile.tensor(
       Right, shape=(inner, columns), access="read",
       block_shape=(BLOCK_K, BLOCK_N),
   )
   output = metile.tensor(
       Output, shape=(rows, columns), access="write",
       block_shape=(BLOCK_M, BLOCK_N),
   )

   left_tile = left.load((row_origin, inner_origin))
   right_tile = right.load((inner_origin, column_origin))
   accumulator = metile.dot(left_tile, right_tile, accumulator)
   output.store((row_origin, column_origin), accumulator)

This is a fragment, not a complete GEMM. The surrounding kernel must supply
tile origins, initialize the accumulator and loop over the reduction dimension
before storing the result.
The tiled path requires two dimensions, positive compile-time block
sizes, scalar origins, device storage, and zero padding. GEMM lowering has
additional supported-pattern checks: contiguous row-major operands with matching
FP16 or FP32 storage types, an FP32 zero-initialized accumulator, one canonical
top-level reduction loop, and a supported epilogue. Matrix dimensions must be
scalar parameters or constants. Operand binding follows the actual ``dot`` and
store dataflow, so parameter names and pointer order do not determine semantics.
Padded or transposed matrix strides, mixed storage types, noncanonical origins,
additional side effects, and arithmetic inserted inside the reduction are
rejected. General strided elementwise access does not imply general strided GEMM
support. Ragged matrix dimensions use the existing SIMD-group tail path when
they cannot use the aligned Metal 4 path.

The existing ``TensorDescriptor(M, N, K)`` has a different purpose: it chooses
an MMA atom. It does not declare the shape or strides of a buffer. Likewise,
the graph frontend's ``TensorSpec`` and runtime ``TensorView`` remain separate
interfaces. Use ``metile.tensor`` for kernel buffer declarations.

Three separate compiler facts
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Fact
     - Meaning
     - Responsibility
   * - Logical tensor
     - Shape, dtype, read/write permissions, indexing and padding semantics
     - Declared by the program and checked by the compiler
   * - Memory layout
     - Element strides, storage address space, allocation and view relationships
     - External storage is explicit; supported matrix staging is compiler selected
   * - Execution layout
     - Which SIMD group, lane and register owns each logical element
     - Selected by an inspectable plan, with checked expert constraints

A transpose changes the logical-to-memory map. Moving values between lanes
changes the execution layout. Staging values in threadgroup memory changes
their storage and requires synchronization. The IR must keep those changes
distinct even when an optimization performs all three.

Tensor declarations and memory-operation metadata preserve logical and memory
contracts in Tile IR. ``ThreadLayout`` adds a checked execution layout to
supported scalar values. The current subset covers register/thread bit
permutations; arbitrary distributed layouts and matrix-fragment conversions
remain unsupported.

Descriptor-based GEMM lowering follows the matrix operation's operands and
output stores. Pointer positions, names such as ``M`` and ``N``, or a tiled
load alone do not establish GEMM semantics. A fast path must preserve the
program it replaces. Unsupported patterns must produce a diagnostic rather
than drop operations.

Legality before optimization
----------------------------

The descriptor path preserves per-access bounds rather than guarding a whole
loop with whichever mask appears first. The current elementwise split pass
removes masks only when it proves they hold throughout the aligned interior.
Independent tensor extents, shifted coordinates, and separate row bounds
retain their checks. The scalar tail retains the original masks and fill
values. This lets supported normalization kernels keep vectorized interiors
without treating every descriptor as contiguous and in bounds.

Vectorization also needs adjacent logical elements to occupy adjacent memory
addresses. Valid indices do not prove contiguity: a stride of two must not
become a four-element contiguous load. The current pass rejects surviving
masks, nonunit lane strides and lane expressions it cannot safely rescale.

Matrix-backend selection needs the same checks. Shape alignment alone does
not prove a reduction loop safe. After choosing fragments and unrolling, the
backend must either prove that K is divisible by the full K step or provide
a correct masked tail. Alignment to 32, for example, does not justify an
unmasked K step of 96.

Precision requirements must reach the actual matrix-operation descriptor.
Checking only a Python configuration flag is insufficient. The generic MPP and
NAX fragment paths propagate ``RELAXED_PRECISION`` through their Metal IR
operations to the emitted descriptor. The current packed NAX FP32 layout is
only validated for relaxed mode: requesting strict FP32 NAX fails with a
diagnostic instead of silently relaxing precision. Strict FP32 remains
available through general MPP lowering; strict FP16 NAX is supported.
``RELAXED_PRECISION=False`` disables the selected descriptor's relaxed mode;
it does not establish bit-exact behavior. The normal
offline runtime still enables fast-math, and reduction order can differ from
MLX. Record the generated path and numerical tolerance with each comparison.

The general autotuner's selection cache includes operand dtype, shape and
strides, caller precision overrides and other compile-policy values in its
identity. Numerical contracts cannot share a cached selection merely because
their declared tuning-key dimensions match. The fixed benchmark still uses
explicit configurations for reproducibility; incompatible packed-NAX requests
are rejected rather than executed with silently relaxed precision.

IR transformations must preserve value identity across regions. An aligned
loop and its tail need independent copies of local operations while retaining
references to values defined outside the loop. Otherwise later common
subexpression elimination can remove a parent value that a detached copy still
references. This requirement is exercised by the migrated normalization tests.

Related compiler designs
-------------------------

Meta's TLX, Triton Low-level Language Extensions, exposes explicit buffers
while inferring their physical layouts from consumers. Its
``local_alloc`` separates allocation from buffer views
and loads; optional layout requirements let experts constrain selected
values. This is a useful model for top-level memory declarations with
compiler-selected defaults. See the primary
`TLX memory reference <https://facebookexperimental.github.io/triton/website/buffers.html>`_
and `layout-control reference <https://facebookexperimental.github.io/triton/website/layouts.html>`_.

Gluon exposes thread, warp, and register distribution through explicit tensor
layouts. Its lower-level interface is useful as a design reference for an
expert escape hatch; requiring those details for every meTile value would
transfer too much scheduling work into ordinary kernels. See Triton's
`Gluon overview <https://triton-lang.org/main/gluon/index.html>`_
and `tensor layout tutorial <https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html>`_.

Meta's AutoWS design separates scheduling decisions from the passes that
materialize partitions, storage, and synchronization. In meTile, a schedule
plan is also separate from the lowering that constructs executable operations.
Keeping these steps separate lets us inspect and test the plan. See
`the compiler pipeline <https://facebookexperimental.github.io/triton/website/triton.html>`_.

An expert control must either enforce a requirement or state that it is a
preference. Unsupported requirements should fail compilation. The compiler
may override preferences with an explanation, but should never silently ignore
an accepted control.

Placement and fusion on Apple GPUs
----------------------------------

Apple's MPP guide recommends direct device access for GEMM and explains that
thread occupancy can overlap memory and computation without explicit software
pipelining. It also identifies tile size, traversal order, static interior
extents, and occasional K-loop synchronization as tuning opportunities.
These are candidate policies, not guarantees for every dtype or workload.
See the `MPP programming guide
<https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>`_.

Automatic placement remains future work; there is no ``placement=auto`` tensor
option. Such a policy would need measured costs and live ranges to compare
direct loads, value reuse and staged storage. Quantized unpacking and irregular
access may need different policies from dense GEMM.

Read-only inspection and offline compilation on October 2, 2026 confirmed
that the installed macOS 26.2 SDK and Metal compiler 32023.864 accept a
Metal 4.0 shader using cooperative-tensor row reduction and a cooperative
matmul result as a second matmul's input. This was a compilation check,
not a runtime correctness or performance result. The installed MPP headers
impose these constraints on that path:

* Cooperative inputs and row reductions use one SIMD group.
* The consuming matmul has a static reduction dimension.
* Producer and consumer dimensions and element types must agree.
* The cooperative input cannot be transposed; layout compatibility must be checked.

The probe also found that constructing the tensor from ``const device float``
failed the SDK's cooperative-input type checks while a mutable tensor view
compiled. Keep logical read permissions in the IR even when the backend
needs a mutable view to satisfy this SDK interface.

These capabilities provide a starting point for exploring
matmul-to-elementwise-to-matmul fusion. They do not establish that arbitrary
meTile chains can stay in cooperative tensors. Incompatible layouts still
need an explicit conversion or a supported memory path.

Pipeline and synchronization requirements
-----------------------------------------

TLX's asynchronous descriptor loads specify a source descriptor, destination
buffer, tile origin, and completion barrier. Its copy-group operations track
completion explicitly. See the
`async memory reference <https://facebookexperimental.github.io/triton/website/async-memory.html>`_.
These interfaces are useful for expressing dependencies, but their
NVIDIA/AMD mechanisms do not establish equivalent public Apple instructions.

Before exposing asynchronous staging in meTile, the IR must represent:

* Read/write effects and aliases of each allocation and view.
* Completion dependencies between a transfer and its consumers.
* Buffer reuse only after every previous consumer has finished.
* Execution scope and uniform participation for each collective operation.
* Pipeline depth constrained by storage capacity and register pressure.

The compiler should derive barriers and rotating-buffer indices from these
dependencies. If an asynchronous request is unsupported, it must be rejected
or use a documented synchronous implementation with identical semantics.
A normal load must not be described as an asynchronous hardware copy. Explicit
SIMD-group roles alone do not provide a general pipeline planner.

Current coverage and next steps
-------------------------------

The implemented foundation includes tensor declarations, bounds/access checks,
canonical descriptor GEMM, pure pointwise epilogues, and inspectable schedule
plans. :doc:`thread-layouts` describes the supported per-value ownership maps,
FP32 register reductions, and verified two-stage software buffering.

Further work includes strided/transposed matrix operands, broader layout
conversion, allocation placement, cooperative-tensor chains, and asynchronous
staging. Each extension needs address, precision, and lifetime checks before
performance tuning. Device-specific costs also need measurements on each Apple
GPU generation.

Reproduce the tensor-kernel comparison
--------------------------------------

.. code-block:: bash

   python3 -m benchmarks.compiler.tensor_memory --output-json /tmp/metile-tensor-memory.json
   python3 -m benchmarks.compiler.tensor_memory --family gemm --rep-ms 200
   python3 -m benchmarks.compiler.tensor_memory --metile-root /path/to/baseline --output-json /tmp/baseline.json

The fixed suite checks FP32 RMSNorm and LayerNorm at aligned and ragged widths,
plus FP16 GEMM at 256 and 1024 square dimensions. Both implementations must
match a NumPy reference and each other before timing. Inputs are resident;
meTile uses prepared dispatch and preallocated output, while MLX manages its
output storage. Alternating execution order measures synchronized wall latency,
with compilation and setup recorded separately.

The October 2, 2026 M5 experiment changed the general FP16 GEMM path to use MPP
cooperative tensors for supported noncooperative 32-by-32 SIMD-group tiles.
FP32 accumulation is converted explicitly to FP16 at the final store through
the SDK's logical element coordinates. Unsupported shapes retain the bounded
SIMD-group path.

With relaxed precision disabled and 64-by-64-by-16 tiles, the recorded GPU
speedups over the earlier compiler were 6.61x at size 256 and 5.17x at size
1024. The separate wall-time comparison with MLX was about 1.08x and 0.95x,
respectively: the compiler improvement did not make both cases faster than MLX.
Normalization was approximately at parity.

See :doc:`benchmarks` for the measurements and artifact links, including
``m5-fp16-tensor-ops.json``, ``m5-tensor-memory.json``, and
``m5-tensor-memory-baseline.json``. The records preserve source hashes and
different timing regimes. Use ``benchmarks/regression/paired_regression.py``
for cross-revision performance checks; a single current-versus-MLX run does
not replace that comparison.
