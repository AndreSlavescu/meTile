Compiler Architecture
=====================

meTile compiles Python kernels to Metal. Its graph frontend can also find
supported algorithm rewrites and combine operations before choosing kernel
boundaries. Both paths use Tile IR and Metal IR to describe the generated work.
The optional MLX backend compares generated kernels with native operations
before deciding which to run.

.. image:: /_static/compilation-pipeline.svg
   :alt: Full meTile compilation flow from graph and eDSL inputs through proof-carrying discovery, fusion, IR lowering, Metal code generation, and guarded runtime dispatch
   :width: 100%

Frontends and Graph Planning
----------------------------

There are two ways to enter the compiler:

* ``@metile.kernel`` traces ordinary Python eDSL operations into kernel-local
  Tile IR.
* ``GraphBuilder`` records a typed compute DAG before kernel boundaries have
  been selected. The graph can come from an MLX integration or from an explicit
  compiler client.

The graph path looks for supported subgraphs, including a query/key matmul,
scale or causal mask, softmax, and value matmul whose intermediate results have
no other users. A reduction certificate checks the algebra needed to replace
that chain with online attention. Fusion planning then estimates launch,
intermediate-storage, and target-resource costs. It uses min-cut for legal
neighborhoods and for bipartite components of the region-conflict graph;
non-bipartite components use a deterministic greedy selection. See
:doc:`graph-fusion` for the scope of that optimization.

Proof-Carrying Discovery
------------------------

Algorithm discovery defines reductions using reusable laws.

.. image:: /_static/algorithm-discovery.svg
   :alt: Proof-carrying algorithm discovery from graph pattern and reduction law through algebraic obligations to a certified rewrite or exact fallback
   :width: 100%

``ReductionLaw`` defines an identity, a binary merge, a singleton lift, and a
finalization expression. The restricted equational checker verifies left and
right identity, generated associativity cases, and pair homomorphism. The
weighted-softmax law summarizes a stream with ``(maximum, normalizer,
numerator)``; only a verified law may replace the materialized attention chain
with ``flash_attention``. These symbolic real-arithmetic checks do not establish
bitwise equivalence under floating-point reassociation. Escaping intermediates, incompatible shapes, an
unsupported reduction axis, or a failed obligation preserve the original graph.

Kernel Compilation
------------------

Tile IR contains hardware-independent loads, stores, loops, reductions, layouts,
and matrix operations. Compiler passes expose target-independent structure before
lowering. Static schedules are canonicalized under the finite symmetry group that
is legal for the concrete kernel shape; equivalent orbit members are not searched
twice. Candidate scalar schedule programs are extracted by a target operation-cost
model with a compressed-description-length tie break.

An immutable schedule plan precedes lowering. It resolves backend, SIMD-group
geometry and supported staging constraints without rewriting the traced Tile IR.
Materialization and optimization passes enforce expert requirements; the final
execution report records actual buffering, vectorization and allocations.
Descriptor GEMM pointwise epilogues lower from a typed scalar dependency DAG,
including shared branches and runtime coefficients, rather than activation-name
templates. See :doc:`execution-schedules` for the supported contracts and limits.

Checked per-value thread ownership lowers into identity forwarding, SIMD
shuffles or verified threadgroup exchanges. Canonical two-stage matrix pipelines
carry typed buffer slots and publication/recycle phases through IR, rather than
relying on an emitter flag. See :doc:`thread-layouts` for the deliberately bounded
layout subset. Explicit layouts support 1, 2, 4, 8, 16, or 32 scalar values per
thread; multi-register programs also support FP32 sums. The compiler keeps those
values live across the reduction and removes proven redundant conversions
before introducing communication.

Lowering produces explicit Metal IR primitives: scalar and SIMD operations,
threadgroup storage, simdgroup matrix fragments, Metal 4 ``tensor_ops``, block
scales, barriers, and multi-output stores. Decomposition and optimization passes
remain ordinary IR-to-IR transformations. The uniform emitter walks Metal IR to
produce MSL, then the runtime uses the offline Metal toolchain when available and
falls back to supported JIT compilation paths.

Experimental AIR emission and native AGX instruction rewriting are described in
:doc:`compiler-bypasses`. These research paths are separate from normal kernel
compilation and require device-specific correctness and performance checks.

Guarded Runtime
---------------

Compilation generates candidates; it does not assume that generated code beats
the framework for every device and shape.

.. image:: /_static/runtime-dispatch.svg
   :alt: Guarded meTile runtime selection across native MLX and generated kernels with validation, finalist tuning, persistent caching, fallback, and zero-copy dispatch
   :width: 100%

Native MLX and generated kernels enter the same selector. Generated results must
first pass a numerical compatibility gate. Provisional and finalist rounds
alternate candidate order. Switching margins depend on the operation family:
attention and RMSNorm require 5 percent, while graph fusion and block-scaled
matmul require 10 percent. Other projection families have their own margins;
see :doc:`mlx-backend`. Decisions persist by device, toolchain or framework
version, source, dtype, shape, launch grid, and candidate family. A cached native
decision deoptimizes directly to the original operation; a generated decision
uses zero-copy MLX custom kernels or prepared Metal dispatch with resource-safe
batching.

Unsupported calls, numerical failures, compilation failures, and inconclusive
timings retain native MLX. Reports identify that selection so a native fallback
is not counted as a generated-kernel win. The general ``metile.autotune`` API
only times configurations; its caller is responsible for numerical validation.
