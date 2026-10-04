Compiler Architecture
=====================

meTile compiles Python kernels to Metal. A second frontend starts from a compute
graph, where it can rewrite supported algorithms and combine operations before
choosing kernel boundaries. Both paths lower through Tile IR and Metal IR. The
optional MLX backend then compares generated kernels with native operations
before choosing which to run.

.. image:: /_static/compilation-pipeline.svg
   :target: ../_static/compilation-pipeline.svg
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

The graph frontend recognizes supported subgraphs, including a query/key
matmul, scale or causal mask, softmax, and value matmul whose intermediate
results have no other users. Before replacing that chain with online attention,
it checks the required algebra through a reduction certificate.

Fusion planning estimates launch, intermediate-storage, and target-resource
costs. It uses min-cut for legal neighborhoods and for bipartite components of
the region-conflict graph. Non-bipartite components use a deterministic greedy
selection instead. :doc:`graph-fusion` explains where the optimization is exact
and where it is heuristic.

Proof-Carrying Discovery
------------------------

Algorithm discovery describes reductions through reusable algebraic laws.

.. image:: /_static/algorithm-discovery.svg
   :target: ../_static/algorithm-discovery.svg
   :alt: Proof-carrying algorithm discovery from graph pattern and reduction law through algebraic obligations to a certified rewrite or exact fallback
   :width: 100%

A ``ReductionLaw`` consists of an identity, a binary merge, a singleton lift,
and a finalization expression. The restricted equational checker verifies
left and right identity, generated associativity cases, and pair homomorphism.
The weighted-softmax law summarizes a stream with ``(maximum, normalizer,
numerator)``; only a verified law may replace the materialized attention chain
with ``flash_attention``. These symbolic real-arithmetic checks do not establish
bitwise equivalence under floating-point reassociation. If intermediates escape
the region, shapes are incompatible, the reduction axis is unsupported, or a
proof obligation fails, the compiler keeps the original graph.

Kernel Compilation
------------------

Tile IR contains hardware-independent loads, stores, loops, reductions, layouts,
and matrix operations. Compiler passes expose target-independent structure
before lowering. Once a kernel shape is fixed, the compiler canonicalizes static
schedules under their legal finite symmetry group. Instead of searching an
orbit of equivalent schedules, it picks one representative. It selects
candidate scalar schedule programs using a target operation-cost model, with
compressed description length as the tie break.

Before lowering, an immutable schedule plan resolves the backend, SIMD-group
geometry, and supported staging constraints without rewriting the traced Tile
IR. Materialization and optimization passes enforce expert requirements, while
the final execution report records the buffering, vectorization, and allocations
actually used.
Descriptor GEMM pointwise epilogues lower from a typed scalar dependency DAG,
including shared branches and runtime coefficients, rather than activation-name
templates. See :doc:`execution-schedules` for the supported contracts and limits.

Once per-value thread ownership has been checked, it lowers to identity
forwarding, SIMD shuffles, or verified threadgroup exchanges. Explicit layouts
support 1, 2, 4, 8, 16, or 32 scalar values per thread; multi-register programs
also support FP32 sums. The compiler keeps those values live across the
reduction and removes proven redundant conversions before introducing
communication. :doc:`thread-layouts` describes the supported layout subset.

Canonical two-stage matrix pipelines use typed buffer slots and
publication/recycle phases in the IR, rather than flags on the emitter.

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

Generating a kernel does not establish that it beats the framework on a given
device and shape.

.. image:: /_static/runtime-dispatch.svg
   :target: ../_static/runtime-dispatch.svg
   :alt: Guarded meTile runtime selection across native MLX and generated kernels with validation, finalist tuning, persistent caching, fallback, and zero-copy dispatch
   :width: 100%

Native MLX and generated kernels compete in the same selector. Generated
results must first pass a numerical compatibility gate. Provisional and
finalist rounds alternate candidate order. Switching margins depend on the
operation family:
attention and RMSNorm require 5 percent, while graph fusion and block-scaled
matmul require 10 percent. Other projection families have their own margins;
see :doc:`mlx-backend`. Decisions persist by device, toolchain or framework
version, source, dtype, shape, launch grid, and candidate family. A cached native
decision deoptimizes directly to the original operation; a generated decision
uses zero-copy MLX custom kernels or prepared Metal dispatch with resource-safe
batching.

The runtime keeps native MLX for unsupported calls, numerical or compilation
failures, and inconclusive timings. Reports make that choice explicit: a native
fallback is not a generated-kernel win. These checks belong to the MLX
integration. The general ``metile.autotune`` API only times configurations; its
caller is responsible for numerical validation.
