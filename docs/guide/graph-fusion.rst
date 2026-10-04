Compute-Graph Fusion
====================

``GraphBuilder`` records tensor operations before kernel boundaries are chosen.
``plan_graph_fusion`` then selects supported groups of operations to lower
together, including groups with more than one output.

See :doc:`architecture` for the full path from graph construction to runtime
dispatch, including proof-carrying attention discovery.

.. code-block:: python

   import metile

   builder = metile.GraphBuilder()
   values = builder.input("values", metile.TensorSpec((1, 2048), "f16"))
   residual = builder.input("residual", metile.TensorSpec((1, 2048), "f16"))
   weight = builder.input("weight", metile.TensorSpec((2048,), "f16"))
   summed = builder.add(values, residual)
   normalized = builder.rms_norm(summed, weight, 1e-5)
   graph = builder.build((summed, normalized))

   plan = metile.plan_graph_fusion(graph)

The residual sum is a graph output. Since fusion must preserve both it and
the normalized result, the selected kernel has two outputs.

Automatic Gated Epilogues
-------------------------

``GraphBuilder.silu`` and ``GraphBuilder.multiply`` express a SwiGLU feed-forward
block in the same graph IR. The default ``ParallelEpilogueRule`` recognizes two
matmuls with a shared activation input, the ``silu(gate) * up`` epilogue, and the
down projection that consumes it. The matcher accepts either multiply-input
order. It rejects escaping intermediates and enforces register and
threadgroup-memory limits before passing the complete region to backend
lowering.

The affine-quantized Metal backend includes a low-register schedule for this
region. It completes the gate reduction, spills the SIMD-group result to
threadgroup scratch, reuses the accumulator lifetime for the up reduction,
synchronizes, and applies SwiGLU on-chip. Native MLX, compiled MLX, and the two
generated schedules remain distinct candidates for autotuning. Generated
reductions must pass a primitive error check before model-level next-token,
KL-divergence, and logit checks.

For one-row affine decode, a second composable epilogue adds the transformer
residual to the down projection in the same generated kernel. The runtime tunes
this stage separately from gate/up, then prepares an MLP executor from the two
selected callables. Each stage retains its own native MLX candidate.

Related work includes the modular epilogues in `SonicMoE
<https://arxiv.org/abs/2512.14080>`_ and the parallel-operator co-optimization
studied by `Magneto <https://doi.org/10.1145/3744906>`_. On Apple silicon,
threadgroup memory is an explicit on-chip scratchpad. Dispatch order can help
keep the resulting hidden tensor cache-hot for the down projection, but Metal
does not expose L2 cache pinning. Unified memory removes a CPU/device copy; it
does not turn global storage into scratchpad memory.

Max-Flow Selection
------------------

Each legal producer/consumer neighborhood becomes an s-t cut network. Keeping
a producer separate cuts an edge weighted by launch and
intermediate-materialization cost; fusing it cuts a target-resource edge.
Infinite-capacity edges encode legality constraints. The source-side vertices
in the residual graph form the candidate region. The project-selection ideas
behind this construction are credited in :doc:`references`.

Candidate regions then form a weighted conflict graph: two vertices conflict
when their regions share an operation. The planner divides this graph into
connected components. For each bipartite component, it finds an exact
maximum-weight independent set by reducing the problem to minimum-weight
vertex cover and then to one s-t cut. Candidate benefit becomes terminal
capacity, and overlap becomes an infinite-capacity edge. This avoids a common
pitfall of greedy selection: choosing one middle region with high individual
benefit that blocks two outer regions with a larger combined benefit.

For non-bipartite components, the planner visits regions in descending estimated
benefit and keeps each one if it does not overlap an earlier selection. This
produces legal, non-overlapping regions but does not guarantee the maximum total
benefit. General weighted region selection is a set-packing problem and cannot
be reduced to the bipartite min-cut construction used here.

The in-tree flow solvers implement deterministic exact combinatorial algorithms,
using floating-point capacities with a ``1e-12`` residual tolerance.
Their optimality applies to the supplied capacities, not to the accuracy of the
performance estimates. ``FlowNetwork`` keeps input capacities separate from the
fresh residual graph built for each solve, allowing differential testing and
repeated autotuning without modifying the original network.

Automatic dispatch uses Dinic directly for compiler networks below 32 vertices.
Larger networks enter a three-round, order-interleaved tournament between Dinic
and highest-label push-relabel with gap and global-relabeling heuristics:
density alone does not reliably predict the winner. A topology-and-capacity
cache dispatches repeat solves directly. Push-relabel must show at least
10 percent headroom to replace the reference solver, limiting the effect of
timing noise and tail cases.

The `almost-linear directed-flow result <https://arxiv.org/abs/2203.00671>`_
uses a different algorithm; neither in-tree engine implements it. The flow
interface is separate from graph construction, so another solver can be tested
against Dinic without changing fusion legality or lowering.

Reproduce the current solver crossover with:

.. code-block:: console

   python -m benchmarks.compiler.max_flow

Measured Dispatch
-----------------

The cost model proposes a fusion; measurement decides whether to use it.
Framework backends benchmark the fused lowering against the unfused graph. The
MLX backend uses a finalist tournament and requires 10 percent headroom before
selecting a graph-fused kernel. Otherwise, it executes native MLX. Projection
and SwiGLU backends have separate selection policies, described in
:doc:`mlx-backend`.
