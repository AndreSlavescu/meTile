Compute-Graph Fusion
====================

``GraphBuilder`` records tensor operations before kernel boundaries are chosen.
``plan_graph_fusion`` then selects supported groups of operations to lower
together, including groups with more than one output.

The full graph-to-runtime flow, including proof-carrying attention discovery, is
shown in :doc:`architecture`.

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

The residual sum is a graph output. Fusion must preserve it as well as the
normalized result, so the selected kernel has two outputs.

Automatic Gated Epilogues
-------------------------

``GraphBuilder.silu`` and ``GraphBuilder.multiply`` express a SwiGLU feed-forward
block in the same graph IR. The default
``ParallelEpilogueRule`` recognizes two matmuls with a shared activation input, the
``silu(gate) * up`` epilogue, and its consuming down projection. The matcher accepts
either multiply-input order, rejects escaping intermediates, enforces register and
threadgroup-memory limits, and presents the complete region to backend lowering.

The affine-quantized Metal backend includes a low-register schedule for this region.
It completes the gate reduction, spills the SIMD-group result to threadgroup scratch,
reuses the accumulator lifetime for the up reduction, synchronizes, and applies SwiGLU
on-chip. Native MLX, compiled MLX, and the two generated schedules remain
independent autotune candidates. Generated reductions must pass a primitive
error check before model-level next-token, KL-divergence, and logit checks.

For one-row affine decode, a second composable epilogue adds the transformer residual to the
down projection in the same generated kernel. The runtime tunes this stage separately from
gate/up, then prepares an MLP executor from the two selected callables. Each
stage keeps its own native MLX candidate.

Related work includes the modular epilogues in `SonicMoE
<https://arxiv.org/abs/2512.14080>`_ with the parallel-operator co-optimization studied by
`Magneto <https://doi.org/10.1145/3744906>`_. On Apple silicon, threadgroup memory is an
explicit on-chip scratchpad. Dispatch order can encourage the resulting hidden tensor to
remain cache-hot for the down projection, but Metal does not expose L2 cache pinning; unified
memory removes a CPU/device copy rather than turning global storage into scratchpad memory.

Max-Flow Selection
------------------

Each legal producer/consumer neighborhood becomes an s-t cut network. Keeping a producer
separate cuts an edge weighted by launch and intermediate-materialization cost. Fusing it
cuts a target-resource edge. Infinite-capacity edges encode legality constraints. The
source-side vertices in the residual graph form the candidate region.

Candidate regions then form a weighted conflict graph: two vertices conflict when their
regions share an operation. The planner decomposes this graph into connected components.
Every bipartite component is selected globally and exactly by reducing maximum-weight
independent set to minimum-weight vertex cover and then to one s-t cut. Candidate benefit
becomes terminal capacity and overlap becomes an infinite-capacity edge. This avoids the
local failure mode where one attractive middle region blocks two outer regions whose
combined benefit is larger.

For non-bipartite components, the planner visits regions in descending estimated
benefit and keeps each one if it does not overlap an earlier selection. This
produces legal, non-overlapping regions but does not guarantee the maximum total
benefit. General weighted region selection is a set-packing problem and cannot
be reduced to the bipartite min-cut construction used here.

The in-tree flow solvers implement deterministic exact combinatorial algorithms,
using floating-point capacities with a ``1e-12`` residual tolerance.
Their optimality does not make the estimated cost model a measured performance
result. ``FlowNetwork`` keeps input capacities separate from the fresh residual
graph built for each solve, which makes differential
testing and repeated autotuning safe. Automatic dispatch uses Dinic directly for compiler
networks below 32 vertices. Larger networks enter a three-round, order-interleaved
tournament between Dinic and highest-label push-relabel with gap and global-relabeling
heuristics because density alone does not predict the winner reliably. A
topology-and-capacity cache then dispatches repeats directly. Push-relabel must demonstrate
at least 10 percent headroom to replace the reference solver, protecting the compiler from
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

Analytical graph cost only proposes a fusion. Framework backends still benchmark the fused
lowering against the unfused graph. The MLX backend uses a finalist tournament and requires
10 percent headroom before selecting a graph-fused kernel; otherwise it executes native MLX.
Projection and SwiGLU backends use separate selection policies, described in
:doc:`mlx-backend`.
