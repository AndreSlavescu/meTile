Compute-Graph Fusion
====================

meTile separates high-level graph partitioning from kernel-local Tile IR. A
``GraphBuilder`` records tensor operations without choosing kernel boundaries, and
``plan_graph_fusion`` selects legal multi-output regions before backend lowering:

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

The residual value remains a graph output, so the selected lowering is a multi-output
kernel rather than a rewrite that changes program semantics.

Automatic Gated Epilogues
-------------------------

``GraphBuilder.silu`` and ``GraphBuilder.multiply`` let the same graph IR express a
SwiGLU feed-forward block without prescribing a kernel template. The default
``ParallelEpilogueRule`` recognizes two matmuls with a shared activation input, the
``silu(gate) * up`` epilogue, and its consuming down projection. The matcher accepts
either multiply-input order, rejects escaping intermediates, enforces register and
threadgroup-memory limits, and presents the complete region to backend lowering.

The affine-quantized Metal backend includes a low-register schedule for this region.
It completes the gate reduction, spills the SIMD-group result to threadgroup scratch,
reuses the accumulator lifetime for the up reduction, synchronizes, and applies SwiGLU
on-chip. Native MLX, compiled MLX, register-fused meTile, and scratch-spilled meTile remain
independent autotune candidates. Scale-normalized primitive error gates generated
reductions before model-level next-token, KL-divergence, and logit checks.

For one-row affine decode, a second composable epilogue adds the transformer residual to the
down projection in the same generated kernel. The runtime tunes this stage separately from
gate/up, then builds a warmed MLP executor from the two selected callables. This preserves the
mainloop-plus-epilogue abstraction: changing the residual operation does not require cloning a
whole MLP kernel template, and native MLX remains a candidate at both boundaries.

This combines the modular epilogue idea used by `SonicMoE
<https://arxiv.org/abs/2512.14080>`_ with the parallel-operator co-optimization studied by
`Magneto <https://doi.org/10.1145/3744906>`_. On Apple silicon, threadgroup memory is an
explicit on-chip scratchpad. Dispatch order can encourage the resulting hidden tensor to
remain cache-hot for the down projection, but Metal does not expose L2 cache pinning; unified
memory removes a CPU/device copy rather than turning global storage into scratchpad memory.

Max-Flow Selection
------------------

Each legal rewrite neighborhood becomes an s-t cut network. Keeping a producer separate
cuts an edge weighted by launch and intermediate-materialization cost. Fusing it cuts a
target-resource edge. Infinite-capacity edges encode legality constraints. The source-side
vertices in the residual graph form the selected region.

The in-tree solvers are deterministic and exact. ``FlowNetwork`` stores an immutable
capacity graph and builds a fresh residual graph per solve, which makes differential
testing and repeated autotuning safe. Automatic dispatch uses Dinic directly for compiler
networks below 32 vertices. Larger networks enter a three-round, order-interleaved
tournament between Dinic and highest-label push-relabel with gap and global-relabeling
heuristics because density alone does not predict the winner reliably. A
topology-and-capacity cache then dispatches repeats directly. Push-relabel must demonstrate
at least 10 percent headroom to replace the reference solver, protecting the compiler from
timing noise and tail cases.

The `almost-linear directed-flow result <https://arxiv.org/abs/2203.00671>`_ based on an
interior-point method, approximate minimum-ratio cycles, and dynamic graph structures is a
separate solver implementation, not a label for either engine above. The flow boundary is
isolated from graph construction so that implementation can be differential-tested against
Dinic before it participates in automatic dispatch, without changing fusion legality or
lowering.

Reproduce the current solver crossover with:

.. code-block:: console

   python benchmarks/max_flow.py

Measured Dispatch
-----------------

Analytical graph cost only proposes a fusion. Framework backends still benchmark the fused
lowering against the unfused graph. The MLX backend uses a finalist tournament and requires
10 percent headroom before selecting a graph-fused kernel; otherwise it executes native MLX.
