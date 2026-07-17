Compute-Graph Fusion
====================

meTile separates high-level graph partitioning from kernel-local Tile IR. A
``GraphBuilder`` records tensor operations without choosing kernel boundaries, and
``plan_graph_fusion`` selects legal multi-output regions before backend lowering:

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

Max-Flow Selection
------------------

Each legal rewrite neighborhood becomes an s-t cut network. Keeping a producer separate
cuts an edge weighted by launch and intermediate-materialization cost. Fusing it cuts a
target-resource edge. Infinite-capacity edges encode legality constraints. The source-side
vertices in the residual graph form the selected region.

The in-tree solver is deterministic and exact. Compiler fusion neighborhoods are normally
small, so a compact Dinic implementation has lower compile-time constants than the
almost-linear theoretical max-flow algorithms based on approximate minimum-ratio cycles
and dynamic graph structures. The solver is isolated from graph construction so a verified
large-graph implementation can replace it without changing fusion legality or lowering.

Measured Dispatch
-----------------

Analytical graph cost only proposes a fusion. Framework backends still benchmark the fused
lowering against the unfused graph. The MLX backend uses a finalist tournament and requires
10 percent headroom before selecting a graph-fused kernel; otherwise it executes native MLX.
