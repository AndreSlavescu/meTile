Fused Activations and SIMDgroup Roles
=======================================

Pointwise fusion computes related expressions in one kernel. Here, two
float32 arrays produce a gated activation:

``output = gelu_approx(gate) * up``, with
``gelu_approx(value) = value / (1 + exp(-1.702 * value))``.

This sigmoid approximation is sometimes called QuickGELU. It differs from
both the exact Gaussian-error-function GELU and its common tanh approximation.
Use the same definition when checking against a model or reference library.

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def geglu(gate_ptr, up_ptr, output_ptr, count, BLOCK: metile.constexpr):
       gates = metile.tensor(gate_ptr, shape=(count,), access="read")
       ups = metile.tensor(up_ptr, shape=(count,), access="read")
       outputs = metile.tensor(output_ptr, shape=(count,), access="write")
       positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
       gate = gates.load((positions,))
       up = ups.load((positions,))
       activated = gate / (1.0 + metile.exp(-1.702 * gate))
       outputs.store((positions,), activated * up)

   count = 1003
   rng = np.random.default_rng(0)
   gate_data = rng.standard_normal(count).astype(np.float32)
   up_data = rng.standard_normal(count).astype(np.float32)
   gate_buffer = metile.Buffer(data=gate_data)
   up_buffer = metile.Buffer(data=up_data)
   output_buffer = metile.Buffer.zeros((count,), dtype=np.float32)
   geglu[(metile.cdiv(count, 256),)](
       gate_buffer, up_buffer, output_buffer, count, BLOCK=256
   )
   reference = gate_data / (1.0 + np.exp(-1.702 * gate_data)) * up_data
   np.testing.assert_allclose(output_buffer.numpy(), reference, rtol=1e-5, atol=1e-6)

Each lane computes the activation and product before storing once. No program
needs to read another program's partially computed output. For SiLU, use
``value / (1.0 + metile.exp(-value))``.

GEMM epilogues
--------------

A supported pointwise expression after a ``dot`` recurrence can be fused
into its epilogue. The :doc:`matmul` example includes a complete GEMM-plus-ReLU
kernel; the same mechanism supports expressions such as this QuickGELU
approximation when they satisfy the compiler's epilogue constraints.

Fusion avoids an intermediate device-memory round trip, but exponentials,
divisions and extra live values still take time and resources. Inspect
``dispatch.explain()`` and the generated MSL to see the selected implementation.

When to use SIMDgroup roles
-----------------------------

``metile.simdgroup_role(role, num_roles=2, num_sgs=0)`` records a region
assigned to a subset of the threadgroup's 32-thread SIMDgroups. This is a
lower-level tool for specialized kernels, not a dependency or synchronization
mechanism.

Python source order does not make one role finish before another starts.
If one role reads another's writes, you must arrange the storage and
synchronization. Every participating thread must reach a threadgroup barrier;
placing a barrier inside only one role can deadlock.

For this gated activation, all dependent operations belong to the same lane,
so ordinary pointwise fusion is sufficient. Independent outputs are a better
starting point for role specialization. See
``kernels/src/metile_kernels/simdgroup_specialized_elementwise.py`` and verify the
generated indexing before introducing communication between roles.
