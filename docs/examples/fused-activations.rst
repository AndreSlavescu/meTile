Fused Activations & Simdgroup Roles
=====================================

This example shows two patterns: simple element-wise activations, and using ``simdgroup_role``
to run different computations on different simdgroup subsets within a single kernel.


Simple Activations
------------------

Element-wise kernels follow the same pattern as vector add — load, compute, store:

.. code-block:: python

   import metile

   @metile.kernel
   def gelu(X, Out, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N
       x = metile.load(X + offs, mask=mask)
       # GELU approximation: x / (1 + exp(-1.702 * x))
       out = x / (1.0 + metile.exp(-1.702 * x))
       metile.store(Out + offs, out, mask=mask)

   @metile.kernel
   def silu(X, Out, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N
       x = metile.load(X + offs, mask=mask)
       # SiLU (Swish): x / (1 + exp(-x))
       out = x / (1.0 + metile.exp(-x))
       metile.store(Out + offs, out, mask=mask)


Fused GEMM + Activation
------------------------

When an activation follows a ``dot`` operation, the compiler fuses it into the GEMM epilogue.
The activation runs on register-resident data — no global memory round-trip:

.. code-block:: python

   @metile.kernel
   def matmul_gelu(A, B, C, M, N, K,
                   BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
                   BLOCK_K: metile.constexpr):
       pid_m = metile.program_id(0)
       pid_n = metile.program_id(1)
       acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
       for k in metile.tile_range(0, K, BLOCK_K):
           a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
           b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
           acc = metile.dot(a, b, acc)
       # Fused GELU epilogue — runs on accumulator registers
       acc = acc / (1.0 + metile.exp(-1.702 * acc))
       metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


Simdgroup Roles
---------------

Apple GPUs organize threads into 32-thread **simdgroups**. A threadgroup can contain
multiple simdgroups. With ``simdgroup_role``, you can assign different work to different
simdgroup subsets — useful for computing multiple outputs in a single dispatch:

.. code-block:: python

   @metile.kernel
   def exp_sqrt(X, out_exp, out_sqrt, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N

       with metile.simdgroup_role(role=0, num_roles=2):
           # First half of simdgroups compute exp
           x = metile.load(X + offs, mask=mask)
           metile.store(out_exp + offs, metile.exp(x), mask=mask)

       with metile.simdgroup_role(role=1, num_roles=2):
           # Second half compute sqrt(abs(x))
           x = metile.load(X + offs, mask=mask)
           metile.store(out_sqrt + offs, metile.sqrt(metile.abs(x)), mask=mask)

With ``num_roles=2``, the threadgroup's simdgroups are split in half. Role 0 computes
exponentials while role 1 computes square roots — simultaneously, in the same kernel launch.

This is useful when you need multiple derived outputs from the same input and want to
avoid the overhead of multiple kernel dispatches.


GEGLU (Gated GELU)
-------------------

A practical use of simdgroup roles — computing the gate and up projections of GEGLU
in parallel:

.. code-block:: python

   @metile.kernel
   def geglu(X_gate, X_up, Out, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N

       with metile.simdgroup_role(role=0, num_roles=2):
           gate = metile.load(X_gate + offs, mask=mask)
           gate = gate / (1.0 + metile.exp(-1.702 * gate))
           metile.store(Out + offs, gate, mask=mask)

       with metile.simdgroup_role(role=1, num_roles=2):
           up = metile.load(X_up + offs, mask=mask)
           gate = metile.load(Out + offs, mask=mask)
           metile.store(Out + offs, gate * up, mask=mask)


Concepts Introduced
-------------------

- Element-wise activation patterns
- ``metile.exp`` for activation functions
- Fused GEMM epilogues — zero-cost post-GEMM operations
- ``metile.simdgroup_role`` — split work across simdgroup subsets
- Multiple outputs from a single kernel
