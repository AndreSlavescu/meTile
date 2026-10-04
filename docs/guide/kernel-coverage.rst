Kernel coverage and training scope
===================================

meTile does not yet cover the full Liger-Kernel training library. This page
separates native forward/backward pairs from forward-only kernels and work that
remains. A matching operator name is not enough: formulas, gradient inputs,
dtypes, layouts and framework integration must also match.

The comparison was checked on 2026-10-04 against Liger-Kernel
`v0.8.4 <https://github.com/linkedin/Liger-Kernel/releases/tag/v0.8.4>`_
and commit ``b297821787949e6102c162f24bf89a0cf3625d09`` on its main branch.
The pinned main snapshot includes partial RoPE support beyond that release.
The `operator exports`_, `functional wrappers`_ and `chunked-loss exports`_
define the inventory here; open pull requests are not counted as released or
merged features.

Status key
----------

**Paired, bounded** means native forward and explicit backward entry points
exist for the stated subset, with numerical tests. It does not mean every
upstream option is supported, that existing inference entry points gained
backward support, or that a framework automatically calls those kernels.

**Forward only** means the existing path has no corresponding native backward
entry point. **Planned** identifies work in progress, not a usable contract.
**Missing** means no corresponding implementation is provided. Priorities below
describe implementation order, not a delivery schedule or performance claim.

Normalization
-------------

The first three pairs live in ``metile_kernels.training_norms``. The explicit
allocation and dispatch API lives separately in
``metile.backends.training_norms``. The older inference normalization modules
remain separate entry points.

.. list-table:: Normalization parity checklist
   :header-rows: 1
   :widths: 22 18 60

   * - Liger family
     - meTile status
     - Remaining contract work
   * - RMSNorm
     - Paired, bounded
     - Standard zero-offset formula; no Liger casting-mode, in-place or
       weight-offset compatibility. See the exact native contract below.
   * - LayerNorm
     - Paired, bounded
     - Two-dimensional rows and affine weight/bias; no arbitrary normalized
       shape, BF16 or framework-autograd adapter.
   * - Fused add + RMSNorm
     - Paired, bounded
     - Both output cotangents are supported. Residual storage and casting
       semantics are explicit, not assumed identical to every upstream mode.
   * - Modulated RMSNorm
     - Missing
     - Input, weight and broadcast scale/shift gradients. Priority 1.
   * - GroupNorm
     - Missing
     - Group/channel layouts and affine gradients. Priority 1.
   * - PolyNorm
     - Missing
     - Matching polynomial normalization and parameter gradients. Priority 1.
   * - Dynamic Tanh (DyT)
     - Missing
     - Trainable alpha, gamma and beta gradients. Priority 1.

Upstream contracts: `RMSNorm`_, `LayerNorm`_, `fused add RMSNorm`_ and the
normalization entries in `functional wrappers`_. In particular, fused add
RMSNorm has two outputs; omitting the residual-output cotangent loses part of
its backward contract.

Activations, projections and positions
--------------------------------------

.. list-table:: Activation and position parity checklist
   :header-rows: 1
   :widths: 22 18 60

   * - Liger family
     - meTile status
     - Remaining contract work
   * - SwiGLU / SiLU multiply
     - Paired, bounded
     - ``training_activations`` supports elementwise SiLU with an optional
       separate gate multiplier and both input gradients. Packed gate/up
       layouts and Liger multiplier variants remain missing.
   * - GeGLU / GELU multiply
     - Paired, bounded
     - ``KIND="gelu_tanh"`` with ``GATED=True`` provides the tanh approximation
       and both gradients. Its FP32 intermediate convention is not a promise
       of bitwise equivalence to upstream intermediate casts.
   * - ReLU squared
     - Missing
     - Ordinary ReLU is paired; squaring it is not yet a named library pair.
       Priority 0.
   * - Fused MLP
     - Forward pieces only
     - A complete gate/up/down projection pair needs input and all weight
       gradients. Fused inference GEMMs do not supply them. Priority 1.
   * - Tiled/checkpointed MLP
     - Missing
     - Recompute policy, chunking and complete backward. Priority 1.
   * - Fused MoE
     - Missing
     - Routed experts, gate/up/down weights and floating router-weight
       gradients; discrete expert indices stay nondifferentiable. Priority 2.
   * - Standard full/partial RoPE
     - Paired, bounded
     - ``metile_kernels.rope`` supports split-half or interleaved rotation and
       unrotated tails. Caller-expanded coefficient tables are required;
       framework broadcasting, position lookup and shared-table reduction
       are not included.
   * - Llama 4 RoPE
     - Partial primitive overlap
     - Interleaved rotation exists; the complete upstream complex-layout
       wrapper and its shape contract remain missing. Priority 1.
   * - Qwen2-VL multi-axis RoPE
     - Missing
     - Sectioned multi-axis coefficient selection and its wrapper. Priority 1.

See upstream `SwiGLU`_, `GeGLU`_, `MLP`_, `tiled MLP`_, `MoE`_ and `RoPE`_.
The older ``metile_kernels.mlp`` GELU/GeGLU helpers use the logistic
``x * sigmoid(1.702 * x)`` approximation: they are not the tanh-GELU variant.
The new activation module keeps these formulas distinct. Native RoPE can also
return per-row cosine/sine gradients; upstream standard RoPE treats those
tables as fixed. Shared-table gradients still need an explicit reduction.

Reductions and attention-like operators
----------------------------------------

.. list-table:: Reduction and attention parity checklist
   :header-rows: 1
   :widths: 22 18 60

   * - Softmax
     - Paired, bounded
     - ``training_losses`` provides last-axis FP32 softmax/log-softmax and
       explicit input gradients, with 1--262144 columns and at most 65536
       flattened rows. FP16/BF16 and framework adapters remain missing.
   * - Sparsemax
     - Missing
     - Support-set derivative and dimension handling. Priority 1.
   * - Multi-token attention
     - Missing
     - This upstream operator convolves attention-score tensors; it is not
       standard scaled dot-product attention. Score, filter and optional
       bias gradients are required. Priority 2.
   * - Fused neighborhood attention
     - Missing
     - Sliding/dilated window Q/K/V backward. A causal dense attention kernel
       is not equivalent. Priority 2.
   * - Attention residuals (AttnRes)
     - Missing
     - Depth-block mixing, query weight and normalization weight gradients.
       Priority 2.
   * - Manifold-constrained hyper-connections (mHC)
     - Missing
     - Coefficient/Sinkhorn, pre-mixing and post-residual pairs with all
       trainable parameter gradients. Priority 2.

Upstream definitions: `softmax`_, `sparsemax`_, `multi-token attention`_,
`neighborhood attention`_, `AttnRes`_ and `mHC`_. Their mathematical contracts
are distinct even when they share reductions or matrix products internally.

Losses
------

.. list-table:: Loss parity checklist
   :header-rows: 1
   :widths: 25 17 58

   * - Liger family
     - meTile status
     - Remaining contract work
   * - Cross-entropy
     - Paired, bounded
     - FP32 row loss/logit gradients support ignore-index,
       none/sum/mean reduction, uniform label smoothing and squared-logsumexp
       z-loss. Class weighting, soft targets, softcap, FP16/BF16 and optional
       metrics remain missing. No fused projection is implied.
   * - Fused linear cross-entropy
     - Missing
     - Chunked projection plus loss, including hidden/weight/optional bias
       gradients; not just CE over materialized logits. Priority 0.
   * - Scaled fused linear cross-entropy
     - Missing
     - Temperature, optional entropy output and all output cotangents;
       tensor-parallel variant also needs collectives. Priorities 1/2.
   * - Vocabulary-parallel cross-entropy
     - Missing
     - Shard-global normalization and backward collectives. Priority 2.
   * - KL divergence
     - Missing
     - Input/log-target and reduction conventions; target is frozen in the
       upstream adjoint contract. Priority 1.
   * - Jensen-Shannon divergence
     - Missing
     - Student gradient, frozen teacher, beta and label masking. Priority 1.
   * - Total variation distance
     - Missing
     - Input gradient, frozen target and reduction conventions. Priority 1.
   * - Fused linear JSD
     - Missing
     - Student hidden/weight gradients and frozen teacher paths. Priority 1.
   * - Fused linear KL divergence
     - Missing
     - Hidden/weight gradients and supported reductions. Priority 1.
   * - Fused CE + TVD
     - Missing
     - Combined loss weighting and backward. Priority 1.
   * - GRPO token loss
     - Missing
     - Policy ratios, clipping, masks and supported GRPO/DAPO-style variants.
       Priority 1.

The pinned `cross-entropy`_, `fused linear cross-entropy`_, `scaled loss`_,
`vocabulary-parallel loss`_, `KL divergence`_, `JSD`_, `TVD`_,
`fused linear JSD`_, `fused linear KL`_, `fused CE TVD`_ and `GRPO loss`_
sources define these variants. Distributed loss support cannot be inferred
from a single-device reduction kernel.

The native ``metile.backends.training_losses`` API uses integer class targets,
valid-row-count mean reduction and zero loss/gradient for entirely ignored
targets. Its softmax and CE paths share the width/row bounds above and a
256 MiB per-call output/workspace limit, excluding NumPy input copies. CE
backward returns only the logit gradient; targets and loss configuration are
fixed. Separate saved maxima and log-denominators avoid recombining large
offsets before the backward normalization.

All eight public `chunked-loss exports`_ are still missing: fused linear
cosine-similarity loss, CPO, DPO, GRPO, JSD, KTO, ORPO and SimPO. These require
their own chunking/recompute and backward contracts; the chunked JSD class is
listed separately from the operator-level fused linear JSD above. Preference,
unpaired-preference, distillation and PPO base implementations are supporting
infrastructure rather than additional public operator checkboxes. Priority 1.

Experimental operators and meTile extras
----------------------------------------

Upstream experimental `embedding`_ has an embedding-table backward, which is
missing here. Repeated indices require correct accumulation into the shared
table; simply gathering or adding a scatter store is insufficient. Priority 1.
The experimental `integer matrix product`_ and its packing helpers do not
define an autograd pair. Integer indices, packed bit patterns and rounding are
not silently treated as differentiable floating operations.

meTile also includes operators outside this Liger inventory:

* ``stable_attention`` and ``metile.backends.attention`` provide bounded dense
  attention with explicit Q/K/V backward, masking, causal offsets and GQA.
* ``gated_delta`` provides bounded recurrent GDN/KDA-style forward/backward,
  including initial-state and final-state cotangents. It is not a complete
  model block or a replacement for the loss families above.
* Dual-chunk attention has a separate bounded implementation and contract;
  it is not proof of neighborhood or multi-token attention parity.
* ``metile.backends.training_matmul`` provides a bounded dense matrix-product
  pair with both input gradients and optional ReLU, SiLU, quick-GELU or
  tanh-GELU. FP16/FP32 inputs are packed to FP32 and the explicit SIMD-group
  backend avoids automatic TF32 selection. This is not a complete gated MLP.
* FFT, other fused GEMMs, block-scaled matrix products, affine INT8 QMV,
  quantized SwiGLU and fixed-arity reduction helpers include inference-oriented
  paths. Their existing names do not imply paired training gradients.
* Native sigmoid, tanh, ordinary ReLU and quick-GELU have elementwise paired
  kernels in ``training_activations``.

No CUDA-versus-Metal performance equivalence is claimed. Each supported Apple
GPU, shape, dtype and resource limit needs its own correctness and performance
measurement.

Exact native normalization contract
-------------------------------------

Install both the compiler and companion kernel package to use these APIs::

   python -m pip install -e . -e ./kernels

``rms_norm_forward(x, weight, epsilon=...)`` computes
``x * inverse * weight``, where
``inverse = 1 / sqrt(mean(x*x) + epsilon)`` over each row.
``layer_norm_forward(x, weight, bias, epsilon=...)`` first centers the row by
its mean, computes the population variance from those centered values, then
applies weight and bias. The default epsilon is ``1e-5``.

``add_rms_norm_forward(x, residual, weight, epsilon=...)`` computes the sum
in FP32 and normalizes that unrounded sum. It returns the normalized output,
the residual sum rounded to the input storage dtype, and a backward context.
``norm_backward`` accepts both the normalized-output cotangent and the
optional ``residual_output_gradient``. The latter defaults to zero. Its
contribution is added to both ``dsource`` and ``dresidual``.

All three APIs accept NumPy arrays or meTile buffers with these bounds:

* Source shape is ``(rows, columns)`` with positive rows and 1--8192 columns.
  Weight and optional bias have shape ``(columns,)``. Residual and cotangents
  match the source shape. NumPy inputs are copied into contiguous buffers.
* Storage is FP16 or FP32; the residual storage dtype must match the source.
  Weight, bias and cotangents may independently use either supported dtype.
  Statistics, arithmetic accumulation and all returned gradients are FP32.
* Epsilon must be positive and representable as finite FP32. Inputs should
  have finite values and finite FP32 intermediate statistics; extreme-value
  overflow is not repaired by the wrapper.
* Parameter gradients use per-row FP32 partials followed by a fixed row-order
  reduction, without atomics. The wrapper limits its conservative two-array
  partial-buffer budget to 128 MiB, or ``rows * columns <= 16_777_216``.
  This does not bound total memory including inputs, outputs and gradients.
* Forward output uses the source storage dtype. Backward differentiates the
  real-valued formula with saved FP32 statistics, ignoring floating-point
  rounding of casts. Returned parameter gradients are FP32 even for FP16
  parameters; callers choose any subsequent cast or optimizer policy.
* No in-place mutation, BF16, automatic framework registration or gradient
  with respect to epsilon is provided. Buffers retained in a forward context
  must not be modified before backward. Independent calls do not accumulate
  gradients into existing parameter buffers.

For example:

.. code-block:: python

   import numpy as np
   from metile.backends.training_norms import layer_norm_forward, norm_backward

   values = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
   weight = np.ones(7, dtype=np.float32)
   bias = np.zeros(7, dtype=np.float32)
   output, context = layer_norm_forward(values, weight, bias)
   gradients = norm_backward(context, np.ones_like(values))
   assert output.shape == values.shape
   assert gradients.source.shape == values.shape
   assert gradients.weight.shape == gradients.bias.shape == (7,)

The raw kernels are ``norm_forward_kernel``, ``norm_backward_rows_kernel`` and
``norm_parameter_reduce_kernel``. The backend supplies their layouts, scratch
buffers and ``STRICT_MATH=True``. Correctness tests cover FP16/FP32 storage,
ragged widths, centered variance, both residual-output seeds, parameter
reduction repeatability and finite-difference derivatives in
``tests/kernels/test_training_norms.py``.

What remains before claiming parity
-----------------------------------

Priority 0 is a dependable training core: bounded norm, activation and RoPE
pairs; softmax and CE; then chunked linear CE and broader GEMM adjoint coverage.
Priority 1 extends formulas and wrappers, additional norms and losses,
embedding and training MLPs. Priority 2 adds distributed and specialized
attention/MoE contracts. All priorities still need framework integration,
dtype coverage and end-to-end training checks.

For each entry, completion requires the exact forward formula and gradient
set, a declared policy for frozen/discrete inputs, dtype and accumulation
rules, masking and empty-input behavior, shape/layout bounds, deterministic
or explicitly nondeterministic accumulation, and validated framework wiring.
Finite differences and independent analytical references precede performance
claims; training-step and convergence checks are separate evidence.

``metile.vjp`` is only a bounded expression-DAG adjoint facility. It can
differentiate supported arithmetic, selected floating unary functions,
selection and row reductions between explicitly requested floating values.
It does not synthesize whole-kernel backward memory traffic, loop adjoints,
matrix-product adjoints, scatter/atomic accumulation or framework hooks.
Hand-written library backwards therefore remain necessary even where a
forward expression uses the DSL.

.. _operator exports: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/__init__.py
.. _functional wrappers: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/transformers/functional.py
.. _chunked-loss exports: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/chunked_loss/__init__.py
.. _RMSNorm: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/rms_norm.py
.. _LayerNorm: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/layer_norm.py
.. _fused add RMSNorm: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_add_rms_norm.py
.. _SwiGLU: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/swiglu.py
.. _GeGLU: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/geglu.py
.. _MLP: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/mlp.py
.. _tiled MLP: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/tiled_mlp.py
.. _MoE: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_moe.py
.. _RoPE: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/rope.py
.. _softmax: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/softmax.py
.. _sparsemax: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/sparsemax.py
.. _multi-token attention: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/multi_token_attention.py
.. _neighborhood attention: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_neighborhood_attention.py
.. _AttnRes: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/attn_res.py
.. _mHC: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/mhc.py
.. _cross-entropy: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/cross_entropy.py
.. _fused linear cross-entropy: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_linear_cross_entropy.py
.. _scaled loss: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_linear_scaled_cross_entropy.py
.. _vocabulary-parallel loss: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/vocab_parallel_cross_entropy.py
.. _KL divergence: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/kl_div.py
.. _JSD: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/jsd.py
.. _TVD: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/tvd.py
.. _fused linear JSD: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_linear_jsd.py
.. _fused linear KL: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_linear_kl_div.py
.. _fused CE TVD: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/fused_ce_tvd.py
.. _GRPO loss: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/grpo_loss.py
.. _embedding: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/experimental/embedding.py
.. _integer matrix product: https://github.com/linkedin/Liger-Kernel/blob/b297821787949e6102c162f24bf89a0cf3625d09/src/liger_kernel/ops/experimental/mm_int8int2.py
