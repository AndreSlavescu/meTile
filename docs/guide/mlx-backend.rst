MLX-LM Backend
==============

The optional MLX backend runs generated Metal kernels inside MLX's lazy graph,
using MLX arrays directly. Native MLX remains a candidate for every supported
operation. The integration selects generated work after numerical checks and
timing comparisons.

.. code-block:: bash

   python -m pip install -e ".[mlx-lm]" -e ./kernels

The integration uses the separate ``metile-kernels`` library as well as the
root compiler project. Installing only the root project's MLX dependencies
does not install that library from this checkout.

Patch a model
-------------

The patch handle restores the original methods when its context exits:

.. code-block:: python

   import mlx.core as mx
   from mlx_lm import load
   from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

   model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
   tokens = mx.array([tokenizer.encode("Explain tiled matrix multiplication.")])

   with apply_metile_to_mlx_lm(model=model):
       logits = model(tokens)
       mx.eval(logits)

Evaluate lazy results while the patch is active. You can also keep the handle
and call ``patch.restore()`` explicitly. Set ``attention=False``,
``rms_norm=False``, ``graph_fusion=False``, or ``quantized_mlp=False`` to disable
individual features. Unsupported calls use the original implementation.

Decode attention supports BF16, FP16, and FP32 MHA, GQA, and MQA with a one-token
query. Prefill attention, masks, sinks, quantized KV caches, and unsupported
dimensions retain MLX-LM's implementation. RMSNorm supports those storage
types with FP32 accumulation.

The integration also recognizes supported transformer residual-add/RMSNorm
patterns. The graph planner preserves the residual as a second output and
compares the fused operation with the original graph. See :doc:`graph-fusion`
for matching and region-selection limits.

Prepare projection backends
---------------------------

Preparation is explicit because it may allocate additional weight views.
For supported affine 4-bit projections, prepare a K-major repack:

.. code-block:: python

   from metile.integrations.mlx_lm import prepare_mlx_lm_affine_prefill

   affine_prefill = prepare_mlx_lm_affine_prefill(model)
   with apply_metile_to_mlx_lm(model=model, affine_prefill=affine_prefill):
       logits = model(tokens)
       mx.eval(logits)

The repack preserves the original nibbles, scales, and biases. The backend
tunes supported row tiles and traversal orders against native MLX. Row masks
handle incomplete tiles. Prepared prefill projections restore their original
class when the row count falls below the configured threshold. Restore each
patch before applying a different configuration to the same model.

For dense BF16 or FP16 checkpoints, use the dense preparation path:

.. code-block:: python

   from metile.integrations.mlx_lm import prepare_mlx_lm_dense_mlp

   model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
   dense_mlp = prepare_mlx_lm_dense_mlp(model)
   tokens = mx.array([tokenizer.encode("Explain tiled matrix multiplication.")])
   with apply_metile_to_mlx_lm(model=model, dense_mlp=dense_mlp):
       logits = model(tokens)
       mx.eval(logits)

Dense preparation builds K-major gate/up views and, when the memory budget
allows, an interleaved gate/up view for decode. By default, estimated model
weights and repacked views must fit within 80% of MLX's recommended working
set. The paired decode view can be omitted independently.

One-row dense candidates share activation loads across gate and up dot
products, then apply SwiGLU before the output store. They vary output count,
SIMD-group count, and K unrolling while preserving the checked lane partition
and reduction order. Exact QMV candidates must match MLX bit for bit and pass
pairwise finalist timing against native MLX. Gate/up confirmation uses 63 rounds
for one row and 31 for multiple rows. The down-projection/residual candidate
is selected independently, so either stage can retain native MLX.

Small batches can also use the SIMD-group QMV candidates when their shapes
satisfy the register and alignment limits. Each row must match MLX's own
single-row result bit for bit; MLX's batched implementation may use another
reduction order. At larger row counts, supported candidates include two
generated projections followed by the MLX epilogue and a fused dual-GEMM
schedule. The fused BF16 path preserves MLX's low-precision boundaries in
sigmoid and the two products. Numerical and model-level checks decide which
candidate is usable for the call.

Quantized SwiGLU also compares eager MLX, compiled MLX, and generated schedules.
One generated schedule writes the gate accumulator to threadgroup scratch,
reuses its registers for up, then applies the epilogue. Its storage and barrier
costs are part of the measured candidate.

Selection and fallback
-----------------------

.. image:: /_static/runtime-dispatch.svg
   :alt: Native MLX and generated kernels pass numerical checks and timing comparisons before dispatch
   :width: 100%

Primitive selection uses family-specific switching margins. These are the
current settings, which may differ from a saved benchmark's policy:

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Family
     - Required headroom
   * - Attention and RMSNorm
     - 5%
   * - Graph fusion and block-scaled matmul
     - 10%
   * - General dense/affine generated projections and quantized SwiGLU
     - 3%
   * - Compiled-MLX quantized SwiGLU
     - 3%
   * - Exact dense gate/up QMV and dense down/residual QMV
     - 1.5%
   * - Affine down/residual QMV
     - 1%

Selections persist by device, MLX version, dtype, shape, source, and candidate
policy. A cached native decision returns to the original operation. Compilation
failures, numerical failures, and insufficient measured headroom also retain it.

Use ``autotune_metile_for_mlx_lm`` to compare feature combinations on an actual
model trajectory. It rejects plans that change the next token or exceed
KL-divergence, mean-logit-error, or maximum-logit-error bounds. It measures
related plans in stages, then checks finalists on a separate holdout of at
least 32 decode steps and seven paired trials before persistence. A model plan
can reject a composition even when its individual kernels win. The result is
specific to the model, calibration workload, device, and backend policy.

Optional weight compression
----------------------------

Compression changes the weight representation. The affine-INT8 path keeps
source BF16 weights and prepares compressed copies for selected one-row decode
projections. Multi-row prefill uses the original projections.

For a dense model, prepare and calibrate the down projections:

.. code-block:: python

   from metile.integrations.mlx_lm import (
       autotune_metile_for_mlx_lm,
       prepare_mlx_lm_compressed_down,
   )

   compressed_down = prepare_mlx_lm_compressed_down(model, format="affine8")
   sample_tokens = mx.array([tokenizer.encode("Explain tiled matrix multiplication.")])
   plan = autotune_metile_for_mlx_lm(
       model, sample_tokens,
       quantized_mlp=False,
       compressed_down=compressed_down,
   )
   patch = apply_metile_to_mlx_lm(
       model=model,
       quantized_mlp=False,
       compressed_down=compressed_down,
       plan=plan,
   )

The strict affine-INT8 calibration policy requires an unchanged next token,
KL divergence at most 0.001, mean logit error at most 0.05, and maximum logit
error at most 0.5. This permits bounded error; it does not establish bitwise
equivalence or accuracy on untested prompts. MXFP8 requires
``allow_approximate=True`` and has a separate, looser policy.

Other independently selectable preparation functions are:

* ``prepare_mlx_lm_compressed_gate_up``: keeps each layer's SwiGLU pair together.
* ``prepare_mlx_lm_compressed_attention``: groups Q, K, V, and output projections
  by layer and preserves projection biases.
* ``prepare_mlx_lm_compressed_vocab``: handles a tied embedding's ``as_linear``
  projection or an untied ``lm_head``; embedding lookup remains unchanged.

Pass each prepared object to both the model tuner and the patch function.
Calibration can leave sensitive layers in BF16 and reject an entire feature
when it does not compose well with the others.

Affine groups of 32, 64, and 128 are available. The default
``group_size="auto"`` times compatible groups at the one-row projection shape.
Attention additionally considers error among groups close to the fastest.
Layer selection uses bounded prefix/suffix searches, local audits, and holdout
checks. Compressed families default to a 90% working-set ceiling and stage
group candidates sequentially. Source weights remain allocated for fallback.

Block-scaled matmul
-------------------

``MLXBlockScaledWeight`` supports K-major MXFP4 and MXFP8 weights:

.. code-block:: python

   from metile.backends.mlx_block_scaled import (
       MLXBlockScaledWeight,
       mlx_block_scaled_matmul,
   )

   weight = MLXBlockScaledWeight.quantize(dense_k_by_n, format="mxfp8")
   output = mlx_block_scaled_matmul(activations, weight)

Here ``dense_k_by_n`` and ``activations`` are MLX arrays. The weight object
keeps both the compiler's K-major layout and MLX's native packed layout.
Generated candidates combine scale/value decoding, register fragments,
``matmul2d``, row masks, and traversal schedules. FP16 and BF16 arrays use
native Metal storage types. The selector checks compatibility and requires
10% headroom before choosing generated work over the native packed candidate.

.. code-block:: bash

   METILE_DISABLE_DISK_CACHE=1 python -m benchmarks.kernels.block_scaled_gemm 2048

Read and reproduce model results
--------------------------------

:doc:`benchmarks` contains model charts, recorded results, and comparison rules.
Distinguish these experiments:

* **Matched weights:** both implementations use the same values and format.
  Repacking without requantization belongs here.
* **Matched quantization:** both implementations use the same quantized
  weights, scales, and biases. This isolates their execution paths.
* **Mixed precision:** the selected meTile plan uses compressed projections
  while native MLX reads source BF16 weights. This measures a deployment
  tradeoff and cannot establish a BF16 kernel speedup.

The seven-model BF16-source capacity suite belongs to the third class. Selected
affine-INT8 projections execute through MLX ``mx.quantized_matmul``. Its decode
gains include reduced weight traffic, and no matched affine-INT8 model control
suite is committed. Fidelity bounds apply to the recorded trajectories.

The suite runner records preparation, model-plan tuning, confirmation, and
steady-state measurement separately. Results include selected native/generated
paths, raw samples, software identities, and MLX allocator peak memory.
When the selected plan is entirely native, both labels share its sample.

.. code-block:: bash

   python -m pip install -e ".[mlx-lm,benchmarks]" -e ./kernels
   python -m benchmarks.mlx.mlx_lm_suite \
     --prompt-tokens 128 --generation-tokens 256 \
     --trials 9 --delay 0 --plan-trials 7 --confirmation-trials 5 \
     --output /tmp/mlx-models.json

Use ``--suite bf16`` for the dense source suite and ``--offline`` for cached
checkpoints. Compression requires explicit flags such as
``--compressed-down-format affine8``, ``--compressed-gate-up``,
``--compressed-vocab``, and ``--compressed-attention``. Each family has a
``--compressed-...-group-size`` option. Use the corresponding
``--disable-attention``, ``--disable-rmsnorm``, ``--disable-graph-fusion``,
``--disable-quantized-mlp``, or ``--disable-affine-prefill`` flags for ablations.
