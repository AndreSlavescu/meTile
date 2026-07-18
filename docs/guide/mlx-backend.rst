MLX-LM Backend
==============

meTile can embed generated Metal bodies in MLX's lazy graph without copying through
NumPy or a separate ``MTLBuffer``. Install the optional integration dependencies:

.. code-block:: bash

   pip install -e ".[mlx-lm]"

Apply the reversible model patch after or before loading an MLX-LM model:

.. code-block:: python

   from mlx_lm import load
   from metile.integrations.mlx_lm import (
       apply_metile_to_mlx_lm,
       prepare_mlx_lm_affine_prefill,
   )

   model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
   affine_prefill = prepare_mlx_lm_affine_prefill(model)
   patch = apply_metile_to_mlx_lm(model=model, affine_prefill=affine_prefill)

   # Generation independently tunes attention, RMSNorm, graph, and MLP primitives.
   patch.restore()

The handle is also a context manager. ``attention=False``, ``rms_norm=False``,
``graph_fusion=False``, or ``quantized_mlp=False`` disables each patch target
independently. Affine prefill is explicit because it creates an additional packed view of
the prepared projections.

Dispatch Policy
---------------

.. image:: /_static/runtime-dispatch.svg
   :alt: Guarded native and generated kernel selection in the meTile MLX runtime
   :width: 100%

Native MLX is an explicit autotune candidate. Attention and RMSNorm use a 5 percent
primitive guard, graph fusion uses 10 percent, generated SwiGLU uses 3 percent, and the
smaller down-projection/residual epilogue uses 1 percent before model-level confirmation.
These family-specific guard bands account for command-graph composition effects that a
standalone kernel benchmark cannot observe. Decisions persist by device architecture, MLX
version, dtype, shape bucket, source, and candidate family.

Decode attention supports FP16/FP32 MHA, GQA, and MQA with a one-token query. Prefill,
attention masks, sinks, quantized KV caches, unsupported dimensions, and unsupported
dtypes retain MLX-LM's original implementation. RMSNorm supports FP16/FP32 values and
weights with FP32 accumulation.

The integration also captures the canonical Llama and Qwen2 residual-add followed by RMSNorm
as a high-level compute DAG. The graph fusion pass uses an exact max-flow/min-cut partition,
preserves the residual as a second output, and measures the fused multi-output kernel against
the original MLX graph. The first supported call initializes this tournament; later MLX winners
return directly to the untouched transformer block.

Affine 4-bit models add eager MLX and an M5-native ``matmul2d`` kernel over an AOT K-major
repack to the same policy. The repack preserves the original affine nibbles, scales, and
biases; it does not requantize model weights. Ragged prefill rows tune Morton, grouped, and
Hilbert schedules against native MLX. Only prepared projection instances use the specialized
class, and the first row below the configured threshold restores the original class. Steady-state
decode therefore does not traverse a wrapper. Exact compiled-MLX SwiGLU candidates use a 0.5
percent switch margin, while generated kernels retain the stricter 3 percent margin. Quantized
decode also evaluates a scratch-spilled SwiGLU schedule that shortens gate/up accumulator
lifetimes before the elementwise epilogue. The M5-native variant removes the dead second
16-row ``matmul2d`` FMA for one-token decode, spills gate fragments through a bank-transposed
threadgroup layout, resets and reuses the same two accumulators for up, then reloads one gate
fragment at a time. Quantized-only plans keep this decode tournament; combined affine-prefill
plans keep it independently available for one-row decode. The selected gate/up implementation
is composed with an affine down-projection/residual epilogue that tunes block width, outputs per
SIMD-group, and FP16/FP32 decode. A warmed shape-specialized executor binds those decisions and
weights once, removing repeated Python dispatch construction. Multi-row calls skip this
decode-only block path and preserve the original MLX-LM implementation.

Model-level tuning rejects plans that change the next token or exceed KL-divergence,
mean-logit-error, or max-logit-error bounds. Surviving plans need a paired TTFT or total-latency
win while preserving bounded decode and total latency. Decode-sensitive plans use a 0.5 percent
confirmation floor; self-deoptimizing prefill-only plans use a 1 percent noise floor. Explicit
backend signatures invalidate model plans whenever primitive candidates, source, or selection
policy changes. The selected feature plan and primitive schedules persist with device, MLX
version, shape, source, and tuner-policy identities.

Block-Scaled MLX Primitive
--------------------------

``MLXBlockScaledWeight`` provides a zero-copy MLX backend for K-major MXFP4 and MXFP8 weights:

.. code-block:: python

   from metile.backends.mlx_block_scaled import (
       MLXBlockScaledWeight,
       mlx_block_scaled_matmul,
   )

   weight = MLXBlockScaledWeight.quantize(dense_k_by_n, format="mxfp8")
   output = mlx_block_scaled_matmul(activations, weight)

The compiler composes E8M0 scale decode, E2M1 or E4M3 value decode, register fragments,
native ``matmul2d``, ragged-row masks, and a schedule pass. The runtime measures linear,
grouped, and Hilbert variants and persists the fastest compatible representation.

Model Benchmark
---------------

The benchmark loads actual MLX-LM models, verifies bounded logit fidelity, tunes a persistent
model-level feature plan, adds a configurable cooldown, and records prefill/decode throughput,
TTFT, total time, environment metadata, raw samples, and every selected dispatch. When the
selected plan is native MLX, both labels share each native sample instead of graphing system
noise.

The committed M5 32 GB suite uses a 128-token prompt, 256 generated tokens, five
end-to-end confirmation pairs, and nine continuous measurement pairs:

.. image:: /_static/mlx-model-throughput.png
   :alt: Native MLX and MLX with meTile median prefill and decode throughput across four 4-bit language models
   :width: 100%

.. image:: /_static/mlx-model-latency.png
   :alt: Native MLX and MLX with meTile TTFT and end-to-end latency across four 4-bit language models
   :width: 100%

.. list-table:: M5 model-level medians
   :header-rows: 1

   * - Model
     - MLX decode
     - MLX + meTile
     - Native TTFT
     - Decode
     - Prefill
     - TTFT speedup
     - End-to-end
   * - Llama 3.2 1B 4-bit
     - 151.36 tok/s
     - 150.98 tok/s
     - 102.0 ms
     - 0.998x
     - 1.339x
     - 1.135x
     - 1.004x
   * - Llama 3.2 3B 4-bit
     - 61.35 tok/s
     - 61.35 tok/s
     - 153.9 ms
     - 1.000x
     - 1.000x
     - 1.000x
     - 1.000x
   * - Qwen 2.5 0.5B 4-bit
     - 309.19 tok/s
     - 304.62 tok/s
     - 70.8 ms
     - 0.994x
     - 1.275x
     - 1.072x
     - 1.001x
   * - Qwen 2.5 1.5B 4-bit
     - 118.54 tok/s
     - 119.65 tok/s
     - 130.6 ms
     - 1.001x
     - 1.331x
     - 1.170x
     - 1.013x

Three workloads selected generated affine prefill; Llama 3.2 3B retained native MLX. Table
speedups are medians of paired ratios, while chart bars are absolute medians. The raw result is committed at
``benchmarks/results/m5-mlx-lm-models.json``.

Reproduce the complete suite and regenerate both figures:

.. code-block:: bash

   pip install -e ".[benchmarks]"

   python benchmarks/mlx_lm_suite.py \
     --prompt-tokens 128 \
     --generation-tokens 256 \
     --trials 9 \
     --delay 0 \
     --plan-trials 7 \
     --confirmation-trials 5 \
     --output benchmarks/results/m5-mlx-lm-models.json

   python benchmarks/render_mlx_lm_results.py \
     benchmarks/results/m5-mlx-lm-models.json

Use ``--disable-attention``, ``--disable-rmsnorm``, ``--disable-graph-fusion``,
``--disable-quantized-mlp``, or ``--disable-affine-prefill`` with either the single-model
or suite runner for ablation runs. ``--offline`` makes the suite use already cached checkpoints. The structured
result lists every native/generated schedule decision so a throughput result cannot
silently attribute an MLX fallback to meTile.
