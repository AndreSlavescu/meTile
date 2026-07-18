MLX-LM Backend
==============

meTile can embed generated Metal bodies in MLX's lazy graph without copying through
NumPy or a separate ``MTLBuffer``. Install the optional integration dependencies:

.. code-block:: bash

   pip install -e ".[mlx-lm]"

Apply the reversible model patch after or before loading an MLX-LM model:

.. code-block:: python

   from mlx_lm import load
   from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

   model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
   patch = apply_metile_to_mlx_lm(model=model)

   # Generation independently tunes attention, RMSNorm, graph, and MLP primitives.
   patch.restore()

The handle is also a context manager. ``attention=False``, ``rms_norm=False``,
``graph_fusion=False``, or ``quantized_mlp=False`` disables each patch target
independently.

Dispatch Policy
---------------

.. image:: /_static/runtime-dispatch.svg
   :alt: Guarded native and generated kernel selection in the meTile MLX runtime
   :width: 100%

Native MLX is an explicit autotune candidate. Generated kernels must beat it by at least
5% in isolated primitive timing before the integration crosses the framework boundary.
This guard band accounts for command-graph composition effects that a standalone kernel
benchmark cannot observe. Decisions persist by device architecture, MLX version, dtype,
shape bucket, source, and candidate family.

Decode attention supports FP16/FP32 MHA, GQA, and MQA with a one-token query. Prefill,
attention masks, sinks, quantized KV caches, unsupported dimensions, and unsupported
dtypes retain MLX-LM's original implementation. RMSNorm supports FP16/FP32 values and
weights with FP32 accumulation.

The integration also captures the canonical Llama residual-add followed by RMSNorm as a
high-level compute DAG. The graph fusion pass uses an exact max-flow/min-cut partition,
preserves the residual as a second output, and measures the fused multi-output kernel against
the original MLX graph. Graph fusion requires 10 percent isolated headroom before switching.

Affine 4-bit Llama MLPs add eager and compiled MLX, an output-major scalar meTile kernel,
and an M5-native ``matmul2d`` kernel over an AOT K-major repack to the same policy. Gate/up
projections and SwiGLU are fused without materializing either projection. Candidate outputs
must first match native MLX; the fastest alternative must then clear a 3 percent guard band.
Decisions are row-bucketed so prefill and decode can choose independently. Repacked weights
are retained only when the NAX representation wins.

Model Benchmark
---------------

The benchmark loads actual MLX-LM models, verifies the next token, tunes a persistent
model-level feature plan, adds a configurable cooldown, and records decode throughput,
TTFT, total time, environment metadata, raw samples, and every selected dispatch. Optimized
plans must preserve TTFT and decode while improving paired total latency. When the selected
plan is native MLX, both labels share each native sample instead of graphing system noise.

The committed M5 32 GB suite uses a 128-token prompt, 256 generated tokens, five
native-fallback trials, and two-second cooldowns:

.. image:: /_static/mlx-model-throughput.png
   :alt: Native MLX and MLX with meTile median decode throughput across four 4-bit language models
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
     - TTFT speedup
     - End-to-end
   * - Llama 3.2 1B 4-bit
     - 149.19 tok/s
     - 149.19 tok/s
     - 149.6 ms
     - 1.000x
     - 1.000x
     - 1.000x
   * - Llama 3.2 3B 4-bit
     - 56.62 tok/s
     - 56.62 tok/s
     - 281.3 ms
     - 1.000x
     - 1.000x
     - 1.000x
   * - Qwen 2.5 0.5B 4-bit
     - 306.93 tok/s
     - 306.93 tok/s
     - 114.1 ms
     - 1.000x
     - 1.000x
     - 1.000x
   * - Qwen 2.5 1.5B 4-bit
     - 119.31 tok/s
     - 119.31 tok/s
     - 214.3 ms
     - 1.000x
     - 1.000x
     - 1.000x

All four workloads selected native MLX in this run. The result demonstrates the guarded
runtime's no-regression fallback rather than a framework speedup. Several isolated kernel
candidates measured faster but failed the paired model-level safety test, so the published
result does not promote them. The raw result is committed at
``benchmarks/results/m5-mlx-lm-models.json``.

Reproduce the complete suite and regenerate both figures:

.. code-block:: bash

   pip install -e ".[benchmarks]"

   python benchmarks/mlx_lm_suite.py \
     --prompt-tokens 128 \
     --generation-tokens 256 \
     --trials 5 \
     --delay 2 \
     --output benchmarks/results/m5-mlx-lm-models.json

   python benchmarks/render_mlx_lm_results.py \
     benchmarks/results/m5-mlx-lm-models.json

Use ``--disable-attention``, ``--disable-rmsnorm``, ``--disable-graph-fusion``, or
``--disable-quantized-mlp`` with either the single-model or suite runner for ablation
runs. ``--offline`` makes the suite use already cached checkpoints. The structured
result lists every native/generated schedule decision so a throughput result cannot
silently attribute an MLX fallback to meTile.
