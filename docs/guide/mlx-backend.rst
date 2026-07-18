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

Affine 4-bit Llama decode MLPs add three representation families to the same policy:
native MLX quantized matmul, an output-major scalar meTile kernel, and an M5-native
``matmul2d`` kernel over an AOT K-major repack. Gate/up projections and SwiGLU are fused
without materializing either projection. Candidate outputs must first match native MLX;
the fastest generated representation must then clear a 10 percent guard band. Repacked
weights are retained only when the NAX representation wins.

Model Benchmark
---------------

The benchmark loads actual MLX-LM models, verifies the next token, warms both paths,
alternates trial order, adds a configurable cooldown, and records decode throughput,
total time, environment metadata, raw samples, and every selected dispatch.

The committed M5 32 GB suite uses a 128-token prompt, 256 generated tokens, five
alternating trials, and two-second cooldowns:

.. image:: /_static/mlx-model-throughput.png
   :alt: Native MLX and MLX with meTile median decode throughput across four 4-bit language models
   :width: 100%

.. image:: /_static/mlx-model-speedups.png
   :alt: Decode and end-to-end percentage change relative to native MLX across four 4-bit language models
   :width: 100%

.. list-table:: M5 model-level medians
   :header-rows: 1

   * - Model
     - MLX decode
     - MLX + meTile
     - Decode
     - End-to-end
   * - Llama 3.2 1B 4-bit
     - 152.34 tok/s
     - 152.24 tok/s
     - 0.999x
     - 1.001x
   * - Llama 3.2 3B 4-bit
     - 59.22 tok/s
     - 61.56 tok/s
     - 1.039x
     - 1.036x
   * - Qwen 2.5 0.5B 4-bit
     - 308.78 tok/s
     - 307.25 tok/s
     - 0.995x
     - 1.028x
   * - Qwen 2.5 1.5B 4-bit
     - 117.95 tok/s
     - 116.35 tok/s
     - 0.986x
     - 0.998x

Llama 3.2 3B is the clear win in this run. Llama 1B is at parity, Qwen 0.5B
improves end-to-end time despite decode parity, and Qwen 1.5B remains slightly below
native decode. The guarded backend is therefore presented as shape- and model-dependent,
not as a universal framework win. The raw result is committed at
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
