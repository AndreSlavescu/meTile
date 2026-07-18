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

The benchmark loads an actual MLX-LM model, warms both paths, alternates trial order,
adds a configurable cooldown, and reports both decode throughput and total time:

.. code-block:: bash

   python benchmarks/mlx_lm_backend.py \
     --model mlx-community/Llama-3.2-1B-Instruct-4bit \
     --prompt-tokens 128 \
     --generation-tokens 256 \
     --trials 5 \
     --delay 2

Use ``--disable-attention``, ``--disable-rmsnorm``, ``--disable-graph-fusion``, or
``--disable-quantized-mlp`` for ablation runs. The final report lists every
native/generated schedule decision so a throughput result cannot silently attribute an
MLX fallback to meTile.
