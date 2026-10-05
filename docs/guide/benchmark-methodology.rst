Benchmark methodology
=====================

Use this page to check a comparison or reproduce a figure. For results, see
:doc:`benchmarks`.

What the measurements mean
--------------------------

The published studies ran on an Apple M5. Each report describes its recorded
software and workload, not every later compiler revision. The separate
:doc:`megakernel comparison <megakernels>` separates single-threadgroup
fixed-context forward latency from GPU-wide end-to-end generation. Both keep
failed FP16 validation separate from measured FP32 results.

* **GPU time** comes from Metal command-buffer timestamps.
* **Wall time** includes dispatch and synchronization. Compiler microbenchmarks
  generally use prepared meTile dispatches and preallocated outputs, while MLX
  includes compiled-call evaluation and framework output allocation. The MLX
  layer sweeps time operation construction and evaluation for both backends.
  Compilation and input setup are excluded from these kernel timings.
* **Time to first token** runs from the start of a streaming generation call
  until its first yielded token. It includes host/framework work and is not
  isolated prefill GPU time.
* **End-to-end time** covers the generation call. Compare it only within the
  same prompt and generation length.
* **Effective weight bandwidth** is weight bytes divided by measured latency.
  It is a timing normalization, not a DRAM counter or a hardware bandwidth limit.

Same weight representation does not imply identical intermediate rounding.
Arithmetic comparisons use the tolerances recorded in each report. The
ownership-transpose study is a pure reorder and checks bitwise equality.

Aggregation and distributions
-----------------------------

.. list-table:: How each chart summarizes its input
   :header-rows: 1
   :widths: 28 34 38

   * - Study
     - Summary
     - Saved samples
   * - Model generation
     - Median of paired per-trial ratios
     - Nine trials per backend for active matched-weight comparisons; seven for compression; shared native fallbacks reuse one series
   * - GPU-wide Qwen3 generation
     - Median of paired per-trial ratios for TTFT, post-first-token decode and total time
     - Five trials per backend at each of four prompt/output lengths; every generated token ID and actual output count is saved and checked
   * - Long-document Qwen3 generation
     - Separate chunk-selection and held-out phases; median of paired per-trial ratios
     - Five pairs for each of four tuning chunks, then five held-out pairs; all 50 timed sequences and counts are checked
   * - Residual MLP batch sweep
     - Median of 25 paired per-round ratios
     - Summary values only; no raw trial distribution
   * - Projection width
     - Ratio of backend median times over 15 paired rounds
     - Summary values only; no raw trial distribution
   * - Effective weight bandwidth
     - Weight bytes divided by each backend's median latency over 15 paired rounds
     - Separate GB/s series, not a speedup ratio or measured memory traffic
   * - RMSNorm heldout validation
     - Three-round summaries in each of two fresh processes
     - Round summaries, not individual dispatch latencies

The TTFT violins use the four 4-bit models only: 128 prompt tokens, 256 requested
output tokens, seed 0, and nine trials per backend for active comparisons.
Model, precision, prompt length, and generation length are never pooled into
one distribution. The dense BF16 matched-weight model requests 64 output tokens
and is not included in that figure. These are configured output limits; the
reports do not record actual output counts or establish full generated-output
equivalence.

Dots show every saved observation; the median marker reports their center.
The density outline is a visual summary within the observed range, not a
confidence interval. Small or constant sample sets use points instead of an
invented density. A shared native fallback is one measurement series, drawn once.
Nine trials describe the recorded session, not variation across prompts, seeds,
devices, or independent benchmark sessions.

.. _qwen3-chunked-methodology:

Long-document chunked prefill
-----------------------------

The :download:`Qwen3 chunked-prefill driver
<../../benchmarks/megakernels/qwen3_chunked_prefill.py>` separates chunk
selection from the final comparison. Request batch size is one. A chunk batches
consecutive token rows of that request, while attention retains its full causal
prefix across chunk boundaries. See :doc:`megakernels` for the kernel structure.

.. list-table:: Separate chunk-selection and held-out workloads
   :header-rows: 1
   :widths: 22 22 20 36

   * - Phase
     - Prompt / output tokens
     - Chunk sizes
     - Measurement
   * - Tuning
     - 2,048 / 8
     - 128, 256, 512, 1024
     - Five paired trials per size; select the lowest candidate median TTFT
   * - Held-out
     - 4,096 / 1,024
     - The selected size only
     - Five new paired trials; not used for chunk selection

Prompts are contiguous token prefixes of local technical documentation, not
random IDs, repeated padding or generated filler. Tuning and held-out sources
must have different paths and file-content hashes, and their token sequences
must differ. Tokenization adds no chat template or special tokens. The report
saves the source hashes, exact input/output IDs, tokenizer and checkpoint hashes,
software versions, and compiler/kernel fingerprints. This is one held-out
document, not a benchmark over document or task diversity.
New reports also record the attention/projection backends, projection tile,
strict-math and relaxed-precision settings, and compiler/runtime environment
overrides. Existing report files are never overwritten; choose a fresh
``--output`` path for each run.
**Held-out refers to chunk-size selection only.** The 4,096-token prompt has
also been used for correctness diagnostics and implementation profiling; it
is not development-unseen test data.

All four candidate sizes must validate before tuning timings begin. The winner
is the smallest median candidate TTFT; smaller chunks break exact ties. It is
then validated separately on the held-out document before that phase is timed.
Each measured workload has two complete warmup generations per backend, followed
by five paired trials with alternating native/chunked order. No tuning samples
are reused as held-out evidence or pooled with them in a distribution.

Validation checks the entire valid K/V prefix in every layer at each chunk
boundary. It then resets the candidate and submits the entire prompt in one
``prefill`` call, matching the timed path, before validating generation. This
second pass also exercises work that can be skipped between internal chunks.
Every generated step checks full vocabulary logits, the exact greedy
token, and the newly written K/V slot in every layer. The whole prefix is checked
again at the first output, every 64 outputs, and the final output. The default
FP32 comparison disables MLX TF32 and uses fixed ``rtol=atol=0.001`` for logits
and cache values. This checks numerical agreement, not bitwise equality.
Every warmup and timed sequence must match its validated token IDs and count.
A failure discards all saved timings from both phases, including earlier tuning
measurements; it cannot leave a successful-looking partial chart.

Timing starts before cache reset or construction and ends after the final token
ID is returned to the host. It includes input/control updates, prefill, all
decode layers and cache writes, full vocabulary projections, GPU argmax,
dispatch submission and synchronization. EOS is ignored to produce exactly
1,024 output tokens. TTFT ends at the first returned token; decode throughput
uses the remaining **1,023 forwards**, not 1,024 divided by decode time.
The first output comes from the final prompt row. Later forwards write cache
positions 4096 through 5118, leaving 5,119 valid entries after generation.

Native MLX batches the first 4,095 prompt tokens and evaluates their cache state,
then forwards the final prompt token. Unused final-layer output work in the
prefix can be pruned. meTile runs all prompt tokens through its chunked decoder
and projects only the final row. For nonterminal chunks within one ``prefill``
call, it stops the final layer after writing K/V: attention, output projection
and the MLP have no remaining consumer for those rows. The last chunk still
runs the complete layer. Both process the same full prompt, but
their execution graphs are not identical. The native baseline calls the
unmodified MLX-LM model in a synchronous greedy loop with a host-observed token
at every step. It is **not** the official pipelined ``mlx_lm.stream_generate``
path. The chunked backend also updates
normalization and key-parallel attention used during decode. The optional
earlier sequential backend is therefore a whole-backend comparison, not an
isolated prefill-only ablation.

Loading, dtype conversion, rotary-table preparation, weight transposition and
packing, compilation, persistent buffer allocation, tokenization, text decoding
and correctness comparisons are excluded. meTile reuses its weight packs and
cache allocation across the sweep; changing chunk size rebuilds only batch
scratch and prepared dispatches outside timing. The extra packed-weight bytes
are recorded. Its cache reset remains timed; native MLX manages intermediate,
output and growing-cache allocations inside timing. These are warmed
token-ID-to-token-ID measurements, not cold-start or text-serving latency.

Lossless weight packing is a separate storage comparison, not a precision
change. When the actual FP32 weights have verified zero low 16 bits, the
current candidate packs two values into one ``uint32`` and reconstructs their
exact FP32 bits inside decode projections and the vocabulary head. That head
also produces the first token. Native MLX and candidate prefill matrix weights
stay unpacked FP32; activations, cache and accumulation stay FP32 as well.
The optional earlier sequential baseline remains unpacked. The report labels
this ``lossless_weight_storage`` and records different physical representations
but identical weight values. Packing requires finite values, even counts,
zero low bits and a bitwise round-trip check; otherwise storage stays unpacked.
The renderer checks the declared scopes, verification flags and byte counts
against the selected model geometry. Original FP32 buffers are retained, so
packed bytes describe an additional read-optimized copy, not lower total
allocation. This path does not relax any model-level correctness tolerance.

The :download:`direct-memory, lossless-packed report
<../../benchmarks/results/m5-qwen3-direct-packed-end-to-end.json>` selects chunk
512. Its held-out median paired speedups are **0.971x for TTFT, 1.655x for
decode and 1.494x for complete generation**. The independent TTFT medians
are 2.360 s for MLX and 6.955 s for meTile. Five highly variable samples do
not establish prefill parity or a prefill win; all samples remain in the
chart. These are whole-backend results, not an attention-only or packing-only
ablation. The :download:`earlier unpacked report
<../../benchmarks/results/m5-qwen3-prefill-shared-reuse-end-to-end.json>` remains
separate and unchanged.

Run a fresh comparison with locally cached weights:

.. code-block:: bash

   python3 -m pip install -e '.[mlx-lm,benchmarks]' -e ./kernels
   MLX_ENABLE_TF32=0 python3 -m benchmarks.megakernels.qwen3_chunked_prefill \
     --model Qwen/Qwen3-0.6B --dtype float32 \
     --attention-backend matrix --projection-backend tensor_ops \
     --projection-tile 64 64 64 \
     --chunk-sizes 128 256 512 1024 \
     --tuning-prompt-tokens 2048 --tuning-output-tokens 8 --tuning-trials 5 \
     --prompt-tokens 4096 --output-tokens 1024 --trials 5 \
     --cache-check-interval 64 --output /tmp/qwen3-direct-packed-end-to-end.json
   python3 -m benchmarks.plots.render_chunked_prefill \
     /tmp/qwen3-direct-packed-end-to-end.json --output /tmp/qwen3-direct-packed-prefill.png

Eligible FP32 decode weights are packed by default. Add
``--no-lossless-decode-weights`` for a fresh unpacked comparison; that does not
restore an earlier compiler, attention implementation or measurement session.
To render the published evidence without running generation:

.. code-block:: bash

   python3 -m benchmarks.plots.render_chunked_prefill \
     benchmarks/results/m5-qwen3-direct-packed-end-to-end.json \
     --output /tmp/qwen3-direct-packed-prefill.png

For the earlier scalar-attention configuration, use ``--attention-backend tiled
--projection-backend simdgroup --projection-tile 64 64 32``. Each report keeps
its own compiler fingerprint; a configuration alone does not reproduce an
older source revision or measurement session.

The renderer accepts only complete successful reports. It recomputes chunk
selection, medians and paired time ratios from raw trials and keeps tuning and
held-out panels separate. Five samples are shown as dots with a median marker,
not a violin density or confidence interval. A ratio above 1.00x favors meTile;
post-first-token tokens/s and whole-generation latency remain separate metrics.

Compiler comparison rules
-------------------------

**RMSNorm.** The selected ``register4_striped`` policy and its frozen baseline
both specialize ``N`` with ``BLOCK=1024``. Tuning uses seed 1741; heldout
validation uses seed 9473 in two fresh processes. Each validation uses three
rounds, 100 ms warmup, and 250 ms measurement budgets. The reports bind the
shader, selection, benchmark driver, and compiler to fingerprints.

Promotion requires at least 1.10x GPU speedup on every aligned 32-row and
256-row case, no more than three percent wall regression on any case, and
no more than three percent GPU regression on ragged-width cases. Both runs
pass the regression guards but fail the speedup requirement.

The primary MLX graph uses FP32 reduction, normalization, and weight
multiplication before the final storage cast. The separate ``mx.fast.rms_norm``
FP16 comparator rounds the normalized value to storage precision before
multiplying by the weight; it is checked against that rounding policy.
These are different comparisons, neither a bitwise-equivalence claim.

**Fusion and staging.** Keep each report's paired baseline and timing boundary.
Do not divide absolute times taken from different fused/two-launch and fused/MLX
pairs: their latency regimes differ. Strict-FP32 fusion disables reduced-precision
matrix computation with ``MLX_ENABLE_TF32=0``. Staging compares eight matched
FP16/FP32 cases, including ragged dimensions, against the preceding pipeline.
See :doc:`execution-schedules`, :doc:`tensor-memory`, and :doc:`thread-layouts`
for the kernel contracts.

**Artifact validity.** The scheduling regression report retains the complete
comparisons, including failures; its expanded five-round ABBA comparison passes
the unchanged 15-percent guard across ten cases. That does not establish the
cause of wall-time variation. For register RMSNorm, use the matched-toolchain
final report. ``m5-register-rmsnorm.json`` and
``m5-register-rmsnorm-broadcast.json`` mix runtime JIT and offline compilation
and are not valid compiler-performance comparisons.

Source reports
--------------

Each report retains its hardware, software, shape, precision, and measurement
metadata where recorded. Tuning artifacts contain candidate measurements, not
independent heldout evidence. AIR inspection describes compiler output rather
than measured performance.

.. list-table:: Evidence by workload
   :header-rows: 1
   :widths: 28 72

   * - Workload
     - Reports
   * - Matched-weight model generation
     - :download:`4-bit models <../../benchmarks/results/m5-mlx-lm-models.json>`; :download:`dense BF16 Qwen 2.5 1.5B <../../benchmarks/results/m5-mlx-lm-bf16-dense-qwen15.json>`
   * - Compression-assisted generation
     - :download:`BF16 capacity suite <../../benchmarks/results/m5-mlx-lm-bf16-models.json>`
   * - Qwen3-0.6B megakernel
     - :download:`matched FP32 decode <../../benchmarks/results/m5-qwen3-megakernel-fp32.json>`; :download:`FP16 validation failure, no timings <../../benchmarks/results/m5-qwen3-megakernel-fp16-validation.json>`
   * - GPU-wide Qwen3-0.6B generation
     - :download:`matched FP32 end-to-end generation <../../benchmarks/results/m5-qwen3-gpu-wide-end-to-end.json>`; :download:`FP16 validation failure, no timings <../../benchmarks/results/m5-qwen3-gpu-wide-fp16-validation.json>`
   * - Long-document Qwen3-0.6B generation
     - :download:`shared-storage attention, 4096 input / 1024 output <../../benchmarks/results/m5-qwen3-prefill-shared-reuse-end-to-end.json>`; :download:`earlier 27-KiB matrix configuration <../../benchmarks/results/m5-qwen3-matrix-prefill-explicit-math-end-to-end.json>`; :download:`earlier scalar-attention configuration <../../benchmarks/results/m5-qwen3-chunked-prefill-end-to-end.json>`
   * - Matrix-attention numerical validation
     - :download:`pre-scaled-query failure, no timings <../../benchmarks/results/m5-qwen3-matrix-prefill-end-to-end.json>`; :download:`earlier base-two softmax failure, no timings <../../benchmarks/results/m5-qwen3-matrix-prefill-base2-end-to-end.json>`
   * - Residual MLP and projection sweeps
     - :download:`matched-representation matrix <../../benchmarks/results/m5-matched-representation-matrix.json>`; :download:`shape sensitivity <../../benchmarks/results/m5-shape-sensitivity.json>`; :download:`model-shaped layers <../../benchmarks/results/m5-model-shape-matrix.json>`
   * - RMSNorm register tiling
     - :download:`heldout run <../../benchmarks/results/m5-rmsnorm-tiling-heldout.json>`; :download:`repeat <../../benchmarks/results/m5-rmsnorm-tiling-heldout-repeat.json>`; :download:`frozen selection <../../benchmarks/results/m5-rmsnorm-tiling-selection.json>`; :download:`tuning matrix <../../benchmarks/results/m5-rmsnorm-tiling-tuning.json>`; :download:`AIR inspection <../../benchmarks/results/m5-rmsnorm-tiling-air.json>`
   * - FP16 tensor-memory GEMM
     - :download:`fixed suite <../../benchmarks/results/m5-tensor-memory.json>`; :download:`compiler baseline <../../benchmarks/results/m5-tensor-memory-baseline.json>`; :download:`MPP probe <../../benchmarks/results/m5-fp16-tensor-ops.json>`
   * - Epilogue fusion and scheduling
     - :download:`fusion run <../../benchmarks/results/m5-schedule-fusion.json>`; :download:`repeat <../../benchmarks/results/m5-schedule-fusion-repeat.json>`; :download:`scheduling regression <../../benchmarks/results/m5-schedule-regression.json>`
   * - Ownership and staging
     - :download:`ownership transpose <../../benchmarks/results/m5-thread-layouts.json>`; :download:`scalar-scatter baseline <../../benchmarks/results/m5-thread-layouts-baseline.json>`; :download:`verified staging <../../benchmarks/results/m5-verified-staging.json>`
   * - Register RMSNorm, loop-based baseline
     - :download:`matched-toolchain final report <../../benchmarks/results/m5-register-rmsnorm-final.json>`; its static-width candidate is not the static-register4 comparison above

Reproduce the figures
---------------------

Run these commands from the repository root. They load saved JSON and generate
SVG/PNG images; no Metal device, MLX installation, or model download is needed.

.. code-block:: bash

   python3 -m pip install -e '.[benchmarks]'
   python3 -m benchmarks.plots.render_model_speedups
   python3 -m benchmarks.plots.render_trial_distributions
   python3 -m benchmarks.plots.render_megakernel_results
   python3 -m benchmarks.plots.render_qwen3_end_to_end
   python3 -m benchmarks.plots.render_chunked_prefill \
     benchmarks/results/m5-qwen3-prefill-shared-reuse-end-to-end.json \
     --output docs/_static/qwen3-shared-prefill.png
   python3 -m benchmarks.plots.render_model_speedups --include-mixed \
     --throughput-output docs/_static/mlx-model-all-speedup.png \
     --latency-output docs/_static/mlx-model-all-latency-speedup.png
   python3 -m benchmarks.plots.render_matched_matrix
   python3 -m benchmarks.plots.render_model_shapes
   python3 -m benchmarks.plots.render_shape_sensitivity
   python3 -m benchmarks.plots.render_compiler_results

The compiler and memory diagrams have a separate, dependency-free SVG renderer:

.. code-block:: bash

   python3 -m benchmarks.plots.render_diagrams

Run fresh measurements
----------------------

Use a compatible Apple GPU and match the report's workload, precision, source,
and toolchain. Offline compiler studies also require the Metal toolchain.
Save new measurements to new files rather than replacing the published evidence.

.. code-block:: bash

   python3 -m pip install -e '.[dev,benchmarks]' -e ./kernels

Model studies also need the ``mlx-lm`` extra. The entry points are
``benchmarks.mlx.mlx_lm_suite`` for models, and
``benchmarks.mlx.matched_representation_matrix``,
``benchmarks.mlx.model_shape_matrix``, and ``benchmarks.mlx.shape_sensitivity``
for layers. Use ``--help`` to set the saved workload and trial counts, including
compression options and plan-confirmation settings.

For RMSNorm tiling, export the intended static-register4 baseline, tune once,
and validate the frozen selection twice. Keep the baseline, tuning report, and
manifest together. Their source fingerprints must match; changing source or
toolchain requires a fresh baseline and selection.

.. code-block:: bash

   python3 -m benchmarks.compiler.rmsnorm_tiling export \
     --root /path/to/pre-tiling-tree --output /tmp/prior-register4.json
   python3 -m benchmarks.compiler.rmsnorm_tiling tune \
     --baseline-json /tmp/prior-register4.json \
     --manifest /tmp/tiling-selection.json --output /tmp/tiling-tuning.json
   python3 -m benchmarks.compiler.rmsnorm_tiling validate \
     --baseline-json /tmp/prior-register4.json \
     --manifest /tmp/tiling-selection.json --output /tmp/tiling-heldout.json
   python3 -m benchmarks.compiler.rmsnorm_tiling validate \
     --baseline-json /tmp/prior-register4.json \
     --manifest /tmp/tiling-selection.json --output /tmp/tiling-heldout-repeat.json

Other compiler workloads expose their shape and timing controls through
``--help``:

.. code-block:: bash

   python3 -m benchmarks.compiler.tensor_memory --warmup-ms 100 --rep-ms 500 \
     --output-json /tmp/tensor-memory.json
   MLX_ENABLE_TF32=0 python3 -m benchmarks.compiler.schedule_fusion \
     --sizes 64 256 1024 --warmup-ms 100 --rep-ms 500 --output /tmp/fusion.json
   python3 -m benchmarks.regression.paired_regression \
     --baseline-root /path/to/pre-scheduling-tree --rounds-per-sample 5
   python3 -m benchmarks.compiler.thread_layouts \
     --shapes 4x8 8x8 8x16 16x16 --dtypes float16 float32 \
     --output-json /tmp/thread-layouts.json
   python3 -m benchmarks.compiler.staged_gemm \
     --baseline-root /path/to/pre-staging-tree --output /tmp/staging.json
   python3 -m benchmarks.compiler.register_rmsnorm \
     --baseline-root /path/to/pre-register-tree --output /tmp/register-rmsnorm.json

``make bench`` runs the numerical regression suite. To select kernel workloads,
use ``make bench BENCH_MODULES="benchmarks.kernels.gemm benchmarks.kernels.softmax"``.
Compiler, hardware, model, and regression drivers live in the corresponding
subdirectories of ``benchmarks/``; plotting code is in ``benchmarks/plots/``
and published reports are in ``benchmarks/results/``.
