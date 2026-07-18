# meTile

Tile-based eDSL and compiler for Apple GPUs. Write tile programs in Python, compile to Metal.

```
@metile.kernel (Python eDSL)
    - Tile IR (hardware-agnostic)
    - Metal IR (simdgroup mappings, threadgroup memory)
    - MSL codegen
    - xcrun metal -O2 (precompiled metallib)
    - dispatch via ctypes Metal bridge
```

## Example: GEMM

```python
@metile.kernel
def gemm(A, B, C, M, N, K,
         BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
         BLOCK_K: metile.constexpr):
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))
```

## Example: Softmax

```python
@metile.kernel
def softmax(X, Out, N, BLOCK: metile.constexpr):
    row = metile.program_id(0)
    m = -1e38
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        m = metile.maximum(m, x)
    m = metile.max(m)

    s = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        s = s + metile.exp(x - m)
    s = metile.sum(s)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        mask = cols < N
        x = metile.load(X + row * N + cols, mask=mask)
        metile.store(Out + row * N + cols, metile.exp(x - m) / s, mask=mask)
```

## Features

**eDSL & Frontend**
- Python-based eDSL: `@metile.kernel`, `program_id`, `arange`, `load`/`store`, `dot`, `tile_load`/`tile_store`
- Explicit `cast` operations keep mixed-precision accumulation and storage choices visible in Tile IR.
- Explicit loop-carried scalar SSA values plus native `simd_sum`, `simd_max`, and `fast_exp` primitives for composable online algorithms
- Autotuner (`@metile.autotune`) with config search over block sizes, SG counts, execution modes, and tile schedules

**Compiler**
- Multi-level IR pipeline: Tile IR (hardware-agnostic) &rarr; Metal IR (decomposed primitives) &rarr; MSL
- CuTe-inspired layout algebra with hierarchical Shape:Stride, composition, complement, and logical divide. Supports arbitrary tile shapes.
- Composable optimization passes that transform IR structure: shared memory padding / XOR swizzle, split-K, vectorized loads, serpentine MMA traversal, preloaded tiles, double-buffered K-loop, block swizzle for L2 locality
- Schedule-algebra pass over finite tile permutations. Generator closure derives the shape-preserving D4, D2, C2, or trivial action; orbit representatives remove equivalent traversals before emitting branch-free linear, grouped-2/4/8, diagonal, Morton, or 4x4 Hilbert schedules.
- Schedule decoders are composable scalar-expression programs, not whole-kernel templates. Extraction strength-reduces exact constant divisions and chooses target-operation cost first, then compressed canonical-program length as a computable minimum-description-length upper bound.
- Runtime remains the primary objective across kernel candidates. Compressed generated MSL only breaks measured latency ties within 0.25% of the fastest representation.
- Proof-carrying reduction discovery models candidate algorithms as finite summary monoids. A restricted equational verifier checks identity, generated associativity, and list-homomorphism obligations before graph rewrites may use sum, max, or stable weighted-softmax states.
- Exact attention discovery recognizes private `softmax(scale(Q @ K.T)) @ V` DAG regions, proves the online `(maximum, normalizer, numerator)` state, and replaces the materializing chain with one `flash_attention` operation.
- Composable MXFP4/MXFP8 Metal IR operations for vectorized fused decode, optional threadgroup staging, register-resident NAX fragments, MPP matrix multiply, and stores.
- NAX setup/run/epilogue/store lowering decomposes into tile-layout, vector-load, cooperative-tensor pack, MMA, per-fragment apply, and fragment-store IR. Shape-tuned reduction epochs can preload adjacent K fragments without owning a whole kernel template.
- Autotuned fused epilogues (ReLU, GELU, SiLU, exp, scale) run directly on cooperative-tensor or NAX register fragments with zero intermediate global memory traffic.

**Codegen**
- Simdgroup matrix (8x8) MMA with decomposed load / MMA / store primitives
- Native unsigned `reverse_bits` lowering for branch-free permutation decoders
- Metal 4 tensor_ops (`matmul2d`) with runtime GPU-family and toolchain checks; the M5 path supports preemptive, cooperative, and direct register-fragment execution
- Per-kernel `max_total_threads_per_threadgroup` specialization and shared op-by-op emission instead of monolithic whole-kernel templates
- AOT compilation via `xcrun metal -O2` with JIT fallback (`newLibraryWithSource`) when Xcode is unavailable

**Runtime**
- Zero-copy unified memory via `metile.Buffer`. CPU and GPU share the same physical memory.
- Interleaved round-robin autotuning uses synchronized end-to-end latency for sub-millisecond kernels and GPU timestamps for sustained workloads, with launch-grid-, shape-, device-, and toolchain-keyed persistent config, measured-latency, and metallib caches.
- Online decode attention is composed from ordinary eDSL kernels. The runtime measures single-pass 2/4/8/16/32-SIMDgroup schedules and long-context two-pass token partitions per head count, context length, and head dimension.
- Automatic dense and block-scaled tile/schedule dispatch across grouped, Morton, Hilbert, staged, and register-resident candidates, including 2- and 4-SIMDgroup MXFP tiles.
- Batched FFT dispatch searches threadgroup width, register-local radix decomposition, bit-reversal placement, and twiddle placement per transform shape instead of selecting a monolithic kernel template.
- Aligned NAX kernels specialize dimensions and bind only matrix buffers on the prepared hot path; reduction epoch and K-fragment preload choices remain runtime-tuned per shape.
- Prepared calls bulk-bind buffers, reuse unchanged encoder state, batch compatible launches, and expose `repeat(count)` to encode repeated work under one lock; measured short kernels receive an adaptive bounded poll-before-sleep budget while longer workloads block immediately.
- Pure Python runtime. meTile has a ctypes Metal bridge with no PyObjC dependency.
- Optional zero-copy MLX graph primitives and a reversible MLX-LM patch select between generated meTile kernels and native MLX per shape, including fused affine-quantized SwiGLU and down-projection/residual decode candidates.
- Discovered FlashAttention regions race native MLX against causal/noncausal row-tiled online kernels and persist only compatible winners that clear the 5% framework boundary.

## Block-Scaled GEMM

MXFP4 and MXFP8 weights use 32-value groups with E8M0 scales. The compiler fuses
dequantization into the same Metal 4 MPP kernel instead of materializing a dense weight:

```python
a = metile.Buffer(data=np.random.randn(128, 1024).astype(np.float32))
w = metile.BlockScaledWeight.quantize(weight, format="mxfp4")  # weight is K x N

out = metile.block_scaled_matmul(a, w)
```

The aligned fast path currently requires `M`/`N` multiples of 64 and `K` a multiple
of 32. `prepare_block_scaled_matmul` autotunes staged and direct register-fragment
representations, including paired K steps that reuse E8M0 scale fragments and 32x64
two-SIMDgroup output tiles, and returns a reusable hot-path dispatcher.

## Decode Attention

The decode kernels keep query values and online-softmax state in registers and merge
SIMDgroup partials without materializing an attention matrix. For long contexts, the
runtime also measures a multi-threadgroup first pass and a second online merge:

```python
from kernels import attention_decode

dispatch = attention_decode[(batch, query_heads)].prepare(
    query,
    key,
    value,
    output,
    context_length,
    head_dim**-0.5,
    D=head_dim,
    KV_HEADS=key_value_heads,
)
dispatch()
```

The public kernel accepts contiguous float32 MHA/GQA/MQA tensors flattened as query/output
``[batch, query_heads, D]`` and key/value ``[batch, key_value_heads, tokens, D]``, with
``query_heads`` divisible by ``key_value_heads`` and ``D`` divisible by 32. The original
one-dimensional ``(heads,)`` launch remains a batch-one MHA shorthand.

## FlashAttention Discovery

The high-level compute DAG can express ordinary matmul, scale, causal-mask, and
softmax nodes without prescribing a kernel boundary. Before backend fusion, the
compiler finds exact attention chains whose score/probability intermediates do not
escape. It then discharges an equational proof over the stable weighted-softmax
summary and emits one proof-carrying `flash_attention` node. Invalid merge equations,
wrong reduction axes, incompatible shapes, and escaping intermediates reject the
rewrite.

The generated Metal candidate assigns one query row per threadgroup, streams K/V
without materializing the score matrix, and merges SIMDgroup-local online states.
It supports aligned causal masks and MHA/GQA shapes with head dimensions divisible by
32. Native MLX is still a candidate, numerical compatibility is checked first, and a
31-round finalist tournament requires 5% headroom before switching. This follows the
exact online-normalizer construction and IO-aware attention decomposition described by
[Milakov and Gimelshein](https://arxiv.org/abs/1805.02867) and
[Dao et al.](https://arxiv.org/abs/2205.14135), but does not assume an NVIDIA warp or
CUDA-specific whole-kernel template.

```bash
python benchmarks/flash_attention_discovery.py --trials 31
```

## Automatic Gated Epilogue Fusion

The graph optimizer recognizes `matmul(x, W_gate)`, `matmul(x, W_up)`,
`silu(gate) * up`, and the consuming down projection as a declarative
`ParallelEpilogueRule`. It rejects escaping intermediates and target-resource violations,
then lets each backend choose its own lowering rather than instantiating a whole MLP template.

For affine 4-bit weights, the Metal runtime now tunes a scratch-spilled schedule alongside
native MLX, compiled MLX, and register-fused meTile. Gate accumulators are reduced into
threadgroup memory before up accumulators become live; after a barrier, SwiGLU consumes the
on-chip values. The native M5 path uses the same lifetime split between `matmul2d` fragment
reductions, reuses two accumulator vectors, and transposes lane/element scratch indices to avoid
32-bank conflicts. Both schedules remain guarded autotune candidates, so losing shapes preserve
MLX. L2 reuse for the immediately following down projection is a scheduling opportunity, not a
cache-pinning guarantee on Metal.

The decode backend also composes the selected gate/up implementation with a down-projection
epilogue that adds the transformer residual before the result leaves the kernel. Its schedule
tournament covers eager and compiled MLX plus generated block width, outputs per SIMD-group,
and FP16/FP32 decode choices. A warmed, shape-specialized MLP executor binds the winning
weights and kernels once, avoiding repeated Python dispatch construction without turning the
MLP into a monolithic compiler template.

## MLX-LM Backend

meTile-generated Metal can execute as a lazy, zero-copy MLX primitive. The integration
uses a Liger-style opt-in patch with independent attention, RMSNorm, graph-fusion,
and quantized-MLP switches:

```python
from mlx_lm import load
from metile.integrations.mlx_lm import (
    apply_metile_to_mlx_lm,
    prepare_mlx_lm_affine_prefill,
)

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
affine_prefill = prepare_mlx_lm_affine_prefill(model)
patch = apply_metile_to_mlx_lm(model=model, affine_prefill=affine_prefill)

# Restore every patched function or use the handle as a context manager.
patch.restore()
```

The dispatcher benchmarks native MLX alongside generated blocks. Attention and RMSNorm
require at least 5% primitive-level headroom before crossing the framework boundary;
otherwise the call stays on MLX. A high-level compute DAG also discovers multi-output
residual-add/RMSNorm fusion using an exact max-flow/min-cut pass and a stricter 10% switch
margin. Unsupported attention modes, masks, sinks, quantized KV caches, and dtypes fall
back exactly. RMSNorm supports FP16/FP32 and accumulates in FP32.

For affine 4-bit model weights, AOT preparation preserves the original quantized values while
transposing packed nibbles and scale/bias groups into a K-major NAX view. Ragged prefill rows
then tune native MLX against generated 32-, 64-, and 128-row NAX workgroups with Morton,
grouped, Hilbert, and linear schedules. Both tile axes are runtime decisions, and ragged row
loads and stores remain masked. Only prepared projection instances change class, so unrelated
linear layers keep the exact MLX path and
decode remains independently eligible for the guarded down/residual path in canonical Llama
and Qwen2 blocks. Unsupported or multi-row calls immediately use the original block, while a
warmed decode executor bypasses repeated compatibility and kernel-construction work. Model
plans require matching next tokens, bounded KL divergence, bounded mean/max logit error, a
measured TTFT or end-to-end win, and bounded decode/total behavior. Decode-sensitive plans keep
the stricter 0.5% confirmation floor; self-deoptimizing prefill-only plans use a 1% noise floor.

The optional MLX primitive backend also accepts MXFP4 and MXFP8 K-major weights. It emits
register-fragment block-scale decode plus NAX ``matmul2d`` directly into the MLX lazy graph,
autotunes multiple tile/schedule representations, and keeps MLX arrays as the only storage.

The committed M5 32 GB suite uses MLX 0.32.0, MLX-LM 0.31.3, a 128-token prompt,
256 generated tokens, five end-to-end confirmation pairs, and nine continuous measurement
pairs. It verifies bounded logit fidelity, tunes the complete model plan, and confirms it on
the full generation workload before measurement. When no optimized plan clears the TTFT,
decode, and end-to-end safety bars, both labels share the same native measurement instead of
presenting system noise as a speedup:

| Prefill and decode throughput | TTFT and end-to-end latency |
|:--:|:--:|
| ![Native MLX and MLX with meTile prefill and decode throughput across four models](docs/_static/mlx-model-throughput.png) | ![Native MLX and MLX with meTile TTFT and end-to-end latency across four models](docs/_static/mlx-model-latency.png) |

| Model | MLX decode | MLX + meTile | Native TTFT | Decode | Prefill | TTFT | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B 4-bit | 151.36 tok/s | 150.98 tok/s | 102.0 ms | 0.998x | 1.339x | 1.135x | 1.004x |
| Llama 3.2 3B 4-bit | 61.35 tok/s | 61.35 tok/s | 153.9 ms | 1.000x | 1.000x | 1.000x | 1.000x |
| Qwen 2.5 0.5B 4-bit | 309.19 tok/s | 304.62 tok/s | 70.8 ms | 0.994x | 1.275x | 1.072x | 1.001x |
| Qwen 2.5 1.5B 4-bit | 118.54 tok/s | 119.65 tok/s | 130.6 ms | 1.001x | 1.331x | 1.170x | 1.013x |

Three workloads selected the generated affine-prefill path; Llama 3.2 3B retained native
MLX. Speedups are medians of paired ratios, while chart bars show absolute medians. Raw trials,
confirmation pairs, environment metadata, fidelity metrics, comparison mode, and selected
dispatches are in `benchmarks/results/m5-mlx-lm-models.json`. Reproduce the suite and regenerate
the PNG bar charts:

```bash
python benchmarks/mlx_lm_suite.py \
  --prompt-tokens 128 --generation-tokens 256 --trials 9 --delay 0 \
  --plan-trials 7 --confirmation-trials 5 \
  --output benchmarks/results/m5-mlx-lm-models.json

python benchmarks/render_mlx_lm_results.py \
  benchmarks/results/m5-mlx-lm-models.json
```

## Install

```bash
pip install -e ".[dev]"

# Optional framework backend
pip install -e ".[mlx-lm]"

# Optional benchmark chart renderer
pip install -e ".[benchmarks]"
```

## Run Tests

```bash
# run individual test
python -m pytest tests/test_gemm.py -v

# run all like so
python -m pytest tests/ -x -q

# or with `make test`
make test
```

## Run Benchmarks

```bash
# run individual benchmark
python benchmarks/gemm.py

# or run `make bench` for running all the benchmarks
make bench

# compare a checkout against this tree on the same Apple runner
python benchmarks/paired_regression.py --baseline-root /path/to/baseline
```

Pull-request CI reports an ABBA-ordered, launch-to-completion comparison against
the checked-out base revision. If one revision cannot compile a group on the hosted
Apple target, the report marks that group unavailable and compares only the exact
supported intersection. Shared-host measurements remain diagnostic rather than
blocking; reproducible performance claims use the interleaved M5 harnesses.

## Architecture

![meTile compiler and runtime architecture](docs/_static/compilation-pipeline.svg)

The [compiler architecture guide](docs/guide/architecture.rst) expands the graph-planning,
proof-carrying discovery, kernel-lowering, and guarded-runtime stages.

| Layer | File | Role |
|-------|------|------|
| Frontend | `frontend/kernel.py` | `@kernel` decorator, compilation pipeline, dispatch |
| Frontend | `frontend/tracing.py` | eDSL ops, constexpr folding, tensor descriptors |
| Tile IR | `ir/tile_ir.py` | Hardware-agnostic tile operations |
| Metal IR | `ir/metal_ir.py` | Decomposed Apple GPU primitives (simdgroup, tensor_ops, cooperative loads) |
| Layout | `ir/layout.py` | CuTe-inspired layout algebra (Shape:Stride, composition, logical divide) |
| Lowering | `compiler/lowering.py` | Tile IR &rarr; Metal IR (GEMM detection, simdgroup/tensor_ops paths) |
| Passes | `compiler/passes.py` | IR &rarr; IR transforms (serpentine, preload, pad, swizzle, split-K, vectorize) |
| Schedule search | `compiler/schedule_search.py` | Permutation-group canonicalization and MDL/locality schedule selection |
| Reduction algebra | `compiler/reduction_algebra.py` | Restricted proof obligations for composable streaming reductions |
| Attention discovery | `compiler/attention_discovery.py` | Certified graph recognition and FlashAttention replacement |
| Block scaling | `compiler/block_scaled.py` | Composes MXFP decode, tensor views, MPP MMA, and store Metal IR |
| Codegen | `codegen/msl_emitter.py` | Metal IR &rarr; MSL (uniform op walker, no per-kernel templates) |
| Runtime | `runtime/metal_device.py` | Metal API via ctypes (compile, capability-gated batching, dispatch, sync) |
| Runtime | `runtime/buffer.py` | Zero-copy unified memory buffers |
| Runtime | `runtime/block_scaled.py` | MXFP quantization and shape-specific tile-family dispatch |
| Attention | `kernels/attention.py` | Composable online-softmax decode kernel and schedule family |
| MLX backend | `backends/mlx.py` | Zero-copy MLX primitives and native/generated guarded dispatch |
| MLX graph backend | `backends/mlx_graph.py` | High-level graph execution, discovery, fusion, and guarded fallback |
| MLX-LM integration | `integrations/mlx_lm.py` | Reversible Liger-style model patching with exact fallbacks |

## Citations

metile's layout algebra is directly inspired by CuTe's hierarchical layout representation:

```bibtex
@misc{cecka2026cute,
    title={CuTe Layout Representation and Algebra},
    author={Cris Cecka},
    year={2026},
    eprint={2603.02298},
    archivePrefix={arXiv},
    primaryClass={cs.MS},
    url={https://arxiv.org/abs/2603.02298}
}
```

The `metile/ir/layout.py` module implements CuTe's core concepts, `Layout(shape, stride)` with hierarchical tuples, colexicographic coordinate mapping, and algebraic operations (coalesce, compose, complement, logical divide, logical product), adapted for Apple GPU tiling patterns (simdgroup 8x8, threadgroup memory banking, cooperative loads).

metile's eDSL design and tile-level programming model draws from Triton, which is a popular tile-based multi-program multi-data pythonic eDSL:

```bibtex
@inproceedings{tillet2019triton,
    title={Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
    author={Philippe Tillet and H. T. Kung and David Cox},
    booktitle={Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year={2019},
    doi={10.1145/3315508.3329973}
}
```

## Links

- [Contributing](.github/CONTRIBUTING.md)
- [Performance Dashboard](https://andreslavescu.github.io/meTile/dev/bench/)

## License

MIT
