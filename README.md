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
- Optional zero-copy MLX graph primitives and a reversible MLX-LM patch select between generated meTile kernels and native MLX per shape, including fused affine-quantized SwiGLU decode candidates.
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

## MLX-LM Backend

meTile-generated Metal can execute as a lazy, zero-copy MLX primitive. The integration
uses a Liger-style opt-in patch with independent attention, RMSNorm, graph-fusion,
and quantized-MLP switches:

```python
from mlx_lm import load
from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
patch = apply_metile_to_mlx_lm(model=model)

# Restore every patched function or use the handle as a context manager.
patch.restore()
```

The dispatcher benchmarks native MLX alongside generated blocks. It requires at least
5% primitive-level headroom before crossing the framework boundary; otherwise the call
stays on MLX. A high-level compute DAG also discovers multi-output residual-add/RMSNorm
fusion using an exact max-flow/min-cut pass and a stricter 10% switch margin. Unsupported
prefill attention, masks, sinks, quantized KV caches, and dtypes also fall back exactly.
RMSNorm supports FP16/FP32 and accumulates in FP32.

For affine 4-bit Llama MLPs, AOT repacking transposes packed nibbles and scale/bias groups
once, then measures native MLX against generated scalar decode and Metal 4 NAX
``matmul2d`` plus fused SwiGLU candidates. Scalar schedules independently tune threadgroup
width, adjacent outputs per SIMDgroup, and FP16/FP32 decode arithmetic while retaining FP32
accumulators. The selector verifies numerical compatibility, requires 10% headroom, persists
the decision by device/source/shape, and discards repacked weights when native MLX wins.

On this M5 32 GB machine with MLX 0.32.0 and MLX-LM 0.31.3, a five-trial interleaved
Llama 3.2 1B 4-bit run at 128 prompt / 256 generated tokens measured 93.39 tok/s for
MLX and 96.06 tok/s with the guarded patch (1.029x decode, 1.035x end to end). The
affine SwiGLU selector retained native MLX for that run; meTile wins came from RMSNorm
and the 512-token attention bucket. Reproduce the model-level benchmark rather than
relying on that machine-specific number:

```bash
python benchmarks/mlx_lm_backend.py \
  --prompt-tokens 128 --generation-tokens 256 --trials 5 --delay 2
```

## Install

```bash
pip install -e ".[dev]"

# Optional framework backend
pip install -e ".[mlx-lm]"
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
```

## Architecture

```
@metile.kernel (Python eDSL)
    - Tile IR (hardware-agnostic ops: program_id, tile_load, dot, ...)
    - Metal IR (decomposed primitives: simdgroup load/MMA/store, cooperative_tensor ops)
    - Optimization passes (serpentine reordering, preload, pad/swizzle, split-K, ...)
    - MSL codegen (op-by-op emission)
    - xcrun metal -O2 (precompiled metallib)
    - dispatch via ctypes Metal bridge
```

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
| Block scaling | `compiler/block_scaled.py` | Composes MXFP decode, tensor views, MPP MMA, and store Metal IR |
| Codegen | `codegen/msl_emitter.py` | Metal IR &rarr; MSL (uniform op walker, no per-kernel templates) |
| Runtime | `runtime/metal_device.py` | Metal API via ctypes (compile, capability-gated batching, dispatch, sync) |
| Runtime | `runtime/buffer.py` | Zero-copy unified memory buffers |
| Runtime | `runtime/block_scaled.py` | MXFP quantization and shape-specific tile-family dispatch |
| Attention | `kernels/attention.py` | Composable online-softmax decode kernel and schedule family |
| MLX backend | `backends/mlx.py` | Zero-copy MLX primitives and native/generated guarded dispatch |
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
