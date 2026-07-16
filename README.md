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
- Autotuner (`@metile.autotune`) with config search over block sizes, SG counts, execution modes, and tile schedules

**Compiler**
- Multi-level IR pipeline: Tile IR (hardware-agnostic) &rarr; Metal IR (decomposed primitives) &rarr; MSL
- CuTe-inspired layout algebra with hierarchical Shape:Stride, composition, complement, and logical divide. Supports arbitrary tile shapes.
- Composable optimization passes that transform IR structure: shared memory padding / XOR swizzle, split-K, vectorized loads, serpentine MMA traversal, preloaded tiles, double-buffered K-loop, block swizzle for L2 locality
- Schedule-algebra pass over finite tile permutations. D4/rectangle group actions remove symmetry-equivalent traversals before emitting branch-free linear, grouped-2/4/8, diagonal, Morton, or 4x4 Hilbert schedules.
- Minimum-description-length selection uses compressed generated source as a computable upper bound on Kolmogorov complexity. Runtime remains the primary objective; source size only breaks measurements within 0.25% of the fastest candidate.
- Composable MXFP4/MXFP8 Metal IR operations for vectorized fused decode, optional threadgroup staging, register-resident NAX fragments, MPP matrix multiply, and stores.
- NAX setup/run/store lowering decomposes into tile-layout, vector-load, cooperative-tensor pack, MMA, and fragment-store IR. Shape-tuned reduction epochs can preload adjacent K fragments without owning a whole kernel template.
- Fused epilogues (ReLU, exp, scale) on register-resident accumulators via `thread_elements()` with zero global memory traffic.

**Codegen**
- Simdgroup matrix (8x8) MMA with decomposed load / MMA / store primitives
- Metal 4 tensor_ops (`matmul2d`) with runtime GPU-family and toolchain checks; the M5 path supports preemptive, cooperative, and direct register-fragment execution
- Per-kernel `max_total_threads_per_threadgroup` specialization and shared op-by-op emission instead of monolithic whole-kernel templates
- AOT compilation via `xcrun metal -O2` with JIT fallback (`newLibraryWithSource`) when Xcode is unavailable

**Runtime**
- Zero-copy unified memory via `metile.Buffer`. CPU and GPU share the same physical memory.
- Interleaved round-robin GPU-timestamp autotuning with device/toolchain-keyed persistent config and metallib caches.
- Automatic dense and block-scaled tile/schedule dispatch across grouped, Morton, Hilbert, staged, and register-resident candidates.
- Aligned NAX kernels specialize dimensions and bind only matrix buffers on the prepared hot path; reduction epoch and K-fragment preload choices remain runtime-tuned per shape.
- Prepared calls batch compatible launches into shared encoders, insert dependency barriers for concurrent dispatch, retain bound resources, and fall back when optional Metal selectors are unavailable.
- Pure Python runtime. meTile has a ctypes Metal bridge with no PyObjC dependency.

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
representations, including paired K steps that reuse E8M0 scale fragments, and returns
a reusable hot-path dispatcher.

## Install

```bash
pip install -e ".[dev]"
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
