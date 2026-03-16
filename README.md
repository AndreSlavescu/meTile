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
- Autotuner (`@metile.autotune`) with config search over block sizes, SG counts, execution modes

**Compiler**
- Multi-level IR pipeline: Tile IR (hardware-agnostic) &rarr; Metal IR (decomposed primitives) &rarr; MSL
- CuTe-inspired layout algebra with hierarchical Shape:Stride, composition, complement, and logical divide. Supports arbitrary tile shapes.
- Composable optimization passes that transform IR structure: shared memory padding / XOR swizzle, split-K, vectorized loads, serpentine MMA traversal, preloaded tiles, double-buffered K-loop, block swizzle for L2 locality
- Fused epilogues (ReLU, exp, scale) on register-resident accumulators via `thread_elements()` with zero global memory traffic.

**Codegen**
- Simdgroup matrix (8x8) MMA with decomposed load / MMA / store primitives
- Metal 4 tensor_ops (`matmul2d`) for TF32 on M5, M5 pro, and M5 max (requires Xcode to be built as well) via preemptive and cooperative execution modes
- AOT compilation via `xcrun metal -O2` with JIT fallback (`newLibraryWithSource`) when Xcode is unavailable

**Runtime**
- Zero-copy unified memory via `metile.Buffer`. CPU and GPU share the same physical memory.
- Pure Python runtime. meTile has a ctypes Metal bridge with no PyObjC dependency.

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
| Codegen | `codegen/msl_emitter.py` | Metal IR &rarr; MSL (uniform op walker, no per-kernel templates) |
| Runtime | `runtime/metal_device.py` | Metal API via ctypes (compile, dispatch, sync) |
| Runtime | `runtime/buffer.py` | Zero-copy unified memory buffers |

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
