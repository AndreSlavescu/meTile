import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import numpy as np
from benchutils import bench_interleaved

import metile
from kernels.gemm import matmul
from kernels.reduce import REDUCE_KERNELS
from metile.runtime.metal_device import MetalDevice

GEMM_CONFIGS = [
    # 4 SGs (2x2)
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, WM=2, WN=2, K_UNROLL=1),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WM=2, WN=2, K_UNROLL=1),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, WM=2, WN=2, K_UNROLL=2),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WM=2, WN=2, K_UNROLL=2),
    # 8 SGs (2x4)
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, WM=2, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, WM=2, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, WM=2, WN=4, K_UNROLL=2),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, WM=2, WN=4, K_UNROLL=2),
    # 16 SGs (4x4)
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=512, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, WM=4, WN=4, K_UNROLL=2),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, WM=4, WN=4, K_UNROLL=2),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, WM=4, WN=4, K_UNROLL=2),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=512, WM=4, WN=4, K_UNROLL=2),
]

autotuned_matmul = metile.autotune(
    configs=GEMM_CONFIGS,
    key=["M", "N", "K"],
    verbose=True,
)(matmul)

COOLDOWN = 3.0

COL_SIZE = 22
COL_T = 12


def _print_table(title, rows):
    print(f"\n  {title}")
    hdr = f"    {'size':>{COL_SIZE}}  {'metile (ms)':>{COL_T}}  {'MLX (ms)':>{COL_T}}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for size_str, dt_mtile, dt_mlx in rows:
        print(f"    {size_str:>{COL_SIZE}}  {dt_mtile:>{COL_T}.2f}  {dt_mlx:>{COL_T}.2f}")


def main():
    print("=== Split-K TF32 GEMM (autotuned) ===\n")

    configs = [
        # Small M*N, large K
        (128, 128, 4096, 4),
        (128, 128, 8192, 8),
        (256, 256, 4096, 4),
        (256, 256, 8192, 8),
        (512, 512, 4096, 4),
        (512, 512, 8192, 8),
        # Larger M*N
        (1024, 1024, 4096, 4),
        (1024, 1024, 8192, 8),
        # Rectangular, tall-skinny
        (1024, 256, 4096, 4),
        (1024, 256, 8192, 8),
        (2048, 256, 4096, 4),
        (2048, 512, 8192, 8),
        # Rectangular, short-wide
        (256, 1024, 4096, 4),
        (256, 1024, 8192, 8),
        (256, 2048, 4096, 4),
        (512, 2048, 8192, 8),
    ]

    dev = MetalDevice.get()
    rows = []

    for M, N, K, SPLIT_K in configs:
        assert K % SPLIT_K == 0
        K_SLICE = K // SPLIT_K
        n_elem = M * N

        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        A_slices, B_slices = [], []
        for s in range(SPLIT_K):
            k0 = s * K_SLICE
            A_slices.append(metile.Buffer(data=A_np[:, k0 : k0 + K_SLICE].copy().ravel()))
            B_slices.append(metile.Buffer(data=B_np[k0 : k0 + K_SLICE, :].copy().ravel()))

        C_partials = [metile.Buffer.zeros((n_elem,)) for _ in range(SPLIT_K)]
        C_out = metile.Buffer.zeros((n_elem,))

        def grid_fn(cfg, M=M, N=N):
            return (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(N, cfg["BLOCK_N"]))

        # Autotune on first slice, then use same config for all slices
        partial_dispatchers = []
        for s in range(SPLIT_K):
            d = autotuned_matmul[grid_fn].prepare(
                A_slices[s], B_slices[s], C_partials[s], M, N, K_SLICE
            )
            partial_dispatchers.append(d)

        reduce_grid = (metile.cdiv(n_elem, 256),)
        reduce_dispatch = REDUCE_KERNELS[SPLIT_K][reduce_grid].prepare(
            *C_partials, C_out, n_elem, BLOCK_SIZE=256
        )

        dev.sync()

        def split_k_fn(pd=partial_dispatchers, rd=reduce_dispatch):
            for d in pd:
                d()
            rd()

        A_mx, B_mx = mx.array(A_np), mx.array(B_np)

        def mlx_fn(a=A_mx, b=B_mx):
            mx.eval(a @ b)

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(split_k_fn, mlx_fn, dev.sync)

        size_str = f"{M}x{N}x{K} k={SPLIT_K}"
        rows.append((size_str, dt_mtile * 1000, dt_mlx * 1000))

    _print_table("split-K matmul vs MLX", rows)
    print()


if __name__ == "__main__":
    main()
