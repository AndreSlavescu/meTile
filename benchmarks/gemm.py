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
from metile.runtime.metal_device import MetalDevice

GEMM_CONFIGS = [
    # 4 SGs (2x2)
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, WM=2, WN=2, K_UNROLL=1),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WM=2, WN=2, K_UNROLL=1),
    # 8 SGs (2x4)
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, WM=2, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, WM=2, WN=4, K_UNROLL=1),
    # 16 SGs (4x4)
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=512, WM=4, WN=4, K_UNROLL=1),
]

autotuned_matmul = metile.autotune(
    configs=GEMM_CONFIGS,
    key=["M", "N", "K"],
    verbose=True,
)(matmul)

COOLDOWN = 3.0

COL_SIZE = 20
COL_T = 12


def _print_table(title, rows):
    print(f"\n  {title}")
    hdr = f"    {'size':>{COL_SIZE}}  {'metile (ms)':>{COL_T}}  {'MLX (ms)':>{COL_T}}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for size_str, dt_mtile, dt_mlx in rows:
        print(f"    {size_str:>{COL_SIZE}}  {dt_mtile:>{COL_T}.2f}  {dt_mlx:>{COL_T}.2f}")


def main():
    sizes = [
        # Square
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        # Rectangular, tall-skinny
        (2048, 512, 1024),
        (4096, 256, 1024),
        (4096, 1024, 512),
        (8192, 512, 1024),
        # Rectangular, short-wide
        (512, 2048, 1024),
        (256, 4096, 1024),
        (1024, 4096, 512),
        (512, 8192, 1024),
    ]
    if len(sys.argv) > 1:
        sizes = [(int(sys.argv[1]),) * 3]

    print("=== TF32 GEMM (autotuned) ===\n")

    dev = MetalDevice.get()
    rows = []

    for M, N, K in sizes:
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        A_buf = metile.Buffer(data=A_np.ravel())
        B_buf = metile.Buffer(data=B_np.ravel())
        C_buf = metile.Buffer.zeros((M * N,))

        def grid_fn(cfg, M=M, N=N):
            return (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(N, cfg["BLOCK_N"]))

        dispatch = autotuned_matmul[grid_fn].prepare(A_buf, B_buf, C_buf, M, N, K)
        dev.sync()

        A_mx, B_mx = mx.array(A_np), mx.array(B_np)

        def mlx_gemm(a=A_mx, b=B_mx):
            mx.eval(a @ b)

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(dispatch, mlx_gemm, dev.sync)

        rows.append((f"{M}x{N}x{K}", dt_mtile * 1000, dt_mlx * 1000))

    _print_table("matmul (metile tensor_ops vs MLX)", rows)
    print()


if __name__ == "__main__":
    main()
