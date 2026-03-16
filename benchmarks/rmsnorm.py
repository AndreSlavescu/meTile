import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from benchutils import bench_interleaved

import metile
from kernels.rmsnorm import rmsnorm
from metile.runtime.metal_device import MetalDevice

RMSNORM_CONFIGS = [
    metile.Config(BLOCK=64),
    metile.Config(BLOCK=128),
    metile.Config(BLOCK=256),
    metile.Config(BLOCK=512),
    metile.Config(BLOCK=1024),
]

autotuned_rmsnorm = metile.autotune(
    configs=RMSNORM_CONFIGS,
    key=["N"],
    verbose=True,
)(rmsnorm)

COOLDOWN = 3.0

COL_SIZE = 12
COL_T = 12


def _print_table(title, rows):
    print(f"\n  {title}")
    hdr = f"    {'size':>{COL_SIZE}}  {'metile (us)':>{COL_T}}  {'MLX (us)':>{COL_T}}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for size_str, dt_mtile, dt_mlx in rows:
        print(f"    {size_str:>{COL_SIZE}}  {dt_mtile:>{COL_T}.1f}  {dt_mlx:>{COL_T}.1f}")


def main():
    print("=== RMSNorm (autotuned) ===\n")

    dev = MetalDevice.get()
    rows = []

    for nrows, hidden in [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]:
        X_np = np.random.randn(nrows, hidden).astype(np.float32)
        W_np = np.random.randn(hidden).astype(np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W_buf = metile.Buffer(data=W_np.ravel())
        Out_buf = metile.Buffer.zeros((nrows * hidden,))

        grid = (nrows,)
        dispatch = autotuned_rmsnorm[grid].prepare(X_buf, W_buf, Out_buf, hidden)

        dev.sync()

        rmsnorm_mlx = nn.RMSNorm(hidden)
        rmsnorm_mlx.weight = mx.array(W_np)
        X_mx = mx.array(X_np)

        def mlx_fn(norm=rmsnorm_mlx, x=X_mx):
            mx.eval(norm(x))

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(dispatch, mlx_fn, dev.sync)

        rows.append((f"{nrows}x{hidden}", dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table("rmsnorm vs MLX", rows)
    print()


if __name__ == "__main__":
    main()
