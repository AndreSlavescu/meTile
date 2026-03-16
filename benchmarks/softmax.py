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
from kernels.softmax import softmax
from metile.runtime.metal_device import MetalDevice

SOFTMAX_CONFIGS = [
    metile.Config(BLOCK=64),
    metile.Config(BLOCK=128),
    metile.Config(BLOCK=256),
    metile.Config(BLOCK=512),
    metile.Config(BLOCK=1024),
]

autotuned_softmax = metile.autotune(
    configs=SOFTMAX_CONFIGS,
    key=["N"],
    verbose=True,
)(softmax)

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
    print("=== Softmax (autotuned) ===\n")

    dev = MetalDevice.get()
    rows = []

    for nrows, hidden in [(128, 512), (256, 1024), (512, 2048), (1024, 4096)]:
        X_np = np.random.randn(nrows, hidden).astype(np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        Out_buf = metile.Buffer.zeros((nrows * hidden,))

        grid = (nrows,)
        dispatch = autotuned_softmax[grid].prepare(X_buf, Out_buf, hidden)

        dev.sync()

        X_mx = mx.array(X_np)

        def mlx_fn(x=X_mx):
            mx.eval(mx.softmax(x, axis=-1))

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(dispatch, mlx_fn, dev.sync)

        rows.append((f"{nrows}x{hidden}", dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table("softmax vs MLX", rows)
    print()


if __name__ == "__main__":
    main()
