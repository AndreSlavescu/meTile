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
from kernels.fft import fft_dispatch
from metile.runtime.metal_device import MetalDevice

COOLDOWN = 3.0

COL_SIZE = 16
COL_T = 12


def _print_table(title, rows):
    print(f"\n  {title}")
    hdr = f"    {'size':>{COL_SIZE}}  {'metile (us)':>{COL_T}}  {'MLX (us)':>{COL_T}}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for size_str, dt_mtile, dt_mlx in rows:
        print(f"    {size_str:>{COL_SIZE}}  {dt_mtile:>{COL_T}.1f}  {dt_mlx:>{COL_T}.1f}")


def main():
    print("=== FFT (batched 1D) ===\n")

    dev = MetalDevice.get()
    rows = []

    configs = [
        (1, 64),
        (1, 256),
        (1, 1024),
        (1, 2048),
        (32, 64),
        (32, 256),
        (32, 1024),
        (32, 2048),
        (128, 256),
        (128, 1024),
        (128, 2048),
    ]

    for batch, N in configs:
        x_np = np.random.randn(batch, N).astype(np.float32)

        xr = metile.Buffer(data=x_np.ravel())
        xi = metile.Buffer(data=np.zeros(batch * N, dtype=np.float32))
        yr = metile.Buffer.zeros((batch * N,))
        yi = metile.Buffer.zeros((batch * N,))

        dispatch = fft_dispatch(batch, N, xr, xi, yr, yi)

        # MLX
        x_mx = mx.array(x_np) + 0j
        mx.eval(mx.fft.fft(x_mx, axis=-1))

        def mlx_fn(x=x_mx):
            mx.eval(mx.fft.fft(x, axis=-1))

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(dispatch, mlx_fn, dev.sync)

        size_str = f"{batch}x{N}"
        rows.append((size_str, dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table("FFT (metile fused vs MLX)", rows)
    print()


if __name__ == "__main__":
    main()
