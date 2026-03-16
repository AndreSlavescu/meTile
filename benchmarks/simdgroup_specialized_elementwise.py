"""Element-wise kernels vs MLX.

Single-output kernels compared against native MLX ops.
Fused dual-output kernels (simdgroup specialization) compared against mx.compile().

Usage:
    python benchmarks/simdgroup_specialized_elementwise.py
"""

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
from kernels.simdgroup_specialized_elementwise import (
    exp_kernel,
    exp_sqrt_kernel,
    geglu_kernel,
    geglu_specialized_kernel,
    gelu_kernel,
    gelu_silu_kernel,
    silu_kernel,
    sqrt_abs_kernel,
)
from metile.runtime.metal_device import MetalDevice

BLOCK = 256
COOLDOWN = 1.0
SIZES = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]

COL_N = 5
COL_T = 12


def _n_str(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    return f"{n // 1024}K"


def _print_table(title, mlx_label, rows):
    """Print a 3-column table: N, metile (us), MLX (us)."""
    print(f"\n  {title}")
    hdr = f"    {'N':>{COL_N}}  {'metile (us)':>{COL_T}}  {mlx_label + ' (us)':>{COL_T}}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for n_str, dt_mtile, dt_mlx in rows:
        print(f"    {n_str:>{COL_N}}  {dt_mtile:>{COL_T}.1f}  {dt_mlx:>{COL_T}.1f}")


def _bench_1in_1out(kernel, mlx_fn_factory, mlx_label):
    """Benchmark a single-input single-output kernel vs MLX."""
    dev = MetalDevice.get()
    rows = []
    for N in SIZES:
        x_np = np.random.randn(N).astype(np.float32)
        x_buf = metile.Buffer(data=x_np)
        out = metile.Buffer(shape=(N,), dtype=np.float32)
        grid = (N + BLOCK - 1) // BLOCK

        kernel[grid](x_buf, out, N, BLOCK=BLOCK)
        dev.sync()

        d = kernel[grid].prepare(x_buf, out, N, BLOCK=BLOCK)

        x_mx = mx.array(x_np)
        mlx_fn = mlx_fn_factory(x_mx)

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(d, mlx_fn, dev.sync)
        rows.append((_n_str(N), dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table(kernel.name, mlx_label, rows)


def _bench_2in_1out(kernel, mlx_fn_factory, mlx_label):
    """Benchmark a two-input single-output kernel vs MLX."""
    dev = MetalDevice.get()
    rows = []
    for N in SIZES:
        a_np = np.random.randn(N).astype(np.float32)
        b_np = np.random.randn(N).astype(np.float32)
        a_buf = metile.Buffer(data=a_np)
        b_buf = metile.Buffer(data=b_np)
        out = metile.Buffer(shape=(N,), dtype=np.float32)
        grid = (N + BLOCK - 1) // BLOCK

        kernel[grid](a_buf, b_buf, out, N, BLOCK=BLOCK)
        dev.sync()

        d = kernel[grid].prepare(a_buf, b_buf, out, N, BLOCK=BLOCK)

        a_mx, b_mx = mx.array(a_np), mx.array(b_np)
        mlx_fn = mlx_fn_factory(a_mx, b_mx)

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(d, mlx_fn, dev.sync)
        rows.append((_n_str(N), dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table(kernel.name, mlx_label, rows)


def _bench_fused_2out(kernel, mlx_fn_factory, mlx_label):
    """Benchmark a fused dual-output kernel vs MLX (compiled)."""
    dev = MetalDevice.get()
    rows = []
    for N in SIZES:
        x_np = np.random.randn(N).astype(np.float32)
        x_buf = metile.Buffer(data=x_np)
        out_a = metile.Buffer(shape=(N,), dtype=np.float32)
        out_b = metile.Buffer(shape=(N,), dtype=np.float32)
        grid = (N + BLOCK - 1) // BLOCK

        kernel[grid](x_buf, out_a, out_b, N, BLOCK=BLOCK)
        dev.sync()

        d = kernel[grid].prepare(x_buf, out_a, out_b, N, BLOCK=BLOCK)

        x_mx = mx.array(x_np)
        mlx_fn = mlx_fn_factory(x_mx)

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(d, mlx_fn, dev.sync)
        rows.append((_n_str(N), dt_mtile * 1e6, dt_mlx * 1e6))

    _print_table(kernel.name, mlx_label, rows)


def main():
    print("=== Element-wise Kernels vs MLX ===\n")

    # --- Single-output kernels (native MLX ops) ---

    _bench_1in_1out(
        exp_kernel,
        lambda x: lambda x=x: mx.eval(mx.exp(x)),
        "MLX",
    )

    _bench_1in_1out(
        sqrt_abs_kernel,
        lambda x: lambda x=x: mx.eval(mx.sqrt(mx.abs(x))),
        "MLX",
    )

    _bench_1in_1out(
        gelu_kernel,
        lambda x: lambda x=x: mx.eval(mx.sigmoid(1.702 * x) * x),
        "MLX",
    )

    _bench_1in_1out(
        silu_kernel,
        lambda x: lambda x=x: mx.eval(x / (1 + mx.exp(-x))),
        "MLX",
    )

    _bench_2in_1out(
        geglu_kernel,
        lambda g, u: lambda g=g, u=u: mx.eval(mx.sigmoid(1.702 * g) * g * u),
        "MLX",
    )

    # --- Fused dual-output kernels (mx.compile for MLX fusion) ---

    def _mlx_exp_sqrt(x):
        @mx.compile
        def f(x):
            return mx.exp(x), mx.sqrt(mx.abs(x))

        f(x)
        mx.eval(f(x))
        return lambda x=x: mx.eval(*f(x))

    _bench_fused_2out(exp_sqrt_kernel, _mlx_exp_sqrt, "MLX (compiled)")

    def _mlx_gelu_silu(x):
        @mx.compile
        def f(x):
            return mx.sigmoid(1.702 * x) * x, x / (1 + mx.exp(-x))

        f(x)
        mx.eval(f(x))
        return lambda x=x: mx.eval(*f(x))

    _bench_fused_2out(gelu_silu_kernel, _mlx_gelu_silu, "MLX (compiled)")

    # --- Specialized kernels (mx.compile for MLX fusion) ---

    def _mlx_geglu_compiled(g, u):
        @mx.compile
        def f(g, u):
            return mx.sigmoid(1.702 * g) * g * u

        f(g, u)
        mx.eval(f(g, u))
        return lambda g=g, u=u: mx.eval(f(g, u))

    _bench_2in_1out(
        geglu_specialized_kernel,
        _mlx_geglu_compiled,
        "MLX (compiled)",
    )

    print()


if __name__ == "__main__":
    main()
