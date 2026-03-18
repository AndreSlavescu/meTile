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
from kernels.mlp import matmul_gelu, matmul_silu
from metile.runtime.metal_device import MetalDevice

# --- Autotune configs ---

GEMM_CONFIGS = [
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, WM=2, WN=2, K_UNROLL=1),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, WM=2, WN=2, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, WM=2, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, WM=2, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, WM=4, WN=4, K_UNROLL=1),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, WM=4, WN=4, K_UNROLL=1),
]

autotuned_gelu = metile.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K"], verbose=True)(
    matmul_gelu
)
autotuned_silu = metile.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K"], verbose=True)(
    matmul_silu
)
autotuned_matmul = metile.autotune(configs=GEMM_CONFIGS, key=["M", "N", "K"], verbose=True)(matmul)

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


def _gelu_ref(x):
    """GELU (sigmoid approx) matching meTile: x / (1 + exp(-1.702 * x))."""
    return x / (1.0 + mx.exp(-1.702 * x))


def _silu_ref(x):
    """SiLU: x / (1 + exp(-x))."""
    return x / (1.0 + mx.exp(-x))


def main():
    # Fused GEMM+activation sizes (batch*seq, hidden, intermediate)
    fused_sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # LLM-typical shapes
        (32, 4096, 4096),
        (128, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
    ]

    # Full MLP sizes: (batch*seq, model_dim, ffn_dim)
    mlp_sizes = [
        (128, 1024, 4096),
        (256, 2048, 8192),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (32, 4096, 4096),
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "--silu":
        act = "silu"
    else:
        act = "gelu"

    autotuned_act = autotuned_gelu if act == "gelu" else autotuned_silu
    act_fn = _gelu_ref if act == "gelu" else _silu_ref

    dev = MetalDevice.get()

    # --- Fused GEMM+activation ---

    print(f"=== Fused GEMM+{act.upper()} (autotuned) ===\n")

    rows = []
    for M, N, K in fused_sizes:
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        A_buf = metile.Buffer(data=A_np.ravel())
        B_buf = metile.Buffer(data=B_np.ravel())
        C_buf = metile.Buffer.zeros((M * N,))

        def grid_fn(cfg, M=M, N=N):
            return (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(N, cfg["BLOCK_N"]))

        dispatch = autotuned_act[grid_fn].prepare(A_buf, B_buf, C_buf, M, N, K)
        dev.sync()

        A_mx, B_mx = mx.array(A_np), mx.array(B_np)

        @mx.compile
        def mlx_fused(a, b):
            return act_fn(a @ b)

        mx.eval(mlx_fused(A_mx, B_mx))

        def mlx_fn(a=A_mx, b=B_mx):
            mx.eval(mlx_fused(a, b))

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(dispatch, mlx_fn, dev.sync)
        rows.append((f"{M}x{N}x{K}", dt_mtile * 1000, dt_mlx * 1000))

    _print_table(f"matmul_{act} (metile fused vs MLX compile)", rows)
    print()

    # --- Full MLP pipeline ---

    print(f"=== Full MLP: {act.upper()}(x @ W1) @ W2 (autotuned) ===\n")

    rows = []
    for M, D, H in mlp_sizes:
        X_np = np.random.randn(M, D).astype(np.float32)
        W1_np = np.random.randn(D, H).astype(np.float32)
        W2_np = np.random.randn(H, D).astype(np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W1_buf = metile.Buffer(data=W1_np.ravel())
        W2_buf = metile.Buffer(data=W2_np.ravel())
        H_buf = metile.Buffer.zeros((M * H,))
        Y_buf = metile.Buffer.zeros((M * D,))

        def grid_up(cfg, M=M, H=H):
            return (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(H, cfg["BLOCK_N"]))

        dispatch_up = autotuned_act[grid_up].prepare(X_buf, W1_buf, H_buf, M, H, D)
        dev.sync()

        def grid_down(cfg, M=M, D=D):
            return (metile.cdiv(M, cfg["BLOCK_M"]), metile.cdiv(D, cfg["BLOCK_N"]))

        dispatch_down = autotuned_matmul[grid_down].prepare(H_buf, W2_buf, Y_buf, M, D, H)
        dev.sync()

        def mtile_mlp(up=dispatch_up, down=dispatch_down):
            up()
            down()

        X_mx = mx.array(X_np)
        W1_mx = mx.array(W1_np)
        W2_mx = mx.array(W2_np)

        @mx.compile
        def mlx_mlp(x, w1, w2):
            return act_fn(x @ w1) @ w2

        mx.eval(mlx_mlp(X_mx, W1_mx, W2_mx))

        def mlx_fn(x=X_mx, w1=W1_mx, w2=W2_mx):
            mx.eval(mlx_mlp(x, w1, w2))

        time.sleep(COOLDOWN)
        dt_mtile, dt_mlx = bench_interleaved(mtile_mlp, mlx_fn, dev.sync)
        rows.append((f"{M}x{D}x{H}", dt_mtile * 1000, dt_mlx * 1000))

    _print_table(f"MLP {act} (metile fused vs MLX compile)", rows)
    print()


if __name__ == "__main__":
    main()
