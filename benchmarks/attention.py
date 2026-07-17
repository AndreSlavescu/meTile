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
from kernels.attention import attention_decode
from metile.runtime.metal_device import MetalDevice

COOLDOWN = 3.0


def main():
    print("=== Decode attention (autotuned) ===\n")
    device = MetalDevice.get()
    random = np.random.default_rng(0)
    dimension = 128
    scale = float(dimension**-0.5)

    print(f"    {'shape':>18}  {'meTile (us)':>12}  {'MLX (us)':>12}  {'speedup':>9}")
    print("    " + "-" * 57)
    for heads, tokens in ((32, 128), (32, 512), (32, 2048), (16, 8192)):
        query = random.standard_normal((heads, dimension), dtype=np.float32)
        key = random.standard_normal((heads, tokens, dimension), dtype=np.float32)
        value = random.standard_normal((heads, tokens, dimension), dtype=np.float32)

        query_buffer = metile.Buffer(data=query.ravel())
        key_buffer = metile.Buffer(data=key.ravel())
        value_buffer = metile.Buffer(data=value.ravel())
        output_buffer = metile.Buffer.zeros((heads * dimension,))
        dispatch = attention_decode[(heads,)].prepare(
            query_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            tokens,
            scale,
            D=dimension,
        )

        query_mlx = mx.array(query.reshape(1, heads, 1, dimension))
        key_mlx = mx.array(key.reshape(1, heads, tokens, dimension))
        value_mlx = mx.array(value.reshape(1, heads, tokens, dimension))

        def mlx_attention(q=query_mlx, k=key_mlx, v=value_mlx):
            mx.eval(
                mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                )
            )

        time.sleep(COOLDOWN)
        metile_time, mlx_time = bench_interleaved(dispatch, mlx_attention, device.sync)
        speedup = mlx_time / metile_time
        shape = f"{heads}x1x{tokens}x{dimension}"
        print(
            f"    {shape:>18}  {metile_time * 1e6:>12.1f}  "
            f"{mlx_time * 1e6:>12.1f}  {speedup:>8.2f}x"
        )


if __name__ == "__main__":
    main()
