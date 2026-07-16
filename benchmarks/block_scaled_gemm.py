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
from metile.runtime.metal_device import MetalDevice


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    formats = [sys.argv[2]] if len(sys.argv) > 2 else ["mxfp4", "mxfp8"]
    rng = np.random.default_rng(17)
    activations = rng.normal(size=(size, size)).astype(np.float32)
    weight = rng.normal(size=(size, size)).astype(np.float32)
    device = MetalDevice.get()

    print(f"=== Block-scaled GEMM ({size}x{size}x{size}) ===")
    for format in formats:
        activations_buffer = metile.Buffer(data=activations)
        quantized = metile.BlockScaledWeight.quantize(weight, format=format)
        output = metile.Buffer.empty((size, size))
        dispatch = metile.prepare_block_scaled_matmul(activations_buffer, quantized, output)
        device.sync()

        mlx_activations = mx.array(activations)
        mlx_values, mlx_scales = mx.quantize(mx.array(weight.T), mode=format)

        def mlx_matmul(
            mlx_activations=mlx_activations,
            mlx_values=mlx_values,
            mlx_scales=mlx_scales,
            format=format,
        ):
            mx.eval(
                mx.quantized_matmul(
                    mlx_activations,
                    mlx_values,
                    mlx_scales,
                    mode=format,
                )
            )

        mlx_matmul()
        time.sleep(1.0)
        metile_time, mlx_time = bench_interleaved(dispatch, mlx_matmul, device.sync)
        print(
            f"{format:>6}: meTile {metile_time * 1e3:7.3f} ms | "
            f"MLX {mlx_time * 1e3:7.3f} ms | {mlx_time / metile_time:5.2f}x"
        )


if __name__ == "__main__":
    main()
