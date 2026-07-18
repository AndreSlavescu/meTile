import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import numpy as np
from benchutils import bench_interleaved

from metile.backends.mlx_block_scaled import (
    MLXBlockScaledWeight,
    mlx_block_scaled_dispatches,
    mlx_block_scaled_matmul,
)


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    formats = [sys.argv[2]] if len(sys.argv) > 2 else ["mxfp4", "mxfp8"]
    rng = np.random.default_rng(17)
    activations = rng.normal(size=(size, size)).astype(np.float32)
    weight = rng.normal(size=(size, size)).astype(np.float32)
    print(f"=== In-graph block-scaled GEMM ({size}x{size}x{size}) ===")
    for format in formats:
        mlx_activations = mx.array(activations)
        quantized = MLXBlockScaledWeight.quantize(weight, format=format)

        def metile_matmul(
            mlx_activations=mlx_activations,
            quantized=quantized,
        ):
            mx.eval(mlx_block_scaled_matmul(mlx_activations, quantized))

        def mlx_matmul(
            mlx_activations=mlx_activations,
            quantized=quantized,
            format=format,
        ):
            mx.eval(
                mx.quantized_matmul(
                    mlx_activations,
                    quantized.native_values,
                    quantized.native_scales,
                    mode=format,
                )
            )

        metile_matmul()
        mlx_matmul()
        time.sleep(1.0)
        metile_time, mlx_time = bench_interleaved(
            metile_matmul,
            mlx_matmul,
            sync=lambda: None,
        )
        selected = next(
            dispatch
            for dispatch in reversed(mlx_block_scaled_dispatches())
            if dispatch["rows"] == size
            and dispatch["reduction"] == size
            and dispatch["output_features"] == size
            and dispatch["format"] == format
        )
        schedule = (
            "native MLX"
            if selected["algorithm"] == "mlx"
            else (
                f"{selected['block_m']}x{selected['block_n']} "
                f"{selected['schedule']} {selected['fragment_type']}"
            )
        )
        print(
            f"{format:>6}: meTile {metile_time * 1e3:7.3f} ms | "
            f"MLX {mlx_time * 1e3:7.3f} ms | {mlx_time / metile_time:5.2f}x | "
            f"{schedule}"
        )


if __name__ == "__main__":
    main()
