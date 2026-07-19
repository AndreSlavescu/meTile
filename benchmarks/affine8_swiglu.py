"""Benchmark matched affine-INT8 SwiGLU representations on a real MLX-LM shape."""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchutils import bench_interleaved

from benchmarks.mlx_lm_backend import (
    _git_revision,
    _hardware_metadata,
    _package_version,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _arguments()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    from metile.backends.mlx_compressed_down import MLXCompressedDownWeight
    from metile.backends.mlx_quantized import (
        mlx_affine_swiglu_dispatches,
        mlx_affine_swiglu_executor,
    )
    from metile.integrations.mlx_lm import _model_layers

    model, _ = load(arguments.model)
    layers = tuple(_model_layers(model))
    if not 0 <= arguments.layer_index < len(layers):
        raise ValueError("layer index is outside the model")
    mlp = layers[arguments.layer_index].mlp
    gate = MLXCompressedDownWeight.quantize(
        mlp.gate_proj.weight,
        group_size=arguments.group_size,
    )
    up = MLXCompressedDownWeight.quantize(
        mlp.up_proj.weight,
        group_size=arguments.group_size,
    )
    mx.random.seed(arguments.seed)
    values = mx.random.normal((1, gate.shape[1])).astype(mlp.gate_proj.weight.dtype)

    def native_mlx():
        gate_output = mx.quantized_matmul(
            values,
            gate.values,
            gate.scales,
            gate.biases,
            group_size=arguments.group_size,
            bits=8,
            mode="affine",
        )
        up_output = mx.quantized_matmul(
            values,
            up.values,
            up.scales,
            up.biases,
            group_size=arguments.group_size,
            bits=8,
            mode="affine",
        )
        return nn.silu(gate_output) * up_output

    executor = mlx_affine_swiglu_executor(
        values,
        gate.values,
        gate.scales,
        gate.biases,
        up.values,
        up.scales,
        up.biases,
        group_size=arguments.group_size,
        bits=8,
    )

    def metile_dispatch():
        return executor(values)

    reference = native_mlx()
    actual = metile_dispatch()
    difference = mx.abs(reference.astype(mx.float32) - actual.astype(mx.float32))
    mx.eval(reference, actual, difference)

    def timed_native():
        mx.eval(native_mlx())

    def timed_metile():
        mx.eval(metile_dispatch())

    metile_seconds, mlx_seconds = bench_interleaved(
        timed_metile,
        timed_native,
        sync=lambda: None,
    )
    dispatch = next(
        result
        for result in reversed(mlx_affine_swiglu_dispatches())
        if result["row_bucket"] == 1
        and result["input_features"] == gate.shape[1]
        and result["output_features"] == gate.shape[0]
        and result["group_size"] == arguments.group_size
        and result["bits"] == 8
    )
    speedup = mlx_seconds / metile_seconds
    print("=== Matched affine-INT8 SwiGLU decode ===")
    print(f"Model shape: 1x{gate.shape[1]} -> {gate.shape[0]} (layer {arguments.layer_index})")
    print(f"Native MLX:     {mlx_seconds * 1e6:.2f} us")
    print(f"meTile dispatch: {metile_seconds * 1e6:.2f} us ({speedup:.3f}x)")
    print(f"Selected:       {dispatch}")
    print(
        "Fidelity:       "
        f"mean={float(mx.mean(difference).item()):.8f}, "
        f"max={float(mx.max(difference).item()):.8f}"
    )

    if arguments.output_json is not None:
        payload = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "revision": _git_revision(),
            "scope": "isolated_primitive",
            "model": arguments.model,
            "layer_index": arguments.layer_index,
            "shape": {
                "rows": 1,
                "input_features": gate.shape[1],
                "output_features": gate.shape[0],
            },
            "hardware": _hardware_metadata(),
            "software": {
                "python": platform.python_version(),
                "mlx": _package_version("mlx"),
                "mlx_lm": _package_version("mlx-lm"),
            },
            "precision_comparison": {
                "class": "matched_precision_affine_int8",
                "same_weight_representation": True,
                "baseline_weights": "affine8",
                "optimized_weights": "affine8",
                "group_size": arguments.group_size,
                "source_dtype": str(mlp.gate_proj.weight.dtype),
            },
            "timings": {
                "mlx_seconds": mlx_seconds,
                "metile_seconds": metile_seconds,
                "speedup": speedup,
            },
            "fidelity": {
                "mean_absolute_error": float(mx.mean(difference).item()),
                "max_absolute_error": float(mx.max(difference).item()),
            },
            "dispatch": dispatch,
            "seed": arguments.seed,
        }
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {arguments.output_json}")


if __name__ == "__main__":
    main()
