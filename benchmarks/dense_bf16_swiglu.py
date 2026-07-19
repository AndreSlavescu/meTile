"""Benchmark exact BF16 SwiGLU dispatch on a real MLX-LM shape."""

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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _arguments()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    from metile.backends.mlx_dense import MLXDenseWeight
    from metile.backends.mlx_dense_swiglu import (
        mlx_dense_swiglu,
        mlx_dense_swiglu_dispatches,
    )
    from metile.integrations.mlx_lm import _model_layers

    model, _ = load(arguments.model)
    layers = tuple(_model_layers(model))
    if not 0 <= arguments.layer_index < len(layers):
        raise ValueError("layer index is outside the model")
    mlp = layers[arguments.layer_index].mlp
    gate = MLXDenseWeight.from_mlx(mlp.gate_proj.weight)
    up = MLXDenseWeight.from_mlx(mlp.up_proj.weight)
    paired = mx.stack((mlp.gate_proj.weight, mlp.up_proj.weight), axis=-1)
    mx.eval(paired)
    mx.random.seed(arguments.seed)
    values = mx.random.normal((1, gate.shape[0])).astype(mlp.gate_proj.weight.dtype)

    def native_mlx():
        gate_output = values @ gate.native_weight.T
        up_output = values @ up.native_weight.T
        return nn.silu(gate_output) * up_output

    def metile_dispatch():
        return mlx_dense_swiglu(values, gate, up, paired_weight=paired)

    reference = native_mlx()
    actual = metile_dispatch()
    mx.eval(reference, actual)
    exact = bool(mx.array_equal(reference, actual).item())
    if not exact:
        raise RuntimeError("generated BF16 SwiGLU did not match native MLX exactly")

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
        for result in reversed(mlx_dense_swiglu_dispatches())
        if result["rows"] == 1
        and result["input_features"] == gate.shape[0]
        and result["output_features"] == gate.shape[1]
        and result["paired_available"]
    )
    speedup = mlx_seconds / metile_seconds
    print("=== Exact BF16 SwiGLU decode ===")
    print(f"Model shape: 1x{gate.shape[0]} -> {gate.shape[1]} (layer {arguments.layer_index})")
    print(f"Native MLX:     {mlx_seconds * 1e6:.2f} us")
    print(f"meTile dispatch: {metile_seconds * 1e6:.2f} us ({speedup:.3f}x)")
    print(f"Selected:       {dispatch}")
    print("Fidelity:       bitwise exact")

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
                "input_features": gate.shape[0],
                "output_features": gate.shape[1],
            },
            "hardware": _hardware_metadata(),
            "software": {
                "python": platform.python_version(),
                "mlx": _package_version("mlx"),
                "mlx_lm": _package_version("mlx-lm"),
            },
            "precision_comparison": {
                "class": "same_precision",
                "same_weight_representation": True,
                "baseline_weights": "bfloat16",
                "optimized_weights": "bfloat16",
            },
            "timings": {
                "mlx_seconds": mlx_seconds,
                "metile_seconds": metile_seconds,
                "speedup": speedup,
            },
            "fidelity": {"bitwise_exact": exact},
            "dispatch": dispatch,
            "seed": arguments.seed,
        }
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {arguments.output_json}")


if __name__ == "__main__":
    main()
