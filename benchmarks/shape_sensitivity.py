"""Measure where MLX leaves performance on the table, and whether meTile picks it up.

Two sweeps, both against native MLX on identical weights in identical formats:

  width   int4 prefill matmul as the output width varies. MLX changes kernel somewhere
          between 2048 and 2560 and the one it uses below that is poor, which is the
          entire reason the 4-bit prefill win depends on which model you run.

  batch   the decode MLP block as rows per dispatch grows. Weights are read once no
          matter how many rows there are, so achieved bandwidth should hold flat. Where
          it collapses, a kernel is re-reading weights per row tile.

Written as one JSON so the charts and the README quote the same numbers.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.mlx_lm_backend import (  # noqa: E402
    _git_revision,
    _hardware_metadata,
    _package_version,
)

_WIDTHS = (1024, 1536, 2048, 2560, 3072, 4096, 5120, 8192)
_ROWS = (1, 2, 4, 8, 16, 32)
# Which model each width is the down projection of, so the chart can name them.
_MODELS = {
    896: "Qwen2.5 0.5B",
    1536: "Qwen2.5 1.5B",
    2048: "Llama 3.2 1B",
    3072: "Llama 3.2 3B",
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=int, default=8192)
    parser.add_argument("--prefill-rows", type=int, default=127)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=8960)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _median(build, mx, inner, rounds, warmup=4):
    for _ in range(warmup):
        mx.eval([build() for _ in range(inner)])
    mx.synchronize()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        mx.eval([build() for _ in range(inner)])
        samples.append((time.perf_counter_ns() - started) / inner / 1e9)
    return statistics.median(samples)


def _paired(native, generated, mx, inner, rounds):
    for _ in range(3):
        mx.eval([native() for _ in range(inner)])
        mx.eval([generated() for _ in range(inner)])
    mx.synchronize()

    def sample(build):
        started = time.perf_counter_ns()
        mx.eval([build() for _ in range(inner)])
        return (time.perf_counter_ns() - started) / inner / 1e9

    left, right = [], []
    for index in range(rounds):
        if index % 2 == 0:
            first, second = sample(native), sample(generated)
        else:
            second, first = sample(generated), sample(native)
        left.append(first)
        right.append(second)
    return statistics.median(left), statistics.median(right)


def _width_sweep(arguments, mx):
    from metile.backends.mlx_affine import (
        MLXAffineWeight,
        _native_affine_matmul,
        mlx_affine_matmul,
    )

    reduction, rows = arguments.reduction, arguments.prefill_rows
    records = []
    print(f"int4 prefill matmul, K={reduction}, rows={rows}")
    print(f"{'N':>6}{'MLX us':>10}{'meTile us':>11}{'speedup':>9}   model")
    for width in _WIDTHS:
        mx.random.seed(0)
        dense = mx.random.normal((width, reduction)).astype(mx.float16)
        packed, scales, biases = mx.quantize(dense, group_size=64, bits=4, mode="affine")
        weight = MLXAffineWeight.from_mlx(packed, scales, biases, group_size=64, bits=4)
        values = mx.random.normal((rows, reduction)).astype(mx.float16)
        mx.eval(packed, scales, biases, weight.packed, weight.scales, weight.biases, values)
        mx.eval(mlx_affine_matmul(values, weight))

        native, generated = _paired(
            lambda: _native_affine_matmul(values, weight),
            lambda: mlx_affine_matmul(values, weight),
            mx,
            4,
            arguments.rounds,
        )
        model = _MODELS.get(width, "")
        records.append(
            {
                "output_features": width,
                "mlx_seconds": native,
                "metile_seconds": generated,
                "speedup": native / generated,
                "model": model,
            }
        )
        print(
            f"{width:>6}{native * 1e6:10.1f}{generated * 1e6:11.1f}"
            f"{native / generated:8.2f}x   {model}"
        )
    return records


def _batch_sweep(arguments, mx, nn):
    from metile.backends.mlx_dense import MLXDenseWeight
    from metile.backends.mlx_dense_residual import mlx_dense_residual_qmv
    from metile.backends.mlx_dense_swiglu import mlx_dense_swiglu

    hidden, intermediate = arguments.hidden, arguments.intermediate
    mx.random.seed(0)
    gate = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    up = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    down = mx.random.normal((hidden, intermediate)).astype(mx.bfloat16)
    paired = mx.stack((gate, up), axis=-1)
    gate_dense, up_dense = MLXDenseWeight.from_mlx(gate), MLXDenseWeight.from_mlx(up)
    mx.eval(gate, up, down, paired, gate_dense.native_weight, up_dense.native_weight)
    dense_bytes = (2 * intermediate * hidden + hidden * intermediate) * 2

    quantized = {}
    for bits in (4, 8):
        quantized[bits] = tuple(
            mx.quantize(tensor.astype(mx.float16), group_size=64, bits=bits, mode="affine")
            for tensor in (gate, up, down)
        )
        mx.eval(quantized[bits])

    records = []
    print(f"\ndecode MLP block {hidden} -> {intermediate} -> {hidden}")
    print(f"{'format':<9}{'rows':>5}{'MLX us':>10}{'MLX GB/s':>10}{'meTile GB/s':>13}")
    for rows in _ROWS:
        values = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
        residual = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
        mx.eval(values, residual)
        inner = max(1, min(8, 64 // rows))

        native = _median(
            lambda: (nn.silu(values @ gate.T) * (values @ up.T)) @ down.T + residual,
            mx,
            inner,
            arguments.rounds,
        )
        generated = _median(
            lambda: mlx_dense_residual_qmv(
                mlx_dense_swiglu(values, gate_dense, up_dense, paired_weight=paired),
                down,
                residual,
            ),
            mx,
            inner,
            arguments.rounds,
        )
        records.append(
            {
                "format": "bf16",
                "rows": rows,
                "weight_bytes": dense_bytes,
                "mlx_seconds": native,
                "metile_seconds": generated,
                "mlx_bandwidth": dense_bytes / native / 1e9,
                "metile_bandwidth": dense_bytes / generated / 1e9,
            }
        )
        print(
            f"{'bf16':<9}{rows:>5}{native * 1e6:10.1f}"
            f"{dense_bytes / native / 1e9:10.1f}{dense_bytes / generated / 1e9:13.1f}"
        )

        for bits in (4, 8):
            (gq, gs, gb), (uq, us, ub), (dq, ds, db) = quantized[bits]
            activations = values.astype(mx.float16)
            residual16 = residual.astype(mx.float16)
            mx.eval(activations, residual16)

            def quantized_matmul(tensor, weight, scales, biases, _bits=bits):
                return mx.quantized_matmul(
                    tensor, weight, scales=scales, biases=biases, transpose=True,
                    group_size=64, bits=_bits, mode="affine",
                )

            quantized_bytes = dense_bytes * bits // 16
            native = _median(
                lambda: quantized_matmul(
                    nn.silu(quantized_matmul(activations, gq, gs, gb))
                    * quantized_matmul(activations, uq, us, ub),
                    dq, ds, db,
                )
                + residual16,
                mx,
                inner,
                arguments.rounds,
            )
            # meTile has no multi-row quantized kernel, so it defers to MLX above one row.
            records.append(
                {
                    "format": f"int{bits}",
                    "rows": rows,
                    "weight_bytes": quantized_bytes,
                    "mlx_seconds": native,
                    "metile_seconds": None,
                    "mlx_bandwidth": quantized_bytes / native / 1e9,
                    "metile_bandwidth": None,
                }
            )
            print(
                f"{'int' + str(bits):<9}{rows:>5}{native * 1e6:10.1f}"
                f"{quantized_bytes / native / 1e9:10.1f}{'-':>13}"
            )
    return records


def main():
    arguments = _arguments()
    import mlx.core as mx
    import mlx.nn as nn

    width = _width_sweep(arguments, mx)
    batch = _batch_sweep(arguments, mx, nn)

    if arguments.output_json is not None:
        payload = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "revision": _git_revision(),
            "scope": "shape_sensitivity",
            "precision_comparison": {
                "class": "same_representation",
                "same_weight_representation": True,
            },
            "reduction": arguments.reduction,
            "prefill_rows": arguments.prefill_rows,
            "shape": {"hidden": arguments.hidden, "intermediate": arguments.intermediate},
            "rounds": arguments.rounds,
            "hardware": _hardware_metadata(),
            "software": {
                "python": platform.python_version(),
                "mlx": _package_version("mlx"),
                "mlx_lm": _package_version("mlx-lm"),
            },
            "width_sweep": width,
            "batch_sweep": batch,
        }
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {arguments.output_json}")


if __name__ == "__main__":
    main()
