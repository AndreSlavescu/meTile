"""Benchmark meTile against MLX at MATCHED weight representation, across batch size.

Every comparison runs the *same weights in the same format* on both sides, so a
speedup here is a kernel win rather than a representation win. This is deliberately
different from the bf16 capacity suite, which compares meTile's INT8 decode
projections against MLX running bf16.

Candidates alternate order every round so thermal drift hits both sides equally, and
the reported figure is the median of per-round paired ratios plus the win rate.
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

from benchmarks.mlx_lm_backend import (
    _git_revision,
    _hardware_metadata,
    _package_version,
)

_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=8960)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-label",
        default="Qwen2.5-1.5B MLP",
        help="Only used to label the recorded shape.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _paired(mx, build_mlx, build_metile, rounds, inner):
    """Interleaved A/B timing. Returns (mlx_seconds, metile_seconds, ratio, win_rate)."""
    for _ in range(3):
        mx.eval([build_mlx() for _ in range(inner)])
        mx.eval([build_metile() for _ in range(inner)])
    mx.synchronize()

    def sample(build):
        start = time.perf_counter_ns()
        mx.eval([build() for _ in range(inner)])
        return (time.perf_counter_ns() - start) / inner / 1e9

    native, generated = [], []
    for index in range(rounds):
        if index % 2 == 0:
            first, second = sample(build_mlx), sample(build_metile)
        else:
            second, first = sample(build_metile), sample(build_mlx)
        native.append(first)
        generated.append(second)
    ratios = sorted(a / b for a, b in zip(native, generated))
    wins = sum(1 for ratio in ratios if ratio > 1.0) / len(ratios)
    return (
        statistics.median(native),
        statistics.median(generated),
        statistics.median(ratios),
        wins,
    )


def main():
    arguments = _arguments()

    import mlx.core as mx
    import mlx.nn as nn

    from metile.backends.mlx_dense import MLXDenseWeight
    from metile.backends.mlx_dense_residual import mlx_dense_residual_qmv
    from metile.backends.mlx_dense_swiglu import mlx_dense_swiglu
    from metile.backends.mlx_quantized import (
        mlx_affine_mlp_executor,
        mlx_affine_swiglu_executor,
    )

    hidden, intermediate = arguments.hidden, arguments.intermediate
    group_size = arguments.group_size
    mx.random.seed(arguments.seed)

    gate = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    up = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    down = mx.random.normal((hidden, intermediate)).astype(mx.bfloat16)
    mx.eval(gate, up, down)

    records = []
    print(f"MLP block {hidden}->{intermediate}->{hidden}, {arguments.rounds} interleaved rounds")
    print(f"{'format':<11}{'rows':>6}{'MLX us':>11}{'meTile us':>11}{'speedup':>9}{'win%':>7}")

    # ---------------------------------------------------------------- bf16
    paired_weight = mx.stack((gate, up), axis=-1)
    gate_dense = MLXDenseWeight.from_mlx(gate)
    up_dense = MLXDenseWeight.from_mlx(up)
    mx.eval(paired_weight, gate_dense.native_weight, up_dense.native_weight)

    for rows in _ROWS:
        inner = max(1, min(4, 4096 // rows))
        values = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
        residual = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
        mx.eval(values, residual)
        native, generated, ratio, wins = _paired(
            mx,
            lambda: (nn.silu(values @ gate.T) * (values @ up.T)) @ down.T + residual,
            lambda: mlx_dense_residual_qmv(
                mlx_dense_swiglu(values, gate_dense, up_dense, paired_weight=paired_weight),
                down,
                residual,
            ),
            arguments.rounds,
            inner,
        )
        records.append(
            {
                "format": "bf16",
                "rows": rows,
                "mlx_seconds": native,
                "metile_seconds": generated,
                "speedup": ratio,
                "win_rate": wins,
                "metile_tokens_per_second": rows / generated,
                "mlx_tokens_per_second": rows / native,
            }
        )
        print(
            f"{'bf16':<11}{rows:>6}{native * 1e6:11.1f}{generated * 1e6:11.1f}"
            f"{ratio:9.3f}{wins * 100:7.0f}"
        )

    # -------------------------------------------------------- affine int4/int8
    dtype = mx.float16
    for bits in (4, 8):
        gate_q, gate_s, gate_b = mx.quantize(
            gate.astype(dtype), group_size=group_size, bits=bits, mode="affine"
        )
        up_q, up_s, up_b = mx.quantize(
            up.astype(dtype), group_size=group_size, bits=bits, mode="affine"
        )
        down_q, down_s, down_b = mx.quantize(
            down.astype(dtype), group_size=group_size, bits=bits, mode="affine"
        )
        mx.eval(gate_q, gate_s, gate_b, up_q, up_s, up_b, down_q, down_s, down_b)

        def quantized_matmul(activations, weight, scales, biases, _bits=bits):
            return mx.quantized_matmul(
                activations,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=group_size,
                bits=_bits,
                mode="affine",
            )

        print()
        for rows in _ROWS:
            inner = max(1, min(4, 4096 // rows))
            values = mx.random.normal((rows, hidden)).astype(dtype)
            residual = mx.random.normal((rows, hidden)).astype(dtype)
            mx.eval(values, residual)

            if bits == 4:
                execute = mlx_affine_mlp_executor(
                    values,
                    gate_q,
                    gate_s,
                    gate_b,
                    up_q,
                    up_s,
                    up_b,
                    down_q,
                    down_s,
                    down_b,
                    residual,
                    group_size=group_size,
                    bits=bits,
                )
                note = ""
            else:
                swiglu = mlx_affine_swiglu_executor(
                    values,
                    gate_q,
                    gate_s,
                    gate_b,
                    up_q,
                    up_s,
                    up_b,
                    group_size=group_size,
                    bits=bits,
                )

                def execute(activations, residual_values, _swiglu=swiglu):
                    projected = quantized_matmul(_swiglu(activations), down_q, down_s, down_b)
                    return projected + residual_values

                note = "down projection stays native: no meTile int8 path"

            native, generated, ratio, wins = _paired(
                mx,
                lambda: (
                    quantized_matmul(
                        nn.silu(quantized_matmul(values, gate_q, gate_s, gate_b))
                        * quantized_matmul(values, up_q, up_s, up_b),
                        down_q,
                        down_s,
                        down_b,
                    )
                    + residual
                ),
                lambda: execute(values, residual),
                arguments.rounds,
                inner,
            )
            label = f"int{bits}"
            records.append(
                {
                    "format": label,
                    "rows": rows,
                    "mlx_seconds": native,
                    "metile_seconds": generated,
                    "speedup": ratio,
                    "win_rate": wins,
                    "metile_tokens_per_second": rows / generated,
                    "mlx_tokens_per_second": rows / native,
                    "note": note,
                }
            )
            print(
                f"{label + ' g' + str(group_size):<11}{rows:>6}{native * 1e6:11.1f}"
                f"{generated * 1e6:11.1f}{ratio:9.3f}{wins * 100:7.0f}"
            )

    if arguments.output_json is not None:
        payload = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "revision": _git_revision(),
            "scope": "matched_representation_matrix",
            "precision_comparison": {
                "class": "same_representation",
                "same_weight_representation": True,
                "note": (
                    "Both sides run identical weights in identical formats, so every "
                    "speedup is a kernel comparison rather than a representation change."
                ),
            },
            "shape": {
                "label": arguments.model_label,
                "hidden": hidden,
                "intermediate": intermediate,
                "group_size": group_size,
            },
            "rounds": arguments.rounds,
            "seed": arguments.seed,
            "hardware": _hardware_metadata(),
            "software": {
                "python": platform.python_version(),
                "mlx": _package_version("mlx"),
                "mlx_lm": _package_version("mlx-lm"),
            },
            "measurements": records,
        }
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {arguments.output_json}")


if __name__ == "__main__":
    main()
