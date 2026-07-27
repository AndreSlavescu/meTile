"""Benchmark what algorithmic discovery and graph fusion actually buy, in multiplier units.

Two things the discovery pipeline produces are measured against native MLX:

  flash attention   `softmax(scale(Q K^T)) V` discovered as one proof-carrying op and
                    executed through the fused graph, versus mx.fast.scaled_dot_product_attention
  add + RMSNorm     the residual-add/RMSNorm region the min-cut selector picks, versus
                    MLX evaluating the two ops separately

Both sides run identical inputs. Candidates alternate order every round so thermal drift
hits them equally, and each timed sample queues several dispatches into one eval, because
a blocking mx.eval round trip costs ~200 us regardless of kernel size and would otherwise
compress the ratio toward 1.0.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from metile.backends.mlx import mlx_add_rms_norm  # noqa: E402
from metile.backends.mlx_graph import compile_mlx_graph  # noqa: E402
from metile.ir.graph_ir import GraphBuilder, TensorSpec  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79)
    return parser.parse_args()


def paired(build_native, build_metile, rounds, inner):
    """Interleaved A/B timing. Returns (native_s, metile_s, ratio, win_rate)."""
    for _ in range(3):
        mx.eval([build_native() for _ in range(inner)])
        mx.eval([build_metile() for _ in range(inner)])
    mx.synchronize()

    def sample(build):
        start = time.perf_counter_ns()
        mx.eval([build() for _ in range(inner)])
        return (time.perf_counter_ns() - start) / inner / 1e9

    native, generated = [], []
    for index in range(rounds):
        if index % 2 == 0:
            first, second = sample(build_native), sample(build_metile)
        else:
            second, first = sample(build_metile), sample(build_native)
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


def _attention_graph(batch, heads, queries, keys, dim, causal, dtype):
    builder = GraphBuilder()
    spec = TensorSpec
    query = builder.input("query", spec((batch, heads, queries, dim), dtype))
    key = builder.input("key", spec((batch, heads, keys, dim), dtype))
    value = builder.input("value", spec((batch, heads, keys, dim), dtype))
    scores = builder.matmul(query, key, transpose_right=True)
    scaled = builder.scale(scores, dim**-0.5)
    if causal:
        scaled = builder.causal_mask(scaled)
    probabilities = builder.softmax(scaled, axis=-1)
    output = builder.matmul(probabilities, value)
    return builder.build([output])


def main():
    arguments = _arguments()
    mx.random.seed(arguments.seed)
    random = np.random.default_rng(arguments.seed)

    print("Discovery and graph fusion vs native MLX")
    print(f"{arguments.rounds} interleaved rounds x {arguments.inner} dispatches per sample\n")
    print(f"{'rewrite':<34}{'MLX us':>10}{'meTile us':>11}{'speedup':>9}{'win%':>7}")

    # ------------------------------------------------------------ flash attention
    for batch, heads, queries, keys, dim, causal in (
        (1, 8, 1, 256, 64, False),
        (1, 8, 1, 1024, 64, False),
        (1, 8, 128, 128, 64, True),
        (1, 8, 512, 512, 64, True),
    ):
        graph = _attention_graph(batch, heads, queries, keys, dim, causal, "f32")
        executable = compile_mlx_graph(graph)
        discovered = [node.op for node in executable.plan.graph.nodes]
        query = mx.array(random.normal(size=(batch, heads, queries, dim)).astype(np.float32))
        key = mx.array(random.normal(size=(batch, heads, keys, dim)).astype(np.float32))
        value = mx.array(random.normal(size=(batch, heads, keys, dim)).astype(np.float32))
        mx.eval(query, key, value)

        mask = "causal" if causal else None
        native, generated, ratio, wins = paired(
            lambda: mx.fast.scaled_dot_product_attention(
                query, key, value, scale=dim**-0.5, mask=mask
            ),
            lambda: executable(query, key, value),
            arguments.rounds,
            arguments.inner,
        )
        label = f"attention q={queries} kv={keys}{' causal' if causal else ''}"
        note = "" if discovered == ["flash_attention"] else f"  [not discovered: {discovered}]"
        print(
            f"{label:<34}{native * 1e6:10.1f}{generated * 1e6:11.1f}"
            f"{ratio:9.3f}{wins * 100:7.0f}{note}"
        )

    # ------------------------------------------------------------ add + RMSNorm
    print()
    for rows, hidden, dtype in (
        (1, 1536, mx.bfloat16),
        (1, 4096, mx.bfloat16),
        (128, 1536, mx.bfloat16),
        (512, 4096, mx.bfloat16),
    ):
        values = mx.random.normal((rows, hidden)).astype(dtype)
        residual = mx.random.normal((rows, hidden)).astype(dtype)
        weight = mx.random.normal((hidden,)).astype(dtype)
        mx.eval(values, residual, weight)

        def native_pair():
            summed = values + residual
            return (summed, mx.fast.rms_norm(summed, weight, 1e-6))

        native, generated, ratio, wins = paired(
            lambda: native_pair()[1],
            lambda: mlx_add_rms_norm(values, residual, weight, 1e-6)[1],
            arguments.rounds,
            arguments.inner,
        )
        print(
            f"{'add+rmsnorm rows=' + str(rows) + ' h=' + str(hidden):<34}"
            f"{native * 1e6:10.1f}{generated * 1e6:11.1f}{ratio:9.3f}{wins * 100:7.0f}"
        )


if __name__ == "__main__":
    main()
