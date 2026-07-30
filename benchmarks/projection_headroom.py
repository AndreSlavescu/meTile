"""Whether attention projections have anything to give, at matched bf16.

The q, k, v and o projections are about a quarter of a decode token and have no meTile path, which
makes them look like the obvious next thing to serve. They are not, and this measures why rather than
arguing it: at one row MLX already runs them at or above the streaming ceiling, so there is nothing
for a kernel to win.

    shape                    rows   MLX GB/s   gap   meTile   bit-exact
    Qwen2.5-1.5B q/o            1        148   0.8x   0.967x   yes
    Qwen2.5-1.5B k/v            1        242   0.5x   0.780x   yes
    Qwen3-8B q/o                1        120   1.0x   0.990x   yes
    Qwen3-8B k/v                1        148   0.8x   0.991x   yes

A gap below 1.0x means the weights are cache-resident and the DRAM ceiling is not even the limit. At
eight rows it is still parity. Only at thirty-two rows does a win appear, 1.30x to 1.57x, and that is
batch or prefill rather than decode; two of the four shapes stop being bit-exact there, so routing it
would trade the logit-equality contract for a win outside the case the task was about.

Amortisation is the whole methodology here and it is easy to get wrong in the direction of good news.
An earlier version of this measurement used enough inner dispatches for 64 MB of weight traffic per
eval. For a 4.7 MB projection that is thirteen dispatches, about 390 us of work against a roughly
200 us `mx.eval` round trip, so nearly a third of every sample was overhead -- and because the two
sides pay it differently it reported MLX at 52 GB/s and meTile at 1.828x. Both were fiction. Targeting
a gigabyte per eval puts the round trip near 2% and the same shape reads 148 GB/s at 0.967x.

If a measurement of a small kernel shows a large win, suspect the harness first.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from metile.target import agx

# Real projection shapes: (label, hidden, output features). Grouped-query attention makes k and v much
# narrower than q and o, and the two behave differently, so both are measured.
SHAPES = (
    ("Qwen2.5-1.5B q/o", 1536, 1536),
    ("Qwen2.5-1.5B k/v", 1536, 256),
    ("Qwen3-8B q/o", 4096, 4096),
    ("Qwen3-8B k/v", 4096, 1024),
)
ROWS = (1, 8, 32)

# Weight traffic per eval. The round trip is fixed, so this decides the error floor: a gigabyte is
# about 8ms at the streaming ceiling, putting a 200us round trip at 2.5%.
TRAFFIC_TARGET = 1e9


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--traffic",
        type=float,
        default=TRAFFIC_TARGET,
        help="weight bytes per eval; lowering this reintroduces round-trip error",
    )
    return parser.parse_args()


def _inner(hidden, output_features, traffic):
    return max(16, min(1024, int(traffic // (output_features * hidden * 2))))


def _median(mx, build, inner, rounds):
    for _ in range(3):
        mx.eval([build() for _ in range(inner)])
    mx.synchronize()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        mx.eval([build() for _ in range(inner)])
        samples.append((time.perf_counter_ns() - started) / inner / 1e9)
    return statistics.median(samples)


def main():
    arguments = _arguments()
    try:
        import mlx.core as mx
    except ImportError:
        print("mlx is required")
        return 1

    from metile.backends.mlx_dense import MLXDenseWeight, mlx_dense_matmul

    ceiling = agx.STREAMING_READ_GBPS
    print(f"attention projections, matched bf16, streaming ceiling {ceiling} GB/s")
    print(f"{int(arguments.traffic / 1e6)} MB of weight traffic per eval\n")
    header = (
        f"{'shape':<20}{'rows':>5}{'inner':>7}{'MLX GB/s':>10}{'gap':>7}{'meTile':>9}{'exact':>7}"
    )
    print(header)
    print("-" * len(header))

    for label, hidden, output_features in SHAPES:
        inner = _inner(hidden, output_features, arguments.traffic)
        for rows in ROWS:
            mx.random.seed(0)
            dense = mx.random.normal((output_features, hidden)).astype(mx.bfloat16)
            values = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
            mx.eval(dense, values)
            weight = MLXDenseWeight.from_mlx(dense)
            mx.eval(weight.k_major)

            def native(values=values, dense=dense):
                return values @ dense.T

            def generated(values=values, weight=weight):
                return mlx_dense_matmul(values, weight)

            produced, reference = generated(), native()
            mx.eval(produced, reference)
            exact = bool(mx.array_equal(produced, reference).item())

            base = _median(mx, native, inner, arguments.rounds)
            ours = _median(mx, generated, inner, arguments.rounds)
            gbps = (output_features * hidden * 2) / base / 1e9
            print(
                f"{label:<20}{rows:>5}{inner:>7}{gbps:>10.0f}{ceiling / gbps:>6.1f}x"
                f"{base / ours:>8.3f}x{'yes' if exact else 'NO':>7}"
            )

    print("\ngap is the ceiling divided by what MLX achieves: what any kernel could win.")
    print("Below 1.0x the weights are cache-resident and DRAM is not the limit at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
