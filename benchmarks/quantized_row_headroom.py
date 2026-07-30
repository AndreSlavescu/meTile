"""How much room MLX leaves at each weight width, which is what decides where a kernel can win.

The multi-row win at four bits is not a general property of batching quantized decode. It exists
because MLX re-reads weights per row tile and, at four bits, that drops its effective weight-read
bandwidth to about a third of what the part can stream. This measures that directly for each width
so the question "should we build a multi-row kernel for N bits" has an answer before anyone builds
one.

What it shows on M5, against a measured streaming ceiling near 120 GB/s:

    rows      int4      int8
       1      60        83
       8      42        65
      16      36        62
      32      37        62

So four bits leaves roughly 3.3x on the table at sixteen rows and eight bits roughly 1.9x. The
four-bit kernel captures about half of its share, measuring 1.45x to 1.73x at rows 8 to 32. Half of
the eight-bit share would be around 1.4x, which is worth wanting, and it is not reachable through
the same path: the matrix-unit affine fragment format is four bits wide and `lower_affine_matmul`
has no bit width to thread through it.

Reading effective bandwidth rather than time is what makes the widths comparable. Eight-bit weights
are twice the bytes, so a slower wall time can still be the better use of the memory system, and a
ratio of times cannot tell the two apart.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from metile.target import agx

HIDDEN, INTERMEDIATE = 1536, 8960
GROUP = 64
ROWS = (1, 2, 4, 8, 16, 32, 64)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    return parser.parse_args()


def _bandwidth(mx, bits, rows, hidden, intermediate, rounds, inner=3):
    """MLX's effective weight-read bandwidth in GB/s for one width and row count."""
    mx.random.seed(0)
    dense = mx.random.normal((intermediate, hidden)).astype(mx.float16)
    packed, scales, biases = mx.quantize(dense, group_size=GROUP, bits=bits, mode="affine")
    activations = mx.random.normal((rows, hidden)).astype(mx.float16)
    mx.eval(packed, scales, biases, activations)

    def run():
        return mx.quantized_matmul(
            activations,
            packed,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=GROUP,
            bits=bits,
            mode="affine",
        )

    for _ in range(3):
        mx.eval([run() for _ in range(inner)])
    mx.synchronize()
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        mx.eval([run() for _ in range(inner)])
        samples.append((time.perf_counter_ns() - started) / inner / 1e9)

    # Weights plus one scale and one bias per group, which is what a single pass has to read.
    parameters = intermediate * (hidden // GROUP) * 2 * 2
    weight_bytes = intermediate * hidden * bits / 8 + parameters
    return weight_bytes / statistics.median(samples) / 1e9


def main():
    arguments = _arguments()
    try:
        import mlx.core as mx
    except ImportError:
        print("mlx is required")
        return 1

    ceiling = agx.STREAMING_READ_GBPS
    print("MLX's effective weight-read bandwidth, GB/s")
    print(f"shape {arguments.hidden}x{arguments.intermediate}, group {GROUP}, ")
    print(f"measured streaming ceiling {ceiling} GB/s\n")
    print(f"{'rows':>6}{'int4':>9}{'int8':>9}{'int4 gap':>11}{'int8 gap':>11}")

    for rows in ROWS:
        four = _bandwidth(mx, 4, rows, arguments.hidden, arguments.intermediate, arguments.rounds)
        eight = _bandwidth(mx, 8, rows, arguments.hidden, arguments.intermediate, arguments.rounds)
        print(
            f"{rows:>6}{four:>9.1f}{eight:>9.1f}{ceiling / four:>10.2f}x{ceiling / eight:>10.2f}x"
        )

    print("\nThe gap columns bound what any kernel could win at that width and row count.")
    print("A width whose gap is near 1.00x has nothing to give however the kernel is written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
