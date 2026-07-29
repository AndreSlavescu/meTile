"""How much instruction-level parallelism does this GPU need, and what can scheduling buy?

Reordering instructions only pays if the hardware stalls without independent work nearby.
This asks the machine directly, with no memory traffic to confuse the answer: a chain of
dependent fmas, replicated into `chains` independent chains.

  chains = 1        every fma waits for the one before it, so throughput is 1 / latency
  chains = enough   independent work covers the pipeline and throughput saturates

The ratio between the two ends is the most any scheduler could ever win on this hardware,
for compute-bound code. Memory-bound code gets less than that, usually nothing.

Measured on M5 (G17): fp32 saturates at 2 chains for a total gain of 1.09x, fp16 at about
6 chains for 1.41x. A single dependent chain already reaches 92% of fp32 peak, because the
GPU covers latency with thread-level parallelism rather than ILP within a thread. The same
run puts scalar peak at 4.1 TFLOP/s fp32 and 6.5 fp16 against a matrix-unit peak of 15.3,
so picking the right functional unit is worth 2.4x where scheduling is worth 1.09x.

Runs on meTile's own Metal runtime. No MLX, nothing to compare against; this measures the
hardware, not a competitor.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

THREADGROUP = 256
THREADGROUPS = 1024
# fmas emitted per chain per loop iteration, so loop overhead stays negligible.
UNROLL = 8
CHAIN_COUNTS = (1, 2, 3, 4, 6, 8, 12)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=9)
    return parser.parse_args()


def kernel_source(chains, dtype):
    """Independent fma chains. Each chain is serial; the chains do not interact."""
    declare = "\n    ".join(f"{dtype} acc{c} = seed + {dtype}({c});" for c in range(chains))
    body = "\n            ".join(
        f"acc{c} = fma(acc{c}, coefficient, offset);" for c in range(chains)
    )
    total = " + ".join(f"acc{c}" for c in range(chains))
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void probe(device const {dtype}* input [[buffer(0)]],
                  device {dtype}* out         [[buffer(1)]],
                  constant uint& iterations   [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {{
    {dtype} seed = input[gid & 255];
    {dtype} coefficient = input[(gid + 1) & 255];
    {dtype} offset = input[(gid + 2) & 255];
    {declare}
    for (uint i = 0; i < iterations; ++i) {{
        for (uint u = 0; u < {UNROLL}; ++u) {{
            {body}
        }}
    }}
    out[gid] = {total};
}}
"""


def main():
    arguments = _arguments()
    import metile
    from metile.runtime.metal_device import MetalDevice

    try:
        from benchmarks.agx_registers import Unavailable, inspect
    except ImportError:  # pragma: no cover - registers are a nicety, not the measurement
        Unavailable, inspect = RuntimeError, None

    device = MetalDevice.get()
    rng = np.random.default_rng(0)
    threads = THREADGROUPS * THREADGROUP
    grid, block = (threads, 1, 1), (THREADGROUP, 1, 1)

    print(f"device: {device.name}")
    print(f"{threads} threads, {arguments.iterations} iterations x {UNROLL} fma per chain")

    for dtype, numpy_dtype in (("float", np.float32), ("half", np.float16)):
        data = metile.Buffer(data=rng.random(256).astype(numpy_dtype))
        out = metile.Buffer(data=np.zeros(threads, dtype=numpy_dtype))
        iterations = metile.Buffer(data=np.array([arguments.iterations], dtype=np.uint32))
        buffers = [data.metal_buffer, out.metal_buffer, iterations.metal_buffer]

        entries = []
        for chains in CHAIN_COUNTS:
            source = kernel_source(chains, dtype)
            registers = -1
            if inspect is not None:
                try:
                    registers = inspect(source, "probe", ".metile-agx")["registers"]
                except (Unavailable, RuntimeError):
                    registers = -1
            entries.append((chains, device.compile_msl(source, "probe"), registers))

        def measure(pipeline, batch=4):
            started = time.perf_counter_ns()
            for _ in range(batch):
                device.dispatch_kernel(pipeline, buffers, grid, block)
            device.sync()
            return (time.perf_counter_ns() - started) / batch / 1e9

        for _, pipeline, _ in entries:
            measure(pipeline)

        # Interleave, because these kernels run for milliseconds and a sequential sweep
        # measures thermal drift as much as it measures the kernels.
        samples = {chains: [] for chains, _, _ in entries}
        for index in range(arguments.rounds):
            ordered = entries[index % len(entries) :] + entries[: index % len(entries)]
            for chains, pipeline, _ in ordered:
                samples[chains].append(measure(pipeline))

        print(f"\n{dtype}")
        print(f"{'chains':>7}{'regs':>6}{'us':>10}{'GFLOP/s':>11}{'vs 1 chain':>12}")
        baseline = None
        for chains, _, registers in entries:
            seconds = statistics.median(samples[chains])
            rate = 2 * threads * arguments.iterations * UNROLL * chains / seconds / 1e9
            baseline = baseline if baseline is not None else rate
            print(
                f"{chains:>7}{registers:>6}{seconds * 1e6:>10.1f}"
                f"{rate:>11.0f}{rate / baseline:>11.2f}x"
            )


if __name__ == "__main__":
    main()
