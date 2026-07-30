"""Where the memory hierarchy changes speed, which is what tiling has to be sized against.

meTile records one bandwidth number, 120.6 GB/s, measured streaming from DRAM. That number is not the
whole story and the gap showed up by accident: a 786 KB weight matrix read at 254 GB/s, more than twice
the recorded ceiling, which means something above DRAM is serving it. A compiler choosing tile sizes
needs to know where that changes, because a tile that fits the fast level and a tile that misses it are
not the same kernel.

The method is a coalesced streaming read over a working set of `size` bytes, looped until a fixed total
of traffic has been read. Bandwidth is total bytes over elapsed time, so a working set that stays
resident reports the resident level's bandwidth and one that does not reports DRAM's. Plateaus are
levels; the knees between them are capacities.

Three things make the numbers trustworthy rather than suggestive:

  occupancy    every working set is measured at several threadgroup counts and the best is kept. A fixed
               count is not safe: the first version of this used 64 for everything, which saturates DRAM
               at 128 MB but starves an 8 MB working set by a factor of twelve, and the resulting table
               described the thread count rather than the hierarchy.
  amortisation each dispatch reads gigabytes and runs for tens of milliseconds, so launch overhead is
               far below the noise. Timing small kernels through a host round trip is what produced
               three fabricated results earlier in this project.
  interleaving sizes are measured in rotating order across rounds, because a sweep that walks from
               small to large measures thermal drift as much as it measures the hierarchy.

One limit remains after the occupancy sweep. The smallest working sets cannot be given both enough
threads to saturate and enough work per thread to amortise the pass loop, because the two demands
conflict once the set is only a few times the thread count. Numbers below about 1 MB are floors rather
than levels, and `metile.target.agx` records the resident regime as a single figure for that reason.

usage:
    python benchmarks/agx_memory_hierarchy.py
    python benchmarks/agx_memory_hierarchy.py --traffic 8 --rounds 7
"""

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import metile
from metile.target import agx

THREADGROUP = 256
VECTOR_BYTES = 16

# Threadgroup counts tried at every working set, with the best taken. A single count cannot serve the
# whole sweep and picking one produced a wrong answer that sat in the target model until the occupancy
# probe contradicted it: at 64 groups an 8 MB working set reads 196 GB/s and at 512 it reads 2403, so the
# "level" recorded at 8 MB was the thread count, not the memory system. Too few threads starves the path;
# too many leaves each thread one inner iteration per pass, where loop bookkeeping competes with the
# loads. Only the maximum over the sweep is a property of the part.
THREADGROUP_COUNTS = (32, 64, 128, 256, 512, 1024)  # float4

SIZES_KB = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=float, default=4.0, help="GB read per dispatch")
    parser.add_argument("--rounds", type=int, default=5)
    return parser.parse_args()


READ_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void probe(device const float4* data [[buffer(0)]],
                  device float4* out        [[buffer(1)]],
                  constant uint& vectors    [[buffer(2)]],
                  constant uint& passes     [[buffer(3)]],
                  constant uint& stride     [[buffer(4)]],
                  uint gid [[thread_position_in_grid]]) {
    float4 total = float4(0.0f);
    for (uint pass = 0; pass < passes; ++pass) {
        for (uint index = gid; index < vectors; index += stride) {
            total += data[index];
        }
    }
    // Written unconditionally so nothing above can be discarded as dead.
    out[gid] = total;
}
"""


def main():
    arguments = _arguments()
    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    pipeline = device.compile_msl(READ_KERNEL, "probe")

    print(f"device: {device.name}")
    print(f"{arguments.traffic:g} GB read per dispatch, best of {THREADGROUP_COUNTS} threadgroups")
    print(f"recorded streaming ceiling: {agx.STREAMING_READ_GBPS} GB/s\n")

    target_bytes = arguments.traffic * 1e9
    largest = max(SIZES_KB) * 1024
    data = metile.Buffer(data=np.ones(largest // 4, dtype=np.float32))
    out = metile.Buffer(data=np.zeros(max(THREADGROUP_COUNTS) * THREADGROUP * 4, dtype=np.float32))

    cases = []
    for size_kb in SIZES_KB:
        size = size_kb * 1024
        vectors = size // VECTOR_BYTES
        passes = max(1, int(target_bytes // size))
        for count in THREADGROUP_COUNTS:
            if vectors < count * THREADGROUP:
                continue
            cases.append((size, count, vectors, passes))

    def measure(case):
        _, count, vectors, passes = case
        threads = count * THREADGROUP
        buffers = [
            data.metal_buffer,
            out.metal_buffer,
            metile.Buffer(data=np.array([vectors], dtype=np.uint32)).metal_buffer,
            metile.Buffer(data=np.array([passes], dtype=np.uint32)).metal_buffer,
            metile.Buffer(data=np.array([threads], dtype=np.uint32)).metal_buffer,
        ]
        started = time.perf_counter_ns()
        device.dispatch_kernel(pipeline, buffers, (threads, 1, 1), (THREADGROUP, 1, 1))
        device.sync()
        seconds = (time.perf_counter_ns() - started) / 1e9
        return vectors * VECTOR_BYTES * passes / seconds / 1e9

    for case in cases:
        measure(case)

    samples = {case: [] for case in cases}
    for index in range(arguments.rounds):
        ordered = cases[index % len(cases) :] + cases[: index % len(cases)]
        for case in ordered:
            samples[case].append(measure(case))

    rates = {case: statistics.median(values) for case, values in samples.items()}

    print(f"{'working set':>13}{'best groups':>13}{'GB/s':>9}{'vs DRAM':>9}{'worst groups':>14}")
    results = []
    for size_kb in SIZES_KB:
        size = size_kb * 1024
        here = [(case, rate) for case, rate in rates.items() if case[0] == size]
        if not here:
            continue
        best_case, best = max(here, key=lambda pair: pair[1])
        worst = min(rate for _, rate in here)
        results.append((size, best))
        label = f"{size // 1024} KB" if size < 1024 * 1024 else f"{size // (1024 * 1024)} MB"
        print(
            f"{label:>13}{best_case[1]:>13}{best:>9.0f}"
            f"{best / agx.STREAMING_READ_GBPS:>8.2f}x{best / worst:>13.1f}x"
        )

    dram = min(gbps for _, gbps in results)
    peak = max(gbps for _, gbps in results)
    print(f"\nslowest {dram:.0f} GB/s, fastest {peak:.0f} GB/s, ratio {peak / dram:.2f}x")
    print(
        "last column is how much the thread count alone moves that size, which is why it is swept."
    )

    # Report the knees rather than leaving them to be eyeballed: a level boundary is where bandwidth
    # drops materially between adjacent sizes.
    print("\ntransitions, where bandwidth drops by more than 15% between adjacent sizes:")
    found = False
    for (small, fast), (large, slow) in itertools.pairwise(results):
        if slow < 0.85 * fast:
            found = True
            print(
                f"  between {small // 1024} KB and {large // 1024} KB: "
                f"{fast:.1f} -> {slow:.1f} GB/s"
            )
    if not found:
        print("  none; the hierarchy looks flat over this range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
