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

  saturation   the sweep has to reach the known 120.6 GB/s at large sizes, or the thread count is too
               low to saturate and every number is a lower bound on something else. Printed as a check
               rather than assumed.
  amortisation each dispatch reads gigabytes and runs for tens of milliseconds, so launch overhead is
               far below the noise. Timing small kernels through a host round trip is what produced
               three fabricated results earlier in this project.
  interleaving sizes are measured in rotating order across rounds, because a sweep that walks from
               small to large measures thermal drift as much as it measures the hierarchy.

One limit of this probe is worth knowing before reading its output. Inside the resident regime bandwidth
*rises* with working set -- 1545, 2006, 2138 and 2386 GB/s at 256 KB, 512 KB, 1 MB and 2 MB -- which
cannot be a property of a cache. A smaller working set means fewer inner iterations per pass, so the
pass loop's bookkeeping takes a larger share, and the effect shrinks as the set grows. The probe
therefore establishes that a resident set runs at 2386 GB/s or better and says nothing reliable about
how that varies below 2 MB. `metile.target.agx` records it as one number for that reason.

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
# Enough threads to saturate DRAM, and few enough that a 256 KB working set still gives every thread
# something to do: the inner loop strides by the total thread count, so a working set smaller than that
# leaves threads idle and measures parallelism instead of bandwidth.
THREADGROUPS = 64
VECTOR_BYTES = 16  # float4

SIZES_KB = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=float, default=4.0, help="GB read per dispatch")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threadgroups", type=int, default=THREADGROUPS)
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
    threads = arguments.threadgroups * THREADGROUP
    grid, block = (threads, 1, 1), (THREADGROUP, 1, 1)
    pipeline = device.compile_msl(READ_KERNEL, "probe")

    print(f"device: {device.name}")
    print(f"{threads} threads, {arguments.traffic:g} GB read per dispatch")
    print(f"recorded streaming ceiling: {agx.STREAMING_READ_GBPS} GB/s\n")

    target_bytes = arguments.traffic * 1e9
    cases = []
    largest = max(SIZES_KB) * 1024
    data = metile.Buffer(data=np.ones(largest // 4, dtype=np.float32))
    out = metile.Buffer(data=np.zeros(threads * 4, dtype=np.float32))

    for size_kb in SIZES_KB:
        size = size_kb * 1024
        vectors = size // VECTOR_BYTES
        if vectors < threads:
            continue
        passes = max(1, int(target_bytes // size))
        buffers = [
            data.metal_buffer,
            out.metal_buffer,
            metile.Buffer(data=np.array([vectors], dtype=np.uint32)).metal_buffer,
            metile.Buffer(data=np.array([passes], dtype=np.uint32)).metal_buffer,
            metile.Buffer(data=np.array([threads], dtype=np.uint32)).metal_buffer,
        ]
        cases.append((size, vectors, passes, buffers))

    def measure(buffers):
        started = time.perf_counter_ns()
        device.dispatch_kernel(pipeline, buffers, grid, block)
        device.sync()
        return (time.perf_counter_ns() - started) / 1e9

    for _, _, _, buffers in cases:
        measure(buffers)

    samples = {size: [] for size, _, _, _ in cases}
    for index in range(arguments.rounds):
        ordered = cases[index % len(cases) :] + cases[: index % len(cases)]
        for size, _, _, buffers in ordered:
            samples[size].append(measure(buffers))

    print(f"{'working set':>13}{'passes':>8}{'ms':>9}{'GB/s':>9}{'vs DRAM':>9}")
    results = []
    for size, vectors, passes, _ in cases:
        seconds = statistics.median(samples[size])
        gbps = (vectors * VECTOR_BYTES * passes) / seconds / 1e9
        results.append((size, gbps))
        label = f"{size // 1024} KB" if size < 1024 * 1024 else f"{size // (1024 * 1024)} MB"
        print(
            f"{label:>13}{passes:>8}{seconds * 1e3:>9.1f}{gbps:>9.1f}"
            f"{gbps / agx.STREAMING_READ_GBPS:>8.2f}x"
        )

    dram = min(gbps for _, gbps in results)
    peak = max(gbps for _, gbps in results)
    print(f"\nslowest {dram:.1f} GB/s, fastest {peak:.1f} GB/s, ratio {peak / dram:.2f}x")
    saturated = dram >= 0.9 * agx.STREAMING_READ_GBPS
    print(
        f"saturation check: largest working sets reach {dram:.1f} GB/s against a recorded "
        f"{agx.STREAMING_READ_GBPS} GB/s -- {'ok' if saturated else 'TOO LOW, raise --threadgroups'}"
    )
    if not saturated:
        print(
            "Until that passes, every number here is a lower bound and the knees may be artefacts."
        )
        return 1

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
