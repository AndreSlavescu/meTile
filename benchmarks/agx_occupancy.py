"""How much parallelism a kernel needs before memory bandwidth saturates.

Decode kernels achieve 80 to 128 GB/s while the memory level their working set sits at delivers 128 to
555. That gap is not tiling -- decode reads each weight once, so there is no reuse to capture -- which
leaves the question of what it is. The first suspect is that the kernel is not running enough
threadgroups to saturate the path in the first place.

That is a property of the part, not of any kernel, so it can be measured directly: hold the working set
fixed and sweep the number of threadgroups. The curve's knee is the parallelism a kernel needs, and any
kernel launching fewer than that is leaving bandwidth behind for a reason a tile-size change could fix.

The number matters for tile selection specifically. A GEMM tiled by output width launches
`output_features / block_n` threadgroups, so a wider tile means fewer of them: at N=8960 a block_n of 64
gives 140 threadgroups and 256 gives 35. If saturation needs more than 35, the widest tiles are
self-limiting and the compiler can rule them out before measuring anything.

Same controls as the other probes here: each dispatch reads gigabytes so launch overhead is negligible,
thread counts are measured in rotating order so drift does not land on one end of the sweep, and elapsed
time has to scale with the pass count or the loop is being collapsed.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import metile

THREADGROUP = 256
VECTOR_BYTES = 16
THREADGROUP_COUNTS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

# Two footprints: one in the resident regime and one streaming, because the parallelism needed to
# saturate a cache and to saturate DRAM are not obviously the same number.
WORKING_SETS = (1024 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024)

KERNEL = """#include <metal_stdlib>
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
    out[gid] = total;
}
"""


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=float, default=4.0, help="GB read per dispatch")
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main():
    arguments = _arguments()
    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    pipeline = device.compile_msl(KERNEL, "probe")
    largest = max(WORKING_SETS)
    data = metile.Buffer(data=np.ones(largest // 4, dtype=np.float32))
    out = metile.Buffer(data=np.zeros(max(THREADGROUP_COUNTS) * THREADGROUP * 4, dtype=np.float32))

    print(f"device: {device.name}")
    print(f"{THREADGROUP} threads per group, {arguments.traffic:g} GB read per dispatch\n")

    def measure(threadgroups, size):
        threads = threadgroups * THREADGROUP
        vectors = size // VECTOR_BYTES
        passes = max(1, int(arguments.traffic * 1e9 // size))
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

    cases = [
        (count, size)
        for size in WORKING_SETS
        for count in THREADGROUP_COUNTS
        if size // VECTOR_BYTES >= count * THREADGROUP
    ]
    for count, size in cases:
        measure(count, size)

    samples = {case: [] for case in cases}
    for index in range(arguments.rounds):
        ordered = cases[index % len(cases) :] + cases[: index % len(cases)]
        for case in ordered:
            samples[case].append(measure(*case))

    rates = {case: statistics.median(values) for case, values in samples.items()}

    header = f"{'groups':>7}{'threads':>9}" + "".join(
        f"{size // (1024 * 1024) if size >= 1024 * 1024 else size // 1024:>10}"
        + ("MB" if size >= 1024 * 1024 else "KB")
        for size in WORKING_SETS
    )
    print(header)
    print("-" * len(header))
    for count in THREADGROUP_COUNTS:
        cells = []
        for size in WORKING_SETS:
            rate = rates.get((count, size))
            cells.append(f"{rate:>10.0f}  " if rate else f"{'-':>12}")
        print(f"{count:>7}{count * THREADGROUP:>9}" + "".join(cells))

    print("\nthreadgroups needed to reach 90% of the best this sweep saw, per working set:")
    knees = {}
    for size in WORKING_SETS:
        curve = [
            (count, rates[count, size]) for count in THREADGROUP_COUNTS if (count, size) in rates
        ]
        peak = max(rate for _, rate in curve)
        knee = next(count for count, rate in curve if rate >= 0.9 * peak)
        knees[size] = knee
        label = f"{size // (1024 * 1024)} MB" if size >= 1024 * 1024 else f"{size // 1024} KB"
        print(f"  {label:>7}: {knee:>4} groups ({knee * THREADGROUP} threads) for {peak:.0f} GB/s")

    # What this means for tile selection, which is the reason to measure it.
    print("\nfor a GEMM tiled by output width, groups launched = output_features / block_n:")
    print(f"{'N':>7}" + "".join(f"{f'bn={bn}':>10}" for bn in (32, 64, 128, 256)))
    worst_knee = max(knees.values())
    for output_features in (4864, 8192, 8960, 17408):
        cells = []
        for block_n in (32, 64, 128, 256):
            groups = output_features // block_n
            cells.append(f"{groups:>7}{'  ok' if groups >= worst_knee else ' LOW'}")
        print(f"{output_features:>7}" + "".join(cells))
    print(f"\nLOW marks fewer than {worst_knee} groups, the most demanding knee above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
