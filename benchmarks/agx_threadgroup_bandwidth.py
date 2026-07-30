"""What threadgroup memory is actually for on this part, and which strides it punishes.

meTile has several passes built around threadgroup memory -- cooperative loads, shared-memory padding
and swizzling, double buffering -- and they all assume it is the fast place to put data. On M5 that
assumption needs checking, because device memory is not slow when it is resident: a working set under
2 MB reads at 2386 GB/s against DRAM's 121. So the question is not whether threadgroup memory is fast,
but whether it beats the cache that would have served the same bytes anyway.

Two things get measured, both over a 32 KB working set per threadgroup so the device arm is resident and
the comparison is staging against a cache hit rather than against DRAM.

  bandwidth   a coalesced read from threadgroup memory against the same read from device memory.
  stride      the same total bytes with consecutive lanes a fixed stride apart, swept, for both spaces.
              This separates how much each cares about indexing, which is the property padding and
              swizzling exist to manage.

The stride sweep is the interesting half, and it corrected a wrong diagnosis on the way here. A
per-thread span of eight vectors made the threadgroup arm about twice as slow, which looked like a bank
conflict, so an arm was added that padded the span by one to break it up. Padding made it worse -- the
unpadded span was seven, and adding one moved it *onto* 128 bytes rather than off it. Padding is not a
direction, it is an arithmetic result, and plus one is not automatically safe.

Controls, for the reasons three fabricated results earlier in this project established: elapsed time has
to scale with the pass count or the loop is being collapsed, and each dispatch runs for milliseconds so
launch overhead sits far below the noise.
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
from metile.target import agx

THREADGROUP = 256
THREADGROUPS = 64
VECTOR_BYTES = 16
# A power of two so the sweep can wrap with a mask rather than a modulo, which would put a divide in the
# inner loop and measure that instead.
TILE_VECTORS = 2048
TILE_BYTES = TILE_VECTORS * VECTOR_BYTES
# Vectors each thread reads per pass, constant across strides so every stride moves the same bytes.
PER_THREAD = 8
STRIDES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 32)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, default=6000)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def _source(space, stride):
    """One arm: read PER_THREAD vectors per pass from `space`, lanes `stride` vectors apart."""
    staging = (
        f"    threadgroup float4 tile[{TILE_VECTORS}];\n"
        f"    for (uint index = lid; index < {TILE_VECTORS}; index += {THREADGROUP}) {{\n"
        f"        tile[index] = data[index];\n"
        f"    }}\n"
        f"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        if space == "threadgroup"
        else ""
    )
    read = "tile" if space == "threadgroup" else "data"
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void probe(device const float4* data [[buffer(0)]],
                  device float4* out        [[buffer(1)]],
                  constant uint& passes     [[buffer(2)]],
                  uint gid [[thread_position_in_grid]],
                  uint lid [[thread_position_in_threadgroup]]) {{
{staging}
    float4 total = float4(0.0f);
    for (uint pass = 0; pass < passes; ++pass) {{
        for (uint step = 0; step < {PER_THREAD}; ++step) {{
            total += {read}[(lid * {stride} + step) & {TILE_VECTORS - 1}];
        }}
    }}
    out[gid] = total;
}}
"""


def main():
    arguments = _arguments()
    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    threads = THREADGROUPS * THREADGROUP
    grid, block = (threads, 1, 1), (THREADGROUP, 1, 1)

    data = metile.Buffer(data=np.ones(TILE_BYTES // 4, dtype=np.float32))
    out = metile.Buffer(data=np.zeros(threads * 4, dtype=np.float32))

    print(f"device: {device.name}")
    print(f"{THREADGROUPS} threadgroups x {THREADGROUP} threads, {TILE_BYTES // 1024} KB per group")
    print(f"resident device read: {agx.RESIDENT_READ_GBPS} GB/s, DRAM: {agx.STREAMING_READ_GBPS}\n")

    arms = {}
    for space in ("threadgroup", "device"):
        for stride in STRIDES:
            arms[space, stride] = device.compile_msl(_source(space, stride), "probe")

    def measure(pipeline, passes):
        buffers = [
            data.metal_buffer,
            out.metal_buffer,
            metile.Buffer(data=np.array([passes], dtype=np.uint32)).metal_buffer,
        ]
        started = time.perf_counter_ns()
        device.dispatch_kernel(pipeline, buffers, grid, block)
        device.sync()
        return (time.perf_counter_ns() - started) / 1e9

    for pipeline in arms.values():
        measure(pipeline, arguments.passes)

    def sweep(passes):
        keys = list(arms)
        samples = {key: [] for key in keys}
        for index in range(arguments.rounds):
            ordered = keys[index % len(keys) :] + keys[: index % len(keys)]
            for key in ordered:
                samples[key].append(measure(arms[key], passes))
        return {key: statistics.median(values) for key, values in samples.items()}

    per_pass = THREADGROUPS * THREADGROUP * PER_THREAD * VECTOR_BYTES
    medians = sweep(arguments.passes)

    def rate(key):
        return per_pass * arguments.passes / medians[key] / 1e9

    print(f"{'stride':>7}{'bytes':>7}{'shared GB/s':>13}{'device GB/s':>13}{'shared/device':>15}")
    for stride in STRIDES:
        shared, plain = rate(("threadgroup", stride)), rate(("device", stride))
        print(
            f"{stride:>7}{stride * VECTOR_BYTES:>7}{shared:>13.0f}{plain:>13.0f}"
            f"{shared / plain:>14.2f}x"
        )

    best_shared = max(rate(("threadgroup", stride)) for stride in STRIDES)
    worst_shared = min(rate(("threadgroup", stride)) for stride in STRIDES)
    best_device = max(rate(("device", stride)) for stride in STRIDES)
    worst_device = min(rate(("device", stride)) for stride in STRIDES)
    print(
        f"\nspread across strides: threadgroup {best_shared / worst_shared:.2f}x, "
        f"device {best_device / worst_device:.2f}x"
    )
    print(f"best threadgroup {best_shared:.0f} GB/s vs best device {best_device:.0f} GB/s")

    penalised = [
        stride
        for stride in STRIDES
        if rate(("threadgroup", stride)) < 0.7 * best_shared
        and rate(("device", stride)) > 0.7 * best_device
    ]
    print(f"strides threadgroup memory punishes and device memory does not: {penalised or 'none'}")

    doubled = sweep(arguments.passes * 2)
    ratios = [doubled[key] / medians[key] for key in arms]
    print(f"\ncontrol, doubling passes: min {min(ratios):.2f}x, max {max(ratios):.2f}x, want ~2.00")
    if not all(1.7 <= ratio <= 2.3 for ratio in ratios):
        print("Time does not scale with work; the loop is being optimised and these are fiction.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
