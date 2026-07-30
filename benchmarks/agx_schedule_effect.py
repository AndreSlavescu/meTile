"""What the native scheduling pass actually changes, measured two ways.

The pass reorders operations so fewer values stay alive at once, on the grounds that reaching
the register budget costs 1.3x to 6.7x while perfect instruction-level parallelism is worth at
most 1.09x. Both halves of that are checkable rather than arguable:

  registers  read out of the compiled binary with `metile.target.agx.inspect`, which reports
             what the Metal backend actually allocated. The number the pass exists to move, and
             a fact rather than a timing.

  time       interleaved, and with each variant compiled once up front.

The compile-once part is the whole difficulty. The obvious way to switch the pass on and off is
to reload the modules between measurements, and doing that inside the timing loop measures the
Metal compiler rather than the kernel: a first version of this benchmark reported the scheduled
variant at 0.83x to 0.92x, which was the extra compile and not the GPU. So each variant is built
once, its compiled kernel object is kept alive, and only dispatches are timed.

Cases where the pass produces byte-identical MSL are the built-in control, and they are the
reason to trust anything else here. Identical source cannot run faster than itself, so whatever
spread those rows show is the harness noise floor, and every other row is measured against it
rather than read off directly. The first version of this file would have reported a 1.28x win on
softmax; the control row beside it was reading 0.75x on identical source at the same moment.
"""

import argparse
import importlib
import os
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice
from metile.target import agx

# Chosen to span the pressure range rather than to flatter the pass. The reductions hold almost
# nothing live and should not move at all; the GEMM tiles are where an order that hoists every
# load has somewhere to go, and the widest tile is the one that approaches the budget.
CASES = (
    ("rmsnorm", (8, 2048)),
    ("softmax", (8, 16384)),
    ("matmul", (256, 256, 256, 32, 32, 32)),
    ("matmul", (512, 512, 512, 64, 64, 32)),
    # 128x128x32 does not fit: it wants 33408 bytes of threadgroup memory against a 32768
    # limit, so the widest admissible tile is the one that gets measured.
    ("matmul", (1024, 1024, 1024, 128, 64, 32)),
    ("matmul", (1024, 1024, 1024, 64, 64, 64)),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--skip-registers", action="store_true")
    return parser.parse_args()


def _fresh(enabled):
    """Reimport the kernel modules with the pass in the requested state.

    Each reload produces distinct kernel objects with their own compilation caches, so the two
    returned module sets hold two genuinely different compiled kernels that can be dispatched
    against each other afterwards without reloading anything.
    """
    os.environ["METILE_SCHEDULE"] = "1" if enabled else "0"
    for name in ("metile.compiler.scheduling", "metile.frontend.kernel"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    modules = {}
    for name in ("kernels.rmsnorm", "kernels.softmax", "kernels.gemm"):
        module = sys.modules.get(name)
        modules[name] = importlib.reload(module) if module else importlib.import_module(name)
    return modules


def _buffers(case, shape):
    random = np.random.default_rng(sum(shape))
    if case in ("rmsnorm", "softmax"):
        rows, width = shape
        values = metile.Buffer(data=random.standard_normal((rows, width), dtype=np.float32).ravel())
        output = metile.Buffer.zeros((rows * width,))
        if case == "rmsnorm":
            return values, metile.Buffer(data=np.ones(width, dtype=np.float32)), output
        return values, output
    m, n, k, *_ = shape
    return (
        random.standard_normal((m, k), dtype=np.float32),
        random.standard_normal((k, n), dtype=np.float32),
        np.zeros((m, n), dtype=np.float32),
    )


def _launcher(case, shape, modules, buffers):
    """A zero-argument callable that dispatches this case once."""
    if case == "rmsnorm":
        rows, width = shape
        values, weight, output = buffers
        kernel = modules["kernels.rmsnorm"].rmsnorm
        return lambda: kernel[(rows,)](values, weight, output, width, 1e-5, BLOCK=256)
    if case == "softmax":
        rows, width = shape
        values, output = buffers
        kernel = modules["kernels.softmax"].softmax
        return lambda: kernel[(rows,)](values, output, width, BLOCK=256)
    m, n, k, bm, bn, bk = shape
    left, right, out = buffers
    kernel = modules["kernels.gemm"].matmul
    grid = ((m + bm - 1) // bm, (n + bn - 1) // bn)
    return lambda: kernel[grid](left, right, out, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)


def _capture(case, shape):
    """Build both variants, returning {enabled: (kernel name, msl, launcher)}.

    Both variants are handed the same buffers. Giving each its own allocation would compare two
    kernels over two pieces of memory, and the placement difference lands in the timing.
    """
    built = {}
    shared = _buffers(case, shape)
    for enabled in (False, True):
        modules = _fresh(enabled)
        frontend = sys.modules["metile.frontend.kernel"]
        original = frontend.emit
        seen = {}

        def spy(func, _original=original, _seen=seen):
            source = _original(func)
            _seen.setdefault("first", (func.name, source))
            return source

        frontend.emit = spy
        try:
            launch = _launcher(case, shape, modules, shared)
            launch()
            MetalDevice.get().sync()
        finally:
            frontend.emit = original
        name, source = seen["first"]
        built[enabled] = (name, source, launch)
    return built


def _time(built, rounds, inner=8):
    """Interleaved dispatch timing, seconds per dispatch. Nothing compiles inside the loop."""
    order = [False, True]
    for enabled in order:
        for _ in range(3):
            for _ in range(inner):
                built[enabled][2]()
            MetalDevice.get().sync()
    samples = {False: [], True: []}
    for index in range(rounds):
        for enabled in order[index % 2 :] + order[: index % 2]:
            started = time.perf_counter_ns()
            for _ in range(inner):
                built[enabled][2]()
            MetalDevice.get().sync()
            samples[enabled].append((time.perf_counter_ns() - started) / inner / 1e9)
    return {enabled: statistics.median(values) for enabled, values in samples.items()}


def main():
    arguments = _arguments()

    print("Effect of metile.compiler.scheduling on real kernels")
    print(f"register budget {agx.REGISTER_BUDGET}, ILP ceiling {agx.ILP_CEILING}\n")
    header = f"{'kernel':<9}{'shape':>22}{'regs off':>10}{'regs on':>9}{'msl':>7}{'time':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for case, shape in CASES:
        built = _capture(case, shape)
        name, plain, _ = built[False]
        _, scheduled, _ = built[True]
        registers = ("-", "-")
        if not arguments.skip_registers:
            try:
                before = agx.inspect(plain, name)
                after = agx.inspect(scheduled, name)
                registers = (
                    f"{before['registers']}{'*' if before['spilling'] else ''}",
                    f"{after['registers']}{'*' if after['spilling'] else ''}",
                )
            except (agx.Unavailable, RuntimeError) as error:
                print(f"  register read unavailable: {str(error)[:60]}")
        medians = _time(built, arguments.rounds)
        ratio = medians[False] / medians[True] if medians[True] else float("nan")
        identical = plain == scheduled
        rows.append((case, shape, registers, identical, ratio))
        printable = "x".join(str(part) for part in shape)
        print(
            f"{case:<9}{printable:>22}{registers[0]:>10}{registers[1]:>9}"
            f"{'control' if identical else 'moved':>9}{ratio:>8.3f}x"
        )

    controls = [abs(ratio - 1.0) for *_, identical, ratio in rows if identical]
    floor = max(controls) if controls else 0.0
    print("\n* marks a register count at or above the budget, which is spilling.")
    print(
        f"Noise floor from {len(controls)} control rows with identical MSL: "
        f"{floor * 100:.1f}%. Nothing smaller than that is a result."
    )
    for case, shape, _, identical, ratio in rows:
        if identical or abs(ratio - 1.0) <= floor:
            continue
        printable = "x".join(str(part) for part in shape)
        print(f"  outside the noise floor: {case} {printable} at {ratio:.3f}x")
    moved = [row for row in rows if not row[3]]
    changed = [row for row in moved if row[2][0] != row[2][1]]
    print(
        f"Registers: {len(changed)} of {len(moved)} reordered kernels allocated a different count."
    )


if __name__ == "__main__":
    main()
