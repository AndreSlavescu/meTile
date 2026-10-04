"""Verify that a rewritten AIR module executes through the existing Metal runtime.

Compile one MSL seed to textual AIR, change its FMA multiplier, then assemble and
execute both modules independently. This bypasses the MSL frontend for the edit;
Apple's GPU backend still selects and schedules the machine instructions.

This is a correctness probe, not a performance benchmark. Generated files remain
in the work directory so the toolchain-specific AIR can be inspected.

Usage: python benchmarks/hardware/agx_air_probe.py --workdir .metile-agx/air
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common.checkout import activate_checkout

activate_checkout(ROOT)

from metile.runtime.buffer import MtileBuffer
from metile.runtime.metal_device import MetalDevice

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void air_probe(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) {
    output[index] = fma(input[index], 2.0f, 1.0f);
}
"""


def rewrite_multiplier(module):
    """Change exactly one known FMA operand, refusing unfamiliar AIR output."""
    pattern = r"(@air\.fma\.f32\(float [^,\n]+, float )2\.000000e\+00(?=, float 1\.000000e\+00\))"
    rewritten, count = re.subn(pattern, r"\g<1>3.000000e+00", module)
    if count != 1:
        raise RuntimeError(f"expected one AIR FMA multiplier, found {count}; inspect baseline.ll")
    return rewritten


def _run_tool(*arguments):
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", *map(str, arguments)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout


def _assemble(module_path):
    air_path = module_path.with_suffix(".air")
    library_path = module_path.with_suffix(".metallib")
    _run_tool("metal-as", module_path, "-o", air_path)
    _run_tool("metallib", air_path, "-o", library_path)
    return library_path


def _execute(device, library_path, values):
    pipeline = device._load_metallib(str(library_path), "air_probe")
    inputs = MtileBuffer.from_numpy(values)
    outputs = MtileBuffer.from_numpy(np.full_like(values, np.nan))
    device.dispatch_kernel(
        pipeline,
        [inputs.metal_buffer, outputs.metal_buffer],
        (values.size, 1, 1),
        (32, 1, 1),
    )
    return outputs.numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".metile-agx/air"))
    arguments = parser.parse_args()
    workdir = arguments.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        version = _run_tool("metal", "--version").strip()
        print(f"toolchain: {version}")
        source_path = workdir / "seed.metal"
        baseline_path = workdir / "baseline.ll"
        rewritten_path = workdir / "rewritten.ll"
        source_path.write_text(SOURCE)
        _run_tool(
            "metal",
            "-std=metal3.1",
            "-fno-fast-math",
            "-S",
            "-emit-llvm",
            source_path,
            "-o",
            baseline_path,
        )
        rewritten_path.write_text(rewrite_multiplier(baseline_path.read_text()))
        baseline_library = _assemble(baseline_path)
        rewritten_library = _assemble(rewritten_path)

        device = MetalDevice.get()
        print(f"device: {device.name}")
        values = np.arange(-128, 129, dtype=np.float32) / np.float32(4)
        for label, library, multiplier in (
            ("baseline", baseline_library, 2),
            ("rewritten", rewritten_library, 3),
        ):
            expected = values * np.float32(multiplier) + np.float32(1)
            actual = _execute(device, library, values)
            np.testing.assert_array_equal(actual, expected)
            print(f"{label}: fma(input, {multiplier}, 1) matches all {values.size} inputs exactly")
    except (OSError, subprocess.TimeoutExpired, RuntimeError, AssertionError) as error:
        print(f"AIR probe failed: {error}", file=sys.stderr)
        return 1

    print(f"AIR rewrite executed successfully; artifacts: {workdir}")
    print("This establishes frontend bypass, not control over GPU scheduling or a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
