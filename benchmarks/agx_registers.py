"""Read how many registers a compiled Metal kernel uses on Apple GPUs.

Metal exposes no register count. maxTotalThreadsPerThreadgroup looks like it should stand
in for one and does not: on an M5 it reads 1024 whether a kernel holds 4 live floats or
512. The compiler does record the number, in a metadata segment of the GPU binary, and it
takes three unwraps to reach:

  1. MTLBinaryArchive.serialize writes a fat file; the applegpu_* slice is the GPU code.
  2. That slice's __compute section is itself a Mach-O.
  3. Inside it, __GPU_METADATA is a FlatBuffer, and __text is the machine code.

The count is field 0 of the table referenced by field 0 of the FlatBuffer root. Read by
path, never by byte offset: the buffer embeds the kernel name and signature, so a fixed
offset drifts between kernels and silently reports a neighbouring value.

There is no disassembler. Apple registers agx1/agx2/agx3 as targets in metal-objdump but
ships them with the instruction printers stripped, so __text can be sized but not decoded.

This is an analysis tool, not something the runtime should call: every reading costs a
Metal compile, on the order of a second. Its use is checking that a scheduling bound keeps
kernels clear of the register budget, rather than assuming it does.

usage:
    python benchmarks/agx_registers.py                 # audit the dense SwiGLU bound
    python benchmarks/agx_registers.py --self-check    # verify against known counts
"""

import argparse
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

HERE = Path(__file__).resolve().parent
SWIFT_SOURCE = HERE / "agx" / "agxdump.swift"

# Measured on M5 (G17) by growing a kernel's live values until the count stopped rising.
# Kernels that reach it are spilling, and measured 1.3x to 6.7x slower than lower-register
# siblings. Other Apple GPU generations have not been checked.
REGISTER_BUDGET = 140


class Unavailable(RuntimeError):
    """The toolchain needed to read register counts is not present."""


def _binary(workdir):
    """Build the harness once per working directory."""
    binary = workdir / "agxdump"
    if binary.exists():
        return binary
    if shutil.which("swiftc") is None:
        raise Unavailable("swiftc not found; Xcode command line tools are required")
    workdir.mkdir(parents=True, exist_ok=True)
    built = subprocess.run(
        ["swiftc", "-O", str(SWIFT_SOURCE), "-o", str(binary)], capture_output=True, text=True
    )
    if built.returncode != 0:
        raise Unavailable(f"could not build the harness: {built.stderr.strip()[:300]}")
    return binary


def _gpu_arch(archive):
    """The GPU slice is named for the chip generation, so read it rather than assume it."""
    listed = subprocess.run(
        ["xcrun", "metal-lipo", "-info", str(archive)], capture_output=True, text=True
    ).stdout
    for name in listed.split():
        if name.startswith("applegpu"):
            return name
    raise Unavailable(f"no applegpu slice in {archive.name}")


def _section(path, segment, section="__compute"):
    """Bytes of one section. Several segments here share a section name, so match both."""
    dumped = subprocess.run(
        ["xcrun", "metal-objdump", "-s", f"--section={section}", str(path)],
        capture_output=True,
        text=True,
    ).stdout
    payload = bytearray()
    inside = False
    for line in dumped.splitlines():
        if line.startswith("Contents of section"):
            label = line.rstrip(":").split()[-1]
            inside = label.endswith(f"{segment},{section}") if segment else True
            continue
        if not inside:
            continue
        match = re.match(r"^\s*[0-9a-f]+\s((?:[0-9a-f]{2,8}\s+){1,4})", line)
        if match:
            for word in match.group(1).split():
                payload += bytes.fromhex(word)
    return bytes(payload)


def _table_fields(blob, table):
    """Yield (index, absolute offset) for each field present in a FlatBuffer table."""
    if not 0 <= table <= len(blob) - 4:
        return
    vtable = table - struct.unpack_from("<i", blob, table)[0]
    if not 0 <= vtable <= len(blob) - 4:
        return
    vtable_bytes = struct.unpack_from("<H", blob, vtable)[0]
    if vtable_bytes < 4 or vtable + vtable_bytes > len(blob):
        return
    for index in range((vtable_bytes - 4) // 2):
        at = vtable + 4 + index * 2
        if at + 2 > len(blob):
            return
        relative = struct.unpack_from("<H", blob, at)[0]
        if relative and table + relative < len(blob):
            yield index, table + relative


def inspect(source, function, workdir):
    """Return {'registers', 'text_bytes', 'spilling'} for one MSL kernel."""
    workdir = Path(workdir)
    binary = _binary(workdir)
    metal = workdir / "kernel.metal"
    archive = workdir / "kernel.bin"
    thin = workdir / "kernel.gpu"
    nested = workdir / "kernel.inner"
    metal.write_text(source)

    built = subprocess.run(
        [str(binary), str(metal), function, str(archive)], capture_output=True, text=True
    )
    if built.returncode != 0:
        raise RuntimeError(built.stderr.strip()[:300])

    subprocess.run(
        ["xcrun", "metal-lipo", str(archive), "-thin", _gpu_arch(archive), "-output", str(thin)],
        capture_output=True,
        check=True,
    )
    nested.write_bytes(_section(thin, None))
    metadata = _section(nested, "__GPU_METADATA")

    root = struct.unpack_from("<I", metadata, 0)[0]
    outer = dict(_table_fields(metadata, root))
    if 0 not in outer:
        raise RuntimeError("unexpected metadata layout: no field 0 on the root table")
    inner_table = outer[0] + struct.unpack_from("<I", metadata, outer[0])[0]
    inner = dict(_table_fields(metadata, inner_table))
    if 0 not in inner:
        raise RuntimeError("unexpected metadata layout: no field 0 on the nested table")

    registers = metadata[inner[0]]
    return {
        "registers": registers,
        "text_bytes": len(_section(nested, None, "__text")),
        "spilling": registers >= REGISTER_BUDGET,
    }


def _pressure_kernel(accumulators):
    """A kernel holding `accumulators` float4 values live across a loop."""
    declare = "\n    ".join(f"float4 acc{i} = float4(x[gid + {i}]);" for i in range(accumulators))
    update = "\n        ".join(
        f"acc{i} = fma(acc{i}, v, acc{(i + 1) % accumulators});" for i in range(accumulators)
    )
    total = " + ".join(f"acc{i}" for i in range(accumulators))
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {{
    {declare}
    for (uint i = 0; i < n; ++i) {{
        float4 v = float4(x[i]);
        {update}
    }}
    float4 total = {total};
    out[gid] = total.x + total.y + total.z + total.w;
}}
"""


def self_check(workdir):
    """Register counts for known kernels, so a bad read is obvious rather than plausible."""
    print(f"{'live floats':>12}{'expected':>10}{'read':>7}")
    ok = True
    for accumulators, expected in ((2, 12), (8, 36), (24, 100), (30, 124)):
        result = inspect(_pressure_kernel(accumulators), "probe", workdir)
        good = result["registers"] == expected
        ok = ok and good
        print(
            f"{accumulators * 4:>12}{expected:>10}{result['registers']:>7}"
            f"{'  ok' if good else '  MISMATCH'}"
        )
    return 0 if ok else 1


def audit_dense_swiglu(workdir, reduction=1536, output_features=8960):
    """Check the accumulator bound really keeps admitted kernels clear of the budget."""
    from metile.backends import mlx_dense_swiglu as backend
    from metile.codegen.msl_emitter import emit
    from metile.compiler.dense import lower_dense_swiglu_qmv

    print(f"dense SwiGLU QMV {reduction} -> {output_features}, budget {REGISTER_BUDGET}")
    print(f"{'rows':>5}{'configs':>9}{'max registers':>15}{'% of budget':>13}")
    worst = 0
    for rows in (1, 2, 4, 8, 16):
        configs = [
            config
            for config in backend._candidate_configs(
                rows, reduction, output_features, paired_available=True
            )
            if config.algorithm == "metile" and config.implementation.startswith("simdgroup")
        ]
        counts = []
        for config in configs:
            name = (
                f"audit_{rows}_{config.outputs_per_simdgroup}"
                f"_{config.simdgroups_per_threadgroup}_{config.k_unroll}"
            )
            metal_ir = lower_dense_swiglu_qmv(
                name,
                output_features,
                reduction,
                outputs_per_simdgroup=config.outputs_per_simdgroup,
                simdgroups_per_threadgroup=config.simdgroups_per_threadgroup,
                interleaved=True,
                k_unroll=config.k_unroll,
                rows=rows,
            )
            source = backend._specialize_mlx_source(emit(metal_ir), "bfloat16")
            try:
                counts.append(inspect(source, name, workdir)["registers"])
            except RuntimeError:
                continue
        peak = max(counts) if counts else 0
        worst = max(worst, peak)
        print(f"{rows:>5}{len(configs):>9}{peak:>15}{peak / REGISTER_BUDGET * 100:>12.0f}%")
    print(f"\nworst admitted kernel uses {worst} of {REGISTER_BUDGET} registers")
    print("spilling" if worst >= REGISTER_BUDGET else "no admitted kernel spills")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path(".metile-agx"))
    arguments = parser.parse_args()
    try:
        if arguments.self_check:
            return self_check(arguments.workdir)
        return audit_dense_swiglu(arguments.workdir)
    except Unavailable as error:
        print(f"unavailable: {error}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
