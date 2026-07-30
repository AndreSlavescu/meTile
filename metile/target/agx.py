"""What meTile knows about Apple GPU hardware, measured rather than assumed.

Every number here was measured on the machine and each one changed a compiler decision or
stopped one from being made. They live in the compiler because they are properties of the
target, not of any benchmark: a pass that wants to know whether a schedule can spill, or
whether reordering instructions could possibly pay, should ask here.

Provenance matters as much as the values, so each is recorded with how it was obtained and
what it ruled in or out. Re-measure with `benchmarks/agx_registers.py` and
`benchmarks/agx_ilp_ceiling.py` on new hardware; nothing below is derived from a datasheet.
"""

import re
import shutil
import struct
import subprocess
from pathlib import Path

PROBE_SOURCE = Path(__file__).resolve().parent / "agx_probe.swift"

# Per-thread register budget. Found by growing a kernel's live values until the reported
# count stopped rising, then confirming that kernels reaching it spill: at 124 live floats
# the count reads 124 and does not spill, at 128 it reads 140 and does. Kernels that reach
# it measured 1.3x to 6.7x slower than lower-register siblings.
REGISTER_BUDGET = 140

# The most any instruction scheduling could win, from dependent-fma chains replicated into
# independent chains. A single dependent chain already reaches 92% of fp32 peak, because the
# GPU covers latency with thread-level parallelism rather than ILP inside a thread. This is
# why scheduling work is not where compiler effort belongs on this target, and why five
# scheduling experiments on the int4 QMV all measured flat.
ILP_CEILING = {"f32": 1.09, "f16": 1.41}

# Throughput ceilings, all measured. The gap between scalar and matrix is the reason
# functional-unit selection outranks scheduling by more than twenty to one.
SCALAR_PEAK_TFLOPS = {"f32": 4.1, "f16": 6.5}
MATRIX_PEAK_TFLOPS = 15.33
STREAMING_READ_GBPS = 120.6


class Unavailable(RuntimeError):
    """The toolchain needed to inspect compiled kernels is not present."""


def ilp_headroom(dtype="f32"):
    """How much a perfect scheduler could win for this element type, at most."""
    return ILP_CEILING.get(dtype, 1.0)


def spills(registers):
    """Whether a kernel using this many registers is spilling."""
    return registers >= REGISTER_BUDGET


def _harness(workdir):
    """Build the Metal harness once per working directory."""
    binary = workdir / "agx_probe"
    if binary.exists():
        return binary
    if shutil.which("swiftc") is None:
        raise Unavailable("swiftc not found; Xcode command line tools are required")
    workdir.mkdir(parents=True, exist_ok=True)
    built = subprocess.run(
        ["swiftc", "-O", str(PROBE_SOURCE), "-o", str(binary)], capture_output=True, text=True
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


def _compiled(source, function, workdir):
    """Compile one MSL kernel and unwrap to the GPU Mach-O inside the binary archive.

    Three unwraps. An MTLBinaryArchive serializes to a fat file whose applegpu_* slice is the
    GPU code; that slice's __compute section is itself a Mach-O; inside it __GPU_METADATA is a
    FlatBuffer and __text is the machine code.

    Costs a Metal compile per call, on the order of a second, so this is for offline analysis
    and never for a dispatch path.
    """
    workdir = Path(workdir)
    binary = _harness(workdir)
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
    return nested


def machine_code(source, function, workdir=".metile-agx"):
    """The kernel's __text bytes: the instructions the GPU will actually run.

    This is what makes a claim about code generation checkable instead of arguable. Two source
    forms that compile to identical bytes cannot differ in speed, which turns questions the
    timing harness answers badly into questions with a yes or no answer.

    Used that way it established that Apple's backend normalises statement order completely:
    two independent fma chains written serially and written interleaved produce byte-identical
    machine code, as do a load placed at its use and the same load hoisted. That is why
    meTile's own scheduling pass is off by default. See benchmarks/agx_source_order.py.
    """
    return _section(_compiled(source, function, workdir), None, "__text")


def inspect(source, function, workdir=".metile-agx"):
    """Return {'registers', 'text_bytes', 'spilling'} for one MSL kernel.

    Metal exposes no register count, and the obvious stand-in does not work:
    maxTotalThreadsPerThreadgroup reads 1024 whether a kernel holds 4 live floats or 512. The
    compiler does record the number, inside the __GPU_METADATA FlatBuffer.

    The count is field 0 of the table referenced by field 0 of the FlatBuffer root. Read by
    path, never by byte offset: the buffer embeds the kernel name and signature, so a fixed
    offset drifts between kernels. Reading byte 188 worked for one probe kernel and silently
    reported a neighbouring field, 24 registers and "spilling", for every real one.
    """
    nested = _compiled(source, function, workdir)
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
        "spilling": spills(registers),
    }
