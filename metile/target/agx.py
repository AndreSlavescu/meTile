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

# Read bandwidth as a function of working-set size, measured by benchmarks/agx_memory_hierarchy.py.
# One number for bandwidth is badly wrong here: a working set that stays resident is served sixteen to
# twenty times faster than one that streams, and that ratio is larger than every other factor in this
# file. A tiling that fits and a tiling that misses are not the same kernel.
#
# Coalesced streaming reads, 16384 threads, six gigabytes of traffic per dispatch, sizes interleaved
# across rounds. The pass loop re-reads the same addresses, so the obvious worry is that the backend
# collapses it into a multiply and the fast numbers are fiction; tripling the traffic triples the
# elapsed time at every size, which it could not if the loop were collapsed.
#
# The resident regime is one entry, not four, because the probe cannot resolve it. Measured 1545, 2006,
# 2138 and 2386 GB/s at 256 KB, 512 KB, 1 MB and 2 MB -- bandwidth *rising* with working set, which
# cannot be a property of a cache. It is the pass loop: a smaller working set means fewer inner
# iterations per pass, so loop bookkeeping takes a larger share, and the effect shrinks as the set
# grows. Only the 2 MB figure is close to uncontaminated, and it is a floor.
#
# So what this establishes is that a resident working set runs at 2386 GB/s or better, and nothing about
# how that varies below 2 MB. Reporting four numbers would claim a resolution the measurement does not
# have; a monotonicity test on the table is what caught the attempt.
BANDWIDTH_BY_WORKING_SET_GBPS = {
    2 * 1024 * 1024: 2386.0,
    4 * 1024 * 1024: 555.0,
    8 * 1024 * 1024: 192.0,
    16 * 1024 * 1024: 161.0,
    32 * 1024 * 1024: 134.0,
    64 * 1024 * 1024: 128.0,
    128 * 1024 * 1024: 124.0,
}

# Largest working set still served by the fast level, and what it delivers. The knee is sharp: 2 MB
# reads 2386 GB/s and 4 MB reads 555, so this is the number a tiling pass should be trying to stay
# under. The rate is a floor for the reason above.
RESIDENT_WORKING_SET_BYTES = 2 * 1024 * 1024
RESIDENT_READ_GBPS = 2386.0

# Threadgroup memory, measured by benchmarks/agx_threadgroup_bandwidth.py against a resident device read
# over the same 32 KB, so this compares staging with a cache hit rather than with DRAM.
#
# It is faster, but only just, and only when read contiguously: 3361 GB/s against the device arm's 2749,
# so 1.22x. That is a far smaller margin than the usual assumption about scratchpad memory, and it means
# a pass cannot justify staging on bandwidth alone.
#
# What it buys in peak it gives back in sensitivity. Across strides the threadgroup arm spreads 7.69x
# and the device arm 1.80x, so threadgroup memory is the more fragile of the two -- the opposite of the
# habit of treating shared memory as forgiving scratch and device memory as the thing needing careful
# access.
THREADGROUP_PEAK_GBPS = 3361.0
THREADGROUP_OVER_RESIDENT = 1.22

# GB/s by per-lane stride in bytes, contiguous first. The collapses are the power-of-two strides from 128
# bytes up; 144 bytes reads 2322 while 128 reads 1216, which is bank aliasing and not a size effect.
THREADGROUP_GBPS_BY_STRIDE = {
    16: 3361.0,
    32: 2658.0,
    48: 2978.0,
    64: 2032.0,
    80: 2737.0,
    96: 2510.0,
    112: 2468.0,
    128: 1216.0,
    144: 2322.0,
    192: 2032.0,
    256: 605.0,
    512: 437.0,
}

# Smallest per-lane stride at which a power of two collapses threadgroup bandwidth. 32 banks of four
# bytes, so 128 bytes puts every lane on the same bank.
THREADGROUP_CONFLICT_STRIDE_BYTES = 128


class Unavailable(RuntimeError):
    """The toolchain needed to inspect compiled kernels is not present."""


def ilp_headroom(dtype="f32"):
    """How much a perfect scheduler could win for this element type, at most."""
    return ILP_CEILING.get(dtype, 1.0)


def spills(registers):
    """Whether a kernel using this many registers is spilling."""
    return registers >= REGISTER_BUDGET


def read_bandwidth_gbps(working_set_bytes):
    """Expected read bandwidth for a working set of this size, in GB/s.

    For a pass deciding a tile size. Interpolating between measured points would invent a smooth curve
    the hardware does not have -- the drop from 2 MB to 4 MB is a factor of four -- so this reports the
    measurement for the smallest size at least as large as the request, which is the conservative
    direction: a tile is served no faster than the next size up was measured at.
    """
    if working_set_bytes <= 0:
        raise ValueError("a working set must be positive")
    for size in sorted(BANDWIDTH_BY_WORKING_SET_GBPS):
        if working_set_bytes <= size:
            return BANDWIDTH_BY_WORKING_SET_GBPS[size]
    return STREAMING_READ_GBPS


def resident(working_set_bytes):
    """Whether a working set of this size is served by the fast level rather than by DRAM."""
    return 0 < working_set_bytes <= RESIDENT_WORKING_SET_BYTES


def threadgroup_conflicts(stride_bytes):
    """Whether this per-lane stride puts threadgroup memory into bank conflict.

    Power-of-two strides from 128 bytes collapse it: 128 reads 1216 GB/s against 3361 contiguous, 256
    reads 605 and 512 reads 437, while 144 bytes -- one vector larger than 128 -- reads 2322. Device
    memory shows nothing comparable, so this is a hazard staging introduces rather than one it avoids.

    `metile.compiler.passes._optimal_pad` already pads to an odd stride, which this confirms is the
    right direction; the docstring's reasoning about 32 four-byte banks was never measured until now.
    """
    if stride_bytes <= 0:
        raise ValueError("a stride must be positive")
    power_of_two = stride_bytes & (stride_bytes - 1) == 0
    return power_of_two and stride_bytes >= THREADGROUP_CONFLICT_STRIDE_BYTES


def tiling_gain(working_set_bytes):
    """How much bandwidth a tiling wins by fitting this working set instead of streaming.

    The figure worth putting beside the other ratios in this file. Fitting under 2 MB is worth about
    19x, where choosing the matrix unit over scalar is worth 2.4x to 3.7x and instruction scheduling is
    worth at most 1.09x and unreachable in practice.

    Available to a pass only where there is reuse, which is the part worth checking before reaching for
    it. Neither of meTile's two main regimes has any:

      decode    each weight element is read exactly once, so the working set is the whole weight and no
                tiling changes that. Real MLP weights run 2.5 MB to 50 MB, all above the knee, and the
                chosen configs achieve 80 to 128 GB/s against their footprint's level of 128 to 555.
      prefill   compute bound, not memory bound. The generated kernels reach 0.96x to 0.97x of
                MATRIX_PEAK_TFLOPS, and MLX reaches 0.95x to 0.96x, so there is nothing for a tiling to
                recover.

    So this is a real property of the part that the current kernels cannot exploit. It becomes reachable
    if a kernel is restructured to reuse a resident tile across more work than it does today.
    """
    return read_bandwidth_gbps(working_set_bytes) / STREAMING_READ_GBPS


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
        message = built.stderr.strip()
        # A device that will not serialize a binary archive cannot be inspected at all, which is a
        # property of the machine and not a fault in the kernel. Reporting it as Unavailable, the
        # way a missing swiftc is reported, lets callers skip; raising RuntimeError made every
        # machine-code test fail on a CI runner rather than opt out of a capability it lacks.
        if "MTLBinaryArchive" in message or "eligible to be serialized" in message:
            raise Unavailable(f"this device does not serialize binary archives: {message[:200]}")
        raise RuntimeError(message[:300])

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
