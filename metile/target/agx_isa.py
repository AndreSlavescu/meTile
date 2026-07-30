"""What meTile has established about G17 machine code, and the method that established it.

The instruction set is undocumented, so everything here was measured, and the method matters as
much as the results. Reading bytes and guessing produces plausible field maps that are wrong.
Every claim below was instead put to the GPU: patch the machine code inside a binary archive,
run it, and check the answer against a prediction made in advance. A hypothesis that survives
that is not an interpretation.

Established, in descending order of confidence:

    execution      A binary archive whose __text has been edited runs the edited code. The
                   driver does not re-derive it from the AIR it also carries, and does not
                   validate it. `execute` below is the primitive everything else rests on.

    boundaries     Overwriting a byte range with nops and running the result finds instruction
                   boundaries behaviourally. On a chain of four `a = fma(a, 2, 1)` steps, only
                   the offsets 0x62, 0x6a and 0x72 yield 15 rather than 31, so exactly one fma
                   was removed at each and a compact f32 fma is eight bytes on an eight-byte
                   stride. Every other alignment gives 0, 1, 7 or a rejected kernel.

    immediates     The eight-bit float operand field is `(e << 4) | (m << 1) | low`, holding
                   `(1 + m/8) * 2**(e - 11)`. Round-trips against every constant the compiler
                   was observed to emit, and — the part that makes it an encoder rather than a
                   table — correctly predicts results for constants the compiler never emitted:
                   rewriting three fmas to `a*6+7` gives 949 from x=1, and to `a*1.25+1.5`
                   gives 11.578125. Both were predicted before running.

    structure      `06 00` is a two-byte nop and `0e 00 00 00` ends a block. Blocks are padded
                   with nops to a 64-byte boundary.

Deliberately not claimed: a general disassembler. One instruction form is mapped, the operand
fields of the rest are not, and `decode` says so rather than inventing mnemonics. Negative
immediates have no known encoding in this field; `encode_immediate` refuses them instead of
producing something untested.

`benchmarks/agx_isa_probe.py` re-derives all of it from scratch, which is how to port this to
new hardware.
"""

import math
import struct
import subprocess
from pathlib import Path

from metile.target import agx

# A two-byte instruction with no effect, and a four-byte one that ends a block. Both read
# straight off the padding the compiler emits between blocks; both confirmed by patching them
# over real instructions and seeing exactly that instruction's contribution disappear.
NOP = bytes.fromhex("0600")
BLOCK_END = bytes.fromhex("0e000000")
BLOCK_ALIGNMENT = 64

# The compact f32 fma. Byte roles, each established by patching that byte and running:
#
#   0  opcode. Low nibble 9. Any other value produced garbage or a dead kernel.
#   1  register selection. Every alternative tried sent the chain's result somewhere the
#      final store did not read.
#   2  operand mode, plus the flag distinguishing the last instruction of a run: the compiler
#      emits 0x2e throughout and 0x0e on the final one. 0x03 turned `a*2+1` into `a*a`,
#      producing 225 from an accumulator holding 15.
#   3  multiplier immediate, in the format above.
#   4  not probed.
#   5  addend immediate, same format with the low bit clear.
#   6  flags. Bit 0x20 disables the instruction; the low bits made no difference to the result.
#   7  register selection, like byte 1.
FMA_LENGTH = 8
FMA_OPCODE_NIBBLE = 0x09
FMA_MULTIPLIER_BYTE = 3
FMA_ADDEND_BYTE = 5

_IMMEDIATE_BIAS = 11
_IMMEDIATE_MANTISSA_STEPS = 8


class EncodingError(ValueError):
    """A value has no encoding in this field, or none that has been verified."""


def encode_immediate(value, low_bit=1):
    """Encode a positive float into the eight-bit operand field.

    `low_bit` is the bit the field shares with its neighbour: set for the multiplier slot,
    clear for the addend slot, both as the compiler emits them. It is passed rather than
    inferred because what that bit belongs to has not been established, and guessing it would
    silently corrupt the adjacent field.
    """
    if value <= 0 or not math.isfinite(value):
        raise EncodingError(
            f"{value} has no verified encoding: the field carries no sign and no special values"
        )
    exponent = math.floor(math.log2(value))
    mantissa = round((value / 2.0**exponent - 1.0) * _IMMEDIATE_MANTISSA_STEPS)
    if mantissa == _IMMEDIATE_MANTISSA_STEPS:  # rounded up to the next power of two
        exponent, mantissa = exponent + 1, 0
    biased = exponent + _IMMEDIATE_BIAS
    if not 0 <= biased <= 15:
        raise EncodingError(f"{value} is outside the field's exponent range")
    if decode_immediate((biased << 4) | (mantissa << 1) | low_bit) != value:
        raise EncodingError(f"{value} is not exactly representable in this field")
    return (biased << 4) | (mantissa << 1) | low_bit


def decode_immediate(byte):
    """The float an operand byte stands for."""
    exponent = (byte >> 4) - _IMMEDIATE_BIAS
    mantissa = (byte >> 1) & 0x07
    return (1.0 + mantissa / _IMMEDIATE_MANTISSA_STEPS) * 2.0**exponent


def find_fma(text):
    """Offsets of compact f32 fma instructions, as a candidate list.

    Pattern matching on an undocumented encoding, so treat these as candidates: confirm one by
    nopping it and checking the arithmetic changed the way removing that operation would.
    `boundaries` does exactly that.
    """
    found = []
    offset = 0
    while offset + FMA_LENGTH <= len(text):
        if text[offset] & 0x0F == FMA_OPCODE_NIBBLE and text[offset + 2] & 0x0F in (0x0E,):
            found.append(offset)
            offset += FMA_LENGTH
        else:
            offset += 2
    return found


def rewrite_fma_immediates(text, offset, multiplier=None, addend=None):
    """Return `text` with one fma's constants replaced.

    Specialises a compiled kernel's constants without recompiling it. The instruction keeps its
    length, so nothing downstream shifts.
    """
    patched = bytearray(text)
    if multiplier is not None:
        patched[offset + FMA_MULTIPLIER_BYTE] = encode_immediate(multiplier, low_bit=1)
    if addend is not None:
        patched[offset + FMA_ADDEND_BYTE] = encode_immediate(addend, low_bit=0)
    return bytes(patched)


def _harness(workdir):
    """Build the executor once per working directory."""
    binary = Path(workdir) / "agx_execute"
    if binary.exists():
        return binary
    source = Path(__file__).resolve().parent / "agx_execute.swift"
    Path(workdir).mkdir(parents=True, exist_ok=True)
    built = subprocess.run(
        ["swiftc", "-O", str(source), "-o", str(binary)], capture_output=True, text=True
    )
    if built.returncode != 0:
        raise agx.Unavailable(f"could not build the executor: {built.stderr.strip()[:300]}")
    return binary


def execute(source, function, inputs, rewrite=None, workdir=".metile-agx"):
    """Compile, optionally rewrite the machine code, run, and return the outputs.

    `rewrite` takes the kernel's __text and returns replacement bytes of the same length. It is
    applied in place inside the serialized archive, and the archive is then the only source the
    driver is allowed to use, so what runs is what was written.

    Same length is required rather than merely advised. The bytes are patched at the offset
    where they were found in the archive file, and a different length would shift everything
    after them, including the metadata the driver reads to set the kernel up.
    """
    workdir = Path(workdir)
    prober = agx._harness(workdir)
    executor = _harness(workdir)
    metal = workdir / "isa.metal"
    archive = workdir / "isa.bin"
    metal.write_text(source)
    built = subprocess.run(
        [str(prober), str(metal), function, str(archive)], capture_output=True, text=True
    )
    if built.returncode != 0:
        raise RuntimeError(built.stderr.strip()[:300])

    raw = archive.read_bytes()
    target = archive
    if rewrite is not None:
        text = agx.machine_code(source, function, workdir)
        replacement = rewrite(text)
        if len(replacement) != len(text):
            raise EncodingError(
                f"a rewrite must keep the length: {len(replacement)} bytes for {len(text)}"
            )
        offset = raw.find(bytes(text))
        if offset < 0 or raw.count(bytes(text)) != 1:
            raise RuntimeError("could not locate __text uniquely inside the archive")
        target = workdir / "isa_patched.bin"
        target.write_bytes(raw[:offset] + replacement + raw[offset + len(text) :])

    values = list(inputs)
    inputs_file, outputs_file = workdir / "isa_in.f32", workdir / "isa_out.f32"
    inputs_file.write_bytes(struct.pack(f"<{len(values)}f", *values))
    ran = subprocess.run(
        [
            str(executor),
            str(target),
            str(metal),
            function,
            str(inputs_file),
            str(outputs_file),
            str(len(values)),
        ],
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0:
        raise RuntimeError(ran.stderr.strip().splitlines()[-1][:200] if ran.stderr else "no output")
    produced = outputs_file.read_bytes()[: 4 * len(values)]
    return list(struct.unpack(f"<{len(values)}f", produced))


def boundaries(source, function, region, inputs, intact, removed, workdir=".metile-agx"):
    """Offsets in `region` where nopping `stride` bytes removes exactly one operation.

    The behavioural instruction finder. `intact` is what the kernel returns untouched and
    `removed` is what it returns with one of the operations gone, both worked out from the
    kernel's arithmetic beforehand. An offset qualifies only when nopping there produces
    `removed` exactly, which is a far stronger signal than the kernel merely still running:
    wrong alignments do run, and return wrong answers rather than failing.
    """
    text = agx.machine_code(source, function, workdir)
    if execute(source, function, inputs, workdir=workdir)[0] != intact:
        raise RuntimeError("the unpatched kernel does not return the expected value")

    start, stop, stride = region
    found = []
    for offset in range(start, stop, 2):

        def rewrite(original, at=offset):
            patched = bytearray(original)
            patched[at : at + stride] = NOP * (stride // len(NOP))
            return bytes(patched)

        try:
            got = execute(source, function, inputs, rewrite=rewrite, workdir=workdir)
        except RuntimeError:
            continue
        if got and got[0] == removed:
            found.append(offset)
    return found, text
