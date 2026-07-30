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

    registers      the register index appears twice in the compact form, as byte 0's high
                   nibble and as `(r << 1) | 1` in byte 1, and the two agreed across every
                   instruction examined. Confirmed by redirecting an instruction onto another
                   chain's register and predicting the whole kernel's output: three redirects,
                   three exact matches.

    assembly       with registers, constants and flags all measured, `encode_fma` assembles the
                   form from scratch. It reproduces the compiler's own bytes exactly for the
                   cases the compiler emits, and four synthesised forms the compiler never
                   emitted — a*3+0.5, a*1.5-2, a*7 with no addend, and -a*2+1 — each ran exactly
                   as predicted on four inputs.

    flags          three bits of the compact fma control its arithmetic, each found by scanning
                   all 256 values of its byte and then predicting the result of setting it
                   across a chain on four inputs. Twelve predictions, all exact: byte 2 bit
                   0x10 negates the product, byte 4 bit 0x10 negates the addend, byte 4 bit
                   0x20 drops the addend entirely and leaves a multiply. So a compiled fma can
                   be rewritten into `-a*m+d`, `a*m-d` or `a*m` without recompiling it.

    structure      `06 00` is a two-byte nop and `0e 00 00 00` ends a block. Blocks are padded
                   with nops to a 64-byte boundary. Giving an eight-byte instruction the nop's
                   opcode nibble is the one edit that makes the driver reject the kernel rather
                   than return a wrong answer, which is what desynchronising the stream would do
                   — but that does not generalise into a length field, see below.

Deliberately not claimed: a general disassembler, and there is now evidence for why rather than
just caution. Instruction length looked like the low nibble of byte 0 — nop 0x06 at two bytes,
block end 0x0e at four, fma 0x09 at eight — and a table fitted to one kernel walked it exactly.
The same table then walked none of eight other kernels exactly. Lengths here come one form at a
time from the behavioural finder in `boundaries`, which cannot be fooled that way.

Negative immediates likewise have no known encoding in this field, and `encode_immediate` refuses
them rather than returning the nearest byte, which would corrupt a kernel silently.

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
#   0  opcode. Low nibble 9. Of all 256 values only the four with nibble 6 are rejected
#      outright; the rest run and return a wrong answer.
#   1  register selection. Every alternative tried sent the chain's result somewhere the
#      final store did not read.
#   2  operand mode and sign. Bit 0x10 negates the product. Bit 0x20 marks all but the last
#      instruction of a run, so the compiler emits 0x2e throughout and 0x0e at the end. Bit
#      0x01 turned `a*2+1` into `a*a`, producing 225 from an accumulator holding 15.
#   3  multiplier immediate, in the format above.
#   4  addend control. Bit 0x20 includes the addend, and clearing it leaves a plain multiply.
#      Bit 0x10 negates it.
#   5  addend immediate, same format with the low bit clear.
#   6  bit 0x20 retires the instruction. Bits 0x40 and 0x80 change where the result goes; the
#      low four bits made no difference to the result at all.
#   7  register selection, like byte 1.
FMA_LENGTH = 8
FMA_OPCODE_NIBBLE = 0x09
FMA_MULTIPLIER_BYTE = 3
FMA_ADDEND_BYTE = 5

# Instruction length is NOT a function of the low nibble of byte 0, and this is the one place a
# plausible shortcut was tried and failed, so it is recorded to stop it being tried again.
#
# The nibble looks like a length field from one kernel: nop is 0x06 and two bytes, the block
# terminator 0x0e and four, the compact fma 0x09 and eight, and setting an eight-byte
# instruction's nibble to 6 is the only edit the driver rejects outright rather than running with
# a wrong answer, which is what desynchronising the stream would do. A table extended to fit one
# kernel walked it end to end and covered all four behaviourally confirmed fma boundaries.
#
# It then walked none of eight other kernels exactly: copy, two fma chains, a reduction loop,
# integer and half-precision mixes, a branch, and a sqrt. Five overran the end of the stream and
# the rest left unknown nibbles behind. Fitting sixteen free values to one 134-byte kernel simply
# is not evidence. Lengths here come from the behavioural finder in `boundaries` instead, one form
# at a time.


class FmaFlag:
    """One bit of a compact fma whose meaning was established by patching and running.

    Each was found by scanning all 256 values of its byte, grouping the outputs by what
    arithmetic they expressed, and then predicting the result of setting the bit across a chain
    of instructions on four different inputs. All twelve predictions were exact.
    """

    def __init__(self, byte, mask, meaning, set_means):
        self.byte = byte
        self.mask = mask
        self.meaning = meaning
        self.set_means = set_means

    def __repr__(self):
        return (
            f"FmaFlag(byte={self.byte}, mask=0x{self.mask:02x}, "
            f"{self.meaning}, set gives {self.set_means})"
        )


# fma(a, m, d) with every flag clear computes a * m + d.
PRODUCT_NEGATE = FmaFlag(2, 0x10, "negate the product", "-a*m + d")
ADDEND_NEGATE = FmaFlag(4, 0x10, "negate the addend", "a*m - d")
# Named for what it selects, after an earlier name got it wrong. It was called ADDEND_ENABLE and
# described as including the addend or not, because clearing it turned a*m+d into a*m. That reading
# survived a prediction on four inputs and was still incomplete: clearing it switches the addend
# slot from an immediate to a *register*, and the byte left in the slot happened to name register
# 88, outside the sixteen the field can reach, which reads zero. The addend was never absent, it
# was zero.
ADDEND_IMMEDIATE = FmaFlag(
    4, 0x20, "the addend slot holds an immediate", "immediate; clear means it names a register"
)
INSTRUCTION_DISABLE = FmaFlag(6, 0x20, "retire the instruction", "no effect, like a nop")
# Not exported: setting 0x01 in byte 2 made `a*2+1` compute `a*a`, giving 225 from an accumulator
# holding 15, so some operand slot is being redirected to the accumulator. Which one was never
# pinned down and it was never checked across several inputs, so it stays a note. Naming it would
# put it on the same footing as the flags above, which were each predicted on four inputs.


# Register selection, established across three independent fma chains and then confirmed by
# redirecting an instruction onto another chain's register and predicting the whole kernel's
# output. Three redirects, three exact matches. The index appears twice: as byte 0's high nibble
# and as `(r << 1) | 1` in byte 1, and the two agreed in all eight instructions examined.
FMA_REGISTER_HIGH_BYTE = 0
FMA_REGISTER_BYTE = 1
FMA_MAX_REGISTER = 15

# The addend slot uses the same shape when it names a register: `r << 1`, low bit ignored. Verified
# by rewriting instructions to `rd = rd * m + rs` for every ordered pair of three live registers at
# two multipliers, predicting all four threads each: eighteen rewrites, seventy-two exact values.
# This is the register-plus-register add, reached as `rd * 1 + rs`.
#
# Indices at or above sixteen read zero, which is what makes a zero addend expressible at all,
# since the immediate field's smallest value is 2**-11 and it has no encoding for zero.
ARCHITECTURAL_REGISTERS = 16
_ZERO_REGISTER_SLOT = 0x58  # names register 44, above the sixteen reachable, so it reads zero

# The remaining constant bytes of the form, taken as the compiler writes them. Byte 7 was 0x22 or
# 0x42 on the first fma of a chain and 0x02 everywhere else, which looks like a dependency or wait
# field; 0x02 is what a synthesised instruction uses, and synthesised instructions run correctly
# with it.
_FMA_ADDEND_CONTROL = 0x21
_FMA_TAIL = bytes((0x02, 0x02))
_FMA_NOT_LAST = 0x20
_FMA_MODE_BASE = 0x0E


def encode_fma(
    register, multiplier, addend=None, last=False, negate_product=False, addend_register=None
):
    """Assemble a complete compact fma: `register = register * multiplier (+/- addend)`.

    Every field comes from a measurement that was checked by prediction, so this is an encoder
    rather than a template with holes: pass the register, the constants and the signs and the
    eight bytes come out. Reproduces the compiler's own encoding exactly for the cases it emits,
    which is the cheapest available check that the assembly is right.

    The addend has three forms. A float is an immediate; a negative one is encoded by setting the
    negate flag, which is how the slot reaches values its unsigned field cannot. `addend_register`
    names a register instead, which is how `rd * 1 + rs` becomes a register-plus-register add.
    Passing neither leaves a zero addend, using a register index above the sixteen the field can
    reach because those read zero and the immediate field has no encoding for zero.
    """
    if addend is not None and addend_register is not None:
        raise EncodingError("an addend is either an immediate or a register, not both")
    if not 0 <= register <= FMA_MAX_REGISTER:
        raise EncodingError(f"register {register} is outside the field")
    mode = _FMA_MODE_BASE
    if not last:
        mode |= _FMA_NOT_LAST
    if negate_product:
        mode |= PRODUCT_NEGATE.mask
    # With the immediate bit clear the slot names a register, so what goes in it is a register
    # index and not a leftover. Writing 0x00 there was tried first and the kernel computed a*m + a,
    # eight-fold growth per step where seven was predicted, because index zero was the accumulator.
    control = _FMA_ADDEND_CONTROL & ~ADDEND_IMMEDIATE.mask
    addend_byte = _ZERO_REGISTER_SLOT
    if addend is not None:
        control = _FMA_ADDEND_CONTROL
        if addend < 0:
            control |= ADDEND_NEGATE.mask
        addend_byte = encode_immediate(abs(addend), low_bit=0)
    elif addend_register is not None:
        if not 0 <= addend_register < ARCHITECTURAL_REGISTERS:
            raise EncodingError(f"addend register {addend_register} is outside the field")
        addend_byte = addend_register << 1
    return bytes(
        (
            (register << 4) | FMA_OPCODE_NIBBLE,
            (register << 1) | 1,
            mode,
            encode_immediate(multiplier, low_bit=1),
            control,
            addend_byte,
            *_FMA_TAIL,
        )
    )


def read_flag(text, offset, flag):
    """Whether one flag is set on the instruction at `offset`."""
    return bool(text[offset + flag.byte] & flag.mask)


def write_flag(text, offset, flag, value):
    """Return `text` with one flag of the instruction at `offset` set or cleared."""
    patched = bytearray(text)
    if value:
        patched[offset + flag.byte] |= flag.mask
    else:
        patched[offset + flag.byte] &= ~flag.mask & 0xFF
    return bytes(patched)


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
