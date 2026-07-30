"""The parts of the G17 encoding meTile claims to understand.

Two kinds of test. The immediate format is arithmetic and is checked directly against the bytes
the Metal compiler was observed to emit, which needs no GPU. Whether an edited archive actually
runs the edited code is not arithmetic and cannot be argued, so that one compiles, patches,
dispatches, and compares against a number worked out beforehand.
"""

import itertools

import pytest

from metile.target import agx_isa

# Constants and the operand bytes the compiler emitted for them, read out of compiled kernels by
# benchmarks/agx_isa_probe.py. The multiplier slot carries a set low bit and the addend slot a
# clear one; what that bit belongs to is not established, which is why it is passed in rather
# than inferred.
OBSERVED_MULTIPLIERS = ((2.0, 0xC1), (3.0, 0xC9), (4.0, 0xD1), (8.0, 0xE1))
OBSERVED_ADDENDS = ((1.0, 0xB0), (3.0, 0xC8), (5.0, 0xD4))


@pytest.mark.parametrize(("value", "byte"), OBSERVED_MULTIPLIERS)
def test_the_multiplier_field_matches_what_the_compiler_emits(value, byte):
    assert agx_isa.encode_immediate(value, low_bit=1) == byte
    assert agx_isa.decode_immediate(byte) == value


@pytest.mark.parametrize(("value", "byte"), OBSERVED_ADDENDS)
def test_the_addend_field_matches_what_the_compiler_emits(value, byte):
    assert agx_isa.encode_immediate(value, low_bit=0) == byte
    assert agx_isa.decode_immediate(byte) == value


def test_every_representable_value_round_trips():
    """The format is (1 + m/8) * 2**(e - 11), so enumerate it and check both directions.

    Enumerating beats sampling here: the field is 128 values wide, so exhaustive is cheap, and a
    sampled test would miss an off-by-one at an exponent boundary.
    """
    for exponent in range(16):
        for mantissa in range(8):
            byte = (exponent << 4) | (mantissa << 1)
            value = agx_isa.decode_immediate(byte)
            assert agx_isa.encode_immediate(value, low_bit=0) == byte


def test_values_outside_the_field_are_refused_rather_than_approximated():
    """Silently encoding the nearest representable value would corrupt a kernel quietly.

    There is no sign bit in this field and no encoding for zero or the specials, and no exponent
    reaches 2**5. A caller asking for one of those has a wrong model of the field and should be
    told, not handed the closest byte.
    """
    for value in (-1.0, 0.0, float("inf"), float("nan"), 1.1, 2.0**6):
        with pytest.raises(agx_isa.EncodingError):
            agx_isa.encode_immediate(value)


def test_rewriting_immediates_keeps_the_instruction_length():
    """Length has to be preserved: the patch lands at a fixed offset inside the archive."""
    instruction = bytes.fromhex("0901 2ec1 21b0 0202")
    rewritten = agx_isa.rewrite_fma_immediates(instruction, 0, multiplier=6.0, addend=7.0)
    assert len(rewritten) == len(instruction)
    assert agx_isa.decode_immediate(rewritten[agx_isa.FMA_MULTIPLIER_BYTE]) == 6.0
    assert agx_isa.decode_immediate(rewritten[agx_isa.FMA_ADDEND_BYTE]) == 7.0


CHAIN = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    float a = x[gid];
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    out[gid] = a;
}
"""


def _machine_code():
    from metile.target import Unavailable, machine_code

    try:
        return machine_code(CHAIN, "probe")
    except Unavailable as error:
        pytest.skip(f"no Metal toolchain: {error}")


def test_an_edited_archive_runs_the_edited_code():
    """The premise the whole ISA effort rests on, so it is asserted rather than assumed.

    Four dependent fmas of a*2+1 turn x=1 into 31. Rewriting the constants of the compact ones
    to a*6+7 must give 949, and that number was computed from the arithmetic before the bytes
    were ever assembled. If the driver were recompiling from the AIR it also carries, or
    ignoring the archive, the answer would still be 31 and nothing here would work.
    """
    text = _machine_code()
    compact = [
        offset
        for offset in range(0, len(text) - agx_isa.FMA_LENGTH, 2)
        if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE
        and text[offset + 2] & 0x0F == 0x0E
        and text[offset + agx_isa.FMA_MULTIPLIER_BYTE] == 0xC1
    ]
    assert len(compact) == 3, f"expected three compact fmas, found {[hex(o) for o in compact]}"

    assert agx_isa.execute(CHAIN, "probe", [1.0, 2.0])[0] == 31.0

    def rewrite(original):
        patched = original
        for offset in compact:
            patched = agx_isa.rewrite_fma_immediates(patched, offset, 6.0, 7.0)
        return patched

    # 1*2+1 = 3, then three steps of a*6+7: 25, 157, 949.
    assert agx_isa.execute(CHAIN, "probe", [1.0, 2.0], rewrite=rewrite)[0] == 949.0


def test_a_rewrite_that_changes_length_is_refused():
    """Shifting the bytes after the patch would move the metadata the driver reads.

    `_machine_code` is called for its skip, not its value: a device that will not serialize a
    binary archive cannot run one either, and this test would otherwise fail there rather than
    opt out. That is exactly what happened on CI.
    """
    _machine_code()
    with pytest.raises(agx_isa.EncodingError, match="keep the length"):
        agx_isa.execute(CHAIN, "probe", [1.0], rewrite=lambda text: text[:-2])


def test_nopping_an_instruction_removes_exactly_its_effect():
    """The behavioural boundary finder, on the case that established the fma length.

    A wrong alignment does not fail; it runs and returns a wrong answer. That is why the test
    asserts the value is exactly what dropping one fma gives, and that the offsets found sit on
    an eight-byte stride, rather than merely checking the kernel survived.
    """
    text = _machine_code()
    offsets, _ = agx_isa.boundaries(
        CHAIN,
        "probe",
        (0x50, len(text) - 12, agx_isa.FMA_LENGTH),
        [1.0, 2.0],
        intact=31.0,
        removed=15.0,
    )
    assert len(offsets) == 4, f"expected four fmas, found {[hex(o) for o in offsets]}"
    assert {b - a for a, b in itertools.pairwise(offsets)} == {agx_isa.FMA_LENGTH}


def _compact_offsets(text):
    return [
        offset
        for offset in range(0, len(text) - agx_isa.FMA_LENGTH, 2)
        if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE
        and text[offset + 2] & 0x0F == 0x0E
        and text[offset + agx_isa.FMA_MULTIPLIER_BYTE] == 0xC1
    ]


@pytest.mark.parametrize(
    ("flag", "clear", "step"),
    (
        (agx_isa.PRODUCT_NEGATE, False, lambda v: -v * 2.0 + 1.0),
        (agx_isa.ADDEND_NEGATE, False, lambda v: v * 2.0 - 1.0),
        (agx_isa.ADDEND_IMMEDIATE, True, lambda v: v * 2.0),
    ),
)
def test_each_arithmetic_flag_does_what_it_claims(flag, clear, step):
    """Set the bit on every compact fma, then check the GPU against arithmetic, not a table.

    Four inputs rather than one. A single input can agree by coincidence -- negating the product
    and negating the addend both happen to move the result by an even amount -- and a flag that
    only holds for x=1 is not understood.
    """
    text = _machine_code()
    offsets = _compact_offsets(text)
    assert len(offsets) == 3

    def rewrite(original):
        patched = original
        for offset in offsets:
            patched = agx_isa.write_flag(patched, offset, flag, not clear)
        return patched

    inputs = [1.0, 2.0, 3.0, 5.0]
    predicted = []
    for value in inputs:
        running = value * 2.0 + 1.0  # the long-form fma, left alone
        for _ in offsets:
            running = step(running)
        predicted.append(running)

    assert agx_isa.execute(CHAIN, "probe", inputs, rewrite=rewrite) == predicted


def test_flags_read_back_the_way_they_were_written():
    instruction = bytes.fromhex("0901 2ec1 21b0 0202")
    assert not agx_isa.read_flag(instruction, 0, agx_isa.PRODUCT_NEGATE)
    assert agx_isa.read_flag(instruction, 0, agx_isa.ADDEND_IMMEDIATE)
    negated = agx_isa.write_flag(instruction, 0, agx_isa.PRODUCT_NEGATE, True)
    assert agx_isa.read_flag(negated, 0, agx_isa.PRODUCT_NEGATE)
    assert len(negated) == len(instruction)
    assert agx_isa.write_flag(negated, 0, agx_isa.PRODUCT_NEGATE, False) == instruction


def test_disabling_an_instruction_by_flag_matches_nopping_it():
    """The flag and the nop should be indistinguishable, and both equal dropping the operation.

    Worth asserting because it ties the two capabilities together: the behavioural boundary
    finder works by overwriting bytes, and this flag achieves the same effect without touching
    the instruction's length or its neighbours.
    """
    text = _machine_code()
    offsets = _compact_offsets(text)
    last = offsets[-1]

    def by_flag(original):
        return agx_isa.write_flag(original, last, agx_isa.INSTRUCTION_DISABLE, True)

    def by_nop(original):
        patched = bytearray(original)
        patched[last : last + agx_isa.FMA_LENGTH] = agx_isa.NOP * (
            agx_isa.FMA_LENGTH // len(agx_isa.NOP)
        )
        return bytes(patched)

    inputs = [1.0, 2.0]
    flagged = agx_isa.execute(CHAIN, "probe", inputs, rewrite=by_flag)
    nopped = agx_isa.execute(CHAIN, "probe", inputs, rewrite=by_nop)
    assert flagged == nopped
    assert flagged[0] == 15.0  # three fmas of a*2+1 from x=1 instead of four


# What the compiler emitted for `a = fma(a, 2.0f, 1.0f)` on register 0, read out of a compiled
# kernel: the same instruction twice with the continue flag set, and once with it clear because it
# ends the run.
COMPILER_FMA = bytes.fromhex("09012ec121b00202")
COMPILER_FMA_LAST = bytes.fromhex("09010ec121b00202")


def test_the_assembler_reproduces_the_compilers_own_bytes():
    """The cheapest check that assembly is right: agree with the only other assembler available."""
    assert agx_isa.encode_fma(0, 2.0, 1.0, last=False) == COMPILER_FMA
    assert agx_isa.encode_fma(0, 2.0, 1.0, last=True) == COMPILER_FMA_LAST
    # Registers 2 and 3 as the compiler wrote them for two other accumulators.
    assert agx_isa.encode_fma(2, 2.0, 1.0)[:2] == bytes((0x29, 0x05))
    assert agx_isa.encode_fma(3, 3.0, 1.0)[:2] == bytes((0x39, 0x07))


def test_the_assembler_refuses_a_register_outside_the_field():
    for register in (-1, 16, 255):
        with pytest.raises(agx_isa.EncodingError, match="register"):
            agx_isa.encode_fma(register, 2.0, 1.0)


def test_dropping_the_addend_does_not_write_zero_into_its_slot():
    """Zero in the immediate slot is not inert, and getting this wrong is silent.

    Writing 0x00 there produced a kernel that computed a*m + a: a synthesised `a*7` grew
    eight-fold per step. Zero selects a register operand rather than meaning "no operand", so the
    slot keeps an ordinary encoded constant and the control bit is what disables it.
    """
    built = agx_isa.encode_fma(0, 7.0, addend=None)
    assert built[agx_isa.FMA_ADDEND_BYTE] != 0x00
    assert not agx_isa.read_flag(built, 0, agx_isa.ADDEND_IMMEDIATE)
    assert agx_isa.read_flag(agx_isa.encode_fma(0, 7.0, 1.0), 0, agx_isa.ADDEND_IMMEDIATE)


def test_a_negative_addend_uses_the_negate_flag():
    """The immediate field is unsigned, so the sign has to live in the control byte."""
    built = agx_isa.encode_fma(0, 1.5, -2.0)
    assert agx_isa.read_flag(built, 0, agx_isa.ADDEND_NEGATE)
    assert agx_isa.decode_immediate(built[agx_isa.FMA_ADDEND_BYTE]) == 2.0


@pytest.mark.parametrize(
    ("fields", "step"),
    (
        ({"multiplier": 3.0, "addend": 0.5}, lambda v: v * 3.0 + 0.5),
        ({"multiplier": 1.5, "addend": -2.0}, lambda v: v * 1.5 - 2.0),
        ({"multiplier": 7.0, "addend": None}, lambda v: v * 7.0),
        ({"multiplier": 2.0, "addend": 1.0, "negate_product": True}, lambda v: -v * 2.0 + 1.0),
    ),
)
def test_synthesised_instructions_run_as_predicted(fields, step):
    """Assemble instructions from scratch, overwrite real ones, and predict the GPU.

    This is the strongest claim in the file. The bytes are not a compiler's output with a field
    edited; every one of the eight is chosen from the measured field map, and the arithmetic they
    produce was worked out before they were assembled.
    """
    text = _machine_code()
    offsets = _compact_offsets(text)
    assert len(offsets) == 3

    def rewrite(original):
        patched = bytearray(original)
        for index, offset in enumerate(offsets):
            patched[offset : offset + agx_isa.FMA_LENGTH] = agx_isa.encode_fma(
                original[offset] >> 4, last=(index == len(offsets) - 1), **fields
            )
        return bytes(patched)

    inputs = [1.0, 2.0, 3.0, 5.0]
    predicted = []
    for value in inputs:
        running = value * 2.0 + 1.0  # the long-form fma, left alone
        for _ in offsets:
            running = step(running)
        predicted.append(running)

    assert agx_isa.execute(CHAIN, "probe", inputs, rewrite=rewrite) == predicted


def test_the_addend_slot_can_name_a_register():
    """`rd * 1 + rs` is the register-plus-register add, which the immediate form cannot express."""
    built = agx_isa.encode_fma(2, 1.0, addend_register=3)
    assert not agx_isa.read_flag(built, 0, agx_isa.ADDEND_IMMEDIATE)
    assert built[agx_isa.FMA_ADDEND_BYTE] >> 1 == 3
    with pytest.raises(agx_isa.EncodingError, match="not both"):
        agx_isa.encode_fma(2, 1.0, addend=1.0, addend_register=3)
    with pytest.raises(agx_isa.EncodingError, match="addend register"):
        agx_isa.encode_fma(2, 1.0, addend_register=agx_isa.ARCHITECTURAL_REGISTERS)


def test_a_zero_addend_names_a_register_beyond_the_reachable_range():
    """There is no zero immediate, so a zero addend has to come from a register that reads zero.

    Asserted rather than left implicit because it is the one place the encoder relies on an index
    the field cannot reach. If a future part reduced that range, this would start adding whatever
    a now-reachable register holds, and silently.
    """
    built = agx_isa.encode_fma(2, 1.0)
    assert not agx_isa.read_flag(built, 0, agx_isa.ADDEND_IMMEDIATE)
    assert built[agx_isa.FMA_ADDEND_BYTE] >> 1 >= agx_isa.ARCHITECTURAL_REGISTERS


THREE_CHAINS = (
    """#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    float a = x[gid + 0];
    float b = x[gid + 1];
    float c = x[gid + 2];
"""
    + "\n".join(
        f"    {v} = fma({v}, {m}f, 1.0f);"
        for m, v in ((2.0, "a"), (3.0, "b"), (5.0, "c"))
        for _ in range(3)
    )
    + """
    out[gid] = a + b + c;
}
"""
)


def test_a_register_addend_adds_that_register_on_the_gpu():
    """Rewrite one instruction to read another chain's register, and predict all four threads.

    Four threads rather than one, because thread `gid` reads x[gid], x[gid+1] and x[gid+2]: an
    earlier version of this prediction used thread 0's inputs for every thread and mispredicted
    three of four while the encoding was perfectly correct.
    """
    from metile.target import Unavailable, machine_code

    multipliers = {"a": 2.0, "b": 3.0, "c": 5.0}
    try:
        text = machine_code(THREE_CHAINS, "probe")
    except Unavailable as error:
        pytest.skip(f"no Metal toolchain: {error}")

    byname = {agx_isa.encode_immediate(m, low_bit=1): name for name, m in multipliers.items()}
    compact = [
        (offset, byname[text[offset + agx_isa.FMA_MULTIPLIER_BYTE]])
        for offset in range(0, len(text) - agx_isa.FMA_LENGTH, 2)
        if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE
        and text[offset + 2] & 0x0F == 0x0E
        and text[offset + agx_isa.FMA_MULTIPLIER_BYTE] in byname
    ]
    registers = {chain: text[offset] >> 4 for offset, chain in compact}
    assert len(registers) == 3

    inputs = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0]
    threads = 4
    victim = [offset for offset, chain in compact if chain == "a"][-1]
    source = registers["b"]
    by_register = {value: name for name, value in registers.items()}

    def rewrite(original):
        patched = bytearray(original)
        patched[victim : victim + agx_isa.FMA_LENGTH] = agx_isa.encode_fma(
            registers["a"], 1.0, addend_register=source, last=False
        )
        return bytes(patched)

    predicted = []
    for gid in range(threads):
        state = {"a": inputs[gid], "b": inputs[gid + 1], "c": inputs[gid + 2]}
        for chain in state:
            hidden = 3 - sum(1 for _, owner in compact if owner == chain)
            for _ in range(hidden):
                state[chain] = state[chain] * multipliers[chain] + 1.0
        for offset, chain in compact:
            if offset == victim:
                state[chain] = state[chain] * 1.0 + state[by_register[source]]
            else:
                state[chain] = state[chain] * multipliers[chain] + 1.0
        predicted.append(sum(state.values()))

    got = agx_isa.execute(THREE_CHAINS, "probe", inputs, rewrite=rewrite)[:threads]
    assert got == predicted
