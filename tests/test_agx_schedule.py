"""Optimising machine code has to be verified by running it, so that is what these do.

The IR scheduler can be checked structurally: its output is IR, and a hazard either was or was
not respected. This pass emits instructions, and an unsound rewrite there produces a kernel that
compiles, dispatches, and returns a wrong number. So the two transformations that claim to be
bit-exact are asserted bit-exact against a GPU run, not against a model of one.
"""

import pytest

from metile.compiler import agx_schedule
from metile.target import agx_isa

CHAINS = ((2.0, "a"), (3.0, "b"), (5.0, "c"))
SOURCE = (
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
    + "\n".join(f"    {v} = fma({v}, {m}f, 1.0f);" for m, v in CHAINS for _ in range(3))
    + """
    out[gid] = a + b + c;
}
"""
)
INPUTS = [1.0, 2.0, 3.0, 4.0]


def _code_and_offsets():
    from metile.target import Unavailable, machine_code

    try:
        text = machine_code(SOURCE, "probe")
    except Unavailable as error:
        pytest.skip(f"no Metal toolchain: {error}")
    wanted = {agx_isa.encode_immediate(m, low_bit=1) for m, _ in CHAINS}
    offsets = [
        offset
        for offset in range(0, len(text) - agx_isa.FMA_LENGTH, 2)
        if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE
        and text[offset + 2] & 0x0F == 0x0E
        and text[offset + agx_isa.FMA_MULTIPLIER_BYTE] in wanted
    ]
    return text, offsets


def test_decoding_recovers_the_arithmetic_the_source_asked_for():
    """Multipliers and registers have to come back, or nothing downstream means anything."""
    text, offsets = _code_and_offsets()
    decoded = agx_schedule.decode(text, offsets)
    assert decoded, "no compact fmas were found"
    multipliers = {instruction.multiplier for instruction in decoded}
    assert multipliers <= {m for m, _ in CHAINS}
    # One register per chain, and each instruction reads and writes the same one.
    assert len({instruction.register for instruction in decoded}) == len(CHAINS)
    for instruction in decoded:
        assert instruction.addend == 1.0
        assert not instruction.negate_product


def test_decoding_rejects_an_offset_that_is_not_a_compact_fma():
    """A wrong offset must fail here, because downstream it silently corrupts the kernel."""
    text, offsets = _code_and_offsets()
    with pytest.raises(ValueError, match="not a compact fma"):
        agx_schedule.decode(text, [offsets[0] + 1])


def test_reordering_moves_instructions_and_changes_no_result():
    """The claim the pass exists to make, checked on the GPU rather than argued.

    Three chains on three registers, so there is real freedom, and distinct multipliers so a
    misordering shows up in the arithmetic instead of hiding behind identical operations.
    """
    text, offsets = _code_and_offsets()
    baseline = agx_isa.execute(SOURCE, "probe", INPUTS)

    reordered, moved = agx_schedule.reorder(text, offsets)
    assert moved > 0, "nothing moved, so this asserts nothing about soundness"
    assert reordered != text

    assert agx_isa.execute(SOURCE, "probe", INPUTS, rewrite=lambda _: reordered) == baseline


def test_reordering_a_single_register_run_is_a_no_op():
    """No freedom means no churn, and no chance to break something for nothing."""
    text, offsets = _code_and_offsets()
    one_register = [
        instruction.offset
        for instruction in agx_schedule.decode(text, offsets)
        if instruction.register == agx_schedule.decode(text, offsets)[0].register
    ]
    reordered, moved = agx_schedule.reorder(text, one_register)
    assert moved == 0
    assert reordered == text


def test_retiring_an_identity_instruction_changes_no_result():
    """`a * 1` computes nothing, so removing it must be invisible.

    The identity is planted rather than waited for, because the compiler does not emit one. That
    is the point of the transformation: it exists for code another pass produced.
    """
    text, offsets = _code_and_offsets()
    planted = bytearray(text)
    victim = offsets[1]
    planted[victim : victim + agx_isa.FMA_LENGTH] = agx_isa.encode_fma(
        text[victim] >> 4,
        1.0,
        None,
        last=not agx_isa.read_flag(text, victim, agx_schedule._CONTINUES),
    )
    planted = bytes(planted)

    assert agx_schedule.summarise(planted, offsets)["identities"] == 1
    simplified, retired = agx_schedule.simplify(planted, offsets)
    assert retired == 1

    before = agx_isa.execute(SOURCE, "probe", INPUTS, rewrite=lambda _: planted)
    after = agx_isa.execute(SOURCE, "probe", INPUTS, rewrite=lambda _: simplified)
    assert after == before


def test_an_identity_is_only_an_identity_without_a_sign_flip():
    """`-a * 1` is a negation, not a no-op, and retiring it would change the answer."""
    negated = agx_isa.encode_fma(0, 1.0, None, negate_product=True)
    assert not agx_schedule.decode(negated, [0])[0].is_identity()
    assert agx_schedule.decode(agx_isa.encode_fma(0, 1.0, None), [0])[0].is_identity()


def test_summarise_reports_runs_rather_than_a_single_count():
    """A run of one offers nothing to reorder however many of them there are."""
    text, offsets = _code_and_offsets()
    summary = agx_schedule.summarise(text, offsets)
    assert summary["instructions"] == len(offsets)
    assert sum(summary["runs"]) == len(offsets)
    assert len(summary["registers"]) == len(CHAINS)


def test_optimize_runs_both_passes_and_reports_what_it_did():
    """One entry point, and its report has to match what the individual passes claim."""
    text, offsets = _code_and_offsets()
    baseline = agx_isa.execute(SOURCE, "probe", INPUTS)

    optimised, report = agx_schedule.optimize(text, offsets)
    assert report["moved"] == agx_schedule.reorder(text, offsets)[1]
    assert report["retired"] == agx_schedule.simplify(text, offsets)[1]
    assert agx_isa.execute(SOURCE, "probe", INPUTS, rewrite=lambda _: optimised) == baseline


def test_optimize_with_everything_switched_off_returns_the_code_untouched():
    text, offsets = _code_and_offsets()
    untouched, report = agx_schedule.optimize(
        text, offsets, simplify_identities=False, reorder_independent=False
    )
    assert untouched == text
    assert report == {"retired": 0, "moved": 0, "folded": 0}


def test_an_fma_adding_a_live_register_is_not_an_identity():
    """`a * 1 + r3` adds a real term, so retiring it would drop it.

    This is the property the addend-flag correction bought. When the flag was read as
    "addend present or absent", `a * 1` with anything in the slot looked like a no-op, and this
    instruction would have been retired and a term silently lost. Deciding it needs the register
    index, because only an index beyond the reachable range reads zero.
    """
    live = agx_isa.encode_fma(2, 1.0, addend_register=3)
    decoded = agx_schedule.decode(live, [0])[0]
    assert decoded.addend_register == 3
    assert not decoded.addend_is_zero()
    assert not decoded.is_identity()

    zero = agx_schedule.decode(agx_isa.encode_fma(2, 1.0), [0])[0]
    assert zero.addend_is_zero()
    assert zero.is_identity()


def test_decoding_round_trips_every_addend_form():
    """Re-encoding a decoded instruction has to reproduce it, or a rewrite silently changes it."""
    for original in (
        agx_isa.encode_fma(2, 2.0, 1.0),
        agx_isa.encode_fma(2, 2.0, -1.0),
        agx_isa.encode_fma(2, 1.5, addend_register=3),
        agx_isa.encode_fma(2, 1.0),
        agx_isa.encode_fma(2, 2.0, 1.0, negate_product=True),
        agx_isa.encode_fma(2, 2.0, 1.0, last=True),
    ):
        assert agx_schedule.decode(original, [0])[0].encode() == original


CONSTANT_CHAIN = """#include <metal_stdlib>
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


def _constant_chain_offsets():
    from metile.target import Unavailable, machine_code

    try:
        text = machine_code(CONSTANT_CHAIN, "probe")
    except Unavailable as error:
        pytest.skip(f"no Metal toolchain: {error}")
    offsets = [
        offset
        for offset in range(0, len(text) - agx_isa.FMA_LENGTH, 2)
        if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE
        and text[offset + 2] & 0x0F == 0x0E
        and text[offset + agx_isa.FMA_MULTIPLIER_BYTE] == 0xC1
    ]
    return text, offsets


def test_collapsing_a_run_folds_it_into_one_instruction():
    """Three steps of a*2+1 are one a*8+7, and the GPU has to agree.

    The closed form for k steps is a*m**k + d*(m**(k-1) + ... + 1). Here the constants are powers
    of two so the fold is exact and can be compared against the untouched kernel as well as against
    the prediction; in general it reassociates, which is why it is off by default.
    """
    text, offsets = _constant_chain_offsets()
    assert len(offsets) == 3
    baseline = agx_isa.execute(CONSTANT_CHAIN, "probe", [1.0, 2.0, 3.0, 5.0])

    folded_code, folded = agx_schedule.collapse(text, offsets)
    assert folded == 1

    remaining = agx_schedule.decode(folded_code, offsets)
    assert remaining[0].multiplier == 8.0
    assert remaining[0].addend == 7.0
    assert all(
        agx_isa.read_flag(folded_code, instruction.offset, agx_isa.INSTRUCTION_DISABLE)
        for instruction in remaining[1:]
    )

    got = agx_isa.execute(
        CONSTANT_CHAIN, "probe", [1.0, 2.0, 3.0, 5.0], rewrite=lambda _: folded_code
    )
    assert got == baseline


def test_collapsing_leaves_a_run_alone_when_the_closed_form_escapes_the_field():
    """Approximating would change the result by more than reassociating does.

    Six steps of a*2+1 close to a*64 + 63, and 63 is not `(1 + m/8) * 2**e` for any m under eight,
    so the fold is declined rather than rounded.
    """
    instruction = agx_isa.encode_fma(0, 2.0, 1.0)
    run = instruction * 6
    offsets = [index * agx_isa.FMA_LENGTH for index in range(6)]
    with pytest.raises(agx_isa.EncodingError):
        agx_isa.encode_immediate(63.0, low_bit=0)
    unchanged, folded = agx_schedule.collapse(run, offsets)
    assert folded == 0
    assert unchanged == run


def test_collapsing_declines_a_register_addend():
    """A register's value can change between steps, so the closed form does not hold."""
    instruction = agx_isa.encode_fma(0, 2.0, addend_register=3)
    run = instruction * 3
    offsets = [index * agx_isa.FMA_LENGTH for index in range(3)]
    unchanged, folded = agx_schedule.collapse(run, offsets)
    assert folded == 0
    assert unchanged == run


def test_optimize_leaves_folding_off_unless_asked():
    """It reassociates, and the model tests assert bit-exact logits."""
    text, offsets = _constant_chain_offsets()
    _, report = agx_schedule.optimize(text, offsets)
    assert report["folded"] == 0
    _, asked = agx_schedule.optimize(text, offsets, fold_runs=True)
    assert asked["folded"] == 1
