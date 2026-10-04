import pytest

from metile.compiler import agx_schedule
from metile.target import agx_isa


def _program(instructions):
    code = b"".join(instructions)
    offsets = list(range(0, len(code), agx_isa.FMA_LENGTH))
    return code, offsets


def _evaluate(code, offsets):
    registers = [float(index + 1) for index in range(agx_isa.ARCHITECTURAL_REGISTERS)]
    for instruction in agx_schedule.decode(code, offsets):
        if instruction.disabled:
            continue
        product = registers[instruction.register] * instruction.multiplier
        if instruction.negate_product:
            product = -product
        if instruction.addend is not None:
            addend = instruction.addend
        elif instruction.addend_is_zero():
            addend = 0.0
        else:
            addend = registers[instruction.addend_register]
            if agx_isa.read_flag(code, instruction.offset, agx_isa.ADDEND_NEGATE):
                addend = -addend
        registers[instruction.register] = product + addend
    return registers


@pytest.mark.parametrize(
    "instructions",
    [
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            agx_isa.encode_fma(1, 3.0, 1.0),
            agx_isa.encode_fma(0, 1.0, addend_register=1),
            agx_isa.encode_fma(2, 2.0, 1.0, last=True),
        ],
        [
            agx_isa.encode_fma(1, 1.0, addend_register=0),
            agx_isa.encode_fma(0, 2.0, 1.0),
            agx_isa.encode_fma(2, 2.0, 1.0, last=True),
        ],
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            agx_isa.encode_fma(1, 3.0, 2.0),
            agx_isa.encode_fma(0, 2.0, 1.0, last=True),
        ],
    ],
    ids=["read_after_write", "write_after_read", "write_after_write"],
)
def test_reordering_preserves_register_dependencies(instructions):
    code, offsets = _program(instructions)
    reordered, _ = agx_schedule.reorder(code, offsets)
    assert _evaluate(reordered, offsets) == _evaluate(code, offsets)


def test_independent_chains_still_interleave():
    code, offsets = _program(
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            agx_isa.encode_fma(1, 3.0, 1.0),
            agx_isa.encode_fma(0, 2.0, 1.0),
            agx_isa.encode_fma(0, 3.0, 1.0, last=True),
        ]
    )
    reordered, moved = agx_schedule.reorder(code, offsets)
    assert moved == 2
    assert [instruction.register for instruction in agx_schedule.decode(reordered, offsets)] == [
        1,
        0,
        1,
        0,
    ]
    assert _evaluate(reordered, offsets) == _evaluate(code, offsets)


def test_reordering_preserves_negative_register_addends():
    negative_addend = agx_isa.write_flag(
        agx_isa.encode_fma(1, 1.0, addend_register=2), 0, agx_isa.ADDEND_NEGATE, True
    )
    code, offsets = _program(
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            negative_addend,
            agx_isa.encode_fma(0, 2.0, 1.0, last=True),
        ]
    )
    assert agx_schedule.decode(negative_addend, [0])[0].encode() == negative_addend
    reordered, moved = agx_schedule.reorder(code, offsets)
    assert moved > 0
    assert _evaluate(reordered, offsets) == _evaluate(code, offsets)


def test_reordering_does_not_reactivate_disabled_instructions():
    code, offsets = _program(
        [agx_isa.encode_fma(1, 2.0, 1.0), agx_isa.encode_fma(0, 2.0, 1.0, last=True)]
    )
    code = agx_isa.write_flag(code, 0, agx_isa.INSTRUCTION_DISABLE, True)
    reordered, moved = agx_schedule.reorder(code, offsets)
    assert reordered == code
    assert moved == 0
    assert _evaluate(reordered, offsets) == _evaluate(code, offsets)


def test_summary_separates_barriers_from_transformable_instructions():
    code, offsets = _program(
        [agx_isa.encode_fma(1, 1.0), agx_isa.encode_fma(0, 2.0, 1.0, last=True)]
    )
    code = agx_isa.write_flag(code, 0, agx_isa.INSTRUCTION_DISABLE, True)
    summary = agx_schedule.summarise(code, offsets)
    assert summary["instructions"] == 2
    assert summary["barriers"] == 1
    assert summary["runs"] == [1]
    assert summary["identities"] == 0


def test_optimization_keeps_folded_instructions_disabled_and_is_idempotent():
    code, offsets = _program(
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            agx_isa.encode_fma(1, 2.0, 1.0),
            agx_isa.encode_fma(0, 2.0, 1.0, last=True),
        ]
    )
    optimized, report = agx_schedule.optimize(code, offsets, fold_runs=True)
    assert report["folded"] == 1
    assert agx_isa.read_flag(optimized, 8, agx_isa.INSTRUCTION_DISABLE)
    assert _evaluate(optimized, offsets) == _evaluate(code, offsets)
    repeated, report = agx_schedule.optimize(optimized, offsets, fold_runs=True)
    assert repeated == optimized
    assert report == {"retired": 0, "moved": 0, "folded": 0}


@pytest.mark.parametrize(
    "byte, mask", [(2, 0x01), (3, 0x01), (4, 0x02), (5, 0x01), (6, 0x40), (7, 0x20)]
)
def test_unknown_operand_modes_and_controls_are_barriers(byte, mask):
    unknown = bytearray(agx_isa.encode_fma(0, 1.0))
    unknown[byte] ^= mask
    code, offsets = _program(
        [
            agx_isa.encode_fma(1, 2.0, 1.0),
            bytes(unknown),
            agx_isa.encode_fma(2, 2.0, 1.0, last=True),
        ]
    )
    decoded = agx_schedule.decode(code, offsets)
    assert decoded[1].encode() == bytes(unknown)
    assert not decoded[1].supported()
    optimized, report = agx_schedule.optimize(code, offsets, fold_runs=True)
    assert optimized == code
    assert report == {"retired": 0, "moved": 0, "folded": 0}


def test_reordering_does_not_cross_terminal_instructions():
    code, offsets = _program(
        [agx_isa.encode_fma(1, 2.0, 1.0, last=True), agx_isa.encode_fma(0, 2.0, 1.0, last=True)]
    )
    reordered, moved = agx_schedule.reorder(code, offsets)
    assert reordered == code
    assert moved == 0


@pytest.mark.parametrize("offsets", [[-8], [1], [0, 0], [0, 2], [8, 0]])
def test_invalid_instruction_offsets_are_rejected(offsets):
    code = agx_isa.encode_fma(0, 2.0, 1.0) * 2
    with pytest.raises(ValueError, match="offsets must be"):
        agx_schedule.decode(code, offsets)
