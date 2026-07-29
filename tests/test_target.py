"""The target model is measured hardware knowledge, so guard what depends on its shape."""

import pytest

from metile.target import (
    ILP_CEILING,
    MATRIX_PEAK_TFLOPS,
    REGISTER_BUDGET,
    SCALAR_PEAK_TFLOPS,
    STREAMING_READ_GBPS,
    ilp_headroom,
    spills,
)


def test_spilling_is_decided_at_the_budget():
    """One register below the budget is fine; reaching it is spilling."""
    assert not spills(REGISTER_BUDGET - 1)
    assert spills(REGISTER_BUDGET)
    assert spills(REGISTER_BUDGET + 40)


def test_ilp_headroom_is_small_and_defaults_to_none():
    """A pass consulting this should learn that scheduling has almost nothing to win.

    The values are measured: a single dependent fma chain already reaches 92% of fp32 peak,
    so no scheduler can find more than 1.09x there. An unknown element type must not report
    imaginary headroom, so it falls back to 1.0 rather than to the largest known value.
    """
    assert ilp_headroom("f32") == pytest.approx(1.09)
    assert ilp_headroom("f16") == pytest.approx(1.41)
    assert ilp_headroom("bf16") == 1.0
    assert max(ILP_CEILING.values()) < 1.5


def test_matrix_unit_beats_scheduling_by_more_than_scheduling_can_offer():
    """The comparison that decides where compiler effort belongs.

    Choosing the right functional unit is worth the matrix-to-scalar ratio; reordering
    instructions is worth at most the ILP ceiling. If that ordering ever inverts on new
    hardware, the guidance built on it needs revisiting rather than silently carrying over.
    """
    for dtype, scalar in SCALAR_PEAK_TFLOPS.items():
        functional_unit_gain = MATRIX_PEAK_TFLOPS / scalar
        assert functional_unit_gain > ILP_CEILING[dtype]


def test_streaming_ceiling_sits_below_the_advertised_bandwidth():
    """Measured read bandwidth, not the spec sheet number, is what a roofline needs."""
    assert 100.0 < STREAMING_READ_GBPS < 153.0
