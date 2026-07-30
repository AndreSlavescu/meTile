"""The target model is measured hardware knowledge, so guard what depends on its shape."""

import itertools

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


def _probe(body):
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void probe(device const float* x [[buffer(0)]],\n"
        "                  device float* out     [[buffer(1)]],\n"
        "                  constant uint& n      [[buffer(2)]],\n"
        "                  uint gid [[thread_position_in_grid]]) {\n" + body + "\n}\n"
    )


_SERIAL = _probe(
    "    float a = x[gid], b = x[gid + 1];\n"
    "    float c = float(n), d = float(n) + 1.0f;\n"
    + "    a = fma(a, c, d);\n" * 4
    + "    b = fma(b, c, d);\n" * 4
    + "    out[gid] = a + b;"
)
_INTERLEAVED = _probe(
    "    float a = x[gid], b = x[gid + 1];\n"
    "    float c = float(n), d = float(n) + 1.0f;\n"
    + "    a = fma(a, c, d); b = fma(b, c, d);\n" * 4
    + "    out[gid] = a + b;"
)


def test_the_backend_normalises_statement_order():
    """The assumption that keeps metile.compiler.scheduling switched off.

    Two independent fma chains, written serially and written interleaved, compile to
    byte-identical machine code. That is a stronger statement than any timing: source forms
    that produce the same instructions cannot differ in speed, so reordering MSL statements
    provably cannot pay, and the scheduling pass is off by default on those grounds.

    If a toolchain update makes this fail, the grounds are gone and the pass is worth enabling
    again. Failing loudly is the point; skipping would let the assumption rot quietly.
    """
    from metile.target import Unavailable, machine_code

    try:
        serial = machine_code(_SERIAL, "probe")
    except Unavailable as error:
        pytest.skip(f"no Metal toolchain to read compiled kernels: {error}")

    assert serial, "the probe kernel produced no machine code, so this proves nothing"
    assert serial == machine_code(_INTERLEAVED, "probe"), (
        "statement order now reaches the machine code. metile.compiler.scheduling is disabled "
        "on the assumption that it does not; re-measure benchmarks/agx_source_order.py and "
        "reconsider the default."
    )


def test_bandwidth_depends_on_working_set_and_the_knee_is_sharp():
    """One bandwidth number is badly wrong for this part, so the model has to be a curve.

    The measured drop from 2 MB to 4 MB is about a factor of four. Interpolating across it would invent
    a smooth ramp the hardware does not have, so the lookup reports the measurement for the smallest
    size at least as large as the request.
    """
    from metile.target import RESIDENT_WORKING_SET_BYTES, read_bandwidth_gbps

    resident_rate = read_bandwidth_gbps(RESIDENT_WORKING_SET_BYTES)
    beyond_rate = read_bandwidth_gbps(RESIDENT_WORKING_SET_BYTES * 2)
    assert resident_rate > 3 * beyond_rate
    assert read_bandwidth_gbps(2**30) == pytest.approx(STREAMING_READ_GBPS)

    # Monotonically non-increasing: a larger working set is never served faster. This is what caught
    # a 256 KB entry reading slower than 512 KB, which is impossible for a smaller working set and was
    # the probe's loop overhead rather than the hierarchy.
    rates = [read_bandwidth_gbps(2**exponent) for exponent in range(14, 31)]
    for faster, slower in itertools.pairwise(rates):
        assert faster >= slower


def test_a_working_set_is_resident_only_up_to_the_measured_capacity():
    from metile.target import RESIDENT_WORKING_SET_BYTES, resident

    assert resident(RESIDENT_WORKING_SET_BYTES)
    assert resident(1024)
    assert not resident(RESIDENT_WORKING_SET_BYTES + 1)
    assert not resident(0)


def test_fitting_a_tile_outranks_every_other_lever_in_this_file():
    """The comparison that should drive where compiler effort goes.

    Keeping a working set resident is worth about 19x. Choosing the matrix unit over scalar is worth
    2.4x to 3.7x. Instruction scheduling is capped at 1.09x and measured unreachable above MSL. If that
    ordering ever changes on new hardware, the guidance built on it needs revisiting rather than being
    carried over.
    """
    from metile.target import RESIDENT_WORKING_SET_BYTES, tiling_gain

    fitting = tiling_gain(RESIDENT_WORKING_SET_BYTES)
    functional_unit = MATRIX_PEAK_TFLOPS / min(SCALAR_PEAK_TFLOPS.values())
    assert fitting > functional_unit > max(ILP_CEILING.values())
    assert tiling_gain(2**30) == pytest.approx(1.0)
