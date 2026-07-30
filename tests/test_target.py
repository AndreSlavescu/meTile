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
