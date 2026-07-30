"""Measured properties of the hardware meTile compiles for.

A compiler that picks tilings, unroll factors and functional units needs to know what the
machine can actually do. Keeping those numbers here, with how each was measured, means a
pass can consult them instead of a comment repeating a value someone once saw.
"""

from metile.target.agx import (
    ILP_CEILING,
    MATRIX_PEAK_TFLOPS,
    REGISTER_BUDGET,
    SCALAR_PEAK_TFLOPS,
    STREAMING_READ_GBPS,
    Unavailable,
    ilp_headroom,
    inspect,
    machine_code,
    spills,
)

__all__ = [
    "ILP_CEILING",
    "MATRIX_PEAK_TFLOPS",
    "REGISTER_BUDGET",
    "SCALAR_PEAK_TFLOPS",
    "STREAMING_READ_GBPS",
    "Unavailable",
    "ilp_headroom",
    "inspect",
    "machine_code",
    "spills",
]
