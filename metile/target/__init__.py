"""Measured properties of the hardware meTile compiles for.

A compiler that picks tilings, unroll factors and functional units needs to know what the
machine can actually do. Keeping those numbers here, with how each was measured, means a
pass can consult them instead of a comment repeating a value someone once saw.
"""

from metile.target.agx import (
    BANDWIDTH_BY_WORKING_SET_GBPS,
    ILP_CEILING,
    MATRIX_PEAK_TFLOPS,
    REGISTER_BUDGET,
    RESIDENT_READ_GBPS,
    RESIDENT_WORKING_SET_BYTES,
    SCALAR_PEAK_TFLOPS,
    STREAMING_READ_GBPS,
    THREADGROUP_CONFLICT_STRIDE_BYTES,
    THREADGROUP_GBPS_BY_STRIDE,
    THREADGROUP_OVER_RESIDENT,
    THREADGROUP_PEAK_GBPS,
    Unavailable,
    ilp_headroom,
    inspect,
    machine_code,
    read_bandwidth_gbps,
    resident,
    spills,
    threadgroup_conflicts,
    tiling_gain,
)

__all__ = [
    "BANDWIDTH_BY_WORKING_SET_GBPS",
    "ILP_CEILING",
    "MATRIX_PEAK_TFLOPS",
    "REGISTER_BUDGET",
    "RESIDENT_READ_GBPS",
    "RESIDENT_WORKING_SET_BYTES",
    "SCALAR_PEAK_TFLOPS",
    "STREAMING_READ_GBPS",
    "THREADGROUP_CONFLICT_STRIDE_BYTES",
    "THREADGROUP_GBPS_BY_STRIDE",
    "THREADGROUP_OVER_RESIDENT",
    "THREADGROUP_PEAK_GBPS",
    "Unavailable",
    "ilp_headroom",
    "inspect",
    "machine_code",
    "read_bandwidth_gbps",
    "resident",
    "spills",
    "threadgroup_conflicts",
    "tiling_gain",
]
