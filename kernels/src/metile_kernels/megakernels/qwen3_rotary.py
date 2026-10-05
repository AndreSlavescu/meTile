"""Generate reusable FP32 RoPE constants entirely with the public DSL.

The operation order follows MLX v0.32.0 rope.metal: float feature division,
base-two exponential of the negative scaled frequency, then fast sine and
cosine of the float position angle. LOG2_BASE is the host-rounded FP32 log2
of the model's rotary base, not the base itself. Launch enough threadgroups
to cover MAX_CONTEXT * (HEAD_DIM // 2) angle pairs.
"""

import math

import metile


@metile.kernel
def qwen3_rotary_table(
    Rotary,
    *,
    HEAD_DIM: metile.constexpr,
    MAX_CONTEXT: metile.constexpr,
    LOG2_BASE: metile.constexpr,
    BLOCK: metile.constexpr = 128,
):
    """Write ``[MAX_CONTEXT, HEAD_DIM // 2, 2]`` interleaved cosine/sine pairs."""
    if type(HEAD_DIM) is not int or HEAD_DIM <= 0 or HEAD_DIM % 2:
        raise ValueError("HEAD_DIM must be a positive even integer")
    if type(MAX_CONTEXT) is not int or MAX_CONTEXT <= 0:
        raise ValueError("MAX_CONTEXT must be a positive integer")
    if MAX_CONTEXT * HEAD_DIM >= 2**31:
        raise ValueError("rotary table exceeds signed 32-bit addressing")
    if type(LOG2_BASE) not in (int, float) or not math.isfinite(LOG2_BASE):
        raise ValueError("LOG2_BASE must be finite")
    if type(BLOCK) is not int or BLOCK < 32 or BLOCK > 1024 or BLOCK % 32:
        raise ValueError("BLOCK must be a multiple of 32 between 32 and 1024")
    rotary = metile.tensor(Rotary, shape=(MAX_CONTEXT, HEAD_DIM // 2, 2), access="write")
    offset = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    position = offset // (HEAD_DIM // 2)
    feature = offset % (HEAD_DIM // 2)
    fraction = metile.cast(feature, "f32") / float(HEAD_DIM // 2)
    frequency = metile.exp2((0.0 - fraction) * LOG2_BASE)
    angle = metile.cast(position, "f32") * frequency
    rotary.store((position, feature, 0), metile.fast_cos(angle))
    rotary.store((position, feature, 1), metile.fast_sin(angle))
