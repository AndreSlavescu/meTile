"""Pure DSL partition, normalization, and gradient-reduction kernels for DCA.

The host backend composes these with stable_attention. Branches are disjoint
same-chunk, previous-chunk, and older-chunk key sets, with global causality.
The merge reconstructs one denominator from separate raw maxima/log sums;
branch outputs must never simply be added. All launches require STRICT_MATH.
"""

import metile


@metile.kernel
def dual_chunk_partition(
    Mask,
    Intra,
    Successive,
    Inter,
    Live,
    Q_LEN,
    K_LEN,
    ROWS,
    *,
    CHUNK_LEN: metile.constexpr,
    QUERY_START: metile.constexpr,
    KEY_START: metile.constexpr,
    HAS_MASK: metile.constexpr,
    BLOCK: metile.constexpr = 32,
):
    if type(CHUNK_LEN) is not int or not 0 < CHUNK_LEN < 2**31:
        raise ValueError("CHUNK_LEN must be a positive signed 32-bit integer")
    if any(type(start) is not int or not 0 <= start < 2**31 for start in (QUERY_START, KEY_START)):
        raise ValueError("query/key starts must be nonnegative signed 32-bit integers")
    if type(HAS_MASK) is not bool or type(BLOCK) is not int or BLOCK != 32:
        raise ValueError("partition requires bool HAS_MASK and BLOCK=32")
    work = metile.program_id(0)
    lane = metile.cast(metile.thread_id(), "i32")
    query_position = work % Q_LEN + QUERY_START
    query_chunk = query_position // CHUNK_LEN
    user_mask = metile.tensor(Mask + work * K_LEN, shape=(K_LEN,), access="read")
    branch_masks = [
        metile.tensor(pointer + work * K_LEN, shape=(K_LEN,), access="write")
        for pointer in (Intra, Successive, Inter)
    ]
    live_memory = metile.tensor(Live + work, shape=(3, 1), strides=(ROWS, 1), access="write")
    counts = [metile.scalar(0.0) for _ in range(3)]
    for start in metile.tile_range(0, K_LEN, BLOCK):
        key_index = start + lane
        key_position = metile.minimum(key_index, K_LEN - 1) + KEY_START
        key_chunk = key_position // CHUNK_LEN
        visible = (key_index < K_LEN) & (key_position <= query_position)
        if HAS_MASK:
            visible = visible & (user_mask.load((key_index,)) != 0)
        branches = (
            visible & (key_chunk == query_chunk),
            visible & (key_chunk == query_chunk - 1),
            visible & (key_chunk < query_chunk - 1),
        )
        for branch in range(3):
            branch_masks[branch].store((key_index,), branches[branch])
            counts[branch] = counts[branch] + metile.where(branches[branch], 1.0, 0.0)
    nonempty = [metile.simd_sum(count) > 0.0 for count in counts]
    for branch in range(3):
        live_memory.store((branch, lane), metile.where(nonempty[branch], 1.0, 0.0))


@metile.kernel
def dual_chunk_merge(
    OutIntra,
    OutSuccessive,
    OutInter,
    MaxIntra,
    MaxSuccessive,
    MaxInter,
    LogIntra,
    LogSuccessive,
    LogInter,
    Live,
    Out,
    OutFloat,
    RowMax,
    RowLogDen,
    ROWS,
    scale,
    *,
    D: metile.constexpr,
    BLOCK: metile.constexpr = 32,
):
    if type(D) is not int or D < 32 or D > 256 or D % 32:
        raise ValueError("D must be a multiple of 32 in [32, 256]")
    if type(BLOCK) is not int or BLOCK != 32:
        raise ValueError("merge requires BLOCK=32")
    work = metile.program_id(0)
    lane = metile.thread_id()
    branch_outputs = [
        metile.tensor(pointer + work * D, shape=(D,), access="read")
        for pointer in (OutIntra, OutSuccessive, OutInter)
    ]
    branch_maxima = [
        metile.tensor(pointer + work, shape=(1,), access="read")
        for pointer in (MaxIntra, MaxSuccessive, MaxInter)
    ]
    branch_denominators = [
        metile.tensor(pointer + work, shape=(1,), access="read")
        for pointer in (LogIntra, LogSuccessive, LogInter)
    ]
    live_memory = metile.tensor(Live + work, shape=(3, 1), strides=(ROWS, 1), access="read")
    output_memory = metile.tensor(Out + work * D, shape=(D,), access="write")
    precise_memory = metile.tensor(OutFloat + work * D, shape=(D,), access="write")
    maximum_memory = metile.tensor(RowMax + work, shape=(1,), access="write")
    denominator_memory = metile.tensor(RowLogDen + work, shape=(1,), access="write")

    live = [live_memory.load((branch, 0)) > 0.0 for branch in range(3)]
    maxima = [memory.load((0,)) for memory in branch_maxima]
    maximum = -3.4028234663852886e38
    for branch in range(3):
        maximum = metile.maximum(maximum, metile.where(live[branch], maxima[branch], maximum))
    any_live = live[0] | live[1] | live[2]
    maximum = metile.where(any_live, maximum, 0.0)
    weights = []
    denominator = 0.0
    for branch in range(3):
        shifted = metile.where(live[branch], maxima[branch] - maximum, 0.0)
        log_denominator = branch_denominators[branch].load((0,))
        weight = metile.where(live[branch], metile.exp(shifted * scale + log_denominator), 0.0)
        weights.append(weight)
        denominator = denominator + weight
    safe_denominator = metile.where(any_live, denominator, 1.0)
    for component in range(D // 32):
        dimension = lane + component * 32
        output = 0.0
        for branch in range(3):
            output = output + branch_outputs[branch].load((dimension,)) * (
                weights[branch] / safe_denominator
            )
        output_memory.store((dimension,), output)
        precise_memory.store((dimension,), output)
    maximum_memory.store((lane,), maximum)
    denominator_memory.store((lane,), metile.log(safe_denominator))


@metile.kernel
def dual_chunk_sum_key_value_gradients(
    KeyIntra,
    KeySuccessive,
    KeyInter,
    ValueIntra,
    ValueSuccessive,
    ValueInter,
    GradKey,
    GradValue,
    ELEMENTS,
    *,
    BLOCK: metile.constexpr = 256,
):
    if type(BLOCK) is not int or BLOCK != 256:
        raise ValueError("gradient reduction requires BLOCK=256")
    key_partials = [
        metile.tensor(pointer, shape=(ELEMENTS,), access="read")
        for pointer in (KeyIntra, KeySuccessive, KeyInter)
    ]
    value_partials = [
        metile.tensor(pointer, shape=(ELEMENTS,), access="read")
        for pointer in (ValueIntra, ValueSuccessive, ValueInter)
    ]
    key_output = metile.tensor(GradKey, shape=(ELEMENTS,), access="write")
    value_output = metile.tensor(GradValue, shape=(ELEMENTS,), access="write")
    indices = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    key_sum = 0.0
    value_sum = 0.0
    for branch in range(3):
        key_sum = key_sum + key_partials[branch].load((indices,))
        value_sum = value_sum + value_partials[branch].load((indices,))
    key_output.store((indices,), key_sum)
    value_output.store((indices,), value_sum)
