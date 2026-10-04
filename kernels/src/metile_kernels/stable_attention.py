"""Bounded FP32-accumulating attention and deterministic output-owner gradients.

This is an independent DSL implementation inspired by KohakuFA's precision
analysis, not a port of its CUDA kernels:
https://github.com/KohakuBlueleaf/KohakuFA/blob/041ac512ac474709ad910e505c545a1ab94853c2/docs/precision.md

Q/Out/OutFloat use contiguous [batch, query_head, query, dimension] storage;
K/V use [batch, kv_head, key, dimension]. Statistics omit the dimension axis.
Mask is uint8 [batch, query_head, query, key], with nonzero entries visible.
OutFloat and both statistics must be FP32; backward reads the unrounded
OutFloat, not the possibly FP16 Out. Gradient buffers should also be FP32.

Launch one 32-thread group per query row for forward/dQ and per KV row for
dK/dV, with STRICT_MATH=True. The caller validates positive finite scale,
positive runtime lengths, matching FP16/FP32 storage, and disjoint buffers.
Finite inputs must also produce finite FP32 dot products. CAUSAL_OFFSET=0
means top-left causality; K_LEN-Q_LEN means bottom-right causality. Fully
masked rows write zero outputs and zero statistics and contribute no gradient.

The saved statistics are raw maximum m and log(sum(exp((QK-m)*scale))).
Keeping them separate avoids losing the denominator when the raw scores are
large. Subtraction precedes scaling in both forward and backward. No attention
matrix or floating-point atomics are required; no throughput claim is implied.
"""

import metile


def _validate_contract(dimension, query_heads, kv_heads, causal, offset, has_mask, block):
    if type(dimension) is not int or dimension < 32 or dimension > 256 or dimension % 32:
        raise ValueError("stable attention requires D to be a multiple of 32 in [32, 256]")
    if (
        type(query_heads) is not int
        or type(kv_heads) is not int
        or query_heads <= 0
        or kv_heads <= 0
        or query_heads % kv_heads
    ):
        raise ValueError("stable attention requires positive Q_HEADS divisible by KV_HEADS")
    if type(causal) is not bool or type(has_mask) is not bool:
        raise TypeError("CAUSAL and HAS_MASK must be bool constexprs")
    if type(offset) is not int or not -(1 << 31) <= offset < (1 << 31):
        raise ValueError("CAUSAL_OFFSET must be a signed 32-bit integer constexpr")
    if type(block) is not int or block != 32:
        raise ValueError("stable attention requires BLOCK=32")


def _visible(mask, query, key, key_length, causal, offset, has_mask, mask_indices):
    visible = key < key_length
    if causal:
        visible = visible & (key <= query + offset)
    if has_mask:
        visible = visible & (mask.load(mask_indices) != 0)
    return visible


def _probability(score, maximum, log_denominator, scale, visible):
    safe_score = metile.where(visible, score, maximum)
    shifted = safe_score - maximum
    return metile.where(visible, metile.exp(shifted * scale - log_denominator), 0.0)


@metile.kernel
def stable_attention_forward(
    Q,
    K,
    V,
    Mask,
    Out,
    OutFloat,
    RowMax,
    RowLogDen,
    Q_LEN,
    K_LEN,
    scale,
    *,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    CAUSAL: metile.constexpr = False,
    CAUSAL_OFFSET: metile.constexpr = 0,
    HAS_MASK: metile.constexpr = False,
    BLOCK: metile.constexpr = 32,
):
    """Online attention with separate raw maxima and log denominators."""
    _validate_contract(D, Q_HEADS, KV_HEADS, CAUSAL, CAUSAL_OFFSET, HAS_MASK, BLOCK)
    work = metile.program_id(0)
    query_index = work % Q_LEN
    query_head = work // Q_LEN % Q_HEADS
    batch = work // (Q_LEN * Q_HEADS)
    kv_head = query_head // (Q_HEADS // KV_HEADS)
    kv_offset = (batch * KV_HEADS + kv_head) * K_LEN * D
    lane = metile.thread_id()
    query_memory = metile.tensor(Q + work * D, shape=(D,), access="read")
    key_memory = metile.tensor(K + kv_offset, shape=(K_LEN, D), access="read")
    value_memory = metile.tensor(V + kv_offset, shape=(K_LEN, D), access="read")
    mask_memory = metile.tensor(Mask + work * K_LEN, shape=(K_LEN,), access="read")
    output_memory = metile.tensor(Out + work * D, shape=(D,), access="write")
    precise_memory = metile.tensor(OutFloat + work * D, shape=(D,), access="write")
    maximum_memory = metile.tensor(RowMax + work, shape=(1,), access="write")
    denominator_memory = metile.tensor(RowLogDen + work, shape=(1,), access="write")

    query = []
    for component in range(D // 32):
        query.append(metile.cast(query_memory.load((lane + component * 32,)), "f32"))
    maximum = metile.scalar(-3.4028234663852886e38)
    denominator = metile.scalar(0.0)
    outputs = [metile.scalar(0.0) for _ in range(D // 32)]
    for key_index in metile.tile_range(0, K_LEN, 1):
        score = 0.0
        for component in range(D // 32):
            dimension = lane + component * 32
            key = metile.cast(key_memory.load((key_index, dimension)), "f32")
            score = score + query[component] * key
        score = metile.simd_sum(score)
        visible = _visible(
            mask_memory,
            query_index,
            key_index,
            K_LEN,
            CAUSAL,
            CAUSAL_OFFSET,
            HAS_MASK,
            (key_index,),
        )
        candidate = metile.where(visible, score, maximum)
        new_maximum = metile.maximum(maximum, candidate)
        old_shift = metile.where(denominator > 0.0, maximum - new_maximum, 0.0)
        old_factor = metile.exp(old_shift * scale)
        safe_score = metile.where(visible, score, new_maximum)
        probability = metile.where(visible, metile.exp((safe_score - new_maximum) * scale), 0.0)
        denominator = denominator * old_factor + probability
        for component in range(D // 32):
            dimension = lane + component * 32
            value = metile.cast(value_memory.load((key_index, dimension)), "f32")
            outputs[component] = outputs[component] * old_factor + probability * value
        maximum = new_maximum

    safe_denominator = metile.where(denominator > 0.0, denominator, 1.0)
    for component in range(D // 32):
        dimension = lane + component * 32
        output = outputs[component] / safe_denominator
        output_memory.store((dimension,), output)
        precise_memory.store((dimension,), output)
    maximum_memory.store((lane,), metile.where(denominator > 0.0, maximum, 0.0))
    denominator_memory.store((lane,), metile.log(safe_denominator))


@metile.kernel
def stable_attention_backward_query(
    Q,
    K,
    V,
    Mask,
    OutFloat,
    RowMax,
    RowLogDen,
    GradOut,
    GradQ,
    Q_LEN,
    K_LEN,
    scale,
    *,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    CAUSAL: metile.constexpr = False,
    CAUSAL_OFFSET: metile.constexpr = 0,
    HAS_MASK: metile.constexpr = False,
    BLOCK: metile.constexpr = 32,
):
    """Accumulate dQ in FP32, applying the scale after the key reduction."""
    _validate_contract(D, Q_HEADS, KV_HEADS, CAUSAL, CAUSAL_OFFSET, HAS_MASK, BLOCK)
    work = metile.program_id(0)
    query_index = work % Q_LEN
    query_head = work // Q_LEN % Q_HEADS
    batch = work // (Q_LEN * Q_HEADS)
    kv_head = query_head // (Q_HEADS // KV_HEADS)
    kv_offset = (batch * KV_HEADS + kv_head) * K_LEN * D
    lane = metile.thread_id()
    query_memory = metile.tensor(Q + work * D, shape=(D,), access="read")
    key_memory = metile.tensor(K + kv_offset, shape=(K_LEN, D), access="read")
    value_memory = metile.tensor(V + kv_offset, shape=(K_LEN, D), access="read")
    mask_memory = metile.tensor(Mask + work * K_LEN, shape=(K_LEN,), access="read")
    precise_memory = metile.tensor(OutFloat + work * D, shape=(D,), access="read")
    maximum_memory = metile.tensor(RowMax + work, shape=(1,), access="read")
    denominator_memory = metile.tensor(RowLogDen + work, shape=(1,), access="read")
    grad_memory = metile.tensor(GradOut + work * D, shape=(D,), access="read")
    result_memory = metile.tensor(GradQ + work * D, shape=(D,), access="write")

    query = []
    gradient = []
    delta = 0.0
    for component in range(D // 32):
        dimension = lane + component * 32
        query.append(metile.cast(query_memory.load((dimension,)), "f32"))
        gradient.append(metile.cast(grad_memory.load((dimension,)), "f32"))
        delta = delta + gradient[component] * precise_memory.load((dimension,))
    delta = metile.simd_sum(delta)
    maximum = maximum_memory.load((0,))
    log_denominator = denominator_memory.load((0,))
    outputs = [metile.scalar(0.0) for _ in range(D // 32)]
    for key_index in metile.tile_range(0, K_LEN, 1):
        score = 0.0
        grad_probability = 0.0
        keys = []
        for component in range(D // 32):
            dimension = lane + component * 32
            keys.append(metile.cast(key_memory.load((key_index, dimension)), "f32"))
            value = metile.cast(value_memory.load((key_index, dimension)), "f32")
            score = score + query[component] * keys[component]
            grad_probability = grad_probability + gradient[component] * value
        score = metile.simd_sum(score)
        grad_probability = metile.simd_sum(grad_probability)
        visible = _visible(
            mask_memory,
            query_index,
            key_index,
            K_LEN,
            CAUSAL,
            CAUSAL_OFFSET,
            HAS_MASK,
            (key_index,),
        )
        probability = _probability(score, maximum, log_denominator, scale, visible)
        grad_score = metile.where(visible, probability * (grad_probability - delta), 0.0)
        for component in range(D // 32):
            outputs[component] = outputs[component] + grad_score * keys[component]

    for component in range(D // 32):
        result_memory.store((lane + component * 32,), outputs[component] * scale)


@metile.kernel
def stable_attention_backward_key_value(
    Q,
    K,
    V,
    Mask,
    OutFloat,
    RowMax,
    RowLogDen,
    GradOut,
    GradK,
    GradV,
    Q_LEN,
    K_LEN,
    scale,
    *,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    CAUSAL: metile.constexpr = False,
    CAUSAL_OFFSET: metile.constexpr = 0,
    HAS_MASK: metile.constexpr = False,
    BLOCK: metile.constexpr = 32,
):
    """Own each KV row and sum its grouped query heads without atomics."""
    _validate_contract(D, Q_HEADS, KV_HEADS, CAUSAL, CAUSAL_OFFSET, HAS_MASK, BLOCK)
    work = metile.program_id(0)
    key_index = work % K_LEN
    kv_head = work // K_LEN % KV_HEADS
    batch = work // (K_LEN * KV_HEADS)
    group_size = Q_HEADS // KV_HEADS
    query_offset = batch * Q_HEADS * Q_LEN * D
    statistic_offset = batch * Q_HEADS * Q_LEN
    lane = metile.thread_id()
    query_memory = metile.tensor(Q + query_offset, shape=(Q_HEADS, Q_LEN, D), access="read")
    key_memory = metile.tensor(K + work * D, shape=(D,), access="read")
    value_memory = metile.tensor(V + work * D, shape=(D,), access="read")
    mask_memory = metile.tensor(
        Mask + statistic_offset * K_LEN, shape=(Q_HEADS, Q_LEN, K_LEN), access="read"
    )
    precise_memory = metile.tensor(
        OutFloat + query_offset, shape=(Q_HEADS, Q_LEN, D), access="read"
    )
    maximum_memory = metile.tensor(RowMax + statistic_offset, shape=(Q_HEADS, Q_LEN), access="read")
    denominator_memory = metile.tensor(
        RowLogDen + statistic_offset, shape=(Q_HEADS, Q_LEN), access="read"
    )
    grad_memory = metile.tensor(GradOut + query_offset, shape=(Q_HEADS, Q_LEN, D), access="read")
    grad_key_memory = metile.tensor(GradK + work * D, shape=(D,), access="write")
    grad_value_memory = metile.tensor(GradV + work * D, shape=(D,), access="write")

    keys = []
    values = []
    for component in range(D // 32):
        dimension = lane + component * 32
        keys.append(metile.cast(key_memory.load((dimension,)), "f32"))
        values.append(metile.cast(value_memory.load((dimension,)), "f32"))
    key_outputs = [metile.scalar(0.0) for _ in range(D // 32)]
    value_outputs = [metile.scalar(0.0) for _ in range(D // 32)]
    for grouped_query in metile.tile_range(0, Q_LEN * group_size, 1):
        query_head = kv_head * group_size + grouped_query // Q_LEN
        query_index = grouped_query % Q_LEN
        query = []
        gradient = []
        score = 0.0
        grad_probability = 0.0
        delta = 0.0
        for component in range(D // 32):
            dimension = lane + component * 32
            coordinates = (query_head, query_index, dimension)
            query.append(metile.cast(query_memory.load(coordinates), "f32"))
            gradient.append(metile.cast(grad_memory.load(coordinates), "f32"))
            score = score + query[component] * keys[component]
            grad_probability = grad_probability + gradient[component] * values[component]
            delta = delta + gradient[component] * precise_memory.load(coordinates)
        score = metile.simd_sum(score)
        grad_probability = metile.simd_sum(grad_probability)
        delta = metile.simd_sum(delta)
        maximum = maximum_memory.load((query_head, query_index))
        log_denominator = denominator_memory.load((query_head, query_index))
        visible = _visible(
            mask_memory,
            query_index,
            key_index,
            K_LEN,
            CAUSAL,
            CAUSAL_OFFSET,
            HAS_MASK,
            (query_head, query_index, key_index),
        )
        probability = _probability(score, maximum, log_denominator, scale, visible)
        grad_score = metile.where(visible, probability * (grad_probability - delta), 0.0)
        for component in range(D // 32):
            key_outputs[component] = key_outputs[component] + grad_score * query[component]
            value_outputs[component] = value_outputs[component] + probability * gradient[component]

    for component in range(D // 32):
        dimension = lane + component * 32
        grad_key_memory.store((dimension,), key_outputs[component] * scale)
        grad_value_memory.store((dimension,), value_outputs[component])
