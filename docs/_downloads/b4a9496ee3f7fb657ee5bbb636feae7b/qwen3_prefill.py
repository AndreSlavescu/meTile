"""Token-batched Qwen3 row operations around compiler-lowered projections.

Rows are token positions within a fixed-size chunk. ``Control`` contains
``[start_position, valid_rows]``; the caller validates both against cache
capacity. Embedding zeroes padding rows, attention zeroes padded outputs,
and QK/RoPE never writes padded tokens into the KV cache. Separate ordered
dispatches make all current-chunk keys and values visible before attention.
Each query attends only through its own position, including earlier chunks.

All buffers are disjoint except the deliberately shared, persistent KV cache.
Projections use their own compiler-lowered matrix kernels. These row stages
declare tensors explicitly and use no raw Metal or cross-threadgroup barriers.
"""

import metile
from metile_kernels.megakernels.qwen3 import (
    _attention_update,
    _silu,
    _stored,
    qwen3_layer_offsets,
)
from metile_kernels.megakernels.qwen3_staged import _row, _validate, _validate_epsilon


@metile.kernel
def qwen3_prefill_embedding(
    Tokens,
    Control,
    Embedding,
    Hidden,
    *,
    CHUNK: metile.constexpr,
    HIDDEN: metile.constexpr,
    VOCAB: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Copy token embeddings and zero padding; grid ``ceil(CHUNK * HIDDEN / BLOCK)``."""
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, HIDDEN, VOCAB)
    tokens = metile.tensor(Tokens, shape=(CHUNK,), access="read")
    control = metile.tensor(Control, shape=(2,), access="read")
    embeddings = metile.tensor(Embedding, shape=(VOCAB, HIDDEN), access="read")
    hidden = metile.tensor(Hidden, shape=(CHUNK, HIDDEN), access="write")
    index = metile.program_id(0) * BLOCK + metile.thread_id()
    row = index // HIDDEN
    feature = index % HIDDEN
    valid = row < control.load((1,))
    token = metile.where(valid, tokens.load((row,)), 0)
    hidden.store((row, feature), metile.where(valid, embeddings.load((token, feature)), 0.0))


@metile.kernel
def qwen3_prefill_rmsnorm(
    Source,
    NormWeights,
    Destination,
    layer,
    *,
    CHUNK: metile.constexpr,
    WIDTH: metile.constexpr,
    WEIGHT_OFFSET: metile.constexpr = 0,
    WEIGHT_STRIDE: metile.constexpr = 0,
    EPS: metile.constexpr = 1e-6,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    FUSED_ARITHMETIC: metile.constexpr = False,
):
    """Normalize each token with contiguous four-value partials; grid ``(CHUNK,)``."""
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, WIDTH)
    _validate_epsilon(EPS)
    if type(FUSED_ARITHMETIC) is not bool:
        raise ValueError("FUSED_ARITHMETIC must be boolean")
    source = metile.tensor(Source, shape=(CHUNK, WIDTH), access="read")
    weights = metile.tensor(
        NormWeights + WEIGHT_OFFSET + layer * WEIGHT_STRIDE, shape=(WIDTH,), access="read"
    )
    destination = metile.tensor(Destination, shape=(CHUNK, WIDTH), access="write")
    reduction_groups = min(metile.cdiv(WIDTH, 128), 32)
    partial_sums = metile.tensor(
        metile.shared(reduction_groups, dtype="f32"), shape=(reduction_groups, 1)
    )
    row = metile.program_id(0)
    lane = metile.simd_lane_id()
    for group in metile.tile_range(metile.thread_id() // 32, reduction_groups, BLOCK // 32):
        square_sum = metile.loop_state(0.0)
        for start in metile.tile_range(0, WIDTH, reduction_groups * 128):
            for component in range(4):
                feature = start + group * 128 + lane * 4 + component
                value = metile.cast(source.load((row, feature)), "f32")
                square_sum.update(
                    metile.fma(value, value, square_sum.value)
                    if FUSED_ARITHMETIC
                    else square_sum.value + value * value
                )
        partial_sums.store((group, lane), metile.simd_sum(square_sum.value))
    metile.barrier()
    total = metile.simd_sum(partial_sums.load((lane, 0)))
    reciprocal = metile.rsqrt(total / WIDTH + EPS)
    for feature in metile.tile_range(metile.thread_id(), WIDTH, BLOCK):
        value = metile.cast(source.load((row, feature)), "f32")
        weight = metile.cast(weights.load((feature,)), "f32")
        destination.store(
            (row, feature),
            _stored(_stored(value * reciprocal, STORAGE_DTYPE) * weight, STORAGE_DTYPE),
        )


@metile.kernel
def qwen3_prefill_qk_rope(
    QKV,
    LayerWeights,
    Rotary,
    Control,
    Queries,
    KVCache,
    layer,
    *,
    CHUNK: metile.constexpr,
    HIDDEN: metile.constexpr,
    INTERMEDIATE: metile.constexpr,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    LAYERS: metile.constexpr,
    MAX_CONTEXT: metile.constexpr,
    EPS: metile.constexpr = 1e-6,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    DECODE: metile.constexpr = False,
    FUSED_ARITHMETIC: metile.constexpr = False,
):
    """Normalize/rotate QK and cache valid rows before causal attention.

    Grid: ``(ceil((QUERY_HEADS + KV_HEADS) / (BLOCK / 32)), CHUNK)``.
    QKV concatenates Q, K, V within each token row. Query output is ``[CHUNK,Q*D]``.
    ``DECODE=True`` requires ``CHUNK=1`` and uses ``Control[token,position]``.
    """
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, LAYERS, MAX_CONTEXT)
    _validate_epsilon(EPS)
    if type(FUSED_ARITHMETIC) is not bool:
        raise ValueError("FUSED_ARITHMETIC must be boolean")
    if type(DECODE) is not bool or (DECODE and CHUNK != 1):
        raise ValueError("DECODE must be boolean and requires CHUNK=1")
    offsets = qwen3_layer_offsets(HIDDEN, INTERMEDIATE, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    heads = QUERY_HEADS + KV_HEADS
    half = HEAD_DIM // 2
    qkv_width = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM
    query_width = QUERY_HEADS * HEAD_DIM
    qk_values = metile.tensor(
        QKV, shape=(CHUNK, heads, HEAD_DIM), strides=(qkv_width, HEAD_DIM, 1), access="read"
    )
    qk_left = metile.tensor(
        QKV, shape=(CHUNK, heads, half), strides=(qkv_width, HEAD_DIM, 1), access="read"
    )
    qk_right = metile.tensor(
        QKV + half, shape=(CHUNK, heads, half), strides=(qkv_width, HEAD_DIM, 1), access="read"
    )
    values = metile.tensor(
        QKV + heads * HEAD_DIM,
        shape=(CHUNK, KV_HEADS, HEAD_DIM),
        strides=(qkv_width, HEAD_DIM, 1),
        access="read",
    )
    norm_weights = metile.tensor(
        LayerWeights + layer * offsets["layer_size"] + offsets["q_norm"],
        shape=(2, HEAD_DIM),
        access="read",
    )
    rotary = metile.tensor(Rotary, shape=(MAX_CONTEXT, half, 2), access="read")
    control = metile.tensor(Control, shape=(2,), access="read")
    query_left = metile.tensor(
        Queries,
        shape=(CHUNK, QUERY_HEADS, half),
        strides=(query_width, HEAD_DIM, 1),
        access="write",
    )
    query_right = metile.tensor(
        Queries + half,
        shape=(CHUNK, QUERY_HEADS, half),
        strides=(query_width, HEAD_DIM, 1),
        access="write",
    )
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="write"
    )
    row = metile.program_id(1)
    head = _row(BLOCK)
    lane = metile.simd_lane_id()
    if DECODE:
        position = control.load((1,))
        valid = row < 1
    else:
        position = control.load((0,)) + row
        valid = row < control.load((1,))
    cache_head = metile.where(valid, head - QUERY_HEADS, KV_HEADS)
    weight_row = metile.where(head < QUERY_HEADS, 0, 1)
    left = [
        metile.cast(qk_left.load((row, head, lane + component * 32)), "f32")
        for component in range((half + 31) // 32)
    ]
    right = [
        metile.cast(qk_right.load((row, head, lane + component * 32)), "f32")
        for component in range((half + 31) // 32)
    ]
    reduction_groups = min(metile.cdiv(HEAD_DIM, 128), 32)
    partial_sums = 0.0
    for group in range(reduction_groups):
        square_sum = metile.loop_state(0.0)
        for start in metile.tile_range(0, HEAD_DIM, reduction_groups * 128):
            for component in range(4):
                feature = start + group * 128 + lane * 4 + component
                value = metile.cast(qk_values.load((row, head, feature)), "f32")
                square_sum.update(
                    metile.fma(value, value, square_sum.value)
                    if FUSED_ARITHMETIC
                    else square_sum.value + value * value
                )
        partial_sums = partial_sums + metile.where(
            lane == group, metile.simd_sum(square_sum.value), 0.0
        )
    reciprocal = metile.rsqrt(metile.simd_sum(partial_sums) / HEAD_DIM + EPS)
    for component, (first, second) in enumerate(zip(left, right, strict=True)):
        feature = lane + component * 32
        left_weight = metile.cast(norm_weights.load((weight_row, feature)), "f32")
        right_weight = metile.cast(norm_weights.load((weight_row, feature + half)), "f32")
        first = _stored(_stored(first * reciprocal, STORAGE_DTYPE) * left_weight, STORAGE_DTYPE)
        second = _stored(_stored(second * reciprocal, STORAGE_DTYPE) * right_weight, STORAGE_DTYPE)
        cosine = rotary.load((position, feature, 0))
        sine = rotary.load((position, feature, 1))
        rotated_left = _stored(
            metile.fma(first, cosine, (0.0 - second) * sine)
            if FUSED_ARITHMETIC
            else first * cosine - second * sine,
            STORAGE_DTYPE,
        )
        rotated_right = _stored(
            metile.fma(first, sine, second * cosine)
            if FUSED_ARITHMETIC
            else second * cosine + first * sine,
            STORAGE_DTYPE,
        )
        query_left.store((row, head, feature), metile.where(valid, rotated_left, 0.0))
        query_right.store((row, head, feature), metile.where(valid, rotated_right, 0.0))
        key_feature = metile.where(feature < half, feature, HEAD_DIM)
        cache.store((0, layer, cache_head, position, key_feature), rotated_left)
        cache.store((0, layer, cache_head, position, feature + half), rotated_right)
    for component in range(HEAD_DIM // 32):
        feature = lane + component * 32
        cache.store(
            (1, layer, cache_head, position, feature),
            values.load((row, head - QUERY_HEADS, feature)),
        )


@metile.kernel
def qwen3_prefill_attention(
    Queries,
    KVCache,
    Control,
    Attention,
    layer,
    *,
    CHUNK: metile.constexpr,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    LAYERS: metile.constexpr,
    MAX_CONTEXT: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    DECODE: metile.constexpr = False,
):
    """Causal online softmax with parallel key partitions and a stable merge.

    Grid: ``(QUERY_HEADS, CHUNK)``. Each threadgroup owns one token/query-head;
    SIMD groups stride over different keys, then merge in threadgroup memory.
    ``DECODE=True`` requires ``CHUNK=1`` and reads the existing staged decoder's
    ``Control[token,position]`` instead of prefill's ``[start,valid_rows]``.
    """
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, QUERY_HEADS, KV_HEADS, HEAD_DIM, LAYERS, MAX_CONTEXT)
    if HEAD_DIM % 32 or QUERY_HEADS % KV_HEADS:
        raise ValueError("HEAD_DIM must be divisible by 32 and QUERY_HEADS by KV_HEADS")
    if type(DECODE) is not bool or (DECODE and CHUNK != 1):
        raise ValueError("DECODE must be boolean and requires CHUNK=1")
    groups = BLOCK // 32
    maxima_memory = metile.shared(groups, dtype="f32")
    sums_memory = metile.shared(groups, dtype="f32")
    outputs_memory = metile.shared(HEAD_DIM * groups, dtype="f32")
    queries = metile.tensor(Queries, shape=(CHUNK, QUERY_HEADS, HEAD_DIM), access="read")
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="read"
    )
    control = metile.tensor(Control, shape=(2,), access="read")
    attention = metile.tensor(Attention, shape=(CHUNK, QUERY_HEADS, HEAD_DIM, 1), access="write")
    partial_maxima = metile.tensor(maxima_memory, shape=(groups, 1))
    partial_sums = metile.tensor(sums_memory, shape=(groups, 1))
    partial_outputs = metile.tensor(outputs_memory, shape=(HEAD_DIM, groups))
    row = metile.program_id(1)
    head = metile.program_id(0)
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    lane = metile.simd_lane_id()
    simdgroup = metile.thread_id() // 32
    if DECODE:
        limit = control.load((1,)) + 1
    else:
        limit = metile.where(row < control.load((1,)), control.load((0,)) + row + 1, 0)
    query = [
        metile.cast(queries.load((row, head, lane * (HEAD_DIM // 32) + component)), "f32")
        for component in range(HEAD_DIM // 32)
    ]
    maximum = metile.loop_state(-1e30)
    denominator = metile.loop_state(0.0)
    outputs = [metile.loop_state(0.0) for _ in range(HEAD_DIM // 32)]
    for previous in metile.tile_range(simdgroup, limit, groups):
        keys = [
            metile.cast(
                cache.load((0, layer, kv_head, previous, lane * (HEAD_DIM // 32) + component)),
                "f32",
            )
            for component in range(HEAD_DIM // 32)
        ]
        values = [
            metile.cast(
                cache.load((1, layer, kv_head, previous, lane * (HEAD_DIM // 32) + component)),
                "f32",
            )
            for component in range(HEAD_DIM // 32)
        ]
        _attention_update(query, keys, values, maximum, denominator, outputs, HEAD_DIM**-0.5)
    partial_maxima.store((simdgroup, lane), maximum.value)
    partial_sums.store((simdgroup, lane), denominator.value)
    for component, output in enumerate(outputs):
        partial_outputs.store((lane * (HEAD_DIM // 32) + component, simdgroup), output.value)
    metile.barrier()
    source_maximum = metile.where(lane < groups, partial_maxima.load((lane, 0)), -1e30)
    global_maximum = metile.simd_max(source_maximum)
    source_factor = metile.where(lane < groups, metile.exp(source_maximum - global_maximum), 0.0)
    source_sum = partial_sums.load((lane, 0))
    merged_denominator = metile.maximum(metile.simd_sum(source_sum * source_factor), 1e-30)
    for feature in metile.tile_range(simdgroup, HEAD_DIM, groups):
        partial = partial_outputs.load((feature, lane))
        merged = metile.simd_sum(partial * source_factor) / merged_denominator
        attention.store(
            (row, head, feature, lane),
            _stored(merged, STORAGE_DTYPE),
        )


@metile.kernel
def qwen3_prefill_residual(
    Projected,
    Residual,
    Destination,
    *,
    CHUNK: metile.constexpr,
    WIDTH: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Add storage-rounded projection and residual; grid ``ceil(CHUNK * WIDTH / BLOCK)``."""
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, WIDTH)
    projected = metile.tensor(Projected, shape=(CHUNK, WIDTH), access="read")
    residual = metile.tensor(Residual, shape=(CHUNK, WIDTH), access="read")
    destination = metile.tensor(Destination, shape=(CHUNK, WIDTH), access="write")
    index = metile.program_id(0) * BLOCK + metile.thread_id()
    row, feature = index // WIDTH, index % WIDTH
    value = metile.cast(projected.load((row, feature)), "f32")
    previous = metile.cast(residual.load((row, feature)), "f32")
    destination.store(
        (row, feature), _stored(_stored(value, STORAGE_DTYPE) + previous, STORAGE_DTYPE)
    )


@metile.kernel
def qwen3_prefill_swiglu(
    GateUp,
    Intermediate,
    *,
    CHUNK: metile.constexpr,
    INTERMEDIATE: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    FUSED_ARITHMETIC: metile.constexpr = False,
):
    """Apply SwiGLU to concatenated ``[CHUNK,2*I]`` projections; grid ``ceil(CHUNK*I/BLOCK)``."""
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, INTERMEDIATE)
    if type(FUSED_ARITHMETIC) is not bool:
        raise ValueError("FUSED_ARITHMETIC must be boolean")
    gates = metile.tensor(
        GateUp, shape=(CHUNK, INTERMEDIATE), strides=(2 * INTERMEDIATE, 1), access="read"
    )
    ups = metile.tensor(
        GateUp + INTERMEDIATE,
        shape=(CHUNK, INTERMEDIATE),
        strides=(2 * INTERMEDIATE, 1),
        access="read",
    )
    destination = metile.tensor(Intermediate, shape=(CHUNK, INTERMEDIATE), access="write")
    index = metile.program_id(0) * BLOCK + metile.thread_id()
    row, feature = index // INTERMEDIATE, index % INTERMEDIATE
    gate = _stored(metile.cast(gates.load((row, feature)), "f32"), STORAGE_DTYPE)
    up = _stored(metile.cast(ups.load((row, feature)), "f32"), STORAGE_DTYPE)
    if FUSED_ARITHMETIC:
        stored_gate = metile.cast(gate, STORAGE_DTYPE)
        one = metile.cast(1.0, STORAGE_DTYPE)
        tail = one / (one + metile.fast_exp(metile.abs(stored_gate)))
        sigmoid = metile.where(stored_gate < 0, tail, one - tail)
        activated = _stored(stored_gate * sigmoid, STORAGE_DTYPE)
    else:
        activated = _silu(gate, STORAGE_DTYPE)
    destination.store((row, feature), _stored(activated * up, STORAGE_DTYPE))


@metile.kernel
def qwen3_prefill_last_hidden(
    Source,
    Control,
    Hidden,
    *,
    CHUNK: metile.constexpr,
    HIDDEN: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Copy the last valid row for decoder final norm/head; grid ``ceil(HIDDEN / BLOCK)``."""
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, HIDDEN)
    source = metile.tensor(Source, shape=(CHUNK, HIDDEN), access="read")
    control = metile.tensor(Control, shape=(2,), access="read")
    hidden = metile.tensor(Hidden, shape=(HIDDEN,), access="write")
    feature = metile.program_id(0) * BLOCK + metile.thread_id()
    hidden.store((feature,), source.load((control.load((1,)) - 1, feature)))
