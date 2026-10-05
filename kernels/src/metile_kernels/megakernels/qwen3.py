"""One-threadgroup, one-token Qwen3 decoder, including its tied language-model head.

Launch exactly one threadgroup. Weights, cache and logits use one declared
storage dtype, FP16 or FP32; accumulation uses FP32. Rotary coefficients are
precomputed FP32 constants.
Current-token keys and values remain in threadgroup memory during attention.
Device cache writes happen only after that layer's attention has finished, so
no same-dispatch device-memory communication or grid-wide barrier is needed.

This is a correctness-first full-model megakernel, not a persistent GEMM and
not a claim of superior throughput. A single threadgroup leaves most GPU
execution resources idle. Host preparation and model loading live elsewhere.
"""

import math

import metile


def qwen3_layer_offsets(hidden, intermediate, query_heads, kv_heads, head_dim):
    """Element offsets into each row of the stacked layer-weight buffer."""
    dimensions = (hidden, intermediate, query_heads, kv_heads, head_dim)
    if any(type(dimension) is not int or dimension <= 0 for dimension in dimensions):
        raise ValueError("Qwen3 dimensions must be positive integers")
    if hidden % 32 or intermediate % 32 or head_dim % 32:
        raise ValueError("hidden, intermediate and head_dim must be multiples of 32")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    sizes = (
        ("input_norm", hidden),
        ("q_proj", query_heads * head_dim * hidden),
        ("k_proj", kv_heads * head_dim * hidden),
        ("v_proj", kv_heads * head_dim * hidden),
        ("o_proj", hidden * query_heads * head_dim),
        ("post_norm", hidden),
        ("gate_proj", intermediate * hidden),
        ("up_proj", intermediate * hidden),
        ("down_proj", hidden * intermediate),
        ("q_norm", head_dim),
        ("k_norm", head_dim),
    )
    offsets = {}
    cursor = 0
    for name, size in sizes:
        offsets[name] = cursor
        cursor += size
    offsets["layer_size"] = cursor
    return offsets


def _stored(value, dtype):
    return metile.cast(metile.cast(value, dtype), "f32")


def _silu(value, dtype="f16"):
    stored_value = metile.cast(value, dtype)
    one = metile.cast(1.0, dtype)
    tail = one / (one + metile.exp(metile.abs(stored_value)))
    sigmoid = metile.where(stored_value < 0, tail, one - tail)
    return _stored(stored_value * sigmoid, dtype)


def _rms_norm(
    source, destination, weights, layer, width, epsilon, thread, lane, block, dtype="f16"
):
    square_sum = metile.loop_state(0.0)
    for feature in metile.tile_range(lane, width, 32):
        value = source.load((feature,))
        square_sum.update(square_sum.value + value * value)
    reciprocal = 1.0 / metile.sqrt(metile.simd_sum(square_sum.value) / width + epsilon)
    for feature in metile.tile_range(thread, width, block):
        weight = metile.cast(weights.load((layer, feature)), "f32")
        normalized = _stored(source.load((feature,)) * reciprocal, dtype)
        destination.store((feature,), _stored(normalized * weight, dtype))
    metile.barrier()


def _gemv(source, weights, destination, layer, rows, columns, simdgroup, lane, groups, dtype):
    for row in metile.tile_range(simdgroup, rows, groups):
        accumulator = metile.loop_state(0.0)
        for column in metile.tile_range(lane, columns, 32):
            weight = metile.cast(weights.load((layer, row, column)), "f32")
            accumulator.update(accumulator.value + source.load((column,)) * weight)
        result = _stored(metile.simd_sum(accumulator.value), dtype)
        destination.store((row, lane), result)
    metile.barrier()


def _residual_gemv(
    source, weights, residual, destination, layer, rows, columns, simdgroup, lane, groups, dtype
):
    for row in metile.tile_range(simdgroup, rows, groups):
        accumulator = metile.loop_state(0.0)
        for column in metile.tile_range(lane, columns, 32):
            weight = metile.cast(weights.load((layer, row, column)), "f32")
            accumulator.update(accumulator.value + source.load((column,)) * weight)
        projected = _stored(metile.simd_sum(accumulator.value), dtype)
        residual_row = metile.where(lane == 0, row, rows)
        destination.store((row, lane), _stored(residual.load((residual_row,)) + projected, dtype))
    metile.barrier()


def _normalize_heads(
    values, weights, layer, heads, dimension, epsilon, simdgroup, lane, groups, dtype
):
    for head in metile.tile_range(simdgroup, heads, groups):
        channels = [
            values.load((head, lane * (dimension // 32) + component))
            for component in range(dimension // 32)
        ]
        square_sum = 0.0
        for channel in channels:
            square_sum = square_sum + channel * channel
        reciprocal = 1.0 / metile.sqrt(metile.simd_sum(square_sum) / dimension + epsilon)
        for component, channel in enumerate(channels):
            feature = lane * (dimension // 32) + component
            weight = metile.cast(weights.load((layer, feature)), "f32")
            values.store(
                (head, feature), _stored(_stored(channel * reciprocal, dtype) * weight, dtype)
            )


def _rotate_heads(left, right, rotary, position, heads, dimension, simdgroup, lane, groups, dtype):
    for head in metile.tile_range(simdgroup, heads, groups):
        for component in range((dimension // 2 + 31) // 32):
            feature = lane + component * 32
            first = left.load((head, feature))
            second = right.load((head, feature))
            cosine = rotary.load((position, feature, 0))
            sine = rotary.load((position, feature, 1))
            left.store((head, feature), _stored(first * cosine - second * sine, dtype))
            right.store((head, feature), _stored(second * cosine + first * sine, dtype))


def _attention_update(query, keys, values, maximum, denominator, outputs, scale):
    score = 0.0
    for component in range(len(query)):
        score = score + (query[component] * scale) * keys[component]
    score = metile.simd_sum(score)
    next_maximum = metile.maximum(maximum.value, score)
    previous_factor = metile.exp(maximum.value - next_maximum)
    probability = metile.exp(score - next_maximum)
    denominator.update(denominator.value * previous_factor + probability)
    for component in range(len(query)):
        outputs[component].update(
            outputs[component].value * previous_factor + probability * values[component]
        )
    maximum.update(next_maximum)


@metile.kernel
def qwen3_decode_megakernel(
    Embedding,
    LayerWeights,
    FinalNorm,
    Rotary,
    KVCache,
    Logits,
    token,
    position,
    *,
    HIDDEN: metile.constexpr,
    INTERMEDIATE: metile.constexpr,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    LAYERS: metile.constexpr,
    VOCAB: metile.constexpr,
    MAX_CONTEXT: metile.constexpr,
    BLOCK: metile.constexpr = 256,
    EPS: metile.constexpr = 1e-6,
    STORAGE_DTYPE: metile.constexpr = "f16",
):
    """Decode one token at ``position`` from a cache containing earlier tokens.

    ``Embedding[VOCAB,HIDDEN]`` also supplies the tied output projection.
    ``LayerWeights[LAYERS,layer_size]`` follows ``qwen3_layer_offsets``.
    ``Rotary[MAX_CONTEXT,HEAD_DIM/2,2]`` stores cosine then sine.
    ``KVCache[2,LAYERS,KV_HEADS,MAX_CONTEXT,HEAD_DIM]`` stores keys then values.
    Only the cache slot at ``position`` and ``Logits[VOCAB]`` are mutated.
    The caller must validate token/position bounds and use grid ``(1,)``.
    """
    offsets = qwen3_layer_offsets(HIDDEN, INTERMEDIATE, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    if STORAGE_DTYPE not in ("f16", "f32"):
        raise ValueError(
            "STORAGE_DTYPE must be f16 or f32 and match every weight/cache/output buffer"
        )
    if type(BLOCK) is not int or BLOCK < 32 or BLOCK > 1024 or BLOCK % 32:
        raise ValueError("BLOCK must be a multiple of 32 between 32 and 1024")
    if any(type(count) is not int or count <= 0 for count in (LAYERS, VOCAB, MAX_CONTEXT)):
        raise ValueError("LAYERS, VOCAB and MAX_CONTEXT must be positive integers")
    if type(EPS) not in (int, float) or not math.isfinite(EPS) or EPS <= 0:
        raise ValueError("EPS must be finite and positive")
    query_width = QUERY_HEADS * HEAD_DIM
    kv_width = KV_HEADS * HEAD_DIM
    layer_size = offsets["layer_size"]
    scratch_size = max(2 * HIDDEN + query_width + 2 * kv_width, 2 * HIDDEN + INTERMEDIATE)
    if scratch_size * 4 > 32768:
        raise ValueError("Qwen3 megakernel scratch must fit within 32 KiB")
    memory = metile.shared(scratch_size, dtype="f32")
    embeddings = metile.tensor(Embedding, shape=(VOCAB, HIDDEN), access="read")
    input_norms = metile.tensor(
        LayerWeights + offsets["input_norm"],
        shape=(LAYERS, HIDDEN),
        strides=(layer_size, 1),
        access="read",
    )
    query_weights = metile.tensor(
        LayerWeights + offsets["q_proj"],
        shape=(LAYERS, query_width, HIDDEN),
        strides=(layer_size, HIDDEN, 1),
        access="read",
    )
    key_weights = metile.tensor(
        LayerWeights + offsets["k_proj"],
        shape=(LAYERS, kv_width, HIDDEN),
        strides=(layer_size, HIDDEN, 1),
        access="read",
    )
    value_weights = metile.tensor(
        LayerWeights + offsets["v_proj"],
        shape=(LAYERS, kv_width, HIDDEN),
        strides=(layer_size, HIDDEN, 1),
        access="read",
    )
    output_weights = metile.tensor(
        LayerWeights + offsets["o_proj"],
        shape=(LAYERS, HIDDEN, query_width),
        strides=(layer_size, query_width, 1),
        access="read",
    )
    post_norms = metile.tensor(
        LayerWeights + offsets["post_norm"],
        shape=(LAYERS, HIDDEN),
        strides=(layer_size, 1),
        access="read",
    )
    gate_weights = metile.tensor(
        LayerWeights + offsets["gate_proj"],
        shape=(LAYERS, INTERMEDIATE, HIDDEN),
        strides=(layer_size, HIDDEN, 1),
        access="read",
    )
    up_weights = metile.tensor(
        LayerWeights + offsets["up_proj"],
        shape=(LAYERS, INTERMEDIATE, HIDDEN),
        strides=(layer_size, HIDDEN, 1),
        access="read",
    )
    down_weights = metile.tensor(
        LayerWeights + offsets["down_proj"],
        shape=(LAYERS, HIDDEN, INTERMEDIATE),
        strides=(layer_size, INTERMEDIATE, 1),
        access="read",
    )
    query_norms = metile.tensor(
        LayerWeights + offsets["q_norm"],
        shape=(LAYERS, HEAD_DIM),
        strides=(layer_size, 1),
        access="read",
    )
    key_norms = metile.tensor(
        LayerWeights + offsets["k_norm"],
        shape=(LAYERS, HEAD_DIM),
        strides=(layer_size, 1),
        access="read",
    )
    final_norms = metile.tensor(FinalNorm, shape=(1, HIDDEN), access="read")
    rotary = metile.tensor(Rotary, shape=(MAX_CONTEXT, HEAD_DIM // 2, 2), access="read")
    cache = metile.tensor(KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM))
    logits = metile.tensor(Logits, shape=(VOCAB, 1), access="write")
    residual = metile.tensor(memory, shape=(HIDDEN,))
    residual_writer = metile.tensor(memory, shape=(HIDDEN, 1), access="write")
    normalized = metile.tensor(memory + HIDDEN, shape=(HIDDEN,))
    queries = metile.tensor(memory + 2 * HIDDEN, shape=(QUERY_HEADS, HEAD_DIM))
    query_vector = metile.tensor(memory + 2 * HIDDEN, shape=(query_width,))
    query_writer = metile.tensor(memory + 2 * HIDDEN, shape=(query_width, 1), access="write")
    query_left = metile.tensor(
        memory + 2 * HIDDEN, shape=(QUERY_HEADS, HEAD_DIM // 2), strides=(HEAD_DIM, 1)
    )
    query_right = metile.tensor(
        memory + 2 * HIDDEN + HEAD_DIM // 2,
        shape=(QUERY_HEADS, HEAD_DIM // 2),
        strides=(HEAD_DIM, 1),
    )
    keys = metile.tensor(memory + 2 * HIDDEN + query_width, shape=(KV_HEADS, HEAD_DIM))
    key_writer = metile.tensor(
        memory + 2 * HIDDEN + query_width, shape=(kv_width, 1), access="write"
    )
    key_left = metile.tensor(
        memory + 2 * HIDDEN + query_width, shape=(KV_HEADS, HEAD_DIM // 2), strides=(HEAD_DIM, 1)
    )
    key_right = metile.tensor(
        memory + 2 * HIDDEN + query_width + HEAD_DIM // 2,
        shape=(KV_HEADS, HEAD_DIM // 2),
        strides=(HEAD_DIM, 1),
    )
    values = metile.tensor(memory + 2 * HIDDEN + query_width + kv_width, shape=(KV_HEADS, HEAD_DIM))
    value_writer = metile.tensor(
        memory + 2 * HIDDEN + query_width + kv_width, shape=(kv_width, 1), access="write"
    )
    intermediate = metile.tensor(memory + 2 * HIDDEN, shape=(INTERMEDIATE,))
    intermediate_writer = metile.tensor(
        memory + 2 * HIDDEN, shape=(INTERMEDIATE, 1), access="write"
    )
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    groups = BLOCK // 32
    for feature in metile.tile_range(thread, HIDDEN, BLOCK):
        residual.store((feature,), metile.cast(embeddings.load((token, feature)), "f32"))
    metile.barrier()
    for layer in metile.tile_range(0, LAYERS, 1):
        _rms_norm(
            residual,
            normalized,
            input_norms,
            layer,
            HIDDEN,
            EPS,
            thread,
            lane,
            BLOCK,
            STORAGE_DTYPE,
        )
        _gemv(
            normalized,
            query_weights,
            query_writer,
            layer,
            query_width,
            HIDDEN,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _gemv(
            normalized,
            key_weights,
            key_writer,
            layer,
            kv_width,
            HIDDEN,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _gemv(
            normalized,
            value_weights,
            value_writer,
            layer,
            kv_width,
            HIDDEN,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _normalize_heads(
            queries,
            query_norms,
            layer,
            QUERY_HEADS,
            HEAD_DIM,
            EPS,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _normalize_heads(
            keys, key_norms, layer, KV_HEADS, HEAD_DIM, EPS, simdgroup, lane, groups, STORAGE_DTYPE
        )
        metile.barrier()
        _rotate_heads(
            query_left,
            query_right,
            rotary,
            position,
            QUERY_HEADS,
            HEAD_DIM,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _rotate_heads(
            key_left,
            key_right,
            rotary,
            position,
            KV_HEADS,
            HEAD_DIM,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        metile.barrier()
        for head in metile.tile_range(simdgroup, QUERY_HEADS, groups):
            kv_head = head // (QUERY_HEADS // KV_HEADS)
            query = [
                queries.load((head, lane * (HEAD_DIM // 32) + component))
                for component in range(HEAD_DIM // 32)
            ]
            maximum = metile.loop_state(-1e30)
            denominator = metile.loop_state(0.0)
            outputs = [metile.loop_state(0.0) for _ in range(HEAD_DIM // 32)]
            for previous in metile.tile_range(0, position, 1):
                cached_keys = [
                    metile.cast(
                        cache.load(
                            (0, layer, kv_head, previous, lane * (HEAD_DIM // 32) + component)
                        ),
                        "f32",
                    )
                    for component in range(HEAD_DIM // 32)
                ]
                cached_values = [
                    metile.cast(
                        cache.load(
                            (1, layer, kv_head, previous, lane * (HEAD_DIM // 32) + component)
                        ),
                        "f32",
                    )
                    for component in range(HEAD_DIM // 32)
                ]
                _attention_update(
                    query, cached_keys, cached_values, maximum, denominator, outputs, HEAD_DIM**-0.5
                )
            current_keys = [
                keys.load((kv_head, lane * (HEAD_DIM // 32) + component))
                for component in range(HEAD_DIM // 32)
            ]
            current_values = [
                values.load((kv_head, lane * (HEAD_DIM // 32) + component))
                for component in range(HEAD_DIM // 32)
            ]
            _attention_update(
                query, current_keys, current_values, maximum, denominator, outputs, HEAD_DIM**-0.5
            )
            for component, output in enumerate(outputs):
                queries.store(
                    (head, lane * (HEAD_DIM // 32) + component),
                    _stored(output.value / denominator.value, STORAGE_DTYPE),
                )
        metile.barrier()
        for feature in metile.tile_range(thread, kv_width, BLOCK):
            kv_head = feature // HEAD_DIM
            dimension = feature % HEAD_DIM
            cache.store(
                (0, layer, kv_head, position, dimension),
                metile.cast(keys.load((kv_head, dimension)), STORAGE_DTYPE),
            )
            cache.store(
                (1, layer, kv_head, position, dimension),
                metile.cast(values.load((kv_head, dimension)), STORAGE_DTYPE),
            )
        _residual_gemv(
            query_vector,
            output_weights,
            residual,
            residual_writer,
            layer,
            HIDDEN,
            query_width,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
        _rms_norm(
            residual, normalized, post_norms, layer, HIDDEN, EPS, thread, lane, BLOCK, STORAGE_DTYPE
        )
        for feature in metile.tile_range(simdgroup, INTERMEDIATE, groups):
            gate = metile.loop_state(0.0)
            up = metile.loop_state(0.0)
            for column in metile.tile_range(lane, HIDDEN, 32):
                value = normalized.load((column,))
                gate_weight = metile.cast(gate_weights.load((layer, feature, column)), "f32")
                up_weight = metile.cast(up_weights.load((layer, feature, column)), "f32")
                gate.update(gate.value + value * gate_weight)
                up.update(up.value + value * up_weight)
            gate_value = _stored(metile.simd_sum(gate.value), STORAGE_DTYPE)
            up_value = _stored(metile.simd_sum(up.value), STORAGE_DTYPE)
            activated = _silu(gate_value, STORAGE_DTYPE)
            intermediate_writer.store((feature, lane), _stored(activated * up_value, STORAGE_DTYPE))
        metile.barrier()
        _residual_gemv(
            intermediate,
            down_weights,
            residual,
            residual_writer,
            layer,
            HIDDEN,
            INTERMEDIATE,
            simdgroup,
            lane,
            groups,
            STORAGE_DTYPE,
        )
    _rms_norm(residual, normalized, final_norms, 0, HIDDEN, EPS, thread, lane, BLOCK, STORAGE_DTYPE)
    for vocabulary_index in metile.tile_range(simdgroup, VOCAB, groups):
        accumulator = metile.loop_state(0.0)
        for column in metile.tile_range(lane, HIDDEN, 32):
            weight = metile.cast(embeddings.load((vocabulary_index, column)), "f32")
            accumulator.update(accumulator.value + normalized.load((column,)) * weight)
        logits.store(
            (vocabulary_index, lane), metile.cast(metile.simd_sum(accumulator.value), STORAGE_DTYPE)
        )
