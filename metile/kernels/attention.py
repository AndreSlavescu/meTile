import metile

ATTENTION_DECODE_CONFIGS = [
    metile.Config(BLOCK=32),
    metile.Config(BLOCK=64),
    metile.Config(BLOCK=128),
    metile.Config(BLOCK=256),
    metile.Config(BLOCK=512),
    metile.Config(BLOCK=1024),
]

ATTENTION_FLASH_CONFIGS = [
    metile.Config(BLOCK=32),
    metile.Config(BLOCK=64),
    metile.Config(BLOCK=128),
    metile.Config(BLOCK=256),
]

ATTENTION_PARTIAL_CONFIGS = [
    (256, 256, 128),
    (512, 256, 128),
    (1024, 256, 128),
    (512, 512, 256),
]


@metile.kernel
def attention_decode_kernel(
    Q,
    K,
    V,
    Out,
    N,
    scale,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    BLOCK: metile.constexpr,
):
    """Online decode attention over flattened batched MHA/GQA/MQA tensors."""
    query_index = metile.program_id(0)
    batch = query_index // Q_HEADS
    query_head = query_index % Q_HEADS
    group_size = Q_HEADS // KV_HEADS
    key_value_head = batch * KV_HEADS + query_head // group_size
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    num_simdgroups = BLOCK // 32
    values_per_lane = D // 32

    partial_maxima = metile.shared(BLOCK, dtype="f32")
    partial_sums = metile.shared(BLOCK, dtype="f32")
    partial_outputs = metile.shared(D * num_simdgroups, dtype="f32")

    query_offset = query_index * D
    query = []
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        # Cast to f32 before scaling. Q, K and V are the model's storage dtype, and for
        # bfloat16 a product of two loads rounds to an 8-bit significand before it ever
        # reaches the f32 accumulator. Over D terms that cost 4x MLX's accuracy and was
        # enough to move a logit by 0.43.
        query.append(metile.cast(metile.load(Q + query_offset + dimension), "f32") * scale)

    local_maximum = metile.scalar(-1e30)
    local_sum = metile.scalar(0.0)
    local_outputs = [metile.scalar(0.0) for _ in range(values_per_lane)]

    for token in metile.tile_range(simdgroup, N, num_simdgroups):
        token_offset = (key_value_head * N + token) * D
        score = 0.0
        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            key_offset = token_offset + dimension
            score = score + query[component] * metile.cast(metile.load(K + key_offset), "f32")

        score = metile.simd_sum(score)
        new_maximum = metile.maximum(local_maximum, score)
        old_factor = metile.fast_exp(local_maximum - new_maximum)
        probability = metile.fast_exp(score - new_maximum)
        local_sum = local_sum * old_factor + probability

        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            value_offset = token_offset + dimension
            value = metile.cast(metile.load(V + value_offset), "f32")
            local_outputs[component] = local_outputs[component] * old_factor + probability * value
        local_maximum = new_maximum

    metile.store(partial_maxima + thread, local_maximum)
    metile.store(partial_sums + thread, local_sum)
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        partial_offset = dimension * num_simdgroups + simdgroup
        metile.store(partial_outputs + partial_offset, local_outputs[component])
    metile.barrier()

    active_source = lane < num_simdgroups
    source_simdgroup = metile.minimum(lane, num_simdgroups - 1)
    source_thread = source_simdgroup * 32
    source_maximum = metile.where(
        active_source,
        metile.load(partial_maxima + source_thread),
        -1e30,
    )
    global_maximum = metile.simd_max(source_maximum)
    source_sum = metile.where(
        active_source,
        metile.load(partial_sums + source_thread),
        0.0,
    )
    source_factor = metile.where(
        active_source,
        metile.fast_exp(source_maximum - global_maximum),
        0.0,
    )
    denominator = metile.simd_sum(source_sum * source_factor)

    output_dimensions = []
    output_values = []
    for chunk_offset in range(0, 32, num_simdgroups):
        chunk = simdgroup + chunk_offset
        for component in range(values_per_lane):
            dimension = chunk * values_per_lane + component
            partial_offset = dimension * num_simdgroups + source_simdgroup
            source_output = metile.where(
                active_source,
                metile.load(partial_outputs + partial_offset),
                0.0,
            )
            output_dimensions.append(dimension)
            output_values.append(metile.simd_sum(source_output * source_factor) / denominator)

    writer = lane == 0
    output_offset = query_index * D
    for output_index in range(len(output_values)):
        metile.store(
            Out + output_offset + output_dimensions[output_index],
            output_values[output_index],
            mask=writer,
        )


@metile.kernel
def attention_decode_partial_kernel(
    Q,
    K,
    V,
    PartialOut,
    PartialMax,
    PartialSum,
    N,
    scale,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    NUM_BLOCKS: metile.constexpr,
    TOKENS_PER_BLOCK: metile.constexpr,
    BLOCK: metile.constexpr,
):
    work = metile.program_id(0)
    query_index = work // NUM_BLOCKS
    token_block = work % NUM_BLOCKS
    batch = query_index // Q_HEADS
    query_head = query_index % Q_HEADS
    group_size = Q_HEADS // KV_HEADS
    key_value_head = batch * KV_HEADS + query_head // group_size
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    num_simdgroups = BLOCK // 32
    values_per_lane = D // 32

    partial_maxima = metile.shared(BLOCK, dtype="f32")
    partial_sums = metile.shared(BLOCK, dtype="f32")
    partial_outputs = metile.shared(D * num_simdgroups, dtype="f32")

    query_offset = query_index * D
    query = []
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        # Cast to f32 before scaling. Q, K and V are the model's storage dtype, and for
        # bfloat16 a product of two loads rounds to an 8-bit significand before it ever
        # reaches the f32 accumulator. Over D terms that cost 4x MLX's accuracy and was
        # enough to move a logit by 0.43.
        query.append(metile.cast(metile.load(Q + query_offset + dimension), "f32") * scale)

    local_maximum = metile.scalar(-1e30)
    local_sum = metile.scalar(0.0)
    local_outputs = [metile.scalar(0.0) for _ in range(values_per_lane)]
    token_start = token_block * TOKENS_PER_BLOCK
    token_end = metile.minimum(token_start + TOKENS_PER_BLOCK, N)

    for token in metile.tile_range(token_start + simdgroup, token_end, num_simdgroups):
        token_offset = (key_value_head * N + token) * D
        score = 0.0
        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            score = score + query[component] * metile.cast(
                metile.load(K + token_offset + dimension), "f32"
            )

        score = metile.simd_sum(score)
        new_maximum = metile.maximum(local_maximum, score)
        old_factor = metile.fast_exp(local_maximum - new_maximum)
        probability = metile.fast_exp(score - new_maximum)
        local_sum = local_sum * old_factor + probability

        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            value = metile.cast(metile.load(V + token_offset + dimension), "f32")
            local_outputs[component] = local_outputs[component] * old_factor + probability * value
        local_maximum = new_maximum

    metile.store(partial_maxima + thread, local_maximum)
    metile.store(partial_sums + thread, local_sum)
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        partial_offset = dimension * num_simdgroups + simdgroup
        metile.store(partial_outputs + partial_offset, local_outputs[component])
    metile.barrier()

    active_source = lane < num_simdgroups
    source_simdgroup = metile.minimum(lane, num_simdgroups - 1)
    source_thread = source_simdgroup * 32
    source_maximum = metile.where(
        active_source,
        metile.load(partial_maxima + source_thread),
        -1e30,
    )
    block_maximum = metile.simd_max(source_maximum)
    source_sum = metile.where(
        active_source,
        metile.load(partial_sums + source_thread),
        0.0,
    )
    source_factor = metile.where(
        active_source,
        metile.fast_exp(source_maximum - block_maximum),
        0.0,
    )
    block_sum = metile.simd_sum(source_sum * source_factor)

    output_dimensions = []
    output_values = []
    for chunk_offset in range(0, 32, num_simdgroups):
        chunk = simdgroup + chunk_offset
        for component in range(values_per_lane):
            dimension = chunk * values_per_lane + component
            source_offset = dimension * num_simdgroups + source_simdgroup
            source_output = metile.where(
                active_source,
                metile.load(partial_outputs + source_offset),
                0.0,
            )
            output_dimensions.append(dimension)
            output_values.append(metile.simd_sum(source_output * source_factor))

    writer = lane == 0
    statistics_offset = work * 32 + simdgroup
    metile.store(PartialMax + statistics_offset, block_maximum, mask=writer)
    metile.store(PartialSum + statistics_offset, block_sum, mask=writer)
    output_offset = work * D
    for output_index in range(len(output_values)):
        metile.store(
            PartialOut + output_offset + output_dimensions[output_index],
            output_values[output_index],
            mask=writer,
        )


@metile.kernel
def attention_decode_merge_kernel(
    PartialOut,
    PartialMax,
    PartialSum,
    Out,
    D: metile.constexpr,
    NUM_BLOCKS: metile.constexpr,
    BLOCK: metile.constexpr,
):
    head = metile.program_id(0)
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    num_simdgroups = BLOCK // 32
    dimensions_per_simdgroup = D // num_simdgroups

    active_source = lane < NUM_BLOCKS
    source_block = metile.minimum(lane, NUM_BLOCKS - 1)
    source_work = head * NUM_BLOCKS + source_block
    statistics_offset = source_work * 32
    source_maximum = metile.where(
        active_source,
        metile.load(PartialMax + statistics_offset),
        -1e30,
    )
    global_maximum = metile.simd_max(source_maximum)
    source_sum = metile.where(
        active_source,
        metile.load(PartialSum + statistics_offset),
        0.0,
    )
    source_factor = metile.where(
        active_source,
        metile.fast_exp(source_maximum - global_maximum),
        0.0,
    )
    denominator = metile.simd_sum(source_sum * source_factor)

    output_dimensions = []
    output_values = []
    for component in range(dimensions_per_simdgroup):
        dimension = simdgroup * dimensions_per_simdgroup + component
        partial_offset = source_work * D + dimension
        source_output = metile.where(
            active_source,
            metile.load(PartialOut + partial_offset),
            0.0,
        )
        output_dimensions.append(dimension)
        output_values.append(metile.simd_sum(source_output * source_factor) / denominator)

    writer = lane == 0
    output_offset = head * D
    for output_index in range(len(output_values)):
        metile.store(
            Out + output_offset + output_dimensions[output_index],
            output_values[output_index],
            mask=writer,
        )


attention_decode_single_pass = metile.autotune(
    configs=ATTENTION_DECODE_CONFIGS,
    key=["N", "D", "Q_HEADS", "KV_HEADS"],
    verbose=False,
)(attention_decode_kernel)


@metile.kernel
def attention_flash_kernel(
    Q,
    K,
    V,
    Out,
    Q_LEN,
    K_LEN,
    scale,
    D: metile.constexpr,
    Q_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    CAUSAL: metile.constexpr,
    BLOCK: metile.constexpr,
):
    """Exact row-tiled attention using a mergeable online-softmax state."""
    work = metile.program_id(0)
    query_token = work % Q_LEN
    query_index = work // Q_LEN
    batch = query_index // Q_HEADS
    query_head = query_index % Q_HEADS
    group_size = Q_HEADS // KV_HEADS
    key_value_head = batch * KV_HEADS + query_head // group_size
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    num_simdgroups = BLOCK // 32
    values_per_lane = D // 32

    partial_maxima = metile.shared(BLOCK, dtype="f32")
    partial_sums = metile.shared(BLOCK, dtype="f32")
    partial_outputs = metile.shared(D * num_simdgroups, dtype="f32")

    query_offset = work * D
    query = []
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        # Cast to f32 before scaling. Q, K and V are the model's storage dtype, and for
        # bfloat16 a product of two loads rounds to an 8-bit significand before it ever
        # reaches the f32 accumulator. Over D terms that cost 4x MLX's accuracy and was
        # enough to move a logit by 0.43.
        query.append(metile.cast(metile.load(Q + query_offset + dimension), "f32") * scale)

    local_maximum = metile.scalar(-1e30)
    local_sum = metile.scalar(0.0)
    local_outputs = [metile.scalar(0.0) for _ in range(values_per_lane)]
    token_end = K_LEN
    if CAUSAL:
        token_end = K_LEN - Q_LEN + query_token + 1

    for token in metile.tile_range(simdgroup, token_end, num_simdgroups):
        token_offset = (key_value_head * K_LEN + token) * D
        score = 0.0
        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            score = score + query[component] * metile.cast(
                metile.load(K + token_offset + dimension), "f32"
            )

        score = metile.simd_sum(score)
        new_maximum = metile.maximum(local_maximum, score)
        old_factor = metile.fast_exp(local_maximum - new_maximum)
        probability = metile.fast_exp(score - new_maximum)
        local_sum = local_sum * old_factor + probability
        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            value = metile.cast(metile.load(V + token_offset + dimension), "f32")
            local_outputs[component] = local_outputs[component] * old_factor + probability * value
        local_maximum = new_maximum

    metile.store(partial_maxima + thread, local_maximum)
    metile.store(partial_sums + thread, local_sum)
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        partial_offset = dimension * num_simdgroups + simdgroup
        metile.store(partial_outputs + partial_offset, local_outputs[component])
    metile.barrier()

    active_source = lane < num_simdgroups
    source_simdgroup = metile.minimum(lane, num_simdgroups - 1)
    source_thread = source_simdgroup * 32
    source_maximum = metile.where(
        active_source,
        metile.load(partial_maxima + source_thread),
        -1e30,
    )
    global_maximum = metile.simd_max(source_maximum)
    source_sum = metile.where(
        active_source,
        metile.load(partial_sums + source_thread),
        0.0,
    )
    source_factor = metile.where(
        active_source,
        metile.fast_exp(source_maximum - global_maximum),
        0.0,
    )
    denominator = metile.simd_sum(source_sum * source_factor)

    output_dimensions = []
    output_values = []
    for chunk_offset in range(0, 32, num_simdgroups):
        chunk = simdgroup + chunk_offset
        for component in range(values_per_lane):
            dimension = chunk * values_per_lane + component
            partial_offset = dimension * num_simdgroups + source_simdgroup
            source_output = metile.where(
                active_source,
                metile.load(partial_outputs + partial_offset),
                0.0,
            )
            output_dimensions.append(dimension)
            output_values.append(metile.simd_sum(source_output * source_factor) / denominator)

    writer = lane == 0
    for output_index in range(len(output_values)):
        metile.store(
            Out + query_offset + output_dimensions[output_index],
            output_values[output_index],
            mask=writer,
        )


__all__ = [
    "ATTENTION_DECODE_CONFIGS",
    "ATTENTION_FLASH_CONFIGS",
    "ATTENTION_PARTIAL_CONFIGS",
    "attention_decode_kernel",
    "attention_decode_merge_kernel",
    "attention_decode_partial_kernel",
    "attention_decode_single_pass",
    "attention_flash_kernel",
]
