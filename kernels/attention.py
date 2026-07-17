import metile

ATTENTION_DECODE_CONFIGS = [
    metile.Config(BLOCK=64),
    metile.Config(BLOCK=128),
    metile.Config(BLOCK=256),
    metile.Config(BLOCK=512),
    metile.Config(BLOCK=1024),
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
    BLOCK: metile.constexpr,
):
    """Online MHA decode attention over flattened ``[head, token, dim]`` tensors."""
    head = metile.program_id(0)
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    num_simdgroups = BLOCK // 32
    values_per_lane = D // 32

    partial_maxima = metile.shared(BLOCK, dtype="f32")
    partial_sums = metile.shared(BLOCK, dtype="f32")
    partial_outputs = metile.shared(D * num_simdgroups, dtype="f32")

    query_offset = head * D
    query = []
    for component in range(values_per_lane):
        dimension = lane * values_per_lane + component
        query.append(metile.load(Q + query_offset + dimension) * scale)

    local_maximum = metile.scalar(-1e30)
    local_sum = metile.scalar(0.0)
    local_outputs = [metile.scalar(0.0) for _ in range(values_per_lane)]

    for token in metile.tile_range(simdgroup, N, num_simdgroups):
        token_offset = (head * N + token) * D
        score = 0.0
        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            key_offset = token_offset + dimension
            score = score + query[component] * metile.load(K + key_offset)

        score = metile.simd_sum(score)
        new_maximum = metile.maximum(local_maximum, score)
        old_factor = metile.fast_exp(local_maximum - new_maximum)
        probability = metile.fast_exp(score - new_maximum)
        local_sum = local_sum * old_factor + probability

        for component in range(values_per_lane):
            dimension = lane * values_per_lane + component
            value_offset = token_offset + dimension
            value = metile.load(V + value_offset)
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
    output_offset = head * D
    for output_index in range(len(output_values)):
        metile.store(
            Out + output_offset + output_dimensions[output_index],
            output_values[output_index],
            mask=writer,
        )


attention_decode = metile.autotune(
    configs=ATTENTION_DECODE_CONFIGS,
    key=["N", "D"],
    verbose=False,
)(attention_decode_kernel)


__all__ = ["ATTENTION_DECODE_CONFIGS", "attention_decode", "attention_decode_kernel"]
