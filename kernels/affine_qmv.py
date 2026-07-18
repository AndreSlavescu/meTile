import metile


@metile.kernel
def affine_qmv(
    X,
    W,
    Scales,
    Biases,
    Out,
    K,
    N,
    GROUP_SIZE: metile.constexpr,
    BITS: metile.constexpr,
    BLOCK: metile.constexpr,
):
    """Affine packed-weight matrix-vector product for decode-shaped rows."""
    thread = metile.thread_id()
    lane = thread % 32
    simdgroup = thread // 32
    outputs_per_threadgroup = BLOCK // 32
    output_index = metile.program_id(0) * outputs_per_threadgroup + simdgroup
    row = output_index // N
    column = output_index % N
    values_per_word = 32 // BITS
    words_per_row = K // values_per_word
    groups_per_row = K // GROUP_SIZE
    words_per_group = GROUP_SIZE // values_per_word
    accumulator = metile.scalar(0.0)

    for word_index in metile.tile_range(lane, words_per_row, 32):
        packed = metile.load(W + column * words_per_row + word_index)
        group = column * groups_per_row + word_index // words_per_group
        scale = metile.cast(metile.load(Scales + group), "f32")
        bias = metile.cast(metile.load(Biases + group), "f32")
        input_base = row * K + word_index * values_per_word
        for element in range(values_per_word):
            quantized = metile.cast((packed >> (element * BITS)) & ((1 << BITS) - 1), "f32")
            value = metile.cast(metile.load(X + input_base + element), "f32")
            accumulator = accumulator + value * (quantized * scale + bias)

    result = metile.simd_sum(accumulator)
    metile.store(Out + output_index, result, mask=lane == 0)


@metile.kernel
def affine_swiglu_qmv(
    X,
    GateW,
    GateScales,
    GateBiases,
    UpW,
    UpScales,
    UpBiases,
    Out,
    K,
    N,
    GROUP_SIZE: metile.constexpr,
    BITS: metile.constexpr,
    BLOCK: metile.constexpr,
):
    """Fuse two affine QMV projections with a SwiGLU epilogue."""
    thread = metile.thread_id()
    lane = thread % 32
    simdgroup = thread // 32
    outputs_per_threadgroup = BLOCK // 32
    output_index = metile.program_id(0) * outputs_per_threadgroup + simdgroup
    row = output_index // N
    column = output_index % N
    values_per_word = 32 // BITS
    words_per_row = K // values_per_word
    groups_per_row = K // GROUP_SIZE
    words_per_group = GROUP_SIZE // values_per_word
    gate_accumulator = metile.scalar(0.0)
    up_accumulator = metile.scalar(0.0)

    for word_index in metile.tile_range(lane, words_per_row, 32):
        packed_offset = column * words_per_row + word_index
        gate_packed = metile.load(GateW + packed_offset)
        up_packed = metile.load(UpW + packed_offset)
        group = column * groups_per_row + word_index // words_per_group
        gate_scale = metile.cast(metile.load(GateScales + group), "f32")
        gate_bias = metile.cast(metile.load(GateBiases + group), "f32")
        up_scale = metile.cast(metile.load(UpScales + group), "f32")
        up_bias = metile.cast(metile.load(UpBiases + group), "f32")
        input_base = row * K + word_index * values_per_word
        for element in range(values_per_word):
            shift = element * BITS
            mask = (1 << BITS) - 1
            gate_quantized = metile.cast((gate_packed >> shift) & mask, "f32")
            up_quantized = metile.cast((up_packed >> shift) & mask, "f32")
            value = metile.cast(metile.load(X + input_base + element), "f32")
            gate_accumulator = gate_accumulator + value * (gate_quantized * gate_scale + gate_bias)
            up_accumulator = up_accumulator + value * (up_quantized * up_scale + up_bias)

    gate = metile.simd_sum(gate_accumulator)
    up = metile.simd_sum(up_accumulator)
    result = gate / (1.0 + metile.fast_exp(0.0 - gate)) * up
    metile.store(Out + output_index, result, mask=lane == 0)
