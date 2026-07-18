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
    OUTPUTS_PER_SIMDGROUP: metile.constexpr,
    HALF_DECODE: metile.constexpr,
):
    """Affine packed-weight matrix-vector product for decode-shaped rows."""
    thread = metile.thread_id()
    lane = thread % 32
    simdgroup = thread // 32
    simdgroups_per_threadgroup = BLOCK // 32
    outputs_per_threadgroup = simdgroups_per_threadgroup * OUTPUTS_PER_SIMDGROUP
    output_base = metile.program_id(0) * outputs_per_threadgroup + simdgroup * OUTPUTS_PER_SIMDGROUP
    row = output_base // N
    column_base = output_base % N
    values_per_word = 32 // BITS
    words_per_row = K // values_per_word
    groups_per_row = K // GROUP_SIZE
    words_per_group = GROUP_SIZE // values_per_word
    decode_dtype = "f16" if HALF_DECODE else "f32"
    accumulators = [metile.scalar(0.0) for _ in range(OUTPUTS_PER_SIMDGROUP)]

    for word_index in metile.tile_range(lane, words_per_row, 32):
        input_base = row * K + word_index * values_per_word
        input_values = [
            metile.cast(metile.load(X + input_base + element), decode_dtype)
            for element in range(values_per_word)
        ]
        for output_offset in range(OUTPUTS_PER_SIMDGROUP):
            column = column_base + output_offset
            packed = metile.load(W + column * words_per_row + word_index)
            group = column * groups_per_row + word_index // words_per_group
            scale = metile.cast(metile.load(Scales + group), decode_dtype)
            bias = metile.cast(metile.load(Biases + group), decode_dtype)
            for element in range(values_per_word):
                quantized = metile.cast(
                    (packed >> (element * BITS)) & ((1 << BITS) - 1),
                    decode_dtype,
                )
                accumulators[output_offset] = accumulators[output_offset] + input_values[
                    element
                ] * (quantized * scale + bias)

    results = [metile.simd_sum(accumulator) for accumulator in accumulators]
    for output_offset in range(OUTPUTS_PER_SIMDGROUP):
        metile.store(
            Out + output_base + output_offset,
            results[output_offset],
            mask=lane == 0,
        )


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
    OUTPUTS_PER_SIMDGROUP: metile.constexpr,
    HALF_DECODE: metile.constexpr,
):
    """Fuse two affine QMV projections with a SwiGLU epilogue."""
    thread = metile.thread_id()
    lane = thread % 32
    simdgroup = thread // 32
    simdgroups_per_threadgroup = BLOCK // 32
    outputs_per_threadgroup = simdgroups_per_threadgroup * OUTPUTS_PER_SIMDGROUP
    output_base = metile.program_id(0) * outputs_per_threadgroup + simdgroup * OUTPUTS_PER_SIMDGROUP
    row = output_base // N
    column_base = output_base % N
    values_per_word = 32 // BITS
    words_per_row = K // values_per_word
    groups_per_row = K // GROUP_SIZE
    words_per_group = GROUP_SIZE // values_per_word
    decode_dtype = "f16" if HALF_DECODE else "f32"
    gate_accumulators = [metile.scalar(0.0) for _ in range(OUTPUTS_PER_SIMDGROUP)]
    up_accumulators = [metile.scalar(0.0) for _ in range(OUTPUTS_PER_SIMDGROUP)]

    for word_index in metile.tile_range(lane, words_per_row, 32):
        input_base = row * K + word_index * values_per_word
        input_values = [
            metile.cast(metile.load(X + input_base + element), decode_dtype)
            for element in range(values_per_word)
        ]
        for output_offset in range(OUTPUTS_PER_SIMDGROUP):
            column = column_base + output_offset
            packed_offset = column * words_per_row + word_index
            gate_packed = metile.load(GateW + packed_offset)
            up_packed = metile.load(UpW + packed_offset)
            group = column * groups_per_row + word_index // words_per_group
            gate_scale = metile.cast(metile.load(GateScales + group), decode_dtype)
            gate_bias = metile.cast(metile.load(GateBiases + group), decode_dtype)
            up_scale = metile.cast(metile.load(UpScales + group), decode_dtype)
            up_bias = metile.cast(metile.load(UpBiases + group), decode_dtype)
            for element in range(values_per_word):
                shift = element * BITS
                mask = (1 << BITS) - 1
                gate_quantized = metile.cast(
                    (gate_packed >> shift) & mask,
                    decode_dtype,
                )
                up_quantized = metile.cast(
                    (up_packed >> shift) & mask,
                    decode_dtype,
                )
                gate_accumulators[output_offset] = gate_accumulators[output_offset] + input_values[
                    element
                ] * (gate_quantized * gate_scale + gate_bias)
                up_accumulators[output_offset] = up_accumulators[output_offset] + input_values[
                    element
                ] * (up_quantized * up_scale + up_bias)

    results = []
    for output_offset in range(OUTPUTS_PER_SIMDGROUP):
        gate = metile.simd_sum(gate_accumulators[output_offset])
        up = metile.simd_sum(up_accumulators[output_offset])
        results.append(gate / (1.0 + metile.fast_exp(0.0 - gate)) * up)
    for output_offset in range(OUTPUTS_PER_SIMDGROUP):
        metile.store(
            Out + output_base + output_offset,
            results[output_offset],
            mask=lane == 0,
        )
