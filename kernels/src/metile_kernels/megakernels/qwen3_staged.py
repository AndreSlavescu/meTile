"""GPU-wide Qwen3 decode stages with ordered dispatch boundaries.

Every matrix stage assigns one output row to each SIMD group. Dependencies
between threadgroups are resolved by the caller's ordered dispatches, never
by device-wide spinning or an unsupported grid barrier. Intermediate vectors
use the declared storage dtype; all matrix and attention sums use FP32.

``Control`` is an int32 tensor containing ``[token, position]``. It must not
change while the prepared pipeline is executing. All tensor buffers passed
as distinct arguments must be disjoint, including residual input and output.
Packed layer weights and cache layout match the single-threadgroup decoder.
Matrix stages optionally accept two losslessly stored BF16 values per uint32
word, lower-indexed value first. The caller verifies that the original FP32
weights round-trip exactly before packing; sources, accumulation and outputs
remain FP32. Logical weight offsets and strides still count unpacked values.
"""

import math

import metile
from metile_kernels.megakernels.qwen3 import (
    _attention_update,
    _silu,
    _stored,
    qwen3_layer_offsets,
)


def _validate(block, storage, *dimensions):
    if type(block) is not int or block < 32 or block > 1024 or block % 32:
        raise ValueError("BLOCK must be a multiple of 32 between 32 and 1024")
    if storage not in ("f16", "f32"):
        raise ValueError("STORAGE_DTYPE must be f16 or f32")
    if any(type(dimension) is not int or dimension <= 0 for dimension in dimensions):
        raise ValueError("tensor dimensions must be positive integers")


def _validate_epsilon(epsilon):
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("EPS must be finite and positive")


def _row(block):
    return metile.program_id(0) * (block // 32) + metile.thread_id() // 32


def _validate_packed_weights(packed, storage, columns, offset=0, stride=0):
    if type(packed) is not bool:
        raise ValueError("PACKED_WEIGHTS must be bool")
    if packed:
        if storage != "f32":
            raise ValueError("PACKED_WEIGHTS requires FP32 source and output storage")
        if any(
            type(value) is not int or value < 0 or value % 2 for value in (columns, offset, stride)
        ):
            raise ValueError("PACKED_WEIGHTS requires even columns, weight offsets and strides")


def _dot(source, weights, row, columns, packed_weights=False):
    if packed_weights and weights.memory.ptr.type.dtype != "u32":
        raise ValueError("PACKED_WEIGHTS requires uint32 weight storage")
    accumulator = metile.loop_state(0.0)
    for column in metile.tile_range(metile.simd_lane_id(), columns, 32):
        value = metile.cast(source.load((column,)), "f32")
        if packed_weights:
            word = weights.load((row, column // 2))
            shift = metile.cast(column % 2, "u32") * 16
            weight = metile.bitcast((word >> shift) << 16, "f32")
        else:
            weight = metile.cast(weights.load((row, column)), "f32")
        accumulator.update(accumulator.value + value * weight)
    return metile.simd_sum(accumulator.value)


@metile.kernel
def qwen3_staged_embedding(
    Embedding,
    Control,
    Hidden,
    *,
    HIDDEN: metile.constexpr,
    VOCAB: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Copy one token embedding; launch ``ceil(HIDDEN / BLOCK)`` threadgroups."""
    _validate(BLOCK, STORAGE_DTYPE, HIDDEN, VOCAB)
    embeddings = metile.tensor(Embedding, shape=(VOCAB, HIDDEN), access="read")
    control = metile.tensor(Control, shape=(2,), access="read")
    hidden = metile.tensor(Hidden, shape=(HIDDEN,), access="write")
    feature = metile.program_id(0) * BLOCK + metile.thread_id()
    hidden.store((feature,), embeddings.load((control.load((0,)), feature)))


@metile.kernel
def qwen3_staged_rmsnorm(
    Source,
    NormWeights,
    Destination,
    layer,
    *,
    WIDTH: metile.constexpr,
    WEIGHT_OFFSET: metile.constexpr = 0,
    WEIGHT_STRIDE: metile.constexpr = 0,
    EPS: metile.constexpr = 1e-6,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Normalize a vector with packed layer weights; launch one threadgroup."""
    _validate(BLOCK, STORAGE_DTYPE, WIDTH)
    _validate_epsilon(EPS)
    source = metile.tensor(Source, shape=(WIDTH,), access="read")
    weights = metile.tensor(
        NormWeights + WEIGHT_OFFSET + layer * WEIGHT_STRIDE, shape=(WIDTH,), access="read"
    )
    destination = metile.tensor(Destination, shape=(WIDTH,), access="write")
    square_sum = metile.loop_state(0.0)
    for feature in metile.tile_range(metile.simd_lane_id(), WIDTH, 32):
        value = metile.cast(source.load((feature,)), "f32")
        square_sum.update(square_sum.value + value * value)
    reciprocal = 1.0 / metile.sqrt(metile.simd_sum(square_sum.value) / WIDTH + EPS)
    for feature in metile.tile_range(metile.thread_id(), WIDTH, BLOCK):
        value = metile.cast(source.load((feature,)), "f32")
        weight = metile.cast(weights.load((feature,)), "f32")
        destination.store(
            (feature,), _stored(_stored(value * reciprocal, STORAGE_DTYPE) * weight, STORAGE_DTYPE)
        )


@metile.kernel
def qwen3_staged_qkv(
    Normalized,
    LayerWeights,
    QKV,
    layer,
    *,
    HIDDEN: metile.constexpr,
    INTERMEDIATE: metile.constexpr,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    PACKED_WEIGHTS: metile.constexpr = False,
):
    """Project contiguous Q, K, V rows; launch ``ceil(rows / (BLOCK / 32))``."""
    _validate(BLOCK, STORAGE_DTYPE)
    offsets = qwen3_layer_offsets(HIDDEN, INTERMEDIATE, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    _validate_packed_weights(
        PACKED_WEIGHTS, STORAGE_DTYPE, HIDDEN, offsets["q_proj"], offsets["layer_size"]
    )
    rows = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM
    source = metile.tensor(Normalized, shape=(HIDDEN,), access="read")
    weights = metile.tensor(
        LayerWeights + (layer * offsets["layer_size"] + offsets["q_proj"]) // 2
        if PACKED_WEIGHTS
        else LayerWeights + layer * offsets["layer_size"] + offsets["q_proj"],
        shape=(rows, HIDDEN // 2 if PACKED_WEIGHTS else HIDDEN),
        access="read",
    )
    destination = metile.tensor(QKV, shape=(rows, 1), access="write")
    row = _row(BLOCK)
    destination.store(
        (row, metile.simd_lane_id()),
        _stored(_dot(source, weights, row, HIDDEN, PACKED_WEIGHTS), STORAGE_DTYPE),
    )


@metile.kernel
def qwen3_staged_qk_rope(
    QKV,
    LayerWeights,
    Rotary,
    Control,
    Queries,
    KVCache,
    layer,
    *,
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
):
    """Normalize/rotate Q and K and store current K/V before attention.

    Launch ``ceil((QUERY_HEADS + KV_HEADS) / (BLOCK / 32))`` groups. Each
    SIMD group handles one Q or K head. Tensor bounds mask the other output.
    """
    _validate(BLOCK, STORAGE_DTYPE, LAYERS, MAX_CONTEXT)
    _validate_epsilon(EPS)
    offsets = qwen3_layer_offsets(HIDDEN, INTERMEDIATE, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    heads = QUERY_HEADS + KV_HEADS
    half = HEAD_DIM // 2
    qk_left = metile.tensor(QKV, shape=(heads, half), strides=(HEAD_DIM, 1), access="read")
    qk_right = metile.tensor(QKV + half, shape=(heads, half), strides=(HEAD_DIM, 1), access="read")
    values = metile.tensor(QKV + heads * HEAD_DIM, shape=(KV_HEADS, HEAD_DIM), access="read")
    norm_weights = metile.tensor(
        LayerWeights + layer * offsets["layer_size"] + offsets["q_norm"],
        shape=(2, HEAD_DIM),
        access="read",
    )
    rotary = metile.tensor(Rotary, shape=(MAX_CONTEXT, half, 2), access="read")
    control = metile.tensor(Control, shape=(2,), access="read")
    query_left = metile.tensor(
        Queries, shape=(QUERY_HEADS, half), strides=(HEAD_DIM, 1), access="write"
    )
    query_right = metile.tensor(
        Queries + half, shape=(QUERY_HEADS, half), strides=(HEAD_DIM, 1), access="write"
    )
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="write"
    )
    head = _row(BLOCK)
    lane = metile.simd_lane_id()
    position = control.load((1,))
    weight_row = metile.where(head < QUERY_HEADS, 0, 1)
    left = [
        metile.cast(qk_left.load((head, lane + component * 32)), "f32")
        for component in range((half + 31) // 32)
    ]
    right = [
        metile.cast(qk_right.load((head, lane + component * 32)), "f32")
        for component in range((half + 31) // 32)
    ]
    square_sum = 0.0
    for first, second in zip(left, right, strict=True):
        square_sum = square_sum + first * first + second * second
    reciprocal = 1.0 / metile.sqrt(metile.simd_sum(square_sum) / HEAD_DIM + EPS)
    for component, (first, second) in enumerate(zip(left, right, strict=True)):
        feature = lane + component * 32
        left_weight = metile.cast(norm_weights.load((weight_row, feature)), "f32")
        right_weight = metile.cast(norm_weights.load((weight_row, feature + half)), "f32")
        first = _stored(_stored(first * reciprocal, STORAGE_DTYPE) * left_weight, STORAGE_DTYPE)
        second = _stored(_stored(second * reciprocal, STORAGE_DTYPE) * right_weight, STORAGE_DTYPE)
        cosine = rotary.load((position, feature, 0))
        sine = rotary.load((position, feature, 1))
        rotated_left = _stored(first * cosine - second * sine, STORAGE_DTYPE)
        rotated_right = _stored(second * cosine + first * sine, STORAGE_DTYPE)
        query_left.store((head, feature), rotated_left)
        query_right.store((head, feature), rotated_right)
        key_feature = metile.where(feature < half, feature, HEAD_DIM)
        cache.store((0, layer, head - QUERY_HEADS, position, key_feature), rotated_left)
        cache.store((0, layer, head - QUERY_HEADS, position, feature + half), rotated_right)
    for component in range(HEAD_DIM // 32):
        feature = lane + component * 32
        cache.store(
            (1, layer, head - QUERY_HEADS, position, feature),
            values.load((head - QUERY_HEADS, feature)),
        )


@metile.kernel
def qwen3_staged_attention(
    Queries,
    KVCache,
    Control,
    Attention,
    layer,
    *,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    LAYERS: metile.constexpr,
    MAX_CONTEXT: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Attend through current position; launch ``ceil(QUERY_HEADS / (BLOCK / 32))``."""
    _validate(BLOCK, STORAGE_DTYPE, QUERY_HEADS, KV_HEADS, HEAD_DIM, LAYERS, MAX_CONTEXT)
    if HEAD_DIM % 32 or QUERY_HEADS % KV_HEADS:
        raise ValueError("HEAD_DIM must be divisible by 32 and QUERY_HEADS by KV_HEADS")
    queries = metile.tensor(Queries, shape=(QUERY_HEADS, HEAD_DIM), access="read")
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="read"
    )
    control = metile.tensor(Control, shape=(2,), access="read")
    attention = metile.tensor(Attention, shape=(QUERY_HEADS, HEAD_DIM), access="write")
    head = _row(BLOCK)
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    lane = metile.simd_lane_id()
    query = [
        metile.cast(queries.load((head, lane * (HEAD_DIM // 32) + component)), "f32")
        for component in range(HEAD_DIM // 32)
    ]
    maximum = metile.loop_state(-1e30)
    denominator = metile.loop_state(0.0)
    outputs = [metile.loop_state(0.0) for _ in range(HEAD_DIM // 32)]
    for previous in metile.tile_range(0, control.load((1,)) + 1, 1):
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
    for component, output in enumerate(outputs):
        attention.store(
            (head, lane * (HEAD_DIM // 32) + component),
            _stored(output.value / denominator.value, STORAGE_DTYPE),
        )


@metile.kernel
def qwen3_staged_gemv(
    Source,
    Weights,
    Destination,
    layer,
    *,
    ROWS: metile.constexpr,
    COLUMNS: metile.constexpr,
    WEIGHT_OFFSET: metile.constexpr = 0,
    WEIGHT_STRIDE: metile.constexpr = 0,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    PACKED_WEIGHTS: metile.constexpr = False,
):
    """Project a vector; launch ``ceil(ROWS / (BLOCK / 32))`` threadgroups."""
    _validate(BLOCK, STORAGE_DTYPE, ROWS, COLUMNS)
    _validate_packed_weights(PACKED_WEIGHTS, STORAGE_DTYPE, COLUMNS, WEIGHT_OFFSET, WEIGHT_STRIDE)
    source = metile.tensor(Source, shape=(COLUMNS,), access="read")
    weights = metile.tensor(
        Weights + (WEIGHT_OFFSET + layer * WEIGHT_STRIDE) // 2
        if PACKED_WEIGHTS
        else Weights + WEIGHT_OFFSET + layer * WEIGHT_STRIDE,
        shape=(ROWS, COLUMNS // 2 if PACKED_WEIGHTS else COLUMNS),
        access="read",
    )
    destination = metile.tensor(Destination, shape=(ROWS, 1), access="write")
    row = _row(BLOCK)
    destination.store(
        (row, metile.simd_lane_id()),
        _stored(_dot(source, weights, row, COLUMNS, PACKED_WEIGHTS), STORAGE_DTYPE),
    )


@metile.kernel
def qwen3_staged_residual(
    Source,
    LayerWeights,
    Residual,
    Destination,
    layer,
    *,
    ROWS: metile.constexpr,
    COLUMNS: metile.constexpr,
    WEIGHT_OFFSET: metile.constexpr,
    WEIGHT_STRIDE: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    PACKED_WEIGHTS: metile.constexpr = False,
):
    """Project and add a disjoint residual; launch ``ceil(ROWS / (BLOCK / 32))``."""
    _validate(BLOCK, STORAGE_DTYPE, ROWS, COLUMNS)
    _validate_packed_weights(PACKED_WEIGHTS, STORAGE_DTYPE, COLUMNS, WEIGHT_OFFSET, WEIGHT_STRIDE)
    source = metile.tensor(Source, shape=(COLUMNS,), access="read")
    weights = metile.tensor(
        LayerWeights + (WEIGHT_OFFSET + layer * WEIGHT_STRIDE) // 2
        if PACKED_WEIGHTS
        else LayerWeights + WEIGHT_OFFSET + layer * WEIGHT_STRIDE,
        shape=(ROWS, COLUMNS // 2 if PACKED_WEIGHTS else COLUMNS),
        access="read",
    )
    residual = metile.tensor(Residual, shape=(ROWS,), access="read")
    destination = metile.tensor(Destination, shape=(ROWS, 1), access="write")
    row = _row(BLOCK)
    projected = _stored(_dot(source, weights, row, COLUMNS, PACKED_WEIGHTS), STORAGE_DTYPE)
    value = _stored(metile.cast(residual.load((row,)), "f32") + projected, STORAGE_DTYPE)
    destination.store((row, metile.simd_lane_id()), value)


@metile.kernel
def qwen3_staged_swiglu(
    Normalized,
    LayerWeights,
    Intermediate,
    layer,
    *,
    HIDDEN: metile.constexpr,
    INTERMEDIATE: metile.constexpr,
    QUERY_HEADS: metile.constexpr,
    KV_HEADS: metile.constexpr,
    HEAD_DIM: metile.constexpr,
    BLOCK: metile.constexpr = 128,
    STORAGE_DTYPE: metile.constexpr = "f32",
    PACKED_WEIGHTS: metile.constexpr = False,
):
    """Fuse gate/up projection and SwiGLU; launch ``ceil(INTERMEDIATE / (BLOCK / 32))``."""
    _validate(BLOCK, STORAGE_DTYPE)
    offsets = qwen3_layer_offsets(HIDDEN, INTERMEDIATE, QUERY_HEADS, KV_HEADS, HEAD_DIM)
    for offset in (offsets["gate_proj"], offsets["up_proj"]):
        _validate_packed_weights(
            PACKED_WEIGHTS, STORAGE_DTYPE, HIDDEN, offset, offsets["layer_size"]
        )
    source = metile.tensor(Normalized, shape=(HIDDEN,), access="read")
    gate_weights = metile.tensor(
        LayerWeights + (layer * offsets["layer_size"] + offsets["gate_proj"]) // 2
        if PACKED_WEIGHTS
        else LayerWeights + layer * offsets["layer_size"] + offsets["gate_proj"],
        shape=(INTERMEDIATE, HIDDEN // 2 if PACKED_WEIGHTS else HIDDEN),
        access="read",
    )
    up_weights = metile.tensor(
        LayerWeights + (layer * offsets["layer_size"] + offsets["up_proj"]) // 2
        if PACKED_WEIGHTS
        else LayerWeights + layer * offsets["layer_size"] + offsets["up_proj"],
        shape=(INTERMEDIATE, HIDDEN // 2 if PACKED_WEIGHTS else HIDDEN),
        access="read",
    )
    destination = metile.tensor(Intermediate, shape=(INTERMEDIATE, 1), access="write")
    row = _row(BLOCK)
    gate = _stored(_dot(source, gate_weights, row, HIDDEN, PACKED_WEIGHTS), STORAGE_DTYPE)
    up = _stored(_dot(source, up_weights, row, HIDDEN, PACKED_WEIGHTS), STORAGE_DTYPE)
    destination.store(
        (row, metile.simd_lane_id()), _stored(_silu(gate, STORAGE_DTYPE) * up, STORAGE_DTYPE)
    )
