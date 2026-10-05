"""Causal Qwen3 attention sharing each KV tile across a group of query rows.

One SIMD group owns one query, retaining partitioned online-softmax states
in FP32 registers. A threadgroup cooperatively loads each key/value tile once
and reuses it across all its queries. Barriers protect only threadgroup memory;
the caller must order the preceding KV-cache writes before this dispatch.
"""

import metile
from metile_kernels.megakernels.qwen3 import _stored
from metile_kernels.megakernels.qwen3_staged import _validate


@metile.kernel
def qwen3_prefill_tiled_attention(
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
    BLOCK: metile.constexpr = 256,
    KEY_TILE: metile.constexpr = 16,
    PARTITIONS: metile.constexpr = 8,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Launch ``(ceil(CHUNK / (BLOCK / 32)), QUERY_HEADS)`` threadgroups.

    Queries and output are contiguous ``[CHUNK, QUERY_HEADS, HEAD_DIM]``.
    Cache is ``[2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM]``. Int32
    ``Control[2]`` contains the starting cache position and valid row count.
    Invalid query rows become zero; future cache entries never participate.
    """
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, QUERY_HEADS, KV_HEADS, HEAD_DIM, LAYERS, MAX_CONTEXT)
    if HEAD_DIM % 32 or QUERY_HEADS % KV_HEADS:
        raise ValueError("HEAD_DIM must be divisible by 32 and QUERY_HEADS by KV_HEADS")
    if type(KEY_TILE) is not int or KEY_TILE < 1:
        raise ValueError("KEY_TILE must be a positive integer")
    if (
        type(PARTITIONS) is not int
        or PARTITIONS < 1
        or PARTITIONS > 32
        or PARTITIONS & (PARTITIONS - 1)
        or KEY_TILE % PARTITIONS
    ):
        raise ValueError("PARTITIONS must be a power of two at most 32 that divides KEY_TILE")
    if 2 * KEY_TILE * HEAD_DIM * 4 > 32768:
        raise ValueError("shared key/value tiles must fit within 32 KiB")
    groups = BLOCK // 32
    components = HEAD_DIM // 32
    tile_memory = metile.shared(2 * KEY_TILE * HEAD_DIM, dtype="f32")
    queries = metile.tensor(Queries, shape=(CHUNK, QUERY_HEADS, HEAD_DIM), access="read")
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="read"
    )
    control = metile.tensor(Control, shape=(2,), access="read")
    output = metile.tensor(Attention, shape=(CHUNK, QUERY_HEADS, HEAD_DIM), access="write")
    tile = metile.tensor(tile_memory, shape=(2, KEY_TILE, HEAD_DIM))
    head = metile.program_id(1)
    row_base = metile.program_id(0) * groups
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    row = row_base + thread // 32
    prefix = control.load((0,))
    valid_rows = control.load((1,))
    valid = row < valid_rows
    position = prefix + row
    limit = metile.where(
        row_base < valid_rows, prefix + metile.minimum(row_base + groups, valid_rows), 0
    )
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    query = [
        metile.cast(queries.load((row, head, lane * components + component)), "f32")
        for component in range(components)
    ]
    maxima = [metile.loop_state(-1e30) for partition in range(PARTITIONS)]
    denominators = [metile.loop_state(0.0) for partition in range(PARTITIONS)]
    accumulators = [
        [metile.loop_state(0.0) for component in range(components)]
        for partition in range(PARTITIONS)
    ]
    for start in metile.tile_range(0, limit, KEY_TILE):
        for index in metile.tile_range(thread, KEY_TILE * HEAD_DIM, BLOCK):
            key = index // HEAD_DIM
            feature = index % HEAD_DIM
            source_position = metile.where(start + key < limit, start + key, MAX_CONTEXT)
            for kind in range(2):
                tile.store(
                    (kind, key, feature),
                    metile.cast(
                        cache.load((kind, layer, kv_head, source_position, feature)), "f32"
                    ),
                )
        metile.barrier()
        for offset in metile.tile_range(0, KEY_TILE, PARTITIONS):
            for partition in range(PARTITIONS):
                key = offset + partition
                visible = valid & (start + key <= position) & (start + key < limit)
                keys = [
                    metile.where(visible, tile.load((0, key, lane * components + component)), 0.0)
                    for component in range(components)
                ]
                values = [
                    metile.where(visible, tile.load((1, key, lane * components + component)), 0.0)
                    for component in range(components)
                ]
                score = 0.0
                for component in range(components):
                    score = score + (query[component] * HEAD_DIM**-0.5) * keys[component]
                score = metile.where(visible, metile.simd_sum(score), -1e30)
                next_maximum = metile.maximum(maxima[partition].value, score)
                previous_factor = metile.exp(maxima[partition].value - next_maximum)
                probability = metile.where(visible, metile.exp(score - next_maximum), 0.0)
                denominators[partition].update(
                    denominators[partition].value * previous_factor + probability
                )
                for component in range(components):
                    accumulators[partition][component].update(
                        accumulators[partition][component].value * previous_factor
                        + probability * values[component]
                    )
                maxima[partition].update(next_maximum)
        metile.barrier()
    merged_maxima = [maximum.value for maximum in maxima]
    width = PARTITIONS
    while width > 1:
        width //= 2
        merged_maxima = [
            metile.maximum(merged_maxima[2 * partition], merged_maxima[2 * partition + 1])
            for partition in range(width)
        ]
    factors = [metile.exp(maximum.value - merged_maxima[0]) for maximum in maxima]
    sums = [
        denominator.value * factor
        for denominator, factor in zip(denominators, factors, strict=True)
    ]
    numerators = [
        [accumulator.value * factor for accumulator in partition]
        for partition, factor in zip(accumulators, factors, strict=True)
    ]
    width = PARTITIONS
    while width > 1:
        width //= 2
        sums = [sums[2 * partition] + sums[2 * partition + 1] for partition in range(width)]
        numerators = [
            [
                numerators[2 * partition][component] + numerators[2 * partition + 1][component]
                for component in range(components)
            ]
            for partition in range(width)
        ]
    normalizer = metile.maximum(sums[0], 1e-30)
    for component in range(components):
        output.store(
            (row, head, lane * components + component),
            _stored(metile.where(valid, numerators[0][component] / normalizer, 0.0), STORAGE_DTYPE),
        )
