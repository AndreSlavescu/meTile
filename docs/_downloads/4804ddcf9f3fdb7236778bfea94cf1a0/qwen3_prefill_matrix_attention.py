"""Causal Qwen3 attention using composable FP32 SIMD-group matrix tiles.

Eight or 32 query rows share a key tile. QK and PV use public DSL dot operations;
online softmax bridges their matrix fragments through bounded shared memory.
Output fragments remain in registers across key tiles. No score matrix,
cross-threadgroup synchronization, or handwritten Metal is required.
Unrolled, unpadded 32-row tiles also cache Q fragments and reuse their shared
allocation for K/V after an explicit threadgroup barrier.

Prepare with STRICT_MATH=True and Schedule(backend="simdgroup_inline").
"""

import metile
from metile_kernels.megakernels.qwen3 import _stored
from metile_kernels.megakernels.qwen3_staged import _validate


@metile.kernel
def qwen3_prefill_matrix_attention(
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
    QUERY_TILE: metile.constexpr = 8,
    KEY_TILE: metile.constexpr = 32,
    SHARED_PADDING: metile.constexpr = 4,
    UNROLL_MMA: metile.constexpr = False,
    SOFTMAX_LANES: metile.constexpr = 32,
    TRANSPOSE_KEYS: metile.constexpr = False,
    REGISTER_STATS: metile.constexpr = False,
    UNROLL_SOFTMAX: metile.constexpr = False,
    LOAD_VECTOR: metile.constexpr = 1,
    SOFTMAX_BASE2: metile.constexpr = False,
    DIRECT_MEMORY: metile.constexpr = False,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Launch ``(ceil(CHUNK / QUERY_TILE), QUERY_HEADS)`` threadgroups.

    Queries and output are contiguous ``[CHUNK, QUERY_HEADS, HEAD_DIM]``.
    Cache is ``[2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM]``. Int32
    ``Control[2]`` contains the starting cache position and valid row count.
    Cache writes must complete before this dispatch. Buffers are disjoint;
    invalid rows become zero, and cache tails may contain arbitrary values.
    Supported query/key tiles are 8/32 and 32/16. The latter assigns eight
    complete query rows to each SIMD group; the former partitions columns.
    SHARED_PADDING=0 retains unpadded shared strides for controlled comparisons.
    REGISTER_STATS retains owned-row statistics in scalar loop state and
    unrolls row processing; UNROLL_SOFTMAX also permits unrolling shared stats.
    SOFTMAX_BASE2 stages unscaled queries, scales completed QK fragments by
    FP32(scale) * FP32(log2(e)), uses base-two online softmax, and divides
    output fragments directly by their row denominators.
    DIRECT_MEMORY uses bounded device matrix views and an explicit per-SIMD
    scratch tile instead of staging Q/K/V and output through shared memory.
    """
    _validate(BLOCK, STORAGE_DTYPE, CHUNK, QUERY_HEADS, KV_HEADS, HEAD_DIM, LAYERS, MAX_CONTEXT)
    if HEAD_DIM % 32 or QUERY_HEADS % KV_HEADS:
        raise ValueError("HEAD_DIM must be divisible by 32 and QUERY_HEADS by KV_HEADS")
    if BLOCK != 128:
        raise ValueError("matrix attention requires BLOCK=128")
    if (
        type(QUERY_TILE) is not int
        or type(KEY_TILE) is not int
        or (QUERY_TILE, KEY_TILE) not in ((8, 32), (32, 16))
    ):
        raise ValueError("matrix attention requires QUERY_TILE/KEY_TILE=8/32 or 32/16")
    if type(SHARED_PADDING) is not int or SHARED_PADDING not in (0, 4):
        raise ValueError("SHARED_PADDING must be 0 or 4")
    if type(UNROLL_MMA) is not bool:
        raise ValueError("UNROLL_MMA must be bool")
    if (
        type(SOFTMAX_LANES) is not int
        or SOFTMAX_LANES not in (8, 16, 32)
        or 32 // SOFTMAX_LANES > QUERY_TILE // 4
    ):
        raise ValueError("SOFTMAX_LANES must be 8, 16 or 32 with enough owned query rows")
    if type(TRANSPOSE_KEYS) is not bool:
        raise ValueError("TRANSPOSE_KEYS must be bool")
    if type(REGISTER_STATS) is not bool or type(UNROLL_SOFTMAX) is not bool:
        raise ValueError("REGISTER_STATS and UNROLL_SOFTMAX must be bool")
    if type(LOAD_VECTOR) is not int or LOAD_VECTOR not in (1, 4, 16):
        raise ValueError("LOAD_VECTOR must be 1, 4 or 16")
    if type(SOFTMAX_BASE2) is not bool:
        raise ValueError("SOFTMAX_BASE2 must be bool")
    if type(DIRECT_MEMORY) is not bool:
        raise ValueError("DIRECT_MEMORY must be bool")
    if DIRECT_MEMORY and (QUERY_TILE != 32 or SHARED_PADDING != 0 or not UNROLL_MMA):
        raise ValueError("DIRECT_MEMORY requires QUERY_TILE=32, SHARED_PADDING=0 and UNROLL_MMA")
    query_stride = HEAD_DIM + SHARED_PADDING
    score_stride = KEY_TILE + SHARED_PADDING
    key_strides = (score_stride, 1) if TRANSPOSE_KEYS else (1, query_stride)
    key_value_elements = (
        max(KEY_TILE * query_stride, HEAD_DIM * score_stride)
        if TRANSPOSE_KEYS
        else KEY_TILE * query_stride
    )
    statistic_elements = 0 if REGISTER_STATS else 2 * QUERY_TILE
    cache_queries = QUERY_TILE == 32 and SHARED_PADDING == 0 and UNROLL_MMA
    query_elements = QUERY_TILE * query_stride
    query_allocation_elements = (
        max(query_elements, key_value_elements) if cache_queries else query_elements
    )
    shared_elements = (
        (0 if DIRECT_MEMORY else query_allocation_elements)
        + (0 if DIRECT_MEMORY or cache_queries else key_value_elements)
        + QUERY_TILE * (score_stride + 8)
        + statistic_elements
    )
    scratch_bytes = BLOCK * 2 * (2 if STORAGE_DTYPE == "f16" else 4) if DIRECT_MEMORY else 0
    if shared_elements * 4 + scratch_bytes > 32768:
        raise ValueError("matrix attention shared tiles must fit within 32 KiB")
    if DIRECT_MEMORY:
        device_scratch = metile.tensor(
            metile.shared(BLOCK * 2, dtype=STORAGE_DTYPE), shape=(BLOCK // 4, 8)
        )
    else:
        query_memory = metile.shared(query_allocation_elements, dtype="f32")
        key_value_memory = (
            query_memory if cache_queries else metile.shared(key_value_elements, dtype="f32")
        )
    score_memory = metile.shared(QUERY_TILE * score_stride, dtype="f32")
    factor_memory = metile.shared(QUERY_TILE * 8, dtype="f32")
    if not REGISTER_STATS:
        statistic_memory = metile.shared(statistic_elements, dtype="f32")
    queries = metile.tensor(Queries, shape=(CHUNK, QUERY_HEADS, HEAD_DIM), access="read")
    cache = metile.tensor(
        KVCache, shape=(2, LAYERS, KV_HEADS, MAX_CONTEXT, HEAD_DIM), access="read"
    )
    control = metile.tensor(Control, shape=(2,), access="read")
    output = metile.tensor(Attention, shape=(CHUNK, QUERY_HEADS, HEAD_DIM), access="write")
    if not DIRECT_MEMORY:
        query_values = metile.tensor(
            query_memory, shape=(QUERY_TILE, HEAD_DIM), strides=(query_stride, 1)
        )
        query_matrix = metile.tensor(
            query_memory,
            shape=(QUERY_TILE, HEAD_DIM),
            strides=(query_stride, 1),
            block_shape=(8, 8),
        )
        key_value_values = metile.tensor(
            key_value_memory, shape=(KEY_TILE, HEAD_DIM), strides=(query_stride, 1)
        )
        key_values = metile.tensor(
            key_value_memory, shape=(HEAD_DIM, KEY_TILE), strides=key_strides
        )
        key_matrix = metile.tensor(
            key_value_memory,
            shape=(HEAD_DIM, KEY_TILE),
            strides=key_strides,
            block_shape=(8, 8),
        )
        value_matrix = metile.tensor(
            key_value_memory,
            shape=(KEY_TILE, HEAD_DIM),
            strides=(query_stride, 1),
            block_shape=(8, 8),
        )
    score_values = metile.tensor(
        score_memory, shape=(QUERY_TILE, KEY_TILE), strides=(score_stride, 1)
    )
    score_matrix = metile.tensor(
        score_memory,
        shape=(QUERY_TILE, KEY_TILE),
        strides=(score_stride, 1),
        block_shape=(8, 8),
    )
    factor_values = metile.tensor(factor_memory, shape=(QUERY_TILE, 8))
    factor_matrix = metile.tensor(factor_memory, shape=(QUERY_TILE, 8), block_shape=(8, 8))
    if not REGISTER_STATS:
        statistics = metile.tensor(statistic_memory, shape=(2, QUERY_TILE, 1))
    head = metile.program_id(1)
    row_base = metile.program_id(0) * QUERY_TILE
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    simdgroup = thread // 32
    groups = BLOCK // 32
    subgroup_lane = lane % SOFTMAX_LANES
    subgroup = lane // SOFTMAX_LANES
    rows_per_iteration = 32 // SOFTMAX_LANES
    row_iterations = QUERY_TILE // (groups * rows_per_iteration)
    if QUERY_TILE == 8:
        matrix_row = 0
        matrix_column = simdgroup * 8
        output_column = simdgroup * 8
        output_step = 32
        row_start = simdgroup
        row_step = groups
        score_fragments = 1
    else:
        matrix_row = simdgroup * 8
        matrix_column = 0
        output_column = 0
        output_step = 8
        row_start = simdgroup * 8
        row_step = 1
        score_fragments = 2
    prefix = control.load((0,))
    valid_rows = control.load((1,))
    limit = metile.where(
        row_base < valid_rows, prefix + metile.minimum(row_base + QUERY_TILE, valid_rows), 0
    )
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    if DIRECT_MEMORY:
        cache_head_elements = MAX_CONTEXT * HEAD_DIM
        key_offset = (layer * KV_HEADS + kv_head) * cache_head_elements
        value_offset = ((LAYERS + layer) * KV_HEADS + kv_head) * cache_head_elements
        query_matrix = metile.tensor(
            Queries + head * HEAD_DIM,
            shape=(valid_rows, HEAD_DIM),
            strides=(QUERY_HEADS * HEAD_DIM, 1),
            block_shape=(8, 8),
            access="read",
        )
        key_matrix = metile.tensor(
            KVCache + key_offset,
            shape=(HEAD_DIM, limit),
            strides=(1, HEAD_DIM),
            block_shape=(8, 8),
            access="read",
        )
        value_matrix = metile.tensor(
            KVCache + value_offset,
            shape=(limit, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            block_shape=(8, 8),
            access="read",
        )
        output_matrix = metile.tensor(
            Attention + head * HEAD_DIM,
            shape=(CHUNK, HEAD_DIM),
            strides=(QUERY_HEADS * HEAD_DIM, 1),
            block_shape=(8, 8),
            access="write",
        )
    if SOFTMAX_BASE2:
        score_scale = metile.cast(HEAD_DIM**-0.5, "f32") * metile.cast(1.4426950408889634, "f32")
    exponential = metile.fast_exp2 if SOFTMAX_BASE2 else metile.exp
    query_copy_count = metile.cdiv(QUERY_TILE * HEAD_DIM, BLOCK * LOAD_VECTOR)
    query_copy_steps = (
        range(0)
        if DIRECT_MEMORY
        else range(query_copy_count)
        if LOAD_VECTOR > 1
        else metile.tile_range(0, query_copy_count, 1)
    )
    for copy in query_copy_steps:
        loaded_queries = []
        for component in range(LOAD_VECTOR):
            index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
            row = index // HEAD_DIM
            feature = index % HEAD_DIM
            source_row = metile.where(
                (row < QUERY_TILE) & (row_base + row < valid_rows), row_base + row, CHUNK
            )
            query_value = metile.cast(queries.load((source_row, head, feature)), "f32")
            if not SOFTMAX_BASE2:
                query_value = query_value * HEAD_DIM**-0.5
            loaded_queries.append(query_value)
        for component in range(LOAD_VECTOR):
            index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
            query_values.store((index // HEAD_DIM, index % HEAD_DIM), loaded_queries[component])
    if REGISTER_STATS:
        row_maxima = [metile.loop_state(-1e30) for row_index in range(row_iterations)]
        row_denominators = [metile.loop_state(0.0) for row_index in range(row_iterations)]
    else:
        statistics.store((0, thread, 0), -1e30)
        statistics.store((1, thread, 0), 0.0)
    outputs = [
        metile.loop_state(metile.zeros((8, 8), dtype="f32"))
        for component in range(HEAD_DIM // output_step)
    ]
    if not DIRECT_MEMORY or not REGISTER_STATS:
        metile.barrier()
    if cache_queries:
        if DIRECT_MEMORY:
            cached_queries = [
                metile.cast(
                    query_matrix.load(
                        (row_base + matrix_row, fragment * 8), scratch=device_scratch
                    ),
                    "f32",
                )
                for fragment in range(HEAD_DIM // 8)
            ]
            if not SOFTMAX_BASE2:
                cached_queries = [query * HEAD_DIM**-0.5 for query in cached_queries]
        else:
            cached_queries = [
                query_matrix.load((matrix_row, fragment * 8)) for fragment in range(HEAD_DIM // 8)
            ]
        if not DIRECT_MEMORY:
            metile.barrier()
    for start in metile.tile_range(0, limit, KEY_TILE):
        key_copy_count = metile.cdiv(KEY_TILE * HEAD_DIM, BLOCK * LOAD_VECTOR)
        key_copy_steps = (
            range(0)
            if DIRECT_MEMORY
            else range(key_copy_count)
            if LOAD_VECTOR > 1
            else metile.tile_range(0, key_copy_count, 1)
        )
        for copy in key_copy_steps:
            loaded_keys = []
            for component in range(LOAD_VECTOR):
                index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
                key = index // HEAD_DIM
                feature = index % HEAD_DIM
                source_position = metile.where(
                    (key < KEY_TILE) & (start + key < limit), start + key, MAX_CONTEXT
                )
                loaded_keys.append(
                    metile.cast(cache.load((0, layer, kv_head, source_position, feature)), "f32")
                )
            for component in range(LOAD_VECTOR):
                index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
                key_values.store((index % HEAD_DIM, index // HEAD_DIM), loaded_keys[component])
        if not DIRECT_MEMORY:
            metile.barrier()
        scores = [
            metile.loop_state(metile.zeros((8, 8), dtype="f32"))
            for column in range(score_fragments)
        ]
        feature_steps = range(0, HEAD_DIM, 8) if UNROLL_MMA else metile.tile_range(0, HEAD_DIM, 8)
        for feature in feature_steps:
            query = (
                cached_queries[feature // 8]
                if cache_queries
                else query_matrix.load((matrix_row, feature))
            )
            for column in range(score_fragments):
                key_fragment = (
                    metile.cast(
                        key_matrix.load(
                            (feature, start + matrix_column + column * 8), scratch=device_scratch
                        ),
                        "f32",
                    )
                    if DIRECT_MEMORY
                    else key_matrix.load((feature, matrix_column + column * 8))
                )
                scores[column].update(
                    metile.dot(
                        query,
                        key_fragment,
                        scores[column].value,
                    )
                )
        for column in range(score_fragments):
            score = scores[column].value
            if SOFTMAX_BASE2:
                score = score * score_scale
            score_matrix.store((matrix_row, matrix_column + column * 8), score)
        metile.barrier()
        value_copy_steps = (
            range(0)
            if DIRECT_MEMORY
            else range(key_copy_count)
            if LOAD_VECTOR > 1
            else metile.tile_range(0, key_copy_count, 1)
        )
        for copy in value_copy_steps:
            loaded_values = []
            for component in range(LOAD_VECTOR):
                index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
                key = index // HEAD_DIM
                feature = index % HEAD_DIM
                source_position = metile.where(
                    (key < KEY_TILE) & (start + key < limit), start + key, MAX_CONTEXT
                )
                loaded_values.append(
                    metile.cast(cache.load((1, layer, kv_head, source_position, feature)), "f32")
                )
            for component in range(LOAD_VECTOR):
                index = thread * LOAD_VECTOR + copy * BLOCK * LOAD_VECTOR + component
                key_value_values.store(
                    (index // HEAD_DIM, index % HEAD_DIM), loaded_values[component]
                )
        row_steps = (
            range(row_iterations)
            if UNROLL_SOFTMAX or REGISTER_STATS
            else metile.tile_range(0, row_iterations, 1)
        )
        for row_index in row_steps:
            row = row_start + (row_index * rows_per_iteration + subgroup) * row_step
            row_scores = []
            row_visibility = []
            local_maximum = -1e30
            for component in range(metile.cdiv(KEY_TILE, SOFTMAX_LANES)):
                key = subgroup_lane * metile.cdiv(KEY_TILE, SOFTMAX_LANES) + component
                visible = (
                    (row_base + row < valid_rows)
                    & (key < KEY_TILE)
                    & (start + key < limit)
                    & (start + key <= prefix + row_base + row)
                )
                score = metile.where(visible, score_values.load((row, key)), -1e30)
                row_scores.append(score)
                row_visibility.append(visible)
                local_maximum = metile.maximum(local_maximum, score)
            if SOFTMAX_LANES == 32:
                local_maximum = metile.simd_max(local_maximum)
            else:
                for shift in (1, 2, 4, 8) if SOFTMAX_LANES == 16 else (1, 2, 4):
                    local_maximum = metile.maximum(
                        local_maximum, metile.simd_shuffle_xor(local_maximum, shift)
                    )
            previous_maximum = (
                row_maxima[row_index].value if REGISTER_STATS else statistics.load((0, row, 0))
            )
            maximum = metile.maximum(previous_maximum, local_maximum)
            factor = exponential(previous_maximum - maximum)
            local_sum = 0.0
            for component in range(len(row_scores)):
                probability = metile.where(
                    row_visibility[component], exponential(row_scores[component] - maximum), 0.0
                )
                local_sum = local_sum + probability
                score_values.store(
                    (row, subgroup_lane * metile.cdiv(KEY_TILE, SOFTMAX_LANES) + component),
                    probability,
                )
            if SOFTMAX_LANES == 32:
                local_sum = metile.simd_sum(local_sum)
            else:
                for shift in (1, 2, 4, 8) if SOFTMAX_LANES == 16 else (1, 2, 4):
                    local_sum = local_sum + metile.simd_shuffle_xor(local_sum, shift)
            previous_denominator = (
                row_denominators[row_index].value
                if REGISTER_STATS
                else statistics.load((1, row, 0))
            )
            denominator = (
                metile.fma(previous_denominator, factor, local_sum)
                if SOFTMAX_BASE2
                else previous_denominator * factor + local_sum
            )
            if REGISTER_STATS:
                row_maxima[row_index].update(maximum)
                row_denominators[row_index].update(denominator)
            else:
                statistics.store((0, row, subgroup_lane), maximum)
                statistics.store((1, row, subgroup_lane), denominator)
            factor_values.store((row, subgroup_lane), factor)
        metile.barrier()
        factor = factor_matrix.load((matrix_row, 0))
        for component in range(HEAD_DIM // output_step):
            outputs[component].update(outputs[component].value * factor)
        key_steps = range(0, KEY_TILE, 8) if UNROLL_MMA else metile.tile_range(0, KEY_TILE, 8)
        for key in key_steps:
            probability = score_matrix.load((matrix_row, key))
            for component in range(HEAD_DIM // output_step):
                value_fragment = (
                    metile.cast(
                        value_matrix.load(
                            (start + key, output_column + component * output_step),
                            scratch=device_scratch,
                        ),
                        "f32",
                    )
                    if DIRECT_MEMORY
                    else value_matrix.load((key, output_column + component * output_step))
                )
                outputs[component].update(
                    metile.dot(
                        probability,
                        value_fragment,
                        outputs[component].value,
                    )
                )
        metile.barrier()
    row_steps = (
        range(row_iterations)
        if UNROLL_SOFTMAX or REGISTER_STATS
        else metile.tile_range(0, row_iterations, 1)
    )
    for row_index in row_steps:
        row = row_start + (row_index * rows_per_iteration + subgroup) * row_step
        source_denominator = (
            row_denominators[row_index].value if REGISTER_STATS else statistics.load((1, row, 0))
        )
        denominator = metile.maximum(source_denominator, 1e-30)
        normalization = denominator if SOFTMAX_BASE2 else 1.0 / denominator
        factor_values.store((row, subgroup_lane), normalization)
    metile.barrier()
    normalizer = factor_matrix.load((matrix_row, 0))
    for component in range(HEAD_DIM // output_step):
        normalized_output = (
            outputs[component].value / normalizer
            if SOFTMAX_BASE2
            else outputs[component].value * normalizer
        )
        if DIRECT_MEMORY:
            output_matrix.store(
                (row_base + matrix_row, output_column + component * output_step),
                metile.cast(normalized_output, STORAGE_DTYPE),
                scratch=device_scratch,
            )
        else:
            query_matrix.store(
                (matrix_row, output_column + component * output_step),
                normalized_output,
            )
    if not DIRECT_MEMORY:
        metile.barrier()
    output_steps = (
        range(0) if DIRECT_MEMORY else metile.tile_range(thread, QUERY_TILE * HEAD_DIM, BLOCK)
    )
    for index in output_steps:
        row = index // HEAD_DIM
        feature = index % HEAD_DIM
        value = metile.where(row_base + row < valid_rows, query_values.load((row, feature)), 0.0)
        output.store((row_base + row, head, feature), _stored(value, STORAGE_DTYPE))
