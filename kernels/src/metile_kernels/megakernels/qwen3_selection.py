"""Two-dispatch GPU greedy selection with stable, lowest-token-ID ties.

Logits may be FP16 or FP32; partial maxima are FP32 and indices are int32.
NaN logits are unsupported. Negative infinity is valid, including a fully
masked chunk. Buffers must be disjoint and the caller orders both dispatches.
"""

import metile


def _validate_selection(vocabulary, chunks, block):
    if type(vocabulary) is not int or not 0 < vocabulary < 2**31:
        raise ValueError("VOCAB must be a positive signed-32-bit integer")
    if type(chunks) is not int or not 0 < chunks < 2**31:
        raise ValueError("CHUNKS must be a positive signed-32-bit integer")
    if type(block) is not int or block < 32 or block > 1024 or block % 32:
        raise ValueError("BLOCK must be a multiple of 32 between 32 and 1024")


def _reduce_selection(value, index, scratch_values, scratch_indices, vocabulary):
    thread = metile.thread_id()
    lane = metile.simd_lane_id()
    group = thread // 32
    maximum = metile.simd_max(value)
    selected = 0 - metile.simd_max(metile.where(value == maximum, 0 - index, -vocabulary))
    scratch_values.store((group, lane), maximum)
    scratch_indices.store((group, lane), selected)
    metile.barrier()
    group_value = scratch_values.load((lane, 0), other=-float("inf"))
    group_index = scratch_indices.load((lane, 0), other=vocabulary)
    maximum = metile.simd_max(group_value)
    selected = 0 - metile.simd_max(
        metile.where(group_value == maximum, 0 - group_index, -vocabulary)
    )
    return maximum, selected


@metile.kernel
def qwen3_argmax_partials(
    Logits,
    PartialValues,
    PartialIndices,
    *,
    VOCAB: metile.constexpr,
    CHUNKS: metile.constexpr,
    BLOCK: metile.constexpr = 256,
    STORAGE_DTYPE: metile.constexpr = "f32",
):
    """Launch ``CHUNKS`` groups, each reducing ``BLOCK`` consecutive logits.

    ``CHUNKS`` must cover the vocabulary; extra, fully padded chunks produce
    ``(-inf, VOCAB)``. Launch the final reduction after all partial writes.
    """
    _validate_selection(VOCAB, CHUNKS, BLOCK)
    if STORAGE_DTYPE not in ("f16", "f32"):
        raise ValueError("STORAGE_DTYPE must be f16 or f32")
    if metile.cdiv(VOCAB, BLOCK) > CHUNKS or CHUNKS * BLOCK >= 2**31:
        raise ValueError("CHUNKS must cover VOCAB within signed-32-bit addressing")
    logits = metile.tensor(Logits, shape=(VOCAB,), access="read")
    partial_values = metile.tensor(PartialValues, shape=(CHUNKS, 1), access="write")
    partial_indices = metile.tensor(PartialIndices, shape=(CHUNKS, 1), access="write")
    scratch_values = metile.tensor(metile.shared(BLOCK // 32, dtype="f32"), shape=(BLOCK // 32, 1))
    scratch_indices = metile.tensor(metile.shared(BLOCK // 32, dtype="i32"), shape=(BLOCK // 32, 1))
    chunk = metile.program_id(0)
    thread = metile.thread_id()
    offset = chunk * BLOCK + thread
    value = metile.cast(logits.load((offset,), other=-float("inf")), "f32")
    index = metile.where(offset < VOCAB, offset, VOCAB)
    maximum, selected = _reduce_selection(value, index, scratch_values, scratch_indices, VOCAB)
    partial_values.store((chunk, thread), maximum)
    partial_indices.store((chunk, thread), selected)


@metile.kernel
def qwen3_argmax_finalize(
    PartialValues,
    PartialIndices,
    Output,
    *,
    VOCAB: metile.constexpr,
    CHUNKS: metile.constexpr,
    BLOCK: metile.constexpr = 256,
):
    """Launch one group to reduce all partials into ``Output[1]`` int32."""
    _validate_selection(VOCAB, CHUNKS, BLOCK)
    partial_values = metile.tensor(PartialValues, shape=(CHUNKS,), access="read")
    partial_indices = metile.tensor(PartialIndices, shape=(CHUNKS,), access="read")
    output = metile.tensor(Output, shape=(1,), access="write")
    scratch_values = metile.tensor(metile.shared(BLOCK // 32, dtype="f32"), shape=(BLOCK // 32, 1))
    scratch_indices = metile.tensor(metile.shared(BLOCK // 32, dtype="i32"), shape=(BLOCK // 32, 1))
    thread = metile.thread_id()
    maximum = metile.loop_state(-float("inf"))
    selected = metile.loop_state(VOCAB)
    for chunk in metile.tile_range(thread, CHUNKS, BLOCK):
        value = partial_values.load((chunk,))
        index = partial_indices.load((chunk,))
        next_index = metile.where(
            value > maximum.value,
            index,
            metile.where(
                value == maximum.value, metile.minimum(index, selected.value), selected.value
            ),
        )
        selected.update(next_index)
        maximum.update(metile.maximum(maximum.value, value))
    _maximum, token = _reduce_selection(
        maximum.value, selected.value, scratch_values, scratch_indices, VOCAB
    )
    output.store((thread,), token)
