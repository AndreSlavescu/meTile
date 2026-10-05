"""Batched Qwen3 projections over setup-time, row-major weight packs.

The host transposes each original ``[N, K]`` projection once into ``[K, N]``
storage, then reuses it across prompt chunks. Concatenate Q/K/V or gate/up
along N before packing to compute those projections together. Every argument
is a direct, disjoint tensor buffer; no layer offset or per-chunk weight copy
is needed. Tensor storage dtypes must match; accumulation is FP32.

For full-precision FP32 comparisons, prepare with ``STRICT_MATH=True`` and
``SCHEDULE=metile.Schedule(backend="simdgroup")``. An explicitly selected
tensor-operations backend additionally requires ``RELAXED_PRECISION=False``.
"""

import metile


@metile.kernel
def qwen3_prefill_projection(
    Source,
    Weight,
    Output,
    M,
    N,
    K,
    *,
    BLOCK_M: metile.constexpr = 32,
    BLOCK_N: metile.constexpr = 64,
    BLOCK_K: metile.constexpr = 16,
):
    """Compute ``Output[M,N] = Source[M,K] @ Weight[K,N]``.

    Launch ``(ceil(M/BLOCK_M), ceil(N/BLOCK_N))`` threadgroups. Bounds checks
    handle incomplete tiles. M is the fixed prepared chunk capacity; padding
    source rows with zero lets row-wise consumers mask the final short chunk.
    """
    source = metile.tensor(Source, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K), access="read")
    weight = metile.tensor(Weight, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N), access="read")
    output = metile.tensor(Output, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N), access="write")
    row = metile.program_id(0) * BLOCK_M
    column = metile.program_id(1) * BLOCK_N
    accumulator = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for feature in metile.tile_range(0, K, BLOCK_K):
        accumulator = metile.dot(
            source.load((row, feature)), weight.load((feature, column)), accumulator
        )
    output.store((row, column), accumulator)
