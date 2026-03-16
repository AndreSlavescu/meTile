import metile


@metile.kernel
def matmul(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    """
    Compiler-inferred tile schedule.
    Automatically chooses the Morton-order 2x2 swizzle pattern (Z-order).
    """
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


@metile.kernel
def matmul_swizzled(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    """
    User-defined tile schedule with explicit Morton swizzle.
    """
    pid_m, pid_n = metile.tile_swizzle(
        metile.program_id(0),
        metile.program_id(1),
        pattern="morton",
        block_size=2,
    )
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


@metile.kernel
def matmul_relu(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    """
    Fused GEMM + ReLU epilogue
    """
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    acc = metile.where(acc > 0, acc, 0)
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))
