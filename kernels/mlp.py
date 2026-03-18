import metile


@metile.kernel
def matmul_gelu(
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
    Fused GEMM + GELU epilogue: C = GELU(A @ B)
    """
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    acc = acc / (1.0 + metile.exp(0.0 - 1.702 * acc))
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))


@metile.kernel
def matmul_silu(
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
    Fused GEMM + SiLU epilogue: C = SiLU(A @ B)
    """
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    acc = acc / (1.0 + metile.exp(0.0 - acc))
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))
