import metile
from metile.compiler.gemm_configs import matmul_candidates

MATMUL_CONFIGS = [metile.Config(**candidate) for candidate in matmul_candidates()]


@metile.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"], verbose=False)
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
    Runtime-tuned GEMM. Explicit BLOCK_M/BLOCK_N/BLOCK_K values bypass tuning.
    """
    left = metile.tensor(A, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K), access="read")
    right = metile.tensor(B, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N), access="read")
    output = metile.tensor(C, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N), access="write")
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = left.load((pid_m * BLOCK_M, k))
        b = right.load((k, pid_n * BLOCK_N))
        acc = metile.dot(a, b, acc)
    output.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc)


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
    left = metile.tensor(A, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K), access="read")
    right = metile.tensor(B, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N), access="read")
    output = metile.tensor(C, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N), access="write")
    pid_m, pid_n = metile.tile_swizzle(
        metile.program_id(0),
        metile.program_id(1),
        pattern="morton",
        block_size=2,
    )
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = left.load((pid_m * BLOCK_M, k))
        b = right.load((k, pid_n * BLOCK_N))
        acc = metile.dot(a, b, acc)
    output.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc)


@metile.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"], verbose=False)
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
    left = metile.tensor(A, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K), access="read")
    right = metile.tensor(B, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N), access="read")
    output = metile.tensor(C, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N), access="write")
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = left.load((pid_m * BLOCK_M, k))
        b = right.load((k, pid_n * BLOCK_N))
        acc = metile.dot(a, b, acc)
    acc = metile.where(acc > 0, acc, 0)
    output.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc)
