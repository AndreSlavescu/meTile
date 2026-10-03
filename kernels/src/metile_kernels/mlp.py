import metile
from metile_kernels.gemm import MATMUL_CONFIGS


@metile.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"], verbose=False)
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
    acc = acc / (1.0 + metile.exp(0.0 - 1.702 * acc))
    output.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc)


@metile.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"], verbose=False)
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
    acc = acc / (1.0 + metile.exp(0.0 - acc))
    output.store((pid_m * BLOCK_M, pid_n * BLOCK_N), acc)
