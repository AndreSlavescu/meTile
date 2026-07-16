import metile

MATMUL_CONFIGS = [
    # Small K tiles keep register pressure low in the measured M5 candidate family.
    # Schedule variants are measured on the concrete workload and persisted.
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="linear"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="diagonal"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="morton"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="hilbert"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="grouped2"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="grouped4"),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, WM=2, WN=2, SWIZZLE="grouped8"),
    metile.Config(
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=2,
        WN=2,
        SWIZZLE="grouped8",
        NAX_FRAGMENTS=True,
    ),
    metile.Config(
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=2,
        WN=2,
        SWIZZLE="grouped8",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=128,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=256,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=2,
        WN=4,
        SWIZZLE="morton",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=256,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=2,
        WN=4,
        SWIZZLE="morton",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=4,
        WN=2,
        SWIZZLE="morton",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=4,
        WN=2,
        SWIZZLE="hilbert",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=256,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=8,
        WN=2,
        SWIZZLE="morton",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
        NAX_SKIP_FIRST_EPOCH_BARRIER=True,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
        NAX_TRAILING_EPOCH_BARRIER=True,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="hilbert",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
        NAX_TRAILING_EPOCH_BARRIER=True,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="hilbert",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=512,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=1024,
        NAX_K_UNROLL=2,
    ),
    metile.Config(
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=16,
        WM=4,
        WN=4,
        SWIZZLE="grouped4",
        NAX_FRAGMENTS=True,
        NAX_OUTER_K=1024,
        NAX_K_UNROLL=2,
        NAX_TRAILING_EPOCH_BARRIER=True,
    ),
    metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, WM=2, WN=2, SWIZZLE="auto"),
    metile.Config(BLOCK_M=64, BLOCK_N=128, BLOCK_K=16, WM=2, WN=4, SWIZZLE="auto"),
    metile.Config(BLOCK_M=64, BLOCK_N=128, BLOCK_K=16, WM=2, WN=4, SWIZZLE="grouped8"),
    metile.Config(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, WM=4, WN=2, SWIZZLE="auto"),
    metile.Config(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, WM=4, WN=4, SWIZZLE="auto"),
]


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
    pid_m = metile.program_id(0)
    pid_n = metile.program_id(1)
    acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for k in metile.tile_range(0, K, BLOCK_K):
        a = metile.tile_load(A, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
        b = metile.tile_load(B, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
        acc = metile.dot(a, b, acc)
    acc = metile.where(acc > 0, acc, 0)
    metile.tile_store(C, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))
