import numpy as np

import metile
from kernels.gemm import matmul
from metile.runtime.metal_device import MetalDevice

# tensor_ops uses TF32 (reduced precision) — need wider tolerance
_TENSOR_OPS = MetalDevice.get().supports_tensor_ops


def _run_matmul(M, N, K, BM, BN, BK, dtype=np.float32):
    """Helper to run matmul and check against numpy."""
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)

    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN

    matmul[(grid_m, grid_n)](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
    )

    expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(dtype)
    if _TENSOR_OPS and dtype == np.float16:
        rtol, atol = 8e-2, 8e-2  # tensor_ops TF32 + f16 accumulation
    elif _TENSOR_OPS:
        rtol, atol = 5e-2, 5e-2  # tensor_ops reduced precision
    elif dtype == np.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-4, 1e-4
    np.testing.assert_allclose(C, expected, rtol=rtol, atol=atol)


class TestNaiveGemm:
    def test_square_32x32(self):
        _run_matmul(32, 32, 32, BM=32, BN=32, BK=32)

    def test_square_64x64(self):
        _run_matmul(64, 64, 64, BM=32, BN=32, BK=32)

    def test_rectangular_64x32(self):
        _run_matmul(64, 32, 32, BM=32, BN=32, BK=32)

    def test_rectangular_32x64(self):
        _run_matmul(32, 64, 32, BM=32, BN=32, BK=32)

    def test_k_multiple_blocks(self):
        _run_matmul(32, 32, 128, BM=32, BN=32, BK=32)

    def test_larger_128x128(self):
        _run_matmul(128, 128, 128, BM=32, BN=32, BK=32)


class TestSimdgroupGemm:
    def test_square_64x64(self):
        _run_matmul(64, 64, 64, BM=64, BN=64, BK=32)

    def test_square_128x128(self):
        _run_matmul(128, 128, 128, BM=64, BN=64, BK=32)

    def test_square_256x256(self):
        _run_matmul(256, 256, 256, BM=64, BN=64, BK=32)

    def test_rectangular_128x64(self):
        _run_matmul(128, 64, 64, BM=64, BN=64, BK=32)

    def test_rectangular_64x128(self):
        _run_matmul(64, 128, 64, BM=64, BN=64, BK=32)

    def test_k_multiple_blocks(self):
        _run_matmul(64, 64, 256, BM=64, BN=64, BK=32)

    def test_larger_512x512(self):
        _run_matmul(512, 512, 512, BM=64, BN=64, BK=32)


class TestF16NaiveGemm:
    def test_square_32x32(self):
        _run_matmul(32, 32, 32, BM=32, BN=32, BK=32, dtype=np.float16)

    def test_square_64x64(self):
        _run_matmul(64, 64, 64, BM=32, BN=32, BK=32, dtype=np.float16)

    def test_larger_128x128(self):
        _run_matmul(128, 128, 128, BM=32, BN=32, BK=32, dtype=np.float16)


class TestF16SimdgroupGemm:
    def test_square_64x64(self):
        _run_matmul(64, 64, 64, BM=64, BN=64, BK=32, dtype=np.float16)

    def test_square_128x128(self):
        _run_matmul(128, 128, 128, BM=64, BN=64, BK=32, dtype=np.float16)

    def test_square_256x256(self):
        _run_matmul(256, 256, 256, BM=64, BN=64, BK=32, dtype=np.float16)

    def test_larger_512x512(self):
        _run_matmul(512, 512, 512, BM=64, BN=64, BK=32, dtype=np.float16)


class TestAutotunedGemm:
    def test_autotune_selects_config(self):
        """Autotuner picks a config and produces correct results."""

        @metile.autotune(
            configs=[
                metile.Config(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32),
                metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32),
            ],
            key=["M", "N", "K"],
            verbose=False,
        )
        @metile.kernel
        def matmul_auto(
            A_ptr,
            B_ptr,
            C_ptr,
            M: int,
            N: int,
            K: int,
            BLOCK_M: metile.constexpr,
            BLOCK_N: metile.constexpr,
            BLOCK_K: metile.constexpr,
        ):
            pid_m = metile.program_id(0)
            pid_n = metile.program_id(1)
            acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
            for k in metile.tile_range(0, K, BLOCK_K):
                a = metile.tile_load(A_ptr, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
                b = metile.tile_load(B_ptr, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
                acc = metile.dot(a, b, acc)
            metile.tile_store(C_ptr, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))

        M, N, K = 128, 128, 128
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        def grid(meta):
            return (metile.cdiv(M, meta["BLOCK_M"]), metile.cdiv(N, meta["BLOCK_N"]))

        matmul_auto[grid](A, B, C, M, N, K)

        expected = A @ B
        tol = 5e-2 if _TENSOR_OPS else 1e-4
        np.testing.assert_allclose(C, expected, rtol=tol, atol=tol)

    def test_autotune_cached(self):
        """Second call uses cached config (no re-tuning)."""
        from metile.frontend.autotune import _autotune_cache

        @metile.autotune(
            configs=[metile.Config(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)],
            key=["M", "N", "K"],
            verbose=False,
        )
        @metile.kernel
        def matmul_cached(
            A_ptr,
            B_ptr,
            C_ptr,
            M: int,
            N: int,
            K: int,
            BLOCK_M: metile.constexpr,
            BLOCK_N: metile.constexpr,
            BLOCK_K: metile.constexpr,
        ):
            pid_m = metile.program_id(0)
            pid_n = metile.program_id(1)
            acc = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
            for k in metile.tile_range(0, K, BLOCK_K):
                a = metile.tile_load(A_ptr, pid_m * BLOCK_M, k, K, (BLOCK_M, BLOCK_K))
                b = metile.tile_load(B_ptr, k, pid_n * BLOCK_N, N, (BLOCK_K, BLOCK_N))
                acc = metile.dot(a, b, acc)
            metile.tile_store(C_ptr, pid_m * BLOCK_M, pid_n * BLOCK_N, N, acc, (BLOCK_M, BLOCK_N))

        M, N, K = 64, 64, 64
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        def grid(meta):
            return (metile.cdiv(M, meta["BLOCK_M"]), metile.cdiv(N, meta["BLOCK_N"]))

        # First call: autotunes
        matmul_cached[grid](A, B, C, M, N, K)
        assert ("matmul_cached", (64, 64, 64)) in _autotune_cache

        # Second call: uses cache
        C2 = np.zeros((M, N), dtype=np.float32)
        matmul_cached[grid](A, B, C2, M, N, K)
        tol = 5e-2 if _TENSOR_OPS else 1e-4
        np.testing.assert_allclose(C2, A @ B, rtol=tol, atol=tol)
