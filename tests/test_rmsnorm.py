import numpy as np

import metile
from kernels.rmsnorm import rmsnorm
from metile.runtime.metal_device import MetalDevice


def _run_rmsnorm(rows, hidden, block=256):
    """Run rmsnorm and check against numpy reference."""
    X_np = np.random.randn(rows, hidden).astype(np.float32)
    W_np = np.random.randn(hidden).astype(np.float32)

    X_buf = metile.Buffer(data=X_np.ravel())
    W_buf = metile.Buffer(data=W_np.ravel())
    Out_buf = metile.Buffer.zeros((rows * hidden,))

    rmsnorm[(rows,)](X_buf, W_buf, Out_buf, hidden, BLOCK=block)
    MetalDevice.get().sync()

    result = Out_buf.numpy().reshape(rows, hidden)

    # numpy reference
    rms = np.sqrt(np.mean(X_np**2, axis=-1, keepdims=True) + 1e-5)
    ref = X_np / rms * W_np

    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)


class TestRMSNorm:
    def test_small_32x64(self):
        _run_rmsnorm(32, 64)

    def test_small_64x128(self):
        _run_rmsnorm(64, 128)

    def test_medium_128x512(self):
        _run_rmsnorm(128, 512)

    def test_medium_256x1024(self):
        _run_rmsnorm(256, 1024)

    def test_large_512x2048(self):
        _run_rmsnorm(512, 2048)

    def test_non_power_of_2_cols(self):
        _run_rmsnorm(64, 300)

    def test_single_row(self):
        _run_rmsnorm(1, 512)

    def test_unit_weights(self):
        """Unit weights should give plain RMS normalization."""
        rows, hidden = 32, 256
        X_np = np.random.randn(rows, hidden).astype(np.float32)
        W_np = np.ones(hidden, dtype=np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W_buf = metile.Buffer(data=W_np.ravel())
        Out_buf = metile.Buffer.zeros((rows * hidden,))

        rmsnorm[(rows,)](X_buf, W_buf, Out_buf, hidden, BLOCK=256)
        MetalDevice.get().sync()

        result = Out_buf.numpy().reshape(rows, hidden)

        rms = np.sqrt(np.mean(X_np**2, axis=-1, keepdims=True) + 1e-5)
        ref = X_np / rms
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_zero_input(self):
        """Zero input should give zero output regardless of weights."""
        rows, hidden = 16, 128
        X_np = np.zeros((rows, hidden), dtype=np.float32)
        W_np = np.random.randn(hidden).astype(np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W_buf = metile.Buffer(data=W_np.ravel())
        Out_buf = metile.Buffer.zeros((rows * hidden,))

        rmsnorm[(rows,)](X_buf, W_buf, Out_buf, hidden, BLOCK=256)
        MetalDevice.get().sync()

        result = Out_buf.numpy().reshape(rows, hidden)
        np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-5)
