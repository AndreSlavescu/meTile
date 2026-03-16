import numpy as np

import metile
from kernels.layernorm import layernorm
from metile.runtime.metal_device import MetalDevice


def _run_layernorm(rows, hidden, block=256):
    X_np = np.random.randn(rows, hidden).astype(np.float32)
    W_np = np.random.randn(hidden).astype(np.float32)
    B_np = np.random.randn(hidden).astype(np.float32)

    X_buf = metile.Buffer(data=X_np.ravel())
    W_buf = metile.Buffer(data=W_np.ravel())
    B_buf = metile.Buffer(data=B_np.ravel())
    Out_buf = metile.Buffer.zeros((rows * hidden,))

    layernorm[(rows,)](X_buf, W_buf, B_buf, Out_buf, hidden, BLOCK=block)
    MetalDevice.get().sync()

    result = Out_buf.numpy().reshape(rows, hidden)

    # numpy reference: y = (x - mean) / sqrt(var + eps) * w + b
    mean = np.mean(X_np, axis=-1, keepdims=True)
    var = np.var(X_np, axis=-1, keepdims=True)
    ref = (X_np - mean) / np.sqrt(var + 1e-5) * W_np + B_np

    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)


class TestLayerNorm:
    def test_small_32x64(self):
        _run_layernorm(32, 64)

    def test_small_64x128(self):
        _run_layernorm(64, 128)

    def test_medium_128x512(self):
        _run_layernorm(128, 512)

    def test_medium_256x1024(self):
        _run_layernorm(256, 1024)

    def test_large_512x2048(self):
        _run_layernorm(512, 2048)

    def test_non_power_of_2_cols(self):
        _run_layernorm(64, 300)

    def test_single_row(self):
        _run_layernorm(1, 512)

    def test_identity_params(self):
        """W=1, B=0 should give standard normalization."""
        rows, hidden = 32, 256
        X_np = np.random.randn(rows, hidden).astype(np.float32)
        W_np = np.ones(hidden, dtype=np.float32)
        B_np = np.zeros(hidden, dtype=np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W_buf = metile.Buffer(data=W_np.ravel())
        B_buf = metile.Buffer(data=B_np.ravel())
        Out_buf = metile.Buffer.zeros((rows * hidden,))

        layernorm[(rows,)](X_buf, W_buf, B_buf, Out_buf, hidden, BLOCK=256)
        MetalDevice.get().sync()

        result = Out_buf.numpy().reshape(rows, hidden)

        mean = np.mean(X_np, axis=-1, keepdims=True)
        var = np.var(X_np, axis=-1, keepdims=True)
        ref = (X_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)

    def test_constant_input(self):
        """Constant input -> zero after normalization, output = bias."""
        rows, hidden = 16, 128
        X_np = np.full((rows, hidden), 5.0, dtype=np.float32)
        W_np = np.random.randn(hidden).astype(np.float32)
        B_np = np.random.randn(hidden).astype(np.float32)

        X_buf = metile.Buffer(data=X_np.ravel())
        W_buf = metile.Buffer(data=W_np.ravel())
        B_buf = metile.Buffer(data=B_np.ravel())
        Out_buf = metile.Buffer.zeros((rows * hidden,))

        layernorm[(rows,)](X_buf, W_buf, B_buf, Out_buf, hidden, BLOCK=256)
        MetalDevice.get().sync()

        result = Out_buf.numpy().reshape(rows, hidden)
        # (x - mean) = 0, so output = 0 * w + b = b
        ref = np.broadcast_to(B_np, (rows, hidden))
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)
