import numpy as np

import metile
from kernels.softmax import softmax
from metile.runtime.metal_device import MetalDevice


def _run_softmax(rows, hidden, block=256):
    """Run softmax and check against numpy reference."""
    X_np = np.random.randn(rows, hidden).astype(np.float32)

    X_buf = metile.Buffer(data=X_np.ravel())
    Out_buf = metile.Buffer.zeros((rows * hidden,))

    softmax[(rows,)](X_buf, Out_buf, hidden, BLOCK=block)
    MetalDevice.get().sync()

    result = Out_buf.numpy().reshape(rows, hidden)

    # numpy reference
    m = np.max(X_np, axis=-1, keepdims=True)
    e = np.exp(X_np - m)
    ref = e / np.sum(e, axis=-1, keepdims=True)

    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)

    # rows should sum to 1
    row_sums = result.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(rows), rtol=1e-4, atol=1e-4)

    # all values in [0, 1]
    assert np.all(result >= 0)
    assert np.all(result <= 1)


class TestSoftmax:
    def test_small_32x64(self):
        _run_softmax(32, 64)

    def test_small_64x128(self):
        _run_softmax(64, 128)

    def test_medium_128x512(self):
        _run_softmax(128, 512)

    def test_medium_256x1024(self):
        _run_softmax(256, 1024)

    def test_large_512x2048(self):
        _run_softmax(512, 2048)

    def test_non_power_of_2_cols(self):
        _run_softmax(64, 300)

    def test_single_row(self):
        _run_softmax(1, 512)

    def test_uniform_input(self):
        """Uniform input should give uniform output."""
        rows, hidden = 32, 128
        X_np = np.ones((rows, hidden), dtype=np.float32) * 3.0

        X_buf = metile.Buffer(data=X_np.ravel())
        Out_buf = metile.Buffer.zeros((rows * hidden,))

        softmax[(rows,)](X_buf, Out_buf, hidden, BLOCK=256)
        MetalDevice.get().sync()

        result = Out_buf.numpy().reshape(rows, hidden)
        expected = np.full((rows, hidden), 1.0 / hidden, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
