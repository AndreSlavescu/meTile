import numpy as np

from kernels.fft import fft


class TestFFT:
    def _run_fft(self, batch, N, tol=1e-3):
        x = np.random.randn(batch, N).astype(np.float32)
        x_re = x.ravel()
        x_im = np.zeros_like(x_re)

        y_re, y_im = fft(x_re, x_im, batch, N)

        ref = np.fft.fft(x.astype(np.float64), axis=-1)
        y = (y_re + 1j * y_im).reshape(batch, N)
        np.testing.assert_allclose(y.real, ref.real, rtol=tol, atol=tol)
        np.testing.assert_allclose(y.imag, ref.imag, rtol=tol, atol=tol)

    def test_8(self):
        self._run_fft(1, 8)

    def test_16(self):
        self._run_fft(1, 16)

    def test_64(self):
        self._run_fft(1, 64)

    def test_256(self):
        self._run_fft(1, 256)

    def test_1024(self):
        self._run_fft(1, 1024)

    def test_batch(self):
        self._run_fft(32, 256)

    def test_large_batch(self):
        self._run_fft(64, 1024)

    def test_2048(self):
        self._run_fft(1, 2048)

    def test_batch_2048(self):
        self._run_fft(32, 2048)
