import numpy as np
import pytest

import metile
from metile.runtime.metal_device import MetalDevice


@pytest.mark.parametrize("format", ["mxfp4", "mxfp8"])
def test_block_scaled_roundtrip(format):
    rng = np.random.default_rng(7)
    weight = rng.normal(size=(32, 64)).astype(np.float32)
    quantized = metile.BlockScaledWeight.quantize(weight, format=format)
    dequantized = quantized.dequantize()
    assert dequantized.shape == weight.shape
    assert np.isfinite(dequantized).all()
    tolerance = 0.15 if format == "mxfp4" else 0.1
    relative_error = np.linalg.norm(weight - dequantized) / np.linalg.norm(weight)
    assert relative_error < tolerance


def test_block_scaled_alignment_validation():
    with pytest.raises(ValueError, match="K must be divisible"):
        metile.BlockScaledWeight.quantize(np.zeros((31, 64), dtype=np.float32))


@pytest.mark.skipif(not MetalDevice.get().supports_tensor_ops, reason="requires Metal 4 MPP")
@pytest.mark.parametrize("format", ["mxfp4", "mxfp8"])
def test_block_scaled_matmul_matches_dequantized_reference(format):
    rng = np.random.default_rng(11)
    activations = rng.normal(size=(64, 64)).astype(np.float32)
    weight = rng.normal(size=(64, 64)).astype(np.float32)
    quantized = metile.BlockScaledWeight.quantize(weight, format=format)
    activations_buffer = metile.Buffer(data=activations)

    output = metile.block_scaled_matmul(activations_buffer, quantized).numpy()
    expected = activations @ quantized.dequantize()
    np.testing.assert_allclose(output, expected, rtol=5e-2, atol=5e-2)
