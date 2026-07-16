import numpy as np
import pytest

import metile
from metile.compiler.block_scaled import lower_block_scaled_matmul
from metile.ir import metal_ir as mir
from metile.runtime.block_scaled import _prepare_block_scaled_dispatch
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


def test_block_scaled_k64_staging_runs_two_mpp_steps_per_barrier_pair():
    function = lower_block_scaled_matmul("test_bsmm", 64, 64, 64, 4, block_k=64)
    loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    assert loop.step == 64
    assert sum(isinstance(op, mir.MMatmul2dRun) for op in loop.body) == 2


def test_block_scaled_register_fragments_eliminate_staging_and_barriers():
    function = lower_block_scaled_matmul(
        "test_bsmm_register", 64, 64, 64, 4, register_fragments=True
    )
    loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    assert loop.step == 16
    assert any(isinstance(op, mir.MNaxBlockScaledRun) for op in loop.body)
    assert not any(isinstance(op, mir.MBlockScaledTileLoad) for op in loop.body)
    assert not any(isinstance(op, mir.MBarrier) for op in loop.body)


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


@pytest.mark.skipif(not MetalDevice.get().supports_tensor_ops, reason="requires Metal 4 MPP")
def test_block_scaled_k64_dispatch_matches_dequantized_reference():
    rng = np.random.default_rng(23)
    activations = rng.normal(size=(64, 64)).astype(np.float32)
    weight = rng.normal(size=(64, 64)).astype(np.float32)
    quantized = metile.BlockScaledWeight.quantize(weight, format="mxfp4")
    activations_buffer = metile.Buffer(data=activations)
    output_buffer = metile.Buffer.empty((64, 64))
    dispatch = _prepare_block_scaled_dispatch(
        activations_buffer, quantized, output_buffer, 64, 64, 64
    )
    dispatch()
    output = output_buffer.numpy()
    expected = activations @ quantized.dequantize()
    np.testing.assert_allclose(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not MetalDevice.get().supports_tensor_ops, reason="requires Metal 4 MPP")
@pytest.mark.parametrize("format", ["mxfp4", "mxfp8"])
def test_block_scaled_register_dispatch_matches_dequantized_reference(format):
    rng = np.random.default_rng(31)
    activations = rng.normal(size=(64, 64)).astype(np.float32)
    weight = rng.normal(size=(64, 64)).astype(np.float32)
    quantized = metile.BlockScaledWeight.quantize(weight, format=format)
    activations_buffer = metile.Buffer(data=activations)
    output_buffer = metile.Buffer.empty((64, 64))
    dispatch = _prepare_block_scaled_dispatch(
        activations_buffer,
        quantized,
        output_buffer,
        64,
        64,
        register_fragments=True,
        schedule="linear",
    )
    dispatch()
    output = output_buffer.numpy()
    expected = activations @ quantized.dequantize()
    np.testing.assert_allclose(output, expected, rtol=5e-2, atol=5e-2)
