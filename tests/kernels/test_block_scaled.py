import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.block_scaled import lower_block_scaled_matmul
from metile.compiler.passes import decompose_nax_fragments
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


def test_block_scaled_register_fragments_support_fp16_model_io():
    function = decompose_nax_fragments(
        lower_block_scaled_matmul(
            "test_bsmm_fp16",
            32,
            64,
            64,
            4,
            block_m=32,
            block_n=64,
            register_fragments=True,
            fragment_type="bfloat",
            activation_type="f16",
            output_type="f16",
        )
    )
    source = emit(function)

    assert "device half* activations" in source
    assert "get_left_input_cooperative_tensor<half, bfloat, float>" in source
    assert "device half* output" in source


def test_block_scaled_register_fragments_mask_ragged_model_rows():
    function = decompose_nax_fragments(
        lower_block_scaled_matmul(
            "test_bsmm_ragged",
            127,
            64,
            64,
            8,
            block_m=32,
            block_n=64,
            register_fragments=True,
            activation_type="f16",
            output_type="f16",
        )
    )
    source = emit(function)

    assert "const uint grid_m = (uint(M) + 32u - 1u) / 32u" in source
    assert "< 127u" in source


def test_block_scaled_staging_rejects_mixed_precision_io():
    with pytest.raises(ValueError, match="requires register fragments"):
        lower_block_scaled_matmul(
            "test_bsmm_fp16_staging",
            64,
            64,
            64,
            4,
            activation_type="f16",
            output_type="f16",
        )


def test_nax_fragment_pass_exposes_composable_native_operations():
    function = lower_block_scaled_matmul(
        "test_bsmm_decomposed", 64, 64, 64, 4, register_fragments=True
    )
    decompose_nax_fragments(function)

    loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    assert not any(isinstance(op, mir.MNaxBlockScaledRun) for op in loop.body)
    assert sum(isinstance(op, mir.MNaxLoadBlockScale) for op in loop.body) == 2
    assert sum(isinstance(op, mir.MNaxLoadBlockScaledFragment) for op in loop.body) == 2
    assert sum(isinstance(op, mir.MNaxLoadFragment) for op in loop.body) == 2
    assert sum(isinstance(op, mir.MNaxFmaFragment) for op in loop.body) == 2
    assert sum(isinstance(op, mir.MNaxStoreFragment) for op in function.ops) == 4


def test_block_scaled_k_unroll_reuses_scale_fragments():
    function = lower_block_scaled_matmul(
        "test_bsmm_unroll",
        64,
        64,
        64,
        4,
        register_fragments=True,
        k_unroll=2,
    )
    decompose_nax_fragments(function)

    loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    assert loop.step == 32
    assert sum(isinstance(op, mir.MNaxLoadBlockScale) for op in loop.body) == 2
    fragments = [op for op in loop.body if isinstance(op, mir.MNaxLoadBlockScaledFragment)]
    assert len(fragments) == 4
    assert {op.k_offset for op in fragments} == {0, 16}


def test_block_scaled_register_reduction_epochs_are_explicit_ir():
    function = lower_block_scaled_matmul(
        "test_bsmm_epochs",
        64,
        64,
        64,
        4,
        register_fragments=True,
        outer_k=64,
    )
    outer_loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    inner_loop = next(op for op in outer_loop.body if isinstance(op, mir.MForLoop))
    assert outer_loop.step == 64
    assert inner_loop.step == 16
    assert any(isinstance(op, mir.MBarrier) for op in outer_loop.body)


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
@pytest.mark.parametrize("fragment_type", ["float", "bfloat"])
@pytest.mark.parametrize("k_unroll", [1, 2])
def test_block_scaled_register_dispatch_matches_dequantized_reference(
    format, fragment_type, k_unroll
):
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
        fragment_type=fragment_type,
        k_unroll=k_unroll,
    )
    dispatch()
    output = output_buffer.numpy()
    expected = activations @ quantized.dequantize()
    np.testing.assert_allclose(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not MetalDevice.get().supports_tensor_ops, reason="requires Metal 4 MPP")
def test_block_scaled_two_simdgroup_tile_matches_dequantized_reference():
    rng = np.random.default_rng(37)
    activations = rng.normal(size=(64, 64)).astype(np.float32)
    weight = rng.normal(size=(64, 64)).astype(np.float32)
    quantized = metile.BlockScaledWeight.quantize(weight, format="mxfp4")
    activations_buffer = metile.Buffer(data=activations)
    output_buffer = metile.Buffer.empty((64, 64))
    dispatch = _prepare_block_scaled_dispatch(
        activations_buffer,
        quantized,
        output_buffer,
        32,
        64,
        register_fragments=True,
        schedule="linear",
        fragment_type="bfloat",
        k_unroll=2,
    )

    dispatch()
    output = output_buffer.numpy()
    expected = activations @ quantized.dequantize()
    np.testing.assert_allclose(output, expected, rtol=5e-2, atol=5e-2)
