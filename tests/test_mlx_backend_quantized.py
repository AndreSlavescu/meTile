"""MLX backend tests: quantized."""

from types import SimpleNamespace

import numpy as np
import pytest

import metile
from tests.module_patching import _patch_mlx_lm


@pytest.mark.parametrize("interleaved", (False, True))
@pytest.mark.parametrize("k_unroll", (1, 2))
def test_mlx_dense_swiglu_qmv_matches_bfloat16_reference_exactly(
    interleaved,
    k_unroll,
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.backends import mlx_dense, mlx_dense_swiglu

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense_swiglu._kernel_cache.clear()
    random = np.random.default_rng(2032)
    activations = mx.array(random.normal(size=(1, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
    gate_native = mx.array(random.normal(size=(256, 256)).astype(np.float32)).astype(mx.bfloat16)
    up_native = mx.array(random.normal(size=(256, 256)).astype(np.float32)).astype(mx.bfloat16)
    gate_weight = mlx_dense.MLXDenseWeight.from_mlx(gate_native)
    up_weight = mlx_dense.MLXDenseWeight.from_mlx(up_native)
    paired_weight = mx.stack((gate_native, up_native), axis=-1) if interleaved else None
    config = mlx_dense_swiglu.MLXDenseSwiGLUConfig(
        "metile",
        implementation="simdgroup_paired" if interleaved else "simdgroup",
        outputs_per_simdgroup=4,
        simdgroups_per_threadgroup=2,
        k_unroll=k_unroll,
    )

    kernel = mlx_dense_swiglu._compile_mlx_dense_swiglu(
        1,
        256,
        256,
        activations.dtype,
        config,
    )
    actual = kernel(activations, gate_weight, up_weight, paired_weight)
    expected = nn.silu(activations @ gate_native.T) * (activations @ up_native.T)
    mx.eval(actual, expected)

    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


def test_mlx_exact_dense_qmv_requires_confirmation_headroom():
    from metile.backends import mlx_dense_swiglu

    native = mlx_dense_swiglu.MLXDenseSwiGLUConfig("mlx")
    exact = mlx_dense_swiglu.MLXDenseSwiGLUConfig(
        "metile",
        implementation="simdgroup",
        outputs_per_simdgroup=4,
        simdgroups_per_threadgroup=2,
    )

    close = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.995, 100, exact)])
    faster = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.98, 100, exact)])

    assert close == native
    assert faster == exact


def test_mlx_dense_qmv_only_offers_paired_schedules_with_paired_weights():
    from metile.backends import mlx_dense_swiglu

    separate = mlx_dense_swiglu._candidate_configs(1, 1536, 8960, False)
    paired = mlx_dense_swiglu._candidate_configs(1, 1536, 8960, True)

    assert all(config.implementation != "simdgroup_paired" for config in separate)
    assert any(config.implementation == "simdgroup_paired" for config in paired)
    assert all(config.implementation != "simdgroup" for config in paired)


@pytest.mark.parametrize("dtype_name", ("bfloat16", "float16"))
def test_mlx_dense_residual_qmv_matches_reference_exactly(dtype_name, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_dense_residual

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense_residual._kernel_cache.clear()
    mlx_dense_residual._schedule_cache.clear()
    random = np.random.default_rng(2033)
    dtype = getattr(mx, dtype_name)
    values = mx.array(random.normal(size=(1, 1, 256)).astype(np.float32)).astype(dtype)
    weight = mx.array(random.normal(size=(128, 256)).astype(np.float32)).astype(dtype)
    residual = mx.array(random.normal(size=(1, 1, 128)).astype(np.float32)).astype(dtype)

    actual = mlx_dense_residual.mlx_dense_residual_qmv(
        values,
        weight,
        residual,
        autotune=False,
    )
    expected = values @ weight.T + residual
    mx.eval(actual, expected)

    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


@pytest.mark.parametrize("rows", (2, 4, 8))
def test_mlx_dense_residual_qmv_matches_per_row_reference_exactly(rows, monkeypatch):
    """A batched step must equal decoding each row on its own, bit for bit."""
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_dense_residual

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense_residual._kernel_cache.clear()
    mlx_dense_residual._schedule_cache.clear()
    random = np.random.default_rng(4177)
    values = mx.array(random.normal(size=(rows, 256)).astype(np.float32)).astype(mx.bfloat16)
    weight = mx.array(random.normal(size=(128, 256)).astype(np.float32)).astype(mx.bfloat16)
    residual = mx.array(random.normal(size=(rows, 128)).astype(np.float32)).astype(mx.bfloat16)

    actual = mlx_dense_residual.mlx_dense_residual_qmv(
        values,
        weight,
        residual,
        autotune=False,
    )
    expected = mx.concatenate(
        [values[row : row + 1] @ weight.T + residual[row : row + 1] for row in range(rows)],
        axis=0,
    )
    mx.eval(actual, expected)

    assert actual.shape == (rows, 128)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


@pytest.mark.parametrize("rows", (2, 4, 8))
def test_mlx_dense_swiglu_qmv_matches_per_row_reference_exactly(rows, monkeypatch):
    """The batched SwiGLU must equal MLX's own one-row SwiGLU for every row."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from metile.backends import mlx_dense_swiglu
    from metile.backends.mlx_dense import MLXDenseWeight

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense_swiglu._kernel_cache.clear()
    mlx_dense_swiglu._schedule_cache.clear()
    random = np.random.default_rng(9311)
    values = mx.array(random.normal(size=(rows, 256)).astype(np.float32)).astype(mx.bfloat16)
    gate = mx.array(random.normal(size=(128, 256)).astype(np.float32)).astype(mx.bfloat16)
    up = mx.array(random.normal(size=(128, 256)).astype(np.float32)).astype(mx.bfloat16)
    paired = mx.stack((gate, up), axis=-1)
    mx.eval(paired)

    actual = mlx_dense_swiglu.mlx_dense_swiglu(
        values,
        MLXDenseWeight.from_mlx(gate),
        MLXDenseWeight.from_mlx(up),
        paired_weight=paired,
        autotune=False,
    )
    expected = mx.concatenate(
        [
            nn.silu(values[row : row + 1] @ gate.T) * (values[row : row + 1] @ up.T)
            for row in range(rows)
        ],
        axis=0,
    )
    mx.eval(actual, expected)

    assert actual.shape == (rows, 128)
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


def test_dense_qmv_candidates_bound_live_accumulators():
    """Row batching must not push the SIMDgroup into register spills."""
    from metile.backends import mlx_dense_residual, mlx_dense_swiglu

    wide = mlx_dense_residual._candidate_configs(31, 256, 128)
    assert all(
        config.algorithm == "mlx" or 31 * config.outputs_per_simdgroup <= 32 for config in wide
    )
    assert any(config.algorithm == "metile" for config in wide)
    # Above the NAX threshold the tile kernels take over from the QMV schedules.
    at_nax = mlx_dense_swiglu._candidate_configs(32, 256, 128, paired_available=True)
    assert all(
        not config.implementation.startswith("simdgroup")
        for config in at_nax
        if config.algorithm == "metile"
    )


@pytest.mark.parametrize("rows", (33, 127))
@pytest.mark.parametrize("dtype_name", ("bfloat16", "float16"))
def test_mlx_block_scaled_matmul_matches_ragged_low_precision_reference(
    rows,
    dtype_name,
    monkeypatch,
):
    from dataclasses import replace

    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_block_scaled

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_block_scaled._kernel_cache.clear()
    mlx_block_scaled._schedule_cache.clear()
    random = np.random.default_rng(52)
    activations = random.normal(size=(1, rows, 64)).astype(np.float32)
    dense_weight = random.normal(size=(64, 64)).astype(np.float32)
    weight = mlx_block_scaled.MLXBlockScaledWeight.quantize(
        dense_weight,
        format="mxfp8",
    )
    weight = replace(weight, native_values=None, native_scales=None)
    reference_weight = metile.BlockScaledWeight.quantize(
        dense_weight,
        format="mxfp8",
    ).dequantize()

    mlx_activations = mx.array(activations).astype(getattr(mx, dtype_name))
    actual = mlx_block_scaled.mlx_block_scaled_matmul(
        mlx_activations,
        weight,
        autotune=False,
    )
    mx.eval(actual)

    expected = np.array(mlx_activations.astype(mx.float32)) @ reference_weight
    assert actual.shape == (1, rows, 64)
    assert actual.dtype == getattr(mx, dtype_name)
    assert mlx_block_scaled.mlx_block_scaled_dispatches()[-1]["algorithm"] == "metile"
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected,
        rtol=5e-2,
        atol=2e-1,
    )


def test_mlx_block_scaled_native_representation_is_an_exact_fallback(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_block_scaled

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_block_scaled._schedule_cache.clear()
    random = np.random.default_rng(58)
    activations = mx.array(random.normal(size=(1, 17, 64)).astype(np.float16))
    weight = mlx_block_scaled.MLXBlockScaledWeight.quantize(
        random.normal(size=(64, 64)).astype(np.float32),
        format="mxfp4",
    )

    actual = mlx_block_scaled.mlx_block_scaled_matmul(activations, weight, autotune=False)
    expected = mlx_block_scaled._native_block_scaled_matmul(activations, weight)
    mx.eval(actual, expected)

    assert mlx_block_scaled.mlx_block_scaled_dispatches()[-1]["algorithm"] == "mlx"
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


def test_mlx_block_scaled_candidates_require_an_available_native_representation():
    from metile.backends import mlx_block_scaled

    with_native = mlx_block_scaled._candidate_configs(127, 2048, 2048, True)
    generated_only = mlx_block_scaled._candidate_configs(127, 2048, 2048, False)

    assert with_native[0].algorithm == "mlx"
    assert all(config.algorithm == "metile" for config in generated_only)
    assert {config.schedule for config in generated_only} >= {
        "grouped4",
        "hilbert",
        "linear",
        "morton",
    }


def test_mlx_native_affine_swiglu_matches_quantized_matmul(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    monkeypatch.setattr(
        mlx_quantized,
        "_AFFINE_SWIGLU_CONFIGS",
        (mlx_quantized.MLXAffineSwiGLUConfig("metile", "scalar", 32),),
    )
    mlx_quantized._affine_swiglu_schedule_cache.clear()
    random = np.random.default_rng(43)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)

    actual = mlx_quantized.mlx_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        autotune=False,
    )
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


def test_mlx_generated_affine8_swiglu_matches_bfloat16_native(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_quantized._affine_swiglu_schedule_cache.clear()
    random = np.random.default_rng(2027)
    input_features = 128
    output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float32)).astype(
        mx.bfloat16
    )
    gate = mx.array(
        random.normal(size=(output_features, input_features)).astype(np.float32)
    ).astype(mx.bfloat16)
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float32)).astype(
        mx.bfloat16
    )
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=128, bits=8)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=128, bits=8)

    actual = mlx_quantized.mlx_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size=128,
        bits=8,
        autotune=False,
    )
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        128,
        8,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=1e-2,
        atol=3e-2,
    )


def test_mlx_affine_swiglu_offers_a_multi_row_candidate():
    """Above one row the fused SwiGLU kernels do not apply, so one candidate must survive.

    Without this the tournament has only scalar kernels and native MLX to choose between
    for rows 2 to 31, and native wins the whole batched band by default.
    """
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    four_bit = mlx_quantized._affine_swiglu_configs(mx.float16, 4)
    assert any(config.implementation == "matmul" for config in four_bit)

    # from_mlx only accepts 4-bit group-64 weights, so the candidate must not be offered
    # at 8 bits where it could only fail to build.
    eight_bit = mlx_quantized._affine_swiglu_configs(mx.float16, 8)
    assert all(config.implementation != "matmul" for config in eight_bit)

    assert any(
        config.algorithm == "metile_matmul" for config in mlx_quantized._AFFINE_RESIDUAL_CONFIGS
    )


@pytest.mark.parametrize("rows", (2, 8))
def test_mlx_affine_matmul_swiglu_candidate_matches_native(rows, monkeypatch):
    """The multi-row candidate has to agree with native MLX before it may be selected."""
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    random = np.random.default_rng(71)
    input_features = 256
    output_features = 128
    values = mx.array(random.normal(size=(rows, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)

    executor, _ = mlx_quantized._make_affine_swiglu_executor(
        mlx_quantized.MLXAffineSwiGLUConfig("metile", "matmul"),
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    actual = executor(values)
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    mx.eval(actual, expected)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


def test_mlx_affine8_swiglu_configs_exclude_nax_and_bfloat16_half_decode():
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    configs = mlx_quantized._affine_swiglu_configs(mx.bfloat16, 8)

    assert all(config.implementation not in {"nax", "nax_scratch"} for config in configs)
    assert all(
        config.algorithm in {"mlx", "mlx_compiled"} or config.decode_dtype == "f32"
        for config in configs
    )


@pytest.mark.parametrize(
    ("block", "outputs_per_simdgroup", "decode_dtype"),
    ((64, 1, "f32"), (64, 2, "f32"), (64, 2, "f16")),
)
def test_mlx_generated_affine_residual_qmv_matches_native(
    block,
    outputs_per_simdgroup,
    decode_dtype,
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    random = np.random.default_rng(54)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float16))
    dense_weight = mx.array(
        random.normal(size=(output_features, input_features)).astype(np.float16)
    )
    residual = mx.array(random.normal(size=(1, 1, output_features)).astype(np.float16))
    weight, scales, biases = mx.quantize(dense_weight, group_size=64, bits=4)
    kernel = mlx_quantized._compile_affine_qmv(
        input_features,
        output_features,
        values.dtype,
        block=block,
        outputs_per_simdgroup=outputs_per_simdgroup,
        decode_dtype=decode_dtype,
        fuse_residual=True,
    )

    actual = kernel(values, weight, scales, biases, residual)
    expected = mlx_quantized._native_affine_residual_qmv(
        values,
        weight,
        scales,
        biases,
        residual,
        64,
        4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("lifetime_schedule", ("parallel", "scratch"))
def test_mlx_nax_affine_swiglu_matches_quantized_matmul(lifetime_schedule, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    random = np.random.default_rng(50)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)
    repacked = mlx_quantized._repacked_affine_pair(
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )
    kernel = mlx_quantized._compile_nax_affine_swiglu_qmv(
        input_features,
        output_features,
        values.dtype,
        block=64,
        lifetime_schedule=lifetime_schedule,
    )

    actual = kernel(values, *repacked)
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-1)


@pytest.mark.parametrize(
    ("compiled_latency", "generated_latency", "expected_algorithm"),
    (
        (0.96, 1.0, "mlx_compiled"),
        # A 1% compiled win is inside the run-to-run noise floor, so it must not switch.
        (0.99, 1.0, "mlx"),
        (0.998, 1.0, "mlx"),
        (1.0, 0.975, "mlx"),
        (1.0, 0.96, "metile"),
    ),
)
def test_mlx_affine_swiglu_uses_separate_exact_and_generated_margins(
    compiled_latency,
    generated_latency,
    expected_algorithm,
):
    from metile.backends import mlx_quantized

    native = mlx_quantized.MLXAffineSwiGLUConfig("mlx")
    compiled = mlx_quantized.MLXAffineSwiGLUConfig("mlx_compiled")
    generated = mlx_quantized.MLXAffineSwiGLUConfig("metile", "nax", 64)

    selected = mlx_quantized._choose_affine_swiglu_config(
        [
            (1.0, 0, native),
            (compiled_latency, 10, compiled),
            (generated_latency, 20, generated),
        ]
    )

    assert selected.algorithm == expected_algorithm


@pytest.mark.parametrize(
    ("compiled_latency", "generated_latency", "expected_algorithm"),
    (
        (0.96, 1.0, "mlx_compiled"),
        # A 1% compiled win is inside the run-to-run noise floor, so it must not switch.
        (0.99, 1.0, "mlx"),
        (0.998, 1.0, "mlx"),
        (1.0, 0.995, "mlx"),
        (1.0, 0.985, "metile"),
    ),
)
def test_mlx_affine_residual_uses_separate_exact_and_generated_margins(
    compiled_latency,
    generated_latency,
    expected_algorithm,
):
    from metile.backends import mlx_quantized

    native = mlx_quantized.MLXAffineResidualConfig("mlx")
    compiled = mlx_quantized.MLXAffineResidualConfig("mlx_compiled")
    generated = mlx_quantized.MLXAffineResidualConfig("metile", 256, 2)

    selected = mlx_quantized._choose_affine_residual_config(
        [
            (1.0, 0, native),
            (compiled_latency, 10, compiled),
            (generated_latency, 20, generated),
        ]
    )

    assert selected.algorithm == expected_algorithm


@pytest.mark.parametrize("rows", (1, 8))
def test_mlx_compiled_affine_swiglu_matches_native(rows, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setattr(mlx_quantized, "_compiled_affine_swiglu", None)
    random = np.random.default_rng(48)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, rows, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)

    actual = mlx_quantized._mlx_compiled_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0, atol=0)


def test_mlx_affine_swiglu_skips_decode_only_nax_for_multiple_rows():
    from metile.backends import mlx_quantized

    config = mlx_quantized.MLXAffineSwiGLUConfig("metile", "nax_scratch", 64)
    values = SimpleNamespace(shape=(1, 2, 64), size=128)
    weight = SimpleNamespace(shape=(64, 32))

    with pytest.raises(ValueError, match="one decode row"):
        mlx_quantized._affine_swiglu_dispatch(
            config,
            values,
            weight,
            None,
            None,
            weight,
            None,
            None,
            64,
            4,
        )


@pytest.mark.parametrize(
    ("outputs_per_simdgroup", "decode_dtype"),
    ((1, "f32"), (4, "f32"), (4, "f16")),
)
def test_mlx_affine_swiglu_schedule_dimensions_match_mlx(
    outputs_per_simdgroup,
    decode_dtype,
):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    random = np.random.default_rng(44)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)

    actual = mlx_quantized.mlx_affine_swiglu_qmv(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        outputs_per_simdgroup=outputs_per_simdgroup,
        decode_dtype=decode_dtype,
    )
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(("rows", "block"), ((1, 64), (33, 128)))
def test_mlx_affine_swiglu_scratch_schedule_matches_mlx(rows, block):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    random = np.random.default_rng(49)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, rows, input_features)).astype(np.float16))
    gate = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    up = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    gate_weight, gate_scales, gate_biases = mx.quantize(gate, group_size=64, bits=4)
    up_weight, up_scales, up_biases = mx.quantize(up, group_size=64, bits=4)
    kernel = mlx_quantized._compile_affine_swiglu_scratch_qmv(
        input_features,
        output_features,
        values.dtype,
        block=block,
        outputs_per_simdgroup=2,
    )

    actual = kernel(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )
    expected = mlx_quantized._native_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        64,
        4,
    )
    register_fused = mlx_quantized._compile_affine_swiglu_qmv(
        input_features,
        output_features,
        values.dtype,
        block=block,
        outputs_per_simdgroup=2,
    )(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )
    mx.eval(actual, expected, register_fused)

    np.testing.assert_allclose(np.array(actual), np.array(register_fused), rtol=0, atol=0)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-1)


def test_mlx_affine_repack_runs_native_tensor_kernel():
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    random = np.random.default_rng(47)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, 1, input_features)).astype(np.float16))
    weight = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
    repacked = mlx_quantized.repack_mlx_affine_weight(packed, scales, biases)
    actual = mlx_quantized.mlx_affine_qmv_nax(
        values,
        *repacked,
        output_features=output_features,
    )
    expected = mx.quantized_matmul(
        values,
        packed,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=64,
        bits=4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


def test_mlx_affine_matmul_matches_native_ragged_prefill(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_affine

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_affine._kernel_cache.clear()
    mlx_affine._schedule_cache.clear()
    random = np.random.default_rng(53)
    rows = 33
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, rows, input_features)).astype(np.float16))
    dense = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    packed, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    weight = mlx_affine.MLXAffineWeight.from_mlx(packed, scales, biases)

    actual = mlx_affine.mlx_affine_matmul(values, weight, autotune=False)
    expected = mx.quantized_matmul(
        values,
        packed,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=64,
        bits=4,
    )
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(("rows", "block_m"), ((65, 64), (129, 128)))
def test_mlx_affine_multi_row_tile_matches_native_ragged_prefill(
    monkeypatch,
    rows,
    block_m,
):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_affine

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_affine._kernel_cache.clear()
    random = np.random.default_rng(59)
    input_features = output_features = 64
    values = mx.array(random.normal(size=(1, rows, input_features)).astype(np.float16))
    dense = mx.array(random.normal(size=(output_features, input_features)).astype(np.float16))
    packed, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    weight = mlx_affine.MLXAffineWeight.from_mlx(packed, scales, biases)
    config = mlx_affine.MLXAffineMatmulConfig("metile", 64, "linear", block_m=block_m)

    actual = mlx_affine._compile_mlx_affine(
        rows,
        input_features,
        output_features,
        values.dtype,
        config,
    )(values, weight)
    expected = mlx_affine._native_affine_matmul(values, weight)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-2, atol=3e-2)


def test_mlx_affine_backend_signature_tracks_candidate_family(monkeypatch):
    from metile.backends import mlx_affine

    original = mlx_affine.mlx_affine_backend_signature()
    monkeypatch.setattr(
        mlx_affine,
        "_CONFIGS",
        (
            *mlx_affine._CONFIGS,
            mlx_affine.MLXAffineMatmulConfig("metile", 256, "linear", block_m=32),
        ),
    )

    assert mlx_affine.mlx_affine_backend_signature() != original


def test_mlx_lm_quantized_block_bypasses_prefill_rows(monkeypatch):
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    calls = []

    def original(self, values, mask=None, cache=None):
        calls.append((self, values, mask, cache))
        return "native"

    monkeypatch.setattr(llama.TransformerBlock, "__call__", original)

    class Model:
        layers = (llama.TransformerBlock.__new__(llama.TransformerBlock),)

        def __call__(self):
            pass

    values = SimpleNamespace(size=128, shape=(1, 2, 64), dtype="mlx.core.float16")
    block = SimpleNamespace(mlp=SimpleNamespace())
    patch = mlx_lm.apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )
    try:
        actual = llama.TransformerBlock.__call__(block, values, "mask", "cache")
    finally:
        patch.restore()

    assert actual == "native"
    assert calls == [(block, values, "mask", "cache")]


def test_mlx_lm_quantized_mlp_fuses_down_projection_with_residual(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    import mlx.nn as nn
    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    gate = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    up = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    down = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    module = SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down)
    calls = []

    def affine_mlp_executor(*args, **kwargs):
        calls.append(("prepare", args, kwargs))

        def execute(values, residual):
            calls.append(("execute", (values, residual), {}))
            return "fused"

        return execute

    _patch_mlx_lm(monkeypatch, "mlx_affine_mlp_executor", affine_mlp_executor)
    _patch_mlx_lm(monkeypatch, "_quantized_mlp_executor_cache", {})

    class Model:
        layers = (llama.TransformerBlock.__new__(llama.TransformerBlock),)

        def __call__(self):
            pass

    class IdentityNorm:
        eps = 1e-5

        def __call__(self, values):
            return values

    values = mx.ones((1, 1, 64), dtype=mx.float16)
    attention = mx.full((1, 1, 64), 2, dtype=mx.float16)
    block = SimpleNamespace(
        input_layernorm=IdentityNorm(),
        post_attention_layernorm=IdentityNorm(),
        self_attn=lambda normalized, mask, cache: attention,
        mlp=module,
    )
    patch = mlx_lm.apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )
    try:
        result = llama.TransformerBlock.__call__(block, values)
        repeated = llama.TransformerBlock.__call__(block, values)
    finally:
        patch.restore()

    assert result == repeated == "fused"
    assert [call[0] for call in calls] == ["prepare", "execute", "execute"]
    np.testing.assert_array_equal(np.array(calls[1][1][1]), np.full((1, 1, 64), 3))


def test_prepare_mlx_lm_affine_prefill_repacks_supported_down_projection():
    pytest.importorskip("mlx_lm")
    from types import SimpleNamespace

    import mlx.nn as nn

    from metile.integrations.mlx_lm import prepare_mlx_lm_affine_prefill

    down_proj = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)

    class Model:
        def __init__(self):
            self.layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=down_proj))]

        def __call__(self):
            pass

    model = Model()
    prepared = prepare_mlx_lm_affine_prefill(model)

    assert prepared.model is model
    assert prepared.projection_count == 1
    assert prepared.weight_for(down_proj).shape == (64, 64)
