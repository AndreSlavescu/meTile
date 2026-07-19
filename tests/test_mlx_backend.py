import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import metile
from metile.backends import mlx as mlx_backend


def test_mlx_lm_benchmark_helpers_import_without_optional_mlx_packages():
    script = """
import builtins

original_import = builtins.__import__

def import_without_mlx(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm" or name.startswith("mlx_lm."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_mlx
from benchmarks import mlx_lm_backend
assert callable(mlx_lm_backend._confirm_plan)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_mlx_kernel_body_rebinds_metal_thread_attributes():
    source = """
#include <metal_stdlib>
using namespace metal;
[[kernel]] void example(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint tgp_id_x [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]]) {
    Y[tgp_id_x + lid] = X[slid];
}
"""

    body = mlx_backend._mlx_kernel_body(source)

    assert "threadgroup_position_in_grid.x" in body
    assert "thread_index_in_threadgroup" in body
    assert "thread_index_in_simdgroup" in body
    assert "tgp_id_x" not in body
    assert " lid" not in body
    assert "slid" not in body


def test_mlx_kernel_body_accepts_kernel_attribute_lists():
    source = """
[[kernel, max_total_threads_per_threadgroup(32)]] void example(
    device half* X [[buffer(0)]],
    uint3 tgp_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    X[tgp_id.x] = half(sgid);
}
"""

    body = mlx_backend._mlx_kernel_body(source)

    assert "threadgroup_position_in_grid.x" in body
    assert "simdgroup_index_in_threadgroup" in body


def test_mlx_bfloat16_source_specialization_uses_native_metal_types_and_stores():
    source = """
device const half4* X;
device half* Out;
Out[thread_index_in_threadgroup] = float(X[0].x);
*((device half4*)(&Out[4])) = half4(values[0], values[1], values[2], values[3]);
"""

    specialized = mlx_backend._specialize_mlx_source(source, "mlx.core.bfloat16")

    assert "device const bfloat4* X" in specialized
    assert "device bfloat* Out" in specialized
    assert "Out[thread_index_in_threadgroup] = bfloat(float(X[0].x));" in specialized
    assert (
        "bfloat4(bfloat(values[0]), bfloat(values[1]), bfloat(values[2]), bfloat(values[3]))"
        in specialized
    )
    assert "half" not in specialized


def test_mlx_source_specialization_preserves_non_bfloat16_source():
    source = "device half* Out;"

    assert mlx_backend._specialize_mlx_source(source, "mlx.core.float16") == source


def test_mlx_dense_matmul_matches_ragged_bfloat16_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_dense

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense._kernel_cache.clear()
    mlx_dense._schedule_cache.clear()
    random = np.random.default_rng(2030)
    activations = mx.array(random.normal(size=(1, 33, 128)).astype(np.float32)).astype(mx.bfloat16)
    native_weight = mx.array(random.normal(size=(128, 128)).astype(np.float32)).astype(mx.bfloat16)
    weight = mlx_dense.MLXDenseWeight.from_mlx(native_weight)

    actual = mlx_dense.mlx_dense_matmul(activations, weight, autotune=False)
    expected = activations @ native_weight.T
    mx.eval(actual, expected)

    assert actual.shape == expected.shape == (1, 33, 128)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=4e-2,
        atol=4e-2,
    )


def test_mlx_dense_swiglu_matches_bfloat16_reference(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.backends import mlx_dense, mlx_dense_swiglu

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_dense_swiglu._kernel_cache.clear()
    mlx_dense_swiglu._schedule_cache.clear()
    random = np.random.default_rng(2031)
    activations = mx.array(random.normal(size=(1, 33, 64)).astype(np.float32)).astype(mx.bfloat16)
    gate_native = mx.array(random.normal(size=(64, 64)).astype(np.float32)).astype(mx.bfloat16)
    up_native = mx.array(random.normal(size=(64, 64)).astype(np.float32)).astype(mx.bfloat16)
    gate_weight = mlx_dense.MLXDenseWeight.from_mlx(gate_native)
    up_weight = mlx_dense.MLXDenseWeight.from_mlx(up_native)

    actual = mlx_dense_swiglu.mlx_dense_swiglu(
        activations,
        gate_weight,
        up_weight,
        autotune=False,
    )
    expected = nn.silu(activations @ gate_native.T) * (activations @ up_native.T)
    mx.eval(actual, expected)

    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )


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


def test_mlx_dense_dispatch_requires_three_percent_headroom():
    from metile.backends import mlx_dense_swiglu

    native = mlx_dense_swiglu.MLXDenseSwiGLUConfig("mlx")
    generated = mlx_dense_swiglu.MLXDenseSwiGLUConfig("metile", 64, 64, "grouped8", 2)

    close = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.98, 100, generated)])
    faster = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.90, 100, generated)])

    assert close == native
    assert faster == generated


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


def test_mlx_dense_residual_requires_exact_speedup_margin():
    from metile.backends import mlx_dense_residual

    native = mlx_dense_residual.MLXDenseResidualConfig("mlx")
    generated = mlx_dense_residual.MLXDenseResidualConfig("metile", 1, 1)

    close = mlx_dense_residual._choose_config([(1.0, 0, native), (0.99, 100, generated)])
    faster = mlx_dense_residual._choose_config([(1.0, 0, native), (0.98, 100, generated)])

    assert close == native
    assert faster == generated


@pytest.mark.parametrize(
    ("format", "mean_limit", "maximum_limit"),
    (("affine8", 0.1, 0.3), ("mxfp8", 0.8, 2.1)),
)
def test_mlx_compressed_down_residual_matches_quantized_reference(
    format,
    mean_limit,
    maximum_limit,
):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_compressed_down import (
        MLXCompressedDownWeight,
        mlx_compressed_down_residual,
    )

    random = np.random.default_rng(2041)
    values = mx.array(random.normal(size=(1, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    dense = mx.array(random.normal(size=(64, 64)).astype(np.float32)).astype(mx.bfloat16)
    residual = mx.array(random.normal(size=(1, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    weight = MLXCompressedDownWeight.quantize(dense, format=format)

    actual = mlx_compressed_down_residual(values, weight, residual)
    expected = values @ dense.T + residual
    mx.eval(actual, expected)

    assert actual.dtype == mx.bfloat16
    assert weight.nbytes < dense.nbytes
    error = np.abs(np.array(actual.astype(mx.float32)) - np.array(expected.astype(mx.float32)))
    assert float(error.mean()) < mean_limit
    assert float(error.max()) < maximum_limit


@pytest.mark.parametrize("group_size", (32, 64, 128))
def test_mlx_compressed_down_affine_group_sizes(group_size):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_compressed_down import MLXCompressedDownWeight

    dense = mx.ones((64, 128), dtype=mx.bfloat16)
    weight = MLXCompressedDownWeight.quantize(
        dense,
        format="affine8",
        group_size=group_size,
    )

    assert weight.group_size == group_size
    assert weight.shape == (64, 128)


def test_mlx_compressed_down_autotunes_strict_affine_groups(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_compressed_down

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_compressed_down._affine8_group_cache.clear()
    dense = mx.ones((64, 128), dtype=mx.bfloat16)

    group_size, tuning = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert group_size in {32, 64, 128}
    assert tuning["group_size"] == group_size
    assert not tuning["cached"]
    assert tuning["objective"] == "balanced"
    assert set(tuning["median_nanoseconds"]) == {"32", "64", "128"}
    assert all(value > 0 for value in tuning["median_nanoseconds"].values())
    assert tuning["native_median_nanoseconds"] > 0
    assert set(tuning["mean_absolute_error"]) == {"32", "64", "128"}
    assert all(value >= 0 for value in tuning["mean_absolute_error"].values())

    cached_group_size, cached = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert cached_group_size == group_size
    assert cached["cached"]


def test_mlx_compressed_down_autotune_omits_invalid_group128(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_compressed_down

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_compressed_down._affine8_group_cache.clear()
    dense = mx.ones((64, 192), dtype=mx.bfloat16)

    group_size, tuning = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert group_size in {32, 64}
    assert set(tuning["median_nanoseconds"]) == {"32", "64"}
    assert set(tuning["mean_absolute_error"]) == {"32", "64"}
    assert tuning["native_median_nanoseconds"] > 0


def test_mlx_lm_model_group_selection_values_fidelity_coverage():
    from metile.integrations.mlx_lm import _select_model_affine8_group

    selected, estimates = _select_model_affine8_group(
        28,
        {32: 26, 64: 14, 128: 1},
        {32: 435_000, 64: 419_000, 128: 405_000},
        620_000,
    )

    assert selected == 32
    assert estimates[32] < estimates[64] < estimates[128]


def test_mlx_lm_model_group_selection_balances_speed_with_broad_coverage():
    from metile.integrations.mlx_lm import _select_model_affine8_group

    selected, estimates = _select_model_affine8_group(
        36,
        {32: 36, 64: 35, 128: 33},
        {32: 421_000, 64: 398_000, 128: 391_000},
        620_000,
    )

    assert selected == 64
    assert estimates[64] < estimates[32]


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


def test_mlx_block_scaled_dispatch_reports_composable_schedule(monkeypatch):
    from metile.backends import mlx_block_scaled

    config = mlx_block_scaled.MLXBlockScaledConfig(32, 64, "linear", "bfloat", 2)
    monkeypatch.setattr(
        mlx_block_scaled,
        "_schedule_cache",
        {(127, 2048, 2048, "mlx.core.float16", "mxfp8", True): config},
    )

    assert mlx_block_scaled.mlx_block_scaled_dispatches() == (
        {
            "rows": 127,
            "reduction": 2048,
            "output_features": 2048,
            "dtype": "mlx.core.float16",
            "format": "mxfp8",
            "native_available": True,
            "block_m": 32,
            "block_n": 64,
            "schedule": "linear",
            "fragment_type": "bfloat",
            "k_unroll": 2,
            "algorithm": "metile",
        },
    )


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


def test_mlx_affine_mlp_executor_reuses_specialized_dispatch(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    random = np.random.default_rng(55)
    features = 64
    values = mx.array(random.normal(size=(1, 1, features)).astype(np.float16))
    residual = mx.array(random.normal(size=(1, 1, features)).astype(np.float16))

    def quantized_weight():
        dense = mx.array(random.normal(size=(features, features)).astype(np.float16))
        return mx.quantize(dense, group_size=64, bits=4)

    gate = quantized_weight()
    up = quantized_weight()
    down = quantized_weight()
    monkeypatch.setattr(
        mlx_quantized,
        "_affine_swiglu_schedule_cache",
        {
            (1, features, features, "mlx.core.float16", 64, 4): mlx_quantized.MLXAffineSwiGLUConfig(
                "mlx"
            )
        },
    )
    monkeypatch.setattr(
        mlx_quantized,
        "_affine_residual_schedule_cache",
        {
            (features, features, "mlx.core.float16", 64, 4): mlx_quantized.MLXAffineResidualConfig(
                "metile", 64
            )
        },
    )

    executor = mlx_quantized.mlx_affine_mlp_executor(
        values,
        *gate,
        *up,
        *down,
        residual,
    )
    actual = executor(values, residual)
    hidden = mlx_quantized._native_affine_swiglu(values, *gate, *up, 64, 4)
    expected = mlx_quantized._native_affine_residual_qmv(hidden, *down, residual, 64, 4)
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
        (0.99, 1.0, "mlx_compiled"),
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
        (0.99, 1.0, "mlx_compiled"),
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


def test_mlx_affine_swiglu_fidelity_uses_scale_aware_error():
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    reference = mx.array([0.0, 1000.0, -2000.0], dtype=mx.float16)
    reduction_drift = mx.array([0.2, 1001.0, -1998.0], dtype=mx.float16)
    material_error = mx.array([0.0, 1015.0, -1960.0], dtype=mx.float16)

    assert mlx_quantized._affine_swiglu_compatible(reduction_drift, reference)
    assert not mlx_quantized._affine_swiglu_compatible(material_error, reference)


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


def test_mlx_affine_swiglu_dispatches_report_row_bucket(monkeypatch):
    from metile.backends import mlx_quantized

    config = mlx_quantized.MLXAffineSwiGLUConfig("mlx_compiled")
    monkeypatch.setattr(
        mlx_quantized,
        "_affine_swiglu_schedule_cache",
        {(128, 2048, 5632, "mlx.core.float16", 64, 4): config},
    )

    assert mlx_quantized.mlx_affine_swiglu_dispatches() == (
        {
            "row_bucket": 128,
            "input_features": 2048,
            "output_features": 5632,
            "dtype": "mlx.core.float16",
            "group_size": 64,
            "bits": 4,
            "algorithm": "mlx_compiled",
            "decode_dtype": "f32",
            "implementation": "",
            "block": 0,
            "outputs_per_simdgroup": 1,
        },
    )


def test_mlx_affine_residual_dispatches_report_selected_schedule(monkeypatch):
    from metile.backends import mlx_quantized

    config = mlx_quantized.MLXAffineResidualConfig("metile", 256, 2, "f16")
    monkeypatch.setattr(
        mlx_quantized,
        "_affine_residual_schedule_cache",
        {(8192, 3072, "mlx.core.float16", 64, 4): config},
    )

    assert mlx_quantized.mlx_affine_residual_qmv_dispatches() == (
        {
            "input_features": 8192,
            "output_features": 3072,
            "dtype": "mlx.core.float16",
            "group_size": 64,
            "bits": 4,
            "algorithm": "metile",
            "block": 256,
            "outputs_per_simdgroup": 2,
            "decode_dtype": "f16",
        },
    )


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


def test_mlx_affine_dispatch_reports_native_or_composable_schedule(monkeypatch):
    from metile.backends import mlx_affine

    config = mlx_affine.MLXAffineMatmulConfig("metile", 64, "grouped4", block_m=64)
    monkeypatch.setattr(
        mlx_affine,
        "_schedule_cache",
        {(127, 8192, 2048, "mlx.core.float16", 64, 4): config},
    )

    assert mlx_affine.mlx_affine_matmul_dispatches() == (
        {
            "rows": 127,
            "input_features": 8192,
            "output_features": 2048,
            "dtype": "mlx.core.float16",
            "group_size": 64,
            "bits": 4,
            "algorithm": "metile",
            "block_m": 64,
            "block_n": 64,
            "schedule": "grouped4",
        },
    )


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


def test_mlx_lm_plan_key_tracks_affine_backend(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    class Model:
        layers = ()

        def __call__(self, values):
            return values

    tokens = mx.zeros((1, 8), dtype=mx.int32)
    requested = mlx_lm.MLXLMPlan(False, False, False, False, True)
    monkeypatch.setattr(mlx_lm, "mlx_affine_backend_signature", lambda: "backend-a")
    first = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, None, 8, 5)
    monkeypatch.setattr(mlx_lm, "mlx_affine_backend_signature", lambda: "backend-b")
    second = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, None, 8, 5)

    assert first != second


def test_framework_dispatch_requires_headroom_over_native():
    native = mlx_backend.MLXAttentionConfig("mlx")
    generated = mlx_backend.MLXAttentionConfig("metile", 256)

    close = mlx_backend._choose_framework_config([(1.0, 0, native), (0.97, 100, generated)])
    faster = mlx_backend._choose_framework_config([(1.0, 0, native), (0.90, 100, generated)])

    assert close == native
    assert faster == generated


def test_framework_dispatch_accepts_larger_graph_fusion_margin():
    native = mlx_backend.MLXAddRMSNormConfig("mlx")
    generated = mlx_backend.MLXAddRMSNormConfig("metile", 256)

    selected = mlx_backend._choose_framework_config(
        [(1.0, 0, native), (0.92, 100, generated)], margin=0.10
    )

    assert selected == native


def test_mlx_lm_plan_candidates_cover_requested_feature_lattice():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(attention=True, rms_norm=True, graph_fusion=False, quantized_mlp=False)
    )

    assert len(candidates) == 4
    assert MLXLMPlan(False, False, False, False) in candidates
    assert MLXLMPlan(True, True, False, False) in candidates
    assert all(not plan.graph_fusion and not plan.quantized_mlp for plan in candidates)


def test_mlx_lm_plan_timing_reuses_prepared_prompt(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.shape), tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    monkeypatch.setattr(mlx_lm, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    tokens = mx.array([[1, 2, 3]])
    prepared_cache = SimpleNamespace(marker="prepared")
    decode_tokens = (mx.array([[7]]), mx.array([[8]]))

    measurement, next_token = mlx_lm._time_mlx_lm_plan(
        Model(),
        tokens,
        mlx_lm.MLXLMPlan(False, False, False, False),
        None,
        None,
        2,
        prepared_prompt=(prepared_cache, 0.25, decode_tokens),
        decode_tokens=decode_tokens,
    )

    assert next_token == 1
    assert len(calls) == 2
    assert [values for _, values, _ in calls] == [(7,), (8,)]
    assert all(shape == (1, 1) for shape, _, _ in calls)
    assert all(cache is not prepared_cache and cache.marker == "prepared" for _, _, cache in calls)
    assert measurement[0] == 0.25
    assert measurement[2] >= 0.25


def test_mlx_lm_prompt_preparation_builds_autoregressive_trajectory(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import cache as cache_module

    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            values = tuple(tokens.flatten().tolist())
            calls.append((values, cache))
            next_token = values[-1] + 1
            logits = [0.0] * 16
            logits[next_token] = 1.0
            return mx.array([[logits]])

    monkeypatch.setattr(
        cache_module,
        "make_prompt_cache",
        lambda _model: SimpleNamespace(marker="prompt"),
    )

    cache, elapsed, decode_tokens = mlx_lm._prepare_mlx_lm_prompt(Model(), mx.array([[1, 2, 3]]), 3)

    assert cache.marker == "prompt"
    assert elapsed >= 0.0
    assert [tuple(token.flatten().tolist()) for token in decode_tokens] == [(4,), (5,), (6,)]
    assert [values for values, _ in calls] == [(1, 2, 3), (4,), (5,)]
    assert all(call_cache is not cache for _, call_cache in calls[1:])


def test_compressed_calibration_candidate_reuses_native_prompt_cache(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    monkeypatch.setattr(mlx_lm, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    prompt_cache = SimpleNamespace(marker="native-prompt")
    reference = mlx_lm._CompressedCalibrationReference(
        mx.array([[9]]), prompt_cache, 2, object(), object()
    )

    mlx_lm._run_compressed_calibration_candidate(
        Model(),
        mx.array([[1, 2, 3]]),
        reference,
        2,
        mlx_lm.MLXLMPlan(False, False, False, False),
    )

    assert [values for values, _ in calls] == [(9,), (9,)]
    assert all(cache is calls[0][1] for _, cache in calls)
    assert calls[0][1] is not prompt_cache
    assert calls[0][1].marker == "native-prompt"


def test_compressed_calibration_candidate_replays_single_token_prompt(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import cache as cache_module

    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    monkeypatch.setattr(mlx_lm, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    monkeypatch.setattr(
        cache_module,
        "make_prompt_cache",
        lambda _model: SimpleNamespace(marker="fresh"),
    )
    reference = mlx_lm._CompressedCalibrationReference(mx.array([[9]]), None, 1, object(), object())

    mlx_lm._run_compressed_calibration_candidate(
        Model(),
        mx.array([[3]]),
        reference,
        1,
        mlx_lm.MLXLMPlan(False, False, False, False),
    )

    assert [values for values, _ in calls] == [(3,), (9,)]
    assert all(cache is calls[0][1] for _, cache in calls)
    assert calls[0][1].marker == "fresh"


def test_mlx_lm_measurement_shares_prompt_only_with_decode_only_plans(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    structural = mlx_lm.MLXLMPlan(True, False, False, False)
    decode_tokens = (object(), object())
    prepared = (object(), 0.25, decode_tokens)
    seen = []

    monkeypatch.setattr(mlx_lm, "_plan_preserves_logits", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mlx_lm, "_prepare_mlx_lm_prompt", lambda *_args: prepared)

    def time_plan(*arguments, prepared_prompt=None, **_options):
        seen.append((arguments[2], prepared_prompt, _options["decode_tokens"]))
        return (0.25, 0.01, 0.27), 1

    monkeypatch.setattr(mlx_lm, "_time_mlx_lm_plan", time_plan)

    class Model:
        def __call__(self, _tokens):
            return mx.array([[[0.0, 1.0]]])

    mlx_lm._measure_mlx_lm_plans(
        Model(),
        mx.array([[1, 2, 3]]),
        (native, compressed, structural),
        None,
        None,
        2,
        1,
    )

    assert seen == [
        (native, prepared, decode_tokens),
        (compressed, prepared, decode_tokens),
        (structural, None, decode_tokens),
    ]


def test_mlx_lm_measurement_extension_reuses_provisional_trials(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    provisional = {
        native: [(1.0, 1.0, 1.0)] * 3,
        compressed: [(1.0, 0.8, 0.9)] * 3,
    }
    calls = []

    prepared_prompt = object()

    def measure(*arguments, **options):
        calls.append((arguments[2], arguments[6], options))
        return {
            native: [(1.0, 1.0, 1.0)] * arguments[6],
            compressed: [(1.0, 0.8, 0.9)] * arguments[6],
        }

    monkeypatch.setattr(mlx_lm, "_measure_mlx_lm_plans", measure)

    measured = mlx_lm._extend_mlx_lm_measurements(
        object(),
        object(),
        provisional,
        (native, compressed),
        None,
        None,
        16,
        7,
        prepared_prompt=prepared_prompt,
    )

    assert calls == [
        (
            (native, compressed),
            4,
            {"prepared_prompt": prepared_prompt, "validate_fidelity": False},
        )
    ]
    assert all(len(samples) == 7 for samples in measured.values())


def test_mlx_lm_model_search_screens_full_lattice_before_successive_halving(monkeypatch):
    from metile.integrations import mlx_lm

    compression_names = (
        "compressed_down",
        "compressed_gate_up",
        "compressed_vocab",
        "compressed_attention",
    )
    candidates = tuple(
        mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            **{name: bool(mask & (1 << index)) for index, name in enumerate(compression_names)},
        )
        for mask in range(1 << len(compression_names))
    )
    native = candidates[0]
    requested = candidates[-1]
    prepared_prompt = object()
    calls = []

    class Tokens:
        ndim = 2
        shape = (1, 128)

    def measure(
        _model,
        _sample_tokens,
        active,
        _affine_prefill,
        _dense_mlp,
        _decode_steps,
        rounds,
        _compressed_down=None,
        _compressed_gate_up=None,
        _compressed_vocab=None,
        _compressed_attention=None,
        *,
        prepared_prompt=None,
        validate_fidelity=True,
    ):
        calls.append((active, rounds, prepared_prompt, validate_fidelity))
        return {
            plan: [
                (
                    1.0,
                    1.0 - 0.03 * plan.feature_count,
                    1.0 - 0.02 * plan.feature_count,
                )
            ]
            * rounds
            for plan in active
        }

    monkeypatch.setattr(mlx_lm, "_mlx_lm_plan_candidates", lambda _requested: candidates)
    monkeypatch.setattr(mlx_lm, "_mlx_lm_warmup_plans", lambda _candidates: (native,))
    monkeypatch.setattr(mlx_lm, "_effective_mlx_lm_plan", lambda plan, *_args: plan)
    monkeypatch.setattr(mlx_lm, "_mlx_lm_plan_key", lambda *_args: "key")
    monkeypatch.setattr(mlx_lm, "_read_mlx_lm_plan", lambda _key: None)
    monkeypatch.setattr(mlx_lm, "_write_mlx_lm_plan", lambda *_args: None)
    monkeypatch.setattr(
        mlx_lm,
        "_prepare_mlx_lm_prompt",
        lambda *_args: prepared_prompt,
    )
    monkeypatch.setattr(mlx_lm, "_measure_mlx_lm_plans", measure)
    monkeypatch.setattr(
        mlx_lm,
        "_mlx_lm_validation_finalists",
        lambda measured: tuple(measured),
    )
    monkeypatch.setattr(
        mlx_lm,
        "_validate_mlx_lm_finalists_repeated",
        lambda *_args: requested,
    )

    selected = mlx_lm.autotune_metile_for_mlx_lm(
        lambda _tokens: None,
        Tokens(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        decode_steps=8,
        trials=5,
    )

    assert selected == requested
    assert [len(active) for active, *_ in calls] == [1, 16, 8, 8]
    assert [rounds for _, rounds, *_ in calls] == [1, 1, 2, 2]
    assert all(prompt is prepared_prompt for _, _, prompt, _ in calls)
    assert [validate for *_, validate in calls] == [True, True, False, False]


def test_mlx_lm_plan_candidates_split_dense_swiglu_and_residual():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            attention=False,
            rms_norm=False,
            graph_fusion=False,
            quantized_mlp=False,
            dense_mlp=True,
            dense_residual=True,
        )
    )

    assert len(candidates) == 4
    assert (
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=True,
            dense_residual=False,
        )
        in candidates
    )
    assert (
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=False,
            dense_residual=True,
        )
        in candidates
    )


def test_mlx_lm_plan_candidates_exclude_competing_down_rewrites():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            attention=False,
            rms_norm=False,
            graph_fusion=False,
            quantized_mlp=False,
            dense_residual=True,
            compressed_down=True,
        )
    )

    assert len(candidates) == 3
    assert any(plan.dense_residual for plan in candidates)
    assert any(plan.compressed_down for plan in candidates)
    assert all(not (plan.dense_residual and plan.compressed_down) for plan in candidates)


def test_mlx_lm_plan_candidates_compose_compressed_projection_families():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            False,
            False,
            False,
            False,
            compressed_down=True,
            compressed_gate_up=True,
            compressed_vocab=True,
            compressed_attention=True,
        )
    )

    assert len(candidates) == 16
    assert any(
        plan.compressed_down
        and plan.compressed_gate_up
        and plan.compressed_vocab
        and plan.compressed_attention
        for plan in candidates
    )


def test_mlx_lm_plan_candidates_bound_structural_cross_product():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    requested = MLXLMPlan(
        attention=True,
        rms_norm=True,
        graph_fusion=True,
        quantized_mlp=False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
        compressed_attention=True,
    )
    candidates = _mlx_lm_plan_candidates(requested)

    assert len(candidates) == 30
    assert requested in candidates
    assert sum(plan.feature_count == 1 for plan in candidates) == 7


def test_mlx_lm_plan_candidates_exclude_bypassed_gate_up_projection():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=True,
            compressed_gate_up=True,
        )
    )

    assert len(candidates) == 3
    assert all(not (plan.dense_mlp and plan.compressed_gate_up) for plan in candidates)


def test_mlx_lm_warmup_plans_scale_with_primitives_not_plan_lattice():
    from metile.integrations.mlx_lm import (
        MLXLMPlan,
        _mlx_lm_plan_candidates,
        _mlx_lm_warmup_plans,
    )

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(**{name: True for name in MLXLMPlan().as_dict()})
    )
    warmups = _mlx_lm_warmup_plans(candidates)
    enabled = {
        frozenset(name for name, active in plan.as_dict().items() if active) for plan in warmups
    }

    assert len(warmups) <= 10
    assert len(warmups) < len(candidates) // 4
    assert frozenset() in enabled
    assert frozenset(("attention",)) in enabled
    assert frozenset(("graph_fusion", "quantized_mlp")) in enabled
    assert frozenset(("dense_mlp", "dense_residual")) in enabled
    assert all("compressed_vocab" not in features for features in enabled)
    assert all("compressed_attention" not in features for features in enabled)


def test_mlx_lm_compressed_subset_candidates_prefer_largest_simple_regions():
    from metile.integrations.mlx_lm import _compressed_down_subset_candidates

    assert tuple(_compressed_down_subset_candidates(4)) == (
        ("all", (0, 1, 2, 3)),
        ("suffix:3", (1, 2, 3)),
        ("prefix:3", (0, 1, 2)),
        ("suffix:2", (2, 3)),
        ("prefix:2", (0, 1)),
        ("suffix:1", (3,)),
        ("prefix:1", (0,)),
    )


def test_mlx_lm_compressed_region_search_finds_largest_boundary_logarithmically():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        limit = 21 if name.startswith("suffix") else 18
        return len(indices) <= limit, {"name": name}

    name, indices, fidelity = _select_compressed_region(28, evaluate)

    assert name == "suffix:21"
    assert indices == tuple(range(7, 28))
    assert fidelity == {"name": "suffix:21"}
    assert len(calls) <= 2 * 28


def test_mlx_lm_compressed_region_search_audits_nonmonotonic_islands():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        size = len(indices)
        compatible = (name.startswith("suffix") and (size <= 9 or size == 21)) or (
            name.startswith("prefix") and size <= 5
        )
        return compatible, {"name": name}

    name, indices, fidelity = _select_compressed_region(28, evaluate)

    assert name == "suffix:21"
    assert indices == tuple(range(7, 28))
    assert fidelity == {"name": "suffix:21"}
    assert len(calls) <= 2 * 28


def test_mlx_lm_compressed_region_search_short_circuits_full_model():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return True, {"name": name}

    assert _select_compressed_region(4, evaluate) == (
        "all",
        (0, 1, 2, 3),
        {"name": "all"},
    )
    assert len(calls) == 1


def test_mlx_lm_compressed_region_search_augments_noncontiguous_subset():
    from metile.integrations.mlx_lm import _select_compressed_region

    compatible = {
        (4,),
        (3, 4),
        (0, 3, 4),
    }

    def evaluate(name, indices):
        return indices in compatible, {"name": name}

    name, indices, fidelity = _select_compressed_region(5, evaluate)

    assert name == "subset:0,3,4"
    assert indices == (0, 3, 4)
    assert fidelity == {"name": "subset:0,3,4"}


def test_mlx_lm_compressed_region_search_can_preserve_interval_mask():
    from metile.integrations.mlx_lm import _select_compressed_region

    compatible = {
        (4,),
        (3, 4),
        (0, 3, 4),
    }

    def evaluate(name, indices):
        return indices in compatible, {"name": name}

    assert _select_compressed_region(5, evaluate, augmentation_budget=0) == (
        "suffix:2",
        (3, 4),
        {"name": "suffix:2"},
    )


def test_mlx_lm_compressed_region_search_bounds_subset_evaluations():
    from metile.integrations.mlx_lm import _augment_compressed_subset

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return False, None

    selected = ("suffix:2", (62, 63), {"name": "suffix:2"})

    assert _augment_compressed_subset(64, evaluate, selected, budget=7) == selected
    assert len(calls) == 7


def test_mlx_lm_compressed_region_search_bounds_interval_directions():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return False, None

    assert _select_compressed_region(128, evaluate) == ("native", (), None)
    assert len(calls) <= 43


def test_mlx_lm_compressed_region_policy_signature_tracks_budgets(monkeypatch):
    from metile.integrations import mlx_lm

    first = mlx_lm._compressed_region_policy_signature()
    monkeypatch.setattr(
        mlx_lm,
        "_COMPRESSED_INTERVAL_DIRECTION_BUDGET",
        mlx_lm._COMPRESSED_INTERVAL_DIRECTION_BUDGET + 1,
    )

    assert mlx_lm._compressed_region_policy_signature() != first


def test_mlx_lm_compressed_region_full_audit_recovers_late_horizon_island():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        return name == "suffix:35", {"name": name}

    selected = ("suffix:18", tuple(range(18, 36)), {"name": "suffix:18"})
    name, indices, fidelity = _audit_larger_compressed_regions(36, evaluate, selected)

    assert name == "suffix:35"
    assert indices == tuple(range(1, 36))
    assert fidelity == {"name": "suffix:35"}
    assert len(calls) <= 9


def test_mlx_lm_compressed_region_full_audit_checks_opposite_edge_escape():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    def evaluate(name, indices):
        return name == "prefix:23", {"name": name}

    selected = ("suffix:7", tuple(range(17, 24)), {"name": "suffix:7"})

    assert _audit_larger_compressed_regions(24, evaluate, selected) == (
        "prefix:23",
        tuple(range(23)),
        {"name": "prefix:23"},
    )


def test_mlx_lm_compressed_region_full_audit_refines_failed_frontier_locally():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        return name == "suffix:19", {"name": name}

    selected = (
        "subset:3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23",
        (3, *range(6, 24)),
        {"name": "short-horizon"},
    )
    name, indices, fidelity = _audit_larger_compressed_regions(
        24,
        evaluate,
        selected,
        selected_compatible=False,
    )

    assert name == "suffix:19"
    assert indices == tuple(range(5, 24))
    assert fidelity == {"name": "suffix:19"}
    assert len(calls) <= 12


def test_mlx_lm_compressed_calibration_cache_restores_layer_mask(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    weights = {id(module): (module, SimpleNamespace(nbytes=100)) for module in modules}
    prepared = mlx_lm.MLXCompressedDown(object(), dict(weights), "affine8", 300)
    prepared.weights = dict(tuple(weights.items())[1:])
    prepared.repack_bytes = 200
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    monkeypatch.setattr(
        mlx_lm,
        "_compressed_down_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_down_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedDown(object(), dict(weights), "affine8", 300)

    assert mlx_lm._restore_compressed_down_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.projection_count == 2
    assert restored.repack_bytes == 200


def test_mlx_lm_compressed_gate_up_cache_restores_layer_pairs(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    layers = {
        id(module): (
            module,
            object(),
            SimpleNamespace(nbytes=100),
            object(),
            SimpleNamespace(nbytes=100),
        )
        for module in modules
    }
    prepared = mlx_lm.MLXCompressedGateUp(object(), dict(layers), 600)
    prepared.layers = dict(tuple(layers.items())[1:])
    prepared.repack_bytes = 400
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    monkeypatch.setattr(
        mlx_lm,
        "_compressed_gate_up_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_gate_up_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedGateUp(object(), dict(layers), 600)

    assert mlx_lm._restore_compressed_gate_up_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.layer_count == 2
    assert restored.projection_count == 4
    assert restored.repack_bytes == 400


def test_mlx_lm_compressed_attention_cache_restores_layer_groups(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    layers = {
        id(module): (
            module,
            tuple((object(), SimpleNamespace(nbytes=100)) for _ in range(4)),
        )
        for module in modules
    }
    prepared = mlx_lm.MLXCompressedAttention(object(), dict(layers), 1200)
    prepared.layers = dict(tuple(layers.items())[1:])
    prepared.repack_bytes = 800
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    monkeypatch.setattr(
        mlx_lm,
        "_compressed_attention_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_attention_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedAttention(object(), dict(layers), 1200)

    assert mlx_lm._restore_compressed_attention_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.layer_count == 2
    assert restored.projection_count == 8
    assert restored.repack_bytes == 800


def test_mlx_lm_compressed_vocab_cache_restores_rejection(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    prepared = mlx_lm.MLXCompressedVocab(object(), object(), object(), True, 100)
    prepared.weight = None
    prepared.repack_bytes = 0
    prepared.calibrated = True
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 8}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    monkeypatch.setattr(
        mlx_lm,
        "_compressed_vocab_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_vocab_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedVocab(object(), object(), object(), True, 100)

    assert mlx_lm._restore_compressed_vocab_calibration(restored, "key")
    assert restored.calibrated
    assert restored.projection_count == 0
    assert restored.repack_bytes == 0


def test_mlx_lm_effective_plan_splits_dense_swiglu_and_residual(monkeypatch):
    from metile.integrations import mlx_lm

    requested = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=True,
        dense_residual=True,
    )
    prepared = SimpleNamespace(implementation="fused")
    monkeypatch.setattr(
        mlx_lm,
        "mlx_dense_swiglu_dispatches",
        lambda: ({"algorithm": "metile"},),
    )
    monkeypatch.setattr(
        mlx_lm,
        "mlx_dense_residual_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(requested, dense_mlp=prepared) == mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=True,
        dense_residual=False,
    )

    monkeypatch.setattr(
        mlx_lm,
        "mlx_dense_swiglu_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )
    monkeypatch.setattr(
        mlx_lm,
        "mlx_dense_residual_dispatches",
        lambda: ({"algorithm": "metile"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(requested, dense_mlp=prepared) == mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=False,
        dense_residual=True,
    )


def test_mlx_lm_effective_plan_prunes_native_wrappers(monkeypatch):
    from metile.integrations import mlx_lm

    monkeypatch.setattr(mlx_lm, "mlx_attention_dispatches", lambda: ({"algorithm": "metile"},))
    monkeypatch.setattr(mlx_lm, "mlx_rms_norm_dispatches", lambda: ({"algorithm": "mlx"},))
    monkeypatch.setattr(mlx_lm, "mlx_add_rms_norm_dispatches", lambda: ())
    monkeypatch.setattr(mlx_lm, "mlx_affine_swiglu_dispatches", lambda: ({"algorithm": "mlx"},))
    monkeypatch.setattr(
        mlx_lm,
        "mlx_affine_residual_qmv_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(mlx_lm.MLXLMPlan()) == mlx_lm.MLXLMPlan(
        attention=True,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
    )


def test_mlx_lm_effective_plan_keeps_compiled_quantized_mlp(monkeypatch):
    from metile.integrations import mlx_lm

    monkeypatch.setattr(mlx_lm, "mlx_attention_dispatches", lambda: ())
    monkeypatch.setattr(mlx_lm, "mlx_rms_norm_dispatches", lambda: ())
    monkeypatch.setattr(mlx_lm, "mlx_add_rms_norm_dispatches", lambda: ())
    monkeypatch.setattr(
        mlx_lm,
        "mlx_affine_swiglu_dispatches",
        lambda: ({"algorithm": "mlx_compiled"},),
    )
    monkeypatch.setattr(mlx_lm, "mlx_affine_residual_qmv_dispatches", lambda: ())

    assert mlx_lm._effective_mlx_lm_plan(mlx_lm.MLXLMPlan()) == mlx_lm.MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )


def test_mlx_lm_plan_requires_decode_ttft_and_total_headroom():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    safe = MLXLMPlan(attention=True, rms_norm=False, graph_fusion=False, quantized_mlp=False)
    ttft_regression = MLXLMPlan(
        attention=False,
        rms_norm=True,
        graph_fusion=False,
        quantized_mlp=False,
    )
    selected = _choose_mlx_lm_plan(
        {
            native: [(0.100, 0.0100, 0.180)] * 5,
            safe: [(0.099, 0.0095, 0.174)] * 5,
            ttft_regression: [(0.102, 0.0090, 0.172)] * 5,
        }
    )

    assert selected == safe


def test_mlx_lm_plan_falls_back_when_total_win_is_too_small():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    close = MLXLMPlan(attention=True, rms_norm=False, graph_fusion=False, quantized_mlp=False)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                close: [(0.100, 0.0099, 0.179)] * 5,
            }
        )
        == native
    )


def test_mlx_lm_plan_accepts_sustained_decode_win_without_latency_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    dense = MLXLMPlan(False, False, False, False, False, True)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                dense: [(0.100, 0.0098, 0.1795)] * 5,
            }
        )
        == dense
    )


def test_mlx_lm_plan_ranking_retains_simpler_validated_fallback():
    from metile.integrations.mlx_lm import MLXLMPlan, _rank_mlx_lm_plans

    native = MLXLMPlan(False, False, False, False)
    combined = MLXLMPlan(
        attention=True,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_down=True,
    )
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )

    assert _rank_mlx_lm_plans(
        {
            native: [(0.100, 0.0100, 0.180)] * 5,
            combined: [(0.098, 0.0085, 0.155)] * 5,
            compressed: [(0.099, 0.0090, 0.165)] * 5,
        }
    ) == (combined, compressed)


def test_mlx_lm_plan_accepts_strong_decode_with_stable_median_ttft():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    ttft_ratios = (0.980, 1.008, 1.011, 1.012, 1.014, 0.978, 0.994)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * len(ttft_ratios),
                compressed: [(0.100 * ttft_ratio, 0.0090, 0.167) for ttft_ratio in ttft_ratios],
            }
        )
        == compressed
    )


def test_mlx_lm_plan_accepts_decode_only_compression_with_noisy_ttft():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 7,
                compressed: [(0.102, 0.0090, 0.167)] * 7,
            }
        )
        == compressed
    )


def test_mlx_lm_decode_only_compositions_ignore_unrelated_prompt_ttft_noise():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    vocab = MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 7,
                vocab: [(0.050, 0.0085, 0.160)] * 7,
                composite: [(0.140, 0.0060, 0.140)] * 7,
            }
        )
        == composite
    )


def test_mlx_lm_plan_rejects_primitive_rewrite_above_ttft_regression_bound():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    rms_norm = MLXLMPlan(False, True, False, False)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 7,
                rms_norm: [(0.102, 0.0090, 0.167)] * 7,
            }
        )
        == native
    )


def test_mlx_lm_plan_validation_uses_longer_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    dense = mlx_lm.MLXLMPlan(False, False, False, False, False, True)
    calls = []

    def measure(
        _model,
        _tokens,
        candidates,
        _affine_prefill,
        _dense_mlp,
        decode_steps,
        rounds,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
        **_options,
    ):
        calls.append((candidates, decode_steps, rounds))
        return {native: [(1.0, 1.0, 1.0)], dense: [(1.0, 0.98, 0.99)]}

    monkeypatch.setattr(mlx_lm, "_measure_mlx_lm_plans", measure)
    monkeypatch.setattr(mlx_lm, "_choose_mlx_lm_plan", lambda measured: dense)

    assert mlx_lm._validate_mlx_lm_plan(object(), object(), dense, None, None, 8, 5) == dense
    assert calls == [((native, dense), 32, 7)]


def test_mlx_lm_plan_validation_retries_one_noisy_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    results = iter((native, compressed))
    calls = []

    def validate(*arguments):
        calls.append(arguments[2])
        return next(results)

    monkeypatch.setattr(mlx_lm, "_validate_mlx_lm_plan", validate)

    assert (
        mlx_lm._validate_mlx_lm_plan_repeated(
            object(),
            object(),
            compressed,
            None,
            None,
            8,
            5,
        )
        == compressed
    )
    assert calls == [compressed, compressed]


def test_mlx_lm_joint_validation_keeps_decode_and_composition_leaders():
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_gate_up=True,
        compressed_vocab=True,
        compressed_attention=True,
    )
    measured = {
        native: [(0.100, 0.010, 0.180)] * 5,
        vocab: [(0.090, 0.009, 0.165)] * 5,
        composite: [(0.106, 0.006, 0.145)] * 5,
    }

    finalists = mlx_lm._mlx_lm_validation_finalists(measured)

    assert finalists[0] == native
    assert vocab in finalists
    assert composite in finalists


def test_mlx_lm_joint_validation_preserves_compression_ladder():
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    down = mlx_lm.MLXLMPlan(False, False, False, False, compressed_down=True)
    gate = mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    down_gate = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
    )
    down_gate_vocab = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
    )
    measured = {
        native: [(0.100, 0.0100, 0.180)] * 5,
        down: [(0.110, 0.0060, 0.160)] * 5,
        gate: [(0.110, 0.0065, 0.162)] * 5,
        vocab: [(0.090, 0.0080, 0.150)] * 5,
        down_gate: [(0.110, 0.0075, 0.165)] * 5,
        down_gate_vocab: [(0.110, 0.0070, 0.160)] * 5,
    }

    finalists = mlx_lm._mlx_lm_validation_finalists(measured)

    assert down_gate in finalists
    assert down_gate_vocab in finalists


def test_mlx_lm_joint_validation_compares_finalists_on_one_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_gate_up=True,
        compressed_attention=True,
    )
    finalists = (native, vocab, composite)
    calls = []

    def measure(
        _model,
        _tokens,
        candidates,
        _affine_prefill,
        _dense_mlp,
        decode_steps,
        rounds,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
        **_options,
    ):
        calls.append((candidates, decode_steps, rounds))
        return {
            native: [(0.100, 0.010, 0.180)] * 7,
            vocab: [(0.099, 0.009, 0.165)] * 7,
            composite: [(0.100, 0.006, 0.140)] * 7,
        }

    monkeypatch.setattr(mlx_lm, "_measure_mlx_lm_plans", measure)

    selected = mlx_lm._validate_mlx_lm_finalists_repeated(
        object(),
        object(),
        finalists,
        None,
        None,
        8,
        5,
    )

    assert selected == composite
    assert calls == [(finalists, 32, 7)]


def test_mlx_lm_joint_validation_halves_large_holdout_before_full_trials(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    down = mlx_lm.MLXLMPlan(False, False, False, False, compressed_down=True)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        compressed_down=True,
        compressed_vocab=True,
    )
    finalists = (native, down, vocab, composite)
    calls = []

    def measure(*arguments, **options):
        active = arguments[2]
        rounds = arguments[6]
        calls.append((active, rounds, options.get("validate_fidelity", True)))
        decode = {native: 1.0, down: 0.82, vocab: 0.90, composite: 0.72}
        return {plan: [(1.0, decode[plan], 0.5 + decode[plan])] * rounds for plan in active}

    monkeypatch.setattr(mlx_lm, "_measure_mlx_lm_plans", measure)

    selected = mlx_lm._validate_mlx_lm_finalists_repeated(
        object(),
        object(),
        finalists,
        None,
        None,
        8,
        5,
    )

    assert selected == composite
    assert calls[0] == (finalists, 3, True)
    assert len(calls[1][0]) == 3
    assert calls[1][1:] == (4, False)


def test_mlx_lm_plan_accepts_strong_ttft_win_without_total_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    prefill = MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        affine_prefill=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                prefill: [(0.090, 0.0100, 0.1795)] * 5,
            }
        )
        == prefill
    )


def test_mlx_lm_plan_accepts_two_percent_ttft_win_without_total_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    prefill = MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        dense_mlp=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.1800)] * 5,
                prefill: [(0.0975, 0.0100, 0.1795)] * 5,
            }
        )
        == prefill
    )


def test_mlx_lm_provisional_finalists_preserve_fastest_ttft_plan():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False, False)
    total = MLXLMPlan(True, False, False, False, False)
    ttft = MLXLMPlan(False, False, False, False, True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        total: [(0.099, 0.010, 0.160)] * 3,
        ttft: [(0.080, 0.010, 0.190)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, (native, total, ttft))

    assert finalists == (native, total, ttft)


def test_mlx_lm_provisional_finalists_preserve_fastest_decode_plan():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False, False)
    total = MLXLMPlan(True, False, False, False, False)
    decode = MLXLMPlan(False, False, False, False, compressed_down=True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        total: [(0.099, 0.0098, 0.160)] * 3,
        decode: [(0.130, 0.0070, 0.190)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, (native, total, decode))

    assert finalists == (native, total, decode)


def test_mlx_lm_provisional_finalists_preserve_promising_single_feature_fallbacks():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False)
    attention = MLXLMPlan(True, False, False, False)
    compressed = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    combined = MLXLMPlan(True, False, False, False, compressed_gate_up=True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        attention: [(0.130, 0.009, 0.200)] * 3,
        compressed: [(0.095, 0.011, 0.200)] * 3,
        combined: [(0.090, 0.008, 0.150)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(
        provisional,
        (native, attention, compressed, combined),
    )

    assert finalists == (native, attention, compressed, combined)


def test_mlx_lm_provisional_finalists_preserve_compression_ladder():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False)
    down = MLXLMPlan(False, False, False, False, compressed_down=True)
    gate = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    vocab = MLXLMPlan(False, False, False, False, compressed_vocab=True)
    down_gate = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
    )
    down_gate_vocab = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
    )
    candidates = (native, down, gate, vocab, down_gate, down_gate_vocab)
    provisional = {
        native: [(0.100, 0.0100, 0.180)] * 3,
        down: [(0.110, 0.0060, 0.160)] * 3,
        gate: [(0.110, 0.0065, 0.162)] * 3,
        vocab: [(0.090, 0.0080, 0.150)] * 3,
        down_gate: [(0.110, 0.0075, 0.165)] * 3,
        down_gate_vocab: [(0.110, 0.0070, 0.160)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, candidates)

    assert down_gate in finalists
    assert down_gate_vocab in finalists


def test_mlx_attention_decode_matches_mlx_gqa(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_kernel_cache.clear()
    mlx_backend._mlx_schedule_cache.clear()
    batch = 2
    query_heads = 4
    key_value_heads = 2
    tokens = 65
    dimension = 64
    random = np.random.default_rng(2026)
    query = mx.array(random.standard_normal((batch, query_heads, 1, dimension)).astype(np.float32))
    key = mx.array(
        random.standard_normal((batch, key_value_heads, tokens, dimension)).astype(np.float32)
    )
    value = mx.array(
        random.standard_normal((batch, key_value_heads, tokens, dimension)).astype(np.float32)
    )

    actual = mlx_backend.mlx_attention_decode(
        query, key, value, scale=dimension**-0.5, autotune=False
    )
    expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=dimension**-0.5)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-5)


def test_mlx_attention_decode_matches_mlx_bfloat16(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_kernel_cache.clear()
    mlx_backend._mlx_schedule_cache.clear()
    random = np.random.default_rng(2027)
    query = mx.array(random.standard_normal((1, 4, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    key = mx.array(random.standard_normal((1, 2, 65, 64)).astype(np.float32)).astype(mx.bfloat16)
    value = mx.array(random.standard_normal((1, 2, 65, 64)).astype(np.float32)).astype(mx.bfloat16)

    actual = mlx_backend.mlx_attention_decode(query, key, value, scale=0.125, autotune=False)
    expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=0.125)
    mx.eval(actual, expected)

    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=2e-2,
        atol=4e-3,
    )


@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_mlx_rms_norm_matches_mlx_low_precision(dtype_name, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_rms_kernel_cache.clear()
    mlx_backend._mlx_rms_schedule_cache.clear()
    random = np.random.default_rng(17)
    dtype = getattr(mx, dtype_name)
    values = mx.array(random.standard_normal((2, 3, 2048)).astype(np.float32)).astype(dtype)
    weight = mx.array(random.standard_normal((2048,)).astype(np.float32)).astype(dtype)

    actual = mlx_backend.mlx_rms_norm(values, weight, 1e-5, autotune=False)
    expected = mx.fast.rms_norm(values, weight, 1e-5)
    mx.eval(actual, expected)

    tolerance = 4e-2 if dtype_name == "bfloat16" else 2e-3
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_mlx_fused_add_rms_norm_preserves_both_outputs(dtype_name, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_add_rms_kernel_cache.clear()
    mlx_backend._mlx_add_rms_schedule_cache.clear()
    random = np.random.default_rng(29)
    dtype = getattr(mx, dtype_name)
    values = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float32)).astype(dtype)
    residual = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float32)).astype(dtype)
    weight = mx.array(random.standard_normal((2048,)).astype(np.float32)).astype(dtype)

    actual_sum, actual_norm = mlx_backend.mlx_add_rms_norm(
        values, residual, weight, 1e-5, autotune=False
    )
    expected_sum = values + residual
    expected_norm = mx.fast.rms_norm(expected_sum, weight, 1e-5)
    mx.eval(actual_sum, actual_norm, expected_sum, expected_norm)

    np.testing.assert_array_equal(
        np.array(actual_sum.astype(mx.float32)),
        np.array(expected_sum.astype(mx.float32)),
    )
    tolerance = 4e-2 if dtype_name == "bfloat16" else 3e-3
    np.testing.assert_allclose(
        np.array(actual_norm.astype(mx.float32)),
        np.array(expected_norm.astype(mx.float32)),
        rtol=tolerance,
        atol=tolerance,
    )


def test_mlx_graph_executor_runs_selected_fusion(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_graph import compile_mlx_graph
    from metile.ir.graph_ir import GraphBuilder, TensorSpec

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    random = np.random.default_rng(31)
    values = mx.array(random.standard_normal((1, 64)).astype(np.float32))
    residual = mx.array(random.standard_normal((1, 64)).astype(np.float32))
    weight = mx.array(random.standard_normal((64,)).astype(np.float32))
    builder = GraphBuilder()
    spec = TensorSpec((1, 64), "f32")
    values_input = builder.input("values", spec)
    residual_input = builder.input("residual", spec)
    weight_input = builder.input("weight", TensorSpec((64,), "f32"))
    summed = builder.add(values_input, residual_input)
    normalized = builder.rms_norm(summed, weight_input, 1e-5)
    executable = compile_mlx_graph(builder.build((summed, normalized)), autotune=False)

    actual_sum, actual_norm = executable(values, residual, weight)
    expected_sum = values + residual
    expected_norm = mx.fast.rms_norm(expected_sum, weight, 1e-5)
    mx.eval(actual_sum, actual_norm, expected_sum, expected_norm)

    assert executable.plan.regions[0].rule.name == "residual_add_rms_norm"
    np.testing.assert_allclose(np.array(actual_sum), np.array(expected_sum), rtol=0, atol=0)
    np.testing.assert_allclose(np.array(actual_norm), np.array(expected_norm), rtol=2e-5, atol=2e-5)


def test_mlx_graph_executor_runs_swiglu_epilogue_pipeline():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.backends.mlx_graph import compile_mlx_graph
    from metile.ir.graph_ir import GraphBuilder, TensorSpec

    random = np.random.default_rng(53)
    values = mx.array(random.standard_normal((1, 4, 8)).astype(np.float16))
    gate_weight = mx.array(random.standard_normal((1, 8, 16)).astype(np.float16))
    up_weight = mx.array(random.standard_normal((1, 8, 16)).astype(np.float16))
    down_weight = mx.array(random.standard_normal((1, 16, 8)).astype(np.float16))
    builder = GraphBuilder()
    values_input = builder.input("values", TensorSpec((1, 4, 8), "f16"))
    gate_input = builder.input("gate_weight", TensorSpec((1, 8, 16), "f16"))
    up_input = builder.input("up_weight", TensorSpec((1, 8, 16), "f16"))
    down_input = builder.input("down_weight", TensorSpec((1, 16, 8), "f16"))
    gate = builder.matmul(values_input, gate_input, name="gate")
    up = builder.matmul(values_input, up_input, name="up")
    hidden = builder.multiply(builder.silu(gate, name="silu"), up, name="hidden")
    output = builder.matmul(hidden, down_input, name="down")
    executable = compile_mlx_graph(builder.build(output), autotune=False)

    actual = executable(values, gate_weight, up_weight, down_weight)
    expected = mx.matmul(
        nn.silu(mx.matmul(values, gate_weight)) * mx.matmul(values, up_weight),
        down_weight,
    )
    mx.eval(actual, expected)

    assert executable.plan.regions[0].rule.name == "parallel_matmul_swiglu_down"
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-3, atol=1e-3)


def test_mlx_lm_patch_restores_modules_loaded_after_application(monkeypatch):
    pytest.importorskip("mlx_lm")
    import mlx.nn as nn
    from mlx_lm.models import base

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    original = base.scaled_dot_product_attention
    original_rms_norm = nn.RMSNorm.__call__
    patch = apply_metile_to_mlx_lm()
    from mlx_lm.models import llama

    original_mlp = llama.MLP.__call__._metile_original
    assert base.scaled_dot_product_attention is not original
    assert llama.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert nn.RMSNorm.__call__ is not original_rms_norm
    assert llama.MLP.__call__ is not original_mlp

    patch.restore()

    assert base.scaled_dot_product_attention is original
    assert llama.scaled_dot_product_attention is original
    assert nn.RMSNorm.__call__ is original_rms_norm
    assert llama.MLP.__call__ is original_mlp


def test_mlx_lm_patch_restores_graph_fused_transformer_block():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    class Model:
        def __init__(self):
            self.layers = [llama.TransformerBlock.__new__(llama.TransformerBlock)]

        def __call__(self):
            pass

    original = llama.TransformerBlock.__call__
    patch = apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=True,
    )

    assert llama.TransformerBlock.__call__ is not original
    patch.restore()
    assert llama.TransformerBlock.__call__ is original


def test_mlx_lm_patch_supports_qwen2_transformer_blocks():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import qwen2

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    class Model:
        def __init__(self):
            self.layers = [qwen2.TransformerBlock.__new__(qwen2.TransformerBlock)]

        def __call__(self):
            pass

    original = qwen2.TransformerBlock.__call__
    patch = apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )

    assert qwen2.TransformerBlock.__call__ is not original
    patch.restore()
    assert qwen2.TransformerBlock.__call__ is original


def test_mlx_lm_graph_fusion_autotunes_on_first_supported_block(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    calls = []

    class Norm:
        eps = 1e-5

        def __init__(self):
            self.weight = mx.ones((64,), dtype=mx.float16)

        def __getitem__(self, key):
            assert key == "weight"
            return self.weight

        def __call__(self, values):
            return values

    def execute_graph(values, residual, norm):
        calls.append((values, residual, norm))
        summed = values + residual
        return summed, summed

    monkeypatch.setattr(mlx_lm, "mlx_add_rms_norm_selection", lambda *_: None)
    monkeypatch.setattr(mlx_lm, "_execute_residual_rms_graph", execute_graph)

    class Model:
        layers = (llama.TransformerBlock.__new__(llama.TransformerBlock),)

        def __call__(self):
            pass

    values = mx.ones((1, 1, 64), dtype=mx.float16)
    attention = mx.full((1, 1, 64), 2, dtype=mx.float16)
    block = SimpleNamespace(
        input_layernorm=Norm(),
        post_attention_layernorm=Norm(),
        self_attn=lambda normalized, mask, cache: attention,
        mlp=lambda normalized: mx.zeros_like(normalized),
    )
    patch = mlx_lm.apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=True,
        quantized_mlp=False,
    )
    try:
        actual = llama.TransformerBlock.__call__(block, values)
        mx.eval(actual)
    finally:
        patch.restore()

    assert len(calls) == 1
    np.testing.assert_array_equal(np.array(actual), np.full((1, 1, 64), 3))


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


def test_mlx_lm_affine_prefill_patch_is_reversible_and_skips_decode(monkeypatch):
    pytest.importorskip("mlx_lm")
    from types import SimpleNamespace

    import mlx.nn as nn

    from metile.integrations import mlx_lm

    module = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)

    class Model:
        def __call__(self):
            pass

    model = Model()
    weight = object()
    prepared = mlx_lm.MLXAffinePrefill(
        model,
        {id(module): (module, weight)},
        min_rows=32,
    )
    calls = []

    def native(self, values):
        calls.append(("native", values))
        return "native"

    def generated(values, weight):
        calls.append(("metile", values, weight))
        return "metile"

    monkeypatch.setattr(nn.QuantizedLinear, "__call__", native)
    monkeypatch.setattr(mlx_lm, "mlx_affine_matmul", generated)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        affine_prefill=prepared,
    )
    prefill_values = SimpleNamespace(size=33 * 64, shape=(1, 33, 64))
    decode_values = SimpleNamespace(size=64, shape=(1, 1, 64))

    assert module(prefill_values) == "metile"
    assert module(decode_values) == "native"
    assert type(module) is nn.QuantizedLinear
    patch.restore()
    assert module(prefill_values) == "native"
    assert [call[0] for call in calls] == ["metile", "native", "native"]


def test_mlx_lm_quantized_mlp_patch_skips_decode(monkeypatch):
    pytest.importorskip("mlx_lm")
    from types import SimpleNamespace

    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    calls = []

    def original(self, values):
        calls.append((self, values))
        return "native"

    monkeypatch.setattr(llama.MLP, "__call__", original)
    mlp = llama.MLP.__new__(llama.MLP)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    model = Model()
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
        affine_prefill=mlx_lm.MLXAffinePrefill(model, {}),
    )
    values = SimpleNamespace(size=64, shape=(1, 1, 64))

    result = llama.MLP.__call__(mlp, values)

    assert result == "native"
    assert calls == [(mlp, values)]
    assert llama.MLP.__call__ is original
    patch.restore()


def test_mlx_lm_quantized_mlp_patch_keeps_decode_dispatch(monkeypatch):
    pytest.importorskip("mlx_lm")
    from types import SimpleNamespace

    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    calls = []

    def original(self, values):
        calls.append((self, values))
        return "native"

    monkeypatch.setattr(llama.MLP, "__call__", original)
    mlp = llama.MLP.__new__(llama.MLP)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    patch = mlx_lm.apply_metile_to_mlx_lm(
        Model(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )
    values = SimpleNamespace(size=64, shape=(1, 1, 64), dtype="f16")

    result = llama.MLP.__call__(mlp, values)

    assert result == "native"
    assert calls == [(mlp, values)]
    assert llama.MLP.__call__ is not original
    patch.restore()
    assert llama.MLP.__call__ is original


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

    monkeypatch.setattr(mlx_lm, "mlx_affine_mlp_executor", affine_mlp_executor)
    monkeypatch.setattr(mlx_lm, "_quantized_mlp_executor_cache", {})

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


def test_mlx_lm_dense_mlp_fuses_down_projection_with_residual(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    from metile.integrations import mlx_lm

    class DenseBlock:
        def __call__(self, values):
            return values

        def down_proj(self, hidden):
            return hidden

    module = DenseBlock()
    gate_weight = SimpleNamespace(shape=(64, 64))
    up_weight = SimpleNamespace(shape=(64, 64))
    down_weight = mx.ones((64, 64), dtype=mx.bfloat16)

    class Model:
        layers = (llama.TransformerBlock.__new__(llama.TransformerBlock),)

        def __call__(self):
            pass

    model = Model()
    prepared = mlx_lm.MLXDenseMLP(
        model,
        {id(module): (module, gate_weight, up_weight, down_weight)},
        min_rows=1,
        implementation="fused",
    )
    calls = []

    def execute_dense(
        active_module,
        values,
        residual,
        active_prepared,
        use_generated_swiglu,
    ):
        calls.append(
            (
                active_module,
                values,
                residual,
                active_prepared,
                use_generated_swiglu,
            )
        )
        return "fused"

    monkeypatch.setattr(mlx_lm, "_execute_dense_mlp", execute_dense)

    class IdentityNorm:
        eps = 1e-5

        def __call__(self, values):
            return values

    values = mx.ones((1, 1, 64), dtype=mx.bfloat16)
    attention = mx.full((1, 1, 64), 2, dtype=mx.bfloat16)
    block = SimpleNamespace(
        input_layernorm=IdentityNorm(),
        post_attention_layernorm=IdentityNorm(),
        self_attn=lambda normalized, mask, cache: attention,
        mlp=module,
    )
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        dense_mlp=prepared,
        plan=mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_residual=True,
        ),
    )
    try:
        assert type(module) is DenseBlock
        result = llama.TransformerBlock.__call__(block, values)
    finally:
        patch.restore()

    assert result == "fused"
    assert calls[0][0] is module
    assert calls[0][3] is prepared
    assert not calls[0][4]
    np.testing.assert_array_equal(
        np.array(calls[0][2].astype(mx.float32)),
        np.full((1, 1, 64), 3),
    )


def test_mlx_lm_compressed_down_patch_is_decode_only():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    from metile.integrations import mlx_lm

    class DenseProjection:
        def __call__(self, values):
            return values

    projection = DenseProjection()

    class Model:
        def __call__(self):
            pass

    model = Model()
    calls = []

    class CompressedWeight:
        shape = (64, 64)

        def __call__(self, values):
            calls.append(values)
            return "compressed"

    compressed = mlx_lm.MLXCompressedDown(
        model,
        {id(projection): (projection, CompressedWeight())},
        "affine8",
        1024,
    )

    values = mx.ones((1, 1, 64), dtype=mx.bfloat16)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_down=compressed,
        plan=mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            compressed_down=True,
        ),
    )
    try:
        decode = projection(values)
        prefill = projection(mx.ones((1, 2, 64), dtype=mx.bfloat16))
    finally:
        patch.restore()

    assert decode == "compressed"
    assert prefill.shape == (1, 2, 64)
    assert calls == [values]
    assert type(projection) is DenseProjection


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


def test_prepare_mlx_lm_dense_mlp_repacks_supported_bfloat16_blocks():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations.mlx_lm import prepare_mlx_lm_dense_mlp

    gate_proj = nn.Linear(64, 64, bias=False)
    up_proj = nn.Linear(64, 64, bias=False)
    down_proj = nn.Linear(64, 64, bias=False)
    gate_proj.weight = gate_proj.weight.astype(mx.bfloat16)
    up_proj.weight = up_proj.weight.astype(mx.bfloat16)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)

    class Model:
        def __init__(self):
            self.layers = [
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        gate_proj=gate_proj,
                        up_proj=up_proj,
                        down_proj=down_proj,
                    )
                )
            ]

        def __call__(self):
            pass

    model = Model()
    prepared = prepare_mlx_lm_dense_mlp(model)
    gate_weight, up_weight, down_weight = prepared.weights_for(model.layers[0].mlp)
    paired_weight = prepared.paired_weight_for(model.layers[0].mlp)
    mx.eval(gate_weight.k_major, up_weight.k_major, paired_weight)

    assert prepared.model is model
    assert prepared.mlp_count == 1
    assert prepared.repack_bytes == 2 * (gate_proj.weight.nbytes + up_proj.weight.nbytes)
    assert gate_weight.shape == up_weight.shape == (64, 64)
    assert down_weight is down_proj.weight
    np.testing.assert_array_equal(
        np.array(gate_weight.k_major.astype(mx.float32)),
        np.array(gate_proj.weight.T.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(up_weight.k_major.astype(mx.float32)),
        np.array(up_proj.weight.T.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(paired_weight[..., 0].astype(mx.float32)),
        np.array(gate_proj.weight.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(paired_weight[..., 1].astype(mx.float32)),
        np.array(up_proj.weight.astype(mx.float32)),
    )


def test_prepare_mlx_lm_dense_mlp_respects_working_set_budget(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations.mlx_lm import prepare_mlx_lm_dense_mlp

    gate_proj = nn.Linear(64, 64, bias=False)
    up_proj = nn.Linear(64, 64, bias=False)
    down_proj = nn.Linear(64, 64, bias=False)
    gate_proj.weight = gate_proj.weight.astype(mx.bfloat16)
    up_proj.weight = up_proj.weight.astype(mx.bfloat16)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)

    class Model:
        layers = (
            SimpleNamespace(
                mlp=SimpleNamespace(
                    gate_proj=gate_proj,
                    up_proj=up_proj,
                    down_proj=down_proj,
                )
            ),
        )

        def __call__(self):
            pass

    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 10_000},
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 9_000)

    with pytest.raises(ValueError, match=r"exceeding the .* working-set budget"):
        prepare_mlx_lm_dense_mlp(Model())


def test_prepare_mlx_lm_compressed_down_quantizes_supported_projection():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations.mlx_lm import prepare_mlx_lm_compressed_down

    down_proj = nn.Linear(64, 64, bias=False)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)
    mlp = SimpleNamespace(down_proj=down_proj)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    model = Model()
    prepared = prepare_mlx_lm_compressed_down(model, group_size=32)

    assert prepared.model is model
    assert prepared.format == "affine8"
    assert prepared.group_size == 32
    assert prepared.projection_count == 1
    assert prepared.weight_for(down_proj).shape == (64, 64)
    assert prepared.repack_bytes < down_proj.weight.nbytes


def test_prepare_mlx_lm_compressed_down_autotunes_group(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    down_proj = nn.Linear(128, 64, bias=False)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)

    class Model:
        layers = (SimpleNamespace(mlp=SimpleNamespace(down_proj=down_proj)),)

        def __call__(self):
            pass

    model = Model()
    tuning = {
        "cached": False,
        "group_size": 128,
        "median_nanoseconds": {"32": 120, "64": 110, "128": 100},
    }
    monkeypatch.setattr(
        mlx_lm,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (128, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_down(model)

    assert prepared.group_size == 128
    assert prepared.group_tuning == tuning
    assert prepared.weight_for(down_proj).group_size == 128


def test_prepare_mlx_lm_compressed_gate_up_preserves_layer_pairs(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    gate = nn.Linear(128, 64, bias=False)
    up = nn.Linear(128, 64, bias=False)
    gate.weight = gate.weight.astype(mx.bfloat16)
    up.weight = up.weight.astype(mx.bfloat16)
    mlp = SimpleNamespace(gate_proj=gate, up_proj=up)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100, "128": 110},
    }
    monkeypatch.setattr(
        mlx_lm,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_gate_up(Model())

    assert prepared.layer_count == 1
    assert len(prepared.source_layers) == 1
    assert prepared.projection_count == 2
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert prepared.weight_for(gate).shape == gate.weight.shape
    assert prepared.weight_for(up).shape == up.weight.shape
    assert prepared.repack_bytes < gate.weight.nbytes + up.weight.nbytes


def test_mlx_lm_compressed_gate_up_patch_is_reversible_and_decode_only(monkeypatch):
    from metile.integrations import mlx_lm

    calls = []

    class Linear:
        def __call__(self, values):
            calls.append(("native", self, values))
            return "native"

    gate = Linear()
    up = Linear()

    gate_weight = SimpleNamespace(values="gate", scales="gate-scale", biases="gate-bias")
    up_weight = SimpleNamespace(values="up", scales="up-scale", biases="up-bias")

    class MLP:
        gate_proj = gate
        up_proj = up

        def __call__(self, values):
            calls.append(("native-mlp", values))
            return "native-mlp"

        def down_proj(self, hidden):
            calls.append(("down", hidden))
            return "projected", hidden

    mlp = MLP()

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(mlp): (mlp, gate, gate_weight, up, up_weight)},
        200,
        calibrated=True,
        implementation="fused",
    )
    monkeypatch.setattr(mlx_lm, "_supports_compressed_gate_up_fusion", lambda _module: True)

    def fused(values, *weights, **options):
        calls.append(("fused", values, weights, options))
        return "fused-hidden"

    monkeypatch.setattr(
        mlx_lm,
        "mlx_affine_swiglu_executor",
        lambda *args, **options: lambda values: fused(values, *args[1:], **options),
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_gate_up=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64), dtype="bf16")
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64), dtype="bf16")

    assert mlp(decode) == ("projected", "fused-hidden")
    assert mlp(prefill) == "native-mlp"
    assert type(gate) is Linear
    assert type(up) is Linear

    patch.restore()

    assert type(mlp) is MLP
    assert mlp(decode) == "native-mlp"
    fused_call = next(call for call in calls if call[0] == "fused")
    assert fused_call[2] == (
        "gate",
        "gate-scale",
        "gate-bias",
        "up",
        "up-scale",
        "up-bias",
    )
    assert fused_call[3] == {"group_size": 64, "bits": 8}


def test_mlx_lm_compressed_gate_up_falls_back_to_projection_patches(monkeypatch):
    from metile.integrations import mlx_lm

    class Linear:
        def __call__(self, _values):
            return "native"

    gate = Linear()
    up = Linear()

    def gate_weight(values):
        return "compressed-gate", values

    def up_weight(values):
        return "compressed-up", values

    mlp = SimpleNamespace(gate_proj=gate, up_proj=up)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(mlp): (mlp, gate, gate_weight, up, up_weight)},
        200,
        calibrated=True,
    )
    monkeypatch.setattr(mlx_lm, "_supports_compressed_gate_up_fusion", lambda _module: False)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_gate_up=prepared,
        plan=mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True),
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))

    assert gate(decode) == ("compressed-gate", decode)
    assert up(decode) == ("compressed-up", decode)
    patch.restore()
    assert type(gate) is Linear
    assert type(up) is Linear


def test_mlx_lm_compressed_gate_up_selects_faster_fused_model_path(monkeypatch):
    from metile.integrations import mlx_lm

    class Model:
        def __call__(self):
            pass

    module = object()
    weight = SimpleNamespace(shape=(64, 64), group_size=64)
    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(module): (module, object(), weight, object(), weight)},
        200,
    )
    reference = SimpleNamespace(full_reference=object())
    fidelity = {
        "next_token": 7,
        "actual_next_token": 7,
        "kl_divergence": 0.0,
        "mean_logit_error": 0.0,
        "max_logit_error": 0.0,
    }
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    monkeypatch.setattr(mlx_lm, "_supports_compressed_gate_up_fusion", lambda _module: True)
    monkeypatch.setattr(mlx_lm, "_compressed_gate_up_implementation_key", lambda *_args: "key")
    monkeypatch.setattr(
        mlx_lm,
        "_prepare_compressed_calibration_reference",
        lambda *_args: reference,
    )
    monkeypatch.setattr(
        mlx_lm,
        "_run_compressed_calibration_candidate",
        lambda *_args, **_options: object(),
    )
    monkeypatch.setattr(mlx_lm, "_logit_fidelity", lambda *_args: fidelity)
    monkeypatch.setattr(
        mlx_lm,
        "_prepare_mlx_lm_prompt",
        lambda *_args: (object(), 0.1, (object(), object())),
    )

    def time_plan(*_args, compressed_gate_up=None, **_options):
        decode = 0.8 if compressed_gate_up.implementation == "fused" else 1.0
        return (0.1, decode, 0.1 + decode), 7

    monkeypatch.setattr(mlx_lm, "_time_mlx_lm_plan", time_plan)

    mlx_lm._select_compressed_gate_up_implementation(
        prepared.model,
        SimpleNamespace(),
        prepared,
        2,
        3,
    )

    assert prepared.implementation == "fused"
    assert prepared.implementation_tuning["reason"] == "timing"
    assert prepared.implementation_tuning["median_nanoseconds"] == {
        "projected": 1_000_000_000,
        "fused": 800_000_000,
    }


def test_mlx_lm_compressed_gate_up_rejects_inexact_fusion(monkeypatch):
    from metile.integrations import mlx_lm

    class Model:
        def __call__(self):
            pass

    module = object()
    weight = SimpleNamespace(shape=(64, 64), group_size=64)
    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(module): (module, object(), weight, object(), weight)},
        200,
    )
    reference = SimpleNamespace(full_reference=object())
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    monkeypatch.setattr(mlx_lm, "_supports_compressed_gate_up_fusion", lambda _module: True)
    monkeypatch.setattr(mlx_lm, "_compressed_gate_up_implementation_key", lambda *_args: "key")
    monkeypatch.setattr(
        mlx_lm,
        "_prepare_compressed_calibration_reference",
        lambda *_args: reference,
    )
    monkeypatch.setattr(
        mlx_lm,
        "_run_compressed_calibration_candidate",
        lambda *_args, **_options: object(),
    )
    monkeypatch.setattr(
        mlx_lm,
        "_logit_fidelity",
        lambda *_args: {
            "next_token": 7,
            "actual_next_token": 8,
            "kl_divergence": 0.0,
            "mean_logit_error": 0.0,
            "max_logit_error": 0.0,
        },
    )
    monkeypatch.setattr(
        mlx_lm,
        "_time_mlx_lm_plan",
        lambda *_args, **_options: pytest.fail("inexact fusion must not be timed"),
    )

    mlx_lm._select_compressed_gate_up_implementation(
        prepared.model,
        SimpleNamespace(),
        prepared,
        2,
        3,
    )

    assert prepared.implementation == "projected"
    assert prepared.implementation_tuning["reason"] == "fidelity"


def test_prepare_mlx_lm_compressed_attention_preserves_layer_groups(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    projections = tuple(nn.Linear(128, 64, bias=index == 0) for index in range(4))
    for projection in projections:
        projection.weight = projection.weight.astype(mx.bfloat16)
        if "bias" in projection:
            projection.bias = projection.bias.astype(mx.bfloat16)
    attention = SimpleNamespace(
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        o_proj=projections[3],
    )

    class Model:
        layers = (SimpleNamespace(self_attn=attention),)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100, "128": 110},
    }
    monkeypatch.setattr(
        mlx_lm,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_attention(Model())

    assert prepared.layer_count == 1
    assert prepared.projection_count == 4
    assert len(prepared.source_layers) == 1
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert all(
        prepared.weight_for(projection).shape == projection.weight.shape
        for projection in projections
    )
    assert prepared.repack_bytes < sum(projection.weight.nbytes for projection in projections)


def test_mlx_lm_compressed_attention_patch_is_reversible_decode_only_and_biased():
    from metile.integrations import mlx_lm

    calls = []

    class Linear(dict):
        def __call__(self, values):
            calls.append(("native", self, values))
            return "native"

    projections = tuple(Linear() for _ in range(4))
    projections[0]["bias"] = 2
    attention = SimpleNamespace(
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        o_proj=projections[3],
    )
    weights = tuple(lambda _values, result=result: result for result in (5, 6, 7, 8))

    class Model:
        layers = (SimpleNamespace(self_attn=attention),)

        def __call__(self):
            pass

    model = Model()
    prepared = mlx_lm.MLXCompressedAttention(
        model,
        {id(attention): (attention, tuple(zip(projections, weights)))},
        400,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_attention=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_attention=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64))

    assert projections[0](decode) == 7
    assert projections[1](decode) == 6
    assert projections[0](prefill) == "native"

    patch.restore()

    assert all(type(projection) is Linear for projection in projections)
    assert projections[0](decode) == "native"


def test_prepare_mlx_lm_compressed_vocab_supports_tied_embedding(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    embedding = nn.Embedding(128, 64)
    embedding.weight = embedding.weight.astype(mx.bfloat16)

    class Model:
        args = SimpleNamespace(tie_word_embeddings=True)
        model = SimpleNamespace(embed_tokens=embedding)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100},
    }
    monkeypatch.setattr(
        mlx_lm,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    model = Model()
    prepared = mlx_lm.prepare_mlx_lm_compressed_vocab(model)

    assert prepared.model is model
    assert prepared.module is embedding
    assert prepared.tied
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert prepared.projection_count == 1
    assert prepared.weight.shape == embedding.weight.shape
    assert prepared.repack_bytes < embedding.weight.nbytes


def test_mlx_lm_compressed_vocab_tied_patch_is_reversible_and_decode_only():
    from metile.integrations import mlx_lm

    calls = []

    class Embedding:
        def as_linear(self, values):
            calls.append(("native", values))
            return "native"

    module = Embedding()

    def weight(values):
        return "compressed", values

    class Model:
        def __call__(self):
            pass

    model = Model()
    prepared = mlx_lm.MLXCompressedVocab(
        model,
        module,
        weight,
        True,
        100,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_vocab=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64))

    assert module.as_linear(decode) == ("compressed", decode)
    assert module.as_linear(prefill) == "native"

    patch.restore()

    assert type(module) is Embedding
    assert module.as_linear(decode) == "native"


def test_mlx_lm_compressed_vocab_untied_patch_uses_linear_call():
    from metile.integrations import mlx_lm

    class Linear:
        def __call__(self, _values):
            return "native"

    module = Linear()

    class Model:
        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedVocab(
        Model(),
        module,
        lambda values: ("compressed", values),
        False,
        100,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    decode = SimpleNamespace(size=64, shape=(1, 64))

    with mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_vocab=prepared,
        plan=plan,
    ):
        assert module(decode) == ("compressed", decode)

    assert module(decode) == "native"


def test_prepare_mlx_lm_mxfp8_down_requires_explicit_approximation():
    pytest.importorskip("mlx.core")

    from metile.integrations.mlx_lm import prepare_mlx_lm_compressed_down

    with pytest.raises(ValueError, match="allow_approximate"):
        prepare_mlx_lm_compressed_down(lambda: None, format="mxfp8")


def test_mlx_lm_compressed_down_fidelity_policy_distinguishes_approximation():
    from metile.integrations.mlx_lm import MLXCompressedDown

    fidelity = {
        "next_token": 7,
        "actual_next_token": 7,
        "kl_divergence": 0.02,
        "mean_logit_error": 0.2,
        "max_logit_error": 1.0,
    }
    strict = MLXCompressedDown(object(), {}, "affine8", 0)
    approximate = MLXCompressedDown(object(), {}, "mxfp8", 0, True)

    assert not strict.fidelity_compatible(fidelity)
    assert approximate.fidelity_compatible(fidelity)
    assert not approximate.fidelity_compatible({**fidelity, "actual_next_token": 8})


def test_mlx_lm_dense_mlp_patch_is_reversible_and_skips_decode(monkeypatch):
    pytest.importorskip("mlx_lm")
    from metile.integrations import mlx_lm

    calls = []

    class DenseBlock:
        def __call__(self, values):
            calls.append(("native", values))
            return "native"

        def down_proj(self, hidden):
            calls.append(("down", hidden))
            return "generated"

    module = DenseBlock()

    class Model:
        layers = (SimpleNamespace(mlp=module),)

        def __call__(self):
            pass

    model = Model()
    gate_weight = object()
    up_weight = object()
    down_weight = object()
    prepared = mlx_lm.MLXDenseMLP(
        model,
        {id(module): (module, gate_weight, up_weight, down_weight)},
        min_rows=32,
        implementation="fused",
    )

    def generated(values, gate, up):
        calls.append(("metile", values, gate, up))
        return "hidden"

    monkeypatch.setattr(mlx_lm, "mlx_dense_swiglu", generated)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        dense_mlp=prepared,
        plan=mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=True,
        ),
    )
    prefill_values = SimpleNamespace(size=33 * 64, shape=(1, 33, 64))
    decode_values = SimpleNamespace(size=64, shape=(1, 1, 64))

    assert module(prefill_values) == "generated"
    assert module(decode_values) == "native"
    assert type(module) is DenseBlock
    patch.restore()
    assert module(prefill_values) == "native"
    assert [call[0] for call in calls] == ["metile", "down", "native", "native"]


def test_mlx_lm_bfloat16_fidelity_uses_precision_aware_limits():
    from metile.integrations import mlx_lm

    fidelity = {
        "reference_dtype": "mlx.core.bfloat16",
        "next_token": 42,
        "actual_next_token": 42,
        "kl_divergence": 7e-4,
        "mean_logit_error": 0.03,
        "max_logit_error": 0.4,
    }

    assert mlx_lm._fidelity_compatible(fidelity)
    assert not mlx_lm._fidelity_compatible({**fidelity, "reference_dtype": "mlx.core.float16"})


def test_mlx_lm_dense_selector_rejects_inexact_fusion(monkeypatch):
    from metile.integrations import mlx_lm

    dense_mlp = SimpleNamespace(implementation="projected")
    exact = {
        "reference_dtype": "mlx.core.bfloat16",
        "next_token": 42,
        "actual_next_token": 42,
        "kl_divergence": 0.0,
        "mean_logit_error": 0.0,
        "max_logit_error": 0.0,
    }

    def fidelity(_model, _tokens, _dense_mlp, implementation):
        if implementation == "projected":
            return exact
        return {
            **exact,
            "kl_divergence": 5e-4,
            "mean_logit_error": 0.01,
            "max_logit_error": 0.1,
        }

    monkeypatch.setattr(mlx_lm, "_cache_aware_dense_fidelity", fidelity)
    monkeypatch.setattr(
        mlx_lm,
        "_time_dense_mlp_implementation",
        lambda _model, _tokens, _dense_mlp, implementation: (
            0.8 if implementation == "fused" else 1.0
        ),
    )

    mlx_lm._select_dense_mlp_implementation(object(), object(), dense_mlp, 3)

    assert dense_mlp.implementation == "projected"


def test_mlx_lm_dense_selector_chooses_faster_exact_fusion(monkeypatch):
    from metile.integrations import mlx_lm

    dense_mlp = SimpleNamespace(implementation="projected")
    exact = {
        "reference_dtype": "mlx.core.bfloat16",
        "next_token": 42,
        "actual_next_token": 42,
        "kl_divergence": 0.0,
        "mean_logit_error": 0.0,
        "max_logit_error": 0.0,
    }

    monkeypatch.setattr(
        mlx_lm,
        "_cache_aware_dense_fidelity",
        lambda _model, _tokens, _dense_mlp, _implementation: exact,
    )
    monkeypatch.setattr(
        mlx_lm,
        "_time_dense_mlp_implementation",
        lambda _model, _tokens, _dense_mlp, implementation: (
            0.8 if implementation == "fused" else 1.0
        ),
    )

    mlx_lm._select_dense_mlp_implementation(object(), object(), dense_mlp, 3)

    assert dense_mlp.implementation == "fused"


@pytest.mark.parametrize(
    ("generated_elapsed", "expected_accepted"),
    ((0.95, True), (1.05, False)),
)
def test_mlx_lm_generation_confirmation_requires_end_to_end_safety(
    generated_elapsed,
    expected_accepted,
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=3, delay=0)
    candidate = MLXLMPlan(False, False, False, False, True)

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(generation_tps=100.0, prompt_tps=1000.0)
        return response, generated_elapsed if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(),
        object(),
        [1, 2, 3],
        arguments,
        candidate,
        None,
        None,
    )

    assert confirmation["accepted"] is expected_accepted
    assert bool(selected.feature_count) is expected_accepted


@pytest.mark.parametrize(
    ("candidate", "expected_accepted"),
    (
        ("prefill", True),
        ("decode", False),
    ),
)
def test_mlx_lm_generation_confirmation_uses_prefill_noise_floor(
    candidate,
    expected_accepted,
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=3, delay=0)
    plan = (
        MLXLMPlan(False, False, False, False, True)
        if candidate == "prefill"
        else MLXLMPlan(True, False, False, False, False)
    )

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=99.2 if patched else 100.0,
            prompt_tps=1100.0 if patched else 1000.0,
        )
        return response, 0.98 if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert confirmation["accepted"] is expected_accepted
    assert bool(selected.feature_count) is expected_accepted


def test_mlx_lm_generation_confirmation_accepts_sustained_decode_win(monkeypatch):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=3, delay=0)
    plan = MLXLMPlan(False, False, False, False, False, True)

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=101.1 if patched else 100.0,
            prompt_tps=1000.0,
        )
        return response, 0.997 if patched else 1.0, 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert confirmation["accepted"]
    assert selected == plan


def test_mlx_lm_generation_confirmation_rejects_unstable_ttft(monkeypatch):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=5, delay=0)
    plan = MLXLMPlan(False, False, False, False, False, True)
    generated_ttft = iter((0.1, 0.1, 0.1, 0.2, 0.2))

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=102.0 if patched else 100.0,
            prompt_tps=1000.0,
        )
        ttft = next(generated_ttft) if patched else 0.1
        return response, 0.98 if patched else 1.0, ttft

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert not confirmation["accepted"]
    assert not selected.feature_count


def test_mlx_lm_generation_confirmation_accepts_strong_decode_with_no_total_losses(
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=5, delay=0)
    plan = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    generated_ttft = iter((0.08, 0.11, 0.09, 0.105, 0.095))

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=110.0 if patched else 100.0,
            prompt_tps=1000.0,
        )
        ttft = next(generated_ttft) if patched else 0.1
        return response, 0.9 if patched else 1.0, ttft

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert confirmation["accepted"]
    assert selected == plan


def test_mlx_lm_generation_confirmation_ignores_prompt_ttft_for_decode_only_compression(
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=5, delay=0)
    plan = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_vocab=True,
    )

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=125.0 if patched else 100.0,
            prompt_tps=1000.0,
        )
        return response, 0.82 if patched else 1.0, 0.11 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert confirmation["accepted"]
    assert confirmation["decode_only_compression"]
    assert confirmation["medians"]["ttft_speedup"] < mlx_lm_backend._TTFT_CONFIRMATION_FLOOR
    assert selected == plan


def test_mlx_lm_graph_fusion_deoptimizes_to_original_block(monkeypatch):
    pytest.importorskip("mlx_lm")
    from types import SimpleNamespace

    from mlx_lm.models import llama

    from metile.backends.mlx import MLXAddRMSNormConfig
    from metile.integrations import mlx_lm

    calls = []

    def original(self, values, mask=None, cache=None):
        calls.append((self, values, mask, cache))
        return "native"

    monkeypatch.setattr(llama.TransformerBlock, "__call__", original)
    monkeypatch.setattr(
        mlx_lm,
        "mlx_add_rms_norm_selection",
        lambda *_: MLXAddRMSNormConfig("mlx"),
    )

    class Model:
        def __init__(self):
            self.layers = [llama.TransformerBlock.__new__(llama.TransformerBlock)]

        def __call__(self):
            pass

    patch = mlx_lm.apply_metile_to_mlx_lm(
        Model(), attention=False, rms_norm=False, graph_fusion=True
    )
    fake_block = SimpleNamespace(post_attention_layernorm=SimpleNamespace(eps=1e-5))
    result = llama.TransformerBlock.__call__(fake_block, "values", "mask", "cache")

    assert result == "native"
    assert calls == [(fake_block, "values", "mask", "cache")]
    patch.restore()
