import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("rows", (33, 127))
def test_mlx_block_scaled_matmul_matches_ragged_fp16_reference(rows, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_block_scaled

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_block_scaled._kernel_cache.clear()
    mlx_block_scaled._schedule_cache.clear()
    random = np.random.default_rng(52)
    activations = random.normal(size=(1, rows, 64)).astype(np.float16)
    dense_weight = random.normal(size=(64, 64)).astype(np.float32)
    weight = mlx_block_scaled.MLXBlockScaledWeight.quantize(
        dense_weight,
        format="mxfp8",
    )
    reference_weight = metile.BlockScaledWeight.quantize(
        dense_weight,
        format="mxfp8",
    ).dequantize()

    actual = mlx_block_scaled.mlx_block_scaled_matmul(
        mx.array(activations),
        weight,
        autotune=False,
    )
    mx.eval(actual)

    expected = activations.astype(np.float32) @ reference_weight
    assert actual.shape == (1, rows, 64)
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(np.array(actual), expected, rtol=5e-2, atol=2e-1)


def test_mlx_block_scaled_dispatch_reports_composable_schedule(monkeypatch):
    from metile.backends import mlx_block_scaled

    config = mlx_block_scaled.MLXBlockScaledConfig(32, 64, "linear", "bfloat", 2)
    monkeypatch.setattr(
        mlx_block_scaled,
        "_schedule_cache",
        {(127, 2048, 2048, "mlx.core.float16", "mxfp8"): config},
    )

    assert mlx_block_scaled.mlx_block_scaled_dispatches() == (
        {
            "rows": 127,
            "reduction": 2048,
            "output_features": 2048,
            "dtype": "mlx.core.float16",
            "format": "mxfp8",
            "block_m": 32,
            "block_n": 64,
            "schedule": "linear",
            "fragment_type": "bfloat",
            "k_unroll": 2,
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


def test_mlx_affine_dispatch_reports_native_or_composable_schedule(monkeypatch):
    from metile.backends import mlx_affine

    config = mlx_affine.MLXAffineMatmulConfig("metile", 64, "grouped4")
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
        (*mlx_affine._CONFIGS, mlx_affine.MLXAffineMatmulConfig("metile", 256, "linear")),
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
    first = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, 8, 5)
    monkeypatch.setattr(mlx_lm, "mlx_affine_backend_signature", lambda: "backend-b")
    second = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, 8, 5)

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


def test_mlx_lm_effective_plan_prunes_native_wrappers(monkeypatch):
    from metile.integrations import mlx_lm

    monkeypatch.setattr(mlx_lm, "mlx_attention_dispatches", lambda: ({"algorithm": "metile"},))
    monkeypatch.setattr(mlx_lm, "mlx_rms_norm_dispatches", lambda: ({"algorithm": "mlx"},))
    monkeypatch.setattr(mlx_lm, "mlx_add_rms_norm_dispatches", lambda: ())
    monkeypatch.setattr(mlx_lm, "mlx_affine_swiglu_dispatches", lambda: ({"algorithm": "mlx"},))

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


def test_mlx_rms_norm_matches_mlx_float16(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_rms_kernel_cache.clear()
    mlx_backend._mlx_rms_schedule_cache.clear()
    random = np.random.default_rng(17)
    values = mx.array(random.standard_normal((2, 3, 2048)).astype(np.float16))
    weight = mx.array(random.standard_normal((2048,)).astype(np.float16))

    actual = mlx_backend.mlx_rms_norm(values, weight, 1e-5, autotune=False)
    expected = mx.fast.rms_norm(values, weight, 1e-5)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-3, atol=2e-3)


def test_mlx_fused_add_rms_norm_preserves_both_outputs(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_add_rms_kernel_cache.clear()
    mlx_backend._mlx_add_rms_schedule_cache.clear()
    random = np.random.default_rng(29)
    values = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float16))
    residual = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float16))
    weight = mx.array(random.standard_normal((2048,)).astype(np.float16))

    actual_sum, actual_norm = mlx_backend.mlx_add_rms_norm(
        values, residual, weight, 1e-5, autotune=False
    )
    expected_sum = values + residual
    expected_norm = mx.fast.rms_norm(expected_sum, weight, 1e-5)
    mx.eval(actual_sum, actual_norm, expected_sum, expected_norm)

    np.testing.assert_array_equal(np.array(actual_sum), np.array(expected_sum))
    np.testing.assert_allclose(np.array(actual_norm), np.array(expected_norm), rtol=3e-3, atol=3e-3)


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

    def generate(_model, _tokenizer, _prompt, _arguments, patched, _plan, _affine_prefill):
        response = SimpleNamespace(generation_tps=100.0)
        return response, generated_elapsed if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(),
        object(),
        [1, 2, 3],
        arguments,
        candidate,
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

    def generate(_model, _tokenizer, _prompt, _arguments, patched, _plan, _affine_prefill):
        response = SimpleNamespace(generation_tps=99.2 if patched else 100.0)
        return response, 0.98 if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None
    )

    assert confirmation["accepted"] is expected_accepted
    assert bool(selected.feature_count) is expected_accepted


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
