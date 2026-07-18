import numpy as np
import pytest

from metile.backends import mlx as mlx_backend


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
