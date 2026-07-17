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


def test_framework_dispatch_requires_headroom_over_native():
    native = mlx_backend.MLXAttentionConfig("mlx")
    generated = mlx_backend.MLXAttentionConfig("metile", 256)

    close = mlx_backend._choose_framework_config([(1.0, 0, native), (0.97, 100, generated)])
    faster = mlx_backend._choose_framework_config([(1.0, 0, native), (0.90, 100, generated)])

    assert close == native
    assert faster == generated


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


def test_mlx_lm_patch_restores_modules_loaded_after_application(monkeypatch):
    pytest.importorskip("mlx_lm")
    import mlx.nn as nn
    from mlx_lm.models import base

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    original = base.scaled_dot_product_attention
    original_rms_norm = nn.RMSNorm.__call__
    patch = apply_metile_to_mlx_lm()
    from mlx_lm.models import llama

    assert base.scaled_dot_product_attention is not original
    assert llama.scaled_dot_product_attention is base.scaled_dot_product_attention
    assert nn.RMSNorm.__call__ is not original_rms_norm

    patch.restore()

    assert base.scaled_dot_product_attention is original
    assert llama.scaled_dot_product_attention is original
    assert nn.RMSNorm.__call__ is original_rms_norm
