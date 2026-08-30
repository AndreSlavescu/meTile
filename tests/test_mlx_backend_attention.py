"""MLX backend tests: attention."""

from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends import mlx as mlx_backend


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


def test_mlx_lm_generation_confirmation_accepts_sustained_decode_win(monkeypatch):

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


def test_mlx_lm_generation_confirmation_accepts_strong_decode_with_no_total_losses(
    monkeypatch,
):

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
