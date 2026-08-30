"""MLX backend tests: graph fusion."""

from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends import mlx as mlx_backend
from tests.module_patching import _patch_mlx_lm


def test_framework_dispatch_accepts_larger_graph_fusion_margin():
    native = mlx_backend.MLXAddRMSNormConfig("mlx")
    generated = mlx_backend.MLXAddRMSNormConfig("metile", 256)

    selected = mlx_backend._choose_framework_config(
        [(1.0, 0, native), (0.92, 100, generated)], margin=0.10
    )

    assert selected == native


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

    _patch_mlx_lm(monkeypatch, "mlx_add_rms_norm_selection", lambda *_: None)
    _patch_mlx_lm(monkeypatch, "_execute_residual_rms_graph", execute_graph)

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

    _patch_mlx_lm(monkeypatch, "_cache_aware_dense_fidelity", fidelity)
    _patch_mlx_lm(
        monkeypatch,
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

    _patch_mlx_lm(
        monkeypatch,
        "_cache_aware_dense_fidelity",
        lambda _model, _tokens, _dense_mlp, _implementation: exact,
    )
    _patch_mlx_lm(
        monkeypatch,
        "_time_dense_mlp_implementation",
        lambda _model, _tokens, _dense_mlp, implementation: (
            0.8 if implementation == "fused" else 1.0
        ),
    )

    mlx_lm._select_dense_mlp_implementation(object(), object(), dense_mlp, 3)

    assert dense_mlp.implementation == "fused"


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
    _patch_mlx_lm(
        monkeypatch,
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
