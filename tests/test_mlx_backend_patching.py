"""MLX backend tests: patching."""

from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends import mlx as mlx_backend
from tests.module_patching import _patch_mlx_lm, _patch_mlx_quantized


def test_mlx_dense_dispatch_requires_three_percent_headroom():
    from metile.backends import mlx_dense_swiglu

    native = mlx_dense_swiglu.MLXDenseSwiGLUConfig("mlx")
    generated = mlx_dense_swiglu.MLXDenseSwiGLUConfig("metile", 64, 64, "grouped8", 2)

    close = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.98, 100, generated)])
    faster = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.90, 100, generated)])

    assert close == native
    assert faster == generated


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
    _patch_mlx_quantized(
        monkeypatch,
        "_affine_swiglu_schedule_cache",
        {
            (1, features, features, "mlx.core.float16", 64, 4): mlx_quantized.MLXAffineSwiGLUConfig(
                "mlx"
            )
        },
    )
    _patch_mlx_quantized(
        monkeypatch,
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


def test_mlx_affine_swiglu_dispatches_report_row_bucket(monkeypatch):
    from metile.backends import mlx_quantized

    config = mlx_quantized.MLXAffineSwiGLUConfig("mlx_compiled")
    _patch_mlx_quantized(
        monkeypatch,
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
    _patch_mlx_quantized(
        monkeypatch,
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


def test_framework_dispatch_requires_headroom_over_native():
    native = mlx_backend.MLXAttentionConfig("mlx")
    generated = mlx_backend.MLXAttentionConfig("metile", 256)

    close = mlx_backend._choose_framework_config([(1.0, 0, native), (0.97, 100, generated)])
    faster = mlx_backend._choose_framework_config([(1.0, 0, native), (0.90, 100, generated)])

    assert close == native
    assert faster == generated


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
    _patch_mlx_lm(monkeypatch, "mlx_affine_matmul", generated)
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

    _patch_mlx_lm(monkeypatch, "mlx_dense_swiglu", generated)
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
