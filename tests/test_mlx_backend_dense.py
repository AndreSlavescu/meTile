"""MLX backend tests: dense."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.module_patching import _patch_mlx_lm


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


@pytest.mark.parametrize("shape", ((1, 6, 256), (2, 3, 256), (1, 1, 256)))
def test_multi_row_dense_tuning_handles_rank_three_activations(shape, monkeypatch):
    """MLX-LM passes [batch, sequence, hidden], not [rows, hidden].

    The per-row reference the tuner builds has to take rows from a flattened view.
    Slicing axis 0 of a rank-3 tensor yields empty rows once batch is smaller than the
    row count, and concatenating those crashes MLX inside eval.
    """
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from metile.backends import mlx_dense_residual, mlx_dense_swiglu
    from metile.backends.mlx_dense import MLXDenseWeight

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    for module in (mlx_dense_swiglu, mlx_dense_residual):
        module._kernel_cache.clear()
        module._schedule_cache.clear()

    hidden, intermediate = shape[-1], 512
    random = np.random.default_rng(5501)

    def sample(*dims):
        return mx.array(random.normal(size=dims).astype(np.float32)).astype(mx.bfloat16)

    gate, up = sample(intermediate, hidden), sample(intermediate, hidden)
    down = sample(hidden, intermediate)
    paired = mx.stack((gate, up), axis=-1)
    values, residual = sample(*shape), sample(*shape)
    mx.eval(gate, up, down, paired, values, residual)

    actual = mlx_dense_residual.mlx_dense_residual_qmv(
        mlx_dense_swiglu.mlx_dense_swiglu(
            values,
            MLXDenseWeight.from_mlx(gate),
            MLXDenseWeight.from_mlx(up),
            paired_weight=paired,
        ),
        down,
        residual,
    )

    rows = values.size // hidden
    flat_values = values.reshape(rows, hidden)
    flat_residual = residual.reshape(rows, hidden)
    expected = mx.concatenate(
        [
            (nn.silu(flat_values[row : row + 1] @ gate.T) * (flat_values[row : row + 1] @ up.T))
            @ down.T
            + flat_residual[row : row + 1]
            for row in range(rows)
        ],
        axis=0,
    ).reshape(shape)
    mx.eval(actual, expected)

    assert actual.shape == shape
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


def test_mlx_dense_residual_requires_exact_speedup_margin():
    from metile.backends import mlx_dense_residual

    native = mlx_dense_residual.MLXDenseResidualConfig("mlx")
    generated = mlx_dense_residual.MLXDenseResidualConfig("metile", 1, 1)

    close = mlx_dense_residual._choose_config([(1.0, 0, native), (0.99, 100, generated)])
    faster = mlx_dense_residual._choose_config([(1.0, 0, native), (0.98, 100, generated)])

    assert close == native
    assert faster == generated


def test_dense_swiglu_selection_prefers_the_faster_ratio():
    """The backend's selection must act on confirmed timings, native included."""
    pytest.importorskip("mlx.core")
    from metile.backends import mlx_dense_swiglu

    native = mlx_dense_swiglu.MLXDenseSwiGLUConfig("mlx")
    fast = mlx_dense_swiglu.MLXDenseSwiGLUConfig(
        "metile", implementation="simdgroup_paired", outputs_per_simdgroup=2
    )
    slow = mlx_dense_swiglu.MLXDenseSwiGLUConfig(
        "metile", implementation="simdgroup_paired", outputs_per_simdgroup=1
    )
    chosen = mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.5, 10, fast), (0.9, 20, slow)])
    assert chosen is fast

    # Nothing clears the switch margin, so native stays.
    assert mlx_dense_swiglu._choose_config([(1.0, 0, native), (0.999, 10, fast)]) is native


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

    _patch_mlx_lm(monkeypatch, "_execute_dense_mlp", execute_dense)

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
