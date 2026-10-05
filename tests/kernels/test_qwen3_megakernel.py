"""The experimental full-token path must match logits and every KV-cache update."""

import sys

import numpy as np
import pytest

from benchmarks.megakernels.qwen3_runtime import Qwen3Megakernel, validate_config


def _config(**changes):
    config = dict(
        model_type="qwen3",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=67,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        rope_scaling=None,
    )
    return config | changes


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"model_type": "qwen3_5"}, "different architectures"),
        ({"tie_word_embeddings": False}, "tied"),
        ({"rope_scaling": {"factor": 2.0}}, "scaled RoPE"),
        ({"hidden_size": 0}, "positive integer"),
        ({"head_dim": 33}, "multiples of 32"),
        ({"num_attention_heads": 3, "num_key_value_heads": 2}, "divisible"),
        ({"hidden_size": 8192}, "scratch budget"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"vocab_size": 2**30}, "32-bit address"),
    ],
)
def test_unsupported_models_fail_closed(changes, match):
    with pytest.raises(ValueError, match=match):
        validate_config(_config(**changes), 16, 128)


@pytest.mark.parametrize("capacity,threads", [(0, 128), (True, 128), (129, 128), (16, 33)])
def test_invalid_allocation_or_launch_contract_is_rejected(capacity, threads):
    with pytest.raises(ValueError):
        validate_config(_config(), capacity, threads)


def test_official_small_model_fits_scratch_without_confusing_query_width_and_hidden():
    config = _config(
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=40960,
    )
    assert validate_config(config, 512, 256) == 24 * 1024


@pytest.fixture(params=[32, 128])
def tiny_model(request):
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    core = pytest.importorskip("mlx.core")
    qwen = pytest.importorskip("mlx_lm.models.qwen3")
    core.random.seed(381)
    model = qwen.Model(qwen.ModelArgs(**_config(head_dim=request.param)))
    model.set_dtype(core.float16)
    for layer in model.layers:
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn.q_norm,
            layer.self_attn.k_norm,
        ):
            norm.weight = core.random.uniform(0.25, 1.75, shape=norm.weight.shape).astype(
                core.float16
            )
    core.eval(model.parameters())
    return model


@pytest.mark.parametrize("threads", [32, 128, 256])
def test_multiple_full_token_steps_match_native_mlx_and_all_cache_entries(tiny_model, threads):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from metile.runtime.metal_device import MetalDevice

    native_cache = make_prompt_cache(tiny_model)
    candidate = Qwen3Megakernel(tiny_model, capacity=8, threads=threads)
    for position, token in enumerate((3, 7, 11, 5)):
        expected = tiny_model(mx.array([[token]]), cache=native_cache)
        mx.eval(expected, [layer.state for layer in native_cache])
        dispatcher = candidate.prepare(token, position)
        actual = candidate.logits.numpy().copy()
        expected_logits = np.asarray(expected).reshape(-1)
        np.testing.assert_allclose(actual, expected_logits, rtol=6e-3, atol=4e-3)
        assert np.argmax(actual) == np.argmax(expected_logits)
        for index, layer in enumerate(native_cache):
            for cache_kind, values in enumerate((layer.keys, layer.values)):
                np.testing.assert_allclose(
                    candidate.cache.numpy()[cache_kind, index, :, : position + 1],
                    np.asarray(values)[0, :, : position + 1],
                    rtol=6e-3,
                    atol=4e-3,
                )
        assert (dispatcher._grid.width, dispatcher._grid.height, dispatcher._grid.depth) == (
            threads,
            1,
            1,
        )
        assert dispatcher._tg.width == threads
        saved_cache = candidate.cache.numpy().copy()
        dispatcher()
        assert MetalDevice.get()._pending_dispatches == 1
        MetalDevice.get().sync()
        np.testing.assert_array_equal(candidate.logits.numpy(), actual)
        np.testing.assert_array_equal(candidate.cache.numpy(), saved_cache)


def test_imported_prefix_and_uninitialized_positions(tiny_model):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    native_cache = make_prompt_cache(tiny_model)
    prefix = tiny_model(mx.array([[3, 7, 11]]), cache=native_cache)
    mx.eval(prefix, [layer.state for layer in native_cache])
    candidate = Qwen3Megakernel(tiny_model, capacity=5, threads=128)
    with pytest.raises(ValueError, match="uninitialized"):
        candidate.prepare(5, 3)
    candidate.load_cache(native_cache)
    assert candidate.cached_tokens == 3
    candidate.prepare(5, 3)
    expected = tiny_model(mx.array([[5]]), cache=native_cache)
    mx.eval(expected)
    np.testing.assert_allclose(
        candidate.logits.numpy(), np.asarray(expected).reshape(-1), rtol=6e-3, atol=4e-3
    )
    for token, position in ((-1, 0), (67, 0), (True, 0), (0, 5), (0, True)):
        with pytest.raises(ValueError):
            candidate.prepare(token, position)
    candidate.reset()
    assert candidate.cached_tokens == 0
    assert np.count_nonzero(candidate.cache.numpy()) == 0


def test_bfloat16_weights_are_not_silently_converted(tiny_model):
    import mlx.core as mx

    tiny_model.set_dtype(mx.bfloat16)
    with pytest.raises(ValueError, match="dense FP16"):
        Qwen3Megakernel(tiny_model, capacity=4)


def test_fp32_mode_keeps_all_buffers_and_intermediates_in_fp32(tiny_model):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    tiny_model.set_dtype(mx.float32)
    native_cache = make_prompt_cache(tiny_model)
    candidate = Qwen3Megakernel(tiny_model, capacity=3, threads=128)
    assert candidate.dtype == np.float32
    assert candidate.config["storage_dtype"] == "float32"
    for position, token in enumerate((3, 7, 11)):
        expected = tiny_model(mx.array([[token]]), cache=native_cache)
        mx.eval(expected)
        candidate.prepare(token, position)
        np.testing.assert_allclose(
            candidate.logits.numpy(), np.asarray(expected).reshape(-1), rtol=1e-4, atol=1e-4
        )
        for index, layer in enumerate(native_cache):
            for kind, values in enumerate((layer.keys, layer.values)):
                np.testing.assert_allclose(
                    candidate.cache.numpy()[kind, index, :, : position + 1],
                    np.asarray(values)[0, :, : position + 1],
                    rtol=1e-4,
                    atol=1e-4,
                )


@pytest.mark.parametrize("mutation", ["bias", "norm", "rope", "adapter"])
def test_modified_model_components_are_rejected_before_packing(tiny_model, mutation):
    import mlx.core as mx
    import mlx.nn as nn

    attention = tiny_model.layers[0].self_attn
    if mutation == "bias":
        attention.q_proj.bias = mx.zeros((64,), dtype=mx.float16)
    elif mutation == "norm":
        attention.q_norm.eps = 1e-4
    elif mutation == "rope":
        attention.rope.traditional = True
    else:

        class AdaptedLinear(nn.Linear):
            pass

        attention.q_proj = AdaptedLinear(32, 64, bias=False)
    with pytest.raises(ValueError):
        Qwen3Megakernel(tiny_model, capacity=4)
