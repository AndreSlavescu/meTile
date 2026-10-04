import numpy as np
import pytest

from metile.backends.dual_chunk_attention import (
    dual_chunk_attention_backward,
    dual_chunk_attention_forward,
)
from metile.backends.gated_delta import gated_delta_reference, gated_delta_reference_backward
from metile.backends.native_attention_graph import compile_native_attention_graph
from metile.ir.attention_graph import (
    dual_chunk_attention,
    gated_delta_attention,
    kimi_delta_attention,
    stable_attention,
)
from metile.ir.graph_ir import GraphBuilder, TensorSpec
from metile.runtime.buffer import MtileBuffer
from tests.kernels.test_stable_attention import _reference as _stable_reference


def _attention_reference(query, key, value, scale, causal_offset):
    query, key, value = (array.astype(np.float64) for array in (query, key, value))
    group = query.shape[1] // key.shape[1]
    key = np.repeat(key, group, axis=1)
    value = np.repeat(value, group, axis=1)
    scores = (query @ key.swapaxes(-1, -2)) * scale
    allowed = np.arange(key.shape[2])[None, :] <= np.arange(query.shape[2])[:, None] + causal_offset
    scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value


def test_gpu_native_stable_attention_chain_matches_fp64_reference():
    generator = np.random.default_rng(742)
    query = generator.normal(size=(1, 2, 3, 32)).astype(np.float32)
    key = generator.normal(size=(1, 1, 5, 32)).astype(np.float32)
    value = generator.normal(size=key.shape).astype(np.float32)
    builder = GraphBuilder()
    query_value = builder.input("query", TensorSpec(query.shape, "f32"))
    key_value = builder.input("key", TensorSpec(key.shape, "f32"))
    values = builder.input("value", TensorSpec(value.shape, "f32"))
    first = stable_attention(builder, query_value, key_value, values, causal=True, scale=0.25)
    second = stable_attention(builder, first, key_value, values, causal=True, scale=0.25)
    executable = compile_native_attention_graph(builder.build((first, second)))
    actual_first, actual_second = executable(query, key, value)
    expected_first = _attention_reference(query, key, value, 0.25, 2)
    expected_second = _attention_reference(expected_first, key, value, 0.25, 2)
    np.testing.assert_allclose(actual_first.numpy(), expected_first, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual_second.numpy(), expected_second, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("channel", [False, True])
def test_gpu_native_delta_chain_threads_state_explicitly(channel):
    generator = np.random.default_rng(528)
    query = (generator.normal(size=(1, 3, 1, 4)) * 0.3).astype(np.float32)
    key = (generator.normal(size=query.shape) * 0.3).astype(np.float32)
    value = generator.normal(size=(1, 3, 1, 3)).astype(np.float32)
    log_decay = np.full(query.shape if channel else query.shape[:-1], -0.2, dtype=np.float32)
    beta = np.full(query.shape[:-1], 0.4, dtype=np.float32)
    initial = generator.normal(size=(1, 1, 4, 3)).astype(np.float32)
    original_state = initial.copy()
    arrays = (query, key, value, log_decay, beta, initial)
    names = ("query", "key", "value", "log_decay", "beta", "initial_state")
    builder = GraphBuilder()
    inputs = [
        builder.input(name, TensorSpec(array.shape, "f32")) for name, array in zip(names, arrays)
    ]
    operation = kimi_delta_attention if channel else gated_delta_attention
    first_output, first_state = operation(builder, *inputs)
    second_output, second_state = operation(builder, *inputs[:-1], first_state)
    executable = compile_native_attention_graph(
        builder.build((first_output, first_state, second_output, second_state))
    )
    actual, tape = executable.forward_with_context(*arrays)
    reference_arrays = [array.astype(np.float64) for array in arrays]
    first = gated_delta_reference(*reference_arrays)
    second = gated_delta_reference(*reference_arrays[:-1], first.final_state)
    for result, expected in zip(
        actual, (first.output, first.final_state, second.output, second.final_state)
    ):
        np.testing.assert_allclose(result.numpy(), expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(initial, original_state)
    output_seeds = [
        generator.normal(size=value.shape).astype(np.float32),
        None,
        generator.normal(size=value.shape).astype(np.float32),
        generator.normal(size=initial.shape).astype(np.float32),
    ]
    derivatives = executable.backward(tape, output_seeds)
    second_derivatives = gated_delta_reference_backward(
        *reference_arrays[:5],
        second.states,
        output_seeds[2].astype(np.float64),
        output_seeds[3].astype(np.float64),
    )
    first_derivatives = gated_delta_reference_backward(
        *reference_arrays[:5],
        first.states,
        output_seeds[0].astype(np.float64),
        second_derivatives.initial_state,
    )
    for index, name in enumerate(("query", "key", "value", "log_decay", "beta", "initial_state")):
        expected = getattr(first_derivatives, name)
        if name != "initial_state":
            expected = expected + getattr(second_derivatives, name)
        np.testing.assert_allclose(
            derivatives[index].numpy(), expected, rtol=5e-5, atol=5e-6, err_msg=name
        )
    for derivative in executable.backward(tape):
        np.testing.assert_array_equal(derivative.numpy(), 0.0)


def test_gpu_native_dca_node_preserves_explicit_branch_inputs():
    generator = np.random.default_rng(691)
    queries = [generator.normal(size=(1, 2, 3, 32)).astype(np.float32) for _ in range(3)]
    key = generator.normal(size=(1, 1, 9, 32)).astype(np.float32)
    value = generator.normal(size=key.shape).astype(np.float32)
    arrays = (*queries, key, value)
    builder = GraphBuilder()
    inputs = [
        builder.input(name, TensorSpec(array.shape, "f32"))
        for name, array in zip(("intra", "successive", "inter", "key", "value"), arrays)
    ]
    options = {"chunk_size": 6, "local_window": 2, "query_start": 6}
    output = dual_chunk_attention(builder, *inputs, **options)
    executable = compile_native_attention_graph(builder.build(output))
    result, tape = executable.forward_with_context(*arrays)
    expected = dual_chunk_attention_forward(*arrays, **options)
    np.testing.assert_allclose(result.numpy(), expected.output.numpy(), rtol=1e-6, atol=1e-6)
    seed = generator.normal(size=queries[0].shape).astype(np.float32)
    actual_gradients = executable.backward(tape, seed)
    expected_gradients = dual_chunk_attention_backward(expected, seed)
    for actual, name in zip(
        actual_gradients, ("query_intra", "query_successive", "query_inter", "key", "value")
    ):
        np.testing.assert_allclose(
            actual.numpy(), getattr(expected_gradients, name).numpy(), rtol=1e-6, atol=1e-6
        )


def test_gpu_stable_graph_vjp_accumulates_shared_leaves_and_multiple_output_seeds():
    generator = np.random.default_rng(771)
    source = generator.normal(size=(1, 1, 3, 32)).astype(np.float32)
    original = source.copy()
    mask = np.tril(np.ones((1, 1, 3, 3), dtype=np.uint8))
    mask_original = mask.copy()
    unused = np.ones((2, 3), dtype=np.float16)
    first_seed = generator.normal(size=source.shape).astype(np.float16)
    second_seed = generator.normal(size=source.shape).astype(np.float32)
    builder = GraphBuilder()
    leaf = builder.input("shared", TensorSpec(source.shape, "f32"))
    mask_value = builder.input("mask", TensorSpec(mask.shape, "u8"))
    builder.input("unused", TensorSpec(unused.shape, "f16"))
    first = stable_attention(builder, leaf, leaf, leaf, scale=0.25, mask=mask_value)
    second = stable_attention(builder, first, leaf, leaf, scale=0.25, mask=mask_value)
    executable = compile_native_attention_graph(builder.build((first, second, mask_value)))
    _, tape = executable.forward_with_context(source, mask, unused)
    source.fill(99.0)
    mask.fill(0)
    first_seed_buffer = MtileBuffer.from_numpy(first_seed)
    actual, mask_gradient, unused_gradient = executable.backward(
        tape, (first_seed_buffer, second_seed, None)
    )
    first_reference = _stable_reference(
        original, original, original, np.zeros_like(original), 0.25, mask_original
    )
    second_reference = _stable_reference(
        first_reference["output"], original, original, second_seed, 0.25, mask_original
    )
    first_derivatives = _stable_reference(
        original,
        original,
        original,
        first_seed.astype(np.float64) + second_reference["grad_query"],
        0.25,
        mask_original,
    )
    expected = sum(first_derivatives[name] for name in ("grad_query", "grad_key", "grad_value"))
    expected += second_reference["grad_key"] + second_reference["grad_value"]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=8e-5, atol=8e-6)
    assert actual.dtype == np.dtype(np.float32)
    assert mask_gradient is None
    np.testing.assert_array_equal(unused_gradient.numpy(), 0.0)
    np.testing.assert_array_equal(source, 99.0)
    np.testing.assert_array_equal(first_seed_buffer.numpy(), first_seed)


def test_gpu_graph_vjp_sums_repeated_output_cotangents():
    builder = GraphBuilder()
    leaf = builder.input("leaf", TensorSpec((3,), "f16"))
    executable = compile_native_attention_graph(builder.build((leaf, leaf)))
    _, tape = executable.forward_with_context(np.ones(3, dtype=np.float16))
    first_seed = np.array([1, 2, 3], dtype=np.float16)
    second_seed = np.array([4, 5, 6], dtype=np.float32)
    (gradient,) = executable.backward(tape, (first_seed, second_seed))
    np.testing.assert_array_equal(gradient.numpy(), [5, 7, 9])
    assert gradient.dtype == np.dtype(np.float32)
