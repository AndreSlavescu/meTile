from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends.dual_chunk_attention import (
    DualChunkAttentionResult,
    dual_chunk_attention_backward,
    dual_chunk_attention_forward,
    dual_chunk_position_indices,
)
from metile.runtime.metal_device import MetalDevice


def _reference(
    queries, key, value, gradient, scale, chunk_length, query_start, key_start, mask=None
):
    queries = [query.astype(np.float64) for query in queries]
    key, value, gradient = (array.astype(np.float64) for array in (key, value, gradient))
    batch, query_heads, query_length, dimension = queries[0].shape
    kv_heads, key_length = key.shape[1:3]
    group = query_heads // kv_heads
    expanded_key = np.repeat(key, group, axis=1)
    expanded_value = np.repeat(value, group, axis=1)
    query_positions = np.arange(query_length) + query_start
    key_positions = np.arange(key_length) + key_start
    distance = query_positions[:, None] // chunk_length - key_positions[None, :] // chunk_length
    visible = np.broadcast_to(
        key_positions[None, :] <= query_positions[:, None],
        (batch, query_heads, query_length, key_length),
    ).copy()
    if mask is not None:
        visible &= mask != 0
    masks = [visible & condition for condition in (distance == 0, distance == 1, distance >= 2)]
    scores = sum(
        np.where(branch_mask, query @ expanded_key.swapaxes(-1, -2), 0.0)
        for query, branch_mask in zip(queries, masks)
    )
    maximum = np.where(visible, scores, -np.inf).max(axis=-1, keepdims=True)
    maximum = np.where(visible.any(axis=-1, keepdims=True), maximum, 0.0)
    shifted = np.where(visible, scores - maximum, 0.0)
    weights = np.where(visible, np.exp(shifted * scale), 0.0)
    denominator = weights.sum(axis=-1, keepdims=True)
    denominator = np.where(denominator > 0.0, denominator, 1.0)
    probability = weights / denominator
    output = probability @ expanded_value
    delta = (gradient * output).sum(axis=-1, keepdims=True)
    score_gradient = probability * (gradient @ expanded_value.swapaxes(-1, -2) - delta)
    grad_queries = []
    grad_key = np.zeros_like(expanded_key)
    for query, branch_mask in zip(queries, masks):
        selected_gradient = np.where(branch_mask, score_gradient, 0.0)
        grad_queries.append((selected_gradient @ expanded_key) * scale)
        grad_key += (selected_gradient.swapaxes(-1, -2) @ query) * scale
    grad_value = probability.swapaxes(-1, -2) @ gradient
    grouped_shape = (batch, kv_heads, group, key_length, dimension)
    return (
        output,
        maximum[..., 0],
        np.log(denominator)[..., 0],
        (
            *grad_queries,
            grad_key.reshape(grouped_shape).sum(axis=2),
            grad_value.reshape(grouped_shape).sum(axis=2),
        ),
    )


def test_position_indices_match_pinned_chunkllama_inclusive_clamp():
    positions = np.array([0, 1, 5, 6, 7, 8, 9, 11])
    result = dual_chunk_position_indices(positions, positions, chunk_size=8, local_window=2)
    np.testing.assert_array_equal(result.query_intra, [0, 1, 5, 0, 1, 2, 3, 5])
    np.testing.assert_array_equal(result.query_successive, [6, 7, 8, 6, 7, 8, 8, 8])
    np.testing.assert_array_equal(result.query_inter, np.full(8, 8))
    np.testing.assert_array_equal(result.key, result.query_intra)
    wide_local = dual_chunk_position_indices(positions, positions, chunk_size=8, local_window=6)
    np.testing.assert_array_equal(wide_local.query_inter, np.full(8, 3))


def test_position_mapping_preserves_local_cross_boundary_distances():
    query = np.array([6, 7])
    key = np.array([4, 5])
    result = dual_chunk_position_indices(query, key, chunk_size=8, local_window=2)
    np.testing.assert_array_equal(
        result.query_successive[:, None] - result.key[None, :], query[:, None] - key[None, :]
    )


@pytest.mark.parametrize(
    "positions",
    [np.array([-1]), np.array([2**31]), np.array([True]), np.array([1.5]), np.ones((2, 2))],
)
def test_invalid_position_indices_reject(positions):
    with pytest.raises((TypeError, ValueError)):
        dual_chunk_position_indices(positions, np.array([0]), chunk_size=8, local_window=2)


@pytest.mark.parametrize("query_start,key_start", [(0, 0), (7, 2)])
def test_all_five_reference_gradients_match_finite_differences(query_start, key_start):
    generator = np.random.default_rng(123)
    queries = [generator.normal(size=(1, 2, 5, 4)) for _ in range(3)]
    key = generator.normal(size=(1, 1, 11, 4))
    value = generator.normal(size=key.shape)
    gradient = generator.normal(size=queries[0].shape)
    mask = (generator.random((1, 2, 5, 11)) > 0.2).astype(np.uint8)
    mask[:, :, 1] = 0
    arguments = (gradient, 0.37, 3, query_start, key_start, mask)
    expected = _reference(queries, key, value, *arguments)[3]
    arrays = [*queries, key, value]
    for index, derivative in enumerate(expected):
        direction = generator.normal(size=arrays[index].shape)
        direction /= np.linalg.norm(direction)
        plus, minus = list(arrays), list(arrays)
        step = 1e-5
        plus[index] = arrays[index] + step * direction
        minus[index] = arrays[index] - step * direction
        upper = _reference(plus[:3], *plus[3:], *arguments)[0]
        lower = _reference(minus[:3], *minus[3:], *arguments)[0]
        numerical = ((upper - lower) * gradient).sum() / (2 * step)
        np.testing.assert_allclose((derivative * direction).sum(), numerical, rtol=3e-8, atol=3e-9)


@pytest.fixture
def forbid_metal(monkeypatch):
    def unexpected():
        raise AssertionError("contract validation must run before Metal initialization")

    monkeypatch.setattr(MetalDevice, "get", unexpected)


def _inputs():
    queries = [np.zeros((1, 2, 5, 32), dtype=np.float32) for _ in range(3)]
    key = np.zeros((1, 1, 7, 32), dtype=np.float32)
    return [*queries, key, key.copy()]


@pytest.mark.usefixtures("forbid_metal")
@pytest.mark.parametrize(
    "options",
    [
        {"chunk_size": 0},
        {"chunk_size": True},
        {"local_window": -1},
        {"local_window": 5},
        {"query_start": -1},
        {"key_start": 2**31 - 1},
        {"query_start": True},
        {"scale": 0.0},
        {"scale": -1.0},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": 1e100},
        {"scale": 1e-100},
        {"scale": True},
        {"max_workspace_bytes": 1},
        {"max_workspace_bytes": True},
    ],
)
def test_configuration_errors_precede_device_initialization(options):
    configuration = {"chunk_size": 5, "local_window": 1, **options}
    with pytest.raises((TypeError, ValueError)):
        dual_chunk_attention_forward(*_inputs(), **configuration)


@pytest.mark.usefixtures("forbid_metal")
@pytest.mark.parametrize(
    "case", ["shape", "dtype", "contiguity", "mask_dtype", "mask_shape", "heads", "dimension"]
)
def test_tensor_contract_errors_precede_device_initialization(case):
    inputs = _inputs()
    options = {"chunk_size": 5, "local_window": 1}
    if case == "shape":
        inputs[1] = inputs[1][:, :, :1].copy()
    elif case == "dtype":
        inputs[2] = inputs[2].astype(np.float64)
    elif case == "contiguity":
        inputs[0] = inputs[0][..., ::-1]
    elif case == "mask_dtype":
        options["mask"] = np.zeros((1, 2, 5, 7), dtype=np.float32)
    elif case == "mask_shape":
        options["mask"] = np.zeros((5, 7), dtype=np.uint8)
    elif case == "heads":
        inputs[3:] = [np.zeros((1, 3, 7, 32), dtype=np.float32) for _ in range(2)]
    else:
        inputs = [array[..., :31].copy() for array in inputs]
    with pytest.raises((TypeError, ValueError)):
        dual_chunk_attention_forward(*inputs, **options)


@pytest.mark.usefixtures("forbid_metal")
def test_backward_budget_rejects_before_device_initialization():
    query, _, _, key, value = _inputs()
    context = SimpleNamespace(query=query, key=key, value=value)
    saved = DualChunkAttentionResult(None, None, None, None, (context, context, context))
    with pytest.raises(ValueError, match="budget"):
        dual_chunk_attention_backward(saved, np.zeros_like(query), max_workspace_bytes=1)
    with pytest.raises(TypeError, match="saved"):
        dual_chunk_attention_backward(None, query)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "query_length,key_length,query_start,key_start,dimension,query_heads,kv_heads",
    [
        (11, 11, 0, 0, 32, 4, 2),
        (5, 13, 8, 0, 64, 4, 1),
        (5, 10, 8, 3, 32, 2, 2),
        (3, 5, 0, 4, 32, 2, 1),
    ],
)
def test_gpu_dca_forward_backward_matches_one_global_fp64_softmax(
    dtype, query_length, key_length, query_start, key_start, dimension, query_heads, kv_heads
):
    generator = np.random.default_rng(72 + key_start)
    queries = [
        generator.normal(size=(2, query_heads, query_length, dimension)).astype(dtype)
        for _ in range(3)
    ]
    key = generator.normal(size=(2, kv_heads, key_length, dimension)).astype(dtype)
    value = generator.normal(size=key.shape).astype(dtype)
    gradient = generator.normal(size=queries[0].shape).astype(dtype)
    mask = (generator.random((2, query_heads, query_length, key_length)) > 0.25).astype(np.uint8)
    mask[:, :, 1] = 0
    scale = float(np.float32(dimension**-0.5))
    expected, maximum, log_denominator, derivatives = _reference(
        queries, key, value, gradient, scale, 4, query_start, key_start, mask
    )
    result = dual_chunk_attention_forward(
        *queries,
        key,
        value,
        chunk_size=6,
        local_window=2,
        query_start=query_start,
        key_start=key_start,
        mask=mask,
        scale=scale,
    )
    actual = dual_chunk_attention_backward(result, gradient)
    np.testing.assert_allclose(result.precise_output.numpy(), expected, rtol=5e-5, atol=9e-6)
    np.testing.assert_allclose(result.row_maximum.numpy(), maximum, rtol=5e-5, atol=9e-6)
    np.testing.assert_allclose(
        result.row_log_denominator.numpy(), log_denominator, rtol=5e-5, atol=9e-6
    )
    tolerance = 1e-3 if dtype is np.float16 else 9e-6
    np.testing.assert_allclose(result.output.numpy(), expected, rtol=tolerance, atol=tolerance)
    for name, derivative in zip(
        ("query_intra", "query_successive", "query_inter", "key", "value"), derivatives
    ):
        np.testing.assert_allclose(
            getattr(actual, name).numpy(), derivative, rtol=6e-5, atol=1e-5, err_msg=name
        )
    assert np.all(result.precise_output.numpy()[:, :, 1] == 0.0)
    for name in ("query_intra", "query_successive", "query_inter"):
        assert np.all(getattr(actual, name).numpy()[:, :, 1] == 0.0)


def test_gpu_dca_merge_weights_partitions_by_their_denominators():
    query = np.zeros((1, 1, 1, 32), dtype=np.float32)
    key = np.zeros((1, 1, 10, 32), dtype=np.float32)
    value = np.broadcast_to(
        np.arange(1, 11, dtype=np.float32)[None, None, :, None], key.shape
    ).copy()
    result = dual_chunk_attention_forward(
        query, query, query, key, value, chunk_size=6, local_window=2, query_start=9, scale=0.25
    )
    np.testing.assert_allclose(result.precise_output.numpy(), 5.5, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.row_log_denominator.numpy(), np.log(10), rtol=1e-6)
    gradients = dual_chunk_attention_backward(result, np.ones_like(query))
    np.testing.assert_allclose(gradients.value.numpy(), 0.1, rtol=1e-6)


def test_gpu_dca_merge_preserves_large_raw_maximum_separately():
    queries = [np.zeros((1, 2, 2, 32), dtype=np.float32) for _ in range(3)]
    key = np.zeros((1, 1, 12, 32), dtype=np.float32)
    for branch, query in enumerate(queries):
        query[..., 0] = 4096
        query[..., 1] = branch + 1
    key[..., 0] = 4096
    key[..., 1] = np.arange(12) * 2
    generator = np.random.default_rng(562)
    value = generator.normal(size=key.shape).astype(np.float32)
    gradient = generator.normal(size=queries[0].shape).astype(np.float32)
    scale = float(np.float32(0.1234567))
    expected, maximum, log_denominator, derivatives = _reference(
        queries, key, value, gradient, scale, 4, 10, 0
    )
    result = dual_chunk_attention_forward(
        *queries, key, value, chunk_size=6, local_window=2, query_start=10, scale=scale
    )
    np.testing.assert_allclose(result.precise_output.numpy(), expected, rtol=8e-6, atol=8e-6)
    np.testing.assert_array_equal(result.row_maximum.numpy(), maximum)
    np.testing.assert_allclose(
        result.row_log_denominator.numpy(), log_denominator, rtol=8e-6, atol=8e-6
    )
    gradients = dual_chunk_attention_backward(result, gradient)
    for name, derivative in zip(
        ("query_intra", "query_successive", "query_inter", "key", "value"), derivatives
    ):
        np.testing.assert_allclose(
            getattr(gradients, name).numpy(), derivative, rtol=3e-5, atol=3e-3, err_msg=name
        )
