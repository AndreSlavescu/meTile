"""Stage-level GPU checks complement end-to-end model and cache comparisons."""

import sys

import numpy as np
import pytest

import metile
from metile_kernels.megakernels.qwen3 import qwen3_layer_offsets
from metile_kernels.megakernels.qwen3_staged import (
    qwen3_staged_attention,
    qwen3_staged_qk_rope,
    qwen3_staged_qkv,
    qwen3_staged_residual,
    qwen3_staged_rmsnorm,
)

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


def _stored(values, dtype):
    return np.asarray(values).astype(dtype).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_parallel_qkv_uses_adjacent_packed_projection_rows(dtype):
    generator = np.random.default_rng(2468)
    geometry = dict(HIDDEN=32, INTERMEDIATE=64, QUERY_HEADS=3, KV_HEADS=1, HEAD_DIM=32)
    offsets = qwen3_layer_offsets(*geometry.values())
    weights = generator.normal(0, 0.1, size=(2, offsets["layer_size"])).astype(dtype)
    source = generator.normal(size=32).astype(dtype)
    output = np.full(160, np.nan, dtype=dtype)
    projection = weights[1, offsets["q_proj"] : offsets["o_proj"]].reshape(160, 32)
    expected = (projection.astype(np.float32) @ source.astype(np.float32)).astype(dtype)
    qwen3_staged_qkv[(40,)].prepare(
        source,
        weights,
        output,
        1,
        **geometry,
        BLOCK=128,
        STORAGE_DTYPE="f16" if dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_parallel_residual_projection_masks_partial_output_group(dtype):
    generator = np.random.default_rng(421)
    source = generator.normal(size=64).astype(dtype)
    weights = generator.normal(0, 0.1, size=67 * 64 + 23).astype(dtype)
    residual = generator.normal(size=67).astype(dtype)
    output = np.full(67, np.nan, dtype=dtype)
    projected = _stored(
        weights[23:].reshape(67, 64).astype(np.float32) @ source.astype(np.float32), dtype
    )
    expected = _stored(residual.astype(np.float32) + projected, dtype)
    qwen3_staged_residual[(metile.cdiv(67, 4),)].prepare(
        source,
        weights,
        residual,
        output,
        0,
        ROWS=67,
        COLUMNS=64,
        WEIGHT_OFFSET=23,
        WEIGHT_STRIDE=0,
        BLOCK=128,
        STORAGE_DTYPE="f16" if dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_staged_norm_matches_storage_rounding_with_multiple_simdgroups(dtype):
    generator = np.random.default_rng(512)
    source = generator.normal(size=160).astype(dtype)
    weights = generator.uniform(0.25, 2.0, size=160).astype(dtype)
    output = np.empty_like(source)
    reciprocal = 1 / np.sqrt(np.mean(source.astype(np.float32) ** 2) + 1e-6)
    expected = _stored(_stored(source.astype(np.float32) * reciprocal, dtype) * weights, dtype)
    qwen3_staged_rmsnorm[(1,)].prepare(
        source,
        weights,
        output,
        0,
        WIDTH=160,
        BLOCK=128,
        STORAGE_DTYPE="f16" if dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("dimension", [32, 64, 96, 128])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_qk_norm_rope_masks_head_and_half_head_tails_without_touching_other_cache_slots(
    dimension, dtype
):
    generator = np.random.default_rng(917)
    geometry = dict(HIDDEN=32, INTERMEDIATE=64, QUERY_HEADS=2, KV_HEADS=1, HEAD_DIM=dimension)
    offsets = qwen3_layer_offsets(*geometry.values())
    weights = generator.uniform(0.25, 2.0, size=(2, offsets["layer_size"])).astype(dtype)
    qkv = generator.normal(size=(4, dimension)).astype(dtype)
    angles = generator.uniform(-1, 1, size=(5, dimension // 2)).astype(np.float32)
    rotary = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    control = np.array([3, 2], dtype=np.int32)
    queries = np.full((2, dimension), np.nan, dtype=dtype)
    cache = np.full((2, 2, 1, 5, dimension), 321, dtype=dtype)
    expected_cache = cache.copy()
    normalized = qkv[:3].astype(np.float32)
    normalized = _stored(
        normalized / np.sqrt(np.mean(normalized**2, axis=1, keepdims=True) + 1e-6), dtype
    )
    query_norm = weights[1, offsets["q_norm"] : offsets["k_norm"]]
    key_norm = weights[1, offsets["k_norm"] :]
    normalized = _stored(normalized * np.stack((query_norm, query_norm, key_norm)), dtype)
    left, right = np.split(normalized, 2, axis=-1)
    cosine, sine = rotary[2, :, 0], rotary[2, :, 1]
    rotated = _stored(
        np.concatenate((left * cosine - right * sine, right * cosine + left * sine), axis=-1), dtype
    )
    expected_cache[0, 1, 0, 2] = rotated[2]
    expected_cache[1, 1, 0, 2] = qkv[3]
    qwen3_staged_qk_rope[(1,)].prepare(
        qkv,
        weights,
        rotary,
        control,
        queries,
        cache,
        1,
        **geometry,
        LAYERS=2,
        MAX_CONTEXT=5,
        BLOCK=128,
        STORAGE_DTYPE="f16" if dtype == np.float16 else "f32",
        STRICT_MATH=True,
    )
    tolerance = 2e-3 if dtype == np.float16 else 1e-6
    np.testing.assert_allclose(queries, rotated[:2], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(cache, expected_cache, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(cache[0, :, :, :2], expected_cache[0, :, :, :2])
    np.testing.assert_array_equal(cache[0, :, :, 3:], expected_cache[0, :, :, 3:])
    np.testing.assert_array_equal(cache[1], expected_cache[1])


@pytest.mark.parametrize("position", [0, 3])
def test_attention_reads_current_cache_slot_but_not_uninitialized_tail(position):
    generator = np.random.default_rng(613)
    queries = generator.normal(size=(3, 96)).astype(np.float32)
    cache = generator.normal(size=(2, 2, 1, 5, 96)).astype(np.float32)
    cache[:, :, :, position + 1 :] = np.nan
    control = np.array([3, position], dtype=np.int32)
    output = np.full_like(queries, np.nan)
    keys = cache[0, 1, 0, : position + 1]
    values = cache[1, 1, 0, : position + 1]
    scores = queries @ keys.T / np.sqrt(96)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities @ values
    qwen3_staged_attention[(1,)].prepare(
        queries,
        cache,
        control,
        output,
        1,
        QUERY_HEADS=3,
        KV_HEADS=1,
        HEAD_DIM=96,
        LAYERS=2,
        MAX_CONTEXT=5,
        BLOCK=128,
        STORAGE_DTYPE="f32",
        STRICT_MATH=True,
    )
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)
