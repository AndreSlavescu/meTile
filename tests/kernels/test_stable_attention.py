import inspect

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile_kernels.stable_attention import (
    stable_attention_backward_key_value,
    stable_attention_backward_query,
    stable_attention_forward,
)

KERNELS = (
    stable_attention_forward,
    stable_attention_backward_query,
    stable_attention_backward_key_value,
)


def _reference(query, key, value, gradient, scale, mask=None, causal=False, offset=0):
    query, key, value, gradient = (
        array.astype(np.float64) for array in (query, key, value, gradient)
    )
    batch, query_heads, query_length, dimension = query.shape
    kv_heads, key_length = key.shape[1:3]
    group_size = query_heads // kv_heads
    expanded_key = np.repeat(key, group_size, axis=1)
    expanded_value = np.repeat(value, group_size, axis=1)
    scores = query @ expanded_key.swapaxes(-1, -2)
    visible = np.ones_like(scores, dtype=bool)
    if mask is not None:
        visible &= mask != 0
    if causal:
        visible &= np.arange(key_length)[None, :] <= np.arange(query_length)[:, None] + offset
    nonempty = visible.any(axis=-1, keepdims=True)
    maximum = np.where(visible, scores, -np.inf).max(axis=-1, keepdims=True)
    maximum = np.where(nonempty, maximum, 0.0)
    shifted = np.where(visible, scores - maximum, 0.0)
    weights = np.where(visible, np.exp(shifted * scale), 0.0)
    denominator = weights.sum(axis=-1, keepdims=True)
    safe_denominator = np.where(nonempty, denominator, 1.0)
    probability = weights / safe_denominator
    output = probability @ expanded_value
    delta = (gradient * output).sum(axis=-1, keepdims=True)
    grad_probability = gradient @ expanded_value.swapaxes(-1, -2)
    grad_score = probability * (grad_probability - delta)
    grad_query = (grad_score @ expanded_key) * scale
    grad_key = (grad_score.swapaxes(-1, -2) @ query) * scale
    grad_value = probability.swapaxes(-1, -2) @ gradient
    grouped_shape = (batch, kv_heads, group_size, key_length, dimension)
    return {
        "output": output,
        "maximum": maximum[..., 0],
        "log_denominator": np.log(safe_denominator)[..., 0],
        "grad_query": grad_query,
        "grad_key": grad_key.reshape(grouped_shape).sum(axis=2),
        "grad_value": grad_value.reshape(grouped_shape).sum(axis=2),
    }


def _trace(kernel, dtype="f32", **overrides):
    options = {
        "D": 32,
        "Q_HEADS": 4,
        "KV_HEADS": 2,
        "CAUSAL": True,
        "CAUSAL_OFFSET": -2,
        "HAS_MASK": True,
        "BLOCK": 32,
    }
    options.update(overrides)
    context = TracingContext(kernel.name)
    context.func.constexprs = {**options, "STRICT_MATH": True}
    with context:
        arguments = []
        for name, parameter in inspect.signature(kernel.fn).parameters.items():
            if parameter.annotation is metile.constexpr:
                continue
            if name in {"Q_LEN", "K_LEN"}:
                datatype = I32
            elif name == "scale":
                datatype = ScalarType("f32")
            elif name == "Mask":
                datatype = PtrType("u8")
            elif name in {"Q", "K", "V", "Out", "GradOut"}:
                datatype = PtrType(dtype)
            else:
                datatype = PtrType("f32")
            is_output = name in {"Out", "GradQ", "GradK", "GradV"} or (
                kernel is stable_attention_forward and name in {"OutFloat", "RowMax", "RowLogDen"}
            )
            context.func.params.append(tir.Param(name, datatype, is_output=is_output))
            arguments.append(TracingProxy(tir.Value(name, datatype)))
        kernel.fn(*arguments, **options)
    return context.func


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("dimension", [32, 96, 256])
def test_stable_attention_traces_to_one_simdgroup_without_atomics(kernel, dtype, dimension):
    function = lower(_trace(kernel, dtype, D=dimension))
    source = emit(function)
    assert function.threadgroup_size == (32, 1, 1)
    assert "simd_sum(" in source
    assert "atomic_" not in source
    assert "fast::exp" not in source
    assert "threadgroup_barrier(" not in source
    assert "for (" in source


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "options,exception,match",
    [
        ({"D": 31}, ValueError, "multiple of 32"),
        ({"D": 288}, ValueError, "multiple of 32"),
        ({"D": True}, ValueError, "multiple of 32"),
        ({"Q_HEADS": 3}, ValueError, "divisible"),
        ({"KV_HEADS": 0}, ValueError, "divisible"),
        ({"KV_HEADS": True}, ValueError, "divisible"),
        ({"CAUSAL": 1}, TypeError, "bool"),
        ({"HAS_MASK": 0}, TypeError, "bool"),
        ({"CAUSAL_OFFSET": 0.5}, ValueError, "signed 32-bit"),
        ({"CAUSAL_OFFSET": 1 << 31}, ValueError, "signed 32-bit"),
        ({"BLOCK": 64}, ValueError, "BLOCK=32"),
    ],
)
def test_stable_attention_rejects_unsupported_constexprs(kernel, options, exception, match):
    with pytest.raises(exception, match=match):
        _trace(kernel, **options)


@pytest.mark.parametrize("causal,offset", [(False, 0), (True, 0), (True, -2)])
def test_fp64_reference_gradients_match_finite_differences(causal, offset):
    generator = np.random.default_rng(872)
    query = generator.normal(size=(1, 2, 5, 4))
    key = generator.normal(size=(1, 1, 3, 4))
    value = generator.normal(size=key.shape)
    gradient = generator.normal(size=query.shape)
    mask = (generator.random((1, 2, 5, 3)) > 0.25).astype(np.uint8)
    mask[:, :, 1, :] = 0
    scale = 0.37
    expected = _reference(query, key, value, gradient, scale, mask, causal, offset)
    arrays = [query, key, value]
    for position, name in enumerate(("grad_query", "grad_key", "grad_value")):
        direction = generator.normal(size=arrays[position].shape)
        direction /= np.linalg.norm(direction)
        plus, minus = list(arrays), list(arrays)
        step = 1e-5
        plus[position] = arrays[position] + step * direction
        minus[position] = arrays[position] - step * direction
        upper = _reference(*plus, gradient, scale, mask, causal, offset)["output"]
        lower_output = _reference(*minus, gradient, scale, mask, causal, offset)["output"]
        numerical = ((upper - lower_output) * gradient).sum() / (2.0 * step)
        analytical = (expected[name] * direction).sum()
        np.testing.assert_allclose(analytical, numerical, rtol=2e-8, atol=2e-9)
    assert np.all(expected["output"][:, :, 1] == 0.0)
    assert np.all(expected["grad_query"][:, :, 1] == 0.0)


def test_large_offset_fixture_detects_combined_logsumexp_rounding():
    scores = np.array([2**24, 2**24 + 2, 2**24 + 4], dtype=np.float32)
    scale = np.float32(0.1234567)
    maximum = scores.max()
    shifted = (scores - maximum) * scale
    log_denominator = np.log(np.exp(shifted).sum())
    separate = np.exp(shifted - log_denominator)
    combined = np.exp(scores * scale - np.float32(maximum * scale + log_denominator))
    expected = np.exp((scores.astype(np.float64) - float(maximum)) * float(scale))
    expected /= expected.sum()
    np.testing.assert_allclose(separate, expected, rtol=3e-7, atol=3e-8)
    assert np.max(np.abs(combined - expected)) > 1e-3


def _run(query, key, value, gradient, scale, mask=None, causal=False, offset=0):
    batch, query_heads, query_length, dimension = query.shape
    kv_heads, key_length = key.shape[1:3]
    mask_array = (
        np.zeros((1,), dtype=np.uint8)
        if mask is None
        else np.ascontiguousarray(mask, dtype=np.uint8)
    )
    output = np.full_like(query, np.nan)
    precise = np.full(query.shape, np.nan, dtype=np.float32)
    maximum = np.full(query.shape[:-1], np.nan, dtype=np.float32)
    log_denominator = np.full_like(maximum, np.nan)
    grad_query = np.full(query.shape, np.nan, dtype=np.float32)
    grad_key = np.full(key.shape, np.nan, dtype=np.float32)
    grad_value = np.full(value.shape, np.nan, dtype=np.float32)
    options = {
        "D": dimension,
        "Q_HEADS": query_heads,
        "KV_HEADS": kv_heads,
        "CAUSAL": causal,
        "CAUSAL_OFFSET": offset,
        "HAS_MASK": mask is not None,
        "BLOCK": 32,
        "STRICT_MATH": True,
    }
    common = (query, key, value, mask_array)
    dimensions = (query_length, key_length, float(np.float32(scale)))
    query_grid = (batch * query_heads * query_length,)
    stable_attention_forward[query_grid](
        *common, output, precise, maximum, log_denominator, *dimensions, **options
    )
    backward_inputs = (*common, precise, maximum, log_denominator, gradient)
    stable_attention_backward_query[query_grid](
        *backward_inputs, grad_query, *dimensions, **options
    )
    stable_attention_backward_key_value[(batch * kv_heads * key_length,)](
        *backward_inputs, grad_key, grad_value, *dimensions, **options
    )
    return {
        "output": precise,
        "rounded_output": output,
        "maximum": maximum,
        "log_denominator": log_denominator,
        "grad_query": grad_query,
        "grad_key": grad_key,
        "grad_value": grad_value,
    }


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "query_heads,kv_heads,query_length,key_length,dimension,causal,offset,masked",
    [
        (2, 2, 3, 7, 32, False, 0, False),
        (4, 2, 5, 3, 64, True, 0, True),
        (4, 1, 5, 3, 32, True, -2, True),
        (2, 1, 3, 7, 96, True, 4, False),
        (2, 1, 3, 7, 256, False, 0, True),
    ],
)
def test_gpu_stable_attention_forward_backward_against_fp64(
    dtype, query_heads, kv_heads, query_length, key_length, dimension, causal, offset, masked
):
    generator = np.random.default_rng(611 + dimension)
    query = generator.normal(size=(2, query_heads, query_length, dimension)).astype(dtype)
    key = generator.normal(size=(2, kv_heads, key_length, dimension)).astype(dtype)
    value = generator.normal(size=key.shape).astype(dtype)
    gradient = generator.normal(size=query.shape).astype(dtype)
    mask = None
    if masked:
        mask = (generator.random((2, query_heads, query_length, key_length)) > 0.35).astype(
            np.uint8
        )
        mask[:, :, 1, :] = 0
        mask[:, :, :, -1] = 0
    scale = float(np.float32(dimension**-0.5))
    expected = _reference(query, key, value, gradient, scale, mask, causal, offset)
    actual = _run(query, key, value, gradient, scale, mask, causal, offset)
    for name, reference in expected.items():
        np.testing.assert_allclose(actual[name], reference, rtol=4e-5, atol=7e-6, err_msg=name)
    tolerance = 8e-4 if dtype is np.float16 else 7e-6
    np.testing.assert_allclose(
        actual["rounded_output"], expected["output"], rtol=tolerance, atol=tolerance
    )
    if masked:
        assert np.all(actual["output"][:, :, 1] == 0.0)
        assert np.all(actual["grad_query"][:, :, 1] == 0.0)
        assert np.all(actual["grad_key"][:, :, -1] == 0.0)
        assert np.all(actual["grad_value"][:, :, -1] == 0.0)
    if causal and offset < 0:
        assert np.all(actual["output"][:, :, :-offset] == 0.0)
        assert np.all(actual["grad_query"][:, :, :-offset] == 0.0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_gpu_stable_attention_large_raw_scores_and_near_ties(dtype, sign):
    generator = np.random.default_rng(903)
    query = np.zeros((1, 2, 2, 32), dtype=dtype)
    key = np.zeros((1, 1, 3, 32), dtype=dtype)
    query[..., 0] = 4096
    query[..., 1] = 1
    key[..., 0] = sign * 4096
    key[..., 1] = [0, 2, 4]
    value = generator.normal(size=key.shape).astype(dtype)
    gradient = generator.normal(size=query.shape).astype(dtype)
    scale = float(np.float32(0.1234567))
    expected = _reference(query, key, value, gradient, scale)
    actual = _run(query, key, value, gradient, scale)
    for name in ("output", "maximum", "log_denominator", "grad_value"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=5e-6, atol=5e-6)
    for name in ("grad_query", "grad_key"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-5, atol=2e-3)
    np.testing.assert_allclose(
        actual["grad_query"][..., 1:], expected["grad_query"][..., 1:], rtol=2e-5, atol=4e-6
    )


def test_gpu_fully_masked_rows_ignore_overflowing_unused_gradient_products():
    query = np.ones((1, 2, 3, 32), dtype=np.float32)
    key = np.ones((1, 1, 5, 32), dtype=np.float32)
    value = np.full_like(key, 1e20)
    gradient = np.full_like(query, 1e20)
    mask = np.zeros((1, 2, 3, 5), dtype=np.uint8)
    actual = _run(query, key, value, gradient, 0.25, mask)
    for name, result in actual.items():
        assert np.all(result == 0.0), name
