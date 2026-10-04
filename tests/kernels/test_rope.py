import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.rope import _contract, rope_backward, rope_forward


def _reference(values, cosine, sine, rotary, interleaved):
    output = values.copy()
    left = slice(0, rotary, 2) if interleaved else slice(0, rotary // 2)
    right = slice(1, rotary, 2) if interleaved else slice(rotary // 2, rotary)
    output[:, left] = values[:, left] * cosine - values[:, right] * sine
    output[:, right] = values[:, right] * cosine + values[:, left] * sine
    return output


@pytest.mark.parametrize("kernel,pointer_count", [(rope_forward, 4), (rope_backward, 7)])
@pytest.mark.parametrize("interleaved", [False, True])
def test_rope_lowers_without_device(kernel, pointer_count, interleaved):
    with TracingContext("rope") as context:
        parameters = [(f"pointer{index}", PtrType("f32")) for index in range(pointer_count)]
        parameters.append(("rows", I32))
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        kernel.fn(*proxies, DIM=69, ROTARY_DIM=36, INTERLEAVED=interleaved, BLOCK=64)
    assert "[[kernel" in emit(lower(context.func))


@pytest.mark.parametrize("rotary,block", [(0, 64), (3, 64), (70, 64), (36, 32)])
def test_rope_rejects_invalid_geometry(rotary, block):
    with pytest.raises(ValueError):
        _contract(69, rotary, False, block)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("dimension,rotary", [(32, 32), (69, 36)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_rope_all_input_adjoints_match_finite_differences(
    interleaved, dimension, rotary, dtype
):
    generator = np.random.default_rng(107)
    values = generator.normal(size=(2, dimension)).astype(dtype)
    cosine = generator.normal(size=(2, rotary // 2)).astype(dtype)
    sine = generator.normal(size=cosine.shape).astype(dtype)
    seed = generator.normal(size=values.shape).astype(np.float32)
    output = np.empty_like(values)
    gradients = [np.full(array.shape, np.nan, dtype=np.float32) for array in (values, cosine, sine)]
    options = dict(
        DIM=dimension, ROTARY_DIM=rotary, INTERLEAVED=interleaved, BLOCK=128, STRICT_MATH=True
    )
    rope_forward[(2,)].prepare(values, cosine, sine, output, 2, **options)
    rope_backward[(2,)].prepare(values, cosine, sine, seed, *gradients, 2, **options)
    arrays = [array.astype(np.float64) for array in (values, cosine, sine)]
    expected = _reference(*arrays, rotary, interleaved)
    np.testing.assert_allclose(
        output, expected, rtol=2e-3 if dtype == np.float16 else 2e-5, atol=2e-5
    )
    for target, actual in enumerate(gradients):
        finite = np.empty_like(arrays[target])
        for index in np.ndindex(finite.shape):
            above = [array.copy() for array in arrays]
            below = [array.copy() for array in arrays]
            above[target][index] += 1e-5
            below[target][index] -= 1e-5
            difference = _reference(*above, rotary, interleaved) - _reference(
                *below, rotary, interleaved
            )
            finite[index] = np.sum(difference * seed) / 2e-5
        np.testing.assert_allclose(actual, finite, rtol=2e-5, atol=2e-6)
