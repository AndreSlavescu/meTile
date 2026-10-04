import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.training_activations import activation_backward, activation_forward

KINDS = ("silu", "sigmoid", "gelu_tanh", "quick_gelu", "relu", "tanh")


def _reference(values, kind):
    if kind == "relu":
        return np.maximum(values, 0)
    if kind == "tanh":
        return np.tanh(values)
    if kind == "gelu_tanh":
        return 0.5 * values * (1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values**3)))
    slope = 1.702 if kind == "quick_gelu" else 1.0
    sigmoid = 1 / (1 + np.exp(-slope * values))
    return sigmoid if kind == "sigmoid" else values * sigmoid


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("kernel,pointers", [(activation_forward, 3), (activation_backward, 5)])
def test_training_activation_lowers_without_device(kind, kernel, pointers):
    with TracingContext("activation") as context:
        parameters = [(f"pointer{index}", PtrType("f32")) for index in range(pointers)]
        parameters.append(("size", I32))
        context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
        proxies = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
        kernel.fn(*proxies, KIND=kind, GATED=True, BLOCK=128)
    assert "[[kernel" in emit(lower(context.func))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_gpu_activation_forward_and_both_gradients(kind, gated, dtype):
    generator = np.random.default_rng(48)
    values = generator.uniform(-4, 4, 139).astype(dtype)
    ups = generator.normal(size=values.shape).astype(dtype)
    seed = generator.normal(size=values.shape).astype(np.float32)
    output = np.full_like(values, np.nan)
    input_gradient = np.full_like(seed, np.nan)
    up_gradient = np.full_like(seed, np.nan)
    options = dict(KIND=kind, GATED=gated, BLOCK=128, STRICT_MATH=True)
    activation_forward[(2,)].prepare(values, ups, output, values.size, **options)
    activation_backward[(2,)].prepare(
        values, ups, seed, input_gradient, up_gradient, values.size, **options
    )
    values64 = values.astype(np.float64)
    expected = _reference(values64, kind)
    step = 1e-5
    derivative = (_reference(values64 + step, kind) - _reference(values64 - step, kind)) / (
        2 * step
    )
    np.testing.assert_allclose(
        output,
        expected * ups if gated else expected,
        rtol=2e-3 if dtype == np.float16 else 2e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        input_gradient, seed * derivative * (ups if gated else 1), rtol=3e-4, atol=2e-6
    )
    if gated:
        np.testing.assert_allclose(up_gradient, seed * expected, rtol=2e-5, atol=2e-6)
    else:
        assert np.isnan(up_gradient).all()


@pytest.mark.parametrize("kind", ["silu", "sigmoid", "gelu_tanh", "quick_gelu", "relu", "tanh"])
def test_gpu_activation_extremes_and_zero_are_finite(kind):
    values = np.array([-3e38, -100, -0.0, 0.0, 100, 3e38], dtype=np.float32)
    gradient = np.empty_like(values)
    unused = np.zeros_like(values)
    activation_backward[(1,)].prepare(
        values,
        unused,
        np.ones_like(values),
        gradient,
        unused,
        values.size,
        KIND=kind,
        BLOCK=32,
        STRICT_MATH=True,
    )
    assert np.isfinite(gradient).all()
    expected_zero = {
        "silu": 0.5,
        "sigmoid": 0.25,
        "gelu_tanh": 0.5,
        "quick_gelu": 0.5,
        "relu": 0.0,
        "tanh": 1.0,
    }[kind]
    np.testing.assert_array_equal(gradient[2:4], expected_zero)
