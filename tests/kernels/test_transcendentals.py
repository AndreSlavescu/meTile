import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def _transcendental(
    source, cotangent, destination, gradient, size, *, OP: metile.constexpr, BLOCK: metile.constexpr
):
    inputs = metile.tensor(source, shape=(size,), access="read")
    seeds = metile.tensor(cotangent, shape=(size,), access="read")
    outputs = metile.tensor(destination, shape=(size,), access="write")
    gradients = metile.tensor(gradient, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    values = inputs.load((positions,))
    seed = seeds.load((positions,))
    if OP == "exp2":
        result = metile.exp2(values)
    elif OP == "fast_exp2":
        result = metile.fast_exp2(values)
    elif OP == "fast_cos":
        result = metile.fast_cos(values)
    elif OP == "fast_sin":
        result = metile.fast_sin(values)
    else:
        raise ValueError("unsupported test operation")
    outputs.store((positions,), result)
    gradients.store((positions,), metile.vjp(result, values, seed))


@pytest.mark.parametrize(
    "name,intrinsic",
    [
        ("exp2", "exp2"),
        ("fast_exp2", "fast::exp2"),
        ("fast_cos", "fast::cos"),
        ("fast_sin", "fast::sin"),
    ],
)
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_transcendental_and_vjp_lower_with_explicit_intrinsics_without_device(
    name, intrinsic, dtype
):
    context = TracingContext("transcendental")
    parameters = [
        (name, PtrType(dtype)) for name in ("source", "cotangent", "destination", "gradient")
    ]
    parameters.append(("size", I32))
    context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
    proxies = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
    with context:
        _transcendental.fn(*proxies, OP=name, BLOCK=64)
    source = emit(lower(context.func))
    assert intrinsic + "(" in source
    assert name in metile.__all__


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "name,reference,derivative",
    [
        ("exp2", np.exp2, lambda values: np.log(2.0) * np.exp2(values)),
        ("fast_exp2", np.exp2, lambda values: np.log(2.0) * np.exp2(values)),
        ("fast_cos", np.cos, lambda values: -np.sin(values)),
        ("fast_sin", np.sin, np.cos),
    ],
)
def test_gpu_transcendentals_and_vjp_match_real_functions(dtype, name, reference, derivative):
    generator = np.random.default_rng(395)
    values = generator.uniform(-3.0, 3.0, 137).astype(dtype)
    seed = generator.uniform(-1.0, 1.0, values.size).astype(dtype)
    output = np.full_like(values, np.nan)
    gradient = np.full_like(values, np.nan)
    _transcendental[(metile.cdiv(values.size, 64),)].prepare(
        values, seed, output, gradient, values.size, OP=name, BLOCK=64, STRICT_MATH=True
    )
    tolerance = 3e-3 if dtype == np.float16 else 3e-6
    np.testing.assert_allclose(
        output, reference(values.astype(np.float64)), rtol=tolerance, atol=1e-6
    )
    np.testing.assert_allclose(
        gradient,
        seed.astype(np.float64) * derivative(values.astype(np.float64)),
        rtol=tolerance,
        atol=1e-6,
    )
