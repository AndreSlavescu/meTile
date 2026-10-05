import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def _reciprocal_square_root(source, destination, size, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(size,), access="read")
    outputs = metile.tensor(destination, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    values = inputs.load((positions,))
    outputs.store((positions,), metile.rsqrt(values))


@metile.kernel
def _reciprocal_square_root_vjp(source, cotangent, destination, size, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(size,), access="read")
    seeds = metile.tensor(cotangent, shape=(size,), access="read")
    gradients = metile.tensor(destination, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    values = inputs.load((positions,))
    seed = seeds.load((positions,))
    gradients.store((positions,), metile.vjp(metile.rsqrt(values), values, seed))


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_reciprocal_square_root_lowers_to_precise_metal_intrinsic_without_device(dtype):
    context = TracingContext("reciprocal_square_root")
    parameters = [("source", PtrType(dtype)), ("destination", PtrType(dtype)), ("size", I32)]
    context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
    proxies = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
    with context:
        _reciprocal_square_root.fn(*proxies, BLOCK=64)
    source = emit(lower(context.func))
    assert "precise::rsqrt(" in source
    assert "fast::rsqrt" not in source
    assert " / sqrt(" not in source


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("size", [37, 257])
def test_gpu_reciprocal_square_root_matches_positive_real_function(dtype, size):
    values = np.geomspace(1e-3, 1e3, size).astype(dtype)
    output = np.full_like(values, np.nan)
    _reciprocal_square_root[(metile.cdiv(size, 64),)].prepare(
        values, output, size, BLOCK=64, STRICT_MATH=True
    )
    expected = (1 / np.sqrt(values.astype(np.float64))).astype(dtype)
    tolerance = 1e-3 if dtype == np.float16 else 2e-7
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_reciprocal_square_root_vjp_matches_finite_differences(dtype):
    generator = np.random.default_rng(47)
    values = generator.uniform(0.25, 4.0, 47).astype(dtype)
    seed = generator.uniform(-1.0, 1.0, 47).astype(dtype)
    output = np.full_like(values, np.nan)
    _reciprocal_square_root_vjp[(1,)].prepare(
        values, seed, output, len(values), BLOCK=64, STRICT_MATH=True
    )
    source = values.astype(np.float64)
    delta = 1e-5
    expected = (
        seed.astype(np.float64)
        * ((source + delta) ** -0.5 - (source - delta) ** -0.5)
        / (2 * delta)
    )
    tolerance = 3e-3 if dtype == np.float16 else 3e-6
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=1e-7)
