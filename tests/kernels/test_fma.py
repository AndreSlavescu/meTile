import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import fold_constants, vectorize_elementwise
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def _fused_multiply_add(
    left,
    right,
    addend,
    cotangent,
    destination,
    left_gradient,
    right_gradient,
    addend_gradient,
    size,
    *,
    BLOCK: metile.constexpr,
):
    left_inputs = metile.tensor(left, shape=(size,), access="read")
    right_inputs = metile.tensor(right, shape=(size,), access="read")
    addend_inputs = metile.tensor(addend, shape=(size,), access="read")
    seeds = metile.tensor(cotangent, shape=(size,), access="read")
    outputs = metile.tensor(destination, shape=(size,), access="write")
    left_gradients = metile.tensor(left_gradient, shape=(size,), access="write")
    right_gradients = metile.tensor(right_gradient, shape=(size,), access="write")
    addend_gradients = metile.tensor(addend_gradient, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    left_values = left_inputs.load((positions,))
    right_values = right_inputs.load((positions,))
    addend_values = addend_inputs.load((positions,))
    seed = seeds.load((positions,))
    result = metile.fma(left_values, right_values, addend_values)
    left_derivative, right_derivative, addend_derivative = metile.vjp(
        result, (left_values, right_values, addend_values), seed
    )
    outputs.store((positions,), result)
    left_gradients.store((positions,), left_derivative)
    right_gradients.store((positions,), right_derivative)
    addend_gradients.store((positions,), addend_derivative)


@metile.kernel
def _fused_cancellation(left, right, addend, fused, separated, size, *, BLOCK: metile.constexpr):
    left_inputs = metile.tensor(left, shape=(size,), access="read")
    right_inputs = metile.tensor(right, shape=(size,), access="read")
    addend_inputs = metile.tensor(addend, shape=(size,), access="read")
    fused_outputs = metile.tensor(fused, shape=(size,), access="write")
    separated_outputs = metile.tensor(separated, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    left_values = left_inputs.load((positions,))
    right_values = right_inputs.load((positions,))
    addend_values = addend_inputs.load((positions,))
    fused_outputs.store((positions,), metile.fma(left_values, right_values, addend_values))
    separated_outputs.store((positions,), left_values * right_values + addend_values)


@pytest.mark.parametrize("dtype,metal_type", [("f16", "half"), ("f32", "float")])
@pytest.mark.parametrize("vectorize", [False, True])
def test_fma_and_vjp_emit_explicit_metal_intrinsic_under_strict_math(dtype, metal_type, vectorize):
    names = (
        "left",
        "right",
        "addend",
        "cotangent",
        "destination",
        "left_gradient",
        "right_gradient",
        "addend_gradient",
    )
    parameters = [(name, PtrType(dtype)) for name in names] + [("size", I32)]
    context = TracingContext("fma_kernel")
    context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
    context.func.constexprs["STRICT_MATH"] = True
    proxies = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
    with context:
        _fused_multiply_add.fn(*proxies, BLOCK=64)
    lowered = fold_constants(lower(context.func))
    if vectorize:
        lowered = vectorize_elementwise(lowered)
    source = emit(lowered)
    assert "fma(" in source
    assert f"{metal_type}" in source
    assert "fast::fma(" not in source


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("size", [37, 257])
def test_gpu_fma_and_vjp_match_all_operand_derivatives(dtype, size):
    generator = np.random.default_rng(327)
    operands = [generator.uniform(-2.0, 2.0, size).astype(dtype) for _ in range(3)]
    seed = generator.uniform(-1.0, 1.0, size).astype(dtype)
    outputs = [np.full(size, np.nan, dtype=dtype) for _ in range(4)]
    _fused_multiply_add[(metile.cdiv(size, 64),)].prepare(
        *operands, seed, *outputs, size, BLOCK=64, STRICT_MATH=True
    )
    left, right, addend = [operand.astype(np.float64) for operand in operands]
    expected = (
        left * right + addend,
        seed.astype(np.float64) * right,
        seed.astype(np.float64) * left,
        seed,
    )
    for output, reference in zip(outputs, expected):
        np.testing.assert_array_equal(output, reference.astype(dtype))


@pytest.mark.parametrize("dtype,delta", [(np.float16, 2.0**-6), (np.float32, 2.0**-13)])
def test_gpu_explicit_fma_preserves_cancellation_that_separated_arithmetic_loses(dtype, delta):
    size = 37
    left = np.full(size, 1.0 + delta, dtype=dtype)
    right = np.full(size, 1.0 - delta, dtype=dtype)
    addend = np.full(size, -1.0, dtype=dtype)
    fused = np.full(size, np.nan, dtype=dtype)
    separated = np.full(size, np.nan, dtype=dtype)
    _fused_cancellation[(1,)].prepare(
        left, right, addend, fused, separated, size, BLOCK=64, STRICT_MATH=True
    )
    np.testing.assert_array_equal(fused, np.full(size, -(delta**2), dtype=dtype))
    np.testing.assert_array_equal(separated, np.zeros(size, dtype=dtype))
