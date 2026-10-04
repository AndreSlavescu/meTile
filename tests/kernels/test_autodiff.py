import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def _mixed_vjp(source, cotangent, destination, size, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(size,), access="read")
    seeds = metile.tensor(cotangent, shape=(size,), access="read")
    gradients = metile.tensor(destination, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    values = inputs.load((positions,))
    seed = seeds.load((positions,))
    squared = values * values
    output = (
        metile.tanh(values) * metile.exp(values * 0.2) + metile.log(squared + 1.0)
    ) / metile.sqrt(squared + 1.5)
    gradients.store((positions,), metile.vjp(output, values, seed))


@metile.kernel
def _softmax_vjp(source, cotangent, destination, rows, columns, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    seeds = metile.tensor(cotangent, shape=(rows, columns), access="read")
    gradients = metile.tensor(destination, shape=(rows, columns), access="write")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK)
    values = inputs.load((row, positions))
    seed = seeds.load((row, positions))
    logits = metile.where(positions < columns, values, -1e9)
    numerator = metile.exp(logits - metile.max(logits))
    probabilities = numerator / metile.sum(numerator)
    gradients.store((row, positions), metile.vjp(probabilities, values, seed))


@metile.kernel
def _row_loss_vjp(
    source, bias, source_gradient, bias_gradient, rows, columns, BLOCK: metile.constexpr
):
    inputs = metile.tensor(source, shape=(rows, columns), access="read")
    biases = metile.tensor(bias, shape=(rows,), access="read")
    source_gradients = metile.tensor(source_gradient, shape=(rows, columns), access="write")
    bias_gradients = metile.tensor(bias_gradient, shape=(rows, 1), access="write")
    row = metile.program_id(0)
    positions = metile.arange(0, BLOCK)
    values = inputs.load((row, positions))
    offset = biases.load((row,))
    shifted = metile.where(positions < columns, values + offset, 0.0)
    loss = metile.sum(shifted * shifted) / columns
    values_gradient, offset_gradient = metile.vjp(loss, (values, offset), 1.0)
    source_gradients.store((row, positions), values_gradient)
    bias_gradients.store((row, positions), offset_gradient)


def _finite_difference(objective, values, step=1e-4):
    values = np.asarray(values, dtype=np.float64)
    gradient = np.empty_like(values)
    for position in np.ndindex(values.shape):
        above = values.copy()
        below = values.copy()
        above[position] += step
        below[position] -= step
        gradient[position] = (objective(above) - objective(below)) / (2 * step)
    return gradient


@pytest.mark.parametrize(
    ("kernel", "pointer_names", "scalar_names"),
    [
        (_mixed_vjp, ("source", "cotangent", "destination"), ("size",)),
        (_softmax_vjp, ("source", "cotangent", "destination"), ("rows", "columns")),
        (
            _row_loss_vjp,
            ("source", "bias", "source_gradient", "bias_gradient"),
            ("rows", "columns"),
        ),
    ],
)
def test_expression_vjp_kernels_lower_without_device(kernel, pointer_names, scalar_names):
    context = TracingContext(kernel.fn.__name__)
    parameters = [(name, PtrType("f32")) for name in pointer_names]
    parameters.extend((name, I32) for name in scalar_names)
    context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
    proxies = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
    with context:
        kernel.fn(*proxies, BLOCK=64)
    source = emit(lower(context.func))
    assert "[[kernel" in source
    assert "vjp" not in source.split("{", 1)[1]


@pytest.mark.parametrize("size", [17, 137])
def test_gpu_mixed_expression_vjp_matches_finite_differences(size):
    generator = np.random.default_rng(size)
    values = generator.uniform(-1.5, 1.5, size).astype(np.float32)
    seed = generator.standard_normal(size).astype(np.float32)
    output = np.full_like(values, np.nan)
    _mixed_vjp[(metile.cdiv(size, 64),)].prepare(
        values, seed, output, size, BLOCK=64, STRICT_MATH=True
    )

    def objective(candidate):
        result = (
            np.tanh(candidate) * np.exp(candidate * 0.2) + np.log(candidate * candidate + 1.0)
        ) / np.sqrt(candidate * candidate + 1.5)
        return np.sum(result * seed)

    expected = _finite_difference(objective, values)
    np.testing.assert_allclose(output, expected, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("columns", [7, 37, 128])
def test_gpu_masked_softmax_vjp_matches_finite_differences(columns):
    generator = np.random.default_rng(columns)
    rows = 2
    values = generator.standard_normal((rows, columns)).astype(np.float32)
    seed = generator.standard_normal(values.shape).astype(np.float32)
    output = np.full_like(values, np.nan)
    _softmax_vjp[(rows,)].prepare(
        values,
        seed,
        output,
        rows,
        columns,
        BLOCK=max(32, metile.next_power_of_2(columns)),
        STRICT_MATH=True,
    )

    def objective(candidate):
        numerator = np.exp(candidate - candidate.max(axis=-1, keepdims=True))
        return np.sum(numerator / numerator.sum(axis=-1, keepdims=True) * seed)

    expected = _finite_difference(objective, values)
    np.testing.assert_allclose(output, expected, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(output.sum(axis=-1), 0.0, atol=2e-6)


@pytest.mark.parametrize("columns", [13, 64])
def test_gpu_row_reduction_and_scalar_broadcast_vjp_matches_finite_differences(columns):
    generator = np.random.default_rng(columns)
    rows = 3
    values = generator.standard_normal((rows, columns)).astype(np.float32)
    bias = generator.standard_normal(rows).astype(np.float32)
    values_gradient = np.full_like(values, np.nan)
    bias_gradient = np.full((rows, 1), np.nan, dtype=np.float32)
    _row_loss_vjp[(rows,)].prepare(
        values,
        bias,
        values_gradient,
        bias_gradient,
        rows,
        columns,
        BLOCK=max(32, metile.next_power_of_2(columns)),
        STRICT_MATH=True,
    )
    expected_values = _finite_difference(
        lambda candidate: np.sum(np.mean((candidate + bias[:, None]) ** 2, axis=-1)), values
    )
    expected_bias = _finite_difference(
        lambda candidate: np.sum(
            np.mean((values.astype(np.float64) + candidate[:, None]) ** 2, axis=-1)
        ),
        bias,
    )
    np.testing.assert_allclose(values_gradient, expected_values, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(bias_gradient[:, 0], expected_bias, rtol=3e-4, atol=3e-6)
