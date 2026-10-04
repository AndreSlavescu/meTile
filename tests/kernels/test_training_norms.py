import inspect

import numpy as np
import pytest

import metile
from metile.backends import training_norms as norms
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile_kernels.training_norms import (
    norm_backward_rows_kernel,
    norm_forward_kernel,
    norm_parameter_reduce_kernel,
)

KINDS = ("rms", "layer", "add_rms")


def _reference(source, weight, bias, residual, seed, residual_seed, kind, epsilon):
    source, weight, bias, residual, seed, residual_seed = (
        array.astype(np.float64) for array in (source, weight, bias, residual, seed, residual_seed)
    )
    values = source + residual if kind == "add_rms" else source
    centered = values - values.mean(axis=-1, keepdims=True) if kind == "layer" else values
    inverse = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + epsilon)
    normalized = centered * inverse
    output = normalized * weight
    weighted = seed * weight
    source_gradient = weighted - normalized * np.mean(weighted * normalized, axis=-1, keepdims=True)
    if kind == "layer":
        output += bias
        source_gradient -= weighted.mean(axis=-1, keepdims=True)
    source_gradient *= inverse
    if kind == "add_rms":
        source_gradient += residual_seed
    return {
        "output": output,
        "residual_output": values,
        "source": source_gradient,
        "residual": source_gradient if kind == "add_rms" else None,
        "weight": (seed * normalized).sum(axis=0),
        "bias": seed.sum(axis=0) if kind == "layer" else None,
    }


def _trace(kernel, kind, dtype, block):
    options = norms._options(block, kind)
    options.pop("STRICT_MATH")
    if kernel is norm_parameter_reduce_kernel:
        options = {"HAS_BIAS": kind == "layer", "BLOCK": 128}
    context = TracingContext(kernel.name)
    context.func.constexprs = {**options, "STRICT_MATH": True}
    storage_names = {"source", "residual", "weight", "bias", "output", "residual_output"}
    output_names = {
        "output",
        "residual_output",
        "source_gradient",
        "residual_gradient",
        "weight_gradient",
        "bias_gradient",
    }
    if kernel is norm_forward_kernel:
        output_names |= {"means", "inverse_scales"}
    elif kernel is norm_backward_rows_kernel:
        output_names |= {"weight_partials", "bias_partials"}
    with context:
        arguments = []
        for name, parameter in inspect.signature(kernel.fn).parameters.items():
            if parameter.annotation is metile.constexpr:
                continue
            if name in {"rows", "columns"}:
                datatype = I32
            elif name == "epsilon":
                datatype = ScalarType("f32")
            else:
                datatype = PtrType(dtype if name in storage_names else "f32")
            context.func.params.append(tir.Param(name, datatype, is_output=name in output_names))
            arguments.append(TracingProxy(tir.Value(name, datatype)))
        kernel.fn(*arguments, **options)
    return lower(context.func)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("block", [64, 512, 8192])
@pytest.mark.parametrize("kernel", [norm_forward_kernel, norm_backward_rows_kernel])
def test_normalization_rows_lower_without_device(kind, dtype, block, kernel):
    function = _trace(kernel, kind, dtype, block)
    source = emit(function)
    assert function.threadgroup_size[0] <= 256
    assert "simd_sum(" in source
    assert "atomic_" not in source
    assert "fast::" not in source


@pytest.mark.parametrize("kind", KINDS)
def test_parameter_reduction_lowers_to_deterministic_row_loop(kind):
    source = emit(_trace(norm_parameter_reduce_kernel, kind, "f32", 128))
    assert "for (" in source
    assert "atomic_" not in source
    assert "simd_sum(" not in source


@pytest.mark.parametrize(
    "source,weight,epsilon,error,match",
    [
        (np.zeros((0, 4), np.float32), np.ones(4, np.float32), 1e-5, ValueError, "rows"),
        (np.zeros((2, 0), np.float32), np.ones(0, np.float32), 1e-5, ValueError, "columns"),
        (np.zeros((2, 8193), np.float32), np.ones(8193, np.float32), 1e-5, ValueError, "columns"),
        (np.zeros((4,), np.float32), np.ones(4, np.float32), 1e-5, ValueError, "rows"),
        (
            np.zeros((2, 4), np.float64),
            np.ones(4, np.float32),
            1e-5,
            TypeError,
            "float16 or float32",
        ),
        (np.zeros((2, 4), np.float32), np.ones(3, np.float32), 1e-5, ValueError, "weight"),
        (np.zeros((2, 4), np.float32), np.ones(4, np.float32), 0.0, ValueError, "epsilon"),
        (np.zeros((2, 4), np.float32), np.ones(4, np.float32), np.nan, ValueError, "epsilon"),
        (np.zeros((2, 4), np.float32), np.ones(4, np.float32), True, ValueError, "epsilon"),
        (np.zeros((2, 4), np.float32), np.ones(4, np.float32), 1e-50, ValueError, "epsilon"),
    ],
)
def test_invalid_normalization_inputs_fail_before_device_creation(
    source, weight, epsilon, error, match
):
    with pytest.raises(error, match=match):
        norms.rms_norm_forward(source, weight, epsilon=epsilon)


def test_residual_and_backward_contract_validation_without_device():
    source = np.zeros((2, 4), np.float32)
    weight = np.ones(4, np.float32)
    with pytest.raises(TypeError, match="matching"):
        norms.add_rms_norm_forward(source, source.astype(np.float16), weight)
    with pytest.raises(ValueError, match="bias"):
        norms.layer_norm_forward(source, weight, np.zeros(3, np.float32))
    with pytest.raises(TypeError, match="NormContext"):
        norms.norm_backward(None, source)
    context = norms.NormContext("rms", source, weight, None, None, None)
    with pytest.raises(ValueError, match="output_gradient"):
        norms.norm_backward(context, source[:, :2])
    with pytest.raises(ValueError, match="requires add-RMSNorm"):
        norms.norm_backward(context, source, residual_output_gradient=source)


def _run(source, weight, bias, residual, seed, residual_seed, kind, epsilon):
    if kind == "rms":
        output, context = norms.rms_norm_forward(source, weight, epsilon=epsilon)
        residual_output = None
    elif kind == "layer":
        output, context = norms.layer_norm_forward(source, weight, bias, epsilon=epsilon)
        residual_output = None
    else:
        output, residual_output, context = norms.add_rms_norm_forward(
            source, residual, weight, epsilon=epsilon
        )
    gradients = norms.norm_backward(
        context, seed, residual_output_gradient=residual_seed if kind == "add_rms" else None
    )
    return output, residual_output, context, gradients


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("columns", [1, 13, 65, 513, 8192])
def test_gpu_training_normalization_forward_and_all_gradients(kind, dtype, columns):
    generator = np.random.default_rng(columns)
    shape = (3, columns)
    source, residual = (generator.normal(size=shape).astype(dtype) for _ in range(2))
    weight = generator.uniform(0.5, 1.5, columns).astype(dtype)
    bias = generator.normal(size=columns).astype(dtype)
    seed, residual_seed = (generator.normal(size=shape).astype(np.float32) for _ in range(2))
    epsilon = 2e-3
    output, residual_output, context, gradients = _run(
        source, weight, bias, residual, seed, residual_seed, kind, epsilon
    )
    expected = _reference(source, weight, bias, residual, seed, residual_seed, kind, epsilon)
    np.testing.assert_allclose(
        output.numpy(),
        expected["output"],
        rtol=1e-3 if dtype == np.float16 else 3e-5,
        atol=1e-3 if dtype == np.float16 else 5e-6,
    )
    for name in ("source", "weight", "bias", "residual"):
        actual = getattr(gradients, name)
        if expected[name] is None:
            assert actual is None
        else:
            assert actual.dtype == np.dtype(np.float32)
            np.testing.assert_allclose(actual.numpy(), expected[name], rtol=2e-4, atol=1e-5)
    if residual_output is not None:
        np.testing.assert_array_equal(
            residual_output.numpy(), expected["residual_output"].astype(dtype)
        )
    repeated = norms.norm_backward(
        context, seed, residual_output_gradient=residual_seed if kind == "add_rms" else None
    )
    np.testing.assert_array_equal(repeated.weight.numpy(), gradients.weight.numpy())
    if kind == "layer":
        np.testing.assert_array_equal(repeated.bias.numpy(), gradients.bias.numpy())


@pytest.mark.parametrize("kind", KINDS)
def test_gpu_training_normalization_gradients_match_finite_differences(kind):
    generator = np.random.default_rng(171)
    source, residual = (generator.normal(size=(2, 5)).astype(np.float32) for _ in range(2))
    weight = generator.uniform(0.5, 1.5, 5).astype(np.float32)
    bias = generator.normal(size=5).astype(np.float32)
    seed, residual_seed = (generator.normal(size=(2, 5)).astype(np.float32) for _ in range(2))
    epsilon = 0.003
    _, _, _, gradients = _run(source, weight, bias, residual, seed, residual_seed, kind, epsilon)
    arrays = dict(source=source, weight=weight, bias=bias, residual=residual)
    names = ["source", "weight"] + (["bias"] if kind == "layer" else [])
    if kind == "add_rms":
        names.append("residual")
    for name in names:
        finite_difference = np.empty_like(arrays[name], dtype=np.float64)
        for position in np.ndindex(finite_difference.shape):
            objectives = []
            for direction in (1, -1):
                candidate = arrays[name].astype(np.float64)
                candidate[position] += direction * 1e-4
                changed = {**arrays, name: candidate}
                reference = _reference(
                    **changed, seed=seed, residual_seed=residual_seed, kind=kind, epsilon=epsilon
                )
                objective = np.sum(reference["output"] * seed)
                if kind == "add_rms":
                    objective += np.sum(reference["residual_output"] * residual_seed)
                objectives.append(objective)
            finite_difference[position] = (objectives[0] - objectives[1]) / 2e-4
        np.testing.assert_allclose(
            getattr(gradients, name).numpy(), finite_difference, rtol=2e-4, atol=3e-6
        )


def test_gpu_layer_norm_centered_variance_and_optional_residual_seed():
    source = np.array([[4095.75, 4096.0, 4096.25]], dtype=np.float32)
    weight = np.ones(3, np.float32)
    output, context = norms.layer_norm_forward(source, weight, np.zeros(3, np.float32))
    centered = source.astype(np.float64) - 4096.0
    expected = centered / np.sqrt(np.mean(centered * centered) + 1e-5)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(
        norms.norm_backward(context, np.ones_like(source)).source.numpy(), 0.0, atol=1e-6
    )
    _, _, context = norms.add_rms_norm_forward(source, -source + 1, weight)
    gradients = norms.norm_backward(context, np.ones_like(source))
    np.testing.assert_array_equal(gradients.source.numpy(), gradients.residual.numpy())
    assert np.isfinite(gradients.weight.numpy()).all()
