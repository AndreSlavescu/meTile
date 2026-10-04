import inspect
from dataclasses import astuple

import numpy as np
import pytest

from metile.backends.gated_delta import (
    gated_delta_backward,
    gated_delta_forward,
    gated_delta_reference,
    gated_delta_reference_backward,
)
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType
from metile_kernels.gated_delta import (
    gated_delta_backward_kernel,
    gated_delta_forward_kernel,
    gated_delta_reduce_gradients_kernel,
)


def _inputs(channel, *, batch=1, sequence=3, heads=2, key_dim=3, value_dim=2, dtype=np.float64):
    random = np.random.default_rng(174)
    prefix = (batch, sequence, heads)
    shapes = [(*prefix, key_dim), (*prefix, key_dim), (*prefix, value_dim)]
    arrays = [random.normal(0, 0.3, shape).astype(dtype) for shape in shapes]
    decay = random.uniform(-0.7, -0.1, (*prefix, key_dim) if channel else prefix).astype(dtype)
    beta = random.uniform(0.1, 0.8, prefix).astype(dtype)
    state = random.normal(0, 0.3, (batch, heads, key_dim, value_dim)).astype(dtype)
    return (*arrays, decay, beta, state)


@pytest.mark.parametrize("channel", [False, True])
def test_reference_vjp_matches_all_input_finite_differences(channel):
    arguments = _inputs(channel)
    result = gated_delta_reference(*arguments, scale=0.7)
    random = np.random.default_rng(175)
    output_gradient = random.normal(size=result.output.shape)
    final_gradient = random.normal(size=result.final_state.shape)
    gradients = gated_delta_reference_backward(
        *arguments[:5], result.states, output_gradient, final_gradient, scale=0.7
    )

    def objective():
        forward = gated_delta_reference(*arguments, scale=0.7)
        return np.sum(forward.output * output_gradient) + np.sum(
            forward.final_state * final_gradient
        )

    epsilon = 1e-6
    for argument, gradient in zip(arguments, astuple(gradients), strict=True):
        numerical = np.empty_like(argument)
        for index in np.ndindex(argument.shape):
            original = argument[index]
            argument[index] = original + epsilon
            plus = objective()
            argument[index] = original - epsilon
            minus = objective()
            argument[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("channel", [False, True])
def test_streaming_forward_and_vjp_compose(channel):
    arguments = _inputs(channel, batch=2, sequence=5)
    complete = gated_delta_reference(*arguments, scale=0.4)
    left_inputs = tuple(np.ascontiguousarray(array[:, :2]) for array in arguments[:5])
    right_inputs = tuple(np.ascontiguousarray(array[:, 2:]) for array in arguments[:5])
    left = gated_delta_reference(*left_inputs, arguments[5], scale=0.4)
    right = gated_delta_reference(*right_inputs, left.final_state, scale=0.4)
    np.testing.assert_array_equal(
        np.concatenate((left.output, right.output), axis=1), complete.output
    )
    np.testing.assert_array_equal(right.final_state, complete.final_state)
    random = np.random.default_rng(176)
    output_gradient = random.normal(size=complete.output.shape)
    final_gradient = random.normal(size=complete.final_state.shape)
    complete_vjp = gated_delta_reference_backward(
        *arguments[:5], complete.states, output_gradient, final_gradient, scale=0.4
    )
    right_vjp = gated_delta_reference_backward(
        *right_inputs,
        right.states,
        np.ascontiguousarray(output_gradient[:, 2:]),
        final_gradient,
        scale=0.4,
    )
    left_vjp = gated_delta_reference_backward(
        *left_inputs,
        left.states,
        np.ascontiguousarray(output_gradient[:, :2]),
        right_vjp.initial_state,
        scale=0.4,
    )
    for expected, left_gradient, right_gradient in zip(
        astuple(complete_vjp)[:5], astuple(left_vjp)[:5], astuple(right_vjp)[:5], strict=True
    ):
        np.testing.assert_allclose(
            np.concatenate((left_gradient, right_gradient), axis=1), expected
        )
    np.testing.assert_allclose(left_vjp.initial_state, complete_vjp.initial_state)


def test_scalar_decay_is_channel_decay_special_case_including_vjp():
    arguments = _inputs(False)
    channel_decay = np.repeat(arguments[3][..., None], arguments[0].shape[-1], axis=-1)
    channel_arguments = (*arguments[:3], channel_decay, *arguments[4:])
    scalar = gated_delta_reference(*arguments)
    channel = gated_delta_reference(*channel_arguments)
    np.testing.assert_array_equal(scalar.output, channel.output)
    np.testing.assert_array_equal(scalar.states, channel.states)
    output_gradient = np.ones_like(scalar.output)
    final_gradient = np.ones_like(scalar.final_state)
    scalar_vjp = gated_delta_reference_backward(
        *arguments[:5], scalar.states, output_gradient, final_gradient
    )
    channel_vjp = gated_delta_reference_backward(
        *channel_arguments[:5], channel.states, output_gradient, final_gradient
    )
    np.testing.assert_allclose(scalar_vjp.log_decay, channel_vjp.log_decay.sum(axis=-1))


def test_decay_precedes_delta_correction_and_reads_updated_state():
    query = np.array([[[[1.0, 2.0]]]])
    key = np.array([[[[0.5, 1.0]]]])
    value = np.array([[[[3.0]]]])
    decay = np.log(np.array([[[[0.2, 0.8]]]]))
    beta = np.array([[[0.7]]])
    initial = np.array([[[[1.0], [2.0]]]])
    result = gated_delta_reference(query, key, value, decay, beta, initial)
    expected_state = np.array([[[[0.655], [2.51]]]])
    np.testing.assert_allclose(result.final_state, expected_state)
    np.testing.assert_allclose(result.output, [[[[5.675]]]])
    np.testing.assert_array_equal(initial, [[[[1.0], [2.0]]]])


def test_no_update_and_zero_scale_keep_final_state_gradient():
    arguments = list(_inputs(True))
    arguments[3].fill(0)
    arguments[4].fill(0)
    result = gated_delta_reference(*arguments, scale=0.0)
    np.testing.assert_array_equal(result.final_state, arguments[5])
    np.testing.assert_array_equal(result.output, np.zeros_like(result.output))
    final_gradient = np.ones_like(result.final_state)
    gradients = gated_delta_reference_backward(
        *arguments[:5], result.states, np.ones_like(result.output), final_gradient, scale=0.0
    )
    np.testing.assert_array_equal(gradients.initial_state, final_gradient)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 1e50])
def test_native_scale_validation_precedes_device_allocation(scale):
    with pytest.raises(ValueError, match="scale"):
        gated_delta_forward(*_inputs(False, dtype=np.float32), scale=scale)


@pytest.mark.parametrize("channel", [False, True])
def test_workspace_budget_precedes_device_allocation(channel):
    arguments = _inputs(channel, dtype=np.float32)
    with pytest.raises(ValueError, match="budget"):
        gated_delta_forward(*arguments, save_states=True, max_workspace_bytes=1)
    result = gated_delta_reference(*arguments)
    with pytest.raises(ValueError, match="budget"):
        gated_delta_backward(
            *arguments[:5], result.states, result.output, result.final_state, max_workspace_bytes=1
        )


def test_native_rejects_wrong_shapes_precision_and_noncontiguous_inputs():
    with pytest.raises(TypeError, match="FP32"):
        gated_delta_forward(*_inputs(False))
    arguments = list(_inputs(False, dtype=np.float32))
    arguments[1] = arguments[1][..., :1].copy()
    with pytest.raises(ValueError, match="key expects shape"):
        gated_delta_forward(*arguments)
    arguments = list(_inputs(False, dtype=np.float32))
    arguments[0] = arguments[0][..., ::-1]
    with pytest.raises(ValueError, match="contiguous"):
        gated_delta_forward(*arguments)
    with pytest.raises(ValueError, match="K,V <= 256"):
        gated_delta_forward(*_inputs(False, key_dim=257, dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        gated_delta_forward(*_inputs(False, sequence=0, dtype=np.float32))
    with pytest.raises(ValueError, match="T <= 4096"):
        gated_delta_forward(*_inputs(False, sequence=4097, dtype=np.float32))
    with pytest.raises(ValueError, match="B <= 64"):
        gated_delta_forward(*_inputs(False, batch=65, dtype=np.float32))


@pytest.mark.parametrize("channel", [False, True])
@pytest.mark.parametrize(
    "kernel",
    [gated_delta_forward_kernel, gated_delta_backward_kernel, gated_delta_reduce_gradients_kernel],
)
def test_recurrent_kernels_trace_and_lower_without_gpu(kernel, channel):
    constants = dict(
        BATCH=2,
        ROWS=12,
        HEADS=2,
        KEY_DIM=5,
        VALUE_DIM=3,
        CHANNEL_DECAY=channel,
        SAVE_STATES=True,
        BLOCK=32,
    )
    arguments = []
    keywords = {}
    with TracingContext(kernel.name) as context:
        for name in inspect.signature(kernel.fn).parameters:
            if name in constants:
                keywords[name] = constants[name]
            else:
                dtype = (
                    ScalarType("i32")
                    if name == "Sequence"
                    else ScalarType("f32")
                    if name == "scale"
                    else PtrType("f32")
                )
                context.func.params.append(tir.Param(name, dtype))
                arguments.append(TracingProxy(tir.Value(name, dtype)))
        context.func.constexprs = {**keywords, "STRICT_MATH": True}
        kernel.fn(*arguments, **keywords)
    operations = []
    pending = list(context.func.ops)
    while pending:
        operation = pending.pop()
        operations.append(operation)
        pending.extend(getattr(operation, "body", ()))
    assert any(isinstance(operation, tir.LoopState) for operation in operations)
    assert any(isinstance(operation, tir.AssignLoopState) for operation in operations)
    assert all(
        operation.tensor is not None
        for operation in operations
        if isinstance(operation, (tir.Load, tir.Store))
    )
    source = emit(lower(context.func))
    assert "[[kernel]]" in source
    assert "atomic" not in source
    assert "for (" in source
