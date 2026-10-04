import sys
from dataclasses import fields

import numpy as np
import pytest

from metile.backends.gated_delta import (
    gated_delta_backward,
    gated_delta_forward,
    gated_delta_reference,
    gated_delta_reference_backward,
)

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


def _inputs(channel, key_dim, value_dim, sequence=5):
    random = np.random.default_rng(214)
    prefix = (2, sequence, 2)
    query = random.normal(0, 0.2, (*prefix, key_dim)).astype(np.float32)
    key = random.normal(0, 0.2, query.shape).astype(np.float32)
    value = random.normal(0, 0.4, (*prefix, value_dim)).astype(np.float32)
    log_decay = random.uniform(-0.7, -0.1, query.shape if channel else prefix).astype(np.float32)
    beta = random.uniform(0.1, 0.8, prefix).astype(np.float32)
    state = random.normal(0, 0.2, (2, 2, key_dim, value_dim)).astype(np.float32)
    return query, key, value, log_decay, beta, state


@pytest.mark.parametrize("channel", [False, True])
@pytest.mark.parametrize("sequence", [1, 5])
@pytest.mark.parametrize(
    "key_dim,value_dim", [(1, 1), (3, 5), (32, 64), (65, 33), (128, 128), (256, 3), (3, 256)]
)
def test_native_forward_backward_matches_fp64_reference(channel, key_dim, value_dim, sequence):
    arguments = _inputs(channel, key_dim, value_dim, sequence=sequence)
    snapshots = [array.copy() for array in arguments]
    reference_arguments = tuple(array.astype(np.float64) for array in arguments)
    reference = gated_delta_reference(*reference_arguments, scale=0.7)
    actual = gated_delta_forward(*arguments, scale=0.7, save_states=True)
    np.testing.assert_allclose(actual.output.numpy(), reference.output, rtol=3e-4, atol=2e-5)
    np.testing.assert_allclose(
        actual.final_state.numpy(), reference.final_state, rtol=3e-4, atol=2e-5
    )
    np.testing.assert_allclose(actual.states.numpy(), reference.states, rtol=3e-4, atol=2e-5)
    random = np.random.default_rng(215)
    output_gradient = random.normal(0, 0.2, reference.output.shape).astype(np.float32)
    final_gradient = random.normal(0, 0.2, reference.final_state.shape).astype(np.float32)
    expected_vjp = gated_delta_reference_backward(
        *reference_arguments[:5],
        reference.states,
        output_gradient.astype(np.float64),
        final_gradient.astype(np.float64),
        scale=0.7,
    )
    actual_vjp = gated_delta_backward(
        *arguments[:5],
        actual.states,
        output_gradient,
        final_gradient,
        scale=0.7,
    )
    for field in fields(expected_vjp):
        np.testing.assert_allclose(
            getattr(actual_vjp, field.name).numpy(),
            getattr(expected_vjp, field.name),
            rtol=5e-4,
            atol=3e-5,
            err_msg=field.name,
        )
    for original, snapshot in zip(arguments, snapshots, strict=True):
        np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("channel", [False, True])
def test_native_streaming_without_saved_states(channel):
    arguments = _inputs(channel, 5, 3)
    complete = gated_delta_forward(*arguments, scale=0.4)
    left = gated_delta_forward(
        *(np.ascontiguousarray(array[:, :2]) for array in arguments[:5]), arguments[5], scale=0.4
    )
    right = gated_delta_forward(
        *(np.ascontiguousarray(array[:, 2:]) for array in arguments[:5]),
        left.final_state,
        scale=0.4,
    )
    assert complete.states is None
    np.testing.assert_allclose(
        np.concatenate((left.output.numpy(), right.output.numpy()), axis=1),
        complete.output.numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        right.final_state.numpy(), complete.final_state.numpy(), rtol=2e-5, atol=2e-6
    )
