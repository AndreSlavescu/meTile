import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import fold_constants
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, TileType


@metile.kernel
def _recurrence(source, output, snapshots, size, steps, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(size,), access="read")
    outputs = metile.tensor(output, shape=(size,), access="write")
    saved = metile.tensor(snapshots, shape=(size,), access="write")
    positions = metile.arange(0, BLOCK)
    state = metile.loop_state(inputs.load((positions,)))
    initial = state.value
    for iteration in metile.tile_range(0, steps, 1):
        before = state.value
        next_value = before * 0.5 + metile.cast(iteration, "f32")
        state.update(metile.where(positions < size, next_value, 0.0))
        updated = state.value
        state.update(updated + metile.sum(updated) / size)
    saved.store((positions,), initial)
    outputs.store((positions,), state.value)


def test_loaded_loop_state_emits_declared_mutation_and_snapshots():
    with TracingContext("explicit_state") as context:
        parameters = [
            ("source", PtrType("f32")),
            ("output", PtrType("f32")),
            ("snapshots", PtrType("f32")),
            ("size", I32),
            ("steps", I32),
        ]
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        proxies = [TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters]
        _recurrence.fn(*proxies, BLOCK=128)
    source = emit(fold_constants(lower(context.func)))
    assert "for (" in source
    assert "simd_sum" in source
    assert "threadgroup_barrier" in source
    assert "_acc_" not in source


def test_loop_state_requires_same_dtype_and_shape():
    with TracingContext("invalid_state"):
        state = metile.loop_state(1.0)
        with pytest.raises(TypeError, match="preserve"):
            state.update(2)
        with pytest.raises(TypeError, match="preserve"):
            state.update(metile.cast(metile.arange(0, 32), "f32"))
        with pytest.raises(ValueError, match="one-dimensional"):
            metile.loop_state(TracingProxy(tir.Value("matrix", TileType((2, 2), "f32"))))


def test_loop_state_rejects_foreign_trace_operands():
    with TracingContext("foreign"):
        foreign = metile.scalar(2.0)
    with TracingContext("current"):
        state = metile.loop_state(0.0)
        with pytest.raises(ValueError, match="contexts"):
            state.update(foreign)
        with pytest.raises(ValueError, match="contexts"):
            metile.loop_state(foreign)


def test_failed_loop_body_restores_outer_capture():
    with TracingContext("failure") as context:
        state = metile.loop_state(0.0)
        outer = context.func.ops
        iterator = metile.tile_range(0, 3)
        next(iterator)
        with pytest.raises(TypeError, match="preserve"):
            state.update(1)
        iterator.close()
        assert context.func.ops is outer
        state.update(2.0)
    assert isinstance(context.func.ops[-1], tir.AssignLoopState)
    assert not any(isinstance(operation, tir.ForRange) for operation in context.func.ops)


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_gpu_loaded_state_masked_recurrence_and_snapshot(steps):
    size = 65
    inputs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    outputs = np.full_like(inputs, np.nan)
    saved = np.full_like(inputs, np.nan)
    _recurrence[(1,)].prepare(inputs, outputs, saved, size, steps, BLOCK=128, STRICT_MATH=True)
    expected = inputs.astype(np.float64)
    for iteration in range(steps):
        expected = expected * 0.5 + iteration
        expected += np.sum(expected) / size
    np.testing.assert_allclose(outputs, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(saved, inputs)
