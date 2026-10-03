import json

import numpy as np
import pytest

import metile
from metile.compiler.lowering.common import LoweringError
from metile.runtime.metal_device import MetalDevice
from metile_kernels.gemm import matmul
from metile_kernels.rmsnorm import rmsnorm
from tests.runtime.test_tensor_memory import descriptor_independent_bounds


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "unknown"},
        {"staging": "registers"},
        {"num_simdgroups": 0},
        {"num_simdgroups": 33},
        {"num_simdgroups": True},
        {"num_simdgroups": 4.0},
        {"vector_width": 2},
        {"vector_width": True},
        {"double_buffer": 1},
    ],
)
def test_schedule_options_reject_invalid_requirements(options):
    with pytest.raises(ValueError):
        metile.Schedule(**options)


@pytest.mark.parametrize("width", [1, 4])
def test_norm_execution_layout_is_materialized_and_reported(width):
    generator = np.random.default_rng(97)
    source = generator.standard_normal((7, 1031)).astype(np.float32)
    weights = generator.standard_normal(1031).astype(np.float32)
    output = np.zeros_like(source)
    dispatch = rmsnorm[(7,)].prepare(
        source,
        weights,
        output,
        1031,
        1e-5,
        BLOCK=128,
        SCHEDULE=metile.Schedule(vector_width=width),
    )
    expected = source / np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + 1e-5) * weights
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
    report = json.loads(dispatch.explain())
    assert report["plan"]["backend"] == "elementwise"
    interiors = [loop for loop in report["loops"] if loop["kind"] == "aligned_interior"]
    assert len(interiors) == 2
    assert {loop["elements_per_lane"] for loop in interiors} == {width}
    assert all(loop["iteration_step"] == 128 * width for loop in interiors)
    tails = [loop for loop in report["loops"] if loop["kind"] == "masked_tail"]
    assert all(loop["elements_per_lane"] == 1 for loop in tails)
    assert dispatch.schedule_plan.threadgroup_size == (128, 1, 1)


def test_unprovable_vector_requirement_fails_instead_of_scalarizing():
    with pytest.raises(LoweringError, match="vector"):
        descriptor_independent_bounds[(1,)].prepare(
            np.ones(37, dtype=np.float32),
            np.zeros(70, dtype=np.float32),
            37,
            70,
            BLOCK=32,
            SCHEDULE=metile.Schedule(vector_width=4),
        )


@pytest.mark.parametrize(
    ("shape", "schedule"),
    [
        ((64, 64, 64), metile.Schedule(backend="simdgroup", vector_width=4, double_buffer=False)),
        ((37, 43, 59), metile.Schedule(backend="simdgroup", vector_width=1, double_buffer=False)),
        ((37, 43, 59), metile.Schedule(backend="simdgroup", double_buffer=True)),
        ((64, 64, 64), metile.Schedule(backend="tensor_ops", staging="device")),
        ((63, 64, 64), metile.Schedule(backend="nax")),
    ],
)
def test_matrix_schedule_requirements_keep_numerics_and_memory_contract(shape, schedule):
    if schedule.backend in {"tensor_ops", "nax"} and not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    rows, columns, reduction = shape
    generator = np.random.default_rng(98)
    left = generator.standard_normal((rows, reduction)).astype(np.float16)
    right = generator.standard_normal((reduction, columns)).astype(np.float16)
    output = np.zeros((rows, columns), dtype=np.float16)
    dispatch = matmul[(metile.cdiv(rows, 64), metile.cdiv(columns, 64))].prepare(
        left,
        right,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        RELAXED_PRECISION=False,
        SCHEDULE=schedule,
    )
    expected = (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=5e-3)
    report = dispatch.execution_report
    assert report.plan.backend == schedule.backend
    assert bool(report.allocations) == (schedule.backend == "simdgroup")
    if schedule.double_buffer is not None:
        assert report.double_buffered is schedule.double_buffer
    if schedule.vector_width == 4:
        assert report.vectorized_loads > 0
    if shape[0] % 64 and schedule.double_buffer is False:
        assert "split_k_loop" not in report.passes


def test_config_simdgroup_count_controls_tensor_backend_geometry():
    if not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    arguments = [np.ones((64, 64), dtype=np.float32) for _ in range(3)]
    launcher = matmul.kernel_fn[(1, 1)]
    launcher(
        *arguments,
        64,
        64,
        64,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        NUM_SG=8,
    )
    assert launcher._last_compiled.threadgroup_size == (256, 1, 1)
    assert launcher._last_compiled.schedule_plan.simdgroup_grid in {(2, 4), (4, 2)}
    np.testing.assert_array_equal(arguments[2], np.full((64, 64), 64, dtype=np.float32))


def test_vector_schedule_does_not_reuse_a_full_tile_proof_for_ragged_rows():
    options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 16,
        "SCHEDULE": metile.Schedule(vector_width=4, double_buffer=False),
    }
    for rows in (64, 96):
        arguments = (
            np.ones((rows, 64), dtype=np.float32),
            np.ones((64, 64), dtype=np.float32),
            np.zeros((rows, 64), dtype=np.float32),
            rows,
            64,
            64,
        )
        if rows == 64:
            dispatch = matmul[(1, 1)].prepare(*arguments, **options)
            assert dispatch.schedule_plan.outer_bounds_proven
            np.testing.assert_array_equal(arguments[2], np.full((rows, 64), 64, np.float32))
        else:
            with pytest.raises(LoweringError, match="vector"):
                matmul[(1, 1)].prepare(*arguments, **options)
