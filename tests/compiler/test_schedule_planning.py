import json
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.compiler.lowering.gemm import _lower_gemm, _lower_tensor_ops_gemm
from metile.compiler.options import Schedule
from metile.compiler.planning import materialize_schedule, plan_schedule
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


def _product(
    left, right, output, rows, columns, reduction, block_rows, block_columns, block_reduction
):
    left_tensor = metile.tensor(
        left, shape=(rows, reduction), block_shape=(block_rows, block_reduction), access="read"
    )
    right_tensor = metile.tensor(
        right,
        shape=(reduction, columns),
        block_shape=(block_reduction, block_columns),
        access="read",
    )
    output_tensor = metile.tensor(
        output, shape=(rows, columns), block_shape=(block_rows, block_columns), access="write"
    )
    row = metile.program_id(0) * block_rows
    column = metile.program_id(1) * block_columns
    accumulator = metile.zeros((block_rows, block_columns), dtype="f32")
    for start in metile.tile_range(0, reduction, block_reduction):
        accumulator = metile.dot(
            left_tensor.load((row, start)), right_tensor.load((start, column)), accumulator
        )
    output_tensor.store((row, column), accumulator)


def _trace(*, dtype="f16", tiles=(64, 64, 16), dimensions=(128, 128, 128), **constexprs):
    context = TracingContext("planned_product")
    arguments = []
    for name in ("left", "right", "output", "rows", "columns", "reduction"):
        value_type = PtrType(dtype) if name in {"left", "right", "output"} else I32
        context.func.params.append(tir.Param(name, value_type, is_output=name == "output"))
        arguments.append(TracingProxy(tir.Value(name, value_type)))
    context.func.constexprs = {
        "_RUNTIME_SCALARS": tuple(zip(("rows", "columns", "reduction"), dimensions, strict=True)),
        "_SCALAR_ALIGNMENT_32": tuple(
            (name, size % 32)
            for name, size in zip(("rows", "columns", "reduction"), dimensions, strict=True)
        ),
        **constexprs,
    }
    with context:
        _product(*arguments, *tiles)
    return context.func


def _elementwise(block=128, **constexprs):
    return tir.Function(
        "planned_elementwise",
        ops=[tir.Arange(size=block)],
        constexprs=constexprs,
    )


def _legacy_product(**constexprs):
    function = _trace(**constexprs)
    function.tensors.clear()
    function.params.append(tir.Param("M", I32))
    for operation in function.ops:
        nested = operation.body if isinstance(operation, tir.ForRange) else [operation]
        for memory in nested:
            if isinstance(memory, (tir.TileLoad, tir.TileStore)):
                memory.tensor = None
    return function


def _specialized_product(**constexprs):
    function = _legacy_product(**constexprs)
    loop = next(operation for operation in function.ops if isinstance(operation, tir.ForRange))
    loads = [operation for operation in loop.body if isinstance(operation, tir.TileLoad)]
    dots = [operation for operation in loop.body if isinstance(operation, tir.Dot)]
    loop.body = [
        operation for operation in loop.body if not isinstance(operation, (tir.TileLoad, tir.Dot))
    ] + [
        tir.SimdgroupRole(role=0, num_roles=2, num_sgs=2, body=loads),
        tir.SimdgroupRole(role=1, num_roles=2, num_sgs=4, body=dots),
    ]
    return function


def _persistent_product(**constexprs):
    function = _legacy_product(**constexprs)
    function.params.append(tir.Param("counter", PtrType("u32")))
    function.ops = [
        tir.PersistentRange(
            counter=tir.Value("counter", PtrType("u32")), total=4, body=function.ops
        )
    ]
    return function


def test_schedule_plan_is_immutable_serializable_and_explains_native_layout():
    plan = plan_schedule(_trace(), supports_tensor_ops=True)
    assert plan.backend == "tensor_ops"
    assert plan.threadgroup_size == (128, 1, 1)
    assert plan.simdgroup_grid == (2, 2)
    assert plan.tile_shape == (64, 64, 16)
    assert plan.staging == "device"
    assert "opaque" in plan.cooperative_layout
    assert json.loads(json.dumps(plan.to_dict()))["threadgroup_size"] == [128, 1, 1]
    assert "backend=tensor_ops" in plan.format()
    assert "double_buffer=auto" in plan.format()
    with pytest.raises(FrozenInstanceError):
        plan.backend = "simdgroup"


@pytest.mark.parametrize("hardware", [False, True])
def test_unconstrained_schedule_preserves_existing_source(hardware):
    function = _trace()
    expected = _lower_tensor_ops_gemm(function) if hardware else _lower_gemm(function)
    plan = plan_schedule(function, supports_tensor_ops=hardware)
    scheduled = materialize_schedule(function, plan)
    lowered = _lower_tensor_ops_gemm(scheduled) if hardware else _lower_gemm(scheduled)
    assert emit(lowered) == emit(expected)
    assert "WM" not in function.constexprs
    assert "NUM_SG" not in function.constexprs


@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize("contract", ["legacy", "schedule", "both"])
def test_requested_group_count_reaches_both_backends(hardware, contract):
    options = {}
    if contract in {"legacy", "both"}:
        options["NUM_SG"] = 8
    if contract in {"schedule", "both"}:
        options["SCHEDULE"] = Schedule(num_simdgroups=8)
    function = _trace(dtype="f32", **options)
    plan = plan_schedule(function, supports_tensor_ops=hardware)
    assert plan.backend == ("tensor_ops" if hardware else "simdgroup")
    assert plan.threadgroup_size == (256, 1, 1)
    assert plan.simdgroup_grid == (2, 4)
    scheduled = materialize_schedule(function, plan)
    lowered = _lower_tensor_ops_gemm(scheduled) if hardware else _lower_gemm(scheduled)
    assert lowered.threadgroup_size == plan.threadgroup_size
    if hardware:
        setup = next(op for op in lowered.ops if isinstance(op, mir.MMatmul2dSetup))
        assert (setup.wm, setup.wn, setup.num_sg) == (2, 4, 8)


@pytest.mark.parametrize("hardware", [False, True])
def test_explicit_geometry_is_respected_by_both_backends(hardware):
    plan = plan_schedule(_trace(dtype="f32", NUM_SG=8, WM=4), supports_tensor_ops=hardware)
    assert plan.simdgroup_grid == (4, 2)
    function = materialize_schedule(_trace(dtype="f32", NUM_SG=8, WM=4), plan)
    if not hardware:
        source = emit(_lower_gemm(function))
        assert "16u" in source
        assert function.constexprs["_PLANNED_SIMDGROUP_GRID"] == (4, 2)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"NUM_SG": 4, "SCHEDULE": Schedule(num_simdgroups=8)}, "conflicts"),
        ({"NUM_SG": 8, "WM": 2, "WN": 2}, "WM/WN conflict"),
        ({"NUM_SG": 0}, "between 1 and 32"),
        ({"NUM_SG": 33}, "between 1 and 32"),
        ({"NUM_SG": True}, "between 1 and 32"),
        ({"WM": 0}, "between 1 and 32"),
        ({"WN": -1}, "between 1 and 32"),
        ({"SCHEDULE": {"backend": "simdgroup"}}, "must be a metile.Schedule"),
    ],
)
def test_conflicting_or_invalid_schedule_requirements_fail(options, message):
    with pytest.raises(LoweringError, match=message):
        plan_schedule(_trace(**options), supports_tensor_ops=True)


@pytest.mark.parametrize("backend", ["tensor_ops", "nax"])
def test_requested_tensor_backend_never_silently_falls_back(backend):
    with pytest.raises(LoweringError, match="requires available Metal tensor operations"):
        plan_schedule(_trace(SCHEDULE=Schedule(backend=backend)), supports_tensor_ops=False)


def test_legacy_nax_request_never_silently_falls_back():
    with pytest.raises(LoweringError, match="requires available Metal tensor operations"):
        plan_schedule(_trace(NAX_FRAGMENTS=True), supports_tensor_ops=False)


def test_forced_nax_materializes_legacy_backend_switch():
    function = _trace(SCHEDULE=Schedule(backend="nax"))
    plan = plan_schedule(function, supports_tensor_ops=True)
    assert plan.backend == "nax"
    assert materialize_schedule(function, plan).constexprs["NAX_FRAGMENTS"] is True
    assert "NAX_FRAGMENTS" not in function.constexprs


def test_strict_float_nax_rejected_before_materialization():
    with pytest.raises(LoweringError, match="Strict f32 NAX packed fragment layout"):
        plan_schedule(
            _trace(dtype="f32", SCHEDULE=Schedule(backend="nax"), RELAXED_PRECISION=False),
            supports_tensor_ops=True,
        )


def test_nax_requires_explicitly_proven_column_and_reduction_alignment():
    with pytest.raises(ValueError, match="aligned N and K"):
        plan_schedule(
            _trace(dimensions=(63, 65, 128), SCHEDULE=Schedule(backend="nax")),
            supports_tensor_ops=True,
        )


@pytest.mark.parametrize("backend", ["simdgroup", "tensor_ops"])
def test_backend_request_conflicts_with_legacy_nax(backend):
    with pytest.raises(LoweringError, match="NAX_FRAGMENTS conflicts"):
        plan_schedule(
            _trace(NAX_FRAGMENTS=True, SCHEDULE=Schedule(backend=backend)),
            supports_tensor_ops=True,
        )


def test_auto_uses_simdgroup_when_half_geometry_cannot_use_native_tensors():
    plan = plan_schedule(_trace(SCHEDULE=Schedule(num_simdgroups=8)), supports_tensor_ops=True)
    assert plan.backend == "simdgroup"
    assert plan.threadgroup_size == (256, 1, 1)


def test_forced_tensor_ops_rejects_unsupported_half_geometry():
    with pytest.raises(LoweringError, match="No legal tensor-operations schedule"):
        plan_schedule(
            _trace(SCHEDULE=Schedule(backend="tensor_ops", num_simdgroups=8)),
            supports_tensor_ops=True,
        )


def test_staging_requirement_selects_backend_not_pointer_address_space():
    function = _trace(SCHEDULE=Schedule(staging="threadgroup"))
    plan = plan_schedule(function, supports_tensor_ops=True)
    assert plan.backend == "simdgroup"
    assert plan.staging == "threadgroup"
    assert all(tensor.address_space == "device" for tensor in function.tensors)
    with pytest.raises(LoweringError, match="requires available Metal tensor operations"):
        plan_schedule(_trace(SCHEDULE=Schedule(staging="device")), supports_tensor_ops=False)


@pytest.mark.parametrize("dimensions", [(63, 128, 128), (128, 63, 128), (128, 128, 63)])
def test_ragged_native_tensor_shapes_fall_back_in_auto_mode(dimensions):
    plan = plan_schedule(_trace(dimensions=dimensions), supports_tensor_ops=True)
    assert plan.backend == "simdgroup"


@pytest.mark.parametrize("dimensions", [(64, 128, 128), (128, 64, 63)])
def test_outer_bounds_proof_uses_semantic_dimension_names(dimensions):
    assert plan_schedule(
        _trace(dimensions=dimensions), supports_tensor_ops=False
    ).outer_bounds_proven


def test_outer_bounds_proof_does_not_invent_more_than_alignment_metadata():
    function = _trace()
    del function.constexprs["_RUNTIME_SCALARS"]
    assert not plan_schedule(function, supports_tensor_ops=False).outer_bounds_proven
    function = _trace(tiles=(32, 32, 16))
    del function.constexprs["_RUNTIME_SCALARS"]
    assert plan_schedule(function, supports_tensor_ops=False).outer_bounds_proven


def test_lowering_attaches_its_exact_plan():
    function = _trace(SCHEDULE=Schedule(backend="simdgroup", num_simdgroups=8))
    with patch("metile.runtime.metal_device.MetalDevice.get", side_effect=AssertionError):
        lowered = lower(function)
    assert lowered.schedule_plan.backend == "simdgroup"
    assert lowered.schedule_plan.threadgroup_size == lowered.threadgroup_size
    assert function.constexprs["SCHEDULE"].num_simdgroups == 8
    assert "NUM_SG" not in function.constexprs


def test_elementwise_plan_reports_existing_lane_geometry():
    plan = plan_schedule(_elementwise(SCHEDULE=Schedule(num_simdgroups=4, vector_width=1)))
    assert plan.backend == "elementwise"
    assert plan.threadgroup_size == (128, 1, 1)
    assert plan.vector_width == 1
    assert plan.staging == "device"


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (Schedule(num_simdgroups=8), "arange/BLOCK lane geometry"),
        (Schedule(backend="tensor_ops"), "require a GEMM dot"),
        (Schedule(staging="threadgroup"), "match the declared memory"),
        (Schedule(double_buffer=True), "double buffering is not implemented"),
    ],
)
def test_elementwise_plan_rejects_controls_that_cannot_be_honored(schedule, message):
    with pytest.raises(LoweringError, match=message):
        plan_schedule(_elementwise(SCHEDULE=schedule))


def test_elementwise_explicit_shared_memory_remains_threadgroup_memory():
    function = _elementwise()
    function.ops.append(tir.SharedAlloc(size=128, dtype="f32"))
    plan = plan_schedule(function)
    assert plan.staging == "threadgroup"
    function.constexprs["SCHEDULE"] = Schedule(staging="device")
    with pytest.raises(LoweringError, match="match the declared memory"):
        plan_schedule(function)


def test_specialized_plan_preserves_producer_and_consumer_geometry():
    function = _specialized_product(SCHEDULE=Schedule(num_simdgroups=6))
    plan = plan_schedule(function, supports_tensor_ops=True)
    assert plan.threadgroup_size == (192, 1, 1)
    assert plan.simdgroup_grid == (2, 2)
    assert plan.double_buffer is True
    lowered = lower(function)
    assert lowered.kernel_type == "specialized_gemm"
    assert lowered.threadgroup_size == plan.threadgroup_size


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (Schedule(num_simdgroups=4), "explicit producer/consumer roles"),
        (Schedule(double_buffer=False), "requires double buffering"),
        (Schedule(backend="nax"), "threadgroup-staged SIMDgroup backend"),
        (Schedule(staging="device"), "threadgroup-staged SIMDgroup backend"),
    ],
)
def test_specialized_plan_rejects_incompatible_requirements(schedule, message):
    with pytest.raises(LoweringError, match=message):
        plan_schedule(_specialized_product(SCHEDULE=schedule), supports_tensor_ops=True)


def test_persistent_plan_honors_group_count_without_changing_algorithm():
    function = _persistent_product(SCHEDULE=Schedule(num_simdgroups=8))
    plan = plan_schedule(function, supports_tensor_ops=True)
    assert plan.backend == "simdgroup"
    assert plan.threadgroup_size == (256, 1, 1)
    lowered = lower(function)
    assert lowered.kernel_type == "persistent_gemm"
    assert lowered.threadgroup_size == plan.threadgroup_size


@pytest.mark.parametrize("backend", ["tensor_ops", "nax"])
def test_persistent_plan_rejects_unsupported_backend(backend):
    with pytest.raises(LoweringError, match="Persistent GEMM currently requires"):
        plan_schedule(
            _persistent_product(SCHEDULE=Schedule(backend=backend)), supports_tensor_ops=True
        )


@pytest.mark.parametrize(
    "schedule", [Schedule(vector_width=1), Schedule(vector_width=4), Schedule(double_buffer=True)]
)
def test_auto_selects_a_backend_that_owns_requested_memory_controls(schedule):
    plan = plan_schedule(_trace(SCHEDULE=schedule), supports_tensor_ops=True)
    assert plan.backend == "simdgroup"
    assert plan.staging == "threadgroup"
    assert plan.vector_width == schedule.vector_width
    assert plan.double_buffer == schedule.double_buffer
    assert "compiler-owned" in plan.reasons[0]


@pytest.mark.parametrize("backend", ["tensor_ops", "nax"])
@pytest.mark.parametrize("width", [1, 4])
def test_native_backend_does_not_promise_scalar_or_vector_lane_access(backend, width):
    with pytest.raises(LoweringError, match="opaque; vector_width"):
        plan_schedule(
            _trace(SCHEDULE=Schedule(backend=backend, vector_width=width)),
            supports_tensor_ops=True,
        )


@pytest.mark.parametrize("backend", ["tensor_ops", "nax"])
def test_native_backend_rejects_explicit_software_double_buffering(backend):
    with pytest.raises(LoweringError, match="double buffering requires"):
        plan_schedule(
            _trace(SCHEDULE=Schedule(backend=backend, double_buffer=True)),
            supports_tensor_ops=True,
        )


@pytest.mark.parametrize("backend", ["auto", "tensor_ops", "nax"])
def test_disabling_double_buffering_is_compatible_with_native_tensors(backend):
    plan = plan_schedule(
        _trace(SCHEDULE=Schedule(backend=backend, double_buffer=False)),
        supports_tensor_ops=True,
    )
    assert plan.backend == ("nax" if backend == "nax" else "tensor_ops")
    assert plan.double_buffer is False
