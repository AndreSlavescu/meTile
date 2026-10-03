import json

import pytest

from metile.compiler.execution_report import (
    execution_report,
    validate_materialized_schedule,
    walk_operations,
)
from metile.compiler.lowering.common import LoweringError
from metile.compiler.planning import SchedulePlan
from metile.frontend.kernel import CompiledKernel, FastDispatcher
from metile.ir import metal_ir as mir


def _function(operations=(), **requirements):
    plan = SchedulePlan(
        backend="elementwise",
        threadgroup_size=(128, 1, 1),
        simdgroup_grid=None,
        tile_shape=(128,),
        staging="device",
        **requirements,
    )
    return mir.MFunction(
        "reported_kernel", ops=list(operations), threadgroup_size=(128, 1, 1), schedule_plan=plan
    )


def _interior(width):
    loop = mir.MForLoop(iv_name="aligned", start=0, end=1024, step=128)
    loop._ew_aligned = True
    loop._vec_size = width
    return loop


def test_report_records_nested_materialized_layouts_and_allocation_sizes():
    interior = _interior(4)
    tail = mir.MForLoop(iv_name="tail", start=1024, end=1031, step=128)
    tail._ew_tail = True
    scratch = mir.MThreadgroupAlloc(alloc_name="scratch", elem_type="half", size=256)
    role = mir.MSimdgroupRoleBlock(body=[interior, tail, scratch])
    function = _function([role])

    report = execution_report(function, ["split_elementwise_loops", "vectorize_elementwise"])

    assert list(walk_operations(function.ops)) == [role, interior, tail, scratch]
    assert [loop.kind for loop in report.loops] == ["aligned_interior", "masked_tail"]
    assert [loop.elements_per_lane for loop in report.loops] == [4, 1]
    assert [loop.iteration_step for loop in report.loops] == [512, 128]
    assert report.allocations[0].name == "scratch"
    assert report.allocations[0].bytes == 512
    assert report.allocations[0].address_space == "threadgroup"
    assert report.passes == ("split_elementwise_loops", "vectorize_elementwise")
    assert json.loads(report.format())["plan"]["threadgroup_size"] == [128, 1, 1]


def test_report_distinguishes_materialized_vector_loads_from_attempted_passes():
    vector_load = mir.MCooperativeLoad(bounds_check=False, vec_size=4)
    scalar_load = mir.MCooperativeLoad(bounds_check=True, vec_size=1)
    function = _function([vector_load, scalar_load])

    report = execution_report(function, ["vectorize_loads", "double_buffer_k_loop"])

    assert report.vectorized_loads == 1
    assert report.double_buffered is False
    assert "No software double buffering materialized." in report.notes


@pytest.mark.parametrize("marker", ["_double_buffered", "_specialized_db"])
def test_report_and_validator_detect_nested_double_buffering(marker):
    loop = mir.MForLoop(iv_name="reduction", step=16)
    setattr(loop, marker, True)
    function = _function([mir.IfBlock(body=[loop])], double_buffer=True)

    validate_materialized_schedule(function)

    assert execution_report(function, []).double_buffered


def test_report_counts_epilogue_operations_for_each_matrix_backend():
    function = _function([mir.MAccElemApply(), mir.MCoopTensorEpilogue(), mir.MNaxApplyFragment()])

    assert execution_report(function, []).epilogue_regions == 3


def test_materialized_schedule_without_a_plan_keeps_legacy_ir_compatible():
    function = mir.MFunction("unplanned")

    validate_materialized_schedule(function)

    assert execution_report(function, []).plan is None


def test_materialized_schedule_rejects_changed_threadgroup_geometry():
    function = _function()
    function.threadgroup_size = (64, 1, 1)

    with pytest.raises(LoweringError, match="geometry"):
        validate_materialized_schedule(function)


@pytest.mark.parametrize(
    "operations",
    [
        [],
        [_interior(1)],
        [mir.MCooperativeLoad(bounds_check=True)],
        [
            mir.MCooperativeLoad(bounds_check=False, vec_size=4),
            mir.MCooperativeLoad(bounds_check=False, vec_size=1),
        ],
    ],
)
def test_materialized_schedule_rejects_unfulfilled_vector_requirement(operations):
    with pytest.raises(LoweringError, match="vector_width=4"):
        validate_materialized_schedule(_function(operations, vector_width=4))


def test_materialized_schedule_allows_scalar_masked_tails_with_vector_interiors():
    tail = mir.MForLoop(iv_name="tail", step=128)
    tail._ew_tail = True
    function = _function([_interior(4), tail], vector_width=4)

    validate_materialized_schedule(function)


def test_materialized_schedule_allows_scalar_checked_loads_with_vector_interiors():
    function = _function(
        [
            mir.MCooperativeLoad(bounds_check=False, vec_size=4),
            mir.MCooperativeLoad(bounds_check=True, vec_size=1),
        ],
        vector_width=4,
    )

    validate_materialized_schedule(function)


def test_materialized_schedule_rejects_unfulfilled_double_buffer_requirement():
    with pytest.raises(LoweringError, match="double_buffer=True"):
        validate_materialized_schedule(_function([mir.MForLoop()], double_buffer=True))


@pytest.mark.parametrize(
    "operation", [_interior(4), mir.MCooperativeLoad(bounds_check=False, vec_size=4)]
)
def test_materialized_schedule_enforces_explicit_scalar_width(operation):
    with pytest.raises(LoweringError, match="vector_width=1"):
        validate_materialized_schedule(_function([operation], vector_width=1))


@pytest.mark.parametrize("marker", ["_double_buffered", "_specialized_db"])
def test_materialized_schedule_enforces_explicit_single_buffering(marker):
    loop = mir.MForLoop()
    setattr(loop, marker, True)

    with pytest.raises(LoweringError, match="double_buffer=False"):
        validate_materialized_schedule(_function([loop], double_buffer=False))


def test_materialized_schedule_accepts_scalar_single_buffered_operations():
    function = _function(
        [_interior(1), mir.MCooperativeLoad(vec_size=1)], vector_width=1, double_buffer=False
    )

    validate_materialized_schedule(function)


def test_compiled_and_prepared_explanations_share_the_same_report_without_gpu_setup():
    report = execution_report(_function([_interior(4)]), ["vectorize_elementwise"])
    compiled = CompiledKernel(None, "", "reported_kernel", (128, 1, 1), execution_report=report)
    dispatch = object.__new__(FastDispatcher)
    dispatch._execution_report = report

    assert compiled.execution_report is dispatch.execution_report
    assert compiled.schedule_plan is dispatch.schedule_plan
    assert json.loads(compiled.explain()) == json.loads(dispatch.explain())
    assert json.loads(compiled.explain())["loops"][0]["elements_per_lane"] == 4
