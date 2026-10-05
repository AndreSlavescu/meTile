from dataclasses import replace

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.execution_report import execution_report
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.compiler.ownership import validate_register_reductions
from metile.frontend.kernel import OutOfResources
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.runtime.metal_device import MetalDevice
from metile_kernels.rmsnorm import rmsnorm_register
from tests.codegen.test_register_lowering import _trace, register_copy


@pytest.mark.parametrize("elements", [2, 4, 8, 16, 32])
def test_register_tiling_changes_threads_not_logical_coverage(elements):
    layout = metile.ThreadLayout.identity(1024, elements_per_thread=elements)
    function = lower(_trace(layout=layout))
    validate_register_reductions(function)
    assert function.threadgroup_size == (1024 // elements, 1, 1)
    assert function.schedule_plan.tile_shape == (1024,)
    loads = [operation for operation in function.ops if isinstance(operation, mir.DeviceLoad)]
    stores = [operation for operation in function.ops if isinstance(operation, mir.DeviceStore)]
    assert len(loads) == 2 * elements
    assert len(stores) == elements
    assert all(operation.mask is not None for operation in (*loads, *stores))
    register_sums = [
        operation
        for operation in function.ops
        if isinstance(operation, mir.MBinOp)
        and operation.result.name.startswith("_metile_register_sum")
    ]
    assert len(register_sums) == elements - 1
    source = emit(function)
    assert source.count("threadgroup_barrier(") == (0 if elements == 32 else 2)
    report = execution_report(function, []).to_dict()
    assert report["register_reductions"][0]["elements_per_thread"] == elements
    assert {value["elements_per_thread"] for value in report["value_layouts"]} == {elements}


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
def test_implicit_tile_ownership_cannot_enter_generalized_register_programs(elements):
    function = _trace(layout=metile.ThreadLayout.identity(1024, elements_per_thread=elements))
    function.add_op(tir.Arange(size=1024))
    with pytest.raises(LoweringError, match="explicit ownership"):
        lower(function)


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
def test_nonidentity_register_conversions_remain_rejected_for_new_counts(elements):
    layout = metile.ThreadLayout.identity(1024, elements_per_thread=elements)
    function = _trace(layout=layout)
    positions = next(
        operation.result for operation in function.ops if isinstance(operation, tir.Arange)
    )
    function.add_op(
        tir.ConvertLayout(
            value=positions,
            layout=metile.ThreadLayout(layout.bit_order, xor_mask=1, elements_per_thread=elements),
        )
    )
    with pytest.raises(LoweringError, match="nonidentity multi-register"):
        lower(function)


def test_one_register_explicit_layout_still_rejects_reductions():
    with pytest.raises(LoweringError, match=r"straight-line.*Reduce"):
        lower(_trace(layout=metile.ThreadLayout.identity(1024)))


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
def test_register_geometry_cannot_mix_distinct_physical_thread_counts(elements):
    layout = metile.ThreadLayout.identity(1024, elements_per_thread=elements)
    function = _trace(layout=layout)
    function.add_op(tir.Arange(size=1024, layout=metile.ThreadLayout.identity(1024)))
    with pytest.raises(LoweringError, match="consistent register and thread geometry"):
        lower(function)


@pytest.mark.parametrize("elements", [1, 3, 64, True, 4.0])
def test_late_reduction_contract_rejects_unsupported_or_noninteger_counts(elements):
    function = lower(_trace())
    function.register_reductions = (
        replace(function.register_reductions[0], elements_per_thread=elements),
    )
    with pytest.raises(LoweringError, match="register reduction"):
        validate_register_reductions(function)


@pytest.mark.parametrize(
    "corruption",
    [
        "valid_record_count",
        "ownership_count",
        "ownership_layout",
        "ownership_shape",
        "missing_ownership",
        "single_group_scratch",
    ],
)
def test_late_reduction_metadata_must_match_actual_thread_and_register_ownership(corruption):
    elements = 32 if corruption == "single_group_scratch" else 4
    function = lower(
        _trace(layout=metile.ThreadLayout.identity(1024, elements_per_thread=elements))
    )
    record = function.register_reductions[0]
    ownership = function.value_layouts[0]
    if corruption == "valid_record_count":
        function.register_reductions = (replace(record, elements_per_thread=8),)
    elif corruption == "ownership_count":
        function.value_layouts = (
            replace(ownership, elements_per_thread=8),
            *function.value_layouts[1:],
        )
    elif corruption == "ownership_layout":
        function.value_layouts = (
            replace(ownership, layout=metile.ThreadLayout.identity(1024, elements_per_thread=8)),
            *function.value_layouts[1:],
        )
    elif corruption == "ownership_shape":
        function.value_layouts = (replace(ownership, shape=(2048,)), *function.value_layouts[1:])
    elif corruption == "missing_ownership":
        function.value_layouts = ()
    else:
        function.register_reductions = (replace(record, scratch="unused_scratch"),)
    with pytest.raises(LoweringError, match="register reduction"):
        validate_register_reductions(function)


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("width", [1009, 1024])
def test_gpu_generalized_register_rmsnorm_preserves_fp32_accumulation(elements, dtype, width):
    generator = np.random.default_rng(elements + width)
    source = generator.standard_normal((3, width)).astype(dtype)
    weights = generator.standard_normal(width).astype(dtype)
    output = np.full_like(source, np.nan)
    identity = metile.ThreadLayout.identity(1024, elements_per_thread=elements)
    layout = metile.ThreadLayout(
        tuple(reversed(identity.bit_order)), xor_mask=7, elements_per_thread=elements
    )
    dispatch = rmsnorm_register[(3,)].prepare(
        source, weights, output, 1e-5, N=width, BLOCK=1024, LAYOUT=layout, RELAXED_PRECISION=False
    )
    promoted = source.astype(np.float32)
    expected = promoted / np.sqrt(np.mean(promoted * promoted, axis=1, keepdims=True) + 1e-5)
    expected *= weights.astype(np.float32)
    tolerance = 2e-3 if dtype == np.float16 else 2e-5
    np.testing.assert_allclose(output, expected.astype(dtype), rtol=tolerance, atol=tolerance)
    assert dispatch.execution_report.plan.threadgroup_size == (1024 // elements, 1, 1)


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
@pytest.mark.parametrize("dtype", [np.float16, np.int32])
@pytest.mark.parametrize("threads", [128, 1024])
def test_gpu_register_tiling_respects_pipeline_thread_limit(monkeypatch, elements, dtype, threads):
    block = elements * threads
    identity = metile.ThreadLayout.identity(block, elements_per_thread=elements)
    layout = metile.ThreadLayout(
        tuple(reversed(identity.bit_order)), xor_mask=3, elements_per_thread=elements
    )
    source = (np.arange(block * 2 - 7, dtype=np.int32) % 127).astype(dtype)
    output = np.full_like(source, -1)
    device = MetalDevice.get()
    pipeline_limits = []
    query_limit = device.pipeline_max_threads

    def record_limit(pipeline):
        limit = query_limit(pipeline)
        pipeline_limits.append(limit)
        return limit

    monkeypatch.setattr(device, "pipeline_max_threads", record_limit)
    try:
        register_copy[(2,)].prepare(source, output, source.size, BLOCK=block, LAYOUT=layout)
    except OutOfResources:
        assert threads == 1024
        assert pipeline_limits and pipeline_limits[-1] < threads
        np.testing.assert_array_equal(output, np.full_like(source, -1))
        return
    np.testing.assert_array_equal(output, source)
