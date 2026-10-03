from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter.gemm import _emit_double_buffered_k_loop
from metile.compiler.lowering.common import _build_kk_loop
from metile.compiler.passes import double_buffer_k_loop, split_k_loop
from metile.compiler.staging import StagingError, validate_pipeline, validate_staging
from metile.ir import metal_ir as mir
from metile.ir.types import I32, PtrType
from metile_kernels.gemm import matmul


def _matrix_function(*, names=("shared_a", "shared_b"), scratch=0, persistent=False):
    function = mir.MFunction(
        "staged_matrix", kernel_type="persistent_gemm" if persistent else "gemm"
    )
    function.threadgroup_size = (32, 1, 1)
    function.params = [
        mir.MParam("left", PtrType("f16")),
        mir.MParam("right", PtrType("f16")),
        mir.MParam("K", I32, is_scalar=True),
    ]
    reduction = mir.MValue("K", I32)
    zero = function.add_op(mir.MConstant(value=0, dtype="i32"), "zero")
    eight = function.add_op(mir.MConstant(value=8, dtype="i32"), "eight")
    lane = function.add_op(mir.ThreadPositionInThreadgroup(), "lane")
    for name in names:
        function.add_op(mir.MThreadgroupAlloc(alloc_name=name, elem_type="half", size=64))
    if scratch:
        function.add_op(
            mir.MThreadgroupAlloc(alloc_name="scratch", elem_type="uchar", size=scratch)
        )
    loads = [
        mir.MCooperativeLoad(
            device_ptr=mir.MValue("left", PtrType("f16")),
            tg_array=names[0],
            row_offset=zero,
            col_offset=None,
            src_stride=reduction,
            tile_rows=8,
            tile_cols=8,
            dst_stride=8,
            tg_size=32,
            linear_tid=lane,
            row_bound=eight,
            col_bound=reduction,
            elem_type="half",
        ),
        mir.MCooperativeLoad(
            device_ptr=mir.MValue("right", PtrType("f16")),
            tg_array=names[1],
            row_offset=None,
            col_offset=zero,
            src_stride=eight,
            tile_rows=8,
            tile_cols=8,
            dst_stride=8,
            tg_size=32,
            linear_tid=lane,
            row_bound=reduction,
            col_bound=eight,
            elem_type="half",
        ),
    ]
    inner = _build_kk_loop(1, 1, zero, zero, names[0], names[1], 8, 8, "half", 8)
    loop = mir.MForLoop(
        iv_name="kb",
        start=0,
        end=reduction,
        step=8,
        body=[
            *loads,
            mir.MBarrier(),
            inner,
            mir.MBarrier(),
        ],
    )
    function.ops.append(mir.MWhileTrue(body=[loop]) if persistent else loop)
    return function, loop


def _source(function, loop):
    lines = []
    _emit_double_buffered_k_loop(loop, lines, 1, function)
    return "\n".join(lines)


def _assert_unchanged(function, original):
    assert repr(function) == original


@pytest.mark.parametrize("persistent", [False, True])
def test_two_stage_contract_has_explicit_publish_release_and_cross_iteration_reuse(persistent):
    function, loop = _matrix_function(persistent=persistent)

    returned, applied = double_buffer_k_loop(function)

    assert returned is function and applied
    assert loop.staging.stages == 2
    assert loop.staging.mechanism == "software"
    assert not hasattr(loop, "_double_buffered")
    validate_staging(function)
    assert [phase.name for phase in loop.staging.phases] == [
        "load_first",
        "publish_first",
        "load_next",
        "consume_current",
        "publish_and_recycle",
        "rotate_slots",
        "consume_last",
        "release_last",
    ]
    exchange = loop.staging.phases[4]
    assert {access.slot for access in exchange.publishes} == {"next"}
    assert {access.slot for access in exchange.recycles} == {"current"}
    assert exchange.barrier_scope == "threadgroup"
    assert exchange.barrier_flags == "mem_threadgroup"
    assert sum(buffer.bytes_per_slot for buffer in loop.staging.buffers) == 256


def test_double_buffer_pass_is_idempotent_and_split_preserves_typed_pipeline():
    function, loop = _matrix_function()
    assert double_buffer_k_loop(function)[1]
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]
    split_k_loop(function)

    _assert_unchanged(function, original)
    assert function.ops[-1] is loop


def test_only_operand_allocations_are_doubled_and_budget_is_exact():
    function, loop = _matrix_function(scratch=100)

    assert double_buffer_k_loop(function, max_tg_bytes=612)[1]

    allocations = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert [allocation.alloc_name for allocation in allocations] == [
        "shared_a_0",
        "shared_a_1",
        "shared_b_0",
        "shared_b_1",
        "scratch",
    ]
    assert sum(buffer.bytes_per_slot * 2 for buffer in loop.staging.buffers) + 100 == 612


def test_budget_rejection_does_not_rename_allocations_or_attach_metadata():
    function, loop = _matrix_function(scratch=100)
    original = repr(function)

    assert not double_buffer_k_loop(function, max_tg_bytes=611)[1]

    _assert_unchanged(function, original)
    assert loop.staging is None


def test_function_without_candidate_loop_is_not_transformed():
    function, _ = _matrix_function()
    function.ops.pop()
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]

    _assert_unchanged(function, original)


@pytest.mark.parametrize(
    "change",
    [
        "extra_operation",
        "missing_publish",
        "conditional_publish",
        "device_publish",
        "weak_publish",
        "missing_release",
        "divergent_release",
        "extra_compute",
        "compute_side_effect",
        "uninitialized_fragment",
        "nonzero_start",
        "invalid_step",
        "nonuniform_end",
        "unchecked_load",
        "overridden_reduction",
        "wrong_bound",
        "wrong_extent",
        "incomplete_threads",
        "mixed_types",
        "aliased_operands",
        "unknown_source",
        "wrong_read_stride",
        "undersized_allocation",
        "read_depends_on_kb",
        "load_depends_on_kb",
        "name_collision",
        "layout_mismatch",
        "different_compute_extent",
    ],
)
def test_unsupported_or_unsafe_loop_is_rejected_without_mutation(change):
    function, loop = _matrix_function()
    first, second, publish, compute, release = loop.body
    if change == "extra_operation":
        loop.body.append(mir.MAccElemApply())
    elif change == "missing_publish":
        loop.body.pop(2)
    elif change == "conditional_publish":
        publish.condition = "slid == 0"
    elif change == "device_publish":
        publish.flags = "mem_device"
    elif change == "weak_publish":
        publish.flags = "mem_none"
    elif change == "missing_release":
        loop.body.pop()
    elif change == "divergent_release":
        release.kind = "simdgroup"
    elif change == "extra_compute":
        loop.body.insert(4, deepcopy(compute))
    elif change == "compute_side_effect":
        compute.body.append(mir.MThreadgroupStore(array_name="shared_a"))
    elif change == "uninitialized_fragment":
        compute.body.insert(0, compute.body.pop())
    elif change == "nonzero_start":
        loop.start = 8
    elif change == "invalid_step":
        loop.step = 7
    elif change == "nonuniform_end":
        loop.end = first.linear_tid
    elif change == "unchecked_load":
        first.bounds_check = False
    elif change == "overridden_reduction":
        first.kb_expr = "kb + 8"
    elif change == "wrong_bound":
        first.col_bound = first.row_bound
    elif change == "wrong_extent":
        first.tile_cols = 16
    elif change == "incomplete_threads":
        first.tg_size = 64
    elif change == "mixed_types":
        second.elem_type = "float"
    elif change == "aliased_operands":
        second.tg_array = first.tg_array
    elif change == "unknown_source":
        compute.body[0].src_array = "other"
    elif change == "wrong_read_stride":
        compute.body[0].stride = 16
    elif change == "undersized_allocation":
        next(
            operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
        ).size = 63
    elif change == "read_depends_on_kb":
        compute.body[0].sg_offset = mir.MValue("kb", I32)
    elif change == "load_depends_on_kb":
        first.row_offset = mir.MValue("kb", I32)
    elif change == "name_collision":
        function.params.append(mir.MParam("sa_curr", I32, is_scalar=True))
    elif change == "layout_mismatch":
        first.load_layout = mir.CooperativeLoadLayout(tile=mir.TileLayout(8, 8, 9), num_threads=32)
    elif change == "different_compute_extent":
        compute.end = 16
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]

    _assert_unchanged(function, original)


@pytest.mark.parametrize("wrapper", [mir.IfBlock, mir.MSimdgroupRoleBlock, mir.MForLoop])
def test_staging_rejects_loops_inside_unproved_execution_scope(wrapper):
    function, loop = _matrix_function()
    function.ops[-1] = wrapper(body=[loop])
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]

    _assert_unchanged(function, original)


@pytest.mark.parametrize(
    "user", [mir.MThreadgroupStore(array_name="shared_a"), mir.MSimdgroupLoad(src_array="shared_b")]
)
def test_storage_with_additional_external_users_is_not_renamed(user):
    function, _ = _matrix_function()
    function.ops.append(user)
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]

    _assert_unchanged(function, original)


def test_source_uses_verified_buffer_identities_and_never_mutates_ir():
    function, loop = _matrix_function(names=("left_stage", "right_stage"))
    assert double_buffer_k_loop(function)[1]
    original = repr(function)

    source = _source(function, loop)

    assert source == _source(function, loop)
    _assert_unchanged(function, original)
    assert "if (K > 0)" in source
    assert "left_stage_curr = left_stage_0" in source
    assert "right_stage_next = right_stage_1" in source
    assert "left_stage_next[_r" in source
    assert "left_stage_curr +" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 3
    assert source.rstrip().endswith("threadgroup_barrier(mem_flags::mem_threadgroup);\n    }")
    assert "shared_a" not in source and "shared_b" not in source


def test_emission_exception_does_not_leave_operand_aliases_patched(monkeypatch):
    import metile.codegen.msl_emitter.gemm as emitter

    function, loop = _matrix_function()
    assert double_buffer_k_loop(function)[1]
    original = repr(function)

    def fail(*arguments):
        raise RuntimeError("source emission failed")

    monkeypatch.setattr(emitter, "_emit_cooperative_load", fail)
    with pytest.raises(RuntimeError, match="source emission failed"):
        _source(function, loop)
    _assert_unchanged(function, original)


@pytest.mark.parametrize(
    "change", ["phase_order", "barrier", "allocation", "side_effect", "scope", "async"]
)
def test_late_verification_detects_invalidated_staging_contract(change):
    function, loop = _matrix_function()
    assert double_buffer_k_loop(function)[1]
    if change == "phase_order":
        phases = list(loop.staging.phases)
        phases[3], phases[4] = phases[4], phases[3]
        loop.staging = replace(loop.staging, phases=tuple(phases))
    elif change == "barrier":
        loop.body[2].flags = "mem_none"
    elif change == "allocation":
        next(
            operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
        ).size -= 1
    elif change == "side_effect":
        loop.body[3].body.append(mir.MThreadgroupStore(array_name="shared_a"))
    elif change == "scope":
        function.ops[-1] = mir.IfBlock(body=[loop])
    elif change == "async":
        loop.staging = replace(loop.staging, mechanism="async_copy")

    with pytest.raises(StagingError):
        validate_staging(function)


def test_untyped_legacy_marker_cannot_bypass_staging_verification():
    function, loop = _matrix_function()
    loop._double_buffered = True

    with pytest.raises(StagingError, match="verified staging contract"):
        _source(function, loop)


def test_phase_program_rejects_missing_recycle_before_overwrite():
    function, loop = _matrix_function()
    assert double_buffer_k_loop(function)[1]
    phases = list(loop.staging.phases)
    phases[4] = replace(phases[4], recycles=())

    with pytest.raises(StagingError, match="publication/recycle"):
        validate_pipeline(replace(loop.staging, phases=tuple(phases)))


@pytest.mark.parametrize(
    "change",
    ["copy_alias", "fragment_out_of_bounds", "fragment_nonuniform", "swizzle", "nested_scratch"],
)
def test_staging_checks_thread_ownership_and_allocation_lifetimes(change):
    function, loop = _matrix_function()
    if change == "copy_alias":
        loop.body[0].linear_tid = function.add_op(
            mir.MConstant(value=0, dtype="i32"), "same_thread"
        )
    elif change == "fragment_out_of_bounds":
        loop.body[3].body[0].tile_offset = 8
    elif change == "fragment_nonuniform":
        loop.body[3].body[0].sg_offset = loop.body[0].linear_tid
    elif change == "swizzle":
        loop.body[0].swizzle_bits = 4
        loop.body[0].swizzle_shift = 3
    elif change == "nested_scratch":
        function.ops.append(
            mir.MWhileTrue(body=[mir.MThreadgroupAlloc(alloc_name="extra", size=1024)])
        )
    original = repr(function)

    assert not double_buffer_k_loop(function)[1]

    _assert_unchanged(function, original)


@pytest.mark.parametrize("change", ["slots", "pointer", "external_slot_user"])
def test_late_validation_rejects_physical_aliases_and_scope_collisions(change):
    function, loop = _matrix_function()
    assert double_buffer_k_loop(function)[1]
    first, second = loop.staging.buffers
    if change == "slots":
        second = replace(second, slots=first.slots)
    elif change == "pointer":
        second = replace(second, current_pointer=first.current_pointer)
    elif change == "external_slot_user":
        function.ops.append(mir.MThreadgroupStore(array_name=first.slots[0]))
    loop.staging = replace(loop.staging, buffers=(first, second))

    with pytest.raises(StagingError):
        validate_staging(function)


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("swizzled", [False, True])
def test_semantic_matrix_bindings_remain_verified_after_optimization(dtype, swizzled):
    from metile.codegen.msl_emitter import emit
    from metile.compiler.lowering.gemm import _lower_gemm
    from metile.compiler.passes import (
        fold_constants,
        pad_shared_memory,
        preload_mma_tiles,
        serpentine_mma,
        swizzle_shared_memory,
    )
    from tests.compiler.test_schedule_planning import _trace

    function = _lower_gemm(_trace(dtype=dtype))
    (swizzle_shared_memory if swizzled else pad_shared_memory)(function)
    assert double_buffer_k_loop(function)[1]
    serpentine_mma(function)
    preload_mma_tiles(function)
    fold_constants(function)

    validate_staging(function)

    source = emit(function)
    assert "if (K > 0)" in source
    assert "shared_a_0" in source


@pytest.mark.parametrize(("dtype", "element_type"), [("f16", "half"), ("f32", "float")])
def test_specialized_pipeline_pointer_types_follow_their_allocations(dtype, element_type):
    from metile.codegen.msl_emitter import emit
    from metile.compiler.lowering.gemm import _lower_specialized_gemm
    from tests.compiler.test_schedule_planning import _specialized_product

    function = _lower_specialized_gemm(_specialized_product(dtype=dtype))
    original = repr(function)

    source = emit(function)

    assert f"threadgroup {element_type}* sa_curr = shared_a_0" in source
    assert f"threadgroup {element_type}* _t = sa_curr" in source
    _assert_unchanged(function, original)


def test_specialized_half_staging_uses_actual_bytes_for_resource_limit():
    from metile.compiler.lowering.common import LoweringError
    from metile.compiler.lowering.gemm import _lower_specialized_gemm
    from tests.compiler.test_schedule_planning import _specialized_product

    function = _lower_specialized_gemm(_specialized_product(dtype="f16", tiles=(64, 64, 48)))
    allocations = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert sum(allocation.size * 2 for allocation in allocations) == 25024
    with pytest.raises(LoweringError, match="threadgroup memory"):
        _lower_specialized_gemm(_specialized_product(dtype="f32", tiles=(64, 64, 48)))


def test_specialized_pipeline_rejects_mixed_storage_element_types():
    from metile.codegen.msl_emitter import emit
    from metile.compiler.lowering.gemm import _lower_specialized_gemm
    from tests.compiler.test_schedule_planning import _specialized_product

    function = _lower_specialized_gemm(_specialized_product(dtype="f16"))
    next(
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ).elem_type = "float"

    with pytest.raises(ValueError, match="matching typed operand allocations"):
        emit(function)


@metile.kernel
def _legacy_specialized_product(left, right, output, M, N, K):
    row = metile.program_id(0) * 64
    column = metile.program_id(1) * 64
    accumulator = metile.zeros((64, 64), dtype="f32")
    for reduction in metile.tile_range(0, K, 16):
        with metile.simdgroup_role(role=0, num_roles=2, num_sgs=2):
            left_tile = metile.tile_load(left, row, reduction, K, (64, 16))
            right_tile = metile.tile_load(right, reduction, column, N, (16, 64))
        with metile.simdgroup_role(role=1, num_roles=2, num_sgs=4):
            accumulator = metile.dot(left_tile, right_tile, accumulator)
    metile.tile_store(output, row, column, N, accumulator, (64, 64))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("shape", [(64, 64, 64), (37, 43, 59)])
def test_gpu_legacy_specialized_pipeline_preserves_storage_and_partial_tiles(dtype, shape):
    rows, columns, reduction = shape
    generator = np.random.default_rng(215)
    left = generator.standard_normal((rows, reduction)).astype(dtype)
    right = generator.standard_normal((reduction, columns)).astype(dtype)
    output = np.full((rows, columns), np.nan, dtype=dtype)

    _legacy_specialized_product[(1, 1)](left, right, output, rows, columns, reduction)

    expected = (left.astype(np.float32) @ right.astype(np.float32)).astype(dtype)
    np.testing.assert_allclose(
        output,
        expected,
        rtol=2e-3 if dtype == np.float16 else 2e-5,
        atol=5e-3 if dtype == np.float16 else 2e-5,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("reduction", [0, 1, 15, 16, 17, 48, 59])
def test_gpu_staged_gemm_empty_single_multiple_and_partial_iterations(dtype, reduction):
    generator = np.random.default_rng(213)
    rows, columns = 37, 43
    left = generator.standard_normal((rows, max(1, reduction))).astype(dtype)
    right = generator.standard_normal((max(1, reduction), columns)).astype(dtype)
    output = np.full((rows, columns), np.nan, dtype=dtype)
    dispatch = matmul[(1, 1)].prepare(
        left,
        right,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        SCHEDULE=metile.Schedule(backend="simdgroup", double_buffer=True),
        RELAXED_PRECISION=False,
    )
    expected = left[:, :reduction].astype(np.float32) @ right[:reduction, :].astype(np.float32)
    np.testing.assert_allclose(
        output,
        expected.astype(dtype),
        rtol=2e-3 if dtype == np.float16 else 2e-5,
        atol=5e-3 if dtype == np.float16 else 2e-5,
    )
    assert dispatch.execution_report.double_buffered
