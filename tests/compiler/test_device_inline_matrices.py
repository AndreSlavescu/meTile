import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.execution_report import validate_materialized_schedule
from metile.compiler.lowering import LoweringError, lower
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType


@metile.kernel
def _device_copy(
    Source,
    Control,
    Output,
    *,
    DTYPE: metile.constexpr,
    TRANSPOSE_SOURCE: metile.constexpr = False,
    TRANSPOSE_OUTPUT: metile.constexpr = False,
    STORE_ROW_OFFSET: metile.constexpr = 0,
    STORE_COLUMN_OFFSET: metile.constexpr = 0,
    BLOCK: metile.constexpr = 64,
):
    scratch = metile.tensor(metile.shared(BLOCK * 2, dtype=DTYPE), shape=(BLOCK // 4, 8))
    control = metile.tensor(Control, shape=(8,), access="read")
    source = metile.tensor(
        Source + control.load((6,)),
        shape=(control.load((0,)), control.load((1,))),
        strides=(1, 24) if TRANSPOSE_SOURCE else (24, 1),
        block_shape=(8, 8),
        access="read",
    )
    output = metile.tensor(
        Output + control.load((7,)),
        shape=(control.load((4,)), control.load((5,))),
        strides=(1, 24) if TRANSPOSE_OUTPUT else (24, 1),
        block_shape=(8, 8),
        access="write",
    )
    row = metile.cast(metile.thread_id() // 32 * 8, "i32")
    fragment = source.load((row + control.load((2,)), control.load((3,))), scratch=scratch)
    output.store((row + STORE_ROW_OFFSET, STORE_COLUMN_OFFSET), fragment, scratch=scratch)


@metile.kernel
def _device_product(
    Left,
    Right,
    Control,
    Output,
    *,
    DTYPE: metile.constexpr,
    TRANSPOSE_RIGHT: metile.constexpr = False,
    BLOCK: metile.constexpr = 64,
):
    scratch = metile.tensor(metile.shared(BLOCK * 2, dtype=DTYPE), shape=(BLOCK // 4, 8))
    control = metile.tensor(Control, shape=(3,), access="read")
    rows = control.load((0,))
    features = control.load((1,))
    columns = control.load((2,))
    left = metile.tensor(
        Left, shape=(rows, features), strides=(24, 1), block_shape=(8, 8), access="read"
    )
    right = metile.tensor(
        Right,
        shape=(features, columns),
        strides=(1, 24) if TRANSPOSE_RIGHT else (24, 1),
        block_shape=(8, 8),
        access="read",
    )
    output = metile.tensor(
        Output, shape=(rows, columns), strides=(24, 1), block_shape=(8, 8), access="write"
    )
    row = metile.program_id(0) * 16 + metile.thread_id() // 32 * 8
    column = metile.program_id(1) * 8
    accumulator = metile.loop_state(metile.zeros((8, 8), dtype="f32"))
    for feature in metile.tile_range(0, features, 8):
        accumulator.update(
            metile.dot(
                left.load((row, feature), scratch=scratch),
                right.load((feature, column), scratch=scratch),
                accumulator.value,
            )
        )
    output.store((row, column), accumulator.value, scratch=scratch)


@metile.kernel
def _device_contract(
    Source, Control, Output, *, CASE: metile.constexpr, BLOCK: metile.constexpr = 64
):
    allocation = metile.shared(256, dtype="f16" if CASE == "scratch_dtype" else "f32")
    control = metile.tensor(Control, shape=(8,), access="read")
    lane = metile.simd_lane_id()
    scratch_pointer = (
        Source
        if CASE == "device_scratch"
        else allocation + 1
        if CASE == "offset_scratch"
        else allocation
    )
    scratch = metile.tensor(
        scratch_pointer,
        shape=(8 if CASE == "scratch_shape" else BLOCK // 4, 8),
        strides=(9, 1) if CASE == "scratch_strides" else (8, 1),
        access="read"
        if CASE == "read_scratch"
        else "write"
        if CASE == "write_scratch"
        else "readwrite",
    )
    base = lane if CASE == "divergent_base" else 0
    rows = lane + 8 if CASE == "divergent_shape" else 16
    stride = (
        control.load((0,))
        if CASE == "dynamic_stride"
        else 0
        if CASE == "zero_stride"
        else -16
        if CASE == "negative_stride"
        else 16
    )
    source = metile.tensor(
        Source + base,
        shape=(rows, 16),
        strides=(2, 3) if CASE == "irregular_strides" else (stride, 1),
        block_shape=(8, 8),
        access="read",
    )
    output = metile.tensor(Output, shape=(16, 16), block_shape=(8, 8), access="write")
    row = lane if CASE == "divergent_origin" else metile.thread_id() // 32 * 8
    if CASE == "scalar_scratch":
        scalar = metile.tensor(Source, shape=(16, 16), access="read")
        scalar.load((0, 0), scratch=scratch)
    elif CASE == "scalar_store_scratch":
        scalar = metile.tensor(Output, shape=(16, 16), access="write")
        scalar.store((0, 0), 1.0, scratch=scratch)
    elif CASE == "shared_scratch":
        shared = metile.tensor(metile.shared(128, dtype="f32"), shape=(16, 8), block_shape=(8, 8))
        shared.load((0, 0), scratch=scratch)
    elif CASE == "shared_store_scratch":
        shared = metile.tensor(metile.shared(128, dtype="f32"), shape=(16, 8), block_shape=(8, 8))
        shared.store((0, 0), metile.zeros((8, 8)), scratch=scratch)
    elif CASE == "missing_load_scratch":
        source.load((row, 0))
    elif CASE == "missing_store_scratch":
        output.store((row, 0), metile.zeros((8, 8)))
    elif CASE == "not_tensor_scratch":
        source.load((row, 0), scratch=allocation)
    elif CASE == "divergent_control":
        for _iteration in metile.tile_range(lane, 32, 1):
            source.load((row, 0), scratch=scratch)
    else:
        if CASE == "scratch_scalar_write":
            scratch.store((0, 0), 1.0)
        fragment = source.load((row, 0), scratch=scratch)
        output.store((row, 0), fragment, scratch=scratch)
        if CASE == "scratch_scalar_read":
            scalar = metile.tensor(Output, shape=(1,), access="write")
            scalar.store((0,), scratch.load((0, 0)))
        if CASE in {"scratch_matrix_read", "scratch_matrix_write"}:
            shared = metile.tensor(allocation, shape=(16, 8), block_shape=(8, 8))
            if CASE == "scratch_matrix_read":
                shared.load((row, 0))
            else:
                shared.store((row, 0), fragment)


@metile.kernel
def _device_alias_contract(
    Source, Control, Output, *, CASE: metile.constexpr, BLOCK: metile.constexpr = 64
):
    scratch = metile.tensor(metile.shared(BLOCK * 2, dtype="f32"), shape=(BLOCK // 4, 8))
    control = metile.tensor(Control, shape=(BLOCK,))
    source = metile.tensor(Source, shape=(16, 16), block_shape=(8, 8))
    target = (
        Output
        if CASE in {"distinct_roots", "unrelated_scalar_in_place"}
        else Source + 64 + control.load((0,))
        if CASE == "offset_alias"
        else Source
    )
    destination = metile.tensor(target, shape=(16, 16), block_shape=(8, 8))
    source_values = metile.tensor(Source, shape=(256,))
    output_values = metile.tensor(Output, shape=(256,), access="write")
    row = metile.thread_id() // 32 * 8
    if CASE == "unrelated_scalar_in_place":
        control.store((metile.thread_id(),), control.load((metile.thread_id(),)) + 1)
    if CASE == "scalar_store_then_tile_load":
        source_values.store((metile.thread_id(),), 1.0)
        metile.barrier()
        fragment = source.load((row, 0), scratch=scratch)
        destination.store((row, 0), fragment, scratch=scratch)
    elif CASE == "tile_store_then_scalar_load":
        destination.store((row, 0), metile.zeros((8, 8)), scratch=scratch)
        metile.barrier()
        output_values.store((metile.thread_id(),), source_values.load((metile.thread_id(),)))
    elif CASE == "tile_store_then_tile_load":
        destination.store((row, 0), metile.zeros((8, 8)), scratch=scratch)
        metile.barrier()
        source.load((row, 0), scratch=scratch)
    else:
        fragment = source.load((row, 0), scratch=scratch)
        metile.barrier()
        if CASE == "tile_load_then_scalar_store":
            source_values.store((metile.thread_id(),), 1.0)
        elif CASE == "write_in_loop":
            for column in metile.tile_range(0, 16, 8):
                destination.store((row, column), fragment, scratch=scratch)
        else:
            destination.store((row, 0), fragment, scratch=scratch)


def _trace(kernel=_device_copy, dtype="f32", **constants):
    names = (
        ("Left", "Right", "Control", "Output")
        if kernel is _device_product
        else ("Source", "Control", "Output")
    )
    with TracingContext(kernel.name) as context:
        context.func.constexprs.update(
            SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
            STRICT_MATH=True,
            BLOCK=64,
            **constants,
        )
        context.func.params = [
            tir.Param(name, PtrType("i32" if name == "Control" else dtype)) for name in names
        ]
        arguments = [
            TracingProxy(tir.Value(parameter.name, parameter.type))
            for parameter in context.func.params
        ]
        if kernel in (_device_contract, _device_alias_contract):
            kernel.fn(*arguments, **constants, BLOCK=64)
        else:
            context.func.constexprs["DTYPE"] = dtype
            kernel.fn(*arguments, DTYPE=dtype, **constants, BLOCK=64)
    _mark_outputs(context.func)
    return context.func


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("transpose_source", [False, True])
@pytest.mark.parametrize("transpose_output", [False, True])
def test_device_fragments_preserve_dynamic_views_and_output_ownership(
    dtype, transpose_source, transpose_output
):
    function = _trace(
        dtype=dtype, TRANSPOSE_SOURCE=transpose_source, TRANSPOSE_OUTPUT=transpose_output
    )
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Output"]
    metal = lower(function)
    validate_materialized_schedule(metal)
    assert metal.kernel_type == "simdgroup_inline"
    assert metal.threadgroup_size == (64, 1, 1)
    allocations = [
        operation for operation in metal.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    assert len(allocations) == 1 and allocations[0].size == 128
    source = emit(metal)
    assert "simdgroup_load(" in source
    assert "simdgroup_store(" in source
    assert "simdgroup_barrier(" in source
    assert "threadgroup_barrier(" not in source
    assert ("half" in source) is (dtype == "f16")


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("transpose_right", [False, True])
def test_device_products_keep_fp32_accumulation_and_explicit_scratch(dtype, transpose_right):
    metal = lower(_trace(_device_product, dtype=dtype, TRANSPOSE_RIGHT=transpose_right))
    products = [
        operation for operation in _walk(metal.ops) if isinstance(operation, mir.MFragmentDot)
    ]
    assert len(products) == 1
    assert products[0].left.type.dtype == dtype
    assert products[0].right.type.dtype == dtype
    assert products[0].accumulator.type.dtype == "f32"
    assert "simdgroup_multiply_accumulate(" in emit(metal)


@pytest.mark.parametrize(
    "case",
    [
        "missing_load_scratch",
        "missing_store_scratch",
        "not_tensor_scratch",
        "device_scratch",
        "offset_scratch",
        "scratch_dtype",
        "scratch_shape",
        "scratch_strides",
        "read_scratch",
        "write_scratch",
        "scalar_scratch",
        "scalar_store_scratch",
        "shared_scratch",
        "shared_store_scratch",
        "scratch_scalar_write",
        "scratch_scalar_read",
        "scratch_matrix_read",
        "scratch_matrix_write",
    ],
)
def test_device_fragment_scratch_contract_fails_closed(case):
    with pytest.raises((LoweringError, TypeError, ValueError), match="scratch"):
        lower(_trace(_device_contract, CASE=case))


@pytest.mark.parametrize(
    "case,match",
    [
        ("dynamic_stride", "stride"),
        ("zero_stride", "stride"),
        ("negative_stride", "stride"),
        ("irregular_strides", "row-major|stride"),
        ("divergent_origin", "uniform"),
        ("divergent_base", "uniform"),
        ("divergent_shape", "uniform"),
        ("divergent_control", "uniform"),
    ],
)
def test_device_fragment_memory_and_control_contracts_fail_closed(case, match):
    with pytest.raises((LoweringError, TypeError, ValueError), match=match):
        lower(_trace(_device_contract, CASE=case))


@pytest.mark.parametrize(
    "case",
    [
        "tile_read_write",
        "tile_store_then_tile_load",
        "scalar_store_then_tile_load",
        "tile_load_then_scalar_store",
        "tile_store_then_scalar_load",
        "offset_alias",
        "write_in_loop",
    ],
)
def test_device_fragment_read_write_aliases_require_more_than_threadgroup_barriers(case):
    with pytest.raises(LoweringError, match="disjoint allocation roots"):
        lower(_trace(_device_alias_contract, CASE=case))


def test_device_fragment_access_modes_do_not_replace_actual_root_access_analysis():
    function = _trace(_device_alias_contract, CASE="distinct_roots")
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Output"]
    lower(function)


def test_device_fragment_restrictions_do_not_ban_unrelated_scalar_in_place_roots():
    lower(_trace(_device_alias_contract, CASE="unrelated_scalar_in_place"))


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "transpose_source,transpose_output",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize(
    "shape,store_origin",
    [
        ((16, 8, 0, 0, 16, 8), (0, 0)),
        ((11, 5, 0, 0, 16, 8), (0, 0)),
        ((11, 8, 0, 0, 13, 8), (0, 0)),
        ((9, 9, -3, -2, 13, 5), (-2, -1)),
        ((0, 0, 0, 0, 16, 8), (0, 0)),
        ((7, 13, 5, 8, 9, 3), (0, 0)),
        ((19, 10, 2, 1, 0, 8), (0, 0)),
    ],
)
def test_gpu_device_copy_transposes_masks_tails_and_preserves_guards(
    dtype, transpose_source, transpose_output, shape, store_origin
):
    rows, columns, row_origin, column_origin, output_rows, output_columns = shape
    source_base, output_base = 17, 29
    source = np.full(24 * 24 + 64, np.nan, dtype=dtype)
    source_strides = (1, 24) if transpose_source else (24, 1)
    output_strides = (1, 24) if transpose_output else (24, 1)
    for row in range(rows):
        for column in range(columns):
            source[source_base + row * source_strides[0] + column * source_strides[1]] = (
                row * 0.5 + column * 0.125 - 2.0
            )
    original = source.copy()
    actual = np.full_like(source, -119.0)
    expected = actual.copy()
    for row in range(16):
        for column in range(8):
            output_row, output_column = row + store_origin[0], column + store_origin[1]
            if not (0 <= output_row < output_rows and 0 <= output_column < output_columns):
                continue
            source_row, source_column = row + row_origin, column + column_origin
            value = 0.0
            if 0 <= source_row < rows and 0 <= source_column < columns:
                value = source[
                    source_base + source_row * source_strides[0] + source_column * source_strides[1]
                ]
            expected[
                output_base + output_row * output_strides[0] + output_column * output_strides[1]
            ] = value
    control = np.array([*shape, source_base, output_base], dtype=np.int32)
    _device_copy[(1,)].prepare(
        source,
        control,
        actual,
        DTYPE="f16" if dtype == np.float16 else "f32",
        TRANSPOSE_SOURCE=transpose_source,
        TRANSPOSE_OUTPUT=transpose_output,
        STORE_ROW_OFFSET=store_origin[0],
        STORE_COLUMN_OFFSET=store_origin[1],
        BLOCK=64,
        STRICT_MATH=True,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("transpose_right", [False, True])
@pytest.mark.parametrize("shape", [(16, 16, 8), (13, 11, 9), (1, 1, 1), (9, 0, 7)])
def test_gpu_device_product_zeros_nan_padding_and_preserves_output_tails(
    dtype, transpose_right, shape
):
    rows, features, columns = shape
    generator = np.random.default_rng(49128)
    left_values = generator.normal(0, 0.3, (rows, features)).astype(dtype)
    right_values = generator.normal(0, 0.3, (features, columns)).astype(dtype)
    left = np.full((24, 24), np.nan, dtype=dtype)
    right = np.full((24, 24), np.nan, dtype=dtype)
    left[:rows, :features] = left_values
    if transpose_right:
        right[:columns, :features] = right_values.T
    else:
        right[:features, :columns] = right_values
    original_left, original_right = left.copy(), right.copy()
    actual = np.full((24, 24), -119.0, dtype=dtype)
    _device_product[(metile.cdiv(rows, 16), metile.cdiv(columns, 8))].prepare(
        left,
        right,
        np.array(shape, dtype=np.int32),
        actual,
        DTYPE="f16" if dtype == np.float16 else "f32",
        TRANSPOSE_RIGHT=transpose_right,
        BLOCK=64,
        STRICT_MATH=True,
        SCHEDULE=metile.Schedule(backend="simdgroup_inline"),
    )
    expected = (left_values.astype(np.float64) @ right_values.astype(np.float64)).astype(dtype)
    tolerance = 2e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(actual[:rows, :columns], expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(actual[rows:], -119.0)
    np.testing.assert_array_equal(actual[:rows, columns:], -119.0)
    np.testing.assert_array_equal(left, original_left)
    np.testing.assert_array_equal(right, original_right)
