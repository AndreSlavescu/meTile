import numpy as np
import pytest

import metile


@metile.kernel
def descriptor_strided_copy(
    source, output, rows, columns, source_stride, output_stride, BLOCK: metile.constexpr
):
    inputs = metile.tensor(source, shape=(rows, columns), strides=(source_stride, 2), access="read")
    outputs = metile.tensor(
        output, shape=(rows, columns), strides=(output_stride, 1), access="write"
    )
    row = metile.program_id(1)
    columns_tile = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    outputs.store((row, columns_tile), inputs.load((row, columns_tile)))


@pytest.mark.parametrize("columns", [1, 31, 37, 65])
def test_tensor_coordinates_keep_axes_strides_and_ragged_bounds(columns):
    rows = 3
    source_stride = columns * 2 + 3
    output_stride = columns + 5
    source = np.arange(rows * source_stride, dtype=np.float32)
    output = np.full(rows * output_stride, -91, dtype=np.float32)
    descriptor_strided_copy[(metile.cdiv(columns, 32), rows)](
        source, output, rows, columns, source_stride, output_stride, BLOCK=32
    )
    expected = np.full_like(output, -91)
    for row in range(rows):
        expected[row * output_stride : row * output_stride + columns] = source[
            row * source_stride : row * source_stride + columns * 2 : 2
        ]
    np.testing.assert_array_equal(output, expected)


@metile.kernel
def descriptor_independent_bounds(
    source, output, input_width, output_width, BLOCK: metile.constexpr
):
    inputs = metile.tensor(source, shape=(input_width,), access="read")
    outputs = metile.tensor(output, shape=(output_width,), access="write")
    for start in metile.tile_range(0, output_width, BLOCK):
        positions = start + metile.arange(0, BLOCK)
        values = inputs.load((positions - 1,), other=-17.0)
        outputs.store((positions,), values)


@pytest.mark.parametrize("input_width", [0, 37])
def test_each_memory_access_keeps_its_own_bounds_and_fill_value(input_width):
    source = np.arange(max(1, input_width), dtype=np.float32)
    output = np.full(77, np.nan, dtype=np.float32)
    descriptor_independent_bounds[(1,)](source, output, input_width, 70, BLOCK=32)
    expected = np.full(70, -17.0, dtype=np.float32)
    expected[1 : input_width + 1] = source[:input_width]
    np.testing.assert_array_equal(output[:70], expected)
    assert np.isnan(output[70:]).all()


@metile.kernel
def descriptor_offset_arange(source, output, width, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(width,), access="read")
    outputs = metile.tensor(output, shape=(width,), access="write")
    positions = metile.arange(-2, BLOCK - 2)
    outputs.store((positions + 2,), inputs.load((positions,), other=19.0))


def test_arange_origin_is_not_discarded():
    source = np.arange(32, dtype=np.float32)
    output = np.zeros_like(source)
    descriptor_offset_arange[(1,)](source, output, 32, BLOCK=32)
    np.testing.assert_array_equal(output, np.r_[19.0, 19.0, source[:-2]])


@metile.kernel
def descriptor_reverse(source, output, width, BLOCK: metile.constexpr):
    inputs = metile.tensor(source + (width - 1), shape=(width,), strides=(-1,), access="read")
    outputs = metile.tensor(output, shape=(width,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    outputs.store((positions,), inputs.load((positions,)))


def test_negative_strides_use_signed_offset_arithmetic():
    source = np.arange(37, dtype=np.float32)
    output = np.zeros_like(source)
    descriptor_reverse[(2,)](source, output, source.size, BLOCK=32)
    np.testing.assert_array_equal(output, source[::-1])


@metile.kernel
def descriptor_shared_fill(source, output, BLOCK: metile.constexpr):
    scratch = metile.shared(BLOCK)
    staged = metile.tensor(
        scratch, shape=(BLOCK - 1,), access="readwrite", address_space="threadgroup"
    )
    inputs = metile.tensor(source, shape=(BLOCK,), access="read")
    outputs = metile.tensor(output, shape=(BLOCK,), access="write")
    positions = metile.arange(0, BLOCK)
    staged.store((positions,), inputs.load((positions,)))
    metile.barrier()
    outputs.store((positions,), staged.load((positions,), other=-23.0))


def test_masked_shared_access_does_not_skip_the_barrier():
    source = np.arange(32, dtype=np.float32)
    output = np.zeros_like(source)
    descriptor_shared_fill[(1,)](source, output, BLOCK=32)
    np.testing.assert_array_equal(output, np.r_[source[:-1], -23.0])
