from dataclasses import replace

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.execution_report import execution_report, validate_materialized_schedule
from metile.compiler.lowering.common import LoweringError
from metile.compiler.ownership import validate_register_reductions
from metile.ir import metal_ir as mir
from tests.codegen.test_register_lowering_audit import _DTYPES, _checked_lower, _interpret, _trace

_BLOCK = 1024
_REGISTER_COUNTS = (2, 4, 8, 16, 32)


def _layout(elements, kind):
    identity = metile.ThreadLayout.identity(_BLOCK, elements_per_thread=elements)
    if kind == "striped":
        return identity
    if kind == "reversed_xor":
        return metile.ThreadLayout(
            tuple(reversed(identity.bit_order)), xor_mask=173, elements_per_thread=elements
        )
    register_bits = elements.bit_length() - 1
    thread_bits = identity.thread_count.bit_length() - 1
    grouped_bits = min(2, register_bits)
    order = (
        *range(thread_bits, thread_bits + grouped_bits),
        *range(thread_bits),
        *range(thread_bits + grouped_bits, thread_bits + register_bits),
    )
    return metile.ThreadLayout(order, elements_per_thread=elements)


def _check_resources(function, elements, reductions):
    validate_register_reductions(function)
    validate_materialized_schedule(function)
    report = execution_report(function, []).to_dict()
    threads = _BLOCK // elements
    groups = threads // 32
    assert function.threadgroup_size == (threads, 1, 1)
    assert function.schedule_plan.tile_shape == (_BLOCK,)
    assert all(value["elements_per_thread"] == elements for value in report["value_layouts"])
    assert all(value["shape"] == (_BLOCK,) for value in report["value_layouts"])
    assert len(report["register_reductions"]) == reductions
    source = emit(function)
    if reductions and groups > 1:
        assert function.schedule_plan.staging == "threadgroup"
        assert len(report["allocations"]) == reductions
        assert len({allocation["name"] for allocation in report["allocations"]}) == reductions
        assert all(allocation["elements"] == groups for allocation in report["allocations"])
        assert sum(allocation["bytes"] for allocation in report["allocations"]) == (
            reductions * groups * 4
        )
        assert source.count("threadgroup_barrier(") == 2 * reductions
        assert source.count(f"(slid < {groups}u)") == reductions
    else:
        assert function.schedule_plan.staging == "device"
        assert not report["allocations"]
        assert "threadgroup_barrier(" not in source
        assert all(record["scratch"] is None for record in report["register_reductions"])
    assert all(record["threads"] == threads for record in report["register_reductions"])
    assert all(
        record["elements_per_thread"] == elements for record in report["register_reductions"]
    )


@pytest.mark.parametrize("elements", _REGISTER_COUNTS)
@pytest.mark.parametrize("kind", ["striped", "blocked4", "reversed_xor"])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("optimized", [False, True])
def test_register_tiling_pointwise_preserves_row_offsets_masks_and_fills(
    elements, kind, dtype, optimized
):
    layout = _layout(elements, kind)

    def body(source, output, width):
        row = metile.program_id(0)
        inputs = metile.tensor(source + row * width, shape=(width,), access="read")
        outputs = metile.tensor(output + row * width, shape=(width,), access="write")
        indices = metile.arange(0, _BLOCK, layout=layout)
        previous = metile.cast(inputs.load((indices - 1,), other=-7), "f32")
        upcoming = metile.cast(inputs.load((indices + 3,), other=11), "f32")
        outputs.store((indices,), previous * 0.25 + upcoming * 2.0)

    lowered = _checked_lower(_trace(body, dtype=dtype), optimized=optimized)
    width = 1009
    rows = 3
    source = ((np.arange(rows * width) % 37) - 16).astype(_DTYPES[dtype])
    output = np.full(source.size + 8, -123, dtype=source.dtype)
    for row in range(rows):
        _interpret(lowered, {"source": source, "output": output, "width": width}, group=row)
    expected = []
    for values in source.reshape(rows, width).astype(np.float32):
        previous = np.r_[-7, values[:-1]]
        upcoming = np.r_[values[3:], 11, 11, 11]
        expected.append(previous * 0.25 + upcoming * 2)
    np.testing.assert_array_equal(output[: source.size], np.asarray(expected).ravel())
    np.testing.assert_array_equal(output[source.size :], -123)
    assert sum(isinstance(operation, mir.DeviceLoad) for operation in lowered.ops) == 2 * elements
    assert sum(isinstance(operation, mir.DeviceStore) for operation in lowered.ops) == elements
    _check_resources(lowered, elements, reductions=0)


@pytest.mark.parametrize("elements", _REGISTER_COUNTS)
@pytest.mark.parametrize("kind", ["striped", "blocked4", "reversed_xor"])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("optimized", [False, True])
def test_register_tiling_multiple_fp32_sums_keep_original_values_live(
    elements, kind, dtype, optimized
):
    layout = _layout(elements, kind)

    def body(source, output, width):
        row = metile.program_id(0)
        inputs = metile.tensor(source + row * width, shape=(width,), access="read")
        outputs = metile.tensor(output + row * width, shape=(width,), access="write")
        indices = metile.arange(0, _BLOCK, layout=layout)
        values = metile.cast(inputs.load((indices - 1,), other=-2), "f32")
        total = metile.sum(values)
        centered = values - total / _BLOCK
        square_sum = metile.sum(centered * centered)
        scaled = centered / metile.sqrt(square_sum / _BLOCK + 1e-5)
        result = metile.where(square_sum > 0, scaled, 0.0) + values * 0.25
        outputs.store((indices,), metile.convert_layout(result, layout))

    lowered = _checked_lower(_trace(body, dtype=dtype), optimized=optimized)
    width = 1009
    rows = 2
    source = np.linspace(-3, 4, rows * width).astype(_DTYPES[dtype])
    output = np.full(source.size + 8, -123, dtype=source.dtype)
    expected = []
    for row, values in enumerate(source.reshape(rows, width)):
        _interpret(lowered, {"source": source, "output": output, "width": width}, group=row)
        padded = np.full(_BLOCK, -2, dtype=np.float32)
        padded[1 : width + 1] = values.astype(np.float32)
        centered = padded - np.sum(padded, dtype=np.float32) / _BLOCK
        scaled = centered / np.sqrt(np.sum(centered * centered, dtype=np.float32) / _BLOCK + 1e-5)
        expected.append((scaled + padded * 0.25)[:width])
    tolerance = 2e-3 if dtype == "f16" else 2e-5
    np.testing.assert_allclose(
        output[: source.size],
        np.asarray(expected).ravel().astype(source.dtype),
        rtol=tolerance,
        atol=1e-5,
    )
    np.testing.assert_array_equal(output[source.size :], -123)
    assert sum(isinstance(operation, mir.DeviceLoad) for operation in lowered.ops) == elements
    assert sum(isinstance(operation, mir.DeviceStore) for operation in lowered.ops) == elements
    _check_resources(lowered, elements, reductions=2)


@pytest.mark.parametrize("elements", _REGISTER_COUNTS)
def test_register_tiling_record_must_match_materialized_ownership_geometry(elements):
    layout = _layout(elements, "striped")

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, _BLOCK, layout=layout)
        values = inputs.load((indices,))
        outputs.store((indices,), values + metile.sum(values))

    lowered = _checked_lower(_trace(body), optimized=True)
    changed = 4 if elements == 2 else 2
    lowered.register_reductions = (
        replace(lowered.register_reductions[0], elements_per_thread=changed),
    )
    with pytest.raises(LoweringError, match="register reduction"):
        validate_register_reductions(lowered)


def test_one_simdgroup_reduction_record_cannot_claim_unmaterialized_scratch():
    layout = _layout(32, "striped")

    def body(source, output, width):
        inputs = metile.tensor(source, shape=(width,), access="read")
        outputs = metile.tensor(output, shape=(width,), access="write")
        indices = metile.arange(0, _BLOCK, layout=layout)
        values = inputs.load((indices,))
        outputs.store((indices,), values + metile.sum(values))

    lowered = _checked_lower(_trace(body), optimized=True)
    lowered.register_reductions = (
        replace(lowered.register_reductions[0], scratch="unmaterialized"),
    )
    with pytest.raises(LoweringError, match="register reduction"):
        validate_register_reductions(lowered)
