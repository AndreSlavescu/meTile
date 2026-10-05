import numpy as np
import pytest

from benchmarks.megakernels.qwen3_staged_runtime import (
    _bfloat16_packable,
    _write_bfloat16_pairs,
)


def test_finite_bfloat16_patterns_roundtrip_including_signed_zero_and_subnormals():
    bits = np.arange(1 << 16, dtype=np.uint32) << np.uint32(16)
    bits = bits[(bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)]
    values = bits.view(np.float32)
    destination = np.zeros(values.size // 2, dtype=np.uint32)
    assert _bfloat16_packable(values)
    _write_bfloat16_pairs(values, destination)
    reconstructed = np.empty_like(bits)
    reconstructed[::2] = destination << np.uint32(16)
    reconstructed[1::2] = destination & np.uint32(0xFFFF0000)
    np.testing.assert_array_equal(reconstructed, bits)


@pytest.mark.parametrize(
    "values",
    [
        np.array([1, 2], dtype=np.float16),
        np.array([1, 2, 3], dtype=np.float32),
        np.array([1, np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32),
        np.array([1, np.inf], dtype=np.float32),
        np.array([np.nan, 0], dtype=np.float32),
        np.array([0, -np.inf], dtype=np.float32),
        np.ones((4,), dtype=np.float32)[::2],
    ],
)
def test_nonexact_or_unsupported_inputs_are_never_rounded(values):
    original = values.copy()
    destination = np.full(4, 0xDEADBEEF, dtype=np.uint32)
    assert not _bfloat16_packable(values)
    with pytest.raises(ValueError, match="lossless packing requires"):
        _write_bfloat16_pairs(values, destination)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(destination, np.full(4, 0xDEADBEEF, dtype=np.uint32))


@pytest.mark.parametrize(
    "dtype,shape", [(np.float32, (2,)), (np.uint32, (3,)), (np.uint32, (1, 2))]
)
def test_packed_destination_contract(dtype, shape):
    with pytest.raises(ValueError, match="one uint32 per pair"):
        _write_bfloat16_pairs(np.ones((4,), dtype=np.float32), np.empty(shape, dtype=dtype))


def test_validation_and_roundtrip_cross_chunk_boundary():
    values = np.tile(np.array([1, -2], dtype=np.float32), (1 << 19) + 1)
    destination = np.zeros(values.size // 2, dtype=np.uint32)
    _write_bfloat16_pairs(values, destination)
    np.testing.assert_array_equal(destination, np.full_like(destination, 0xC0003F80))
    values[-1] = np.nextafter(np.float32(1), np.float32(2))
    assert not _bfloat16_packable(values)


def test_packed_output_cannot_alias_immutable_weights():
    values = np.ones((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="one uint32 per pair"):
        _write_bfloat16_pairs(values, values.view(np.uint32)[:2])
    np.testing.assert_array_equal(values, np.ones((4,), dtype=np.float32))
