import sys

import numpy as np
import pytest

import metile
from metile_kernels.megakernels.qwen3 import _rms_norm, _silu


@metile.kernel
def _weighted_norm(Input, Weight, Output, WIDTH: metile.constexpr):
    source = metile.tensor(Input, shape=(WIDTH,), access="read")
    weights = metile.tensor(Weight, shape=(1, WIDTH), access="read")
    output = metile.tensor(Output, shape=(WIDTH,), access="write")
    _rms_norm(
        source, output, weights, 0, WIDTH, 1e-6, metile.thread_id(), metile.simd_lane_id(), 32
    )


@metile.kernel
def _silu_values(Input, Output, SIZE, BLOCK: metile.constexpr):
    source = metile.tensor(Input, shape=(SIZE,), access="read")
    output = metile.tensor(Output, shape=(SIZE,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    output.store((positions,), _silu(source.load((positions,))))


@pytest.fixture
def mlx_core():
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    return pytest.importorskip("mlx.core")


def test_weighted_norm_rounds_normalized_half_before_multiplying_gamma(mlx_core):
    generator = np.random.default_rng(823)
    values = generator.normal(size=32).astype(np.float16)
    weights = generator.uniform(0.5, 4.0, size=32).astype(np.float16)
    output = np.empty(32, dtype=np.float32)
    _weighted_norm[(1,)].prepare(
        values.astype(np.float32), weights, output, WIDTH=32, BLOCK=32, STRICT_MATH=True
    )
    expected = np.array(
        mlx_core.fast.rms_norm(mlx_core.array(values), mlx_core.array(weights), 1e-6)
    )
    np.testing.assert_array_equal(output, expected.astype(np.float32))
    inverse = 1.0 / np.sqrt(np.mean(values.astype(np.float32) ** 2) + 1e-6)
    wrong = (values.astype(np.float32) * inverse * weights.astype(np.float32)).astype(np.float16)
    assert np.count_nonzero(expected != wrong) >= 4


def test_silu_preserves_half_sigmoid_and_product_rounding(mlx_core):
    values = np.linspace(-10, 10, 20000).astype(np.float16)
    output = np.empty_like(values)
    _silu_values[(metile.cdiv(values.size, 128),)].prepare(
        values, output, values.size, BLOCK=128, STRICT_MATH=True
    )
    inputs = mlx_core.array(values)
    expected = np.array(inputs * mlx_core.sigmoid(inputs))
    np.testing.assert_array_equal(output, expected)
