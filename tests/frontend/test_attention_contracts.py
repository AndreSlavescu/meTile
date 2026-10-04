import numpy as np
import pytest

from metile.backends.attention import attention_backward, attention_forward
from metile.runtime.metal_device import MetalDevice


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    def fail():
        raise AssertionError("invalid metadata must be rejected before device access")

    monkeypatch.setattr(MetalDevice, "get", staticmethod(fail))


@pytest.mark.parametrize(
    "options",
    [
        {"scale": 0},
        {"scale": -1},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": True},
        {"scale": 1e-100},
        {"causal": 1},
        {"causal_offset": 1},
        {"causal": True, "causal_offset": 1 << 31},
        {"mask": np.ones((2, 2), dtype=np.float32)},
        {"mask": np.ones((9, 9), dtype=bool)},
    ],
)
def test_bad_attention_options_fail_before_device(options):
    inputs = np.ones((1, 2, 3, 32), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        attention_forward(inputs, inputs, inputs, **options)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 3, 32), np.float32),
        ((1, 2, 3, 33), np.float32),
        ((1, 2, 3, 32), np.float64),
        ((1, 2, 0, 32), np.float32),
    ],
)
def test_bad_attention_inputs_fail_before_device(shape, dtype):
    inputs = np.ones(shape, dtype=dtype)
    with pytest.raises(ValueError):
        attention_forward(inputs, inputs, inputs)


def test_attention_rejects_invalid_context():
    with pytest.raises(TypeError, match="AttentionContext"):
        attention_backward(None, np.ones((1,), dtype=np.float32))
