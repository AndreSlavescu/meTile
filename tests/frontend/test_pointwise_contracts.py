import numpy as np
import pytest

from metile.backends.pointwise import (
    activation_backward,
    activation_forward,
    rope_backward,
    rope_forward,
)
from metile.runtime.metal_device import MetalDevice


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    def fail():
        raise AssertionError("validation must precede device access")

    monkeypatch.setattr(MetalDevice, "get", staticmethod(fail))


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "gelu"},
        {"up": np.ones((3,), dtype=np.float32)},
        {"up": np.ones((2, 4), dtype=np.float16)},
    ],
)
def test_activation_rejects_ambiguous_variant_or_bad_up(options):
    with pytest.raises((TypeError, ValueError)):
        activation_forward(np.ones((2, 4), dtype=np.float32), **options)


@pytest.mark.parametrize("options", [{"rotary_dim": 3}, {"rotary_dim": 6}, {"interleaved": 1}])
def test_rotary_rejects_bad_geometry(options):
    values = np.ones((2, 4), dtype=np.float32)
    table = np.ones((2, 2), dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        rope_forward(values, table, table, **options)


@pytest.mark.parametrize("backward", [activation_backward, rope_backward])
def test_pointwise_backward_rejects_wrong_context(backward):
    with pytest.raises(TypeError):
        backward(None, np.ones(4, dtype=np.float32))
