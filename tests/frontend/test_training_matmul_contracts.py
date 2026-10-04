import numpy as np
import pytest

from metile.backends.training_matmul import matmul_forward
from metile.runtime.metal_device import MetalDevice


@pytest.mark.parametrize(
    "options", [{"max_workspace_bytes": 1}, {"max_workspace_bytes": True}, {"activation": "gelu"}]
)
def test_training_matmul_rejects_invalid_options_before_device(monkeypatch, options):
    def fail():
        raise AssertionError("validation must precede allocation")

    monkeypatch.setattr(MetalDevice, "get", staticmethod(fail))
    with pytest.raises(ValueError):
        matmul_forward(np.ones((3, 4), np.float32), np.ones((4, 5), np.float32), **options)
