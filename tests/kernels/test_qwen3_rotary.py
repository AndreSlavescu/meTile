import math
import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType
from metile_kernels.megakernels.qwen3_rotary import qwen3_rotary_table


def _trace(**overrides):
    constants = dict(
        HEAD_DIM=128, MAX_CONTEXT=5120, LOG2_BASE=float(np.float32(math.log2(1e6))), BLOCK=128
    )
    constants.update(overrides)
    with TracingContext("qwen3_rotary_table") as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param("Rotary", PtrType("f32"))]
        qwen3_rotary_table.fn(TracingProxy(tir.Value("Rotary", PtrType("f32"))), **constants)
    _mark_outputs(context.func)
    return context.func


def test_rotary_table_uses_public_exp2_and_fast_trigonometric_intrinsics():
    function = _trace()
    source = emit(lower(function))
    assert "exp2(" in source
    assert "fast::cos(" in source
    assert "fast::sin(" in source
    assert "Rotary" in {parameter.name for parameter in function.params if parameter.is_output}
    assert tuple(dimension.defining_op.value for dimension in function.tensors[0].shape) == (
        5120,
        64,
        2,
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"HEAD_DIM": 3}, "HEAD_DIM"),
        ({"HEAD_DIM": False}, "HEAD_DIM"),
        ({"MAX_CONTEXT": 0}, "MAX_CONTEXT"),
        ({"MAX_CONTEXT": 2**30}, "32-bit"),
        ({"LOG2_BASE": math.nan}, "LOG2_BASE"),
        ({"LOG2_BASE": True}, "LOG2_BASE"),
        ({"BLOCK": 33}, "BLOCK"),
    ],
)
def test_rotary_geometry_is_validated_before_lowering(overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(**overrides)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("head_dim,capacity", [(32, 7), (128, 5120)])
def test_gpu_rotary_table_matches_native_mlx_basis_at_long_positions(head_dim, capacity):
    mx = pytest.importorskip("mlx.core")
    half = head_dim // 2
    storage = np.full(capacity * head_dim + 19, -17.0, dtype=np.float32)
    qwen3_rotary_table[(metile.cdiv(capacity * half, 128) + 1,)].prepare(
        storage,
        HEAD_DIM=head_dim,
        MAX_CONTEXT=capacity,
        LOG2_BASE=float(np.float32(math.log2(1e6))),
        BLOCK=128,
        STRICT_MATH=True,
    )
    actual = storage[: capacity * head_dim].reshape(capacity, half, 2)
    basis = mx.concatenate(
        (mx.ones((1, 1, capacity, half)), mx.zeros((1, 1, capacity, half))), axis=-1
    )
    rotated = mx.fast.rope(basis, dims=head_dim, traditional=False, base=1e6, scale=1.0, offset=0)
    mx.eval(rotated)
    native = np.asarray(rotated)[0, 0]
    expected = np.stack((native[:, :half], native[:, half:]), axis=-1)
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
    np.testing.assert_array_equal(storage[capacity * head_dim :], -17.0)
