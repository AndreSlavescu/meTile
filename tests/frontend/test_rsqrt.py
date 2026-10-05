import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import ScalarType, TileType


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("shape", [(), (37,), (8, 8)])
def test_rsqrt_traces_one_unary_and_preserves_shape_and_storage_dtype(dtype, shape):
    datatype = TileType(shape, dtype) if shape else ScalarType(dtype)
    with TracingContext("reciprocal_square_root") as context:
        context.func.params.append(tir.Param("source", datatype))
        source = TracingProxy(tir.Value("source", datatype))
        result = metile.rsqrt(source)
    assert "rsqrt" in metile.__all__
    assert result._value.type == datatype
    assert len(context.func.ops) == 1
    operation = context.func.ops[0]
    assert isinstance(operation, tir.Unary)
    assert operation.op == "rsqrt"
    assert operation.operand is source._value


def test_rsqrt_vjp_reuses_the_primal_result_without_division_or_recomputation():
    with TracingContext("reciprocal_square_root_vjp") as context:
        context.func.params.append(tir.Param("source", ScalarType("f32")))
        source = TracingProxy(tir.Value("source", ScalarType("f32")))
        output = metile.rsqrt(source)
        gradient = metile.vjp(output, source, 2.0)
    assert gradient._value.type == source._value.type
    operations = context.func.ops
    assert (
        sum(
            isinstance(operation, tir.Unary) and operation.op == "rsqrt" for operation in operations
        )
        == 1
    )
    assert not any(
        isinstance(operation, tir.BinOp) and operation.op == "div" for operation in operations
    )
