import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import MatrixFragmentType, PtrType, ScalarType, TileType


def _input(context, datatype):
    context.func.params.append(tir.Param("source", datatype))
    return TracingProxy(tir.Value("source", datatype))


@pytest.mark.parametrize("source_dtype", ["f32", "i32", "u32"])
@pytest.mark.parametrize("target_dtype", ["f32", "i32", "u32"])
@pytest.mark.parametrize("shape", [(), (37,), (8, 8)])
def test_bitcast_preserves_shape_and_only_changes_element_interpretation(
    source_dtype, target_dtype, shape
):
    source_type = TileType(shape, source_dtype) if shape else ScalarType(source_dtype)
    target_type = TileType(shape, target_dtype) if shape else ScalarType(target_dtype)
    with TracingContext("reinterpret") as context:
        source = _input(context, source_type)
        output = metile.bitcast(source, target_dtype)
    assert "bitcast" in metile.__all__
    assert output._value.type == target_type
    if source_dtype == target_dtype:
        assert output is source
        assert context.func.ops == []
    else:
        assert len(context.func.ops) == 1
        operation = context.func.ops[0]
        assert isinstance(operation, tir.Bitcast)
        assert operation.value is source._value
        assert operation.dtype == target_dtype


@pytest.mark.parametrize("dtype", ["f32", "i32", "u32"])
def test_bitcast_retains_explicit_register_ownership(dtype):
    layout = metile.ThreadLayout.identity(128, elements_per_thread=4)
    with TracingContext("owned") as context:
        source = _input(context, TileType((128,), "f32", layout))
        output = metile.bitcast(source, dtype)
    assert output._value.type == TileType((128,), dtype, layout)
    assert output._value.type.layout is layout


@pytest.mark.parametrize("target", ["f16", "bf16", "f64", "i64", "u8", "bool", None, 32, True])
def test_bitcast_rejects_unsupported_target_widths_and_types(target):
    with TracingContext("invalid_target") as context:
        source = _input(context, ScalarType("f32"))
        with pytest.raises(ValueError, match="target dtype"):
            metile.bitcast(source, target)


@pytest.mark.parametrize(
    "datatype",
    [
        ScalarType("f16"),
        ScalarType("bf16"),
        ScalarType("bool"),
        ScalarType("u8"),
        TileType((8,), "f16"),
        TileType((8,), "bool"),
        PtrType("f32"),
        PtrType("u32"),
        MatrixFragmentType("f32"),
    ],
)
def test_bitcast_rejects_unsupported_sources_including_pointers(datatype):
    with TracingContext("invalid_source") as context:
        source = _input(context, datatype)
        with pytest.raises(TypeError, match="scalar or tile"):
            metile.bitcast(source, "u32")


@pytest.mark.parametrize("literal", [0, 1.0, True, None, "f32", [1.0]])
def test_bitcast_requires_explicitly_typed_proxies_not_python_literals(literal):
    with TracingContext("literal"), pytest.raises(TypeError, match="explicitly typed"):
        metile.bitcast(literal, "u32")


def test_bitcast_accepts_explicit_scalar_without_numeric_cast_or_constant_fold():
    with TracingContext("constant") as context:
        source = metile.scalar(0x80000000, dtype="u32")
        output = metile.bitcast(source, "f32")
    assert output._value.type == ScalarType("f32")
    assert [type(operation) for operation in context.func.ops] == [tir.Constant, tir.Bitcast]


@pytest.mark.parametrize("target", ["f32", "u32"])
def test_bitcast_rejects_cross_trace_operands_even_when_no_conversion_is_needed(target):
    with TracingContext("first") as context:
        source = _input(context, ScalarType("f32"))
    with TracingContext("second"), pytest.raises(ValueError, match="cross tracing contexts"):
        metile.bitcast(source, target)


@pytest.mark.parametrize("shape", [(), (37,)])
def test_bitcast_round_trip_has_no_synthetic_autodiff_rule(shape):
    with TracingContext("nondifferentiable") as context:
        source = _input(context, TileType(shape, "f32") if shape else ScalarType("f32"))
        output = metile.bitcast(metile.bitcast(source, "u32"), "f32")
        with pytest.raises(NotImplementedError, match="bit reinterpretation"):
            metile.vjp(output, source, 1.0)
