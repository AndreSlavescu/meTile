import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import LoweringError, lower
from metile.compiler.passes import fold_constants, split_elementwise_loops, vectorize_elementwise
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType

_DTYPES = {"f32": np.float32, "i32": np.int32, "u32": np.uint32}
_CONVERSIONS = [(source, target) for source in _DTYPES for target in _DTYPES if source != target]
_PATTERNS = np.array(
    [
        0x00000000,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7FA12345,
        0xFFC54321,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0xFFFFFFFF,
        0x01234567,
        0xDEADBEEF,
    ],
    dtype=np.uint32,
)


@metile.kernel
def _reinterpret(
    Source,
    Destination,
    size,
    *,
    TARGET: metile.constexpr,
    MODE: metile.constexpr = "tile",
    BLOCK: metile.constexpr = 128,
):
    source = metile.tensor(Source, shape=(size,), access="read")
    destination = metile.tensor(Destination, shape=(size,), access="write")
    if MODE == "scalar":
        positions = metile.program_id(0) * BLOCK + metile.thread_id()
    else:
        layout = (
            metile.ThreadLayout.identity(BLOCK, elements_per_thread=4)
            if MODE == "registers"
            else None
        )
        positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK, layout=layout)
    destination.store((positions,), metile.bitcast(source.load((positions,)), TARGET))


@metile.kernel
def _round_trip(Source, Destination, *, BLOCK: metile.constexpr = 32):
    source = metile.tensor(Source, shape=(256,), access="read")
    destination = metile.tensor(Destination, shape=(256,), access="write")
    for offset in metile.tile_range(0, 256, 32):
        positions = offset + metile.arange(0, 32)
        bits = metile.bitcast(source.load((positions,)), "u32")
        destination.store((positions,), metile.bitcast(bits, "f32"))


def _trace(source_dtype, target_dtype, mode):
    with TracingContext("reinterpret") as context:
        context.func.constexprs.update(BLOCK=128, STRICT_MATH=True)
        parameters = [
            ("Source", PtrType(source_dtype)),
            ("Destination", PtrType(target_dtype)),
            ("size", I32),
        ]
        context.func.params = [tir.Param(name, datatype) for name, datatype in parameters]
        arguments = [TracingProxy(tir.Value(name, datatype)) for name, datatype in parameters]
        _reinterpret.fn(*arguments, TARGET=target_dtype, MODE=mode, BLOCK=128)
    _mark_outputs(context.func)
    return context.func


@pytest.mark.parametrize("source_dtype,target_dtype", _CONVERSIONS)
@pytest.mark.parametrize("mode", ["scalar", "tile", "registers"])
def test_bitcast_lowers_to_metal_reinterpretation_without_numeric_conversion(
    source_dtype, target_dtype, mode
):
    function = _trace(source_dtype, target_dtype, mode)
    metal = lower(function)
    split_elementwise_loops(metal)
    vectorize_elementwise(metal)
    fold_constants(metal)
    emitted = emit(metal)
    target = {"f32": "float", "i32": "int", "u32": "uint"}[target_dtype]
    assert f"as_type<{target}>" in emitted
    assert [parameter.name for parameter in function.params if parameter.is_output] == [
        "Destination"
    ]


def test_bitcast_constants_are_not_replaced_by_numeric_constant_folding():
    function = mir.MFunction("constant_reinterpretation")
    source = function.add_op(mir.MConstant(value=0x80000000, dtype="u32"), "bits")
    operation = mir.MBitcast(value=source, target_dtype="f32")
    result = function.add_op(operation, "negative_zero")
    fold_constants(function)
    assert result.defining_op is operation
    assert operation in function.ops


def test_bitcast_round_trip_survives_vectorized_float_memory_loops():
    with TracingContext("vector_reinterpretation") as context:
        context.func.constexprs.update(BLOCK=32, STRICT_MATH=True)
        context.func.params = [
            tir.Param(name, PtrType("f32")) for name in ("Source", "Destination")
        ]
        _round_trip.fn(
            *[
                TracingProxy(tir.Value(parameter.name, parameter.type))
                for parameter in context.func.params
            ],
            BLOCK=32,
        )
    _mark_outputs(context.func)
    metal = lower(context.func)
    split_elementwise_loops(metal)
    vectorize_elementwise(metal)
    fold_constants(metal)
    emitted = emit(metal)
    assert "as_type<uint4>" in emitted
    assert "as_type<float4>" in emitted


def test_bitcast_cannot_reinterpret_opaque_matrix_fragment_element_ownership():
    with TracingContext("matrix_reinterpretation") as context:
        context.func.constexprs.update(
            BLOCK=32, SCHEDULE=metile.Schedule(backend="simdgroup_inline")
        )
        matrix = metile.tensor(metile.shared(64, dtype="f32"), shape=(8, 8), block_shape=(8, 8))
        metile.bitcast(matrix.load((0, 0)), "u32")
    with pytest.raises(LoweringError, match=r"f16/f32|fragment|matrix"):
        lower(context.func)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("source_dtype,target_dtype", _CONVERSIONS)
@pytest.mark.parametrize("mode", ["scalar", "tile", "registers"])
@pytest.mark.parametrize("size", [37, 257])
def test_gpu_bitcast_preserves_every_bit_including_nan_payloads_and_signed_zero(
    source_dtype, target_dtype, mode, size
):
    patterns = np.resize(_PATTERNS, size)
    source = patterns.view(_DTYPES[source_dtype]).copy()
    destination = np.zeros(size, dtype=_DTYPES[target_dtype])
    _reinterpret[(metile.cdiv(size, 128),)].prepare(
        source, destination, size, TARGET=target_dtype, MODE=mode, BLOCK=128, STRICT_MATH=True
    )
    np.testing.assert_array_equal(destination.view(np.uint32), patterns)
    np.testing.assert_array_equal(source.view(np.uint32), patterns)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
def test_gpu_vectorized_bitcast_round_trip_preserves_subnormals_and_nan_payloads():
    patterns = np.resize(_PATTERNS, 256)
    source = patterns.view(np.float32).copy()
    destination = np.zeros(256, dtype=np.float32)
    _round_trip[(1,)].prepare(source, destination, BLOCK=32, STRICT_MATH=True)
    np.testing.assert_array_equal(destination.view(np.uint32), patterns)
