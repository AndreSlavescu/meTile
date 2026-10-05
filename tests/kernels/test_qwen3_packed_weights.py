import inspect
import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels import qwen3_staged as stages
from metile_kernels.megakernels.qwen3 import qwen3_layer_offsets

PACKED_STAGES = (
    stages.qwen3_staged_qkv,
    stages.qwen3_staged_gemv,
    stages.qwen3_staged_residual,
    stages.qwen3_staged_swiglu,
)


def _trace(stage, *, weight_dtype=None, **overrides):
    geometry = (
        dict(
            HIDDEN=32,
            INTERMEDIATE=64,
            QUERY_HEADS=2,
            KV_HEADS=1,
            HEAD_DIM=32,
            ROWS=17,
            COLUMNS=34,
            WEIGHT_OFFSET=2,
            WEIGHT_STRIDE=582,
            BLOCK=128,
            STORAGE_DTYPE="f32",
            PACKED_WEIGHTS=True,
        )
        | overrides
    )
    signature = inspect.signature(stage.fn)
    constants = {
        name: geometry[name]
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }
    parameters = []
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if name == "layer":
            dtype = I32
        elif name in ("Weights", "LayerWeights"):
            dtype = PtrType(weight_dtype or ("u32" if geometry["PACKED_WEIGHTS"] else "f32"))
        else:
            dtype = PtrType("f32")
        parameters.append((name, dtype))
    with TracingContext(stage.name) as context:
        context.func.constexprs.update(constants, STRICT_MATH=True)
        context.func.params = [tir.Param(name, dtype) for name, dtype in parameters]
        stage.fn(*[TracingProxy(tir.Value(name, dtype)) for name, dtype in parameters], **constants)
    _mark_outputs(context.func)
    return context.func


@pytest.mark.parametrize("stage", PACKED_STAGES, ids=lambda stage: stage.name)
@pytest.mark.parametrize("packed", [False, True])
def test_packed_weight_stages_emit_explicit_bit_reinterpretation(stage, packed):
    function = _trace(stage, PACKED_WEIGHTS=packed)
    source = emit(lower(function))
    assert ("as_type<float>" in source) == packed
    assert "half" not in source
    assert "fma(" not in source
    assert "threadgroup_barrier" not in source
    assert "simd_sum" in source
    weight_parameter = next(
        parameter for parameter in function.params if parameter.name in ("Weights", "LayerWeights")
    )
    assert weight_parameter.type.dtype == ("u32" if packed else "f32")
    assert not weight_parameter.is_output


@pytest.mark.parametrize("stage", PACKED_STAGES, ids=lambda stage: stage.name)
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"PACKED_WEIGHTS": 1}, "PACKED_WEIGHTS must be bool"),
        ({"STORAGE_DTYPE": "f16"}, "FP32 source and output"),
    ],
)
def test_packed_weight_stages_reject_invalid_precision_policy(stage, overrides, match):
    with pytest.raises(ValueError, match=match):
        _trace(stage, **overrides)


@pytest.mark.parametrize("stage", PACKED_STAGES, ids=lambda stage: stage.name)
@pytest.mark.parametrize("weight_dtype", ["i32", "f16", "f32"])
def test_packed_weight_stages_require_unsigned_words(stage, weight_dtype):
    with pytest.raises(ValueError, match="uint32 weight storage"):
        _trace(stage, weight_dtype=weight_dtype)


@pytest.mark.parametrize("stage", [stages.qwen3_staged_gemv, stages.qwen3_staged_residual])
@pytest.mark.parametrize(
    "overrides",
    [
        {"COLUMNS": 33},
        {"WEIGHT_OFFSET": 1},
        {"WEIGHT_STRIDE": 581},
        {"WEIGHT_OFFSET": -2},
        {"WEIGHT_STRIDE": True},
    ],
)
def test_packed_weight_stages_require_even_logical_addressing(stage, overrides):
    with pytest.raises(ValueError, match="even columns, weight offsets and strides"):
        _trace(stage, **overrides)


def _bf16_values(generator, shape):
    values = generator.normal(0, 0.125, shape).astype(np.float32)
    return (values.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _pack(values):
    bits = np.ascontiguousarray(values).view(np.uint32).ravel()
    assert bits.size % 2 == 0
    assert not np.any(bits & np.uint32(65535))
    words = bits >> np.uint32(16)
    return words[0::2] | (words[1::2] << np.uint32(16))


@pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")
@pytest.mark.parametrize("stage", PACKED_STAGES, ids=lambda stage: stage.name)
@pytest.mark.parametrize("block", [32, 128, 256])
def test_gpu_packed_weights_preserve_all_staged_projection_bits(stage, block):
    generator = np.random.default_rng(33985)
    matrix_stage = stage in (stages.qwen3_staged_gemv, stages.qwen3_staged_residual)
    if matrix_stage:
        rows, columns, offset, stride = 17, 34, 2, 582
        weights = _bf16_values(generator, (2 * stride,))
        constants = dict(ROWS=rows, COLUMNS=columns, WEIGHT_OFFSET=offset, WEIGHT_STRIDE=stride)
    else:
        columns = 32
        layout = qwen3_layer_offsets(32, 64, 2, 1, 32)
        weights = _bf16_values(generator, (2 * layout["layer_size"],))
        rows = 128 if stage is stages.qwen3_staged_qkv else 64
        constants = dict(HIDDEN=32, INTERMEDIATE=64, QUERY_HEADS=2, KV_HEADS=1, HEAD_DIM=32)
    source = metile.Buffer(data=generator.normal(size=columns).astype(np.float32))
    original_weights = metile.Buffer(data=weights)
    packed_weights = metile.Buffer(data=_pack(weights))
    residual = metile.Buffer(data=generator.normal(size=rows).astype(np.float32))
    reference = metile.Buffer.empty((rows,), dtype=np.float32)
    output = metile.Buffer.empty((rows,), dtype=np.float32)
    launcher = stage[(metile.cdiv(rows, block // 32),)]
    constants.update(BLOCK=block, STORAGE_DTYPE="f32", STRICT_MATH=True)

    def arguments(weight_buffer, destination):
        if stage is stages.qwen3_staged_residual:
            return source, weight_buffer, residual, destination, 1
        return source, weight_buffer, destination, 1

    launcher.prepare(*arguments(original_weights, reference), PACKED_WEIGHTS=False, **constants)
    launcher.prepare(*arguments(packed_weights, output), PACKED_WEIGHTS=True, **constants)
    np.testing.assert_array_equal(output.numpy().view(np.uint32), reference.numpy().view(np.uint32))
    np.testing.assert_array_equal(original_weights.numpy(), weights)
    np.testing.assert_array_equal(packed_weights.numpy(), _pack(weights))
