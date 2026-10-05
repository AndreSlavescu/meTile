import sys

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.lowering.common import LoweringError
from metile.frontend.kernel import _mark_outputs
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType
from metile_kernels.megakernels.qwen3_prefill_projection import qwen3_prefill_projection


def _trace(dtype="f32", **overrides):
    constants = {
        "BLOCK_M": 32,
        "BLOCK_N": 64,
        "BLOCK_K": 16,
        "SCHEDULE": metile.Schedule(backend="simdgroup"),
        "RELAXED_PRECISION": False,
        "STRICT_MATH": True,
    }
    constants.update(overrides)
    parameters = [
        (name, PtrType(dtype) if name in ("Source", "Weight", "Output") else I32)
        for name in ("Source", "Weight", "Output", "M", "N", "K")
    ]
    with TracingContext(qwen3_prefill_projection.name) as context:
        context.func.constexprs.update(constants)
        context.func.params = [tir.Param(name, kind) for name, kind in parameters]
        proxies = [TracingProxy(tir.Value(name, kind)) for name, kind in parameters]
        qwen3_prefill_projection.fn(
            *proxies,
            **{name: constants[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K")},
        )
    _mark_outputs(context.func)
    return context.func


@pytest.mark.parametrize(
    "block_rows,block_columns,block_reduction",
    [(16, 64, 16), (32, 64, 16), (32, 64, 32), (64, 64, 16), (64, 64, 32), (64, 128, 16)],
)
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_prefill_projection_uses_compiler_matrix_tiles_with_fp32_accumulation(
    block_rows, block_columns, block_reduction, dtype
):
    function = _trace(dtype, BLOCK_M=block_rows, BLOCK_N=block_columns, BLOCK_K=block_reduction)
    metal = lower(function)
    source = emit(metal)
    assert metal.schedule_plan.backend == "simdgroup"
    assert metal.schedule_plan.tile_shape == (block_rows, block_columns, block_reduction)
    assert "simdgroup_matrix<float, 8, 8> acc" in source
    assert "simdgroup_multiply_accumulate" in source
    if dtype == "f32":
        assert "half" not in source
    assert [tensor.ptr.name for tensor in function.tensors] == ["Source", "Weight", "Output"]
    assert all(tensor.ptr.defining_op is None for tensor in function.tensors)
    assert [parameter.name for parameter in function.params if parameter.is_output] == ["Output"]
    loops = [operation for operation in function.ops if isinstance(operation, tir.ForRange)]
    assert len(loops) == 1
    assert loops[0].end.name == "K"
    assert any(isinstance(operation, mir.MSimdgroupAccDecl) for operation in metal.ops)


@pytest.mark.parametrize(
    "config",
    [
        {"BLOCK_M": 7},
        {"BLOCK_N": 31},
        {"BLOCK_K": 0},
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256},
    ],
)
def test_prefill_projection_rejects_unsupported_tile_contracts(config):
    with pytest.raises((ValueError, LoweringError)):
        lower(_trace(**config))


@pytest.fixture(params=["simdgroup", "tensor_ops"])
def backend(request):
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    from metile.runtime.metal_device import MetalDevice

    if request.param == "tensor_ops" and not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    return request.param


def _prepare(source, weights, output, backend, tiles=(32, 64, 16)):
    rows, reduction = source.shape
    columns = weights.shape[1]
    block_rows, block_columns, block_reduction = tiles
    launcher = qwen3_prefill_projection[
        (metile.cdiv(rows, block_rows), metile.cdiv(columns, block_columns))
    ]
    dispatcher = launcher.prepare(
        source,
        weights,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=block_rows,
        BLOCK_N=block_columns,
        BLOCK_K=block_reduction,
        SCHEDULE=metile.Schedule(
            backend=backend,
            num_simdgroups=2 if backend == "tensor_ops" and source.dtype == np.float16 else None,
        ),
        RELAXED_PRECISION=False,
        STRICT_MATH=True,
    )
    return launcher._last_compiled, dispatcher


@pytest.mark.parametrize(
    "rows,columns,reduction", [(32, 96, 128), (64, 192, 96), (128, 64, 128), (256, 128, 64)]
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_batched_projection_matches_original_transposed_weights(
    backend, rows, columns, reduction, dtype
):
    generator = np.random.default_rng(527)
    inputs = generator.normal(0, 0.125, (rows, reduction)).astype(dtype)
    original = generator.normal(0, 0.125, (columns, reduction)).astype(dtype)
    weight_pack = np.ascontiguousarray(original.T)
    source = metile.Buffer(data=inputs)
    weights = metile.Buffer(data=weight_pack)
    output = metile.Buffer.empty((rows, columns), dtype=dtype)
    compiled, dispatcher = _prepare(source, weights, output, backend)
    reference = (inputs.astype(np.float64) @ original.astype(np.float64).T).astype(dtype)
    tolerance = 0.001 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(output.numpy(), reference, rtol=tolerance, atol=tolerance)
    assert dispatcher.schedule_plan.backend == backend
    assert compiled.strict_math
    saved_weight_pack = weights.numpy().copy()
    source.numpy().fill(0)
    dispatcher()
    np.testing.assert_array_equal(output.numpy(), np.zeros_like(reference))
    np.testing.assert_array_equal(weights.numpy(), saved_weight_pack)


def test_gpu_strict_fp32_projection_preserves_bits_below_half_precision(backend):
    rows, columns, reduction = 32, 64, 64
    inputs = np.full((rows, reduction), 1 + 2**-12, dtype=np.float32)
    weight_pack = np.full((reduction, columns), 1 + 2**-13, dtype=np.float32)
    source = metile.Buffer(data=inputs)
    weights = metile.Buffer(data=weight_pack)
    output = metile.Buffer.empty((rows, columns))
    compiled, _dispatcher = _prepare(source, weights, output, backend)
    reference = (inputs.astype(np.float64) @ weight_pack.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(output.numpy(), reference, rtol=0, atol=2e-6)
    assert np.min(output.numpy()) > reduction
    assert "half" not in compiled.msl_source


@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 67, 35), (33, 129, 79)])
def test_gpu_simdgroup_projection_masks_partial_rows_columns_and_reduction(shape):
    if sys.platform != "darwin":
        pytest.skip("requires Apple Metal")
    rows, columns, reduction = shape
    generator = np.random.default_rng(382)
    inputs = generator.normal(0, 0.125, (rows, reduction)).astype(np.float32)
    weight_pack = generator.normal(0, 0.125, (reduction, columns)).astype(np.float32)
    source = metile.Buffer(data=inputs)
    weights = metile.Buffer(data=weight_pack)
    output = metile.Buffer.empty((rows, columns))
    _prepare(source, weights, output, "simdgroup")
    reference = (inputs.astype(np.float64) @ weight_pack.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(output.numpy(), reference, rtol=2e-6, atol=2e-6)
