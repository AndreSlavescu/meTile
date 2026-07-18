import numpy as np

from metile.codegen.msl_emitter import emit
from metile.compiler.affine_quantized import (
    lower_affine_matmul,
    lower_affine_qmv,
    lower_affine_swiglu_qmv,
)
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import optimize_tile_schedules
from metile.frontend.kernel import CompiledKernel, FastDispatcher
from metile.ir import metal_ir as mir
from metile.runtime.buffer import MtileBuffer
from metile.runtime.metal_device import MetalDevice


def _walk(operations):
    for operation in operations:
        yield operation
        if isinstance(operation, mir.MForLoop):
            yield from _walk(operation.body)


def test_affine_qmv_decomposes_to_reusable_nax_primitives():
    function = decompose_nax_fragments(lower_affine_qmv("affine_ir", 64, 64))
    operations = tuple(_walk(function.ops))

    assert not any(isinstance(operation, mir.MNaxAffineRun) for operation in operations)
    assert sum(isinstance(operation, mir.MNaxLoadAffineParameters) for operation in operations) == 2
    assert sum(isinstance(operation, mir.MNaxLoadAffineFragment) for operation in operations) == 8
    assert any(isinstance(operation, mir.MNaxMatmul2dDecl) for operation in operations)


def test_affine_matmul_masks_ragged_prefill_rows():
    function = lower_affine_matmul("affine_prefill_ir", 33, 64, 64, schedule="morton")
    operations = tuple(_walk(function.ops))

    schedule = next(
        operation for operation in operations if isinstance(operation, mir.MTileSchedule)
    )
    setup = next(operation for operation in operations if isinstance(operation, mir.MNaxGemmSetup))
    runs = [operation for operation in operations if isinstance(operation, mir.MNaxAffineRun)]
    store = next(operation for operation in operations if isinstance(operation, mir.MNaxGemmStore))

    assert schedule.grid_m == 2
    assert schedule.pattern == "morton"
    assert setup.m == 33
    assert all(run.row_bound == 33 for run in runs)
    assert store.row_bound == 33


def test_affine_matmul_maps_multiple_row_simdgroups():
    function = lower_affine_matmul("affine_prefill_64x64", 65, 64, 64, block_m=64)
    operations = tuple(_walk(function.ops))

    schedule = next(
        operation for operation in operations if isinstance(operation, mir.MTileSchedule)
    )
    setup = next(operation for operation in operations if isinstance(operation, mir.MNaxGemmSetup))

    assert function.threadgroup_size == (128, 1, 1)
    assert schedule.grid_m == 2
    assert setup.block_m == 64
    assert setup.wm == 2
    assert setup.wn == 2


def test_affine_swiglu_is_composed_from_binary_fragment_ir():
    function = lower_affine_swiglu_qmv("affine_swiglu_ir", 64, 64)
    operations = tuple(_walk(function.ops))

    combines = [
        operation for operation in operations if isinstance(operation, mir.MNaxBinaryFragment)
    ]
    assert len(combines) == 2
    assert all(operation.operation == "swiglu" for operation in combines)
    assert sum(isinstance(operation, mir.MNaxLoadFragment) for operation in operations) == 4
    assert sum(isinstance(operation, mir.MNaxFmaFragment) for operation in operations) == 8


def test_affine_swiglu_scratch_reuses_two_accumulator_fragments():
    function = lower_affine_swiglu_qmv(
        "affine_swiglu_scratch_ir",
        64,
        64,
        lifetime_schedule="scratch",
    )
    operations = tuple(_walk(function.ops))

    initialization = next(
        operation for operation in operations if isinstance(operation, mir.MNaxAccumulatorInit)
    )
    allocation = next(
        operation for operation in operations if isinstance(operation, mir.MThreadgroupAlloc)
    )
    assert initialization.names == ("gate00", "gate01")
    assert allocation.size == 2 * 2 * 32 * 8
    assert sum(isinstance(operation, mir.MNaxSpillFragment) for operation in operations) == 2
    assert sum(isinstance(operation, mir.MNaxReloadFragment) for operation in operations) == 2
    assert sum(isinstance(operation, mir.MNaxAccumulatorReset) for operation in operations) == 1
    assert sum(isinstance(operation, mir.MNaxFmaFragment) for operation in operations) == 8


def test_affine_swiglu_scratch_emits_per_simdgroup_spills():
    source = emit(
        lower_affine_swiglu_qmv(
            "affine_swiglu_scratch_source",
            64,
            64,
            lifetime_schedule="scratch",
        )
    )

    assert "threadgroup float nax_gate_scratch[1024];" in source
    assert "(sgid * 2u + 0u) * 256u + uint(i) * 32u + slid" in source
    assert "simdgroup_barrier(mem_flags::mem_threadgroup);" in source
    assert "gate00 = metal::vec<float, 8>(0.0f);" in source


def test_affine_qmv_native_tensor_ops_match_reference():
    device = MetalDevice.get()
    if not device.supports_tensor_ops:
        return

    input_features = output_features = 64
    random = np.random.default_rng(211)
    activations = random.normal(size=(1, input_features)).astype(np.float16)
    quantized = random.integers(0, 16, size=(output_features, input_features), dtype=np.uint8)
    scales = random.uniform(0.01, 0.2, size=(1, output_features)).astype(np.float16)
    biases = random.uniform(-0.5, 0.5, size=(1, output_features)).astype(np.float16)
    k_major = np.ascontiguousarray(quantized.T).reshape(-1)
    packed = np.ascontiguousarray(k_major[0::2] | (k_major[1::2] << 4), dtype=np.uint8)
    expected_weight = quantized.astype(np.float32) * scales.T.astype(np.float32) + biases.T.astype(
        np.float32
    )
    expected = (activations.astype(np.float32) @ expected_weight.T).astype(np.float16)

    function_name = "affine_qmv_native_test"
    function = decompose_nax_fragments(
        optimize_tile_schedules(lower_affine_qmv(function_name, output_features, input_features))
    )
    source = emit(function)
    pipeline, _ = device.compile_msl_precompiled(source, function_name, metal_std="metal4.0")
    inputs = [MtileBuffer.from_numpy(array) for array in (activations, packed, scales, biases)]
    output = MtileBuffer.zeros((1, output_features), np.float16)
    compiled = CompiledKernel(
        pipeline,
        source,
        function_name,
        function.threadgroup_size,
        is_gemm=True,
        output_indices=(4,),
    )
    dispatch = FastDispatcher(
        compiled,
        [buffer.metal_buffer for buffer in inputs] + [output.metal_buffer],
        (1, 1),
        device,
        resources=(*inputs, output),
    )
    dispatch()

    np.testing.assert_allclose(output.numpy(), expected, rtol=3e-2, atol=3e-2)
