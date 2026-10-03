import inspect

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.compiler.lowering.common import LoweringError, _analyze_gemm, _tensor_ops_aligned
from metile.compiler.lowering.gemm import _lower_gemm, _lower_tensor_ops_gemm
from metile.compiler.passes import decompose_nax_fragments
from metile.frontend.kernel import KernelLauncher
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType


@metile.kernel
def descriptor_product(
    destination,
    unused,
    columns,
    right,
    rows,
    left,
    reduction,
    TILE_ROWS: metile.constexpr,
    TILE_COLUMNS: metile.constexpr,
    TILE_REDUCTION: metile.constexpr,
):
    left_tensor = metile.tensor(
        left, shape=(rows, reduction), block_shape=(TILE_ROWS, TILE_REDUCTION), access="read"
    )
    right_tensor = metile.tensor(
        right, shape=(reduction, columns), block_shape=(TILE_REDUCTION, TILE_COLUMNS), access="read"
    )
    output_tensor = metile.tensor(
        destination, shape=(rows, columns), block_shape=(TILE_ROWS, TILE_COLUMNS), access="write"
    )
    row = metile.program_id(0) * TILE_ROWS
    column = metile.program_id(1) * TILE_COLUMNS
    accumulator = metile.zeros((TILE_ROWS, TILE_COLUMNS), dtype="f32")
    for start in metile.tile_range(0, reduction, TILE_REDUCTION):
        accumulator = metile.dot(
            left_tensor.load((row, start)), right_tensor.load((start, column)), accumulator
        )
    output_tensor.store((row, column), accumulator)


@metile.kernel
def permuted_dimensions(destination, N, right, K, left, M):
    left_tensor = metile.tensor(left, shape=(N, M), block_shape=(32, 16), access="read")
    right_tensor = metile.tensor(right, shape=(M, K), block_shape=(16, 32), access="read")
    output_tensor = metile.tensor(destination, shape=(N, K), block_shape=(32, 32), access="write")
    row = metile.program_id(0) * 32
    column = metile.program_id(1) * 32
    accumulator = metile.zeros((32, 32), dtype="f32")
    for start in metile.tile_range(0, M, 16):
        accumulator = metile.dot(
            left_tensor.load((row, start)), right_tensor.load((start, column)), accumulator
        )
    output_tensor.store((row, column), accumulator)


def _trace(kernel=descriptor_product, dtype="f16", **config):
    defaults = {"TILE_ROWS": 32, "TILE_COLUMNS": 32, "TILE_REDUCTION": 16}
    defaults.update(config)
    function = kernel.fn
    signature = inspect.signature(function)
    context = TracingContext(kernel.name)
    arguments = []
    constexprs = {}
    for name, parameter in signature.parameters.items():
        if parameter.annotation is metile.constexpr:
            constexprs[name] = defaults[name]
            continue
        if name in {"destination", "left", "right"}:
            dtype_value = PtrType(dtype)
        elif name == "unused":
            dtype_value = PtrType("u32")
        else:
            dtype_value = I32
        context.func.params.append(tir.Param(name, dtype_value, is_output=name == "destination"))
        arguments.append(TracingProxy(tir.Value(name, dtype_value)))
    context.func.constexprs = {**defaults, **config}
    with context:
        function(*arguments, **constexprs)
    return context.func


def test_descriptor_gemm_binds_operands_and_tiles_from_dataflow():
    function = _trace()
    binding = _analyze_gemm(function)
    assert [binding.left.ptr.name, binding.right.ptr.name, binding.output.ptr.name] == [
        "left",
        "right",
        "destination",
    ]
    assert [dimension.name for dimension in binding.dimensions] == ["rows", "columns", "reduction"]
    assert binding.tiles == (32, 32, 16)
    lowered = _lower_gemm(function)
    loop = next(operation for operation in lowered.ops if isinstance(operation, mir.MForLoop))
    loads = [operation for operation in loop.body if isinstance(operation, mir.MCooperativeLoad)]
    assert [operation.device_ptr.name for operation in loads] == ["left", "right"]
    assert all(operation.elem_type == "half" for operation in loads)
    assert lowered.dimension_bindings["M"].name == "rows"
    source = emit(lowered)
    assert "const int _metile_dimension_M = rows;" in source


def test_descriptor_gemm_tensor_ops_supports_permuted_dimension_names():
    lowered = _lower_tensor_ops_gemm(_trace(permuted_dimensions, dtype="f32"))
    source = emit(lowered)
    assert "const int _metile_dimension_M = N;" in source
    assert "const int _metile_dimension_N = K;" in source
    assert "const int _metile_dimension_K = M;" in source
    assert source.index("const int _metile_dimension_K = M;") < source.index(
        "const int M = _metile_dimension_M;"
    )


def test_generic_half_tensor_ops_preserves_float_accumulation_and_strict_default():
    function = _trace(dtype="f16", TILE_ROWS=64, TILE_COLUMNS=64)
    source = emit(_lower_tensor_ops_gemm(function))
    assert "get_left_input_cooperative_tensor<half, half, float>" in source
    assert "false, false, false," in source
    assert "mC[_coordinate] = half(cT[_index]);" in source


@pytest.mark.parametrize("change", ["strides", "dimensions", "origin", "tile", "mixed_dtype"])
def test_descriptor_gemm_rejects_unsupported_memory_contracts(change):
    function = _trace()
    binding = _analyze_gemm(function)
    if change == "strides":
        binding.left.tensor.strides = tuple(reversed(binding.left.tensor.strides))
        expected = "contiguous row-major"
    elif change == "dimensions":
        binding.right.tensor.shape = tuple(reversed(binding.right.tensor.shape))
        binding.right.tensor.strides = (
            binding.right.tensor.shape[1],
            binding.right.tensor.strides[1],
        )
        expected = "dimensions do not agree"
    elif change == "origin":
        binding.left.row_offset = binding.left.col_offset
        expected = "canonical tiled product"
    elif change == "tile":
        function.constexprs["BLOCK_M"] = 64
        expected = "match BLOCK"
    else:
        binding.output.ptr.type = PtrType("f32")
        expected = "matching f16 or f32"
    with pytest.raises(LoweringError, match=expected):
        _lower_gemm(function)


def test_descriptor_gemm_rejects_unrecognized_epilogue():
    function = _trace()
    output = next(operation for operation in function.ops if isinstance(operation, tir.TileStore))
    extra = tir.Cast(value=output.value, dtype="f16")
    extra.result = tir.Value("unsupported_epilogue", extra.result_type(), extra)
    output.value = extra.result
    function.ops.insert(function.ops.index(output), extra)
    with pytest.raises(LoweringError, match="epilogue is not supported"):
        _lower_gemm(function)


@pytest.mark.parametrize("change", ["loop_arithmetic", "store_initializer", "outer_loop"])
def test_descriptor_gemm_rejects_programs_the_matmul_lowerer_cannot_preserve(change):
    function = _trace()
    loop = next(operation for operation in function.ops if isinstance(operation, tir.ForRange))
    output = next(operation for operation in function.ops if isinstance(operation, tir.TileStore))
    dot = next(operation for operation in loop.body if isinstance(operation, tir.Dot))
    if change == "loop_arithmetic":
        constant = tir.Constant(value=2.0, dtype="f32")
        constant.result = tir.Value("scale", constant.result_type(), constant)
        operation = tir.BinOp(op="mul", lhs=dot.result, rhs=constant.result)
        operation.result = tir.Value("scaled_accumulator", operation.result_type(), operation)
        loop.body.extend((constant, operation))
        output.value = operation.result
        expected = "only tensor loads and the dot recurrence"
    elif change == "store_initializer":
        output.value = dot.acc
        expected = "epilogue is not supported"
    else:
        outer_loop = tir.ForRange(start=loop.start, end=loop.end, step=1, iv=loop.iv, body=[loop])
        function.ops[function.ops.index(loop)] = outer_loop
        expected = "one reduction loop"
    with pytest.raises(LoweringError, match=expected):
        _lower_gemm(function)


def test_descriptor_gemm_nax_specializes_actual_dimension_bindings():
    function = _trace(
        dtype="f32",
        TILE_ROWS=64,
        TILE_COLUMNS=64,
        NAX_FRAGMENTS=True,
        _RUNTIME_SCALARS=(("rows", 63), ("columns", 64), ("reduction", 96)),
    )
    lowered = _lower_tensor_ops_gemm(function)
    assert [parameter.name for parameter in lowered.params] == [
        "destination",
        "unused",
        "right",
        "left",
    ]
    source = emit(decompose_nax_fragments(lowered))
    assert "constexpr uint M = 63u;" in source
    assert "constexpr uint N = 64u;" in source
    assert "constexpr uint K = 96u;" in source


@pytest.mark.parametrize("unroll", [2, 4])
def test_tensor_ops_alignment_must_cover_the_entire_unrolled_reduction_step(unroll):
    function = _trace(
        dtype="f32",
        TILE_REDUCTION=32,
        K_UNROLL=unroll,
        _SCALAR_ALIGNMENT_32=(("rows", 0), ("columns", 0), ("reduction", 0)),
    )
    assert not _tensor_ops_aligned(function)
    function.constexprs["_RUNTIME_SCALARS"] = (("reduction", 96),)
    assert not _tensor_ops_aligned(function)
    function.constexprs["_RUNTIME_SCALARS"] = (("reduction", 128),)
    assert _tensor_ops_aligned(function)


def test_get_compiled_returns_the_launchers_specialized_compilation(monkeypatch):
    compiled = object()

    def launch(launcher, *arguments, **options):
        assert arguments == (37,)
        assert options == {"NAX_FRAGMENTS": True}
        launcher._last_compiled = compiled

    monkeypatch.setattr(KernelLauncher, "__call__", launch)
    assert descriptor_product.get_compiled(37, NAX_FRAGMENTS=True) is compiled


def test_descriptor_gemm_keeps_contiguous_legacy_codegen():
    function = _trace(permuted_dimensions)
    parameter_names = {"N": "M", "K": "N", "M": "K"}
    for parameter in function.params:
        parameter.name = parameter_names.get(parameter.name, parameter.name)
    visited = set()

    def rename(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if value.defining_op is None:
            value.name = parameter_names.get(value.name, value.name)

    for tensor in function.tensors:
        for value in (*tensor.shape, *tensor.strides):
            rename(value)
    descriptor_source = emit(_lower_gemm(function))
    for operation in function.ops:
        if isinstance(operation, tir.TileStore):
            operation.tensor = None
        if isinstance(operation, tir.ForRange):
            for nested in operation.body:
                if isinstance(nested, tir.TileLoad):
                    nested.tensor = None
    assert descriptor_source == emit(_lower_gemm(function))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_descriptor_gemm_handles_renamed_reordered_arguments_and_ragged_shapes(dtype):
    rows, columns, reduction = 37, 45, 23
    generator = np.random.default_rng(803)
    left = generator.normal(size=(rows, reduction)).astype(dtype)
    right = generator.normal(size=(reduction, columns)).astype(dtype)
    output = np.zeros((rows, columns), dtype=dtype)
    unused = np.zeros(1, dtype=np.uint32)
    descriptor_product[(2, 2)](
        output,
        unused,
        columns,
        right,
        rows,
        left,
        reduction,
        TILE_ROWS=32,
        TILE_COLUMNS=32,
        TILE_REDUCTION=16,
    )
    expected = (left.astype(np.float32) @ right.astype(np.float32)).astype(dtype)
    np.testing.assert_allclose(output, expected, rtol=5e-3, atol=1e-2)


def test_gpu_descriptor_gemm_handles_permuted_dimension_names():
    rows, columns, reduction = 37, 45, 23
    generator = np.random.default_rng(807)
    left = generator.normal(size=(rows, reduction)).astype(np.float32)
    right = generator.normal(size=(reduction, columns)).astype(np.float32)
    output = np.zeros((rows, columns), dtype=np.float32)
    permuted_dimensions[(2, 2)](output, rows, right, columns, left, reduction)
    np.testing.assert_allclose(output, left @ right, rtol=5e-3, atol=1e-2)


def test_gpu_descriptor_gemm_nax_cache_tracks_renamed_runtime_shapes():
    from metile.runtime.metal_device import MetalDevice

    if not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    generator = np.random.default_rng(811)
    for rows, columns, reduction in ((31, 64, 32), (61, 64, 48)):
        left = generator.normal(size=(rows, reduction)).astype(np.float32)
        right = generator.normal(size=(reduction, columns)).astype(np.float32)
        output = np.zeros((rows, columns), dtype=np.float32)
        descriptor_product[(1, 1)](
            output,
            np.zeros(1, dtype=np.uint32),
            columns,
            right,
            rows,
            left,
            reduction,
            TILE_ROWS=64,
            TILE_COLUMNS=64,
            TILE_REDUCTION=16,
            NAX_FRAGMENTS=True,
        )
        np.testing.assert_allclose(output, left @ right, rtol=5e-3, atol=1e-2)


def test_gpu_descriptor_gemm_nax_rejects_misaligned_permuted_dimensions():
    from metile.runtime.metal_device import MetalDevice

    if not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    rows, columns, reduction = 64, 32, 23
    with pytest.raises(ValueError, match="aligned N and K"):
        permuted_dimensions[(2, 1)](
            np.zeros((rows, columns), dtype=np.float32),
            rows,
            np.zeros((reduction, columns), dtype=np.float32),
            columns,
            np.zeros((rows, reduction), dtype=np.float32),
            reduction,
            NAX_FRAGMENTS=True,
            WM=1,
            WN=1,
        )


@pytest.mark.parametrize("unroll", [2, 4])
def test_gpu_descriptor_gemm_unrolled_k_tail_uses_bounded_lowering(unroll):
    rows, columns, reduction = 64, 64, 96
    generator = np.random.default_rng(823)
    left = generator.normal(size=(rows, reduction)).astype(np.float32)
    right = generator.normal(size=(reduction, columns)).astype(np.float32)
    output = np.zeros((rows, columns), dtype=np.float32)
    launcher = descriptor_product[(1, 1)]
    launcher(
        output,
        np.zeros(1, dtype=np.uint32),
        columns,
        right,
        rows,
        left,
        reduction,
        TILE_ROWS=64,
        TILE_COLUMNS=64,
        TILE_REDUCTION=32,
        K_UNROLL=unroll,
    )
    assert "<metal_simdgroup_matrix>" in launcher._last_compiled.msl_source
    np.testing.assert_allclose(output, left @ right, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(64, 64, 64), (96, 96, 64), (128, 64, 96)])
def test_gpu_generic_half_tensor_ops_is_selected_from_memory_and_tile_contract(shape):
    from metile.runtime.metal_device import MetalDevice

    if not MetalDevice.get().supports_tensor_ops:
        pytest.skip("requires Metal tensor operations")
    rows, columns, reduction = shape
    generator = np.random.default_rng(829)
    left = generator.normal(size=(rows, reduction)).astype(np.float16)
    right = generator.normal(size=(reduction, columns)).astype(np.float16)
    output = np.zeros((rows, columns), dtype=np.float16)
    launcher = descriptor_product[(metile.cdiv(rows, 64), metile.cdiv(columns, 64))]
    launcher(
        output,
        np.zeros(1, dtype=np.uint32),
        columns,
        right,
        rows,
        left,
        reduction,
        TILE_ROWS=64,
        TILE_COLUMNS=64,
        TILE_REDUCTION=16,
    )
    source = launcher._last_compiled.msl_source
    assert "<metal_tensor>" in source
    assert "get_left_input_cooperative_tensor<half, half, float>" in source
    assert "false, false, false," in source
    expected = (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=5e-3)
