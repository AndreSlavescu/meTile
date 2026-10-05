from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import metile
from metile.codegen.msl_emitter import emit
from metile.codegen.msl_emitter.elementwise import _emit_epilogue_captures, _emit_epilogue_chain
from metile.compiler.epilogue import EpilogueError, EpilogueProgram, build_epilogue
from metile.compiler.lowering.common import LoweringError, _detect_epilogue
from metile.compiler.lowering.gemm import _lower_gemm, _lower_tensor_ops_gemm
from metile.compiler.passes import decompose_nax_fragments
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType, TileType


def _accumulate(left, right, destination, rows, columns, inner, block_m, block_n, block_k):
    left_tensor = metile.tensor(
        left, shape=(rows, inner), block_shape=(block_m, block_k), access="read"
    )
    right_tensor = metile.tensor(
        right, shape=(inner, columns), block_shape=(block_k, block_n), access="read"
    )
    output_tensor = metile.tensor(
        destination, shape=(rows, columns), block_shape=(block_m, block_n), access="write"
    )
    row = metile.program_id(0) * block_m
    column = metile.program_id(1) * block_n
    accumulator = metile.zeros((block_m, block_n), dtype="f32")
    for reduction in metile.tile_range(0, inner, block_k):
        accumulator = metile.dot(
            left_tensor.load((row, reduction)),
            right_tensor.load((reduction, column)),
            accumulator,
        )
    return output_tensor, row, column, accumulator


def _trace(
    expression,
    *,
    dtype="f32",
    extra_parameters=None,
    dimension_names=("rows", "columns", "inner"),
):
    parameter_types = {
        "left": PtrType(dtype),
        "right": PtrType(dtype),
        "destination": PtrType(dtype),
        **dict.fromkeys(dimension_names, I32),
        "alpha": ScalarType("f32"),
        "beta": ScalarType("f32"),
    }
    parameter_types.update(extra_parameters or {})
    context = TracingContext("epilogue_dag")
    context.func.params = [
        tir.Param(name, datatype, is_output=name == "destination")
        for name, datatype in parameter_types.items()
    ]
    proxies = {
        name: TracingProxy(tir.Value(name, datatype)) for name, datatype in parameter_types.items()
    }
    context.func.constexprs = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 16,
        "WM": 2,
        "WN": 2,
        "SWIZZLE": "linear",
        "_RUNTIME_SCALARS": tuple((name, 64) for name in dimension_names),
    }
    with context:
        output, row, column, accumulator = _accumulate(
            *(proxies[name] for name in ("left", "right", "destination", *dimension_names)),
            64,
            64,
            16,
        )
        output.store((row, column), expression(accumulator, proxies))
    return context.func


def _evaluate(program, accumulator, parameters):
    values = {}
    unary = {
        "exp": np.exp,
        "exp2": np.exp2,
        "fast_cos": np.cos,
        "fast_exp": np.exp,
        "fast_exp2": np.exp2,
        "fast_sin": np.sin,
        "log": np.log,
        "sqrt": np.sqrt,
        "rsqrt": lambda values: 1 / np.sqrt(values),
        "abs": np.abs,
        "neg": np.negative,
        "tanh": np.tanh,
    }
    binary = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "max": np.maximum,
        "min": np.minimum,
        "bitand": np.bitwise_and,
    }
    compare = {
        "lt": np.less,
        "le": np.less_equal,
        "gt": np.greater,
        "ge": np.greater_equal,
        "eq": np.equal,
        "ne": np.not_equal,
    }
    dtypes = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "u32": np.uint32, "bool": bool}
    for instruction in program.instructions:
        arguments = [values[operand] for operand in instruction.operands]
        if instruction.kind == "accumulator":
            result = accumulator
        elif instruction.kind == "parameter":
            result = parameters[instruction.parameter]
        elif instruction.kind == "constant":
            result = instruction.value
        elif instruction.kind == "cast":
            result = arguments[0]
        elif instruction.kind == "unary":
            result = unary[instruction.operation](*arguments)
        elif instruction.kind == "binary":
            result = binary[instruction.operation](*arguments)
        elif instruction.kind == "compare":
            result = compare[instruction.operation](*arguments)
        else:
            assert instruction.kind == "select"
            result = np.where(*arguments)
        values[instruction.name] = np.asarray(result, dtype=dtypes[instruction.type.dtype])
    return values[program.result]


def _branched_expression(accumulator, parameters):
    scaled = accumulator * parameters["alpha"] + parameters["beta"]
    nonlinear = metile.exp(0.0 - accumulator * accumulator)
    selected = metile.where(accumulator >= 0, scaled + nonlinear, parameters["beta"] - scaled)
    return metile.minimum(metile.maximum(selected, -1.5), 2.0)


def test_fused_program_preserves_branches_runtime_coefficients_and_clamps():
    program = build_epilogue(_trace(_branched_expression))
    samples = np.array([-3.0, -1.0, -0.2, 0.0, 0.3, 1.0, 3.0], dtype=np.float32)
    actual = _evaluate(program, samples, {"alpha": 0.75, "beta": -0.125})
    scaled = samples * 0.75 - 0.125
    expected = np.clip(
        np.where(samples >= 0, scaled + np.exp(-samples * samples), -0.125 - scaled), -1.5, 2.0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert {
        instruction.parameter
        for instruction in program.instructions
        if instruction.kind == "parameter"
    } == {"alpha", "beta"}
    assert {instruction.kind for instruction in program.instructions} >= {
        "compare",
        "select",
        "binary",
        "unary",
    }
    with pytest.raises(FrozenInstanceError):
        program.result = "changed"
    with pytest.raises(FrozenInstanceError):
        program.instructions[0].name = "changed"


def test_reused_subexpression_is_emitted_once_and_dead_branches_are_omitted():
    def expression(accumulator, parameters):
        metile.sqrt(accumulator)
        shared = accumulator * parameters["alpha"] + parameters["beta"]
        return shared * shared + metile.exp(shared)

    program = build_epilogue(_trace(expression))
    operations = [instruction.operation for instruction in program.instructions]
    assert operations.count("mul") == 2
    assert "sqrt" not in operations
    squared = next(
        instruction
        for instruction in program.instructions
        if instruction.operation == "mul" and instruction.operands[0] == instruction.operands[1]
    )
    exponential = next(
        instruction for instruction in program.instructions if instruction.operation == "exp"
    )
    assert squared.operands[0] == exponential.operands[0]
    lines = []
    _emit_epilogue_chain([program], "accumulator_element", lines, "")
    source = "\n".join(lines)
    assert source.count(" = exp(") == 1
    assert source.count(f" {squared.operands[0]} = ") == 1


def test_reciprocal_square_root_epilogue_uses_precise_intrinsic():
    function = _trace(lambda accumulator, parameters: metile.rsqrt(accumulator))
    program = build_epilogue(function)
    assert _detect_epilogue(function.ops) == [("unary", "rsqrt")]
    samples = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(_evaluate(program, samples, {}), 1 / np.sqrt(samples))
    assert any(instruction.operation == "rsqrt" for instruction in program.instructions)
    lines = []
    _emit_epilogue_chain([program], "accumulator_element", lines, "")
    source = "\n".join(lines)
    assert "precise::rsqrt(" in source


@pytest.mark.parametrize(
    "operation,reference,intrinsic",
    [
        (metile.exp2, np.exp2, "exp2"),
        (metile.fast_exp2, np.exp2, "fast::exp2"),
        (metile.fast_cos, np.cos, "fast::cos"),
        (metile.fast_sin, np.sin, "fast::sin"),
    ],
)
def test_exp2_and_fast_trigonometric_epilogues_use_explicit_intrinsics(
    operation, reference, intrinsic
):
    function = _trace(lambda accumulator, parameters: operation(accumulator))
    program = build_epilogue(function)
    assert _detect_epilogue(function.ops) == [("unary", operation.__name__)]
    values = np.array([-2.0, -0.3, 0.0, 0.7, 2.0], dtype=np.float32)
    np.testing.assert_allclose(_evaluate(program, values, {}), reference(values))
    lines = []
    _emit_epilogue_chain([program], "accumulator_element", lines, "")
    assert intrinsic + "(" in "\n".join(lines)


def test_fma_epilogue_preserves_explicit_fused_operation():
    function = _trace(
        lambda accumulator, parameters: metile.fma(
            accumulator, parameters["alpha"], parameters["beta"]
        )
    )
    program = build_epilogue(function)
    instruction = next(item for item in program.instructions if item.kind == "fma")
    assert len(instruction.operands) == 3
    lines = []
    _emit_epilogue_chain([program], "accumulator_element", lines, "")
    source = "\n".join(lines)
    assert source.count(" = fma(") == 1


@pytest.mark.parametrize("kind", ["silu", "gelu", "relu", "reverse_where"])
def test_existing_activation_semantics_and_reversed_where_branches(kind):
    def expression(accumulator, parameters):
        if kind == "silu":
            return accumulator / (1.0 + metile.exp(0.0 - accumulator))
        if kind == "gelu":
            return accumulator / (1.0 + metile.exp(0.0 - 1.702 * accumulator))
        if kind == "relu":
            return metile.where(accumulator > 0, accumulator, 0)
        return metile.where(accumulator > 0, 0.0, accumulator)

    program = build_epilogue(_trace(expression))
    samples = np.linspace(-3.0, 3.0, 19, dtype=np.float32)
    if kind in {"silu", "gelu"}:
        scale = 1.702 if kind == "gelu" else 1.0
        expected = samples / (1.0 + np.exp(-scale * samples))
    elif kind == "relu":
        expected = np.maximum(samples, 0)
    else:
        expected = np.minimum(samples, 0)
    np.testing.assert_allclose(_evaluate(program, samples, {}), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", ["simdgroup", "tensor_ops", "nax"])
def test_all_gemm_backends_consume_the_same_frozen_epilogue_program(backend):
    function = _trace(_branched_expression, dtype="f16")
    if backend == "simdgroup":
        lowered = _lower_gemm(function)
        application = next(
            operation for operation in lowered.ops if isinstance(operation, mir.MAccElemApply)
        )
    else:
        function.constexprs["NAX_FRAGMENTS"] = backend == "nax"
        lowered = _lower_tensor_ops_gemm(function)
        if backend == "nax":
            lowered = decompose_nax_fragments(lowered)
            application = next(
                operation
                for operation in lowered.ops
                if isinstance(operation, mir.MNaxApplyFragment)
            )
        else:
            application = next(
                operation
                for operation in lowered.ops
                if isinstance(operation, mir.MCoopTensorEpilogue)
            )
    assert len(application.operations) == 1
    assert isinstance(application.operations[0], EpilogueProgram)
    source = emit(lowered)
    assert " = alpha;" in source
    assert " = beta;" in source
    assert "const bool _metile_epilogue_" in source
    assert " ? " in source
    assert " = max(" in source
    assert " = min(" in source


def test_identity_epilogue_requires_no_program():
    function = _trace(lambda accumulator, parameters: accumulator)
    assert build_epilogue(function) is None
    assert _detect_epilogue(function.ops, func=function) == []


@pytest.mark.parametrize("operation", ["memory", "reduce", "simd", "cast", "control_flow"])
def test_unsupported_dependencies_are_rejected_instead_of_dropped(operation):
    def expression(accumulator, parameters):
        if operation == "memory":
            return accumulator + metile.load(parameters["left"] + 0)
        if operation == "reduce":
            return accumulator + metile.sum(accumulator)
        if operation == "simd":
            return metile.simd_sum(accumulator)
        if operation == "cast":
            return metile.cast(accumulator, "f16")
        for _index in metile.tile_range(0, parameters["inner"], 1):
            accumulator = accumulator * 2.0
        return accumulator

    function = _trace(expression)
    with pytest.raises(EpilogueError, match="epilogue is not supported"):
        build_epilogue(function)


@pytest.mark.parametrize("dimension", ["rows", "columns", "inner", "M"])
def test_dimension_parameter_dependencies_require_explicit_binding_support(dimension):
    function = _trace(
        lambda accumulator, parameters: accumulator * parameters[dimension],
        extra_parameters={"M": ScalarType("f32")},
    )
    with pytest.raises(EpilogueError, match="preserved scalar binding"):
        build_epilogue(function)
    with pytest.raises(LoweringError, match="preserved scalar binding"):
        _lower_tensor_ops_gemm(function)


def test_independent_integer_scalar_coefficient_and_explicit_f32_cast():
    function = _trace(
        lambda accumulator, parameters: accumulator * metile.cast(parameters["factor"], "f32"),
        extra_parameters={"factor": I32},
    )
    program = build_epilogue(function)
    lines = []
    _emit_epilogue_chain([program], "element", lines, "")
    assert "static_cast<float>" in "\n".join(lines)
    np.testing.assert_array_equal(
        _evaluate(program, np.array([-1.0, 2.0]), {"factor": 3}), [-3.0, 6.0]
    )


@pytest.mark.parametrize("operation", ["integer_exp", "float_bitwise", "integer_overflow"])
def test_unsupported_scalar_operations_fail_before_emitting_msl(operation):
    def expression(accumulator, parameters):
        if operation == "integer_exp":
            return accumulator + metile.exp(parameters["factor"])
        if operation == "float_bitwise":
            return accumulator + (parameters["factor"] & parameters["alpha"])
        return accumulator + (1 << 40)

    function = _trace(expression, extra_parameters={"factor": I32})
    with pytest.raises(EpilogueError, match="epilogue is not supported"):
        build_epilogue(function)


def test_boolean_predicate_branches_remain_typed_boolean():
    def expression(accumulator, parameters):
        inside = (accumulator >= -1.0) & (accumulator <= 1.0)
        return metile.where(inside, accumulator * parameters["alpha"], accumulator)

    program = build_epilogue(_trace(expression))
    combined = next(
        instruction for instruction in program.instructions if instruction.operation == "bitand"
    )
    assert combined.type == ScalarType("bool")
    samples = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(
        _evaluate(program, samples, {"alpha": 2.0}), [-2.0, -1.0, 1.0, 2.0]
    )


def test_epilogue_temporary_names_do_not_shadow_scalar_parameters():
    parameter = "_metile_epilogue_0"
    function = _trace(
        lambda accumulator, parameters: accumulator * parameters[parameter],
        extra_parameters={parameter: ScalarType("f32")},
    )
    program = build_epilogue(function)
    assert all(instruction.name != parameter for instruction in program.instructions)


@pytest.mark.parametrize("backend", ["simdgroup", "tensor_ops", "nax"])
@pytest.mark.parametrize("canonical_dimensions", [False, True])
@pytest.mark.parametrize("parameter", ["tile_row", "sg_row", "_metile_epilogue_0"])
def test_epilogue_scalars_are_captured_before_backend_scopes(
    backend, canonical_dimensions, parameter
):
    function = _trace(
        lambda accumulator, parameters: accumulator * parameters[parameter],
        dtype="f16",
        extra_parameters={parameter: ScalarType("f32")},
        dimension_names=("M", "N", "K") if canonical_dimensions else ("rows", "columns", "inner"),
    )
    function.constexprs["NAX_FRAGMENTS"] = backend == "nax"
    program = build_epilogue(function)
    scalar = next(
        instruction for instruction in program.instructions if instruction.kind == "parameter"
    )
    lowered = _lower_gemm(function) if backend == "simdgroup" else _lower_tensor_ops_gemm(function)
    if backend == "nax":
        lowered = decompose_nax_fragments(lowered)
    source = emit(lowered)
    capture = f"const float {scalar.capture_name} = {parameter};"
    assert source.count(capture) == 1
    assert source.index(capture) < source.index("\n    {")
    assert f"const float {scalar.name} = {scalar.capture_name};" in source
    assert f"const float {scalar.name} = {parameter};" not in source


@pytest.mark.parametrize("expression", [lambda accumulator: accumulator, metile.exp])
def test_epilogues_without_runtime_scalar_leaves_emit_no_capture_scope(expression):
    lowered = _lower_gemm(_trace(lambda accumulator, parameters: expression(accumulator)))
    lines = []
    assert not _emit_epilogue_captures(lowered, lines)
    assert lines == []
    assert "_capture" not in emit(lowered)


@pytest.mark.parametrize("mutation", ["removed", "type_changed"])
def test_epilogue_emitter_rejects_lost_scalar_bindings(mutation):
    lowered = _lower_gemm(_trace(lambda accumulator, parameters: accumulator * parameters["alpha"]))
    if mutation == "removed":
        lowered.params = [parameter for parameter in lowered.params if parameter.name != "alpha"]
    else:
        next(parameter for parameter in lowered.params if parameter.name == "alpha").type = I32
    with pytest.raises(ValueError, match="scalar binding is not preserved: alpha"):
        emit(lowered)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tile_condition", [False, True])
def test_select_promotes_integer_float_branches_in_both_orders(reverse, tile_condition):
    condition_type = TileType((8, 8), "bool") if tile_condition else ScalarType("bool")
    condition = tir.Value("condition", condition_type)
    integer = tir.Value("integer", I32)
    floating = tir.Value("floating", ScalarType("f32"))
    true_value, false_value = (floating, integer) if reverse else (integer, floating)
    result = tir.Select(
        condition=condition, true_val=true_value, false_val=false_value
    ).result_type()
    assert result == (TileType((8, 8), "f32") if tile_condition else ScalarType("f32"))
    assert condition.type == condition_type


@pytest.mark.parametrize("reverse", [False, True])
def test_select_infers_tile_shape_from_either_value_branch(reverse):
    integer = tir.Value("integer", I32)
    tile = tir.Value("tile", TileType((8, 8), "f32"))
    true_value, false_value = (tile, integer) if reverse else (integer, tile)
    condition = tir.Value("condition", ScalarType("bool"))
    assert (
        tir.Select(condition=condition, true_val=true_value, false_val=false_value).result_type()
        == tile.type
    )


def test_select_preserves_pointer_types_and_rejects_mismatches():
    condition = tir.Value("condition", ScalarType("bool"))
    pointer = tir.Value("pointer", PtrType("f32"))
    assert (
        tir.Select(condition=condition, true_val=pointer, false_val=pointer).result_type()
        == pointer.type
    )
    with pytest.raises(TypeError, match="matching pointer types"):
        tir.Select(
            condition=condition, true_val=pointer, false_val=tir.Value("other", I32)
        ).result_type()


@metile.kernel
def fused_branched_gemm(
    left,
    right,
    destination,
    rows,
    columns,
    inner,
    alpha,
    beta,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    output, row, column, accumulator = _accumulate(
        left, right, destination, rows, columns, inner, BLOCK_M, BLOCK_N, BLOCK_K
    )
    output.store((row, column), _branched_expression(accumulator, {"alpha": alpha, "beta": beta}))


@pytest.mark.parametrize(
    ("backend", "dtype", "shape"),
    [
        ("simdgroup", np.float32, (33, 47, 29)),
        ("tensor_ops", np.float32, (64, 64, 64)),
        ("tensor_ops", np.float16, (64, 64, 64)),
        ("nax", np.float16, (63, 64, 64)),
    ],
)
def test_fused_epilogue_gpu_matches_composed_numpy_expression(backend, dtype, shape):
    from metile.runtime.metal_device import MetalDevice

    if backend != "simdgroup" and not MetalDevice.get().supports_tensor_ops:
        pytest.skip("tensor operations are unavailable")
    rows, columns, inner = shape
    generator = np.random.default_rng(193)
    left = generator.normal(scale=0.25, size=(rows, inner)).astype(dtype)
    right = generator.normal(scale=0.25, size=(inner, columns)).astype(dtype)
    output = np.zeros((rows, columns), dtype=dtype)
    alpha = 0.75
    beta = -0.125
    launcher = fused_branched_gemm[((rows + 63) // 64, (columns + 63) // 64)]
    launcher(
        left,
        right,
        output,
        rows,
        columns,
        inner,
        alpha,
        beta,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=2,
        WN=2,
        NAX_FRAGMENTS=backend == "nax",
        RELAXED_PRECISION=False,
        SWIZZLE="linear",
    )
    product = left.astype(np.float64) @ right.astype(np.float64)
    scaled = product * alpha + beta
    expected = np.clip(
        np.where(product >= 0, scaled + np.exp(-product * product), beta - scaled), -1.5, 2.0
    ).astype(dtype)
    tolerance = 2e-5 if dtype == np.float32 else 2e-3
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
    assert "_metile_epilogue_" in launcher._last_compiled.msl_source


@metile.kernel
def fused_shadowed_scalar_gemm(
    left,
    right,
    destination,
    rows,
    columns,
    inner,
    tile_row,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    output, row, column, accumulator = _accumulate(
        left, right, destination, rows, columns, inner, BLOCK_M, BLOCK_N, BLOCK_K
    )
    output.store((row, column), accumulator * tile_row)


@metile.kernel
def fused_canonical_shadowed_scalar_gemm(
    left,
    right,
    destination,
    M,
    N,
    K,
    tile_row,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    output, row, column, accumulator = _accumulate(
        left, right, destination, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K
    )
    output.store((row, column), accumulator * tile_row)


@pytest.mark.parametrize("canonical_dimensions", [False, True])
@pytest.mark.parametrize("backend", ["simdgroup", "tensor_ops", "nax"])
def test_fused_epilogue_gpu_preserves_scalar_shadowed_by_tile_row(backend, canonical_dimensions):
    from metile.runtime.metal_device import MetalDevice

    if backend != "simdgroup" and not MetalDevice.get().supports_tensor_ops:
        pytest.skip("tensor operations are unavailable")
    dtype = np.float16 if backend == "nax" else np.float32
    size = 63 if backend == "simdgroup" else 128
    generator = np.random.default_rng(194)
    left = generator.normal(scale=0.25, size=(size, size)).astype(dtype)
    right = generator.normal(scale=0.25, size=(size, size)).astype(dtype)
    output = np.zeros_like(left)
    coefficient = -0.375
    kernel = (
        fused_canonical_shadowed_scalar_gemm if canonical_dimensions else fused_shadowed_scalar_gemm
    )
    launcher = kernel[((size + 63) // 64, (size + 63) // 64)]
    launcher(
        left,
        right,
        output,
        size,
        size,
        size,
        coefficient,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=16,
        WM=2,
        WN=2,
        NAX_FRAGMENTS=backend == "nax",
        RELAXED_PRECISION=False,
        SWIZZLE="linear",
    )
    expected = ((left.astype(np.float64) @ right.astype(np.float64)) * coefficient).astype(dtype)
    tolerance = 2e-5 if dtype == np.float32 else 2e-3
    np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
    assert "_capture = tile_row;" in launcher._last_compiled.msl_source


@metile.kernel
def mixed_branch_where(
    source,
    destination,
    length,
    REVERSE: metile.constexpr,
    BLOCK: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(length,), access="read")
    outputs = metile.tensor(destination, shape=(length,), access="write")
    indices = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    values = inputs.load(indices)
    result = metile.where(values > 0, values, 0) if REVERSE else metile.where(values > 0, 0, values)
    outputs.store(indices, result)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_elementwise_select_lowers_both_branches_to_the_promoted_value_type(reverse, dtype):
    from metile.compiler.lowering.elementwise import _ElementwiseLoweringContext

    context = TracingContext("promoted_select")
    arguments = []
    for name, datatype in (
        ("source", PtrType(dtype)),
        ("destination", PtrType(dtype)),
        ("length", I32),
    ):
        context.func.params.append(tir.Param(name, datatype, is_output=name == "destination"))
        arguments.append(TracingProxy(tir.Value(name, datatype)))
    with context:
        mixed_branch_where.fn(*arguments, REVERSE=reverse, BLOCK=32)
    lowered = _ElementwiseLoweringContext(context.func).lower()
    select = next(operation for operation in lowered.ops if isinstance(operation, mir.MSelect))
    assert select.result.type == ScalarType(dtype)
    assert select.true_val.type == ScalarType(dtype)
    assert select.false_val.type == ScalarType(dtype)
    assert select.condition.type == ScalarType("bool")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_elementwise_where_gpu_preserves_fractional_values_in_either_branch(reverse, dtype):
    source = np.array([-2.75, -1.5, -0.125, 0.0, 0.25, 1.75, 3.5], dtype=dtype)
    destination = np.empty_like(source)
    mixed_branch_where[(1,)](
        source,
        destination,
        source.size,
        REVERSE=reverse,
        BLOCK=32,
    )
    expected = np.where(source > 0, source, 0) if reverse else np.where(source > 0, 0, source)
    np.testing.assert_array_equal(destination, expected)
