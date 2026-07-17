from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.frontend.tracing import (
    TracingContext,
    TracingProxy,
    fast_exp,
    load,
    maximum,
    scalar,
    store,
    tile_range,
)
from metile.ir import tile_ir as tir
from metile.ir.printer import print_metal_ir, print_tile_ir
from metile.ir.types import I32, U32, PtrType, ScalarType


def _build_vector_add_ir() -> tir.Function:
    """Build Tile IR for vector_add manually."""
    func = tir.Function(
        name="vector_add",
        params=[
            tir.Param("a", PtrType("f32"), is_output=False),
            tir.Param("b", PtrType("f32"), is_output=False),
            tir.Param("c", PtrType("f32"), is_output=True),
            tir.Param("N", I32),
        ],
        constexprs={"BLOCK": 256},
    )

    a_val = tir.Value("a", PtrType("f32"))
    b_val = tir.Value("b", PtrType("f32"))
    c_val = tir.Value("c", PtrType("f32"))
    n_val = tir.Value("N", I32)

    pid = func.add_op(tir.ProgramId(axis=0))
    block_const = func.add_op(tir.Constant(value=256, dtype="i32"))
    pid_x_block = func.add_op(tir.BinOp(op="mul", lhs=pid, rhs=block_const))
    arange_val = func.add_op(tir.Arange(start=pid_x_block, size=256))
    offs = func.add_op(tir.BinOp(op="add", lhs=pid_x_block, rhs=arange_val))
    mask = func.add_op(tir.Compare(predicate="lt", lhs=offs, rhs=n_val))
    a_ptr_off = func.add_op(tir.PtrOffset(ptr=a_val, offsets=offs))
    a_tile = func.add_op(tir.Load(ptr=a_ptr_off, offsets=offs, mask=mask))
    b_ptr_off = func.add_op(tir.PtrOffset(ptr=b_val, offsets=offs))
    b_tile = func.add_op(tir.Load(ptr=b_ptr_off, offsets=offs, mask=mask))
    sum_val = func.add_op(tir.BinOp(op="add", lhs=a_tile, rhs=b_tile))
    c_ptr_off = func.add_op(tir.PtrOffset(ptr=c_val, offsets=offs))
    func.add_op(tir.Store(ptr=c_ptr_off, offsets=offs, value=sum_val, mask=mask))

    return func


def test_tile_ir_construction():
    """Test that Tile IR can be constructed without errors."""
    func = _build_vector_add_ir()
    assert func.name == "vector_add"
    assert len(func.params) == 4
    assert len(func.ops) > 0


def test_tile_ir_printer():
    """Test that Tile IR prints to readable text."""
    func = _build_vector_add_ir()
    text = print_tile_ir(func)
    assert "func @vector_add" in text
    assert "program_id" in text
    assert "arange" in text
    assert "load" in text
    assert "store" in text


def test_lowering():
    """Test Tile IR -> Metal IR lowering."""
    func = _build_vector_add_ir()
    metal_func = lower(func)
    assert metal_func.name == "mtile_vector_add"
    assert len(metal_func.params) == 4
    assert metal_func.threadgroup_size == (256, 1, 1)


def test_metal_ir_printer():
    """Test Metal IR pretty printing."""
    func = _build_vector_add_ir()
    metal_func = lower(func)
    text = print_metal_ir(metal_func)
    assert "metal_func @mtile_vector_add" in text
    assert "thread_position_in_grid" in text


def test_msl_codegen():
    """Test MSL code generation produces valid Metal code."""
    func = _build_vector_add_ir()
    metal_func = lower(func)
    msl = emit(metal_func)
    assert "#include <metal_stdlib>" in msl
    assert "[[kernel]]" in msl
    assert "[[buffer(0)]]" in msl
    assert "[[thread_position_in_grid]]" in msl
    assert "device const float*" in msl


def test_msl_compiles():
    """Test that generated MSL actually compiles on the Metal device."""
    from metile.runtime.metal_device import MetalDevice

    func = _build_vector_add_ir()
    metal_func = lower(func)
    msl = emit(metal_func)
    dev = MetalDevice.get()
    pipeline = dev.compile_msl(msl, metal_func.name)
    assert pipeline is not None


def test_reverse_bits_lowers_to_native_unsigned_msl():
    from metile.ir import metal_ir as mir

    func = tir.Function(name="reverse_bits", params=[tir.Param("value", I32)])
    value = tir.Value("value", I32)
    reversed_value = func.add_op(tir.Unary(op="reverse_bits", operand=value))

    assert reversed_value.type == U32
    metal_func = lower(func)
    reverse_op = next(op for op in metal_func.ops if isinstance(op, mir.MUnary))
    assert reverse_op.result_type() == U32
    assert "reverse_bits" in emit(metal_func)


def test_explicit_scalar_is_carried_across_runtime_loop():
    ctx = TracingContext("scalar_recurrence")
    input_value = tir.Value("input", PtrType("f32"))
    output_value = tir.Value("output", PtrType("f32"))
    length_value = tir.Value("length", I32)
    ctx.func.params = [
        tir.Param("input", PtrType("f32")),
        tir.Param("output", PtrType("f32"), is_output=True),
        tir.Param("length", I32),
    ]

    with ctx:
        input_proxy = TracingProxy(input_value)
        output_proxy = TracingProxy(output_value)
        length_proxy = TracingProxy(length_value)
        local_max = scalar(-1e30)
        for index in tile_range(0, length_proxy):
            value = load(input_proxy + index)
            new_max = maximum(local_max, value)
            fast_exp(local_max - new_max)
            local_max = new_max
        store(output_proxy + 0, local_max)

    msl = emit(lower(ctx.func))
    assert "float _acc_0 = -1e+30f;" in msl
    assert "= _acc_0 -" in msl
    assert "fast::exp(" in msl
    assert "simd_sum" not in msl


def test_native_simd_math_primitives_emit_metal_intrinsics():
    func = tir.Function(name="simd_math", params=[tir.Param("value", ScalarType("f32"))])
    value = tir.Value("value", ScalarType("f32"))
    summed = func.add_op(tir.Unary(op="simd_sum", operand=value))
    maximum_value = func.add_op(tir.Unary(op="simd_max", operand=summed))
    func.add_op(tir.Unary(op="fast_exp", operand=maximum_value))

    msl = emit(lower(func))
    assert "simd_sum(" in msl
    assert "simd_max(" in msl
    assert "fast::exp(" in msl


def _build_simdgroup_role_ir() -> tir.Function:
    """Build Tile IR with simdgroup_role blocks."""
    func = tir.Function(
        name="role_test",
        params=[
            tir.Param("x", PtrType("f32"), is_output=False),
            tir.Param("out", PtrType("f32"), is_output=True),
            tir.Param("N", I32),
        ],
        constexprs={"BLOCK": 256},
    )

    x_val = tir.Value("x", PtrType("f32"))
    out_val = tir.Value("out", PtrType("f32"))
    n_val = tir.Value("N", I32)

    pid = func.add_op(tir.ProgramId(axis=0))
    block_const = func.add_op(tir.Constant(value=256, dtype="i32"))
    pid_x_block = func.add_op(tir.BinOp(op="mul", lhs=pid, rhs=block_const))
    arange_val = func.add_op(tir.Arange(start=pid_x_block, size=256))
    offs = func.add_op(tir.BinOp(op="add", lhs=pid_x_block, rhs=arange_val))
    mask = func.add_op(tir.Compare(predicate="lt", lhs=offs, rhs=n_val))
    x_ptr = func.add_op(tir.PtrOffset(ptr=x_val, offsets=offs))
    x_tile = func.add_op(tir.Load(ptr=x_ptr, offsets=offs, mask=mask))

    # Role 0: store x * 2
    two = func.add_op(tir.Constant(value=2.0, dtype="f32"))
    x_times_2 = func.add_op(tir.BinOp(op="mul", lhs=x_tile, rhs=two))
    out_ptr0 = func.add_op(tir.PtrOffset(ptr=out_val, offsets=offs))

    role0_body = [tir.Store(ptr=out_ptr0, offsets=offs, value=x_times_2, mask=mask)]
    func.ops.append(tir.SimdgroupRole(role=0, num_roles=2, body=role0_body))

    # Role 1: store x * 3
    three = func.add_op(tir.Constant(value=3.0, dtype="f32"))
    x_times_3 = func.add_op(tir.BinOp(op="mul", lhs=x_tile, rhs=three))
    out_ptr1 = func.add_op(tir.PtrOffset(ptr=out_val, offsets=offs))

    role1_body = [tir.Store(ptr=out_ptr1, offsets=offs, value=x_times_3, mask=mask)]
    func.ops.append(tir.SimdgroupRole(role=1, num_roles=2, body=role1_body))

    return func


def test_simdgroup_role_ir():
    """Test SimdgroupRole Tile IR construction."""
    func = _build_simdgroup_role_ir()
    role_ops = [op for op in func.ops if isinstance(op, tir.SimdgroupRole)]
    assert len(role_ops) == 2
    assert role_ops[0].role == 0
    assert role_ops[1].role == 1
    assert role_ops[0].num_roles == 2


def test_simdgroup_role_lowering():
    """Test SimdgroupRole lowers to MSimdgroupRoleBlock."""
    from metile.ir import metal_ir as mir

    func = _build_simdgroup_role_ir()
    metal_func = lower(func)
    role_blocks = [op for op in metal_func.ops if isinstance(op, mir.MSimdgroupRoleBlock)]
    assert len(role_blocks) == 2
    assert role_blocks[0].role == 0
    assert role_blocks[0].first_sg == 0
    assert role_blocks[1].role == 1
    assert role_blocks[1].first_sg > 0


def test_simdgroup_role_codegen():
    """Test SimdgroupRole generates valid MSL with sgid guards."""
    func = _build_simdgroup_role_ir()
    metal_func = lower(func)
    msl = emit(metal_func)
    assert "sgid" in msl
    assert "simdgroup_index_in_threadgroup" in msl


def test_simdgroup_role_compiles():
    """Test that SimdgroupRole MSL compiles on Metal device."""
    from metile.runtime.metal_device import MetalDevice

    func = _build_simdgroup_role_ir()
    metal_func = lower(func)
    msl = emit(metal_func)
    dev = MetalDevice.get()
    pipeline = dev.compile_msl(msl, metal_func.name)
    assert pipeline is not None


def test_nax_preload_pass_decomposes_two_k_steps_before_mma():
    from metile.compiler.passes import decompose_nax_fragments
    from metile.ir import metal_ir as mir

    ptr_a = mir.MValue("A", PtrType("f32"))
    ptr_b = mir.MValue("B", PtrType("f32"))
    function = mir.MFunction("nax_preload", kernel_type="tensor_ops_gemm")
    function.ops = [
        mir.MNaxGemmSetup(),
        mir.MForLoop(
            iv_name="k",
            end=64,
            step=32,
            body=[
                mir.MNaxGemmRun(ptr_a=ptr_a, ptr_b=ptr_b),
                mir.MNaxGemmRun(ptr_a=ptr_a, ptr_b=ptr_b, k_offset=16),
            ],
        ),
    ]

    decompose_nax_fragments(function)
    loop = next(op for op in function.ops if isinstance(op, mir.MForLoop))
    first_mma = next(
        index for index, op in enumerate(loop.body) if isinstance(op, mir.MNaxFmaFragment)
    )
    assert sum(isinstance(op, mir.MNaxLoadFragment) for op in loop.body[:first_mma]) == 8
    assert sum(isinstance(op, mir.MNaxFmaFragment) for op in loop.body) == 4
    assert not any(isinstance(op, mir.MNaxGemmRun) for op in loop.body)


def test_nax_epilogue_decomposes_over_each_accumulator_fragment():
    from metile.compiler.passes import decompose_nax_fragments
    from metile.ir import metal_ir as mir

    function = mir.MFunction("nax_epilogue", kernel_type="tensor_ops_gemm")
    function.ops = [
        mir.MNaxAccumulatorInit(),
        mir.MNaxGemmEpilogue(operations=[("relu",)]),
    ]

    decompose_nax_fragments(function)
    applications = [op for op in function.ops if isinstance(op, mir.MNaxApplyFragment)]
    assert [op.source for op in applications] == ["d00", "d01", "d10", "d11"]
    assert all(op.operations == [("relu",)] for op in applications)
    assert len({id(op.operations) for op in applications}) == 4
    assert not any(isinstance(op, mir.MNaxGemmEpilogue) for op in function.ops)
