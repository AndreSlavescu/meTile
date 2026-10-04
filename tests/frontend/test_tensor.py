import operator

import numpy as np
import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.printer import print_metal_ir, print_tile_ir
from metile.ir.types import PtrType, ScalarType, TileType


def _pointer(name="values", address_space="device"):
    return TracingProxy(tir.Value(name, PtrType("f32", address_space)))


def _scalar(name, dtype="i32"):
    return TracingProxy(tir.Value(name, ScalarType(dtype)))


def _evaluate(value, arguments):
    if value.name in arguments:
        return arguments[value.name]
    operation = value.defining_op
    if isinstance(operation, tir.Constant):
        return operation.value
    if isinstance(operation, tir.BinOp):
        implementation = {
            "add": operator.add,
            "mul": operator.mul,
            "bitand": operator.and_,
        }[operation.op]
        return implementation(
            _evaluate(operation.lhs, arguments), _evaluate(operation.rhs, arguments)
        )
    if isinstance(operation, tir.Compare):
        implementation = {"ge": operator.ge, "lt": operator.lt}[operation.predicate]
        return implementation(
            _evaluate(operation.lhs, arguments), _evaluate(operation.rhs, arguments)
        )
    raise AssertionError(f"unexpected index operation: {operation}")


def test_tensor_memory_contract_survives_tracing():
    pointer = _pointer()
    rows = _scalar("rows")
    columns = _scalar("columns")
    with TracingContext("descriptor") as context:
        view = metile.tensor(pointer, shape=(rows, columns), access="read")
        loaded = view.load((1, 2), other=-3.5)

    memory = context.func.tensors[0]
    assert view.memory is memory
    assert memory.ptr is pointer._value
    assert memory.shape == (rows._value, columns._value)
    assert memory.strides[0] is columns._value
    assert _evaluate(memory.strides[1], {}) == 1
    assert memory.access == "read"
    assert memory.address_space == "device"
    assert loaded._value.defining_op.tensor is memory
    assert _evaluate(loaded._value.defining_op.other, {}) == -3.5


@pytest.mark.parametrize(
    ("coordinates", "expected_offset", "expected_valid"),
    [
        ((1, 2, 3), 43, True),
        ((1, 2, 4), 44, False),
        ((2, 0, 0), 48, False),
        ((0, -1, 0), -8, False),
    ],
)
def test_dynamic_strided_indexing_and_each_axis_bounds(
    coordinates, expected_offset, expected_valid
):
    with TracingContext("strided"):
        view = metile.tensor(
            _pointer(),
            shape=(_scalar("depth"), 3, 4),
            strides=(_scalar("plane_stride"), 8, 1),
        )
        loaded = view.load(tuple(_scalar(f"coordinate_{axis}") for axis in range(3)))
    operation = loaded._value.defining_op
    arguments = {"depth": 2, "plane_stride": 24}
    arguments.update({f"coordinate_{axis}": index for axis, index in enumerate(coordinates)})
    assert _evaluate(operation.offsets, arguments) == expected_offset
    assert bool(_evaluate(operation.mask, arguments)) is expected_valid


def test_row_major_strides_and_empty_extent():
    with TracingContext("contiguous"):
        view = metile.tensor(_pointer(), shape=(2, 3, 4))
        empty = metile.tensor(_pointer("empty"), shape=(0, 4))
        loaded = empty.load((0, 0))

    assert tuple(_evaluate(stride, {}) for stride in view.memory.strides) == (12, 4, 1)
    assert not _evaluate(loaded._value.defining_op.mask, {})


def test_vector_coordinates_use_elementwise_indexing_and_bounds():
    row_indices = TracingProxy(tir.Value("rows", TileType((4,), "i32")))
    column_indices = TracingProxy(tir.Value("columns", TileType((4,), "i32")))
    with TracingContext("vector"):
        view = metile.tensor(_pointer(), shape=(2, 3), strides=(1, 2))
        loaded = view.load((row_indices, column_indices))
    operation = loaded._value.defining_op
    arguments = {"rows": np.array([0, 1, -1, 2]), "columns": np.array([2, 1, 0, 0])}
    np.testing.assert_array_equal(_evaluate(operation.offsets, arguments), [4, 3, -1, 2])
    np.testing.assert_array_equal(_evaluate(operation.mask, arguments), [True, True, False, False])
    assert loaded._value.type == TileType((4,), "f32")


def test_store_keeps_its_own_bounds_and_descriptor():
    coordinate = _scalar("index")
    with TracingContext("independent_bounds") as context:
        source = metile.tensor(_pointer("source"), shape=(8,), access="read")
        destination = metile.tensor(_pointer("destination"), shape=(4,), access="write")
        loaded = source.load(coordinate)
        destination.store(coordinate, loaded)
    stored = next(operation for operation in context.func.ops if isinstance(operation, tir.Store))
    assert stored.tensor is destination.memory
    assert _evaluate(loaded._value.defining_op.mask, {"index": 6})
    assert not _evaluate(stored.mask, {"index": 6})
    assert stored.value is loaded._value


def test_threadgroup_address_space_is_inherited_without_a_cast():
    with TracingContext("shared"):
        allocation = metile.shared(32)
        view = metile.tensor(allocation, shape=(32,))
        loaded = view.load(0)
    assert view.memory.address_space == "threadgroup"
    assert loaded._value.defining_op.ptr.type.address_space == "threadgroup"


def test_matrix_tile_accesses_preserve_shape_stride_and_origins():
    row = _scalar("row")
    column = _scalar("column")
    with TracingContext("matrix_tile") as context:
        source = metile.tensor(
            _pointer("source"), shape=(20, 30), block_shape=(8, 16), access="read"
        )
        destination = metile.tensor(
            _pointer("destination"), shape=(20, 30), block_shape=(8, 16), access="write"
        )
        loaded = source.load((row, column))
        destination.store((row, column), loaded)
    load_operation = loaded._value.defining_op
    store_operation = next(
        operation for operation in context.func.ops if isinstance(operation, tir.TileStore)
    )
    assert isinstance(load_operation, tir.TileLoad)
    assert load_operation.row_offset is row._value
    assert load_operation.col_offset is column._value
    assert load_operation.tile_shape == (8, 16)
    assert load_operation.tensor is source.memory
    assert store_operation.tensor is destination.memory
    assert _evaluate(load_operation.stride, {}) == 30


@pytest.mark.parametrize(
    ("options", "exception", "message"),
    [
        ({"shape": ()}, ValueError, "nonempty"),
        ({"shape": (-1,)}, ValueError, "nonnegative"),
        ({"shape": (1.5,)}, TypeError, "integer"),
        ({"shape": (True,)}, TypeError, "integer"),
        ({"shape": (4,), "strides": (1, 2)}, ValueError, "rank"),
        ({"shape": (4,), "strides": (0.5,)}, TypeError, "integer"),
        ({"shape": (4,), "access": "mutable"}, ValueError, "access"),
        ({"shape": (4,), "address_space": "threadgroup"}, ValueError, "allocation"),
        ({"shape": (4,), "block_shape": (4, 4)}, ValueError, "rank-two"),
        ({"shape": (4, 4), "block_shape": (0, 4)}, ValueError, "positive"),
        ({"shape": (4, 4), "block_shape": (4.5, 4)}, ValueError, "compile-time"),
    ],
)
def test_invalid_memory_contracts_are_rejected(options, exception, message):
    with TracingContext("invalid"), pytest.raises(exception, match=message):
        metile.tensor(_pointer(), **options)


def test_dynamic_dimensions_and_strides_must_be_integer_scalars():
    with TracingContext("invalid_types"):
        with pytest.raises(TypeError, match="scalar integer"):
            metile.tensor(_pointer(), shape=(_scalar("dimension", "f32"),))
        with pytest.raises(TypeError, match="scalar integer"):
            metile.tensor(
                _pointer(), shape=(TracingProxy(tir.Value("tile", TileType((4,), "i32"))),)
            )
        with pytest.raises(TypeError, match="scalar integer"):
            metile.tensor(_pointer(), shape=(4,), strides=(_scalar("stride", "f32"),))
        with pytest.raises(TypeError, match="memory allocation"):
            metile.tensor(_scalar("value"), shape=(4,))


def test_access_permissions_are_enforced():
    with TracingContext("permissions"):
        readonly = metile.tensor(_pointer(), shape=(4,), access="read")
        writeonly = metile.tensor(_pointer(), shape=(4,), access="write")
        with pytest.raises(ValueError, match="read-only"):
            readonly.store(0, 1.0)
        with pytest.raises(ValueError, match="write-only"):
            writeonly.load(0)
        with pytest.raises(ValueError, match="read-only"):
            metile.tensor(_pointer(address_space="constant"), shape=(4,))
        constant = metile.tensor(_pointer(address_space="constant"), shape=(4,), access="read")
        assert constant.memory.address_space == "constant"


def test_coordinate_rank_dtype_and_vector_shapes_are_checked():
    with TracingContext("invalid_coordinates"):
        view = metile.tensor(_pointer(), shape=(4, 4))
        with pytest.raises(ValueError, match="rank"):
            view.load((0,))
        with pytest.raises(TypeError, match="integer"):
            view.load((0.5, 0))
        short = TracingProxy(tir.Value("short", TileType((4,), "i32")))
        long = TracingProxy(tir.Value("long", TileType((8,), "i32")))
        with pytest.raises(ValueError, match="identical"):
            view.load((short, long))
        with pytest.raises(ValueError, match="coordinate shape"):
            view.store((short, 0), long)
        with pytest.raises(TypeError, match="fill must be scalar"):
            view.load((0, 0), other=short)


def test_matrix_tile_unsupported_controls_fail_explicitly():
    with TracingContext("invalid_matrix_controls"):
        view = metile.tensor(_pointer(), shape=(16, 16), block_shape=(8, 8))
        with pytest.raises(ValueError, match="zero fill"):
            view.load((0, 0), other=1)
        with pytest.raises(ValueError, match="zero fill"):
            view.load((0, 0), other=_scalar("fill", "f32"))
        with pytest.raises(TypeError, match="scalar integer"):
            view.load((TracingProxy(tir.Value("vector", TileType((8,), "i32"))), 0))
        with pytest.raises(ValueError, match="block shape"):
            view.store((0, 0), metile.zeros((4, 4)))
        with pytest.raises(ValueError, match="device memory"):
            metile.tensor(_pointer(address_space="threadgroup"), shape=(16, 16), block_shape=(8, 8))


def test_constexpr_dimensions_strides_coordinates_and_fill():
    with TracingContext("constexpr"):
        view = metile.tensor(
            _pointer(), shape=(metile.constexpr(8),), strides=(metile.constexpr(2),)
        )
        loaded = view.load(metile.constexpr(3), other=metile.constexpr(-1.0))
    assert _evaluate(loaded._value.defining_op.offsets, {}) == 6
    assert _evaluate(loaded._value.defining_op.other, {}) == -1.0


def test_existing_mma_atom_descriptor_remains_compatible():
    atom = metile.TensorDescriptor(16, 32, 16)
    assert atom == metile.TensorDescriptor(16, 32, 16)
    assert atom.M == 16
    assert atom.N == 32
    assert atom.K == 16


def test_tensor_requires_active_kernel_tracing():
    with pytest.raises(RuntimeError, match=r"inside a @metile\.kernel"):
        metile.tensor(_pointer(), shape=(4,))


def test_ir_printer_declares_dynamic_memory_contracts_and_distinguishes_aliases():
    with TracingContext("visible_contracts") as context:
        source = metile.tensor(_pointer(), shape=(_scalar("rows"), 8), access="read")
        destination = metile.tensor(
            _pointer(), shape=(_scalar("rows"), 4), strides=(8, 1), access="write"
        )
        value = source.load((0, 0), other=-7.0)
        destination.store((0, 0), value)
    printed = print_tile_ir(context.func)
    assert (
        "tensor @tensor0(%values, shape=(%rows, 8), strides=(8, 1), access=read, address_space=device)"
        in printed
    )
    assert (
        "tensor @tensor1(%values, shape=(%rows, 4), strides=(8, 1), access=write, address_space=device)"
        in printed
    )
    load_line = next(line for line in printed.splitlines() if " = load(" in line)
    store_line = next(line for line in printed.splitlines() if "  store(" in line)
    assert "tensor=@tensor0" in load_line
    assert "other=%" in load_line
    assert "mask=%" in load_line
    assert "tensor=@tensor1" in store_line
    assert "mask=%" in store_line


def test_ir_printer_keeps_tensor_references_inside_loops():
    with TracingContext("visible_loop") as context:
        view = metile.tensor(_pointer(), shape=(8,))
        for index in metile.tile_range(0, 8, 1):
            value = view.load(index)
            view.store(index, value)
    printed = print_tile_ir(context.func)
    assert "shape=(8,), strides=(1,)" in printed
    assert printed.count("tensor=@tensor0") == 2


def test_ir_printer_shows_matrix_tile_block_shapes():
    with TracingContext("visible_tile") as context:
        view = metile.tensor(_pointer(), shape=(16, 32), block_shape=(8, 8))
        value = view.load((0, 0))
        view.store((0, 0), value)
    printed = print_tile_ir(context.func)
    assert "block_shape=(8, 8)" in printed
    assert printed.count("tensor=@tensor0") == 2


@pytest.mark.parametrize("address_space", ["device", "threadgroup"])
def test_lowered_ir_printer_preserves_predicates_and_fill_values(address_space):
    from metile.compiler.lowering import lower

    with TracingContext("visible_masked_access") as context:
        if address_space == "threadgroup":
            pointer = metile.shared(8)
        else:
            pointer = _pointer()
            context.func.params.append(tir.Param("values", PtrType("f32"), is_output=True))
        view = metile.tensor(pointer, shape=(8,))
        value = view.load(0, other=-7.0)
        view.store(0, value)
    printed = print_metal_ir(lower(context.func))
    load_line = next(line for line in printed.splitlines() if f"{address_space}_load(" in line)
    store_line = next(line for line in printed.splitlines() if f"{address_space}_store(" in line)
    assert "mask=%" in load_line
    assert "other=%" in load_line
    assert "mask=%" in store_line


@pytest.mark.parametrize("value", [-(1 << 31) - 1, 1 << 31])
@pytest.mark.parametrize("field", ["dimension", "stride", "coordinate"])
def test_static_index_fields_reject_values_outside_signed_int32(value, field):
    with TracingContext("index_range"), pytest.raises(ValueError, match="signed 32-bit"):
        if field == "dimension":
            metile.tensor(_pointer(), shape=(value,))
        elif field == "stride":
            metile.tensor(_pointer(), shape=(1,), strides=(value,))
        else:
            metile.tensor(_pointer(), shape=(1,)).load(value)


def test_constexpr_and_scalar_constant_indices_cannot_bypass_range_validation():
    with TracingContext("constant_index_range"):
        with pytest.raises(ValueError, match="signed 32-bit"):
            metile.tensor(_pointer(), shape=(metile.constexpr(1 << 31),))
        with pytest.raises(ValueError, match="signed 32-bit"):
            metile.tensor(_pointer(), shape=(metile.scalar(1 << 31, dtype="u32"),))
        with pytest.raises(ValueError, match="signed 32-bit"):
            metile.tensor(_pointer(), shape=(1,)).load(metile.scalar(1 << 31))


def test_contiguous_stride_multiplication_rejects_overflow():
    with TracingContext("stride_product_range"), pytest.raises(ValueError, match="product"):
        metile.tensor(_pointer(), shape=(1, 65536, 65536))


def test_constant_index_multiplication_rejects_overflow_with_dynamic_shape():
    with TracingContext("index_product_range"):
        view = metile.tensor(_pointer(), shape=(_scalar("length"),), strides=((1 << 31) - 1,))
        with pytest.raises(ValueError, match="product"):
            view.load(2)


def test_constant_index_addition_rejects_overflow_with_dynamic_shape():
    with TracingContext("index_sum_range"):
        view = metile.tensor(
            _pointer(),
            shape=(_scalar("rows"), _scalar("columns")),
            strides=((1 << 31) - 1, (1 << 31) - 1),
        )
        with pytest.raises(ValueError, match="element offset"):
            view.load((1, 1))


@pytest.mark.parametrize(
    ("shape", "strides"),
    [
        ((65536, 65536), None),
        ((2, 2), ((1 << 31) - 1, (1 << 31) - 1)),
        ((3,), (-(1 << 30) - 1,)),
    ],
)
def test_static_address_span_rejects_overflow(shape, strides):
    with TracingContext("span_range"), pytest.raises(ValueError, match="offset"):
        metile.tensor(_pointer(), shape=shape, strides=strides)


def test_int32_boundary_offsets_and_negative_strides_remain_valid():
    with TracingContext("valid_index_range"):
        forward = metile.tensor(_pointer(), shape=(2,), strides=((1 << 31) - 1,))
        backward = metile.tensor(_pointer(), shape=(2,), strides=(-(1 << 31),))
        strided = metile.tensor(_pointer(), shape=(2, 3), strides=(-4, 1))
        forward_value = forward.load(1)
        backward_value = backward.load(1)
        strided_value = strided.load((1, 2))
    assert _evaluate(forward_value._value.defining_op.offsets, {}) == (1 << 31) - 1
    assert _evaluate(backward_value._value.defining_op.offsets, {}) == -(1 << 31)
    assert _evaluate(strided_value._value.defining_op.offsets, {}) == -2


def test_empty_views_do_not_require_representable_unreachable_address_spans():
    with TracingContext("empty_span"):
        contiguous = metile.tensor(_pointer(), shape=(0, 65536, 65536))
        explicit = metile.tensor(
            _pointer(), shape=(0, 2, 2), strides=(1, (1 << 31) - 1, -(1 << 31))
        )
        loaded = contiguous.load((0, 0, 0))
    assert not _evaluate(loaded._value.defining_op.mask, {})
    assert tuple(_evaluate(stride, {}) for stride in contiguous.memory.strides) == (0, 65536, 1)
    assert explicit.memory.shape[0].defining_op.value == 0
