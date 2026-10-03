from __future__ import annotations

from metile.frontend.tracing import TracingProxy, _get_ctx, _to_value, constexpr
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType, ScalarType, TileType

_INDEX_MIN = -(1 << 31)
_INDEX_MAX = (1 << 31) - 1


def _unwrap(value):
    return value.value if isinstance(value, constexpr) else value


def _validate_integer_range(value: int, label: str):
    if value < _INDEX_MIN or value > _INDEX_MAX:
        raise ValueError(f"{label} must fit in signed 32-bit indexing")


def _integer_value(value, label: str, *, allow_tile: bool = False) -> tir.Value:
    value = _unwrap(value)
    if isinstance(value, bool) or not isinstance(value, (int, TracingProxy)):
        raise TypeError(f"{label} must be an integer or an integer tracing value")
    if isinstance(value, int):
        _validate_integer_range(value, label)
    result = _to_value(value)
    allowed_types = (ScalarType, TileType) if allow_tile else (ScalarType,)
    if not isinstance(result.type, allowed_types) or result.type.dtype not in {"i32", "u32"}:
        raise TypeError(f"{label} must be {'scalar or tile' if allow_tile else 'scalar'} integer")
    if isinstance(result.defining_op, tir.Constant):
        _validate_integer_range(result.defining_op.value, label)
    return result


def _constant(value: tir.Value):
    if isinstance(value.defining_op, tir.Constant):
        return value.defining_op.value
    return None


def _product(left: tir.Value, right: tir.Value) -> tir.Value:
    left_constant = _constant(left)
    right_constant = _constant(right)
    if left_constant == 1:
        return right
    if right_constant == 1:
        return left
    if left_constant is not None and right_constant is not None:
        return _integer_value(left_constant * right_constant, "tensor stride or index product")
    return _get_ctx().add_op(tir.BinOp(op="mul", lhs=left, rhs=right))


def _index_sum(left: tir.Value, right: tir.Value) -> tir.Value:
    left_constant = _constant(left)
    right_constant = _constant(right)
    if left_constant is not None and right_constant is not None:
        return _integer_value(left_constant + right_constant, "tensor element offset")
    return _get_ctx().add_op(tir.BinOp(op="add", lhs=left, rhs=right))


def _validate_static_span(dimensions: tuple[tir.Value, ...], strides: tuple[tir.Value, ...]):
    extents = tuple(_constant(dimension) for dimension in dimensions)
    steps = tuple(_constant(stride) for stride in strides)
    if 0 in extents or None in extents or None in steps:
        return
    axis_offsets = tuple((extent - 1) * step for extent, step in zip(extents, steps))
    _validate_integer_range(sum(min(0, offset) for offset in axis_offsets), "tensor minimum offset")
    _validate_integer_range(sum(max(0, offset) for offset in axis_offsets), "tensor maximum offset")


class Tensor:
    """A traced memory view with compiler-generated indexing and access bounds.

    Coordinates are elementwise: vector coordinates must have identical shapes.
    A block shape instead selects a two-dimensional matrix tile from scalar origins.
    """

    def __init__(self, memory: tir.TensorMemory):
        self.memory = memory

    def _coordinates(self, indices) -> tuple[tir.Value, ...]:
        if not isinstance(indices, (tuple, list)):
            indices = (indices,)
        if len(indices) != len(self.memory.shape):
            raise ValueError("tensor coordinates must match the tensor rank")
        coordinates = tuple(
            _integer_value(index, "tensor coordinate", allow_tile=self.memory.block_shape is None)
            for index in indices
        )
        vector_shapes = {
            coordinate.type.shape
            for coordinate in coordinates
            if isinstance(coordinate.type, TileType)
        }
        if len(vector_shapes) > 1:
            raise ValueError("vector tensor coordinates must have identical shapes")
        return coordinates

    def _access(self, coordinates: tuple[tir.Value, ...]):
        context = _get_ctx()
        offset = None
        mask = None
        zero = _to_value(0)
        for coordinate, dimension, stride in zip(
            coordinates, self.memory.shape, self.memory.strides
        ):
            axis_offset = _product(coordinate, stride)
            offset = axis_offset if offset is None else _index_sum(offset, axis_offset)
            lower = context.add_op(tir.Compare(predicate="ge", lhs=coordinate, rhs=zero))
            upper = context.add_op(tir.Compare(predicate="lt", lhs=coordinate, rhs=dimension))
            axis_mask = context.add_op(tir.BinOp(op="bitand", lhs=lower, rhs=upper))
            mask = (
                axis_mask
                if mask is None
                else context.add_op(tir.BinOp(op="bitand", lhs=mask, rhs=axis_mask))
            )
        pointer = context.add_op(tir.PtrOffset(ptr=self.memory.ptr, offsets=offset))
        return pointer, offset, mask

    def load(self, indices, other=0) -> TracingProxy:
        """Load at element coordinates, filling out-of-bounds coordinates with ``other``."""
        if self.memory.access == "write":
            raise ValueError("cannot load from a write-only tensor")
        coordinates = self._coordinates(indices)
        context = _get_ctx()
        if self.memory.block_shape is not None:
            other = _unwrap(other)
            if not isinstance(other, (int, float)) or other != 0:
                raise ValueError("matrix tile loads currently require zero fill")
            operation = tir.TileLoad(
                ptr=self.memory.ptr,
                row_offset=coordinates[0],
                col_offset=coordinates[1],
                stride=self.memory.strides[0],
                tile_shape=self.memory.block_shape,
                tensor=self.memory,
            )
        else:
            fill = _to_value(_unwrap(other))
            if not isinstance(fill.type, ScalarType):
                raise TypeError("tensor load fill must be scalar")
            pointer, offset, mask = self._access(coordinates)
            operation = tir.Load(
                ptr=pointer, offsets=offset, mask=mask, other=fill, tensor=self.memory
            )
        return TracingProxy(context.add_op(operation))

    def store(self, indices, value):
        """Store at element coordinates, skipping out-of-bounds coordinates."""
        if self.memory.access == "read":
            raise ValueError("cannot store to a read-only tensor")
        coordinates = self._coordinates(indices)
        stored = _to_value(_unwrap(value))
        if not isinstance(stored.type, (ScalarType, TileType)):
            raise TypeError("tensor store value must be scalar or tile")
        context = _get_ctx()
        if self.memory.block_shape is not None:
            if (
                not isinstance(stored.type, TileType)
                or stored.type.shape != self.memory.block_shape
            ):
                raise ValueError("stored tile shape must match the tensor block shape")
            operation = tir.TileStore(
                ptr=self.memory.ptr,
                row_offset=coordinates[0],
                col_offset=coordinates[1],
                stride=self.memory.strides[0],
                value=stored,
                tile_shape=self.memory.block_shape,
                tensor=self.memory,
            )
        else:
            coordinate_shape = next(
                (
                    coordinate.type.shape
                    for coordinate in coordinates
                    if isinstance(coordinate.type, TileType)
                ),
                None,
            )
            if isinstance(stored.type, TileType) and stored.type.shape != coordinate_shape:
                raise ValueError("stored vector shape must match the tensor coordinate shape")
            pointer, offset, mask = self._access(coordinates)
            operation = tir.Store(
                ptr=pointer, offsets=offset, value=stored, mask=mask, tensor=self.memory
            )
        context.add_op(operation)


def tensor(
    pointer,
    *,
    shape: tuple,
    strides: tuple | None = None,
    access: str = "readwrite",
    block_shape: tuple[int, int] | None = None,
    address_space: str | None = None,
) -> Tensor:
    """Declare a tensor's memory contract at the top of a kernel.

    Shapes and element strides may contain runtime integer scalar values. Omitted
    strides describe contiguous row-major storage. Address space is inferred from
    the allocation and cannot be changed by a view. ``block_shape`` selects the
    matrix-tile path; backend lowering validates supported matrix layouts.
    """
    context = _get_ctx()
    if not isinstance(pointer, TracingProxy) or not isinstance(pointer._value.type, PtrType):
        raise TypeError("tensor pointer must refer to a traced memory allocation")
    pointer_value = pointer._value
    pointer_space = pointer_value.type.address_space
    if pointer_space not in {"device", "threadgroup", "constant"}:
        raise ValueError(f"unsupported tensor address space: {pointer_space}")
    if address_space is not None and address_space != pointer_space:
        raise ValueError("tensor address space must match its allocation")
    if access not in {"read", "write", "readwrite"}:
        raise ValueError("tensor access must be read, write, or readwrite")
    if pointer_space == "constant" and access != "read":
        raise ValueError("constant memory tensors must be read-only")
    if not isinstance(shape, tuple) or not shape:
        raise ValueError("tensor shape must be a nonempty tuple")
    dimensions = tuple(_integer_value(dimension, "tensor dimension") for dimension in shape)
    if any(
        _constant(dimension) is not None and _constant(dimension) < 0 for dimension in dimensions
    ):
        raise ValueError("tensor dimensions must be nonnegative")
    if strides is None:
        stride = _to_value(1)
        reversed_strides = [stride]
        empty = any(_constant(dimension) == 0 for dimension in dimensions)
        for dimension in reversed(dimensions[1:]):
            dimension_constant = _constant(dimension)
            stride_constant = _constant(stride)
            if (
                empty
                and dimension_constant is not None
                and stride_constant is not None
                and dimension_constant * stride_constant > _INDEX_MAX
            ):
                stride = _to_value(0)
            else:
                stride = _product(dimension, stride)
            reversed_strides.append(stride)
        stride_values = tuple(reversed(reversed_strides))
    else:
        if not isinstance(strides, tuple) or len(strides) != len(dimensions):
            raise ValueError("tensor strides must match the tensor rank")
        stride_values = tuple(_integer_value(stride, "tensor stride") for stride in strides)
    _validate_static_span(dimensions, stride_values)
    if block_shape is not None:
        if not isinstance(block_shape, tuple) or len(block_shape) != 2 or len(dimensions) != 2:
            raise ValueError("matrix block shapes require a rank-two tensor and two dimensions")
        block_shape = tuple(_unwrap(dimension) for dimension in block_shape)
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in block_shape
        ):
            raise ValueError("matrix block dimensions must be positive compile-time integers")
        if pointer_space != "device":
            raise ValueError("matrix tile tensors currently require device memory")
    memory = tir.TensorMemory(
        ptr=pointer_value,
        shape=dimensions,
        strides=stride_values,
        access=access,
        address_space=pointer_space,
        block_shape=block_shape,
    )
    context.func.tensors.append(memory)
    return Tensor(memory)
