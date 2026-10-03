"""Checked thread ownership and communication choices for scalar tile values."""

from dataclasses import dataclass, fields

from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import ScalarType, TileType


@dataclass(frozen=True)
class ValueOwnership:
    value: str
    dtype: str
    shape: tuple[int, ...]
    layout: ThreadLayout
    elements_per_thread: int = 1


@dataclass(frozen=True)
class LayoutConversion:
    value: str
    source: ThreadLayout
    destination: ThreadLayout
    mechanism: str
    scratch: str | None = None
    publish: str | None = None
    recycle: str | None = None


@dataclass(frozen=True)
class RegisterReduction:
    value: str
    operation: str
    dtype: str
    elements_per_thread: int
    threads: int
    scratch: str | None
    broadcast: str = "simdgroup"


def conversion_map(source: ThreadLayout, destination: ThreadLayout) -> ThreadLayout:
    if (
        source.size != destination.size
        or source.elements_per_thread != destination.elements_per_thread
    ):
        raise ValueError("layout conversion requires identical tile and threadgroup sizes")
    inverse = tuple(source.bit_order.index(bit) for bit in range(len(source.bit_order)))
    order = tuple(destination.bit_order[bit] for bit in inverse)
    difference = source.xor_mask ^ destination.xor_mask
    mask = sum(((difference >> bit) & 1) << index for index, bit in enumerate(inverse))
    return ThreadLayout(order, mask, elements_per_thread=source.elements_per_thread)


def conversion_mechanism(source: ThreadLayout, destination: ThreadLayout) -> str:
    if source == destination:
        return "identity"
    if source.elements_per_thread != 1 or destination.elements_per_thread != 1:
        raise ValueError("nonidentity multi-register layout conversions are not supported")
    ownership = conversion_map(source, destination)
    if ownership == ThreadLayout.identity(source.size):
        return "identity"
    if all(ownership.logical_index(thread) // 32 == thread // 32 for thread in range(source.size)):
        return "simd_shuffle"
    return "threadgroup"


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def has_thread_layouts(function: tir.Function) -> bool:
    return any(
        isinstance(operation, tir.ConvertLayout)
        or (
            operation.result is not None
            and isinstance(operation.result.type, TileType)
            and operation.result.type.layout is not None
        )
        for operation in _walk(function.ops)
    )


def validate_thread_layouts(function: tir.Function) -> tuple[ValueOwnership, ...]:
    from metile.compiler.lowering.common import LoweringError

    if not has_thread_layouts(function):
        return ()
    supported = (
        tir.Constant,
        tir.ProgramId,
        tir.Arange,
        tir.BinOp,
        tir.Unary,
        tir.Compare,
        tir.Select,
        tir.Cast,
        tir.Load,
        tir.Store,
        tir.PtrOffset,
        tir.ConvertLayout,
    )
    if not function.tensors:
        raise LoweringError("ThreadLayout requires descriptor-based tensor memory operations")
    sizes = {operation.size for operation in function.ops if isinstance(operation, tir.Arange)}
    if len(sizes) != 1:
        raise LoweringError("ThreadLayout requires one consistent one-dimensional arange size")
    size = next(iter(sizes))
    geometries = {
        operation.result.type.layout.elements_per_thread
        for operation in _walk(function.ops)
        if operation.result is not None
        and isinstance(operation.result.type, TileType)
        and operation.result.type.layout is not None
    }
    if len(geometries) > 1:
        raise LoweringError("ThreadLayout requires one consistent register and thread geometry")
    elements_per_thread = next(iter(geometries), 1)
    if elements_per_thread > 1:
        supported += (tir.Reduce,)
    try:
        automatic = ThreadLayout.identity(size, elements_per_thread=elements_per_thread)
    except ValueError as error:
        raise LoweringError(str(error)) from error
    ownerships = []
    nonuniform = set()
    for operation in function.ops:
        if not isinstance(operation, supported):
            raise LoweringError(
                f"ThreadLayout supports straight-line pointwise operations, not {type(operation).__name__}"
            )
        if isinstance(operation, tir.Unary) and operation.op.startswith("simd_"):
            raise LoweringError("raw SIMD collectives cannot preserve explicit logical ownership")
        if isinstance(operation, (tir.Load, tir.Store)) and operation.tensor is None:
            raise LoweringError("ThreadLayout requires descriptor-based loads and stores")
        if isinstance(operation, tir.Reduce) and (
            operation.op != "sum"
            or operation.operand is None
            or not isinstance(operation.operand.type, TileType)
            or operation.operand.type.dtype != "f32"
        ):
            raise LoweringError("register-owned reductions require an FP32 tile sum")
        if (
            isinstance(operation, tir.Arange)
            and operation.start is not None
            and (
                not isinstance(operation.start.type, ScalarType)
                or operation.start.type.dtype not in {"i32", "u32"}
                or operation.start.name in nonuniform
            )
        ):
            raise LoweringError("ThreadLayout arange origins must be uniform integer scalars")
        if operation.result is not None:
            result = operation.result
            dependencies = [
                getattr(operation, field.name)
                for field in fields(operation)
                if field.name != "result" and isinstance(getattr(operation, field.name), tir.Value)
            ]
            if not isinstance(operation, tir.Reduce) and (
                isinstance(result.type, TileType)
                or any(value.name in nonuniform for value in dependencies)
            ):
                nonuniform.add(result.name)
                if isinstance(result.type, ScalarType):
                    raise LoweringError("nonuniform scalar values cannot preserve tile ownership")
            if isinstance(result.type, TileType):
                if result.type.shape != (size,):
                    raise LoweringError("ThreadLayout requires matching one-dimensional tile sizes")
                if elements_per_thread > 1 and result.type.layout is None:
                    raise LoweringError("multi-register tiles require explicit ownership")
                layout = result.type.layout or automatic
                if layout.size != size:
                    raise LoweringError("ThreadLayout size must match the threadgroup and tile")
                ownerships.append(
                    ValueOwnership(
                        result.name, result.type.dtype, (size,), layout, elements_per_thread
                    )
                )
        if (
            isinstance(operation, tir.ConvertLayout)
            and elements_per_thread > 1
            and operation.value.type.layout != operation.layout
        ):
            raise LoweringError("nonidentity multi-register layout conversions are not supported")
        if isinstance(operation, tir.ConvertLayout) and operation.value.type.dtype not in {
            "f32",
            "f16",
            "i32",
            "u32",
            "bool",
        }:
            raise LoweringError("layout conversion supports f32, f16, i32, u32 and bool values")
    for tensor in function.tensors:
        if tensor.ptr.name in nonuniform:
            raise LoweringError("ThreadLayout tensor base pointers must be uniform")
        if tensor.ptr.type.address_space != "device":
            raise LoweringError("ThreadLayout currently requires device-backed tensor declarations")
    return tuple(ownerships)


def requires_exchange(function: tir.Function) -> bool:
    return any(
        conversion_mechanism(
            operation.value.type.layout or ThreadLayout.identity(operation.value.type.numel),
            operation.layout,
        )
        == "threadgroup"
        for operation in function.ops
        if isinstance(operation, tir.ConvertLayout)
    )


def validate_register_reductions(function: mir.MFunction):
    from metile.compiler.lowering.common import LoweringError

    recorded = [record.value for record in function.register_reductions]
    materialized = [
        operation.result.name
        for operation in _walk(function.ops)
        if isinstance(operation, mir.MThreadgroupReduce) and operation.replicate_partials
    ]
    if len(set(recorded)) != len(recorded) or sorted(recorded) != sorted(materialized):
        raise LoweringError(
            "register reduction requires exactly one checked contract per operation"
        )
    for record in function.register_reductions:
        reductions = [
            operation
            for operation in function.ops
            if isinstance(operation, mir.MThreadgroupReduce)
            and operation.result.name == record.value
        ]
        if len(reductions) != 1:
            raise LoweringError("register reduction must execute in uniform top-level control flow")
        reduction = reductions[0]
        if (
            reduction.reduce_op != record.operation
            or record.operation != "sum"
            or reduction.dtype != record.dtype
            or record.dtype != "f32"
            or reduction.operand.type != ScalarType("f32")
            or reduction.result.type != ScalarType("f32")
            or reduction.block_size != record.threads
            or function.threadgroup_size != (record.threads, 1, 1)
            or type(record.elements_per_thread) is not int
            or record.elements_per_thread not in {2, 4, 8, 16, 32}
            or not function.value_layouts
            or any(
                not isinstance(ownership.layout, ThreadLayout)
                or type(ownership.elements_per_thread) is not int
                or ownership.elements_per_thread != ownership.layout.elements_per_thread
                or ownership.layout.elements_per_thread != record.elements_per_thread
                or ownership.layout.thread_count != record.threads
                or ownership.shape != (ownership.layout.size,)
                for ownership in function.value_layouts
            )
            or (record.threads == 32 and record.scratch is not None)
            or record.broadcast != "simdgroup"
            or not reduction.replicate_partials
        ):
            raise LoweringError("register reduction violates its geometry or FP32 sum contract")
        if record.threads > 32:
            allocations = [
                operation
                for operation in function.ops[: function.ops.index(reduction)]
                if isinstance(operation, mir.MThreadgroupAlloc)
                and operation.alloc_name == record.scratch
            ]
            if (
                reduction.shared_name != record.scratch
                or len(allocations) != 1
                or allocations[0].size != record.threads // 32
                or allocations[0].elem_type != "float"
            ):
                raise LoweringError("register reduction scratch does not match its contract")
            for operation in _walk(function.ops):
                if operation is reduction or operation is allocations[0]:
                    continue
                for definition in fields(operation):
                    if definition.name == "result":
                        continue
                    value = getattr(operation, definition.name)
                    if (isinstance(value, str) and value == record.scratch) or (
                        isinstance(value, mir.MValue) and value.name == record.scratch
                    ):
                        raise LoweringError(
                            "register reduction scratch must remain private and immutable"
                        )


def validate_layout_exchanges(function: mir.MFunction):
    from metile.compiler.lowering.common import LoweringError

    for conversion in function.layout_conversions:
        if conversion.mechanism != "threadgroup":
            continue
        loads = [
            (index, operation)
            for index, operation in enumerate(function.ops)
            if isinstance(operation, mir.MThreadgroupLoad)
            and operation.result is not None
            and operation.result.name == conversion.value
        ]
        if len(loads) != 1:
            raise LoweringError("layout exchange must remain in uniform top-level control flow")
        index, operation = loads[0]
        allocations = [
            allocation
            for allocation in function.ops
            if isinstance(allocation, mir.MThreadgroupAlloc)
            and allocation.alloc_name == conversion.scratch
        ]
        mapping = operation.index.defining_op if operation.index is not None else None
        if (
            len(allocations) != 1
            or allocations[0].size != conversion.source.size
            or allocations[0].elem_type != operation.result.type.to_msl()
            or not isinstance(mapping, mir.MThreadIndexMap)
            or mapping.layout != conversion_map(conversion.source, conversion.destination)
            or not isinstance(mapping.thread.defining_op, mir.ThreadPositionInThreadgroup)
            or mapping.thread.defining_op.axis != 0
            or function.threadgroup_size != (conversion.source.size, 1, 1)
        ):
            raise LoweringError("layout exchange storage or owner mapping violates its contract")
        preceding = function.ops[max(0, index - 2) : index]
        following = function.ops[index + 1 : index + 2]
        if (
            len(preceding) != 2
            or not isinstance(preceding[0], mir.MThreadgroupStore)
            or preceding[0].array_name != conversion.scratch
            or preceding[0].mask is not None
            or operation.mask is not None
            or preceding[0].index is not mapping.thread
            or operation.array_name != conversion.scratch
            or not isinstance(preceding[1], mir.MBarrier)
            or len(following) != 1
            or not isinstance(following[0], mir.MBarrier)
            or any(
                barrier.kind != "threadgroup"
                or barrier.flags != "mem_threadgroup"
                or barrier.condition
                for barrier in (preceding[1], following[0])
            )
        ):
            raise LoweringError("layout exchange requires uniform publish and recycle barriers")
