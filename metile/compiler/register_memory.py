"""Pack contiguous accesses within one scalarized tensor memory operation."""

from dataclasses import fields

from metile.ir import metal_ir as mir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import BOOL, PtrType, ScalarType, VectorType


def _operands(operation):
    for definition in fields(operation):
        if definition.name == "result":
            continue
        value = getattr(operation, definition.name)
        if isinstance(value, mir.MValue):
            yield value
        elif isinstance(value, tuple):
            yield from (item for item in value if isinstance(item, mir.MValue))


def _affine(value, threads, cache):
    if not isinstance(value, mir.MValue):
        return None
    value = mir.resolve(value)
    key = id(value)
    if key in cache:
        return cache[key]
    cache[key] = None
    if value.type not in (ScalarType("i32"), ScalarType("u32")):
        return None
    operation = value.defining_op
    result = None
    zeros = (0,) * threads
    if operation is None or isinstance(operation, mir.ThreadgroupPositionInGrid):
        result = ({key: 1}, zeros)
    elif isinstance(operation, mir.MConstant) and type(operation.value) is int:
        result = ({}, (operation.value,) * threads)
    elif isinstance(operation, mir.ThreadPositionInThreadgroup) and operation.axis == 0:
        result = ({}, tuple(range(threads)))
    elif isinstance(operation, mir.MCast):
        result = _affine(operation.value, threads, cache)
    elif isinstance(operation, mir.MThreadIndexMap):
        source = _affine(operation.thread, threads, cache)
        if source is not None and not source[0] and isinstance(operation.layout, ThreadLayout):
            layout = operation.layout
            result = (
                {},
                tuple(
                    layout.xor_mask
                    ^ sum(
                        ((packed >> source_bit) & 1) << logical_bit
                        for logical_bit, source_bit in enumerate(layout.bit_order)
                    )
                    for packed in source[1]
                ),
            )
    elif isinstance(operation, mir.MBinOp):
        left = _affine(operation.lhs, threads, cache)
        right = _affine(operation.rhs, threads, cache)
        if left is not None and right is not None:
            if operation.op == "bitand" and not left[0] and not right[0]:
                result = ({}, tuple(first & second for first, second in zip(left[1], right[1])))
            elif operation.op in {"add", "sub"}:
                sign = 1 if operation.op == "add" else -1
                symbols = dict(left[0])
                for symbol, coefficient in right[0].items():
                    symbols[symbol] = symbols.get(symbol, 0) + sign * coefficient
                result = (
                    {symbol: coefficient for symbol, coefficient in symbols.items() if coefficient},
                    tuple(first + sign * second for first, second in zip(left[1], right[1])),
                )
            elif operation.op == "mul":
                for constant, operand in ((left, right), (right, left)):
                    if not constant[0] and len(set(constant[1])) == 1:
                        scale = constant[1][0]
                        result = (
                            {
                                symbol: coefficient * scale
                                for symbol, coefficient in operand[0].items()
                                if coefficient * scale
                            },
                            tuple(item * scale for item in operand[1]),
                        )
                        break
                if result is None and len(set(left[1])) == len(set(right[1])) == 1:
                    result = ({key: 1}, zeros)
    cache[key] = result
    return result


def _consecutive(indices, threads):
    cache = {}
    expressions = [_affine(index, threads, cache) for index in indices]
    if any(expression is None for expression in expressions):
        return False
    symbols, first = expressions[0]
    return all(
        coefficients == symbols
        and all(current - base == lane for base, current in zip(first, values))
        for lane, (coefficients, values) in enumerate(expressions)
    )


def _depends_on(value, results, seen):
    value = mir.resolve(value)
    if id(value) in results:
        return True
    if id(value) in seen or value.defining_op is None:
        return False
    seen.add(id(value))
    return any(_depends_on(operand, results, seen) for operand in _operands(value.defining_op))


def group_register_memory(operations, threads, fresh_name):
    """Group one original Load/Store expansion without moving any memory effect.

    The affine check identifies useful candidates. Emitted widened-index guards
    additionally preserve scalar semantics if a runtime 32-bit origin wraps.
    """
    if not operations or len(operations) % 4:
        return operations
    kind = type(operations[0])
    if kind not in (mir.DeviceLoad, mir.DeviceStore) or any(
        type(op) is not kind for op in operations
    ):
        return operations
    results = {id(operation.result) for operation in operations if operation.result is not None}
    if any(
        _depends_on(operand, results, set())
        for operation in operations
        for operand in _operands(operation)
    ):
        return operations
    grouped = []
    for offset in range(0, len(operations), 4):
        accesses = operations[offset : offset + 4]
        pointer = mir.resolve(accesses[0].ptr)
        indices = tuple(access.index for access in accesses)
        if (
            not isinstance(pointer.type, PtrType)
            or pointer.type.address_space != "device"
            or pointer.type.dtype not in {"f16", "f32"}
            or any(mir.resolve(access.ptr) is not pointer for access in accesses)
            or not _consecutive(indices, threads)
            or (
                kind is mir.DeviceLoad
                and any(access.dtype != pointer.type.dtype for access in accesses)
            )
        ):
            grouped.extend(accesses)
            continue
        arguments = {
            "ptr": pointer,
            "indices": indices,
            "masks": tuple(access.mask for access in accesses),
            "dtype": pointer.type.dtype,
        }
        if kind is mir.DeviceStore:
            grouped.append(
                mir.MVectorStore(**arguments, values=tuple(access.value for access in accesses))
            )
            continue
        vector = mir.MVectorLoad(**arguments, others=tuple(access.other for access in accesses))
        vector.result = mir.MValue(fresh_name("_metile_vector_load"), vector.result_type(), vector)
        grouped.append(vector)
        for lane, access in enumerate(accesses):
            extract = mir.MVectorExtract(value=vector.result, lane=lane)
            extract.result = access.result
            extract.result.defining_op = extract
            grouped.append(extract)
    return grouped


def validate_register_memory(function):
    from metile.compiler.lowering.common import LoweringError

    vector_operations = (mir.MVectorLoad, mir.MVectorStore, mir.MVectorExtract)
    loaded = set()

    def fail():
        raise LoweringError(
            "register vector memory violates its typed ownership or contiguity contract"
        )

    def walk(operations):
        for operation in operations:
            yield operation
            yield from walk(getattr(operation, "body", ()))

    grouped = [
        operation for operation in walk(function.ops) if isinstance(operation, vector_operations)
    ]
    if not grouped:
        return
    geometry = function.threadgroup_size
    ownerships = function.value_layouts
    if (
        not isinstance(geometry, tuple)
        or len(geometry) != 3
        or any(type(dimension) is not int for dimension in geometry)
        or geometry[1:] != (1, 1)
        or not ownerships
        or any(not isinstance(record.layout, ThreadLayout) for record in ownerships)
    ):
        fail()
    threads = geometry[0]
    layout = ownerships[0].layout
    if any(
        record.layout.thread_count != threads
        or type(record.elements_per_thread) is not int
        or record.elements_per_thread != record.layout.elements_per_thread
        or record.layout.elements_per_thread != layout.elements_per_thread
        or record.layout.elements_per_thread < 4
        or record.layout.size != layout.size
        or record.shape != (layout.size,)
        for record in ownerships
    ):
        fail()
    for operation in grouped:
        if not any(operation is top_level for top_level in function.ops) or (
            function.schedule_plan is not None and function.schedule_plan.vector_width == 1
        ):
            fail()
        if isinstance(operation, mir.MVectorExtract):
            if (
                not isinstance(operation.value, mir.MValue)
                or not isinstance(operation.value.type, VectorType)
                or id(mir.resolve(operation.value)) not in loaded
                or type(operation.lane) is not int
                or not 0 <= operation.lane < 4
                or not isinstance(operation.result, mir.MValue)
                or operation.result.type != ScalarType(operation.value.type.dtype)
                or operation.result.defining_op is not operation
            ):
                fail()
            continue
        if (
            operation.dtype not in {"f16", "f32"}
            or not isinstance(operation.ptr, mir.MValue)
            or operation.ptr.type != PtrType(operation.dtype)
            or not isinstance(operation.indices, tuple)
            or len(operation.indices) != 4
            or any(
                not isinstance(index, mir.MValue)
                or index.type not in (ScalarType("i32"), ScalarType("u32"))
                for index in operation.indices
            )
            or not isinstance(operation.masks, tuple)
            or len(operation.masks) != 4
            or any(
                mask is not None and (not isinstance(mask, mir.MValue) or mask.type != BOOL)
                for mask in operation.masks
            )
            or not _consecutive(operation.indices, threads)
        ):
            fail()
        if isinstance(operation, mir.MVectorLoad):
            if (
                not isinstance(operation.result, mir.MValue)
                or operation.result.type != VectorType(operation.dtype, 4)
                or operation.result.defining_op is not operation
                or not isinstance(operation.others, tuple)
                or len(operation.others) != 4
                or any(
                    value is not None
                    and (
                        not isinstance(value, mir.MValue) or not isinstance(value.type, ScalarType)
                    )
                    for value in operation.others
                )
            ):
                fail()
            loaded.add(id(operation.result))
        elif (
            operation.result is not None
            or not isinstance(operation.values, tuple)
            or len(operation.values) != 4
            or any(
                not isinstance(value, mir.MValue) or not isinstance(value.type, ScalarType)
                for value in operation.values
            )
        ):
            fail()
