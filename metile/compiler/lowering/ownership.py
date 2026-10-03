"""Materialize scalar tile ownership without changing logical tensor indexing."""

from metile.compiler.ownership import LayoutConversion, conversion_map, conversion_mechanism
from metile.ir import metal_ir as mir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import I32, ScalarType


def _name(context, prefix):
    names = context._ownership_names
    while prefix in names:
        prefix += "_"
    names.add(prefix)
    return prefix


def _map(context, thread, layout, name):
    operation = mir.MThreadIndexMap(thread=thread, layout=layout)
    operation.result = mir.MValue(_name(context, name), I32, operation)
    return operation


def lower_owned_arange(context, operation):
    context.block_size = operation.size
    layout = operation.layout or ThreadLayout.identity(operation.size)
    index = _map(context, context.lid_value, layout, f"_metile_index_{operation.result.name}")
    operations = [index]
    value = index.result
    if operation.start is not None:
        addition = mir.MBinOp(op="add", lhs=value, rhs=context._resolve(operation.start))
        addition.result = mir.MValue(operation.result.name, I32, addition)
        operations.append(addition)
        value = addition.result
    context.value_map[operation.result.name] = value
    return operations


def lower_layout_conversion(context, operation):
    source = operation.value.type.layout or ThreadLayout.identity(operation.value.type.numel)
    destination = operation.layout
    mechanism = conversion_mechanism(source, destination)
    value = context._resolve(operation.value)
    name = operation.result.name
    scratch = None
    operations = []
    if mechanism == "identity":
        result = value
    else:
        owners = _map(
            context,
            context.lid_value,
            conversion_map(source, destination),
            f"_metile_owner_{name}",
        )
        operations.append(owners)
        dtype = operation.value.type.dtype
        if mechanism == "simd_shuffle":
            mask = mir.MConstant(value=31, dtype="i32")
            mask.result = mir.MValue(_name(context, f"_metile_mask_{name}"), I32, mask)
            lane = mir.MBinOp(op="bitand", lhs=owners.result, rhs=mask.result)
            lane.result = mir.MValue(_name(context, f"_metile_lane_{name}"), I32, lane)
            shuffle = mir.MSimdShuffle(value=value, lane=lane.result, dtype=dtype)
            shuffle.result = mir.MValue(name, ScalarType(dtype), shuffle)
            operations.extend((mask, lane, shuffle))
            result = shuffle.result
        else:
            scratch = context._ownership_scratch.get(dtype)
            if scratch is None:
                scratch = _name(context, f"_metile_exchange_{dtype}")
                context._ownership_scratch[dtype] = scratch
                operations.append(
                    mir.MThreadgroupAlloc(
                        alloc_name=scratch, elem_type=ScalarType(dtype).to_msl(), size=source.size
                    )
                )
            store = mir.MThreadgroupStore(array_name=scratch, index=context.lid_value, value=value)
            load = mir.MThreadgroupLoad(array_name=scratch, index=owners.result, dtype=dtype)
            load.result = mir.MValue(name, ScalarType(dtype), load)
            operations.extend((store, mir.MBarrier(), load, mir.MBarrier()))
            result = load.result
    context.value_map[name] = result
    context.mfunc.layout_conversions += (
        LayoutConversion(
            name,
            source,
            destination,
            mechanism,
            scratch,
            "threadgroup/mem_threadgroup" if scratch else None,
            "threadgroup/mem_threadgroup" if scratch else None,
        ),
    )
    return operations
