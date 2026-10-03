"""Scalarize checked register ownership into ordinary reusable Metal IR operations."""

from copy import copy
from dataclasses import fields

from metile.compiler.lowering.ownership import _map, _name
from metile.compiler.ownership import LayoutConversion, RegisterReduction
from metile.compiler.register_memory import group_register_memory
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, TileType


class RegisterLowering:
    def __init__(self, context):
        self.context = context
        self.layout = context.mfunc.value_layouts[0].layout
        self.values = {}
        context.block_size = self.layout.thread_count

    def lower(self, operation):
        if isinstance(operation, tir.Arange):
            return self._arange(operation)
        if isinstance(operation, tir.ConvertLayout):
            self.values[operation.result.name] = self.values[operation.value.name]
            self.context.mfunc.layout_conversions += (
                LayoutConversion(
                    operation.result.name, operation.value.type.layout, operation.layout, "identity"
                ),
            )
            return []
        if isinstance(operation, tir.Reduce):
            return self._reduce(operation)
        dependencies = {
            field.name: getattr(operation, field.name)
            for field in fields(operation)
            if field.name != "result" and isinstance(getattr(operation, field.name), tir.Value)
        }
        distributed = any(value.name in self.values for value in dependencies.values()) or (
            operation.result is not None and isinstance(operation.result.type, TileType)
        )
        if not distributed:
            return self.context._lower_scalar_op(operation)
        operations = []
        results = []
        for register in range(self.layout.elements_per_thread):
            changes = {}
            for field, value in dependencies.items():
                if value.name not in self.values:
                    continue
                name = _name(self.context, f"_metile_{value.name}_r{register}")
                self.context.value_map[name] = self.values[value.name][register]
                changes[field] = tir.Value(name, value.type, value.defining_op)
            if operation.result is not None:
                name = _name(self.context, f"_metile_{operation.result.name}_r{register}")
                changes["result"] = tir.Value(name, operation.result.type)
            scalar = copy(operation)
            for field, value in changes.items():
                setattr(scalar, field, value)
            operations.extend(self.context._lower_scalar_op(scalar) or [])
            if operation.result is not None:
                results.append(self.context.value_map[scalar.result.name])
        if operation.result is not None:
            self.values[operation.result.name] = tuple(results)
        schedule = self.context.func.constexprs.get("SCHEDULE")
        if (
            isinstance(operation, (tir.Load, tir.Store))
            and getattr(schedule, "vector_width", None) != 1
        ):
            operations = group_register_memory(
                operations, self.layout.thread_count, lambda stem: _name(self.context, stem)
            )
        return operations

    def _arange(self, operation):
        context = self.context
        thread = mir.MCast(value=context.lid_value, target_dtype="i32")
        thread.result = mir.MValue(_name(context, "_metile_register_thread"), I32, thread)
        mask = mir.MConstant(value=self.layout.thread_count - 1, dtype="i32")
        mask.result = mir.MValue(_name(context, "_metile_thread_mask"), I32, mask)
        bounded = mir.MBinOp(op="bitand", lhs=thread.result, rhs=mask.result)
        bounded.result = mir.MValue(_name(context, "_metile_bounded_thread"), I32, bounded)
        operations = [thread, mask, bounded]
        values = []
        for register in range(self.layout.elements_per_thread):
            offset = mir.MConstant(value=register * self.layout.thread_count, dtype="i32")
            offset.result = mir.MValue(_name(context, "_metile_register_offset"), I32, offset)
            packed = mir.MBinOp(op="add", lhs=bounded.result, rhs=offset.result)
            packed.result = mir.MValue(_name(context, "_metile_register_packed"), I32, packed)
            mapped = _map(context, packed.result, operation.layout, "_metile_register_index")
            operations.extend((offset, packed, mapped))
            value = mapped.result
            if operation.start is not None:
                addition = mir.MBinOp(op="add", lhs=value, rhs=context._resolve(operation.start))
                addition.result = mir.MValue(
                    _name(context, "_metile_register_origin"), I32, addition
                )
                operations.append(addition)
                value = addition.result
            values.append(value)
        self.values[operation.result.name] = tuple(values)
        return operations

    def _reduce(self, operation):
        context = self.context
        values = self.values[operation.operand.name]
        operations = []
        while len(values) > 1:
            partials = []
            for index in range(0, len(values), 2):
                addition = mir.MBinOp(op="add", lhs=values[index], rhs=values[index + 1])
                addition.result = mir.MValue(
                    _name(context, "_metile_register_sum"), values[index].type, addition
                )
                operations.append(addition)
                partials.append(addition.result)
            values = partials
        name = _name(context, "_metile_register_partial")
        context.value_map[name] = values[0]
        reduced = copy(operation)
        reduced.operand = tir.Value(name, values[0].type)
        lowered = context._lower_reduce(reduced)
        reduction = lowered[-1]
        reduction.replicate_partials = True
        scratch = _name(context, "_metile_register_reduction")
        reduction.shared_name = scratch
        for lowered_operation in lowered:
            if isinstance(lowered_operation, mir.MThreadgroupAlloc):
                lowered_operation.alloc_name = scratch
        context.mfunc.register_reductions += (
            RegisterReduction(
                operation.result.name,
                "sum",
                "f32",
                self.layout.elements_per_thread,
                self.layout.thread_count,
                reduction.shared_name if self.layout.thread_count > 32 else None,
            ),
        )
        return [*operations, *lowered]
