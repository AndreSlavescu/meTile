"""Composable SIMDgroup matrices inside ordinary scalar and loop programs."""

from __future__ import annotations

from dataclasses import replace

from metile.compiler.lowering.common import LoweringError
from metile.compiler.lowering.elementwise import _ElementwiseLoweringContext
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, MatrixFragmentType, PtrType, ScalarType, TileType


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def _constant(value):
    operation = value.defining_op
    return operation.value if isinstance(operation, tir.Constant) else None


def _shared_root(value):
    while isinstance(value.defining_op, tir.PtrOffset):
        value = value.defining_op.ptr
    return value if isinstance(value.defining_op, tir.SharedAlloc) else None


def _matrix(value):
    return isinstance(value.type, TileType) and len(value.type.shape) == 2


def _overlaps(access, effects):
    root, bounds = access
    return any(
        root == other_root
        and (
            bounds is None
            or other_bounds is None
            or (bounds[0] <= other_bounds[1] and other_bounds[0] <= bounds[1])
        )
        for other_root, other_bounds in effects
    )


class _Contract:
    def __init__(self, function, threads):
        self.function = function
        self.threads = threads
        self.facts = {}
        self.memory = {}
        self.scratch_roots = set()

    def device_fragment(self, operation):
        tensor, scratch = operation.tensor, operation.scratch
        if scratch is None:
            raise LoweringError("device inline matrix tiles require explicit shared scratch")
        pointer = tensor.ptr
        while isinstance(pointer.defining_op, tir.PtrOffset):
            pointer = pointer.defining_op.ptr
        if pointer.defining_op is not None or not any(
            parameter.name == pointer.name
            and parameter.type == pointer.type
            and isinstance(parameter.type, PtrType)
            and parameter.type.address_space == "device"
            for parameter in self.function.params
        ):
            raise LoweringError(
                "device inline matrices require a parameter-backed device allocation"
            )
        root = _shared_root(scratch.ptr)
        if (
            root is None
            or scratch.ptr is not root
            or scratch.address_space != "threadgroup"
            or scratch.access != "readwrite"
            or scratch.ptr.type.dtype != tensor.ptr.type.dtype
            or tuple(_constant(value) for value in scratch.shape) != (self.threads // 4, 8)
            or tuple(_constant(value) for value in scratch.strides) != (8, 1)
            or root.defining_op.size < self.threads * 2
        ):
            raise LoweringError(
                "device matrix scratch requires a matching-dtype, readwrite shared allocation "
                "with shape (BLOCK / 4, 8) and strides (8, 1)"
            )
        operands = (tensor.ptr, *tensor.shape, operation.row_offset, operation.col_offset)
        if any(self.scalar_facts(value)[0] < 1 for value in operands):
            raise LoweringError("device matrix bases, extents and origins must be SIMD-uniform")
        strides = tuple(_constant(value) for value in tensor.strides)
        if any(type(value) is not int or value <= 0 for value in strides) or 1 not in strides:
            raise LoweringError(
                "device inline matrices require positive row-major or column-major strides"
            )
        self.memory[id(operation)] = (tensor.ptr, 0, strides, strides[1] != 1)
        return None

    def device_roots(self, pointer):
        operation = pointer.defining_op
        if isinstance(operation, tir.PtrOffset):
            return self.device_roots(operation.ptr)
        if isinstance(operation, tir.SharedAlloc):
            return set()
        if isinstance(operation, tir.Select):
            return self.device_roots(operation.true_val) | self.device_roots(operation.false_val)
        parameters = {
            parameter.name
            for parameter in self.function.params
            if isinstance(parameter.type, PtrType) and parameter.type.address_space == "device"
        }
        if operation is None and pointer.name in parameters:
            return {pointer.name}
        return parameters

    def validate_device_effects(self, operations):
        fragment_roots, reads, writes = set(), set(), set()
        for operation in operations:
            if not isinstance(operation, (tir.Load, tir.Store, tir.TileLoad, tir.TileStore)):
                continue
            roots = self.device_roots(operation.ptr)
            if isinstance(operation, (tir.Load, tir.TileLoad)):
                reads.update(roots)
            else:
                writes.update(roots)
            if (
                isinstance(operation, (tir.TileLoad, tir.TileStore))
                and operation.tensor is not None
                and operation.tensor.address_space == "device"
            ):
                fragment_roots.update(roots)
        if fragment_roots & reads & writes:
            raise LoweringError(
                "device inline matrix inputs and outputs require disjoint allocation roots"
            )

    def scalar_facts(self, value):
        """Return uniformity (lane/SIMD/threadgroup) and conservative integer bounds."""
        if value.name in self.facts:
            return self.facts[value.name]
        self.facts[value.name] = (0, None)
        operation = value.defining_op
        uniform, bounds = 0, None
        if operation is None:
            uniform = 2 if any(value.name == item.name for item in self.function.params) else 0
        elif isinstance(operation, (tir.Constant, tir.SharedAlloc)):
            uniform = 2
            if isinstance(operation, tir.Constant) and type(operation.value) is int:
                bounds = (operation.value, operation.value)
        elif isinstance(operation, tir.ProgramId):
            uniform, bounds = 2, (0, 2**31 - 1)
        elif isinstance(operation, tir.ThreadId):
            bounds = (0, self.threads - 1)
        elif isinstance(operation, tir.SimdLaneId):
            bounds = (0, 31)
        elif isinstance(operation, tir.Cast):
            uniform, bounds = self.scalar_facts(operation.value)
            if operation.value.type.dtype not in {"i32", "u32"} or value.type.dtype not in {
                "i32",
                "u32",
            }:
                bounds = None
        elif isinstance(operation, tir.Bitcast):
            uniform = self.scalar_facts(operation.value)[0]
        elif isinstance(operation, tir.PtrOffset):
            uniform = min(
                self.scalar_facts(operation.ptr)[0], self.scalar_facts(operation.offsets)[0]
            )
        elif isinstance(operation, tir.Load):
            operands = [operation.ptr, operation.offsets]
            if operation.mask is not None:
                operands.append(operation.mask)
            if operation.other is not None:
                operands.append(operation.other)
            uniform = min(self.scalar_facts(operand)[0] for operand in operands)
        elif isinstance(operation, tir.Unary):
            uniform = self.scalar_facts(operation.operand)[0]
            if operation.op in {"simd_sum", "simd_max", "simd_min"}:
                uniform = max(uniform, 1)
        elif isinstance(operation, tir.SimdBroadcast):
            uniform = 1 if self.scalar_facts(operation.lane)[0] else 0
        elif isinstance(operation, tir.Select):
            operands = (operation.condition, operation.true_val, operation.false_val)
            uniform = min(self.scalar_facts(operand)[0] for operand in operands)
            choices = [self.scalar_facts(operand)[1] for operand in operands[1:]]
            if all(choice is not None for choice in choices):
                bounds = (
                    min(choice[0] for choice in choices),
                    max(choice[1] for choice in choices),
                )
        elif isinstance(operation, tir.Fma):
            uniform = min(
                self.scalar_facts(operand)[0]
                for operand in (operation.left, operation.right, operation.addend)
            )
        elif isinstance(operation, (tir.BinOp, tir.Compare)):
            left_uniform, left = self.scalar_facts(operation.lhs)
            right_uniform, right = self.scalar_facts(operation.rhs)
            uniform = min(left_uniform, right_uniform)
            if isinstance(operation, tir.Compare):
                bounds = (0, 1)
            else:
                divisor = _constant(operation.rhs)
                if (
                    isinstance(operation.lhs.defining_op, tir.ThreadId)
                    and type(divisor) is int
                    and (
                        (operation.op == "div" and divisor >= 32 and divisor % 32 == 0)
                        or (operation.op == "shr" and divisor >= 5)
                    )
                ):
                    uniform = 1
                if left is not None and right is not None:
                    if operation.op == "add":
                        bounds = (left[0] + right[0], left[1] + right[1])
                    elif operation.op == "sub":
                        bounds = (left[0] - right[1], left[1] - right[0])
                    elif operation.op == "mul":
                        products = [first * second for first in left for second in right]
                        bounds = (min(products), max(products))
                    elif right[0] == right[1] and right[0] > 0 and left[0] >= 0:
                        if operation.op == "div":
                            bounds = (left[0] // right[0], left[1] // right[0])
                        elif operation.op == "mod":
                            bounds = (0, right[0] - 1)
                    if operation.op == "min":
                        bounds = (min(left[0], right[0]), min(left[1], right[1]))
                    elif operation.op == "max":
                        bounds = (max(left[0], right[0]), max(left[1], right[1]))
        if bounds is not None and value.type.dtype in {"i32", "u32"}:
            minimum, maximum = (
                (0, 2**32 - 1) if value.type.dtype == "u32" else (-(2**31), 2**31 - 1)
            )
            if bounds[0] < minimum or bounds[1] > maximum:
                bounds = None
        self.facts[value.name] = (uniform, bounds)
        return uniform, bounds

    def scalar_memory(self, operation):
        pointer = operation.ptr
        root = _shared_root(pointer)
        if root is None:
            if isinstance(pointer.type, PtrType) and pointer.type.address_space == "threadgroup":
                raise LoweringError(
                    "inline scalar shared memory must reference a declared shared allocation "
                    "through pointer offsets only"
                )
            return None
        offsets = []
        compatible = pointer.type.dtype == root.type.dtype
        while isinstance(pointer.defining_op, tir.PtrOffset):
            offsets.append(pointer.defining_op.offsets)
            pointer = pointer.defining_op.ptr
            compatible &= pointer.type.dtype == root.type.dtype
        if not offsets:
            offsets.append(operation.offsets)
        bounds = (0, 0)
        dtype = offsets[-1].type.dtype
        minimum, maximum = (0, 2**32 - 1) if dtype == "u32" else (-(2**31), 2**31 - 1)
        for offset in reversed(offsets):
            part = self.scalar_facts(offset)[1]
            if (
                not compatible
                or dtype not in {"i32", "u32"}
                or offset.type.dtype not in {"i32", "u32"}
                or part is None
                or part[0] < minimum
                or part[1] > maximum
            ):
                return root.name, None
            bounds = (bounds[0] + part[0], bounds[1] + part[1])
            if bounds[0] < minimum or bounds[1] > maximum:
                return root.name, None
        if bounds[0] < 0 or bounds[1] >= root.defining_op.size:
            bounds = None
        return root.name, bounds

    def fragment_memory(self, operation):
        tensor = operation.tensor
        if tensor is None or operation.tile_shape != (8, 8):
            raise LoweringError("inline matrices require tensor descriptors with 8x8 blocks")
        if len(tensor.shape) != 2:
            raise LoweringError("inline matrix tiles require rank-two tensors")
        if tensor.ptr.type.dtype not in {"f16", "f32"}:
            raise LoweringError("inline matrix memory requires f16 or f32 storage")
        if tensor.address_space == "device":
            return self.device_fragment(operation)
        if operation.scratch is not None:
            raise LoweringError("scratch is only valid for device inline matrix tiles")
        pointer, base = tensor.ptr, 0
        while isinstance(pointer.defining_op, tir.PtrOffset):
            offset = _constant(pointer.defining_op.offsets)
            if type(offset) is not int:
                raise LoweringError(
                    "inline matrix bases require constant shared allocation offsets"
                )
            base += offset
            pointer = pointer.defining_op.ptr
        if not isinstance(pointer.defining_op, tir.SharedAlloc):
            raise LoweringError("inline matrix memory must reference a declared shared allocation")
        if pointer.name in self.scratch_roots:
            raise LoweringError("device matrix scratch is exclusive to masked tile accesses")
        shape = tuple(_constant(value) for value in tensor.shape)
        strides = tuple(_constant(value) for value in tensor.strides)
        if any(type(value) is not int or value <= 0 for value in (*shape, *strides)):
            raise LoweringError("inline matrix extents and strides must be positive constants")
        row_major = strides[1] == 1 and strides[0] >= shape[1]
        column_major = strides[0] == 1 and strides[1] >= shape[0]
        if not (row_major or column_major):
            raise LoweringError("inline matrices require row-major or transposed row-major strides")
        span = base + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
        if base < 0 or span >= pointer.defining_op.size:
            raise LoweringError("inline matrix tensor exceeds its shared allocation")
        minimum, maximum = base, base
        for origin, extent, stride in zip(
            (operation.row_offset, operation.col_offset), shape, strides
        ):
            uniform, bounds = self.scalar_facts(origin)
            if uniform < 1:
                raise LoweringError("inline matrix origins must be SIMD-uniform")
            if bounds is None or bounds[0] < 0 or bounds[1] + 8 > extent:
                raise LoweringError(
                    "inline matrix accesses require provably complete in-bounds tiles"
                )
            minimum += bounds[0] * stride
            maximum += (bounds[1] + 7) * stride
        binding = (pointer, base, strides, not row_major)
        self.memory[id(operation)] = binding
        return pointer.name, (minimum, maximum)

    def validate(self):
        operations = tuple(_walk(self.function.ops))
        self.validate_device_effects(operations)
        self.scratch_roots = {
            root.name
            for operation in operations
            if isinstance(operation, (tir.TileLoad, tir.TileStore))
            and operation.scratch is not None
            and (root := _shared_root(operation.scratch.ptr)) is not None
        }
        allocations = [
            operation for operation in operations if isinstance(operation, tir.SharedAlloc)
        ]
        if any(
            not any(operation is root for root in self.function.ops) for operation in allocations
        ):
            raise LoweringError("inline matrix shared allocations must be defined at the top level")
        if (
            sum(
                operation.size * (2 if operation.dtype == "f16" else 4) for operation in allocations
            )
            > 32768
        ):
            raise LoweringError("inline matrix program exceeds 32 KiB of shared memory")
        if not any(
            isinstance(operation, (tir.TileLoad, tir.TileStore, tir.Dot))
            for operation in operations
        ):
            raise LoweringError("simdgroup_inline requires matrix fragment operations")
        scalar_writes, matrix_reads, matrix_writes = set(), set(), set()

        def walk(body, control_uniform=2):
            for operation in body:
                if isinstance(operation, tir.ForRange):
                    if type(operation.step) is not int or operation.step < 1:
                        raise LoweringError("inline matrix loops require a positive constant step")
                    start_uniform, start = self.scalar_facts(operation.start)
                    end_uniform, end = self.scalar_facts(operation.end)
                    uniform = min(control_uniform, start_uniform, end_uniform)
                    bounds = None
                    if start is not None and end is not None and operation.step > 0:
                        last = (
                            start[0]
                            + max(0, (end[1] - 1 - start[0]) // operation.step) * operation.step
                            if start[0] == start[1]
                            else end[1] - 1
                        )
                        bounds = (start[0], max(start[1], last))
                    self.facts[operation.iv.name] = (uniform, bounds)
                    incoming = (scalar_writes.copy(), matrix_reads.copy(), matrix_writes.copy())
                    walk(operation.body, uniform)
                    walk(operation.body, uniform)
                    if start is None or end is None or start[1] >= end[0]:
                        scalar_writes.update(incoming[0])
                        matrix_reads.update(incoming[1])
                        matrix_writes.update(incoming[2])
                    continue
                if isinstance(
                    operation,
                    (tir.PersistentRange, tir.SimdgroupRole, tir.Arange, tir.ConvertLayout),
                ):
                    raise LoweringError(
                        "inline matrices currently require scalar indices and explicit tile_range loops"
                    )
                if isinstance(operation, tir.Barrier):
                    if control_uniform < 2:
                        raise LoweringError(
                            "inline matrix barriers require threadgroup-uniform control flow"
                        )
                    scalar_writes.clear()
                    matrix_reads.clear()
                    matrix_writes.clear()
                    continue
                fragment = (
                    isinstance(operation, (tir.TileLoad, tir.TileStore, tir.Dot))
                    or (operation.result is not None and _matrix(operation.result))
                    or (isinstance(operation, tir.AssignLoopState) and _matrix(operation.state))
                )
                if fragment and control_uniform < 1:
                    raise LoweringError(
                        "inline matrix operations require SIMD-uniform control flow"
                    )
                if isinstance(operation, (tir.TileLoad, tir.TileStore)):
                    access = self.fragment_memory(operation)
                    if access is None:
                        continue
                    if isinstance(operation, tir.TileLoad):
                        if _overlaps(access, scalar_writes) or _overlaps(access, matrix_writes):
                            raise LoweringError(
                                "shared writes require a barrier before matrix loads"
                            )
                        matrix_reads.add(access)
                    else:
                        if _overlaps(access, matrix_reads):
                            raise LoweringError(
                                "matrix reads require a barrier before shared overwrite"
                            )
                        if _overlaps(access, scalar_writes):
                            raise LoweringError(
                                "scalar shared writes require a barrier before matrix stores"
                            )
                        matrix_writes.add(access)
                elif isinstance(operation, (tir.Load, tir.Store)):
                    access = self.scalar_memory(operation)
                    if access is None:
                        continue
                    if access[0] in self.scratch_roots:
                        raise LoweringError(
                            "device matrix scratch is exclusive to masked tile accesses"
                        )
                    if isinstance(operation, tir.Load) and _overlaps(access, matrix_writes):
                        raise LoweringError(
                            "matrix stores require a barrier before scalar shared loads"
                        )
                    if isinstance(operation, tir.Store):
                        if _overlaps(access, matrix_reads):
                            raise LoweringError(
                                "matrix reads require a barrier before shared overwrite"
                            )
                        if _overlaps(access, matrix_writes):
                            raise LoweringError(
                                "matrix stores require a barrier before scalar shared overwrite"
                            )
                        scalar_writes.add(access)

        walk(self.function.ops)
        return self


def inline_matrix_plan(function, schedule, count):
    from metile.compiler.planning import SchedulePlan

    block = function.constexprs.get("BLOCK", 32 * (count or 4))
    if type(block) is not int or not 32 <= block <= 1024 or block % 32:
        raise LoweringError("inline matrix BLOCK must be a multiple of 32 between 32 and 1024")
    if count is not None and block != count * 32:
        raise LoweringError("inline matrix BLOCK conflicts with the requested SIMDgroup count")
    if schedule.staging not in {"auto", "threadgroup"} or schedule.double_buffer:
        raise LoweringError(
            "inline matrix staging and buffering are explicitly declared by the kernel"
        )
    if schedule.vector_width not in {None, 1}:
        raise LoweringError("inline matrices do not support forced vectorized memory operations")
    if any(name in function.constexprs for name in ("WM", "WN")) or any(
        function.constexprs.get(name, False) for name in ("NAX_FRAGMENTS", "COOPERATIVE")
    ):
        raise LoweringError(
            "inline matrix SIMDgroup assignment is explicitly declared by the kernel"
        )
    _Contract(function, block).validate()
    return SchedulePlan(
        backend="simdgroup_inline",
        threadgroup_size=(block, 1, 1),
        simdgroup_grid=None,
        tile_shape=(8, 8),
        staging="threadgroup",
        vector_width=1,
        double_buffer=False,
        reasons=(
            "Composable 8x8 matrix fragments use declared tensors and explicit shared scratch.",
        ),
        cooperative_layout="opaque 8x8 SIMDgroup register matrices",
    )


class InlineMatrixLowering(_ElementwiseLoweringContext):
    def __init__(self, function):
        super().__init__(function)
        count = function.constexprs.get("SCHEDULE").num_simdgroups or function.constexprs.get(
            "NUM_SG", 4
        )
        self.threads = function.constexprs.get("BLOCK", count * 32)
        self.contract = _Contract(function, self.threads).validate()

    def lower(self):
        function = super().lower()
        function.kernel_type = "simdgroup_inline"
        function.threadgroup_size = (self.threads, 1, 1)
        return function

    def _result(self, operation, lowered):
        lowered.result = mir.MValue(operation.result.name, lowered.result_type(), lowered)
        self.value_map[operation.result.name] = lowered.result
        return [lowered]

    def _fragment(self, value):
        if (
            not _matrix(value)
            or value.type.shape != (8, 8)
            or value.type.dtype not in {"f16", "f32"}
        ):
            raise LoweringError("inline matrix values must be f16/f32 8x8 tiles")
        lowered = self._resolve(value)
        if not isinstance(lowered.type, MatrixFragmentType):
            raise LoweringError("inline matrix value was not lowered as a register fragment")
        return lowered

    def _full_tile_loop(self, operation):
        if (
            _constant(operation.start) != 0
            or operation.iv.type.dtype != "i32"
            or operation.end.type.dtype != "i32"
            or not 8 <= operation.step <= 2**31 - 1
            or any(getattr(nested, "body", None) is not None for nested in operation.body)
        ):
            return False

        def relative(value):
            if value.name == operation.iv.name:
                return 0
            definition = value.defining_op
            if value.type.dtype != "i32" or not isinstance(definition, tir.BinOp):
                return None
            if definition.op != "add" or any(
                operand.type.dtype != "i32" for operand in (definition.lhs, definition.rhs)
            ):
                return None
            for dynamic, constant in (
                (definition.lhs, definition.rhs),
                (definition.rhs, definition.lhs),
            ):
                offset = _constant(constant)
                if (
                    type(offset) is int
                    and offset >= 0
                    and (origin := relative(dynamic)) is not None
                    and origin + offset <= operation.step - 8
                ):
                    return origin + offset
            return None

        found = False
        for nested in operation.body:
            if not isinstance(nested, (tir.TileLoad, tir.TileStore)) or nested.scratch is None:
                continue
            found = True
            for origin, extent in zip(
                (nested.row_offset, nested.col_offset), nested.tensor.shape, strict=True
            ):
                constant_extent = _constant(extent)
                bounds = self.contract.scalar_facts(origin)[1]
                if (
                    type(constant_extent) is int
                    and bounds is not None
                    and 0 <= bounds[0] <= bounds[1] <= constant_extent - 8
                ):
                    continue
                offset = relative(origin)
                if (
                    extent.name != operation.end.name
                    or offset is None
                    or not 0 <= offset <= operation.step - 8
                ):
                    return False
        return found

    def _lower_op(self, operation):
        if isinstance(operation, tir.ForRange):
            self.value_map[operation.iv.name] = mir.MValue(operation.iv.name, I32)
            body = []
            for nested in operation.body:
                body.extend(self._lower_op(nested) or ())
            loop = mir.MForLoop(
                iv_name=operation.iv.name,
                start=self._resolve(operation.start),
                end=self._resolve(operation.end),
                step=operation.step,
                body=body,
            )
            if not self._full_tile_loop(operation):
                return [loop]
            setup = []

            def append(name, nested):
                nested.result = mir.MValue(f"{operation.iv.name}_{name}", I32, nested)
                setup.append(nested)
                return nested.result

            step = append("step", mir.MConstant(value=operation.step, dtype="i32"))
            nonnegative = append("positive_end", mir.MBinOp(op="max", lhs=loop.end, rhs=loop.start))
            count = append("complete_steps", mir.MBinOp(op="div", lhs=nonnegative, rhs=step))
            complete_end = append("complete_end", mir.MBinOp(op="mul", lhs=count, rhs=step))
            full_body = []
            for nested in operation.body:
                full_body.extend(self._lower_op(nested) or ())
            for nested in full_body:
                if (
                    isinstance(nested, (mir.MFragmentLoad, mir.MFragmentStore))
                    and nested.shape is not None
                ):
                    nested.full_tile = True
            return [
                *setup,
                replace(loop, end=complete_end, body=full_body),
                replace(loop, start=complete_end),
            ]
        if isinstance(operation, (tir.TileLoad, tir.TileStore)):
            pointer, base, strides, transpose = self.contract.memory[id(operation)]
            resolved_pointer = self.value_map.get(pointer.name)
            if isinstance(resolved_pointer, tuple):
                resolved_pointer, base = resolved_pointer
            else:
                resolved_pointer = self._resolve(pointer)
            arguments = dict(
                ptr=resolved_pointer,
                row=self._resolve(operation.row_offset),
                column=self._resolve(operation.col_offset),
                row_stride=strides[0],
                column_stride=strides[1],
                base_offset=base,
                transpose=transpose,
            )
            if operation.scratch is not None:
                arguments.update(
                    shape=tuple(self._resolve(value) for value in operation.tensor.shape),
                    scratch=self._resolve(operation.scratch.ptr),
                )
            if isinstance(operation, tir.TileLoad):
                return self._result(
                    operation, mir.MFragmentLoad(**arguments, dtype=pointer.type.dtype)
                )
            value = self._fragment(operation.value)
            operations = []
            if value.type.dtype != pointer.type.dtype:
                cast = mir.MFragmentElementwise(
                    operation="cast", operands=(value,), dtype=pointer.type.dtype
                )
                cast.result = mir.MValue(
                    f"{value.name}_store_{len(self.value_map)}", cast.result_type(), cast
                )
                operations.append(cast)
                value = cast.result
            return [*operations, mir.MFragmentStore(**arguments, value=value)]
        if isinstance(operation, tir.Dot):
            left, right, accumulator = (
                self._fragment(value) for value in (operation.a, operation.b, operation.acc)
            )
            if left.type != right.type or accumulator.type.dtype != "f32":
                raise LoweringError(
                    "inline dot requires matching operand dtypes and an FP32 accumulator"
                )
            return self._result(
                operation, mir.MFragmentDot(left=left, right=right, accumulator=accumulator)
            )
        if isinstance(operation, tir.AssignLoopState) and _matrix(operation.state):
            state, value = self._fragment(operation.state), self._fragment(operation.value)
            if state.type != value.type:
                raise LoweringError("matrix loop state updates must preserve the fragment dtype")
            return [mir.MFragmentStateAssign(state_name=state.name, value=value)]
        if operation.result is not None and _matrix(operation.result):
            if operation.result.type.shape != (8, 8) or operation.result.type.dtype not in {
                "f16",
                "f32",
            }:
                raise LoweringError("inline matrix values must be f16/f32 8x8 tiles")
            if isinstance(operation, tir.Zeros):
                return self._result(operation, mir.MFragmentInit(dtype=operation.dtype))
            if isinstance(operation, tir.LoopState):
                value = self._fragment(operation.value)
                self.value_map[operation.result.name] = mir.MValue(
                    operation.result.name, value.type
                )
                return [mir.MFragmentStateInit(state_name=operation.result.name, value=value)]
            if isinstance(operation, tir.ReadLoopState):
                return self._result(
                    operation, mir.MFragmentStateRead(state=self._fragment(operation.state))
                )
            if isinstance(operation, (tir.BinOp, tir.Fma)):
                name = "fma" if isinstance(operation, tir.Fma) else operation.op
                if name not in {"add", "sub", "mul", "div", "min", "max", "fma"}:
                    raise LoweringError("unsupported inline matrix binary operation")
                arguments = (
                    (operation.left, operation.right, operation.addend)
                    if isinstance(operation, tir.Fma)
                    else (operation.lhs, operation.rhs)
                )
                for operand in arguments:
                    if _matrix(operand):
                        self._fragment(operand)
                        if operand.type.dtype != operation.result.type.dtype:
                            raise LoweringError("matrix pointwise operands require matching dtypes")
                    elif (
                        not isinstance(operand.type, ScalarType)
                        or self.contract.scalar_facts(operand)[0] < 1
                    ):
                        raise LoweringError("matrix scalar operands must be SIMD-uniform")
                operands = tuple(self._resolve(operand) for operand in arguments)
                return self._result(
                    operation,
                    mir.MFragmentElementwise(
                        operation=name, operands=operands, dtype=operation.result.type.dtype
                    ),
                )
            if isinstance(operation, (tir.Cast, tir.Unary)):
                name = "cast" if isinstance(operation, tir.Cast) else operation.op
                if name not in {
                    "cast",
                    "exp",
                    "exp2",
                    "fast_exp",
                    "fast_exp2",
                    "fast_cos",
                    "fast_sin",
                    "log",
                    "sqrt",
                    "rsqrt",
                    "abs",
                    "neg",
                    "tanh",
                }:
                    raise LoweringError("unsupported inline matrix unary operation")
                value = operation.value if isinstance(operation, tir.Cast) else operation.operand
                fragment = self._fragment(value)
                if isinstance(operation, tir.Cast) and fragment.type.dtype == operation.dtype:
                    self.value_map[operation.result.name] = fragment
                    return []
                return self._result(
                    operation,
                    mir.MFragmentElementwise(
                        operation=name,
                        operands=(fragment,),
                        dtype=operation.result.type.dtype,
                    ),
                )
            raise LoweringError(f"unsupported inline matrix operation: {type(operation).__name__}")
        if isinstance(operation, tir.Reduce) and _matrix(operation.operand):
            raise LoweringError("matrix reductions require an explicit shared-memory scalar bridge")
        return super()._lower_op(operation)
