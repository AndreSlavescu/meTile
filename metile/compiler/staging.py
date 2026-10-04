"""Verified software staging with explicit publication and buffer-reuse dependencies."""

import re
from dataclasses import dataclass, fields, replace

from metile.ir import metal_ir as mir


class StagingError(ValueError):
    """A software pipeline cannot preserve the staged loop's dependencies."""


@dataclass(frozen=True)
class StagingBuffer:
    logical_name: str
    slots: tuple[str, str]
    element_type: str
    elements: int
    current_pointer: str
    next_pointer: str

    @property
    def bytes_per_slot(self) -> int:
        return self.elements * _ELEMENT_BYTES[self.element_type]


@dataclass(frozen=True)
class StageAccess:
    buffer: str
    slot: str


@dataclass(frozen=True)
class PipelinePhase:
    name: str
    region: str
    kind: str
    reads: tuple[StageAccess, ...] = ()
    writes: tuple[StageAccess, ...] = ()
    publishes: tuple[StageAccess, ...] = ()
    recycles: tuple[StageAccess, ...] = ()
    rotates: tuple[str, ...] = ()
    barrier_scope: str | None = None
    barrier_flags: str | None = None


@dataclass(frozen=True)
class SoftwarePipeline:
    buffers: tuple[StagingBuffer, ...]
    phases: tuple[PipelinePhase, ...]
    stages: int = 2
    mechanism: str = "software"


_ELEMENT_BYTES = {"float": 4, "half": 2, "int": 4, "uint": 4, "uchar": 1}


def _phases(buffers: tuple[StagingBuffer, ...]) -> tuple[PipelinePhase, ...]:
    current = tuple(StageAccess(buffer.logical_name, "current") for buffer in buffers)
    upcoming = tuple(StageAccess(buffer.logical_name, "next") for buffer in buffers)
    return (
        PipelinePhase("load_first", "prologue", "copy", writes=current),
        PipelinePhase(
            "publish_first",
            "prologue",
            "barrier",
            publishes=current,
            barrier_scope="threadgroup",
            barrier_flags="mem_threadgroup",
        ),
        PipelinePhase("load_next", "steady", "copy", writes=upcoming),
        PipelinePhase("consume_current", "steady", "compute", reads=current),
        PipelinePhase(
            "publish_and_recycle",
            "steady",
            "barrier",
            publishes=upcoming,
            recycles=current,
            barrier_scope="threadgroup",
            barrier_flags="mem_threadgroup",
        ),
        PipelinePhase(
            "rotate_slots",
            "steady",
            "rotate",
            rotates=tuple(buffer.logical_name for buffer in buffers),
        ),
        PipelinePhase("consume_last", "drain", "compute", reads=current),
        PipelinePhase(
            "release_last",
            "drain",
            "barrier",
            recycles=current,
            barrier_scope="threadgroup",
            barrier_flags="mem_threadgroup",
        ),
    )


def validate_pipeline(pipeline: SoftwarePipeline):
    """Check both the emitted phase vocabulary and cross-iteration slot dependencies."""
    if pipeline.mechanism != "software" or pipeline.stages != 2:
        raise StagingError("only verified two-stage software pipelines are supported")
    if len(pipeline.buffers) != 2:
        raise StagingError("software matrix staging requires two independent operand buffers")
    names = [buffer.logical_name for buffer in pipeline.buffers]
    if len(set(names)) != len(names):
        raise StagingError("staging buffers must have distinct logical identities")
    identifiers = []
    for buffer in pipeline.buffers:
        if (
            len(buffer.slots) != 2
            or buffer.element_type not in {"half", "float"}
            or type(buffer.elements) is not int
            or buffer.elements <= 0
        ):
            raise StagingError("staging buffers require two typed positive-size slots")
        identifiers.extend((*buffer.slots, buffer.current_pointer, buffer.next_pointer))
    if len(set(identifiers)) != len(identifiers) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name in names for name in identifiers
    ):
        raise StagingError("staging slot and pointer identities must be distinct valid names")
    if pipeline.phases != _phases(pipeline.buffers):
        raise StagingError(
            "pipeline phases do not match the supported publication/recycle contract"
        )
    state = {StageAccess(name, slot): "free" for name in names for slot in ("current", "next")}
    regions = ("prologue", "steady", "steady", "drain")
    for region in regions:
        for phase in pipeline.phases:
            if phase.region != region:
                continue
            for access in phase.writes:
                if state[access] != "free":
                    raise StagingError("staging writes require a released buffer slot")
                state[access] = "written"
            for access in phase.reads:
                if state[access] != "published":
                    raise StagingError("staging reads require a published buffer slot")
                state[access] = "consumed"
            for access in phase.publishes:
                if state[access] != "written":
                    raise StagingError("publication must follow a staged write")
                state[access] = "published"
            for access in phase.recycles:
                if state[access] != "consumed":
                    raise StagingError("slot reuse must follow consumption and a release barrier")
                state[access] = "free"
            for name in phase.rotates:
                current, upcoming = StageAccess(name, "current"), StageAccess(name, "next")
                state[current], state[upcoming] = state[upcoming], state[current]
    if any(status != "free" for status in state.values()):
        raise StagingError("pipeline drain must release every buffer slot")


def _walk(operations):
    for operation in operations:
        yield operation
        yield from _walk(getattr(operation, "body", ()))


def _uniform(value, function: mir.MFunction) -> bool:
    if isinstance(value, int):
        return not isinstance(value, bool)
    if not isinstance(value, mir.MValue):
        return False
    value = mir.resolve(value)
    operation = value.defining_op
    if operation is None:
        binding = function.dimension_bindings.get(value.name)
        if isinstance(binding, int):
            return True
        if isinstance(binding, mir.MValue):
            return any(
                parameter.name == binding.name and parameter.is_scalar
                for parameter in function.params
            )
        return any(
            parameter.name == value.name and parameter.is_scalar for parameter in function.params
        )
    if isinstance(operation, (mir.MConstant, mir.ThreadgroupPositionInGrid)):
        return True
    if isinstance(operation, mir.MBinOp):
        return _uniform(operation.lhs, function) and _uniform(operation.rhs, function)
    if isinstance(operation, mir.MCast):
        return _uniform(operation.value, function)
    return False


def _thread_value(value, thread: int) -> int:
    if not isinstance(value, mir.MValue):
        raise StagingError("staged thread indices require an SSA value")
    value = mir.resolve(value)
    operation = value.defining_op
    if isinstance(operation, mir.MConstant) and type(operation.value) is int:
        return operation.value
    if isinstance(operation, mir.MSimdgroupId):
        return thread // 32
    if isinstance(operation, mir.MThreadInSimdgroup):
        return thread % 32
    if isinstance(operation, mir.ThreadPositionInThreadgroup) and operation.axis == 0:
        return thread
    if isinstance(operation, mir.MCast):
        return _thread_value(operation.value, thread)
    if isinstance(operation, mir.MBinOp):
        left, right = _thread_value(operation.lhs, thread), _thread_value(operation.rhs, thread)
        if operation.op == "add":
            return left + right
        if operation.op == "sub":
            return left - right
        if operation.op == "mul":
            return left * right
        if operation.op == "div" and right > 0 and left >= 0:
            return left // right
        if operation.op == "mod" and right > 0 and left >= 0:
            return left % right
    raise StagingError("staged thread ownership must have a statically provable index mapping")


def _same_value(left, right) -> bool:
    if isinstance(left, mir.MValue) and isinstance(right, mir.MValue):
        left, right = mir.resolve(left), mir.resolve(right)
        return left is right or (
            left.defining_op is None and right.defining_op is None and left.name == right.name
        )
    return isinstance(left, int) and isinstance(right, int) and left == right


def _depends_on(value: mir.MValue | None, names: set[str], seen=None) -> bool:
    if value is None:
        return False
    seen = set() if seen is None else seen
    value = mir.resolve(value)
    if id(value) in seen:
        return False
    seen.add(id(value))
    if value.name in names:
        return True
    operation = value.defining_op
    if operation is None:
        return False
    return any(
        _depends_on(operand, names, seen)
        for field in fields(operation)
        if field.name != "result"
        and isinstance(operand := getattr(operation, field.name), mir.MValue)
    )


def _barrier(operation):
    return (
        isinstance(operation, mir.MBarrier)
        and operation.kind == "threadgroup"
        and operation.flags == "mem_threadgroup"
        and operation.condition is None
    )


def _loop_parts(loop: mir.MForLoop, function: mir.MFunction):
    if (
        type(loop.start) is not int
        or loop.start != 0
        or type(loop.step) is not int
        or loop.step <= 0
        or loop.step % 8
        or not _uniform(loop.end, function)
        or loop.index_alias is not None
        or loop.index_expression is not None
        or getattr(loop, "_aligned", False)
        or getattr(loop, "_is_tail", False)
        or getattr(loop, "_specialized_db", False)
    ):
        raise StagingError("staging requires a uniform zero-based unsplit matrix reduction loop")
    if len(loop.body) != 5:
        raise StagingError("staging cannot discard additional reduction-loop operations")
    first, second, publish, compute, release = loop.body
    if not all(isinstance(operation, mir.MCooperativeLoad) for operation in (first, second)):
        raise StagingError("staging requires exactly two leading cooperative loads")
    if not _barrier(publish) or not _barrier(release):
        raise StagingError(
            "staging requires unconditional threadgroup publish and release barriers"
        )
    if (
        not isinstance(compute, mir.MForLoop)
        or type(compute.start) is not int
        or compute.start != 0
        or compute.end != loop.step
        or compute.step != 8
        or not getattr(compute, "_unroll", False)
        or compute.staging is not None
        or compute.index_alias is not None
        or compute.index_expression is not None
    ):
        raise StagingError("staging requires one complete unrolled SIMD-group reduction")
    loads = (first, second)
    if first.tg_array == second.tg_array:
        raise StagingError("staged operands cannot share an allocation")
    if first.elem_type != second.elem_type or first.elem_type not in {"half", "float"}:
        raise StagingError("staged matrix operands must use matching supported element types")
    load_by_name = {load.tg_array: load for load in loads}
    roles = set()
    for load in loads:
        if (load.row_offset is None) == (load.col_offset is None):
            raise StagingError("staged loads must advance exactly one reduction dimension")
        is_right = load.row_offset is None
        roles.add(is_right)
        reduction_bound = load.row_bound if is_right else load.col_bound
        reduction_extent = load.tile_rows if is_right else load.tile_cols
        if reduction_extent != loop.step or not _same_value(reduction_bound, loop.end):
            raise StagingError("staged load bounds must match the reduction-loop extent")
        if (
            not load.bounds_check
            or load.row_bound is None
            or load.col_bound is None
            or load.kb_expr is not None
            or load.vec_size != 1
            or load.tile_rows <= 0
            or load.tile_cols <= 0
            or load.tile_rows % 8
            or load.tile_cols % 8
            or load.dst_stride < load.tile_cols
            or load.tg_size != function.threadgroup_size[0]
            or function.threadgroup_size[1:] != (1, 1)
            or load.tg_size <= 0
            or load.tg_size % 32
        ):
            raise StagingError("staging requires complete bounds-checked whole-threadgroup loads")
        if load.load_layout is not None:
            layout = load.load_layout
            if (
                layout.tile.rows,
                layout.tile.cols,
                layout.tile.smem_stride,
                layout.num_threads,
            ) != (load.tile_rows, load.tile_cols, load.dst_stride, load.tg_size):
                raise StagingError("staged load geometry conflicts with its memory layout")
        if (
            load.swizzle_bits < 0
            or load.swizzle_shift < 0
            or (
                load.swizzle_bits
                and (
                    load.dst_stride != load.tile_cols
                    or load.tile_cols & (load.tile_cols - 1)
                    or load.swizzle_shift != load.tile_cols.bit_length() - 1
                    or (1 << load.swizzle_bits) > load.tile_cols
                )
            )
        ):
            raise StagingError("staged swizzles must preserve the allocated tile extent")
        if load.linear_tid is None or any(
            _thread_value(load.linear_tid, thread) != thread for thread in range(load.tg_size)
        ):
            raise StagingError("staged copies require complete disjoint thread ownership")
        for field in fields(load):
            value = getattr(load, field.name)
            if isinstance(value, mir.MValue) and _depends_on(
                value, {loop.iv_name, compute.iv_name}
            ):
                raise StagingError("staged load operands must be invariant across the reduction")
    if roles != {False, True}:
        raise StagingError("staging requires complementary left and right matrix operands")
    initialized = set()
    consumed = set()
    multiply_count = 0
    for operation in compute.body:
        if isinstance(operation, mir.MSimdgroupLoad):
            load = load_by_name.get(operation.src_array)
            if (
                load is None
                or operation.in_type != load.elem_type
                or operation.stride != load.dst_stride
                or operation.kk_var != compute.iv_name
                or operation.is_b != (load.row_offset is None)
                or operation.tile_offset < 0
                or operation.tile_idx < 0
                or operation.swizzle_bits != load.swizzle_bits
                or operation.swizzle_shift != load.swizzle_shift
                or _depends_on(operation.sg_offset, {loop.iv_name, compute.iv_name})
            ):
                raise StagingError(
                    "SIMD-group reads must match staged memory identities and geometry"
                )
            extent = load.tile_cols if operation.is_b else load.tile_rows
            if operation.sg_offset is None:
                raise StagingError("staged fragment origins require a proven SIMD-group mapping")
            for thread in range(load.tg_size):
                origin = _thread_value(operation.sg_offset, thread) + operation.tile_offset
                group_origin = (
                    _thread_value(operation.sg_offset, (thread // 32) * 32) + operation.tile_offset
                )
                if origin != group_origin or origin < 0 or origin + 8 > extent:
                    raise StagingError("SIMD-group fragments must stay within the staged tile")
            initialized.add((operation.tile_name, operation.tile_idx))
            consumed.add(operation.src_array)
        elif isinstance(operation, mir.MSimdgroupMMA):
            if (operation.a_tile, operation.mi) not in initialized or (
                operation.b_tile,
                operation.ni,
            ) not in initialized:
                raise StagingError(
                    "staged matrix multiply reads a fragment before it is initialized"
                )
            multiply_count += 1
        else:
            raise StagingError("unsupported operation in staged matrix compute region")
    if not multiply_count or consumed != set(load_by_name):
        raise StagingError("staged compute must consume both matrix operands")
    return loads, compute


def _eligible_locations(operations):
    for operation in operations:
        if isinstance(operation, mir.MForLoop) and operation.iv_name == "kb":
            yield operation
        elif isinstance(operation, mir.MWhileTrue):
            yield from _eligible_locations(operation.body)


def _storage_users(function, loop, names):
    inside = {id(operation) for operation in _walk(loop.body)}
    for operation in _walk(function.ops):
        if id(operation) in inside or isinstance(operation, mir.MThreadgroupAlloc):
            continue
        if any(
            isinstance(value := getattr(operation, field.name), str) and value in names
            for field in fields(operation)
            if field.name != "result"
        ):
            raise StagingError("staged storage has additional users outside its verified lifetime")


def _allocation_map(function):
    allocations = [
        operation for operation in function.ops if isinstance(operation, mir.MThreadgroupAlloc)
    ]
    names = [allocation.alloc_name for allocation in allocations]
    if len(allocations) != sum(
        isinstance(operation, mir.MThreadgroupAlloc) for operation in _walk(function.ops)
    ):
        raise StagingError("staging requires top-level threadgroup allocation lifetimes")
    if len(set(names)) != len(names):
        raise StagingError("threadgroup allocations must have unique names")
    if any(
        allocation.elem_type not in _ELEMENT_BYTES or allocation.size <= 0
        for allocation in allocations
    ):
        raise StagingError("threadgroup allocations require supported types and positive sizes")
    return {allocation.alloc_name: allocation for allocation in allocations}


def _new_pipeline(function, loop, allocations):
    loads, _ = _loop_parts(loop, function)
    _storage_users(function, loop, {load.tg_array for load in loads})
    occupied = {parameter.name for parameter in function.params} | set(allocations)
    occupied.update(
        operation.result.name for operation in _walk(function.ops) if operation.result is not None
    )
    occupied.update(
        operation.iv_name
        for operation in _walk(function.ops)
        if isinstance(operation, mir.MForLoop)
    )
    occupied.update({"_stage_swap", *function.dimension_bindings})
    buffers = []
    for load in loads:
        allocation = allocations.get(load.tg_array)
        if (
            allocation is None
            or allocation.elem_type != load.elem_type
            or allocation.size < load.tile_rows * load.dst_stride
        ):
            raise StagingError("staged allocation must cover the complete padded operand tile")
        prefix = {"shared_a": "sa", "shared_b": "sb"}.get(load.tg_array, load.tg_array)
        buffer = StagingBuffer(
            load.tg_array,
            (f"{load.tg_array}_0", f"{load.tg_array}_1"),
            load.elem_type,
            allocation.size,
            f"{prefix}_curr",
            f"{prefix}_next",
        )
        generated = (*buffer.slots, buffer.current_pointer, buffer.next_pointer)
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name in occupied
            for name in generated
        ):
            raise StagingError("staging buffer names collide with existing kernel identifiers")
        occupied.update(generated)
        buffers.append(buffer)
    buffers = tuple(buffers)
    pipeline = SoftwarePipeline(buffers, _phases(buffers))
    validate_pipeline(pipeline)
    return pipeline


def materialize_double_buffer(function: mir.MFunction, max_tg_bytes: int = 30720) -> bool:
    """Atomically materialize one verified staging region, or leave the function untouched."""
    if function.kernel_type not in {"gemm", "persistent_gemm"}:
        return False
    if any(
        isinstance(operation, mir.MForLoop) and operation.staging is not None
        for operation in _walk(function.ops)
    ):
        validate_staging(function)
        return False
    candidates = tuple(_eligible_locations(function.ops))
    if len(candidates) != 1:
        return False
    loop = candidates[0]
    try:
        allocations = _allocation_map(function)
        pipeline = _new_pipeline(function, loop, allocations)
        original_bytes = sum(
            allocation.size * _ELEMENT_BYTES[allocation.elem_type]
            for allocation in allocations.values()
        )
        if (
            original_bytes + sum(buffer.bytes_per_slot for buffer in pipeline.buffers)
            > max_tg_bytes
        ):
            return False
    except StagingError:
        return False
    staged = {buffer.logical_name: buffer for buffer in pipeline.buffers}
    operations = []
    for operation in function.ops:
        if isinstance(operation, mir.MThreadgroupAlloc) and operation.alloc_name in staged:
            operations.extend(
                replace(operation, alloc_name=name) for name in staged[operation.alloc_name].slots
            )
        else:
            operations.append(operation)
    function.ops = operations
    loop.staging = pipeline
    return True


def validate_staging(function: mir.MFunction):
    """Recheck materialized staging after optimization and before source emission."""
    loops = [
        operation
        for operation in _walk(function.ops)
        if isinstance(operation, mir.MForLoop) and operation.staging is not None
    ]
    if not loops:
        return
    if len(loops) != 1 or loops[0] not in tuple(_eligible_locations(function.ops)):
        raise StagingError("staging requires one uniformly executed reduction region")
    loop = loops[0]
    pipeline = loop.staging
    validate_pipeline(pipeline)
    loads, _ = _loop_parts(loop, function)
    allocations = _allocation_map(function)
    if tuple(load.tg_array for load in loads) != tuple(
        buffer.logical_name for buffer in pipeline.buffers
    ):
        raise StagingError("staging buffer identities no longer match cooperative loads")
    for buffer, load in zip(pipeline.buffers, loads):
        if buffer.logical_name in allocations:
            raise StagingError("materialized staging must replace the logical allocation")
        if (
            buffer.element_type != load.elem_type
            or buffer.elements < load.tile_rows * load.dst_stride
        ):
            raise StagingError("staging buffer metadata conflicts with the lowered tile")
        for slot in buffer.slots:
            allocation = allocations.get(slot)
            if (
                allocation is None
                or allocation.elem_type != buffer.element_type
                or allocation.size != buffer.elements
            ):
                raise StagingError(
                    "staging physical allocations do not match the pipeline contract"
                )
    _storage_users(
        function,
        loop,
        {name for buffer in pipeline.buffers for name in (buffer.logical_name, *buffer.slots)},
    )
