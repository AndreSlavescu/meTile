"""Inspectable decisions from the materialized compiler pipeline, not GPU ISA claims."""

import json
from dataclasses import asdict, dataclass

from metile.ir import metal_ir as mir


@dataclass(frozen=True)
class LoopLayout:
    induction_variable: str
    kind: str
    lanes: int
    elements_per_lane: int
    iteration_step: int


@dataclass(frozen=True)
class MemoryAllocation:
    name: str
    address_space: str
    element_type: str
    elements: int
    bytes: int


@dataclass(frozen=True)
class ExecutionReport:
    plan: object
    passes: tuple[str, ...]
    loops: tuple[LoopLayout, ...]
    allocations: tuple[MemoryAllocation, ...]
    double_buffered: bool
    vectorized_loads: int
    epilogue_regions: int
    notes: tuple[str, ...]
    value_layouts: tuple = ()
    layout_conversions: tuple = ()
    pipelines: tuple = ()
    register_reductions: tuple = ()
    layout_optimizations: object = None
    grouped_loads: int = 0
    grouped_stores: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def walk_operations(operations):
    for operation in operations:
        yield operation
        yield from walk_operations(getattr(operation, "body", ()))


def execution_report(function: mir.MFunction, passes: list[str]) -> ExecutionReport:
    operations = tuple(walk_operations(function.ops))
    loops = []
    allocations = []
    element_sizes = {"float": 4, "half": 2, "int": 4, "uint": 4, "uchar": 1, "bool": 1}
    for operation in operations:
        if isinstance(operation, mir.MForLoop):
            width = getattr(operation, "_vec_size", 1)
            kind = (
                "aligned_interior"
                if getattr(operation, "_ew_aligned", False)
                else "masked_tail"
                if getattr(operation, "_ew_tail", False)
                else "loop"
            )
            loops.append(
                LoopLayout(
                    operation.iv_name,
                    kind,
                    function.threadgroup_size[0],
                    width,
                    operation.step * width,
                )
            )
        if isinstance(operation, mir.MThreadgroupAlloc):
            allocations.append(
                MemoryAllocation(
                    operation.alloc_name,
                    "threadgroup",
                    operation.elem_type,
                    operation.size,
                    operation.size * element_sizes[operation.elem_type],
                )
            )
    double_buffered = any(
        getattr(operation, "staging", None) is not None
        or getattr(operation, "_double_buffered", False)
        or getattr(operation, "_specialized_db", False)
        for operation in operations
    )
    vectorized_loads = sum(
        (isinstance(operation, mir.MCooperativeLoad) and operation.vec_size > 1)
        or isinstance(operation, mir.MVectorLoad)
        for operation in operations
    )
    grouped_loads = sum(isinstance(operation, mir.MVectorLoad) for operation in operations)
    grouped_stores = sum(isinstance(operation, mir.MVectorStore) for operation in operations)
    epilogue_regions = sum(
        isinstance(operation, (mir.MAccElemApply, mir.MCoopTensorEpilogue, mir.MNaxApplyFragment))
        for operation in operations
    )
    notes = [
        "Layout widths describe emitted Metal operations, not final machine instructions.",
        "Masked tails may use scalar accesses even when aligned interiors are vectorized.",
    ]
    if grouped_loads or grouped_stores:
        notes.append(
            "Grouped packed accesses retain independently masked scalar fallbacks; "
            "native vector values do not guarantee hardware vector register placement."
        )
    if function.schedule_plan is not None and function.schedule_plan.double_buffer is None:
        notes.append(
            "Automatic software double buffering selected."
            if double_buffered
            else "No software double buffering materialized."
        )
    return ExecutionReport(
        function.schedule_plan,
        tuple(passes),
        tuple(loops),
        tuple(allocations),
        double_buffered,
        vectorized_loads,
        epilogue_regions,
        tuple(notes),
        function.value_layouts,
        function.layout_conversions,
        tuple(
            operation.staging
            for operation in operations
            if getattr(operation, "staging", None) is not None
        ),
        function.register_reductions,
        function.layout_optimizations,
        grouped_loads,
        grouped_stores,
    )


def validate_materialized_schedule(function: mir.MFunction):
    from metile.compiler.lowering.common import LoweringError

    plan = function.schedule_plan
    if plan is None:
        return
    if plan.threadgroup_size != function.threadgroup_size:
        raise LoweringError("materialized threadgroup geometry does not match its schedule plan")
    operations = tuple(walk_operations(function.ops))
    if plan.vector_width == 1 and any(
        (isinstance(operation, mir.MForLoop) and getattr(operation, "_vec_size", 1) > 1)
        or (isinstance(operation, mir.MCooperativeLoad) and operation.vec_size > 1)
        or isinstance(operation, (mir.MVectorLoad, mir.MVectorStore))
        for operation in operations
    ):
        raise LoweringError("vector_width=1 conflicts with materialized vector operations")
    if plan.vector_width == 4:
        interiors = [
            operation
            for operation in operations
            if isinstance(operation, mir.MForLoop) and getattr(operation, "_ew_aligned", False)
        ]
        loads = [
            operation
            for operation in operations
            if isinstance(operation, mir.MCooperativeLoad) and not operation.bounds_check
        ]
        if (
            not (interiors or loads)
            or any(getattr(operation, "_vec_size", 1) != 4 for operation in interiors)
            or any(operation.vec_size != 4 for operation in loads)
        ):
            raise LoweringError("vector_width=4 requires provably vectorizable aligned interiors")
    double_buffered = any(
        getattr(operation, "staging", None) is not None
        or getattr(operation, "_double_buffered", False)
        or getattr(operation, "_specialized_db", False)
        for operation in operations
    )
    if plan.double_buffer is True and not double_buffered:
        raise LoweringError(
            "double_buffer=True cannot be satisfied within this schedule's resources"
        )
    if plan.double_buffer is False and double_buffered:
        raise LoweringError("double_buffer=False conflicts with materialized double buffering")
