"""Select and explain execution schedules before materializing Metal operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from metile.compiler.ownership import requires_exchange, validate_thread_layouts
from metile.ir import tile_ir as tir


@dataclass(frozen=True)
class SchedulePlan:
    """An immutable execution contract, independent of emitted Metal source."""

    backend: str
    threadgroup_size: tuple[int, int, int]
    simdgroup_grid: tuple[int, int] | None
    tile_shape: tuple[int, ...] | None
    staging: str
    vector_width: int | None = None
    double_buffer: bool | None = None
    reasons: tuple[str, ...] = ()
    cooperative_layout: str = "scalar lanes"
    outer_bounds_proven: bool = False

    def to_dict(self) -> dict:
        """Return JSON-compatible report data."""
        values = asdict(self)
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in values.items()
        }

    def format(self) -> str:
        """Describe the selected schedule without hiding unresolved pass decisions."""
        geometry = "x".join(str(size) for size in self.threadgroup_size)
        grid = "none" if self.simdgroup_grid is None else "x".join(map(str, self.simdgroup_grid))
        vector = "auto" if self.vector_width is None else str(self.vector_width)
        buffering = "auto" if self.double_buffer is None else str(self.double_buffer).lower()
        description = (
            f"backend={self.backend}, threads={geometry}, simdgroups={grid}, "
            f"staging={self.staging}, vector_width={vector}, double_buffer={buffering}"
        )
        return "\n".join((description, *(f"- {reason}" for reason in self.reasons)))


def _walk_ops(operations):
    for operation in operations:
        yield operation
        yield from _walk_ops(getattr(operation, "body", ()))


def _positive_count(value, name: str) -> int:
    from metile.compiler.lowering.common import LoweringError

    if type(value) is not int or value < 1 or value > 32:
        raise LoweringError(f"{name} must be an integer between 1 and 32")
    return value


def _requested_count(constexprs: dict, schedule) -> int | None:
    from metile.compiler.lowering.common import LoweringError

    legacy_count = constexprs.get("NUM_SG")
    if legacy_count is not None:
        legacy_count = _positive_count(legacy_count, "NUM_SG")
    if schedule.num_simdgroups is not None:
        count = _positive_count(schedule.num_simdgroups, "Schedule.num_simdgroups")
        if legacy_count is not None and legacy_count != count:
            raise LoweringError("NUM_SG conflicts with Schedule.num_simdgroups")
        return count
    return legacy_count


def _grid_candidates(constexprs: dict, count: int | None, default: tuple[int, int]):
    from metile.compiler.lowering.common import LoweringError

    rows = constexprs.get("WM")
    columns = constexprs.get("WN")
    if rows is not None:
        rows = _positive_count(rows, "WM")
    if columns is not None:
        columns = _positive_count(columns, "WN")
    if count is None:
        candidates = [(rows or default[0], columns or default[1])]
    else:
        candidates = [
            (factor, count // factor)
            for factor in range(1, count + 1)
            if count % factor == 0
            and (rows is None or factor == rows)
            and (columns is None or count // factor == columns)
        ]
        candidates.sort(key=lambda grid: (abs(grid[0] - grid[1]), grid))
    if not candidates:
        raise LoweringError("WM/WN conflict with the requested SIMDgroup count")
    return [grid for grid in candidates if grid[0] * grid[1] <= 32]


def _tensor_grid_legal(tiles, grid, dtype, constexprs, nax):
    block_rows, block_columns, block_reduction = tiles
    warp_rows, warp_columns = grid
    if block_rows % warp_rows or block_columns % warp_columns:
        return False
    subtile_rows, subtile_columns = block_rows // warp_rows, block_columns // warp_columns
    if subtile_rows > 32 or subtile_columns > 32:
        return False
    if nax:
        return (
            subtile_rows == subtile_columns == 32
            and block_reduction == 16
            and not constexprs.get("COOPERATIVE", False)
        )
    return dtype == "f32" or (
        dtype == "f16"
        and subtile_rows == subtile_columns == 32
        and not constexprs.get("COOPERATIVE", False)
    )


def _with_geometry(func, grid, *, nax=False):
    constexprs = dict(func.constexprs)
    constexprs.update(WM=grid[0], WN=grid[1], NUM_SG=grid[0] * grid[1])
    constexprs["NAX_FRAGMENTS"] = nax
    constexprs["_PLANNED_SIMDGROUP_GRID"] = grid
    return replace(func, constexprs=constexprs)


def _nax_alignment(func, binding):
    constexprs = func.constexprs
    runtime = dict(constexprs.get("_RUNTIME_SCALARS", ()))
    for axis, dimension, tile_size in zip(
        ("M", "N", "K"), binding.dimensions, binding.tiles, strict=True
    ):
        if axis == "M":
            continue
        if isinstance(dimension.defining_op, tir.Constant):
            aligned = (
                dimension.defining_op.value > 0 and dimension.defining_op.value % tile_size == 0
            )
        elif dimension.name in runtime:
            aligned = runtime[dimension.name] > 0 and runtime[dimension.name] % tile_size == 0
        else:
            aligned = constexprs.get(f"_ALIGNED_{axis}", False)
        if not aligned:
            raise ValueError("NAX fragments currently require aligned N and K")


def _outer_bounds_proven(func, binding):
    runtime = dict(func.constexprs.get("_RUNTIME_SCALARS", ()))
    alignment = dict(func.constexprs.get("_SCALAR_ALIGNMENT_32", ()))
    for dimension, block in zip(binding.dimensions[:2], binding.tiles[:2], strict=True):
        if isinstance(dimension.defining_op, tir.Constant):
            if dimension.defining_op.value % block:
                return False
        elif dimension.name in runtime:
            if runtime[dimension.name] % block:
                return False
        elif 32 % block or alignment.get(dimension.name, -1) % block:
            return False
    return True


def _elementwise_plan(func, schedule, count):
    from metile.compiler.lowering.common import LoweringError

    if schedule.backend not in {"auto", "elementwise"}:
        raise LoweringError("Matrix backends require a GEMM dot recurrence")
    operations = list(_walk_ops(func.ops))
    ownerships = validate_thread_layouts(func)
    if ownerships and schedule.vector_width not in {None, 1}:
        raise LoweringError(
            "ThreadLayout uses explicit scalar register ownership, not forced vector memory accesses"
        )
    sizes = [operation.size for operation in operations if isinstance(operation, tir.Arange)]
    if func.tensors and len(set(sizes)) > 1:
        raise LoweringError("Tensor accesses require one consistent arange size")
    block = sizes[-1] if sizes else func.constexprs.get("BLOCK", 1)
    block = block or 1
    logical_size = block
    if ownerships:
        block = ownerships[0].layout.thread_count
    if type(block) is not int or not 1 <= block <= 1024:
        raise LoweringError("Elementwise threadgroup size must be between 1 and 1024")
    if count is not None and block != count * 32:
        raise LoweringError(
            "Requested SIMDgroup count conflicts with the arange/BLOCK lane geometry"
        )
    shared = any(isinstance(operation, tir.SharedAlloc) for operation in operations) or (
        bool(ownerships)
        and (
            requires_exchange(func)
            or (block > 32 and any(isinstance(operation, tir.Reduce) for operation in operations))
        )
    )
    staging = "threadgroup" if shared else "device"
    if schedule.staging != "auto" and schedule.staging != staging:
        raise LoweringError("Elementwise staging must match the declared memory operations")
    if schedule.double_buffer:
        raise LoweringError("Elementwise double buffering is not implemented")
    return SchedulePlan(
        backend="elementwise",
        threadgroup_size=(block, 1, 1),
        simdgroup_grid=None,
        tile_shape=(logical_size,),
        staging=staging,
        vector_width=schedule.vector_width,
        double_buffer=schedule.double_buffer,
        reasons=("Lane geometry follows the traced arange/BLOCK contract.",),
        cooperative_layout=(
            f"checked register/thread-bit ownership; {ownerships[0].elements_per_thread} scalar(s) per thread"
            if ownerships
            else "scalar lanes"
        ),
    )


def _specialized_plan(func, binding, schedule, count):
    from metile.compiler.lowering.common import LoweringError, _compute_simdgroup_layout

    if schedule.backend not in {"auto", "simdgroup"} or schedule.staging == "device":
        raise LoweringError("Specialized GEMM requires the threadgroup-staged SIMDgroup backend")
    if schedule.double_buffer is False:
        raise LoweringError("Specialized producer/consumer GEMM requires double buffering")
    if func.constexprs.get("NAX_FRAGMENTS", False):
        raise LoweringError("Specialized GEMM does not support NAX fragments")
    roles = [
        operation for operation in _walk_ops(func.ops) if isinstance(operation, tir.SimdgroupRole)
    ]
    producers = next(
        (
            role.num_sgs or 2
            for role in roles
            if any(isinstance(op, tir.TileLoad) for op in role.body)
        ),
        None,
    )
    consumers = next(
        (role.num_sgs or 4 for role in roles if any(isinstance(op, tir.Dot) for op in role.body)),
        None,
    )
    if producers is None or consumers is None:
        raise LoweringError("Specialized GEMM needs producer and consumer roles")
    _positive_count(producers, "Producer SIMDgroups")
    _positive_count(consumers, "Consumer SIMDgroups")
    total = _positive_count(producers + consumers, "Specialized SIMDgroups")
    if count is not None and count != total:
        raise LoweringError(
            "Requested SIMDgroup count conflicts with explicit producer/consumer roles"
        )
    if "WM" in func.constexprs or "WN" in func.constexprs:
        raise LoweringError("Specialized GEMM defines geometry through explicit roles, not WM/WN")
    layout = _compute_simdgroup_layout(*binding.tiles[:2], consumers)
    return SchedulePlan(
        backend="simdgroup",
        threadgroup_size=(total * 32, 1, 1),
        simdgroup_grid=(layout.sg_rows, layout.sg_cols),
        tile_shape=binding.tiles,
        staging="threadgroup",
        vector_width=schedule.vector_width,
        double_buffer=True,
        reasons=("Explicit producer/consumer roles determine geometry and double buffering.",),
        cooperative_layout="8x8 simdgroup_matrix fragments; consumer SIMDgroup grid",
        outer_bounds_proven=_outer_bounds_proven(func, binding),
    )


def plan_schedule(func: tir.Function, *, supports_tensor_ops: bool | None = None) -> SchedulePlan:
    """Resolve legal backend, lane-group geometry and temporary placement."""
    from metile.compiler.lowering.common import (
        LoweringError,
        _analyze_gemm,
        _check_tg_memory,
        _compute_simdgroup_layout,
        _is_gemm,
        _is_persistent_gemm,
        _is_specialized_gemm,
        _select_num_sg,
        _tensor_ops_aligned,
    )
    from metile.compiler.options import Schedule

    schedule = func.constexprs.get("SCHEDULE", Schedule())
    if not isinstance(schedule, Schedule):
        raise LoweringError("SCHEDULE must be a metile.Schedule")
    constexprs = func.constexprs
    validate_thread_layouts(func)
    count = _requested_count(constexprs, schedule)
    if not _is_gemm(func):
        return _elementwise_plan(func, schedule, count)
    binding = _analyze_gemm(func)
    if _is_specialized_gemm(func):
        return _specialized_plan(func, binding, schedule, count)
    persistent = _is_persistent_gemm(func)
    if schedule.backend == "elementwise":
        raise LoweringError("The elementwise backend cannot lower a GEMM dot recurrence")
    nax = constexprs.get("NAX_FRAGMENTS", False) or schedule.backend == "nax"
    if nax and schedule.backend not in {"auto", "nax"}:
        raise LoweringError("NAX_FRAGMENTS conflicts with the requested backend")
    if persistent and (nax or schedule.backend not in {"auto", "simdgroup"}):
        raise LoweringError("Persistent GEMM currently requires the SIMDgroup backend")
    dtype = binding.left.ptr.type.dtype
    relaxed = constexprs.get("RELAXED_PRECISION", dtype != "f16" or nax)
    if nax and dtype == "f32" and not relaxed:
        raise LoweringError(
            "Strict f32 NAX packed fragment layout is unsupported; disable NAX_FRAGMENTS"
        )
    reasons = []
    controlled_memory = schedule.vector_width is not None or schedule.double_buffer is True
    try_tensor = (
        not persistent
        and schedule.backend != "simdgroup"
        and schedule.staging != "threadgroup"
        and not controlled_memory
    )
    require_tensor = nax or schedule.backend == "tensor_ops" or schedule.staging == "device"
    if require_tensor and controlled_memory:
        if schedule.vector_width is not None:
            raise LoweringError(
                "Native tensor memory access widths are opaque; vector_width requires the SIMDgroup backend"
            )
        raise LoweringError(
            "Explicit double buffering requires the threadgroup-staged SIMDgroup backend"
        )
    if supports_tensor_ops is None and (try_tensor or require_tensor):
        from metile.runtime.metal_device import MetalDevice

        supports_tensor_ops = MetalDevice.get().supports_tensor_ops
    if require_tensor and (not supports_tensor_ops or not try_tensor):
        raise LoweringError(
            "The requested backend/placement requires available Metal tensor operations"
        )
    if try_tensor and supports_tensor_ops:
        for grid in _grid_candidates(constexprs, count, (2, 2)):
            if not _tensor_grid_legal(binding.tiles, grid, dtype, constexprs, nax):
                continue
            scheduled = _with_geometry(func, grid, nax=nax)
            if nax:
                _nax_alignment(scheduled, binding)
            elif not _tensor_ops_aligned(scheduled):
                continue
            return SchedulePlan(
                backend="nax" if nax else "tensor_ops",
                threadgroup_size=(grid[0] * grid[1] * 32, 1, 1),
                simdgroup_grid=grid,
                tile_shape=binding.tiles,
                staging="device",
                vector_width=schedule.vector_width,
                double_buffer=schedule.double_buffer,
                reasons=(
                    "Metal tensor operations support the dtype, shape and selected SIMDgroup grid.",
                    "Inputs load directly from device into register-resident cooperative tensors.",
                    "Native cooperative-tensor lane ownership is opaque to meTile.",
                ),
                cooperative_layout="native SDK cooperative_tensor (opaque lane ownership)",
                outer_bounds_proven=_outer_bounds_proven(func, binding),
            )
    if require_tensor:
        raise LoweringError(
            "No legal tensor-operations schedule satisfies the requested geometry, dtype and alignment"
        )
    if schedule.backend == "auto":
        reasons.append(
            "Explicit memory controls require compiler-owned SIMDgroup staging."
            if controlled_memory
            else "Tensor operations are unavailable or incompatible with the requested contract."
        )
    if count is None and "WM" not in constexprs and "WN" not in constexprs:
        count = _select_num_sg(*binding.tiles[:2])
    if count is None and ("WM" not in constexprs or "WN" not in constexprs):
        count = _select_num_sg(*binding.tiles[:2])
    candidates = _grid_candidates(constexprs, count, (2, 2))
    layout = None
    for grid in candidates:
        try:
            layout = _compute_simdgroup_layout(
                *binding.tiles[:2], grid[0] * grid[1], simdgroup_grid=grid
            )
            break
        except LoweringError:
            continue
    if layout is None:
        raise LoweringError(
            "No legal SIMDgroup layout satisfies the requested tile and group count"
        )
    block_rows, block_columns, block_reduction = binding.tiles
    _check_tg_memory(
        {"left": block_rows * block_reduction, "right": block_reduction * block_columns},
        "half" if dtype == "f16" else "float",
        "GEMM schedule",
    )
    reasons.extend(
        (
            "The selected SIMDgroup grid partitions 8x8 accumulator fragments without overlap.",
            "Cooperative loads stage operand tiles in threadgroup memory.",
        )
    )
    return SchedulePlan(
        backend="simdgroup",
        threadgroup_size=(layout.num_sg * 32, 1, 1),
        simdgroup_grid=(layout.sg_rows, layout.sg_cols),
        tile_shape=binding.tiles,
        staging="threadgroup",
        vector_width=schedule.vector_width,
        double_buffer=schedule.double_buffer,
        reasons=tuple(reasons),
        cooperative_layout="8x8 simdgroup_matrix fragments",
        outer_bounds_proven=_outer_bounds_proven(func, binding),
    )


def materialize_schedule(func: tir.Function, plan: SchedulePlan) -> tir.Function:
    """Copy compiler decisions into lowering inputs without changing the traced IR."""
    from metile.compiler.lowering.common import _is_specialized_gemm

    if plan.simdgroup_grid is None or _is_specialized_gemm(func):
        return replace(func, constexprs=dict(func.constexprs))
    return _with_geometry(func, plan.simdgroup_grid, nax=plan.backend == "nax")
