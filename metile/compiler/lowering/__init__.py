"""Lower tile IR to Metal IR."""

from __future__ import annotations

from metile.compiler.layout_optimizations import optimize_layout_conversions
from metile.compiler.lowering.common import (
    LoweringError as LoweringError,
)
from metile.compiler.lowering.common import (
    _compute_coop_load_layout as _compute_coop_load_layout,
)
from metile.compiler.lowering.common import (
    _compute_simdgroup_layout as _compute_simdgroup_layout,
)
from metile.compiler.lowering.common import (
    _is_persistent_gemm,
    _is_specialized_gemm,
)
from metile.compiler.lowering.elementwise import (
    _ElementwiseLoweringContext,
)
from metile.compiler.lowering.gemm import (
    _lower_gemm,
    _lower_persistent_gemm,
    _lower_specialized_gemm,
    _lower_tensor_ops_gemm,
)
from metile.compiler.ownership import validate_thread_layouts
from metile.compiler.planning import materialize_schedule, plan_schedule
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir


def lower(func: tir.Function) -> mir.MFunction:
    """Lower a Tile IR function to Metal IR."""
    validate_thread_layouts(func)
    func, layout_optimizations = optimize_layout_conversions(func)
    plan = plan_schedule(func)
    if plan.backend != "simdgroup_inline":
        pending = list(func.ops)
        while pending:
            operation = pending.pop()
            if (
                isinstance(operation, (tir.TileLoad, tir.TileStore))
                and operation.scratch is not None
            ):
                raise LoweringError("explicit matrix scratch requires simdgroup_inline")
            pending.extend(getattr(operation, "body", ()))
    func = materialize_schedule(func, plan)
    if plan.backend == "simdgroup_inline":
        from metile.compiler.lowering.fragments import InlineMatrixLowering

        lowered = InlineMatrixLowering(func).lower()
    elif _is_persistent_gemm(func):
        lowered = _lower_persistent_gemm(func)
    elif _is_specialized_gemm(func):
        lowered = _lower_specialized_gemm(func)
    elif plan.backend in {"tensor_ops", "nax"}:
        lowered = _lower_tensor_ops_gemm(func)
    elif plan.backend == "simdgroup":
        lowered = _lower_gemm(func)
    else:
        lowered = _ElementwiseLoweringContext(func).lower()
    if lowered.threadgroup_size != plan.threadgroup_size:
        raise LoweringError("Lowering changed the planned threadgroup geometry")
    lowered.schedule_plan = plan
    lowered.layout_optimizations = layout_optimizations
    return lowered
