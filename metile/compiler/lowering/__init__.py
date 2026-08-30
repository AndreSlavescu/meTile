"""Lower tile IR to Metal IR."""

from __future__ import annotations

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
    _detect_dtype,
    _is_gemm,
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
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir


def lower(func: tir.Function) -> mir.MFunction:
    """Lower a Tile IR function to Metal IR."""
    if _is_persistent_gemm(func):
        return _lower_persistent_gemm(func)
    if _is_specialized_gemm(func):
        return _lower_specialized_gemm(func)
    if _is_gemm(func):
        from metile.runtime.metal_device import MetalDevice

        dtype, _ = _detect_dtype(func)
        constexprs = func.constexprs
        low_precision_nax = dtype == "f16" and constexprs.get("NAX_FRAGMENTS", False)
        if MetalDevice.get().supports_tensor_ops and (dtype == "f32" or low_precision_nax):
            # tensor_ops matmul2d requires SM,SN <= 32 for valid descriptor
            BM = constexprs.get("BLOCK_M", 128)
            BN = constexprs.get("BLOCK_N", 64)
            WM = constexprs.get("WM", 2)
            WN = constexprs.get("WN", 2)
            SM, SN = BM // WM, BN // WN
            if SM <= 32 and SN <= 32:
                return _lower_tensor_ops_gemm(func)
        return _lower_gemm(func)
    ctx = _ElementwiseLoweringContext(func)
    return ctx.lower()
