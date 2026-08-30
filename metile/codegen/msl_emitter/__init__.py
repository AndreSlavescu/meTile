"""Emit Metal Shading Language from Metal IR."""

from __future__ import annotations

from metile.codegen.msl_emitter.common import (
    _val_name as _val_name,
)
from metile.codegen.msl_emitter.elementwise import (
    _emit_elementwise,
)
from metile.codegen.msl_emitter.gemm import (
    _emit_gemm,
    _emit_tensor_ops_kernel,
)
from metile.codegen.msl_emitter.nax import (
    _emit_nax_binary_fragment as _emit_nax_binary_fragment,
)
from metile.codegen.msl_emitter.nax import (
    _emit_nax_load_block_scale as _emit_nax_load_block_scale,
)
from metile.codegen.msl_emitter.nax import (
    _emit_nax_load_block_scaled_fragment as _emit_nax_load_block_scaled_fragment,
)
from metile.codegen.msl_emitter.nax import (
    _emit_nax_store_fragment as _emit_nax_store_fragment,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_cooperative_load as _emit_cooperative_load,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_dot_accumulate as _emit_dot_accumulate,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_dot_accumulator_init as _emit_dot_accumulator_init,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_dot_residual_store as _emit_dot_residual_store,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_paired_dot_accumulate as _emit_paired_dot_accumulate,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_paired_dot_accumulator_init as _emit_paired_dot_accumulator_init,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_paired_dot_swiglu_store as _emit_paired_dot_swiglu_store,
)
from metile.codegen.msl_emitter.simdgroup import (
    _emit_simdgroup_qmv_layout as _emit_simdgroup_qmv_layout,
)
from metile.ir import metal_ir as mir


def emit(func: mir.MFunction) -> str:
    """Generate MSL source code from a Metal IR function."""
    if func.kernel_type == "tensor_ops_gemm":
        return _emit_tensor_ops_kernel(func)
    if func.kernel_type in ("gemm", "persistent_gemm", "specialized_gemm"):
        return _emit_gemm(func)
    return _emit_elementwise(func)
