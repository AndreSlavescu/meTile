"""Affine-quantized MLX backends."""

from metile.backends.mlx_quantized.common import (
    _SWITCH_MARGIN as _SWITCH_MARGIN,
)
from metile.backends.mlx_quantized.common import (
    _affine_residual_schedule_cache as _affine_residual_schedule_cache,
)
from metile.backends.mlx_quantized.common import (
    _affine_swiglu_schedule_cache as _affine_swiglu_schedule_cache,
)
from metile.backends.mlx_quantized.common import (
    _compiled_affine_swiglu as _compiled_affine_swiglu,
)
from metile.backends.mlx_quantized.common import (
    _repacked_affine_pair as _repacked_affine_pair,
)
from metile.backends.mlx_quantized.common import (
    repack_mlx_affine_weight as repack_mlx_affine_weight,
)
from metile.backends.mlx_quantized.dispatch import (
    mlx_affine_mlp_executor as mlx_affine_mlp_executor,
)
from metile.backends.mlx_quantized.dispatch import (
    mlx_affine_residual_qmv as mlx_affine_residual_qmv,
)
from metile.backends.mlx_quantized.dispatch import (
    mlx_affine_swiglu as mlx_affine_swiglu,
)
from metile.backends.mlx_quantized.dispatch import (
    mlx_affine_swiglu_backend_signature as mlx_affine_swiglu_backend_signature,
)
from metile.backends.mlx_quantized.dispatch import (
    mlx_affine_swiglu_executor as mlx_affine_swiglu_executor,
)
from metile.backends.mlx_quantized.qmv import (
    mlx_affine_qmv_nax as mlx_affine_qmv_nax,
)
from metile.backends.mlx_quantized.residual import (
    _AFFINE_RESIDUAL_CONFIGS as _AFFINE_RESIDUAL_CONFIGS,
)
from metile.backends.mlx_quantized.residual import (
    MLXAffineResidualConfig as MLXAffineResidualConfig,
)
from metile.backends.mlx_quantized.residual import (
    _affine_residual_dispatch as _affine_residual_dispatch,
)
from metile.backends.mlx_quantized.residual import (
    _choose_affine_residual_config as _choose_affine_residual_config,
)
from metile.backends.mlx_quantized.residual import (
    _compile_affine_qmv as _compile_affine_qmv,
)
from metile.backends.mlx_quantized.residual import (
    _native_affine_residual_qmv as _native_affine_residual_qmv,
)
from metile.backends.mlx_quantized.residual import (
    mlx_affine_qmv as mlx_affine_qmv,
)
from metile.backends.mlx_quantized.residual import (
    mlx_affine_residual_qmv_dispatches as mlx_affine_residual_qmv_dispatches,
)
from metile.backends.mlx_quantized.swiglu import (
    _AFFINE_SWIGLU_CONFIGS as _AFFINE_SWIGLU_CONFIGS,
)
from metile.backends.mlx_quantized.swiglu import (
    MLXAffineSwiGLUConfig as MLXAffineSwiGLUConfig,
)
from metile.backends.mlx_quantized.swiglu import (
    _affine_swiglu_compatible as _affine_swiglu_compatible,
)
from metile.backends.mlx_quantized.swiglu import (
    _affine_swiglu_configs as _affine_swiglu_configs,
)
from metile.backends.mlx_quantized.swiglu import (
    _affine_swiglu_dispatch as _affine_swiglu_dispatch,
)
from metile.backends.mlx_quantized.swiglu import (
    _choose_affine_swiglu_config as _choose_affine_swiglu_config,
)
from metile.backends.mlx_quantized.swiglu import (
    _compile_affine_swiglu_qmv as _compile_affine_swiglu_qmv,
)
from metile.backends.mlx_quantized.swiglu import (
    _compile_affine_swiglu_scratch_qmv as _compile_affine_swiglu_scratch_qmv,
)
from metile.backends.mlx_quantized.swiglu import (
    _compile_nax_affine_swiglu_qmv as _compile_nax_affine_swiglu_qmv,
)
from metile.backends.mlx_quantized.swiglu import (
    _make_affine_swiglu_executor as _make_affine_swiglu_executor,
)
from metile.backends.mlx_quantized.swiglu import (
    _mlx_compiled_affine_swiglu as _mlx_compiled_affine_swiglu,
)
from metile.backends.mlx_quantized.swiglu import (
    _native_affine_swiglu as _native_affine_swiglu,
)
from metile.backends.mlx_quantized.swiglu import (
    mlx_affine_swiglu_dispatches as mlx_affine_swiglu_dispatches,
)
from metile.backends.mlx_quantized.swiglu import (
    mlx_affine_swiglu_qmv as mlx_affine_swiglu_qmv,
)
from metile.backends.mlx_quantized.swiglu import (
    mlx_affine_swiglu_qmv_nax as mlx_affine_swiglu_qmv_nax,
)

__all__ = [
    "MLXAffineResidualConfig",
    "MLXAffineSwiGLUConfig",
    "mlx_affine_mlp_executor",
    "mlx_affine_qmv",
    "mlx_affine_qmv_nax",
    "mlx_affine_residual_qmv",
    "mlx_affine_residual_qmv_dispatches",
    "mlx_affine_swiglu",
    "mlx_affine_swiglu_backend_signature",
    "mlx_affine_swiglu_dispatches",
    "mlx_affine_swiglu_executor",
    "mlx_affine_swiglu_qmv",
    "mlx_affine_swiglu_qmv_nax",
    "repack_mlx_affine_weight",
]
