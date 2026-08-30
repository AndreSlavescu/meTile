"""Shared helpers for the affine-quantized MLX backends."""

from __future__ import annotations

import threading

from metile.runtime.cache import cache_root

_affine_qmv_kernel_cache = {}
_affine_swiglu_kernel_cache = {}
_nax_affine_qmv_kernel_cache = {}
_nax_affine_swiglu_kernel_cache = {}
_affine_swiglu_schedule_cache = {}
_affine_residual_schedule_cache = {}
_affine_weight_cache = {}
_compiled_affine_swiglu = None
_compiled_affine_residual_qmv = None
_affine_cache_lock = threading.RLock()
_affine_cache_path = cache_root() / "mlx-affine-swiglu-autotune-v7.json"
_affine_residual_cache_path = cache_root() / "mlx-affine-residual-autotune-v3.json"
_SWITCH_MARGIN = 0.03
# Switching to the mx.compile variant has to clear the run-to-run noise floor, which is
# several percent on this hardware. At the previous 0.005 it won tournaments on noise and
# then measured 0.88-0.90x against eager MLX in steady state.
_COMPILED_SWITCH_MARGIN = 0.03
_RESIDUAL_SWITCH_MARGIN = 0.01
_AFFINE_SWIGLU_TUNER_VERSION = 7
_AFFINE_RESIDUAL_TUNER_VERSION = 3


def _tensor_kernel_header():
    return (
        "#include <metal_tensor>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace mpp::tensor_ops;"
    )


def repack_mlx_affine_weight(weight, scales, biases):
    """Repack MLX output-major affine uint4 weights into K-major NAX layout.

    Four bits throughout, matching the matrix unit's affine fragment format. Generalising this to
    eight is a few lines and does not help: the consumer, `lower_affine_matmul`, emits NAX affine
    fragments with block_size=4 and takes no bit width, so a wider repack only produces weights the
    kernel will decode as nibbles. See the guard in MLXAffineWeight.from_mlx.
    """
    import mlx.core as mx

    if biases is None:
        raise ValueError("native affine QMV requires affine biases")
    if weight.ndim != 2 or scales.ndim != 2 or biases.shape != scales.shape:
        raise ValueError("expected packed weight and matching two-dimensional parameters")
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    quantized = ((weight[..., None] >> shifts) & 15).astype(mx.uint8)
    quantized = quantized.reshape(weight.shape[0], weight.shape[1] * 8).T.reshape(-1)
    packed = quantized[0::2] | (quantized[1::2] << 4)
    repacked = packed, scales.T, biases.T
    mx.eval(*repacked)
    return repacked


def _repacked_affine_pair(
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
):
    key = (id(gate_weight), id(up_weight))
    cached = _affine_weight_cache.get(key)
    if cached is not None and cached[0] is gate_weight and cached[1] is up_weight:
        return cached[2]
    repacked = (
        *repack_mlx_affine_weight(gate_weight, gate_scales, gate_biases),
        *repack_mlx_affine_weight(up_weight, up_scales, up_biases),
    )
    _affine_weight_cache[key] = (gate_weight, up_weight, repacked)
    return repacked


def _discard_repacked_affine_pair(gate_weight, up_weight):
    key = (id(gate_weight), id(up_weight))
    cached = _affine_weight_cache.get(key)
    if cached is not None and cached[0] is gate_weight and cached[1] is up_weight:
        del _affine_weight_cache[key]
