from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np

import metile
from kernels.affine_qmv import (
    affine_qmv,
    affine_residual_qmv,
    affine_swiglu_qmv,
    affine_swiglu_scratch_qmv,
)
from metile.backends.mlx import (
    _mlx_compiler_dtype,
    _mlx_dtype_to_numpy,
    _mlx_kernel_body,
    _replace_identifier,
    _specialize_mlx_source,
    _token_bucket,
    calibrate_tournament_batch,
)
from metile.codegen.msl_emitter import emit
from metile.compiler.affine_quantized import lower_affine_qmv, lower_affine_swiglu_qmv
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

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


@dataclass(frozen=True)
class MLXAffineSwiGLUConfig:
    algorithm: str
    implementation: str = ""
    block: int = 0
    outputs_per_simdgroup: int = 1
    decode_dtype: str = "f32"


@dataclass(frozen=True)
class MLXAffineResidualConfig:
    algorithm: str
    block: int = 0
    outputs_per_simdgroup: int = 1
    decode_dtype: str = "f32"


_AFFINE_SWIGLU_CONFIGS = tuple(
    [MLXAffineSwiGLUConfig("mlx"), MLXAffineSwiGLUConfig("mlx_compiled")]
    + [
        MLXAffineSwiGLUConfig("metile", "scalar", block, outputs_per_simdgroup)
        for block, outputs_per_simdgroup in (
            (32, 1),
            (32, 2),
            (32, 4),
            (32, 8),
            (64, 1),
            (64, 2),
            (64, 4),
            (128, 1),
            (128, 2),
            (128, 4),
            (256, 1),
            (256, 2),
            (256, 4),
        )
    ]
    + [
        MLXAffineSwiGLUConfig("metile", "scalar", block, outputs_per_simdgroup, "f16")
        for block, outputs_per_simdgroup in (
            (32, 1),
            (64, 1),
            (128, 1),
            (128, 4),
            (256, 1),
            (256, 2),
        )
    ]
    + [MLXAffineSwiGLUConfig("metile", "nax", block) for block in (32, 64, 128)]
    + [MLXAffineSwiGLUConfig("metile", "nax_scratch", block) for block in (32, 64, 128)]
    # The fused SwiGLU kernels above are single-row, so from two rows up the only
    # candidates left were the scalar ones, which lose, and the tournament settled on
    # native MLX for the whole batched band. This one builds the block from two multi-row
    # affine matmuls, which keep weight traffic flat as rows grow.
    + [MLXAffineSwiGLUConfig("metile", "matmul")]
    + [
        MLXAffineSwiGLUConfig("metile", "scratch", block, outputs_per_simdgroup, decode_dtype)
        for block, outputs_per_simdgroup, decode_dtype in (
            (64, 1, "f32"),
            (64, 2, "f32"),
            (128, 1, "f32"),
            (128, 2, "f32"),
            (256, 1, "f32"),
            (256, 2, "f32"),
            (256, 1, "f16"),
        )
    ]
)

_AFFINE_RESIDUAL_CONFIGS = tuple(
    [
        MLXAffineResidualConfig("mlx"),
        MLXAffineResidualConfig("mlx_compiled"),
        MLXAffineResidualConfig("metile_matmul"),
    ]
    + [
        MLXAffineResidualConfig("metile", block, outputs_per_simdgroup, decode_dtype)
        for block, outputs_per_simdgroup, decode_dtype in (
            (32, 2, "f16"),
            (64, 1, "f32"),
            (64, 2, "f32"),
            (64, 2, "f16"),
            (128, 1, "f32"),
            (128, 1, "f16"),
            (256, 1, "f32"),
            (256, 2, "f32"),
            (256, 1, "f16"),
            (256, 2, "f16"),
        )
    ]
)


def _affine_swiglu_configs(dtype, bits):
    if bits == 4:
        return _AFFINE_SWIGLU_CONFIGS
    return tuple(
        config
        for config in _AFFINE_SWIGLU_CONFIGS
        if config.algorithm in {"mlx", "mlx_compiled"}
        or (
            config.implementation in {"scalar", "scratch"}
            and (str(dtype) != "mlx.core.bfloat16" or config.decode_dtype == "f32")
        )
    )


@dataclass(frozen=True)
class _MLXAffineQMVKernel:
    operation: object
    output_features: int
    block: int
    outputs_per_simdgroup: int
    description_bits: int

    def __call__(self, values, weight, scales, biases):
        rows = values.size // values.shape[-1]
        output_shape = (*values.shape[:-1], self.output_features)
        outputs_per_threadgroup = self.block // 32 * self.outputs_per_simdgroup
        threadgroups = (rows * self.output_features + outputs_per_threadgroup - 1) // (
            outputs_per_threadgroup
        )
        return self.operation(
            inputs=[values, weight, scales, biases],
            grid=(threadgroups * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[values.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXAffineResidualQMVKernel:
    operation: object
    output_features: int
    block: int
    outputs_per_simdgroup: int
    description_bits: int

    def __call__(self, values, weight, scales, biases, residual):
        rows = values.size // values.shape[-1]
        output_shape = (*values.shape[:-1], self.output_features)
        outputs_per_threadgroup = self.block // 32 * self.outputs_per_simdgroup
        threadgroups = (rows * self.output_features + outputs_per_threadgroup - 1) // (
            outputs_per_threadgroup
        )
        return self.operation(
            inputs=[values, weight, scales, biases, residual],
            grid=(threadgroups * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[values.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXAffineSwiGLUKernel:
    operation: object
    output_features: int
    block: int
    outputs_per_simdgroup: int
    description_bits: int

    def __call__(
        self,
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    ):
        rows = values.size // values.shape[-1]
        output_shape = (*values.shape[:-1], self.output_features)
        outputs_per_threadgroup = self.block // 32 * self.outputs_per_simdgroup
        threadgroups = (rows * self.output_features + outputs_per_threadgroup - 1) // (
            outputs_per_threadgroup
        )
        return self.operation(
            inputs=[
                values,
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
            ],
            grid=(threadgroups * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[values.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXNaxAffineQMVKernel:
    operation: object
    output_features: int
    block: int
    description_bits: int

    def __call__(self, values, packed, scales, biases):
        if values.size != values.shape[-1]:
            raise ValueError("native affine QMV currently supports one decode row")
        return self.operation(
            inputs=[values, packed, scales, biases],
            grid=(self.block, self.output_features // self.block, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[(*values.shape[:-1], self.output_features)],
            output_dtypes=[values.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXNaxAffineSwiGLUKernel:
    operation: object
    output_features: int
    block: int
    description_bits: int

    def __call__(
        self,
        values,
        gate_packed,
        gate_scales,
        gate_biases,
        up_packed,
        up_scales,
        up_biases,
    ):
        if values.size != values.shape[-1]:
            raise ValueError("native affine SwiGLU currently supports one decode row")
        return self.operation(
            inputs=[
                values,
                gate_packed,
                gate_scales,
                gate_biases,
                up_packed,
                up_scales,
                up_biases,
            ],
            grid=(self.block, self.output_features // self.block, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[(*values.shape[:-1], self.output_features)],
            output_dtypes=[values.dtype],
        )[0]


def _tensor_kernel_header():
    return (
        "#include <metal_tensor>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace mpp::tensor_ops;"
    )


def _compile_nax_affine_qmv(
    input_features,
    output_features,
    dtype,
    group_size=64,
    bits=4,
    block=32,
):
    import mlx.core as mx

    if bits != 4:
        raise ValueError("native affine QMV requires 4-bit weights")
    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    if numpy_dtype != np.dtype(np.float16):
        raise TypeError("native affine QMV requires float16 activations")
    kernel_key = (input_features, output_features, numpy_dtype.str, group_size, bits, block)
    cached = _nax_affine_qmv_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    function_name = f"metile_nax_affine_qmv_{stable_digest(kernel_key)[:16]}"
    metal_ir = decompose_nax_fragments(
        optimize_tile_schedules(
            lower_affine_qmv(
                function_name,
                output_features,
                input_features,
                block_n=block,
                group_size=group_size,
            )
        )
    )
    source = emit(metal_ir)
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "packed", "scales", "biases"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=_tensor_kernel_header(),
    )
    kernel = _MLXNaxAffineQMVKernel(
        operation,
        output_features,
        block,
        compressed_description_bits(source),
    )
    _nax_affine_qmv_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_nax_affine_swiglu_qmv(
    input_features,
    output_features,
    dtype,
    group_size=64,
    bits=4,
    block=32,
    lifetime_schedule="parallel",
):
    import mlx.core as mx

    if bits != 4:
        raise ValueError("native affine SwiGLU requires 4-bit weights")
    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    if numpy_dtype != np.dtype(np.float16):
        raise TypeError("native affine SwiGLU requires float16 activations")
    if lifetime_schedule not in {"parallel", "scratch"}:
        raise ValueError("native affine SwiGLU schedule must be parallel or scratch")
    kernel_key = (
        input_features,
        output_features,
        numpy_dtype.str,
        group_size,
        bits,
        block,
        lifetime_schedule,
    )
    cached = _nax_affine_swiglu_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    function_name = f"metile_nax_affine_swiglu_{stable_digest(kernel_key)[:16]}"
    metal_ir = optimize_tile_schedules(
        lower_affine_swiglu_qmv(
            function_name,
            output_features,
            input_features,
            block_n=block,
            group_size=group_size,
            lifetime_schedule=lifetime_schedule,
        )
    )
    source = emit(metal_ir)
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=[
            "activations",
            "gate_packed",
            "gate_scales",
            "gate_biases",
            "up_packed",
            "up_scales",
            "up_biases",
        ],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=_tensor_kernel_header(),
    )
    kernel = _MLXNaxAffineSwiGLUKernel(
        operation,
        output_features,
        block,
        compressed_description_bits(source),
    )
    _nax_affine_swiglu_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_affine_qmv(
    input_features,
    output_features,
    dtype,
    group_size=64,
    bits=4,
    block=32,
    outputs_per_simdgroup=1,
    decode_dtype="f32",
    fuse_residual=False,
):
    import mlx.core as mx

    numpy_dtype = _mlx_compiler_dtype(dtype)
    outputs_per_threadgroup = block // 32 * outputs_per_simdgroup
    if outputs_per_simdgroup < 1 or output_features % outputs_per_threadgroup:
        raise ValueError("output features must tile the affine QMV threadgroups")
    if decode_dtype not in {"f16", "f32"}:
        raise ValueError("affine QMV decode dtype must be f16 or f32")
    kernel_key = (
        "residual" if fuse_residual else "qmv",
        input_features,
        output_features,
        numpy_dtype.str,
        group_size,
        bits,
        block,
        outputs_per_simdgroup,
        decode_dtype,
    )
    cached = _affine_qmv_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    values = metile.Buffer.empty((input_features,), dtype=numpy_dtype)
    weight = metile.Buffer.empty((output_features, input_features * bits // 32), dtype=np.uint32)
    scales = metile.Buffer.empty((output_features, input_features // group_size), dtype=numpy_dtype)
    biases = metile.Buffer.empty(scales.shape, dtype=numpy_dtype)
    output = metile.Buffer.empty((output_features,), dtype=numpy_dtype)
    compile_arguments = [values, weight, scales, biases]
    if fuse_residual:
        compile_arguments.append(metile.Buffer.empty((output_features,), dtype=numpy_dtype))
    compile_arguments.extend((output, input_features, output_features))
    kernel_function = affine_residual_qmv if fuse_residual else affine_qmv
    compiled = kernel_function.get_compiled(
        *compile_arguments,
        GROUP_SIZE=group_size,
        BITS=bits,
        BLOCK=block,
        OUTPUTS_PER_SIMDGROUP=outputs_per_simdgroup,
        HALF_DECODE=decode_dtype == "f16",
    )
    source = _mlx_kernel_body(_specialize_mlx_source(compiled.msl_source, dtype))
    source = _replace_identifier(source, "K", f"{input_features}")
    source = _replace_identifier(source, "N", f"{output_features}")
    operation = mx.fast.metal_kernel(
        name=f"metile_affine_{'residual_' if fuse_residual else ''}qmv_{stable_digest(kernel_key)[:16]}",
        input_names=["X", "W", "Scales", "Biases"] + (["Residual"] if fuse_residual else []),
        output_names=["Out"],
        source=source,
    )
    kernel_class = _MLXAffineResidualQMVKernel if fuse_residual else _MLXAffineQMVKernel
    kernel = kernel_class(
        operation,
        output_features,
        block,
        outputs_per_simdgroup,
        compiled.description_bits,
    )
    _affine_qmv_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_affine_swiglu_qmv(
    input_features,
    output_features,
    dtype,
    group_size=64,
    bits=4,
    block=32,
    outputs_per_simdgroup=1,
    decode_dtype="f32",
):
    import mlx.core as mx

    numpy_dtype = _mlx_compiler_dtype(dtype)
    outputs_per_threadgroup = block // 32 * outputs_per_simdgroup
    if outputs_per_simdgroup < 1 or output_features % outputs_per_threadgroup:
        raise ValueError("output features must tile the affine SwiGLU threadgroups")
    if decode_dtype not in {"f16", "f32"}:
        raise ValueError("affine SwiGLU decode dtype must be f16 or f32")
    kernel_key = (
        input_features,
        output_features,
        numpy_dtype.str,
        group_size,
        bits,
        block,
        outputs_per_simdgroup,
        decode_dtype,
    )
    cached = _affine_swiglu_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    values = metile.Buffer.empty((input_features,), dtype=numpy_dtype)
    packed_shape = (output_features, input_features * bits // 32)
    parameter_shape = (output_features, input_features // group_size)
    gate_weight = metile.Buffer.empty(packed_shape, dtype=np.uint32)
    gate_scales = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    gate_biases = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    up_weight = metile.Buffer.empty(packed_shape, dtype=np.uint32)
    up_scales = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    up_biases = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    output = metile.Buffer.empty((output_features,), dtype=numpy_dtype)
    compiled = affine_swiglu_qmv.get_compiled(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        output,
        input_features,
        output_features,
        GROUP_SIZE=group_size,
        BITS=bits,
        BLOCK=block,
        OUTPUTS_PER_SIMDGROUP=outputs_per_simdgroup,
        HALF_DECODE=decode_dtype == "f16",
    )
    source = _mlx_kernel_body(_specialize_mlx_source(compiled.msl_source, dtype))
    source = _replace_identifier(source, "K", f"{input_features}")
    source = _replace_identifier(source, "N", f"{output_features}")
    operation = mx.fast.metal_kernel(
        name=f"metile_affine_swiglu_qmv_{stable_digest(kernel_key)[:16]}",
        input_names=[
            "X",
            "GateW",
            "GateScales",
            "GateBiases",
            "UpW",
            "UpScales",
            "UpBiases",
        ],
        output_names=["Out"],
        source=source,
    )
    kernel = _MLXAffineSwiGLUKernel(
        operation,
        output_features,
        block,
        outputs_per_simdgroup,
        compiled.description_bits,
    )
    _affine_swiglu_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_affine_swiglu_scratch_qmv(
    input_features,
    output_features,
    dtype,
    group_size=64,
    bits=4,
    block=64,
    outputs_per_simdgroup=1,
    decode_dtype="f32",
):
    import mlx.core as mx

    numpy_dtype = _mlx_compiler_dtype(dtype)
    outputs_per_threadgroup = block // 32 * outputs_per_simdgroup
    if outputs_per_simdgroup < 1 or output_features % outputs_per_threadgroup:
        raise ValueError("output features must tile the scratch SwiGLU threadgroups")
    if decode_dtype not in {"f16", "f32"}:
        raise ValueError("scratch SwiGLU decode dtype must be f16 or f32")
    kernel_key = (
        "scratch",
        input_features,
        output_features,
        numpy_dtype.str,
        group_size,
        bits,
        block,
        outputs_per_simdgroup,
        decode_dtype,
    )
    cached = _affine_swiglu_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    values = metile.Buffer.empty((input_features,), dtype=numpy_dtype)
    packed_shape = (output_features, input_features * bits // 32)
    parameter_shape = (output_features, input_features // group_size)
    gate_weight = metile.Buffer.empty(packed_shape, dtype=np.uint32)
    gate_scales = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    gate_biases = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    up_weight = metile.Buffer.empty(packed_shape, dtype=np.uint32)
    up_scales = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    up_biases = metile.Buffer.empty(parameter_shape, dtype=numpy_dtype)
    output = metile.Buffer.empty((output_features,), dtype=numpy_dtype)
    compiled = affine_swiglu_scratch_qmv.get_compiled(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        output,
        input_features,
        output_features,
        GROUP_SIZE=group_size,
        BITS=bits,
        BLOCK=block,
        OUTPUTS_PER_SIMDGROUP=outputs_per_simdgroup,
        HALF_DECODE=decode_dtype == "f16",
    )
    source = _mlx_kernel_body(_specialize_mlx_source(compiled.msl_source, dtype))
    source = _replace_identifier(source, "K", f"{input_features}")
    source = _replace_identifier(source, "N", f"{output_features}")
    operation = mx.fast.metal_kernel(
        name=f"metile_affine_swiglu_scratch_{stable_digest(kernel_key)[:16]}",
        input_names=[
            "X",
            "GateW",
            "GateScales",
            "GateBiases",
            "UpW",
            "UpScales",
            "UpBiases",
        ],
        output_names=["Out"],
        source=source,
    )
    kernel = _MLXAffineSwiGLUKernel(
        operation,
        output_features,
        block,
        outputs_per_simdgroup,
        compiled.description_bits,
    )
    _affine_swiglu_kernel_cache[kernel_key] = kernel
    return kernel


def mlx_affine_qmv(
    values,
    weight,
    scales,
    biases,
    *,
    group_size=64,
    bits=4,
    block=32,
    outputs_per_simdgroup=1,
    decode_dtype="f32",
):
    """Execute one generated affine packed-weight QMV inside an MLX graph."""
    kernel = _compile_affine_qmv(
        values.shape[-1],
        weight.shape[0],
        values.dtype,
        group_size,
        bits,
        block,
        outputs_per_simdgroup,
        decode_dtype,
    )
    return kernel(values, weight, scales, biases)


def mlx_affine_swiglu_qmv(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    *,
    group_size=64,
    bits=4,
    block=32,
    outputs_per_simdgroup=1,
    decode_dtype="f32",
):
    """Execute fused affine gate/up QMV projections with a SwiGLU epilogue."""
    kernel = _compile_affine_swiglu_qmv(
        values.shape[-1],
        gate_weight.shape[0],
        values.dtype,
        group_size,
        bits,
        block,
        outputs_per_simdgroup,
        decode_dtype,
    )
    return kernel(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )


def repack_mlx_affine_weight(weight, scales, biases):
    """Repack MLX output-major affine uint4 weights into K-major NAX layout."""
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


def mlx_affine_qmv_nax(
    values,
    packed,
    scales,
    biases,
    *,
    output_features,
    group_size=64,
    bits=4,
    block=32,
):
    """Execute an M5-native affine QMV over pre-repacked MLX weights."""
    kernel = _compile_nax_affine_qmv(
        values.shape[-1],
        output_features,
        values.dtype,
        group_size,
        bits,
        block,
    )
    return kernel(values, packed, scales, biases)


def mlx_affine_swiglu_qmv_nax(
    values,
    gate_packed,
    gate_scales,
    gate_biases,
    up_packed,
    up_scales,
    up_biases,
    *,
    output_features,
    group_size=64,
    bits=4,
    block=32,
):
    """Execute fused M5-native affine gate/up QMV with a SwiGLU epilogue."""
    kernel = _compile_nax_affine_swiglu_qmv(
        values.shape[-1],
        output_features,
        values.dtype,
        group_size,
        bits,
        block,
    )
    return kernel(
        values,
        gate_packed,
        gate_scales,
        gate_biases,
        up_packed,
        up_scales,
        up_biases,
    )


def _native_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    group_size,
    bits,
):
    import mlx.core as mx

    projected = mx.quantized_matmul(
        values,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    return projected + residual


def _mlx_compiled_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    group_size,
    bits,
):
    import mlx.core as mx

    global _compiled_affine_residual_qmv
    if _compiled_affine_residual_qmv is None:
        _compiled_affine_residual_qmv = mx.compile(_native_affine_residual_qmv)
    return _compiled_affine_residual_qmv(
        values,
        weight,
        scales,
        biases,
        residual,
        group_size,
        bits,
    )


def _native_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    import mlx.core as mx
    import mlx.nn as nn

    gate = mx.quantized_matmul(
        values,
        gate_weight,
        scales=gate_scales,
        biases=gate_biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    up = mx.quantized_matmul(
        values,
        up_weight,
        scales=up_scales,
        biases=up_biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    return nn.silu(gate) * up


def _mlx_compiled_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    import mlx.core as mx

    global _compiled_affine_swiglu
    if _compiled_affine_swiglu is None:
        _compiled_affine_swiglu = mx.compile(_native_affine_swiglu)
    return _compiled_affine_swiglu(
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )


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


def _make_affine_swiglu_executor(
    config,
    sample_values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    output_features = gate_weight.shape[0]
    if config.algorithm == "mlx":
        return (
            lambda values: _native_affine_swiglu(
                values,
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
                group_size,
                bits,
            ),
            0,
        )
    if config.algorithm == "mlx_compiled":
        return (
            lambda values: _mlx_compiled_affine_swiglu(
                values,
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
                group_size,
                bits,
            ),
            compressed_description_bits(inspect.getsource(_native_affine_swiglu)),
        )
    if config.implementation == "scalar":
        kernel = _compile_affine_swiglu_qmv(
            sample_values.shape[-1],
            output_features,
            sample_values.dtype,
            group_size,
            bits,
            config.block,
            config.outputs_per_simdgroup,
            config.decode_dtype,
        )
        return (
            lambda values: kernel(
                values,
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
            ),
            kernel.description_bits,
        )
    if config.implementation == "scratch":
        kernel = _compile_affine_swiglu_scratch_qmv(
            sample_values.shape[-1],
            output_features,
            sample_values.dtype,
            group_size,
            bits,
            config.block,
            config.outputs_per_simdgroup,
            config.decode_dtype,
        )
        return (
            lambda values: kernel(
                values,
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
            ),
            kernel.description_bits,
        )
    if config.implementation == "matmul":
        # Imported here because mlx_affine imports this module for weight repacking.
        import mlx.nn as nn

        from metile.backends.mlx_affine import MLXAffineWeight, mlx_affine_matmul

        gate = MLXAffineWeight.from_mlx(
            gate_weight, gate_scales, gate_biases, group_size=group_size, bits=bits
        )
        up = MLXAffineWeight.from_mlx(
            up_weight, up_scales, up_biases, group_size=group_size, bits=bits
        )
        return (
            lambda values: nn.silu(mlx_affine_matmul(values, gate)) * mlx_affine_matmul(values, up),
            compressed_description_bits(inspect.getsource(mlx_affine_matmul)),
        )
    if config.implementation in {"nax", "nax_scratch"}:
        if sample_values.size != sample_values.shape[-1]:
            raise ValueError("native affine SwiGLU schedules require one decode row")
        repacked = _repacked_affine_pair(
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
        )
        kernel = _compile_nax_affine_swiglu_qmv(
            sample_values.shape[-1],
            output_features,
            sample_values.dtype,
            group_size,
            bits,
            config.block,
            "scratch" if config.implementation == "nax_scratch" else "parallel",
        )
        return (
            lambda values: kernel(values, *repacked),
            kernel.description_bits,
        )
    raise ValueError(f"unknown affine SwiGLU implementation: {config.implementation}")


def _affine_swiglu_dispatch(
    config,
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    executor, description_bits = _make_affine_swiglu_executor(
        config,
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )
    return lambda: executor(values), description_bits


def _make_affine_residual_executor(
    config,
    sample_values,
    weight,
    scales,
    biases,
    group_size,
    bits,
):
    if config.algorithm == "mlx":
        return (
            lambda values, residual: _native_affine_residual_qmv(
                values,
                weight,
                scales,
                biases,
                residual,
                group_size,
                bits,
            ),
            0,
        )
    if config.algorithm == "mlx_compiled":
        return (
            lambda values, residual: _mlx_compiled_affine_residual_qmv(
                values,
                weight,
                scales,
                biases,
                residual,
                group_size,
                bits,
            ),
            compressed_description_bits(inspect.getsource(_native_affine_residual_qmv)),
        )
    if config.algorithm == "metile_matmul":
        # Same reasoning as the SwiGLU matmul candidate: the generated residual QMV below
        # is single-row, so the down projection had nothing to offer above one row.
        # from_mlx rejects anything but 4-bit group-64, and the tuner drops candidates it
        # cannot build, so this simply does not compete at other formats.
        from metile.backends.mlx_affine import MLXAffineWeight, mlx_affine_matmul

        projection = MLXAffineWeight.from_mlx(
            weight, scales, biases, group_size=group_size, bits=bits
        )
        return (
            lambda values, residual: mlx_affine_matmul(values, projection) + residual,
            compressed_description_bits(inspect.getsource(mlx_affine_matmul)),
        )
    if sample_values.size != sample_values.shape[-1]:
        raise ValueError("generated affine residual QMV requires one decode row")
    kernel = _compile_affine_qmv(
        sample_values.shape[-1],
        weight.shape[0],
        sample_values.dtype,
        group_size,
        bits,
        config.block,
        config.outputs_per_simdgroup,
        config.decode_dtype,
        True,
    )
    return (
        lambda values, residual: kernel(values, weight, scales, biases, residual),
        kernel.description_bits,
    )


def _affine_residual_dispatch(
    config,
    values,
    weight,
    scales,
    biases,
    residual,
    group_size,
    bits,
):
    executor, description_bits = _make_affine_residual_executor(
        config,
        values,
        weight,
        scales,
        biases,
        group_size,
        bits,
    )
    return lambda: executor(values, residual), description_bits


def _choose_affine_swiglu_config(results):
    native = next(result for result in results if result[2].algorithm == "mlx")
    eligible = [
        result
        for result in results
        if (
            result[2].algorithm == "mlx_compiled"
            and result[0] < native[0] * (1.0 - _COMPILED_SWITCH_MARGIN)
        )
        or (result[2].algorithm == "metile" and result[0] < native[0] * (1.0 - _SWITCH_MARGIN))
    ]
    return choose_mdl_tie(eligible) if eligible else native[2]


def _choose_affine_residual_config(results):
    native = next(result for result in results if result[2].algorithm == "mlx")
    eligible = [
        result
        for result in results
        if (
            result[2].algorithm == "mlx_compiled"
            and result[0] < native[0] * (1.0 - _COMPILED_SWITCH_MARGIN)
        )
        or (
            result[2].algorithm == "metile"
            and result[0] < native[0] * (1.0 - _RESIDUAL_SWITCH_MARGIN)
        )
    ]
    return choose_mdl_tie(eligible) if eligible else native[2]


def _affine_swiglu_compatible(result, reference):
    import mlx.core as mx

    if result.shape != reference.shape or result.dtype != reference.dtype:
        return False
    result_f32 = result.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    difference = result_f32 - reference_f32
    reference_rms = mx.sqrt(mx.mean(reference_f32 * reference_f32))
    difference_rms = mx.sqrt(mx.mean(difference * difference))
    reference_peak = mx.max(mx.abs(reference_f32))
    difference_peak = mx.max(mx.abs(difference))
    finite = mx.all(mx.isfinite(result_f32))
    normalized_rms = difference_rms / mx.maximum(reference_rms, mx.array(1e-6))
    normalized_peak = difference_peak / mx.maximum(reference_peak, mx.array(1e-6))
    mx.eval(finite, normalized_rms, normalized_peak)
    return bool(finite.item()) and normalized_rms.item() <= 0.005 and normalized_peak.item() <= 0.01


def _tune_affine_dispatches(configs, make_dispatch, choose_config):
    import mlx.core as mx

    kernels = []
    for config in configs:
        try:
            dispatch, description_bits = make_dispatch(config)
            result = dispatch()
            mx.eval(result)
        except (RuntimeError, TypeError, ValueError):
            if config.algorithm == "mlx":
                raise
            continue
        kernels.append((config, dispatch, description_bits))

    native_dispatch = next(dispatch for config, dispatch, _ in kernels if config.algorithm == "mlx")
    reference = native_dispatch()
    mx.eval(reference)
    compatible = []
    for config, dispatch, description_bits in kernels:
        result = dispatch()
        mx.eval(result)
        if config.algorithm == "mlx" or _affine_swiglu_compatible(result, reference):
            compatible.append((config, dispatch, description_bits))
    kernels = compatible

    # One eval per batch rather than per dispatch: the blocking round trip costs roughly
    # 200 us whatever the kernel does, so evaluating per dispatch adds that constant to
    # every candidate and compresses their ratios toward 1.0, letting the switch margins
    # admit a kernel that is actually slower than native MLX.
    batch = calibrate_tournament_batch(native_dispatch)
    samples = {config: [] for config, _, _ in kernels}
    for round_index in range(11):
        ordered = kernels[round_index % len(kernels) :] + kernels[: round_index % len(kernels)]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval([dispatch() for _ in range(batch)])
            samples[config].append((time.perf_counter_ns() - start) * 1e-9 / batch)

    provisional = {
        config: statistics.median(config_samples) for config, config_samples in samples.items()
    }
    configs = tuple(config for config, _, _ in kernels)
    native = next(config for config in configs if config.algorithm == "mlx")
    alternatives = tuple(config for config in configs if config.algorithm != "mlx")
    if not alternatives:
        return native
    fastest_alternative = min(alternatives, key=provisional.__getitem__)
    best = min(provisional.values())
    finalists = {
        config
        for config, latency in provisional.items()
        if latency <= best * 1.10 or config in {native, fastest_alternative}
    }
    finalist_kernels = [candidate for candidate in kernels if candidate[0] in finalists]
    samples = {config: [] for config in finalists}
    for round_index in range(31):
        ordered = (
            finalist_kernels[round_index % len(finalist_kernels) :]
            + finalist_kernels[: round_index % len(finalist_kernels)]
        )
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval([dispatch() for _ in range(batch)])
            samples[config].append((time.perf_counter_ns() - start) * 1e-9 / batch)
    return choose_config(
        [
            (statistics.median(samples[config]), description_bits, config)
            for config, _, description_bits in finalist_kernels
        ]
    )


def _tune_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    group_size,
    bits,
):
    return _tune_affine_dispatches(
        _affine_swiglu_configs(values.dtype, bits),
        lambda config: _affine_swiglu_dispatch(
            config,
            values,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            group_size,
            bits,
        ),
        _choose_affine_swiglu_config,
    )


def _tune_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    group_size,
    bits,
):
    return _tune_affine_dispatches(
        _AFFINE_RESIDUAL_CONFIGS,
        lambda config: _affine_residual_dispatch(
            config,
            values,
            weight,
            scales,
            biases,
            residual,
            group_size,
            bits,
        ),
        _choose_affine_residual_config,
    )


def mlx_affine_swiglu_backend_signature():
    """Return the code/config identity that can change affine SwiGLU dispatch."""
    return stable_digest(
        {
            "compiled": inspect.getsource(_mlx_compiled_affine_swiglu),
            "compiled_switch_margin": _COMPILED_SWITCH_MARGIN,
            "configs": [vars(config) for config in _AFFINE_SWIGLU_CONFIGS],
            "config_filter": inspect.getsource(_affine_swiglu_configs),
            "dispatch": inspect.getsource(mlx_affine_swiglu),
            "executor": inspect.getsource(mlx_affine_swiglu_executor),
            "fidelity": inspect.getsource(_affine_swiglu_compatible),
            "lowering": inspect.getsource(lower_affine_swiglu_qmv),
            "native": inspect.getsource(_native_affine_swiglu),
            "nax": inspect.getsource(_compile_nax_affine_swiglu_qmv),
            "residual_compiled": inspect.getsource(_mlx_compiled_affine_residual_qmv),
            "residual_configs": [vars(config) for config in _AFFINE_RESIDUAL_CONFIGS],
            "residual_dispatch": inspect.getsource(_affine_residual_dispatch),
            "residual_kernel": inspect.getsource(affine_residual_qmv.fn),
            "residual_margin": _RESIDUAL_SWITCH_MARGIN,
            "residual_native": inspect.getsource(_native_affine_residual_qmv),
            "residual_selection": inspect.getsource(_choose_affine_residual_config),
            "residual_tune": inspect.getsource(_tune_affine_residual_qmv),
            "residual_tuner": _AFFINE_RESIDUAL_TUNER_VERSION,
            "runtime_executor": inspect.getsource(mlx_affine_mlp_executor),
            "scalar": inspect.getsource(affine_swiglu_qmv.fn),
            "scalar_compile": inspect.getsource(_compile_affine_swiglu_qmv),
            "scratch": inspect.getsource(affine_swiglu_scratch_qmv.fn),
            "scratch_compile": inspect.getsource(_compile_affine_swiglu_scratch_qmv),
            "selection": inspect.getsource(_choose_affine_swiglu_config),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_affine_swiglu),
            "tuning_measure": inspect.getsource(_tune_affine_dispatches),
            "tuner": _AFFINE_SWIGLU_TUNER_VERSION,
        }
    )


def _affine_swiglu_persistent_key(values, gate_weight, group_size, bits):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "bits": bits,
            "configs": [vars(config) for config in _affine_swiglu_configs(values.dtype, bits)],
            "dtype": str(values.dtype),
            "group_size": group_size,
            "input_features": values.shape[-1],
            "mlx": mx.__version__,
            "output_features": gate_weight.shape[0],
            "rows": _token_bucket(values.size // values.shape[-1]),
            "source": mlx_affine_swiglu_backend_signature(),
            "compiled_switch_margin": _COMPILED_SWITCH_MARGIN,
            "switch_margin": _SWITCH_MARGIN,
            "tuner": _AFFINE_SWIGLU_TUNER_VERSION,
        }
    )


def _read_affine_swiglu_config(key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_affine_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in _AFFINE_SWIGLU_CONFIGS
            if config.algorithm == payload.get("algorithm")
            and config.implementation == payload.get("implementation", "")
            and config.block == payload.get("block", 0)
            and config.outputs_per_simdgroup == payload.get("outputs_per_simdgroup", 1)
            and config.decode_dtype == payload.get("decode_dtype", "f32")
        ),
        None,
    )


def _write_affine_swiglu_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_affine_cache_path, {})
    payload[key] = {
        "algorithm": config.algorithm,
        "block": config.block,
        "decode_dtype": config.decode_dtype,
        "implementation": config.implementation,
        "outputs_per_simdgroup": config.outputs_per_simdgroup,
    }
    atomic_write_json(_affine_cache_path, payload)


def _affine_residual_persistent_key(values, weight, group_size, bits):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "bits": bits,
            "configs": [vars(config) for config in _AFFINE_RESIDUAL_CONFIGS],
            "dtype": str(values.dtype),
            "group_size": group_size,
            "input_features": values.shape[-1],
            "mlx": mx.__version__,
            "output_features": weight.shape[0],
            "source": mlx_affine_swiglu_backend_signature(),
            "switch_margin": _RESIDUAL_SWITCH_MARGIN,
            "tuner": _AFFINE_RESIDUAL_TUNER_VERSION,
        }
    )


def _read_affine_residual_config(key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_affine_residual_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in _AFFINE_RESIDUAL_CONFIGS
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
            and config.outputs_per_simdgroup == payload.get("outputs_per_simdgroup", 1)
            and config.decode_dtype == payload.get("decode_dtype", "f32")
        ),
        None,
    )


def _write_affine_residual_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_affine_residual_cache_path, {})
    payload[key] = {
        "algorithm": config.algorithm,
        "block": config.block,
        "decode_dtype": config.decode_dtype,
        "outputs_per_simdgroup": config.outputs_per_simdgroup,
    }
    atomic_write_json(_affine_residual_cache_path, payload)


def mlx_affine_residual_qmv(
    values,
    weight,
    scales,
    biases,
    residual,
    *,
    group_size=64,
    bits=4,
    autotune=True,
):
    """Dispatch affine QMV plus residual to the fastest compatible kernel."""
    if group_size != 64 or bits != 4:
        raise ValueError("affine residual QMV requires 4-bit weights with group size 64")
    if biases is None:
        raise ValueError("affine residual QMV requires affine quantization biases")
    input_features = values.shape[-1]
    output_features = weight.shape[0]
    parameter_shape = (output_features, input_features // group_size)
    if (
        weight.ndim != 2
        or weight.shape[1] * 32 // bits != input_features
        or scales.shape != parameter_shape
        or biases.shape != parameter_shape
    ):
        raise ValueError("affine residual QMV received incompatible quantization parameters")
    if scales.dtype != values.dtype or biases.dtype != values.dtype:
        raise ValueError("affine residual QMV parameters must match the input dtype")
    expected_shape = (*values.shape[:-1], output_features)
    if residual.shape != expected_shape or residual.dtype != values.dtype:
        raise ValueError("residual must match the affine QMV output shape and dtype")
    if values.size != values.shape[-1]:
        return _native_affine_residual_qmv(
            values,
            weight,
            scales,
            biases,
            residual,
            group_size,
            bits,
        )
    schedule_key = (
        values.shape[-1],
        weight.shape[0],
        str(values.dtype),
        group_size,
        bits,
    )
    selected = _affine_residual_schedule_cache.get(schedule_key)
    if selected is None:
        with _affine_cache_lock:
            selected = _affine_residual_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _affine_residual_persistent_key(
                    values,
                    weight,
                    group_size,
                    bits,
                )
                selected = _read_affine_residual_config(persistent_key)
            if selected is None:
                selected = (
                    _tune_affine_residual_qmv(
                        values,
                        weight,
                        scales,
                        biases,
                        residual,
                        group_size,
                        bits,
                    )
                    if autotune
                    else MLXAffineResidualConfig("metile", 64)
                )
                _write_affine_residual_config(persistent_key, selected)
            _affine_residual_schedule_cache[schedule_key] = selected
    dispatch, _ = _affine_residual_dispatch(
        selected,
        values,
        weight,
        scales,
        biases,
        residual,
        group_size,
        bits,
    )
    return dispatch()


def mlx_affine_residual_qmv_dispatches():
    """Return in-process affine residual-QMV schedule decisions."""
    return tuple(
        {
            "input_features": key[0],
            "output_features": key[1],
            "dtype": key[2],
            "group_size": key[3],
            "bits": key[4],
            "algorithm": config.algorithm,
            "block": config.block,
            "outputs_per_simdgroup": config.outputs_per_simdgroup,
            "decode_dtype": config.decode_dtype,
        }
        for key, config in sorted(_affine_residual_schedule_cache.items())
    )


def mlx_affine_swiglu(
    values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    *,
    group_size=64,
    bits=4,
    autotune=True,
):
    """Dispatch affine SwiGLU to eager/compiled MLX, scalar, or M5 NAX kernels."""
    if not ((bits == 4 and group_size == 64) or (bits == 8 and group_size in {32, 64, 128})):
        raise ValueError("affine SwiGLU requires 4-bit group-64 or 8-bit group-32/64/128 weights")
    if gate_biases is None or up_biases is None:
        raise ValueError("affine SwiGLU requires affine quantization biases")
    if (
        values.dtype != gate_scales.dtype
        or values.dtype != gate_biases.dtype
        or gate_weight.shape != up_weight.shape
        or gate_scales.shape != up_scales.shape
        or gate_biases.shape != up_biases.shape
    ):
        raise ValueError("affine SwiGLU requires matching gate/up weights and parameters")
    schedule_key = (
        _token_bucket(values.size // values.shape[-1]),
        values.shape[-1],
        gate_weight.shape[0],
        str(values.dtype),
        group_size,
        bits,
    )
    selected = _affine_swiglu_schedule_cache.get(schedule_key)
    if selected is None:
        with _affine_cache_lock:
            selected = _affine_swiglu_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _affine_swiglu_persistent_key(
                    values, gate_weight, group_size, bits
                )
                selected = _read_affine_swiglu_config(persistent_key)
            if selected is None:
                selected = (
                    _tune_affine_swiglu(
                        values,
                        gate_weight,
                        gate_scales,
                        gate_biases,
                        up_weight,
                        up_scales,
                        up_biases,
                        group_size,
                        bits,
                    )
                    if autotune
                    else MLXAffineSwiGLUConfig("metile", "scalar", 32)
                )
                _write_affine_swiglu_config(persistent_key, selected)
            _affine_swiglu_schedule_cache[schedule_key] = selected
    if selected.implementation not in {"nax", "nax_scratch"}:
        _discard_repacked_affine_pair(gate_weight, up_weight)
    dispatch, _ = _affine_swiglu_dispatch(
        selected,
        values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )
    return dispatch()


def mlx_affine_swiglu_executor(
    sample_values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    *,
    group_size=64,
    bits=4,
):
    """Autotune once and return the selected shape-specialized SwiGLU callable."""
    mlx_affine_swiglu(
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
    )
    schedule_key = (
        _token_bucket(sample_values.size // sample_values.shape[-1]),
        sample_values.shape[-1],
        gate_weight.shape[0],
        str(sample_values.dtype),
        group_size,
        bits,
    )
    executor, _ = _make_affine_swiglu_executor(
        _affine_swiglu_schedule_cache[schedule_key],
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size,
        bits,
    )
    return executor


def mlx_affine_mlp_executor(
    sample_values,
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    down_weight,
    down_scales,
    down_biases,
    sample_residual,
    *,
    group_size=64,
    bits=4,
):
    """Autotune once and return a shape-specialized affine MLP callable."""
    swiglu_executor = mlx_affine_swiglu_executor(
        sample_values,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
    )
    hidden = swiglu_executor(sample_values)
    residual_key = (
        hidden.shape[-1],
        down_weight.shape[0],
        str(hidden.dtype),
        group_size,
        bits,
    )
    if residual_key not in _affine_residual_schedule_cache:
        import mlx.core as mx

        mx.eval(hidden)
    mlx_affine_residual_qmv(
        hidden,
        down_weight,
        down_scales,
        down_biases,
        sample_residual,
        group_size=group_size,
        bits=bits,
    )
    residual_executor, _ = _make_affine_residual_executor(
        _affine_residual_schedule_cache[residual_key],
        hidden,
        down_weight,
        down_scales,
        down_biases,
        group_size,
        bits,
    )

    def execute(values, residual):
        return residual_executor(swiglu_executor(values), residual)

    return execute


def mlx_affine_swiglu_dispatches():
    """Return in-process affine SwiGLU schedule decisions."""
    return tuple(
        {
            "row_bucket": key[0],
            "input_features": key[1],
            "output_features": key[2],
            "dtype": key[3],
            "group_size": key[4],
            "bits": key[5],
            "algorithm": config.algorithm,
            "decode_dtype": config.decode_dtype,
            "implementation": config.implementation,
            "block": config.block,
            "outputs_per_simdgroup": config.outputs_per_simdgroup,
        }
        for key, config in sorted(_affine_swiglu_schedule_cache.items())
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
