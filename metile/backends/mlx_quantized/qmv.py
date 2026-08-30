"""Affine-quantized matrix-vector kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metile.backends.mlx import (
    _mlx_dtype_to_numpy,
    _mlx_kernel_body,
)
from metile.backends.mlx_quantized.common import (
    _nax_affine_qmv_kernel_cache,
    _tensor_kernel_header,
)
from metile.codegen.msl_emitter import emit
from metile.compiler.affine_quantized import lower_affine_qmv
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import (
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.cache import stable_digest


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
