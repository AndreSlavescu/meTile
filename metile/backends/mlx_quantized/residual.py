"""Affine-quantized residual kernels."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import numpy as np

import metile
from metile.backends.mlx import (
    _mlx_compiler_dtype,
    _mlx_kernel_body,
    _replace_identifier,
    _specialize_mlx_source,
)
from metile.backends.mlx_quantized.common import (
    _COMPILED_SWITCH_MARGIN,
    _RESIDUAL_SWITCH_MARGIN,
    _affine_qmv_kernel_cache,
    _affine_residual_cache_path,
    _affine_residual_schedule_cache,
    _compiled_affine_residual_qmv,
)
from metile.backends.mlx_quantized.qmv import (
    _MLXAffineQMVKernel,
)
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
)
from metile.kernels.affine_qmv import (
    affine_qmv,
    affine_residual_qmv,
)
from metile.runtime.cache import atomic_write_json, read_json, stable_digest


@dataclass(frozen=True)
class MLXAffineResidualConfig:
    algorithm: str
    block: int = 0
    outputs_per_simdgroup: int = 1
    decode_dtype: str = "f32"


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
