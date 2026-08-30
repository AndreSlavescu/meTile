"""Affine-quantized SwiGLU kernels."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import numpy as np

import metile
from metile.backends.mlx import (
    _mlx_compiler_dtype,
    _mlx_dtype_to_numpy,
    _mlx_kernel_body,
    _replace_identifier,
    _specialize_mlx_source,
)
from metile.backends.mlx_quantized.common import (
    _COMPILED_SWITCH_MARGIN,
    _SWITCH_MARGIN,
    _affine_cache_path,
    _affine_swiglu_kernel_cache,
    _affine_swiglu_schedule_cache,
    _compiled_affine_swiglu,
    _nax_affine_swiglu_kernel_cache,
    _repacked_affine_pair,
    _tensor_kernel_header,
)
from metile.codegen.msl_emitter import emit
from metile.compiler.affine_quantized import lower_affine_swiglu_qmv
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.kernels.affine_qmv import (
    affine_swiglu_qmv,
    affine_swiglu_scratch_qmv,
)
from metile.runtime.cache import atomic_write_json, read_json, stable_digest


@dataclass(frozen=True)
class MLXAffineSwiGLUConfig:
    algorithm: str
    implementation: str = ""
    block: int = 0
    outputs_per_simdgroup: int = 1
    decode_dtype: str = "f32"


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


# mx.compile is not offered for affine SwiGLU. Measured against eager MLX, interleaved and
# batched, it is 0.938x at one row, 0.946x at two, 1.014x at four and 1.005x at eight: never
# faster than noise, and clearly slower exactly where it kept winning the tournament and
# then losing in steady state. Raising _COMPILED_SWITCH_MARGIN twice did not stop that, so
# the candidate is withdrawn rather than margined against.
_AFFINE_SWIGLU_CONFIGS = tuple(
    [MLXAffineSwiGLUConfig("mlx")]
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
