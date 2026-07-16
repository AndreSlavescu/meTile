from __future__ import annotations

import os
import statistics
import threading
from dataclasses import dataclass

import numpy as np

from metile.codegen.msl_emitter import emit
from metile.compiler.block_scaled import lower_block_scaled_matmul
from metile.compiler.schedule_search import optimize_tile_schedules
from metile.frontend.kernel import CompiledKernel, FastDispatcher
from metile.runtime.buffer import MtileBuffer
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest
from metile.runtime.metal_device import MetalDevice

_GROUP_SIZE = 32
_compiled_block_scaled: dict[tuple, CompiledKernel] = {}
_block_scaled_config_cache: dict[tuple, tuple[int, int]] = {}
_block_scaled_cache_lock = threading.RLock()
_block_scaled_cache_path = cache_root() / "block-scaled-autotune-v1.json"
_BLOCK_SCALED_TILES = ((64, 64), (64, 128), (128, 64), (128, 128))


def _decode_e8m0(scales: np.ndarray) -> np.ndarray:
    exponents = scales.astype(np.int16) - 127
    return np.ldexp(np.ones(scales.shape, dtype=np.float32), exponents)


def _encode_e8m0(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exponents = np.full(scales.shape, -127, dtype=np.int16)
    nonzero = scales > 0
    exponents[nonzero] = np.rint(np.log2(scales[nonzero])).astype(np.int16)
    np.clip(exponents, -127, 127, out=exponents)
    encoded = (exponents + 127).astype(np.uint8)
    return encoded, _decode_e8m0(encoded)


def _encode_e2m1(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    bits = np.select(
        [
            magnitude > 5.0,
            magnitude >= 3.5,
            magnitude > 2.5,
            magnitude >= 1.75,
            magnitude > 1.25,
            magnitude >= 0.75,
            magnitude > 0.25,
        ],
        [7, 6, 5, 4, 3, 2, 1],
        default=0,
    ).astype(np.uint8)
    return bits | (np.signbit(values).astype(np.uint8) << 3)


def _decode_e2m1(bits: np.ndarray) -> np.ndarray:
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    values = magnitudes[bits & 7]
    return np.where(bits & 8, -values, values)


def _encode_e4m3(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    float_bits = values.view(np.uint32)
    sign = float_bits & np.uint32(0x80000000)
    magnitude_bits = float_bits ^ sign
    output = np.empty(values.shape, dtype=np.uint8)

    saturated = magnitude_bits >= np.uint32(543 << 21)
    denormal = (~saturated) & (magnitude_bits < np.uint32(121 << 23))
    normal = ~(saturated | denormal)
    output[saturated] = 0x7E

    denorm_mask = np.uint32(141 << 23)
    if np.any(denormal):
        adjusted = magnitude_bits[denormal].view(np.float32) + denorm_mask.view(np.float32)
        output[denormal] = (adjusted.view(np.uint32) - denorm_mask).astype(np.uint8)
    if np.any(normal):
        normal_bits = magnitude_bits[normal].astype(np.uint64)
        mantissa_odd = (normal_bits >> 20) & 1
        bias = ((7 - 127) << 23) & 0xFFFFFFFF
        rounded = (normal_bits + bias + 0x7FFFF + mantissa_odd) & 0xFFFFFFFF
        output[normal] = (rounded >> 20).astype(np.uint8)

    output |= (sign >> 24).astype(np.uint8)
    return output


def _decode_e4m3(bits: np.ndarray) -> np.ndarray:
    raw = ((bits & 127).astype(np.uint16) << 7).view(np.float16)
    magnitude = raw.astype(np.float32) * 256.0
    return np.where(bits & 128, -magnitude, magnitude)


@dataclass
class BlockScaledWeight:
    """A GPU-resident MX block-scaled KxN weight matrix."""

    values: MtileBuffer
    scales: MtileBuffer
    shape: tuple[int, int]
    format: str

    @property
    def bits(self) -> int:
        return 4 if self.format == "mxfp4" else 8

    @classmethod
    def quantize(cls, weight: np.ndarray, format: str = "mxfp4") -> BlockScaledWeight:
        if format not in {"mxfp4", "mxfp8"}:
            raise ValueError("format must be 'mxfp4' or 'mxfp8'")
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("block-scaled weights must be a KxN matrix")
        k, n = weight.shape
        if k % _GROUP_SIZE or n % 64:
            raise ValueError("K must be divisible by 32 and N by 64")

        groups = weight.reshape(k // _GROUP_SIZE, _GROUP_SIZE, n)
        maximum = np.max(np.abs(groups), axis=1)
        limit = 6.0 if format == "mxfp4" else 448.0
        encoded_scales, decoded_scales = _encode_e8m0(maximum / limit)
        normalized = groups / decoded_scales[:, None, :]
        normalized = normalized.reshape(k, n)

        if format == "mxfp4":
            quantized = _encode_e2m1(normalized).reshape(-1)
            packed = quantized[0::2] | (quantized[1::2] << 4)
        else:
            packed = _encode_e4m3(normalized).reshape(-1)

        return cls(
            values=MtileBuffer(data=np.ascontiguousarray(packed)),
            scales=MtileBuffer(data=np.ascontiguousarray(encoded_scales)),
            shape=(k, n),
            format=format,
        )

    def dequantize(self) -> np.ndarray:
        k, n = self.shape
        packed = self.values.numpy().reshape(-1)
        if self.bits == 4:
            values = np.empty(k * n, dtype=np.uint8)
            values[0::2] = packed & 15
            values[1::2] = packed >> 4
            decoded = _decode_e2m1(values)
        else:
            decoded = _decode_e4m3(packed)
        scales = _decode_e8m0(self.scales.numpy()).reshape(k // _GROUP_SIZE, 1, n)
        return (decoded.reshape(k // _GROUP_SIZE, _GROUP_SIZE, n) * scales).reshape(k, n)


def prepare_block_scaled_matmul(
    activations: MtileBuffer,
    weight: BlockScaledWeight,
    output: MtileBuffer,
) -> FastDispatcher:
    """Compile and bind the fused dequantize-plus-MPP matmul fast path."""
    if not isinstance(activations, MtileBuffer) or not isinstance(output, MtileBuffer):
        raise TypeError("activations and output must be metile.Buffer instances")
    if len(activations.shape) != 2 or len(output.shape) != 2:
        raise ValueError("activations and output must retain their two-dimensional shapes")
    m, k = activations.shape
    weight_k, n = weight.shape
    if k != weight_k or output.shape != (m, n):
        raise ValueError("expected A[M,K], weight[K,N], and output[M,N]")
    if m % 64 or n % 64 or k % 32:
        raise ValueError("fast block-scaled matmul requires M/N multiples of 64 and K of 32")

    dev = MetalDevice.get()
    if not dev.supports_tensor_ops:
        raise RuntimeError("block-scaled matmul requires Metal 4 tensor operations")
    tuning_key = (m, n, k, weight.bits, dev.name, dev.metal_compiler_version)
    candidates = [
        candidate
        for candidate in _BLOCK_SCALED_TILES
        if m % candidate[0] == 0 and n % candidate[1] == 0
    ]
    with _block_scaled_cache_lock:
        tile = _block_scaled_config_cache.get(tuning_key)
        persistent_key = stable_digest(tuning_key)
        if tile is None and os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
            stored = read_json(_block_scaled_cache_path, {}).get(persistent_key)
            if isinstance(stored, list) and tuple(stored) in _BLOCK_SCALED_TILES:
                tile = tuple(stored)
        if tile not in candidates:
            tile = _tune_block_scaled_tile(
                activations,
                weight,
                output,
                candidates,
                tuning_key,
            )
            if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
                payload = read_json(_block_scaled_cache_path, {})
                payload[persistent_key] = list(tile)
                atomic_write_json(_block_scaled_cache_path, payload)
        _block_scaled_config_cache[tuning_key] = tile
    return _prepare_block_scaled_dispatch(activations, weight, output, *tile)


def _prepare_block_scaled_dispatch(
    activations: MtileBuffer,
    weight: BlockScaledWeight,
    output: MtileBuffer,
    block_m: int,
    block_n: int,
) -> FastDispatcher:
    m, k = activations.shape
    n = weight.shape[1]
    dev = MetalDevice.get()
    cache_key = (
        m,
        n,
        k,
        weight.bits,
        block_m,
        block_n,
        dev.name,
        dev.metal_compiler_version,
    )
    with _block_scaled_cache_lock:
        compiled = _compiled_block_scaled.get(cache_key)
        if compiled is None:
            function_name = f"mtile_bsmm_{weight.format}_{m}_{n}_{k}_{block_m}_{block_n}"
            metal_ir = optimize_tile_schedules(
                lower_block_scaled_matmul(
                    function_name,
                    m,
                    n,
                    k,
                    weight.bits,
                    block_m=block_m,
                    block_n=block_n,
                )
            )
            source = emit(metal_ir)
            pipeline, _ = dev.compile_msl_precompiled(source, function_name, metal_std="metal4.0")
            compiled = CompiledKernel(
                pipeline=pipeline,
                msl_source=source,
                func_name=function_name,
                threadgroup_size=metal_ir.threadgroup_size,
                is_gemm=True,
            )
            _compiled_block_scaled[cache_key] = compiled

    buffers = [
        activations.metal_buffer,
        weight.values.metal_buffer,
        weight.scales.metal_buffer,
        output.metal_buffer,
    ]
    grid = (m // block_m, n // block_n)
    return FastDispatcher(compiled, buffers, grid, dev)


def _tune_block_scaled_tile(
    activations: MtileBuffer,
    weight: BlockScaledWeight,
    output: MtileBuffer,
    candidates: list[tuple[int, int]],
    tuning_key: tuple,
) -> tuple[int, int]:
    if not candidates:
        raise ValueError("no aligned block-scaled tile is available for this shape")
    dev = MetalDevice.get()
    results = []
    for block_m, block_n in candidates:
        dispatch = _prepare_block_scaled_dispatch(activations, weight, output, block_m, block_n)
        for _ in range(3):
            dispatch()
            dev.sync()
        samples = []
        for _ in range(7):
            dispatch()
            dev.sync()
            elapsed = dev.gpu_elapsed()
            if elapsed > 0:
                samples.append(elapsed)
        if samples:
            results.append((statistics.median(samples), block_m, block_n))
    if not results:
        raise RuntimeError(f"GPU timing failed while tuning block-scaled shape {tuning_key[:4]}")
    _, block_m, block_n = min(results)
    return block_m, block_n


def block_scaled_matmul(
    activations: MtileBuffer,
    weight: BlockScaledWeight,
    output: MtileBuffer | None = None,
) -> MtileBuffer:
    """Run a fused MXFP4/MXFP8 weight-only matmul and return its output buffer."""
    if len(activations.shape) != 2:
        raise ValueError("activations must retain their MxK shape")
    m, _ = activations.shape
    n = weight.shape[1]
    output = output or MtileBuffer.empty((m, n), dtype=np.float32)
    prepare_block_scaled_matmul(activations, weight, output)()
    return output
