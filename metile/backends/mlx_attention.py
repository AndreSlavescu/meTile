from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

import metile
from kernels.attention import ATTENTION_FLASH_CONFIGS, attention_flash_kernel
from metile.backends.mlx import (
    _choose_framework_config,
    _mlx_dtype_to_numpy,
    _mlx_kernel_body,
    _replace_identifier,
    MLXAttentionConfig,
    calibrate_tournament_batch,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_flash_kernel_cache = {}
_flash_schedule_cache = {}
_flash_lock = threading.RLock()
_flash_cache_path = cache_root() / "mlx-flash-attention-autotune-v1.json"
_FLASH_CONFIGS = tuple(
    [MLXAttentionConfig("mlx")]
    + [MLXAttentionConfig("metile", config.kwargs["BLOCK"]) for config in ATTENTION_FLASH_CONFIGS]
)


@dataclass(frozen=True)
class _MLXFlashAttentionKernel:
    operation: object
    block: int
    description_bits: int

    def __call__(self, query, key, value):
        batch, query_heads, query_length, _ = query.shape
        return self.operation(
            inputs=[query, key, value],
            grid=(batch * query_heads * query_length * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[query.shape],
            output_dtypes=[query.dtype],
        )[0]


def _compile_flash_attention(
    query_heads,
    key_value_heads,
    dimension,
    dtype,
    scale,
    causal,
    block,
):
    import mlx.core as mx

    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    key = (
        query_heads,
        key_value_heads,
        dimension,
        numpy_dtype.str,
        float(scale),
        bool(causal),
        block,
    )
    cached = _flash_kernel_cache.get(key)
    if cached is not None:
        return cached

    query = metile.Buffer.empty((query_heads * dimension,), dtype=numpy_dtype)
    key_buffer = metile.Buffer.empty((key_value_heads * dimension,), dtype=numpy_dtype)
    value = metile.Buffer.empty(key_buffer.shape, dtype=numpy_dtype)
    output = metile.Buffer.empty(query.shape, dtype=numpy_dtype)
    compiled = attention_flash_kernel.get_compiled(
        query,
        key_buffer,
        value,
        output,
        1,
        1,
        float(scale),
        D=dimension,
        Q_HEADS=query_heads,
        KV_HEADS=key_value_heads,
        CAUSAL=causal,
        BLOCK=block,
    )
    source = _mlx_kernel_body(compiled.msl_source)
    source = _replace_identifier(source, "Q_LEN", "Q_shape[2]")
    source = _replace_identifier(source, "K_LEN", "K_shape[2]")
    source = _replace_identifier(source, "scale", f"{float(scale):.12g}f")
    operation = mx.fast.metal_kernel(
        name=f"metile_flash_attention_{stable_digest(key)[:16]}",
        input_names=["Q", "K", "V"],
        output_names=["Out"],
        source=source,
    )
    kernel = _MLXFlashAttentionKernel(operation, block, compiled.description_bits)
    _flash_kernel_cache[key] = kernel
    return kernel


def _native_attention(query, key, value, scale, causal):
    import mlx.core as mx

    arguments = {"scale": scale}
    if causal:
        # MLX masks causally without materializing anything. Building the bias tensor
        # instead costs a full [queries, keys] allocation on every call, which made the
        # native reference look slower than it is and biased the tournament toward the
        # generated kernel. The two agree bitwise, including when queries < keys, where
        # both align the mask to the bottom right.
        arguments["mask"] = "causal"
    return mx.fast.scaled_dot_product_attention(query, key, value, **arguments)


def _tune_flash_attention(query, key, value, scale, causal):
    import mlx.core as mx

    kernels = []
    for config in _FLASH_CONFIGS:
        try:
            if config.algorithm == "mlx":

                def dispatch():
                    return _native_attention(query, key, value, scale, causal)

                description_bits = 0
            else:
                kernel = _compile_flash_attention(
                    query.shape[1],
                    key.shape[1],
                    query.shape[-1],
                    query.dtype,
                    scale,
                    causal,
                    config.block,
                )

                def dispatch(kernel=kernel):
                    return kernel(query, key, value)

                description_bits = kernel.description_bits
            result = dispatch()
            mx.eval(result)
        except (RuntimeError, TypeError, ValueError):
            if config.algorithm == "mlx":
                raise
            continue
        kernels.append((config, dispatch, description_bits, result))

    reference = kernels[0][1]()
    mx.eval(reference)
    tolerance = 3e-3
    compatible = [
        candidate
        for candidate in kernels
        if candidate[0].algorithm == "mlx"
        or bool(mx.allclose(candidate[3], reference, rtol=tolerance, atol=tolerance).item())
    ]
    # One eval per batch; see calibrate_tournament_batch.
    batch = calibrate_tournament_batch(compatible[0][1])
    samples = {config: [] for config, _, _, _ in compatible}
    for round_index in range(11):
        ordered = (
            compatible[round_index % len(compatible) :]
            + compatible[: round_index % len(compatible)]
        )
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval([dispatch() for _ in range(batch)])
            samples[config].append((time.perf_counter_ns() - start) * 1e-9 / batch)
    provisional = {
        config: statistics.median(config_samples) for config, config_samples in samples.items()
    }
    native = next(config for config in provisional if config.algorithm == "mlx")
    generated = tuple(config for config in provisional if config.algorithm == "metile")
    if not generated:
        return native
    fastest_generated = min(generated, key=provisional.__getitem__)
    best = min(provisional.values())
    finalists = {
        config
        for config, latency in provisional.items()
        if latency <= best * 1.10 or config in {native, fastest_generated}
    }
    finalist_kernels = [candidate for candidate in compatible if candidate[0] in finalists]
    samples = {config: [] for config in finalists}
    for round_index in range(31):
        ordered = (
            finalist_kernels[round_index % len(finalist_kernels) :]
            + finalist_kernels[: round_index % len(finalist_kernels)]
        )
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval([dispatch() for _ in range(batch)])
            samples[config].append((time.perf_counter_ns() - start) * 1e-9 / batch)
    return _choose_framework_config(
        [
            (statistics.median(samples[config]), description_bits, config)
            for config, _, description_bits, _ in finalist_kernels
        ]
    )


def _persistent_key(query, key, scale, causal):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "causal": causal,
            "configs": [vars(config) for config in _FLASH_CONFIGS],
            "dtype": str(query.dtype),
            "key_shape": tuple(key.shape),
            "mlx": mx.__version__,
            # The tournament compares against _native_attention and times candidates with
            # _tune_flash_attention, so a change to either invalidates the stored pick.
            "native": inspect.getsource(_native_attention),
            "measure": inspect.getsource(_tune_flash_attention),
            "query_shape": tuple(query.shape),
            "scale": float(scale),
            "source": inspect.getsource(attention_flash_kernel.fn),
        }
    )


def _read_config(key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_flash_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in _FLASH_CONFIGS
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
        ),
        None,
    )


def _write_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_flash_cache_path, {})
    payload[key] = {"algorithm": config.algorithm, "block": config.block}
    atomic_write_json(_flash_cache_path, payload)


def mlx_flash_attention(
    query,
    key,
    value,
    *,
    scale,
    causal=False,
    autotune=True,
):
    """Run exact discovered attention through a guarded generated/native tournament."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("flash attention expects rank-four query, key, and value arrays")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("flash attention requires matching batch and key/value shapes")
    if query.shape[1] % key.shape[1]:
        raise ValueError("key/value heads must divide query heads")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] % 32:
        raise ValueError("flash attention head dimensions must match and be a multiple of 32")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("flash attention requires matching input dtypes")
    if causal and query.shape[2] > key.shape[2]:
        raise ValueError("causal attention requires at least as many keys as queries")
    _mlx_dtype_to_numpy(query.dtype)

    schedule_key = (
        tuple(query.shape),
        tuple(key.shape),
        str(query.dtype),
        float(scale),
        bool(causal),
    )
    selected = _flash_schedule_cache.get(schedule_key)
    if selected is None:
        with _flash_lock:
            selected = _flash_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _persistent_key(query, key, scale, causal)
                selected = _read_config(persistent_key)
            if selected is None:
                selected = (
                    _tune_flash_attention(query, key, value, scale, causal)
                    if autotune
                    else MLXAttentionConfig("metile", 128)
                )
                _write_config(persistent_key, selected)
            _flash_schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        return _native_attention(query, key, value, scale, causal)
    kernel = _compile_flash_attention(
        query.shape[1],
        key.shape[1],
        query.shape[-1],
        query.dtype,
        scale,
        causal,
        selected.block,
    )
    return kernel(query, key, value)


def _causal_attention_bias(query, key):
    import mlx.core as mx

    rows, columns = query.shape[-2], key.shape[-2]
    row = mx.arange(rows)[:, None]
    column = mx.arange(columns)[None, :]
    allowed = column <= row + columns - rows
    return mx.where(allowed, 0.0, mx.array(float("-inf"), dtype=query.dtype))


def mlx_flash_attention_dispatches():
    """Return in-process discovered-attention schedule decisions."""
    return tuple(
        {
            "query_shape": key[0],
            "key_shape": key[1],
            "dtype": key[2],
            "scale": key[3],
            "causal": key[4],
            "algorithm": config.algorithm,
            "block": config.block,
        }
        for key, config in sorted(_flash_schedule_cache.items())
    )


__all__ = ["mlx_flash_attention", "mlx_flash_attention_dispatches"]
