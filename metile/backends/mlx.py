from __future__ import annotations

import inspect
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np

import metile
from kernels.add_rmsnorm import add_rmsnorm
from kernels.attention import ATTENTION_DECODE_CONFIGS, attention_decode_kernel
from kernels.rmsnorm import rmsnorm
from metile.compiler.schedule_search import choose_mdl_tie
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_mlx_kernel_cache = {}
_mlx_schedule_cache = {}
_mlx_cache_lock = threading.RLock()
_mlx_cache_path = cache_root() / "mlx-attention-autotune-v1.json"
_mlx_rms_kernel_cache = {}
_mlx_rms_schedule_cache = {}
_mlx_rms_cache_path = cache_root() / "mlx-rmsnorm-autotune-v1.json"
_mlx_add_rms_kernel_cache = {}
_mlx_add_rms_schedule_cache = {}
_mlx_add_rms_cache_path = cache_root() / "mlx-add-rmsnorm-autotune-v2.json"
_FRAMEWORK_SWITCH_MARGIN = 0.05
_GRAPH_FUSION_SWITCH_MARGIN = 0.10


@dataclass(frozen=True)
class MLXAttentionConfig:
    algorithm: str
    block: int = 0


@dataclass(frozen=True)
class MLXRMSNormConfig:
    algorithm: str
    block: int = 0


@dataclass(frozen=True)
class MLXAddRMSNormConfig:
    algorithm: str
    block: int = 0


_MLX_ATTENTION_CONFIGS = tuple(
    [MLXAttentionConfig("mlx")]
    + [MLXAttentionConfig("metile", config.kwargs["BLOCK"]) for config in ATTENTION_DECODE_CONFIGS]
)
_MLX_RMSNORM_CONFIGS = tuple(
    [MLXRMSNormConfig("mlx")]
    + [MLXRMSNormConfig("metile", block) for block in (32, 64, 128, 256, 512, 1024)]
)
_MLX_ADD_RMSNORM_CONFIGS = tuple(
    [MLXAddRMSNormConfig("mlx")]
    + [MLXAddRMSNormConfig("metile", block) for block in (32, 64, 128, 256, 512, 1024)]
)


@dataclass(frozen=True)
class _MLXKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    description_bits: int

    def __call__(self, query, key, value):
        batch, query_heads, _, _ = query.shape
        return self.operation(
            inputs=[query, key, value],
            grid=(batch * query_heads * self.threadgroup[0], 1, 1),
            threadgroup=self.threadgroup,
            output_shapes=[query.shape],
            output_dtypes=[query.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXRMSNormKernel:
    operation: object
    block: int
    description_bits: int

    def __call__(self, values, weight):
        rows = values.size // values.shape[-1]
        return self.operation(
            inputs=[values, weight],
            grid=(rows * self.block, 1, 1),
            threadgroup=(self.block, 1, 1),
            output_shapes=[values.shape],
            output_dtypes=[values.dtype],
        )[0]


@dataclass(frozen=True)
class _MLXAddRMSNormKernel:
    operation: object
    block: int
    description_bits: int

    def __call__(self, values, residual, weight):
        rows = values.size // values.shape[-1]
        return tuple(
            self.operation(
                inputs=[values, residual, weight],
                grid=(rows * self.block, 1, 1),
                threadgroup=(self.block, 1, 1),
                output_shapes=[values.shape, values.shape],
                output_dtypes=[values.dtype, values.dtype],
            )
        )


def _require_mlx():
    try:
        import mlx.core as mx
    except ImportError as error:
        raise ImportError("The meTile MLX backend requires the optional 'mlx' package") from error
    return mx


def _mlx_dtype_to_numpy(dtype):
    name = str(dtype)
    if name == "mlx.core.float16":
        return np.dtype(np.float16)
    if name == "mlx.core.float32":
        return np.dtype(np.float32)
    raise TypeError(f"meTile MLX attention does not support {name}")


def _replace_identifier(source, identifier, expression):
    return re.sub(rf"\b{re.escape(identifier)}\b", expression, source)


def _mlx_kernel_body(msl_source):
    kernel_start = msl_source.index("[[kernel")
    body_start = msl_source.index("{", kernel_start)
    header = msl_source[kernel_start:body_start]
    body = msl_source[body_start + 1 : msl_source.rfind("}")]

    attributes = re.findall(
        r"\b(?:u?int|u?int[234])\s+(\w+)\s+\[\[(\w+)\]\]",
        header,
    )
    for identifier, attribute in attributes:
        if attribute in ("thread_position_in_threadgroup", "thread_index_in_threadgroup"):
            expression = "thread_index_in_threadgroup"
        elif attribute in ("threadgroup_position_in_grid", "thread_position_in_grid"):
            axis = identifier.rsplit("_", 1)[-1]
            expression = attribute if axis not in ("x", "y", "z") else f"{attribute}.{axis}"
        else:
            expression = attribute
        body = _replace_identifier(body, identifier, expression)
    return body


def _compile_mlx_attention(query_heads, key_value_heads, dimension, dtype, scale, block):
    mx = _require_mlx()
    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    kernel_key = (query_heads, key_value_heads, dimension, numpy_dtype.str, scale, block)
    cached = _mlx_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    query = metile.Buffer.empty((query_heads * dimension,), dtype=numpy_dtype)
    key = metile.Buffer.empty((key_value_heads * dimension,), dtype=numpy_dtype)
    value = metile.Buffer.empty(key.shape, dtype=numpy_dtype)
    output = metile.Buffer.empty(query.shape, dtype=numpy_dtype)
    compiled = attention_decode_kernel.get_compiled(
        query,
        key,
        value,
        output,
        1,
        float(scale),
        D=dimension,
        Q_HEADS=query_heads,
        KV_HEADS=key_value_heads,
        BLOCK=block,
    )
    source = _mlx_kernel_body(compiled.msl_source)
    source = _replace_identifier(source, "N", "K_shape[2]")
    source = _replace_identifier(source, "scale", f"{float(scale):.12g}f")
    operation_name = f"metile_attention_{stable_digest(kernel_key)[:16]}"
    operation = mx.fast.metal_kernel(
        name=operation_name,
        input_names=["Q", "K", "V"],
        output_names=["Out"],
        source=source,
    )
    kernel = _MLXKernel(operation, compiled.threadgroup_size, compiled.description_bits)
    _mlx_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_mlx_rms_norm(hidden, dtype, eps, block):
    mx = _require_mlx()
    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    kernel_key = (hidden, numpy_dtype.str, float(eps), block)
    cached = _mlx_rms_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    values = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    weight = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    output = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    compiled = rmsnorm.get_compiled(
        values,
        weight,
        output,
        hidden,
        float(eps),
        BLOCK=block,
    )
    source = _mlx_kernel_body(compiled.msl_source)
    source = _replace_identifier(source, "N", "X_shape[X_ndim - 1]")
    source = _replace_identifier(source, "eps", f"{float(eps):.12g}f")
    operation = mx.fast.metal_kernel(
        name=f"metile_rmsnorm_{stable_digest(kernel_key)[:16]}",
        input_names=["X", "W"],
        output_names=["Out"],
        source=source,
    )
    kernel = _MLXRMSNormKernel(operation, block, compiled.description_bits)
    _mlx_rms_kernel_cache[kernel_key] = kernel
    return kernel


def _compile_mlx_add_rms_norm(hidden, dtype, eps, block):
    mx = _require_mlx()
    numpy_dtype = _mlx_dtype_to_numpy(dtype)
    kernel_key = (hidden, numpy_dtype.str, float(eps), block)
    cached = _mlx_add_rms_kernel_cache.get(kernel_key)
    if cached is not None:
        return cached

    values = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    residual = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    weight = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    summed = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    output = metile.Buffer.empty((hidden,), dtype=numpy_dtype)
    compiled = add_rmsnorm.get_compiled(
        values,
        residual,
        weight,
        summed,
        output,
        hidden,
        float(eps),
        BLOCK=block,
    )
    source = _mlx_kernel_body(compiled.msl_source)
    source = _replace_identifier(source, "N", "X_shape[X_ndim - 1]")
    source = _replace_identifier(source, "eps", f"{float(eps):.12g}f")
    operation = mx.fast.metal_kernel(
        name=f"metile_add_rmsnorm_{stable_digest(kernel_key)[:16]}",
        input_names=["X", "Residual", "W"],
        output_names=["Sum", "Out"],
        source=source,
    )
    kernel = _MLXAddRMSNormKernel(operation, block, compiled.description_bits)
    _mlx_add_rms_kernel_cache[kernel_key] = kernel
    return kernel


def _token_bucket(tokens):
    return 1 << max(tokens - 1, 0).bit_length()


def _persistent_key(query, key, scale, configs):
    mx = _require_mlx()
    device = mx.device_info()
    return stable_digest(
        {
            "architecture": device.get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(query.dtype),
            "key_value_heads": key.shape[1],
            "mlx": mx.__version__,
            "query_heads": query.shape[1],
            "scale": scale,
            "source": inspect.getsource(attention_decode_kernel.fn),
            "tokens": _token_bucket(key.shape[2]),
        }
    )


def _read_config(key, configs):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_mlx_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in configs
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
        ),
        None,
    )


def _write_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_mlx_cache_path, {})
    payload[key] = {"algorithm": config.algorithm, "block": config.block}
    atomic_write_json(_mlx_cache_path, payload)


def _rms_persistent_key(values, eps, configs):
    mx = _require_mlx()
    device = mx.device_info()
    return stable_digest(
        {
            "architecture": device.get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(values.dtype),
            "eps": float(eps),
            "hidden": values.shape[-1],
            "mlx": mx.__version__,
            "rows": _token_bucket(values.size // values.shape[-1]),
            "source": inspect.getsource(rmsnorm.fn),
        }
    )


def _read_rms_config(key, configs):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_mlx_rms_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in configs
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
        ),
        None,
    )


def _write_rms_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_mlx_rms_cache_path, {})
    payload[key] = {"algorithm": config.algorithm, "block": config.block}
    atomic_write_json(_mlx_rms_cache_path, payload)


def _add_rms_persistent_key(values, eps, configs):
    mx = _require_mlx()
    device = mx.device_info()
    return stable_digest(
        {
            "architecture": device.get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(values.dtype),
            "eps": float(eps),
            "hidden": values.shape[-1],
            "mlx": mx.__version__,
            "rows": _token_bucket(values.size // values.shape[-1]),
            "source": inspect.getsource(add_rmsnorm.fn),
            "switch_margin": _GRAPH_FUSION_SWITCH_MARGIN,
            "tuner": 2,
        }
    )


def _read_add_rms_config(key, configs):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_mlx_add_rms_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    return next(
        (
            config
            for config in configs
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
        ),
        None,
    )


def _write_add_rms_config(key, config):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_mlx_add_rms_cache_path, {})
    payload[key] = {"algorithm": config.algorithm, "block": config.block}
    atomic_write_json(_mlx_add_rms_cache_path, payload)


def _tune_mlx_attention(query, key, value, scale, configs):
    mx = _require_mlx()
    dimension = query.shape[-1]
    kernels = []
    for config in configs:
        if config.algorithm == "mlx":
            kernels.append(
                (
                    config,
                    lambda: mx.fast.scaled_dot_product_attention(query, key, value, scale=scale),
                    0,
                )
            )
        else:
            kernel = _compile_mlx_attention(
                query.shape[1], key.shape[1], dimension, query.dtype, scale, config.block
            )
            kernels.append(
                (config, lambda kernel=kernel: kernel(query, key, value), kernel.description_bits)
            )
    for _, dispatch, _ in kernels:
        mx.eval(dispatch())

    samples = {config: [] for config in configs}
    for round_index in range(7):
        ordered = kernels[round_index % len(kernels) :] + kernels[: round_index % len(kernels)]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval(dispatch())
            samples[config].append((time.perf_counter_ns() - start) * 1e-9)

    results = [
        (statistics.median(samples[config]), description_bits, config)
        for config, _, description_bits in kernels
    ]
    return _choose_framework_config(results)


def _tune_mlx_rms_norm(values, weight, eps, configs):
    mx = _require_mlx()
    kernels = []
    for config in configs:
        if config.algorithm == "mlx":
            kernels.append(
                (
                    config,
                    lambda: mx.fast.rms_norm(values, weight, eps),
                    0,
                )
            )
        else:
            kernel = _compile_mlx_rms_norm(values.shape[-1], values.dtype, eps, config.block)
            kernels.append(
                (config, lambda kernel=kernel: kernel(values, weight), kernel.description_bits)
            )
    for _, dispatch, _ in kernels:
        mx.eval(dispatch())

    samples = {config: [] for config in configs}
    for round_index in range(7):
        ordered = kernels[round_index % len(kernels) :] + kernels[: round_index % len(kernels)]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval(dispatch())
            samples[config].append((time.perf_counter_ns() - start) * 1e-9)

    results = [
        (statistics.median(samples[config]), description_bits, config)
        for config, _, description_bits in kernels
    ]
    return _choose_framework_config(results)


def _tune_mlx_add_rms_norm(values, residual, weight, eps, configs):
    mx = _require_mlx()
    kernels = []
    for config in configs:
        if config.algorithm == "mlx":

            def native_dispatch():
                summed = values + residual
                return summed, mx.fast.rms_norm(summed, weight, eps)

            kernels.append((config, native_dispatch, 0))
        else:
            kernel = _compile_mlx_add_rms_norm(values.shape[-1], values.dtype, eps, config.block)
            kernels.append(
                (
                    config,
                    lambda kernel=kernel: kernel(values, residual, weight),
                    kernel.description_bits,
                )
            )
    for _, dispatch, _ in kernels:
        mx.eval(*dispatch())

    samples = {config: [] for config in configs}
    for round_index in range(11):
        ordered = kernels[round_index % len(kernels) :] + kernels[: round_index % len(kernels)]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch, _ in ordered:
            start = time.perf_counter_ns()
            mx.eval(*dispatch())
            samples[config].append((time.perf_counter_ns() - start) * 1e-9)

    provisional = {
        config: statistics.median(config_samples) for config, config_samples in samples.items()
    }
    best = min(provisional.values())
    native_config = next(config for config in configs if config.algorithm == "mlx")
    fastest_generated_config = min(
        (config for config in configs if config.algorithm == "metile"),
        key=provisional.__getitem__,
    )
    finalists = {
        config
        for config, latency in provisional.items()
        if latency <= best * 1.10 or config is native_config or config is fastest_generated_config
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
            mx.eval(*dispatch())
            samples[config].append((time.perf_counter_ns() - start) * 1e-9)

    results = [
        (statistics.median(samples[config]), description_bits, config)
        for config, _, description_bits in finalist_kernels
    ]
    return _choose_framework_config(results, margin=_GRAPH_FUSION_SWITCH_MARGIN)


def _choose_framework_config(results, *, margin=_FRAMEWORK_SWITCH_MARGIN):
    native = next(result for result in results if result[2].algorithm == "mlx")
    generated = [result for result in results if result[2].algorithm == "metile"]
    fastest_generated = min(generated, key=lambda result: result[0])
    if fastest_generated[0] >= native[0] * (1.0 - margin):
        return native[2]
    return choose_mdl_tie(generated)


def mlx_attention_decode(query, key, value, *, scale, autotune=True):
    """Run a zero-copy meTile decode-attention primitive on MLX arrays.

    Inputs use MLX's ``[batch, heads, sequence, dimension]`` convention. The
    operation is intentionally decode-only: the query sequence length must be
    one. Generated Metal is embedded in MLX's lazy graph without NumPy copies.
    """
    _require_mlx()
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("meTile MLX attention expects rank-four query, key, and value arrays")
    if query.shape[2] != 1:
        raise ValueError("meTile MLX attention currently supports decode queries of length one")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise ValueError("meTile MLX attention requires matching batch and key/value shapes")
    if query.shape[1] % key.shape[1]:
        raise ValueError("key/value heads must divide query heads")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] % 32:
        raise ValueError("head dimensions must match and be a multiple of 32")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("meTile MLX attention requires matching input dtypes")
    _mlx_dtype_to_numpy(query.dtype)

    schedule_key = (
        query.shape[0],
        query.shape[1],
        key.shape[1],
        _token_bucket(key.shape[2]),
        query.shape[-1],
        str(query.dtype),
        float(scale),
    )
    selected = _mlx_schedule_cache.get(schedule_key)
    if selected is None:
        with _mlx_cache_lock:
            selected = _mlx_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _persistent_key(query, key, scale, _MLX_ATTENTION_CONFIGS)
                selected = _read_config(persistent_key, _MLX_ATTENTION_CONFIGS)
            if selected is None:
                selected = (
                    _tune_mlx_attention(query, key, value, scale, _MLX_ATTENTION_CONFIGS)
                    if autotune
                    else MLXAttentionConfig("metile", 256)
                )
                _write_config(persistent_key, selected)
            _mlx_schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        mx = _require_mlx()
        return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)

    kernel = _compile_mlx_attention(
        query.shape[1],
        key.shape[1],
        query.shape[-1],
        query.dtype,
        scale,
        selected.block,
    )
    return kernel(query, key, value)


def mlx_rms_norm(values, weight, eps, *, autotune=True):
    """Run an autotuned zero-copy RMSNorm primitive inside an MLX graph."""
    mx = _require_mlx()
    if values.ndim < 1 or weight.ndim != 1 or values.shape[-1] != weight.shape[0]:
        raise ValueError("meTile MLX RMSNorm requires a matching one-dimensional weight")
    if values.dtype != weight.dtype:
        raise TypeError("meTile MLX RMSNorm requires matching input and weight dtypes")
    _mlx_dtype_to_numpy(values.dtype)

    rows = values.size // values.shape[-1]
    schedule_key = (
        _token_bucket(rows),
        values.shape[-1],
        str(values.dtype),
        float(eps),
    )
    selected = _mlx_rms_schedule_cache.get(schedule_key)
    if selected is None:
        with _mlx_cache_lock:
            selected = _mlx_rms_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _rms_persistent_key(values, eps, _MLX_RMSNORM_CONFIGS)
                selected = _read_rms_config(persistent_key, _MLX_RMSNORM_CONFIGS)
            if selected is None:
                selected = (
                    _tune_mlx_rms_norm(values, weight, eps, _MLX_RMSNORM_CONFIGS)
                    if autotune
                    else MLXRMSNormConfig("metile", 256)
                )
                _write_rms_config(persistent_key, selected)
            _mlx_rms_schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        return mx.fast.rms_norm(values, weight, eps)
    kernel = _compile_mlx_rms_norm(values.shape[-1], values.dtype, eps, selected.block)
    return kernel(values, weight)


def mlx_add_rms_norm(values, residual, weight, eps, *, autotune=True):
    """Fuse residual addition and RMSNorm while preserving both outputs."""
    mx = _require_mlx()
    if values.shape != residual.shape or values.dtype != residual.dtype:
        raise ValueError("meTile fused add/RMSNorm requires matching residual inputs")
    if values.ndim < 1 or weight.ndim != 1 or values.shape[-1] != weight.shape[0]:
        raise ValueError("meTile fused add/RMSNorm requires a matching RMSNorm weight")
    if values.dtype != weight.dtype:
        raise TypeError("meTile fused add/RMSNorm requires matching input and weight dtypes")
    _mlx_dtype_to_numpy(values.dtype)

    rows = values.size // values.shape[-1]
    schedule_key = (
        _token_bucket(rows),
        values.shape[-1],
        str(values.dtype),
        float(eps),
    )
    selected = _mlx_add_rms_schedule_cache.get(schedule_key)
    if selected is None:
        with _mlx_cache_lock:
            selected = _mlx_add_rms_schedule_cache.get(schedule_key)
            if selected is None:
                persistent_key = _add_rms_persistent_key(values, eps, _MLX_ADD_RMSNORM_CONFIGS)
                selected = _read_add_rms_config(persistent_key, _MLX_ADD_RMSNORM_CONFIGS)
            if selected is None:
                selected = (
                    _tune_mlx_add_rms_norm(
                        values,
                        residual,
                        weight,
                        eps,
                        _MLX_ADD_RMSNORM_CONFIGS,
                    )
                    if autotune
                    else MLXAddRMSNormConfig("metile", 256)
                )
                _write_add_rms_config(persistent_key, selected)
            _mlx_add_rms_schedule_cache[schedule_key] = selected

    if selected.algorithm == "mlx":
        summed = values + residual
        return summed, mx.fast.rms_norm(summed, weight, eps)
    kernel = _compile_mlx_add_rms_norm(values.shape[-1], values.dtype, eps, selected.block)
    return kernel(values, residual, weight)


def mlx_attention_dispatches():
    """Return the in-process MLX attention schedule decisions."""
    return tuple(
        {
            "batch": key[0],
            "query_heads": key[1],
            "key_value_heads": key[2],
            "token_bucket": key[3],
            "dimension": key[4],
            "dtype": key[5],
            "algorithm": config.algorithm,
            "block": config.block,
        }
        for key, config in sorted(_mlx_schedule_cache.items())
    )


def mlx_rms_norm_dispatches():
    """Return the in-process MLX RMSNorm schedule decisions."""
    return tuple(
        {
            "row_bucket": key[0],
            "hidden": key[1],
            "dtype": key[2],
            "eps": key[3],
            "algorithm": config.algorithm,
            "block": config.block,
        }
        for key, config in sorted(_mlx_rms_schedule_cache.items())
    )


def mlx_add_rms_norm_dispatches():
    """Return the in-process fused residual/RMSNorm schedule decisions."""
    return tuple(
        {
            "row_bucket": key[0],
            "hidden": key[1],
            "dtype": key[2],
            "eps": key[3],
            "algorithm": config.algorithm,
            "block": config.block,
        }
        for key, config in sorted(_mlx_add_rms_schedule_cache.items())
    )


def mlx_add_rms_norm_selection(values, eps):
    """Return the cached graph-fusion decision for a runtime shape, if known."""
    rows = values.size // values.shape[-1]
    return _mlx_add_rms_schedule_cache.get(
        (
            _token_bucket(rows),
            values.shape[-1],
            str(values.dtype),
            float(eps),
        )
    )


__all__ = [
    "MLXAddRMSNormConfig",
    "MLXAttentionConfig",
    "MLXRMSNormConfig",
    "mlx_add_rms_norm",
    "mlx_add_rms_norm_dispatches",
    "mlx_add_rms_norm_selection",
    "mlx_attention_decode",
    "mlx_attention_dispatches",
    "mlx_rms_norm",
    "mlx_rms_norm_dispatches",
]
