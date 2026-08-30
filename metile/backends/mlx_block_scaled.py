from __future__ import annotations

import inspect
import statistics
import threading
from dataclasses import dataclass

import numpy as np

from metile.backends.mlx import (
    _batched_evaluator,
    _mlx_compiler_dtype,
    _mlx_kernel_body,
    _tune_framework_kernels,
    batched_measure,
    calibrate_tournament_batch,
)
from metile.codegen import msl_emitter
from metile.codegen.msl_emitter import emit
from metile.compiler import passes as compiler_passes
from metile.compiler.block_scaled import lower_block_scaled_matmul
from metile.compiler.passes import decompose_nax_fragments
from metile.compiler.schedule_search import (
    choose_mdl_tie,
    compressed_description_bits,
    optimize_tile_schedules,
)
from metile.runtime.block_scaled import _quantize_block_scaled_arrays
from metile.runtime.cache import (
    cache_root,
    read_cached_config,
    stable_digest,
    write_cached_config,
)
from metile.tuning import confirm_pairwise, round_robin

_kernel_cache = {}
_schedule_cache = {}
_cache_lock = threading.RLock()
_cache_path = cache_root() / "mlx-block-scaled-autotune-v2.json"
_SWITCH_MARGIN = 0.10
_TUNER_VERSION = 3


@dataclass(frozen=True)
class MLXBlockScaledConfig:
    block_m: int
    block_n: int
    schedule: str
    fragment_type: str
    k_unroll: int
    algorithm: str = "metile"


_CONFIGS = (
    MLXBlockScaledConfig(0, 0, "", "", 0, "mlx"),
    MLXBlockScaledConfig(32, 64, "linear", "bfloat", 2),
    MLXBlockScaledConfig(32, 64, "hilbert", "bfloat", 2),
    MLXBlockScaledConfig(64, 64, "linear", "bfloat", 2),
    MLXBlockScaledConfig(64, 64, "diagonal", "float", 1),
    MLXBlockScaledConfig(128, 64, "grouped4", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "linear", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "grouped4", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "morton", "bfloat", 2),
    MLXBlockScaledConfig(64, 128, "hilbert", "float", 1),
)


@dataclass(frozen=True)
class MLXBlockScaledWeight:
    """K-major MXFP weight arrays owned by MLX."""

    values: object
    scales: object
    shape: tuple[int, int]
    format: str
    native_values: object | None = None
    native_scales: object | None = None

    @property
    def bits(self):
        return 4 if self.format == "mxfp4" else 8

    @classmethod
    def quantize(cls, weight, *, format="mxfp8"):
        import mlx.core as mx

        dense = np.array(weight, dtype=np.float32)
        packed, scales = _quantize_block_scaled_arrays(dense, format)
        values = mx.array(packed)
        encoded_scales = mx.array(scales)
        native_values, native_scales = mx.quantize(mx.array(dense.T), mode=format)
        mx.eval(values, encoded_scales, native_values, native_scales)
        return cls(
            values,
            encoded_scales,
            tuple(dense.shape),
            format,
            native_values,
            native_scales,
        )


@dataclass(frozen=True)
class _MLXBlockScaledKernel:
    operation: object
    threadgroup: tuple[int, int, int]
    block_m: int
    block_n: int
    output_features: int
    description_bits: int

    def __call__(self, activations, weight):
        rows = activations.size // activations.shape[-1]
        output_shape = (*activations.shape[:-1], self.output_features)
        threadgroups_m = (rows + self.block_m - 1) // self.block_m
        return self.operation(
            inputs=[activations, weight.values, weight.scales],
            grid=(threadgroups_m * self.threadgroup[0], self.output_features // self.block_n, 1),
            threadgroup=self.threadgroup,
            output_shapes=[output_shape],
            output_dtypes=[activations.dtype],
        )[0]


def _candidate_configs(rows, output_features, reduction, native_available):
    return tuple(
        config
        for config in _CONFIGS
        if (config.algorithm == "mlx" and native_available)
        or (
            config.algorithm == "metile"
            and output_features % config.block_n == 0
            and reduction % (16 * config.k_unroll) == 0
            and (rows >= config.block_m or config.block_m == 32)
        )
    )


def _compile_mlx_block_scaled(rows, reduction, output_features, dtype, format, config):
    import mlx.core as mx

    numpy_dtype = _mlx_compiler_dtype(dtype)
    if numpy_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise TypeError("MLX block-scaled matmul requires float16 or float32 activations")
    if config.algorithm != "metile":
        raise ValueError("only meTile block-scaled configs compile a Metal kernel")
    key = (
        rows,
        reduction,
        output_features,
        numpy_dtype.str,
        format,
        config,
    )
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    scalar_type = {
        "mlx.core.bfloat16": "bf16",
        "mlx.core.float16": "f16",
        "mlx.core.float32": "f32",
    }[str(dtype)]
    function_name = f"metile_mlx_bsmm_{stable_digest(key)[:16]}"
    metal_ir = decompose_nax_fragments(
        optimize_tile_schedules(
            lower_block_scaled_matmul(
                function_name,
                rows,
                output_features,
                reduction,
                4 if format == "mxfp4" else 8,
                block_m=config.block_m,
                block_n=config.block_n,
                register_fragments=True,
                schedule=config.schedule,
                fragment_type=config.fragment_type,
                k_unroll=config.k_unroll,
                activation_type=scalar_type,
                output_type=scalar_type,
            )
        )
    )
    source = emit(metal_ir)
    kernel_start = source.index("[[kernel")
    operation = mx.fast.metal_kernel(
        name=function_name,
        input_names=["activations", "packed", "scales"],
        output_names=["output"],
        source=_mlx_kernel_body(source),
        header=source[:kernel_start],
    )
    kernel = _MLXBlockScaledKernel(
        operation,
        metal_ir.threadgroup_size,
        config.block_m,
        config.block_n,
        output_features,
        compressed_description_bits(source),
    )
    _kernel_cache[key] = kernel
    return kernel


def _persistent_key(rows, reduction, output_features, dtype, format, configs):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "configs": [vars(config) for config in configs],
            "dtype": str(dtype),
            "format": format,
            "mlx": mx.__version__,
            "output_features": output_features,
            "reduction": reduction,
            "rows": rows,
            "source": mlx_block_scaled_backend_signature(),
            "switch_margin": _SWITCH_MARGIN,
            "tuner": _TUNER_VERSION,
        }
    )


def mlx_block_scaled_backend_signature():
    """Return every source/config identity that can alter MXFP dispatch."""
    return stable_digest(
        {
            "candidates": inspect.getsource(_candidate_configs),
            "compile": inspect.getsource(_compile_mlx_block_scaled),
            "configs": [vars(config) for config in _CONFIGS],
            "decompose": inspect.getsource(compiler_passes._block_scaled_nax_steps),
            "decompose_step": inspect.getsource(compiler_passes._block_scaled_nax_step),
            "dispatch": inspect.getsource(mlx_block_scaled_matmul),
            "emit": inspect.getsource(msl_emitter._emit_nax_load_block_scaled_fragment),
            "emit_scale": inspect.getsource(msl_emitter._emit_nax_load_block_scale),
            "emit_store": inspect.getsource(msl_emitter._emit_nax_store_fragment),
            "lowering": inspect.getsource(lower_block_scaled_matmul),
            "native": inspect.getsource(_native_block_scaled_matmul),
            "switch_margin": _SWITCH_MARGIN,
            "tune": inspect.getsource(_tune_config),
            "tuner": _TUNER_VERSION,
        }
    )


def _tune_config(activations, weight, configs):
    import mlx.core as mx

    kernels = []
    for config in configs:
        if config.algorithm == "mlx":
            kernels.append(
                (
                    config,
                    lambda: _native_block_scaled_matmul(activations, weight),
                    0,
                )
            )
            continue
        try:
            kernel = _compile_mlx_block_scaled(
                activations.size // activations.shape[-1],
                activations.shape[-1],
                weight.shape[1],
                activations.dtype,
                weight.format,
                config,
            )
            kernels.append(
                (
                    config,
                    lambda kernel=kernel: kernel(activations, weight),
                    kernel.description_bits,
                )
            )
        except (RuntimeError, TypeError, ValueError):
            continue
    if not kernels:
        raise RuntimeError("no MLX block-scaled kernel compiled for this shape")

    native = next((candidate for candidate in kernels if candidate[0].algorithm == "mlx"), None)
    if native is not None:
        reference = native[1]()
        mx.eval(reference)
        compatible = [native]
        for candidate in kernels:
            if candidate is native:
                continue
            actual = candidate[1]()
            mx.eval(actual)
            if bool(mx.allclose(actual, reference, rtol=5e-3, atol=5e-3).item()):
                compatible.append(candidate)
        return _tune_framework_kernels(
            compatible,
            _batched_evaluator(),
            margin=_SWITCH_MARGIN,
        )

    # One eval per batch; see calibrate_tournament_batch for why evaluating a single
    # dispatch per sample compresses the ratios between candidates toward 1.0.
    measure = batched_measure(calibrate_tournament_batch(kernels[0][1]))

    provisional = round_robin(kernels, 9, measure)
    medians = {config: statistics.median(samples) for config, samples in provisional.items()}
    best = min(medians.values())
    finalists = [candidate for candidate in kernels if medians[candidate[0]] <= best * 1.08]
    # Confirm head to head. There is no native candidate to pair against here, so the
    # provisional fastest serves as the reference; what matters is that every finalist is
    # measured in an identically sized context rather than in one crowded rotation.
    reference = min(medians, key=medians.__getitem__)
    timings = confirm_pairwise(finalists, reference, 21, measure)
    return choose_mdl_tie(
        [
            (timings[config], description_bits, config)
            for config, _, description_bits in finalists
            if config in timings
        ]
    )


def _native_block_scaled_matmul(activations, weight):
    import mlx.core as mx

    if weight.native_values is None or weight.native_scales is None:
        raise ValueError("block-scaled weight has no native MLX representation")
    return mx.quantized_matmul(
        activations,
        weight.native_values,
        weight.native_scales,
        mode=weight.format,
    )


def mlx_block_scaled_matmul(activations, weight, *, autotune=True):
    """Run a zero-copy M5 MXFP matmul inside an MLX lazy graph."""
    if not isinstance(weight, MLXBlockScaledWeight):
        raise TypeError("weight must be an MLXBlockScaledWeight")
    if activations.ndim < 2 or activations.shape[-1] != weight.shape[0]:
        raise ValueError("expected activations[..., K] and a KxN block-scaled weight")
    if weight.format not in {"mxfp4", "mxfp8"}:
        raise ValueError("block-scaled weight format must be mxfp4 or mxfp8")
    _mlx_compiler_dtype(activations.dtype)
    rows = activations.size // activations.shape[-1]
    reduction, output_features = weight.shape
    native_available = weight.native_values is not None and weight.native_scales is not None
    configs = _candidate_configs(
        rows,
        output_features,
        reduction,
        native_available,
    )
    if not configs:
        raise ValueError("no MLX block-scaled schedule supports this shape")
    schedule_key = (
        rows,
        reduction,
        output_features,
        str(activations.dtype),
        weight.format,
        native_available,
    )
    selected = _schedule_cache.get(schedule_key)
    if selected is None:
        with _cache_lock:
            selected = _schedule_cache.get(schedule_key)
            if selected is None:
                key = _persistent_key(
                    rows,
                    reduction,
                    output_features,
                    activations.dtype,
                    weight.format,
                    configs,
                )
                selected = read_cached_config(_cache_path, key, configs)
            if selected is None:
                selected = _tune_config(activations, weight, configs) if autotune else configs[0]
                write_cached_config(_cache_path, key, selected)
            _schedule_cache[schedule_key] = selected
    if selected.algorithm == "mlx":
        return _native_block_scaled_matmul(activations, weight)
    kernel = _compile_mlx_block_scaled(
        rows,
        reduction,
        output_features,
        activations.dtype,
        weight.format,
        selected,
    )
    return kernel(activations, weight)


def mlx_block_scaled_dispatches():
    """Return in-process MLX block-scaled schedule decisions."""
    return tuple(
        {
            "rows": key[0],
            "reduction": key[1],
            "output_features": key[2],
            "dtype": key[3],
            "format": key[4],
            "native_available": key[5],
            **vars(config),
        }
        for key, config in sorted(_schedule_cache.items())
    )


__all__ = [
    "MLXBlockScaledConfig",
    "MLXBlockScaledWeight",
    "mlx_block_scaled_backend_signature",
    "mlx_block_scaled_dispatches",
    "mlx_block_scaled_matmul",
]
