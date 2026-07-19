from __future__ import annotations

import inspect
import os
import statistics
import threading
import time
from dataclasses import dataclass

from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_AFFINE8_GROUP_SIZES = (32, 64, 128)
_AFFINE8_GROUP_NEAR_PEAK_MARGIN = 0.025
_AFFINE8_GROUP_REPETITIONS = 4
_AFFINE8_GROUP_TUNER_VERSION = 5
_affine8_group_cache = {}
_affine8_group_lock = threading.RLock()
_affine8_group_cache_path = cache_root() / "mlx-affine8-group-autotune-v5.json"


@dataclass(frozen=True)
class MLXCompressedDownWeight:
    """An MLX-owned affine INT8 or MXFP8 output-major weight."""

    values: object
    scales: object
    biases: object | None
    shape: tuple[int, int]
    format: str
    group_size: int

    @classmethod
    def quantize(cls, weight, *, format="affine8", group_size=64):
        import mlx.core as mx

        if weight.ndim != 2:
            raise ValueError("compressed down-projection weights must be matrices")
        if format == "affine8":
            if group_size not in {32, 64, 128}:
                raise ValueError("affine8 group size must be 32, 64, or 128")
            values, scales, biases = mx.quantize(
                weight,
                group_size=group_size,
                bits=8,
                mode="affine",
            )
        elif format == "mxfp8":
            values, scales = mx.quantize(weight, mode="mxfp8")
            biases = None
            group_size = 32
        else:
            raise ValueError("compressed down format must be affine8 or mxfp8")
        arrays = (values, scales) if biases is None else (values, scales, biases)
        mx.eval(*arrays)
        return cls(values, scales, biases, tuple(weight.shape), format, group_size)

    @property
    def nbytes(self):
        return (
            self.values.nbytes
            + self.scales.nbytes
            + (self.biases.nbytes if self.biases is not None else 0)
        )

    def __call__(self, values):
        import mlx.core as mx

        if self.format == "affine8":
            return mx.quantized_matmul(
                values,
                self.values,
                self.scales,
                self.biases,
                group_size=self.group_size,
                bits=8,
                mode="affine",
            )
        return mx.quantized_matmul(
            values,
            self.values,
            self.scales,
            mode="mxfp8",
        )


def _affine8_group_tuning_key(weights, mx, objective):
    shapes = sorted({(tuple(weight.shape), str(weight.dtype)) for weight in weights})
    return stable_digest(
        {
            "backend": mlx_compressed_down_backend_signature(),
            "device": mx.device_info(),
            "mlx": mx.__version__,
            "objective": objective,
            "shapes": shapes,
            "version": _AFFINE8_GROUP_TUNER_VERSION,
        }
    )


def _restore_affine8_group_tuning(key, group_sizes, objective):
    record = _affine8_group_cache.get(key)
    if record is None and os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        record = read_json(_affine8_group_cache_path, {}).get(key)
    if not isinstance(record, dict):
        return None
    group_size = record.get("group_size")
    if record.get("objective") != objective:
        return None
    timings = record.get("median_nanoseconds")
    native_timing = record.get("native_median_nanoseconds")
    errors = record.get("mean_absolute_error")
    if group_size not in group_sizes or not isinstance(timings, dict):
        return None
    expected = {str(group) for group in group_sizes}
    if set(timings) != expected or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in timings.values()
    ):
        return None
    if (
        not isinstance(native_timing, int)
        or isinstance(native_timing, bool)
        or native_timing <= 0
        or not isinstance(errors, dict)
        or set(errors) != expected
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0.0
            for value in errors.values()
        )
    ):
        return None
    record = {
        "group_size": group_size,
        "mean_absolute_error": errors,
        "median_nanoseconds": timings,
        "native_median_nanoseconds": native_timing,
        "objective": objective,
    }
    _affine8_group_cache[key] = record
    return record


def _write_affine8_group_tuning(key, record):
    _affine8_group_cache[key] = record
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_affine8_group_cache_path, {})
    payload[key] = record
    atomic_write_json(_affine8_group_cache_path, payload)


def tune_mlx_affine8_group_size(weights, *, trials=15, objective="balanced"):
    """Measure and cache the fastest strict affine-INT8 decode group."""
    if trials < 3:
        raise ValueError("affine8 group tuning requires at least three trials")
    if objective not in {"balanced", "throughput"}:
        raise ValueError("affine8 group objective must be balanced or throughput")
    weights = tuple(weights)
    if not weights:
        raise ValueError("affine8 group tuning requires at least one weight")
    if any(weight.ndim != 2 for weight in weights):
        raise ValueError("affine8 group tuning requires matrix weights")
    group_sizes = tuple(
        group
        for group in _AFFINE8_GROUP_SIZES
        if all(weight.shape[-1] % group == 0 for weight in weights)
    )
    if 64 not in group_sizes:
        raise ValueError("affine8 auto groups require input features divisible by 64")

    import mlx.core as mx

    key = _affine8_group_tuning_key(weights, mx, objective)
    with _affine8_group_lock:
        cached = _restore_affine8_group_tuning(key, group_sizes, objective)
    if cached is not None:
        return cached["group_size"], {**cached, "cached": True}

    representatives = {}
    for weight in weights:
        representatives.setdefault((tuple(weight.shape), str(weight.dtype)), weight)
    samples = {group: [] for group in group_sizes}
    native_samples = []
    errors = {group: [] for group in group_sizes}
    for weight in representatives.values():
        candidates = {
            group: MLXCompressedDownWeight.quantize(
                weight,
                format="affine8",
                group_size=group,
            )
            for group in group_sizes
        }
        values = mx.ones((1, weight.shape[-1]), dtype=weight.dtype)
        reference = values @ weight.T
        mx.eval(reference)
        for group in group_sizes:
            for _ in range(6):
                mx.eval(candidates[group](values))
            actual = candidates[group](values)
            error = mx.mean(mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32)))
            errors[group].append(float(error.item()))
        for _ in range(6):
            mx.eval(values @ weight.T)
        for trial in range(trials):
            offset = trial % len(group_sizes)
            order = group_sizes[offset:] + group_sizes[:offset]
            for group in order:
                start = time.perf_counter_ns()
                for _ in range(_AFFINE8_GROUP_REPETITIONS):
                    mx.eval(candidates[group](values))
                samples[group].append((time.perf_counter_ns() - start) / _AFFINE8_GROUP_REPETITIONS)
            start = time.perf_counter_ns()
            for _ in range(_AFFINE8_GROUP_REPETITIONS):
                mx.eval(values @ weight.T)
            native_samples.append((time.perf_counter_ns() - start) / _AFFINE8_GROUP_REPETITIONS)

    timings = {group: round(statistics.median(samples[group])) for group in group_sizes}
    errors = {group: statistics.median(errors[group]) for group in group_sizes}
    fastest_time = min(timings.values())
    near_peak = tuple(
        group
        for group in group_sizes
        if timings[group] <= fastest_time * (1.0 + _AFFINE8_GROUP_NEAR_PEAK_MARGIN)
    )
    if objective == "throughput":
        group_size = min(group_sizes, key=lambda group: (timings[group], -group))
    else:
        group_size = min(near_peak, key=lambda group: (errors[group], timings[group], group))
    record = {
        "group_size": group_size,
        "mean_absolute_error": {str(group): errors[group] for group in group_sizes},
        "median_nanoseconds": {str(group): timings[group] for group in group_sizes},
        "native_median_nanoseconds": round(statistics.median(native_samples)),
        "objective": objective,
    }
    with _affine8_group_lock:
        _write_affine8_group_tuning(key, record)
    return group_size, {**record, "cached": False}


def mlx_compressed_down_residual(values, weight, residual):
    """Execute a compressed down projection and typed residual epilogue."""
    if not isinstance(weight, MLXCompressedDownWeight):
        raise TypeError("weight must be an MLXCompressedDownWeight")
    if values.ndim < 2 or residual.ndim != values.ndim:
        raise ValueError("compressed down projection requires rank-compatible inputs")
    if values.shape[-1] != weight.shape[1]:
        raise ValueError("compressed down input features must match the weight")
    if residual.shape != (*values.shape[:-1], weight.shape[0]):
        raise ValueError("compressed down residual shape must match the projected output")
    if values.dtype != residual.dtype:
        raise TypeError("compressed down projection requires matching input dtypes")
    if str(values.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16"):
        raise TypeError("compressed down projection requires bfloat16 or float16")
    return weight(values) + residual


def mlx_compressed_down_backend_signature():
    """Return the implementation identity for model-plan cache invalidation."""
    return stable_digest(
        {
            "dispatch": inspect.getsource(mlx_compressed_down_residual),
            "quantize": inspect.getsource(MLXCompressedDownWeight.quantize),
            "run": inspect.getsource(MLXCompressedDownWeight.__call__),
            "tune_group": inspect.getsource(tune_mlx_affine8_group_size),
        }
    )


__all__ = [
    "MLXCompressedDownWeight",
    "mlx_compressed_down_backend_signature",
    "mlx_compressed_down_residual",
    "tune_mlx_affine8_group_size",
]
