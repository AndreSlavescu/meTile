"""Compare one complete Qwen3 decode forward with native MLX, using cached weights."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
LOGIT_TOLERANCE = {"rtol": 0.02, "atol": 0.05}
CACHE_TOLERANCE = {"rtol": 0.02, "atol": 0.02}
FP32_TOLERANCE = {"rtol": 0.001, "atol": 0.001}
VALIDATION_STEPS = 3


class ValidationFailure(AssertionError):
    def __init__(self, context, checks):
        self.context = context
        self.checks = checks
        super().__init__(f"correctness failed at context {context}: {checks[-1]}")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--contexts", type=int, nargs="+", default=[0, 16])
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--threads", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/m5-qwen3-megakernel.json")
    )
    arguments = parser.parse_args(argv)
    if any(context < 0 for context in arguments.contexts):
        parser.error("contexts must be nonnegative")
    if len(set(arguments.contexts)) != len(arguments.contexts):
        parser.error("contexts must be distinct")
    if arguments.trials < 3:
        parser.error("at least three paired trials are required")
    return arguments


def _array_fidelity(reference, actual, tolerance):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} versus {actual.shape}")
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("correctness requires nonempty, finite arrays")
    difference = np.abs(actual - reference)
    return {
        "passed": bool(np.allclose(actual, reference, equal_nan=False, **tolerance)),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        **tolerance,
    }


def _logit_fidelity(reference, actual, tolerance=None):
    reference = np.asarray(reference).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    result = _array_fidelity(reference, actual, LOGIT_TOLERANCE if tolerance is None else tolerance)
    result["mlx_next_token"] = int(np.argmax(reference))
    result["metile_next_token"] = int(np.argmax(actual))
    result["greedy_token_equal"] = result["mlx_next_token"] == result["metile_next_token"]
    result["passed"] = result["passed"] and result["greedy_token_equal"]
    return result


def _summarize(samples):
    if len(samples) < 3:
        raise ValueError("at least three paired samples are required")
    for index, sample in enumerate(samples):
        expected_order = ["MLX", "meTile"] if index % 2 == 0 else ["meTile", "MLX"]
        if sample.get("order") != expected_order:
            raise ValueError("paired samples must alternate measurement order")
        for key in ("mlx_wall_seconds", "metile_wall_seconds"):
            elapsed = sample[key]
            if isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("sample durations must be positive and finite")
    return {
        "mlx_wall_seconds": statistics.median(sample["mlx_wall_seconds"] for sample in samples),
        "metile_wall_seconds": statistics.median(
            sample["metile_wall_seconds"] for sample in samples
        ),
        "paired_speedup": statistics.median(
            sample["mlx_wall_seconds"] / sample["metile_wall_seconds"] for sample in samples
        ),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_snapshot(model, revision):
    from huggingface_hub import snapshot_download

    supplied_path = Path(model).expanduser()
    if supplied_path.is_dir():
        return supplied_path.resolve()
    return Path(snapshot_download(model, revision=revision, local_files_only=True))


def _benchmark_checkout():
    from benchmarks.common.checkout import load_kernel

    root = Path(__file__).resolve().parents[2]
    load_kernel(root, "megakernels.qwen3")
    return root


def _validate_checkpoint_config(config):
    if not isinstance(config, dict) or config.get("model_type") != "qwen3":
        raise ValueError("this benchmark requires a dense Qwen3 checkpoint")
    if "quantization" in config or "quantization_config" in config:
        raise ValueError("this benchmark requires an unquantized checkpoint")
    if "model_file" in config:
        raise ValueError("custom executable model files are not supported")


def _require_precision_environment(dtype):
    if dtype == "float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
        raise RuntimeError(
            "float32 comparisons require MLX_ENABLE_TF32=0 before importing MLX; run "
            "MLX_ENABLE_TF32=0 python -m benchmarks.megakernels.qwen3 --dtype float32"
        )


def _fidelity_tolerances(dtype):
    if dtype == "float32":
        return FP32_TOLERANCE, FP32_TOLERANCE
    if dtype == "float16":
        return LOGIT_TOLERANCE, CACHE_TOLERANCE
    raise ValueError("supported comparison dtypes are float16 and float32")


def _prefix_cache(model, prefix):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    if prefix:
        output = model(mx.array([prefix], dtype=mx.int32), cache=cache)
        mx.eval(output)
    return cache


def _padded_cache(prefix_cache, capacity):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    result = []
    for previous in prefix_cache:
        current = KVCache()
        if previous.keys is not None:
            context = previous.offset
            key_shape = (*previous.keys.shape[:2], capacity, previous.keys.shape[-1])
            value_shape = (*previous.values.shape[:2], capacity, previous.values.shape[-1])
            prefix_keys = np.array(previous.keys[:, :, :context])
            prefix_values = np.array(previous.values[:, :, :context])
            keys = np.zeros(key_shape, dtype=prefix_keys.dtype)
            values = np.zeros(value_shape, dtype=prefix_values.dtype)
            keys[:, :, :context] = prefix_keys
            values[:, :, :context] = prefix_values
            current.keys, current.values = mx.array(keys), mx.array(values)
            current.offset = context
            mx.eval(current.keys, current.values)
        result.append(current)
    return result


def _cache_fidelity(native_cache, candidate, valid_tokens, tolerance=None):
    tolerance = CACHE_TOLERANCE if tolerance is None else tolerance
    storage = candidate.cache.numpy()
    maximum = 0.0
    passed = True
    for layer_index, layer in enumerate(native_cache):
        for kind, native in enumerate((layer.keys, layer.values)):
            reference = np.array(native[:, :, :valid_tokens])[0]
            actual = storage[kind, layer_index, :, :valid_tokens]
            fidelity = _array_fidelity(reference, actual, tolerance)
            maximum = max(maximum, fidelity["max_absolute_error"])
            passed = passed and fidelity["passed"]
    return {
        "passed": passed,
        "max_absolute_error": maximum,
        "layers_checked": len(native_cache),
        "valid_tokens_checked": valid_tokens,
        "keys_and_values_checked": True,
        **tolerance,
    }


def _validate_context(model, candidate, prefix_cache, token, context, capacity, dtype="float16"):
    import mlx.core as mx

    native_cache = _padded_cache(prefix_cache, capacity)
    candidate.load_cache(prefix_cache)
    logit_tolerance, cache_tolerance = _fidelity_tolerances(dtype)
    checks = []
    for step in range(VALIDATION_STEPS):
        position = context + step
        native_output = model(mx.array([[token]], dtype=mx.int32), cache=native_cache)
        mx.eval(native_output)
        candidate.prepare(token, position)
        try:
            logits = _logit_fidelity(
                np.array(native_output), candidate.logits.numpy(), logit_tolerance
            )
            cache = _cache_fidelity(native_cache, candidate, position + 1, cache_tolerance)
        except ValueError as error:
            checks.append({"position": position, "input_token": token, "error": str(error)})
            raise ValidationFailure(context, checks) from error
        checks.append(
            {"position": position, "input_token": token, "logits": logits, "cache": cache}
        )
        if not logits["passed"] or not cache["passed"]:
            raise ValidationFailure(context, checks)
        token = logits["mlx_next_token"]
    return checks


def _measure_context(model, candidate, prefix_cache, token, context, capacity, trials):
    import mlx.core as mx

    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    native_cache = _padded_cache(prefix_cache, capacity)
    inputs = mx.array([[token]], dtype=mx.int32)
    mx.eval(inputs)
    candidate.load_cache(prefix_cache)
    dispatch = candidate.prepare(token, context)

    def native():
        output = model(inputs, cache=native_cache)
        mx.eval(output)
        mx.synchronize()

    def generated():
        dispatch()
        device.sync()

    def restore_native_position():
        for layer in native_cache:
            layer.offset = context

    for _ in range(3):
        restore_native_position()
        native()
        generated()
    if any(layer.keys.shape[2] <= context for layer in native_cache):
        raise RuntimeError("native KV capacity was not prepared before timing")
    initial_capacity = [(layer.keys.shape[2], layer.values.shape[2]) for layer in native_cache]
    samples = []
    for trial in range(trials):
        ordered = (("MLX", native), ("meTile", generated))
        if trial % 2:
            ordered = ordered[::-1]
        sample = {"trial": trial, "order": [label for label, _ in ordered]}
        for label, function in ordered:
            restore_native_position()
            mx.synchronize()
            device.sync()
            start = time.perf_counter_ns()
            function()
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            sample["mlx_wall_seconds" if label == "MLX" else "metile_wall_seconds"] = elapsed
        samples.append(sample)
    if [(layer.keys.shape[2], layer.values.shape[2]) for layer in native_cache] != initial_capacity:
        raise RuntimeError("native KV cache grew inside the timed region")
    return {
        "context_tokens": context,
        "input_token": token,
        "samples": samples,
        "medians": _summarize(samples),
    }


def _context_results(arguments, model, candidate, capacity, tokens):
    results = []
    prepared = []
    for context in arguments.contexts:
        print(
            f"Validating context {context} ({VALIDATION_STEPS} autoregressive steps)...", flush=True
        )
        prefix_cache = _prefix_cache(model, tokens[:context])
        try:
            correctness = _validate_context(
                model, candidate, prefix_cache, tokens[context], context, capacity, arguments.dtype
            )
        except ValidationFailure as failure:
            results.append(
                {
                    "context_tokens": context,
                    "input_token": tokens[context],
                    "status": "validation_failed",
                    "correctness": failure.checks,
                    "error": str(failure),
                }
            )
            return "validation_failed", results
        result = {
            "context_tokens": context,
            "input_token": tokens[context],
            "status": "validated",
            "correctness": correctness,
        }
        results.append(result)
        prepared.append(prefix_cache)
    for result, prefix_cache in zip(results, prepared):
        context = result["context_tokens"]
        result.update(
            _measure_context(
                model, candidate, prefix_cache, tokens[context], context, capacity, arguments.trials
            )
        )
        result["status"] = "ok"
        medians = result["medians"]
        print(
            f"context={context}: MLX {medians['mlx_wall_seconds'] * 1000:.3f} ms, "
            f"meTile {medians['metile_wall_seconds'] * 1000:.3f} ms, "
            f"paired speedup {medians['paired_speedup']:.4f}x",
            flush=True,
        )
    return "ok", results


def run(arguments):
    _require_precision_environment(arguments.dtype)
    root = _benchmark_checkout()

    import mlx.core as mx
    from mlx_lm.utils import load_model

    from benchmarks.common.checkout import implementation_hash
    from benchmarks.megakernels import qwen3_runtime
    from metile.runtime.metal_device import MetalDevice

    compiler_hash = implementation_hash(root)
    source_paths = [Path(__file__), Path(qwen3_runtime.__file__)]
    source_hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths}
    snapshot = _local_snapshot(arguments.model, arguments.revision)
    checkpoint_config = json.loads((snapshot / "config.json").read_text())
    _validate_checkpoint_config(checkpoint_config)
    model, _ = load_model(snapshot)
    model.set_dtype(getattr(mx, arguments.dtype))
    mx.eval(model.parameters())
    capacity = max(arguments.contexts) + VALIDATION_STEPS
    candidate = qwen3_runtime.Qwen3Megakernel(model, capacity, threads=arguments.threads)
    generator = np.random.default_rng(arguments.seed)
    tokens = generator.integers(1, model.args.vocab_size, size=capacity).tolist()
    status, results = _context_results(arguments, model, candidate, capacity, tokens)
    if compiler_hash != implementation_hash(root):
        raise RuntimeError("compiler or kernel sources changed during measurement")
    if source_hashes != {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths}:
        raise RuntimeError("benchmark or runtime sources changed during measurement")
    return {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": arguments.model,
        "checkpoint_revision": snapshot.name,
        "checkpoint_sha256": {
            path.name: _sha256(path)
            for path in sorted([snapshot / "config.json", *snapshot.glob("model*.safetensors")])
        },
        "hardware": mx.device_info(),
        "software": {
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "platform": platform.platform(),
            "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
        },
        "metal_compiler": MetalDevice.get().metal_compiler_version,
        "source_sha256": source_hashes,
        "compiler_and_kernels_sha256": compiler_hash,
        "geometry": {**candidate.config, "dtype": arguments.dtype},
        "parameter_count": candidate.parameter_count,
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "bitwise_exact": False,
            "storage_dtype": arguments.dtype,
            "accumulation_dtype": "float32",
            "policy": "Identical converted checkpoint weights and storage dtype; reduction order may differ. Full logits, greedy tokens and KV caches must pass the recorded tolerances before timing.",
        },
        "weights": {
            "source_dtype": checkpoint_config.get("torch_dtype") or checkpoint_config.get("dtype"),
            "both_backends": arguments.dtype,
            "same_values": True,
            "quantized": False,
        },
        "workload": {
            "batch": 1,
            "query_tokens": 1,
            "contexts": arguments.contexts,
            "capacity": capacity,
            "seed": arguments.seed,
            "synthetic_tokens": tokens,
            "trials": arguments.trials,
            "validation_steps_per_context": VALIDATION_STEPS,
        },
        "execution": {
            "metile_dispatches_per_forward": 1,
            "metile_threadgroups": 1,
            "native_backend": "unpatched MLX-LM Qwen3",
            "scope": "embedding, every decoder layer, final RMSNorm and full vocabulary logits",
        },
        "timing": {
            "metric": "synchronized wall seconds per fixed-context, one-token full forward",
            "included": [
                "native model graph construction/evaluation and synchronization",
                "prepared meTile dispatch and synchronization",
                "KV update at the same fixed position",
                "complete vocabulary logits",
            ],
            "excluded": [
                f"checkpoint loading and {arguments.dtype} conversion",
                "weight packing and compilation",
                "prefill and cache preparation",
                "input token allocation",
                "cache offset reset",
                "argmax, sampling and tokenization",
            ],
            "allocation_difference": "meTile reuses prepared buffers; native MLX manages intermediate/output allocations. Native KV capacity is warmed and does not grow during timing.",
            "aggregation": "median of within-trial MLX/meTile wall-time ratios",
            "limitation": "same-position replay, not autoregressive generation throughput or end-to-end latency; correctness checks separately advance three tokens per context",
        },
        "results": results,
    }


def main(argv=None):
    arguments = _arguments(argv)
    report = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {arguments.output}")
    if report["status"] != "ok":
        print("Validation failed; no timings collected.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
