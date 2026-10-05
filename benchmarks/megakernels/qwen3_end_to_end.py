"""Measure actual Qwen3 greedy generation with GPU-wide DSL kernels and native MLX."""

import argparse
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

from benchmarks.megakernels.qwen3 import (
    DEFAULT_MODEL,
    _cache_fidelity,
    _fidelity_tolerances,
    _local_snapshot,
    _logit_fidelity,
    _require_precision_environment,
    _sha256,
    _validate_checkpoint_config,
)


class GenerationValidationFailure(AssertionError):
    def __init__(self, checks):
        self.checks = checks
        super().__init__(f"generation correctness failed: {checks[-1]}")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--threads", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m5-qwen3-gpu-wide-end-to-end.json"),
    )
    arguments = parser.parse_args(argv)
    for name, values, minimum in (
        ("prompt lengths", arguments.prompt_lengths, 1),
        ("output lengths", arguments.output_lengths, 2),
    ):
        if any(value < minimum for value in values):
            parser.error(f"{name} must be at least {minimum}")
        if len(set(values)) != len(values):
            parser.error(f"{name} must be distinct")
    if arguments.trials < 3:
        parser.error("at least three alternating paired trials are required")
    if arguments.seed < 0:
        parser.error("seed must be nonnegative")
    return arguments


def _benchmark_checkout():
    from benchmarks.common.checkout import load_kernel

    root = Path(__file__).resolve().parents[2]
    load_kernel(root, "megakernels.qwen3_staged")
    return root


def _check_step(native_output, native_cache, candidate, token, position, dtype):
    logit_tolerance, cache_tolerance = _fidelity_tolerances(dtype)
    try:
        logits = _logit_fidelity(native_output, candidate.logits.numpy(), logit_tolerance)
        cache = _cache_fidelity(native_cache, candidate, position + 1, cache_tolerance)
        selected = candidate.greedy()
        selection = {"passed": selected == logits["mlx_next_token"], "token": selected}
        return {
            "position": position,
            "input_token": token,
            "logits": logits,
            "cache": cache,
            "gpu_greedy_selection": selection,
            "passed": logits["passed"] and cache["passed"] and selection["passed"],
        }
    except ValueError as error:
        return {"position": position, "input_token": token, "passed": False, "error": str(error)}


def _validate_workload(model, candidate, prompt, output_tokens, dtype):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    native_cache = make_prompt_cache(model)
    native_output = model(mx.array([prompt], dtype=mx.int32), cache=native_cache)
    mx.eval(native_output, [layer.state for layer in native_cache])
    prompt_logits = np.asarray(native_output)
    candidate.reset()
    checks = []
    generated = []
    for position, token in enumerate(prompt):
        candidate.forward(token, position, project=True)
        check = _check_step(
            prompt_logits[:, position, :], native_cache, candidate, token, position, dtype
        )
        checks.append(check)
        if not check["passed"]:
            raise GenerationValidationFailure(checks)
    generated.append(checks[-1]["gpu_greedy_selection"]["token"])
    for offset in range(output_tokens - 1):
        position = len(prompt) + offset
        token = generated[-1]
        native_output = model(mx.array([[token]], dtype=mx.int32), cache=native_cache)
        mx.eval(native_output, [layer.state for layer in native_cache])
        candidate.forward(token, position, project=True)
        check = _check_step(
            np.asarray(native_output), native_cache, candidate, token, position, dtype
        )
        checks.append(check)
        if not check["passed"]:
            raise GenerationValidationFailure(checks)
        generated.append(check["gpu_greedy_selection"]["token"])
    return {
        "passed": True,
        "checks": checks,
        "actual_output_tokens": len(generated),
        "generated_token_ids": generated,
        "validation_prefill": "native batched prompt logits versus sequential DSL prompt forwards",
        "validation_decode": "every forward advances its own complete KV cache",
    }


def _generation_sample(start, first_token, end, generated):
    return {
        "total_wall_seconds": (end - start) * 1e-9,
        "time_to_first_token_seconds": (first_token - start) * 1e-9,
        "decode_wall_seconds": (end - first_token) * 1e-9,
        "actual_output_tokens": len(generated),
        "generated_token_ids": generated,
    }


def _native_generation(model, prompt, output_tokens):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    start = time.perf_counter_ns()
    cache = make_prompt_cache(model)
    if len(prompt) > 1:
        model(mx.array([prompt[:-1]], dtype=mx.int32), cache=cache)
        mx.eval([layer.state for layer in cache])
    generated = []
    token = prompt[-1]
    for _ in range(output_tokens):
        output = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        selected = mx.argmax(output[:, -1, :], axis=-1)
        mx.eval(selected, [layer.state for layer in cache])
        token = int(selected.item())
        mx.synchronize()
        generated.append(token)
        if len(generated) == 1:
            first_token = time.perf_counter_ns()
    end = time.perf_counter_ns()
    return _generation_sample(start, first_token, end, generated)


def _candidate_generation(candidate, prompt, output_tokens):
    start = time.perf_counter_ns()
    candidate.reset()
    for position, token in enumerate(prompt[:-1]):
        candidate.forward(token, position, project=False)
    generated = []
    token = prompt[-1]
    for offset in range(output_tokens):
        candidate.forward(token, len(prompt) - 1 + offset, project=True)
        token = candidate.greedy()
        generated.append(token)
        if len(generated) == 1:
            first_token = time.perf_counter_ns()
    end = time.perf_counter_ns()
    return _generation_sample(start, first_token, end, generated)


def _check_generated(sample, expected):
    if sample["actual_output_tokens"] != len(expected) or sample["generated_token_ids"] != expected:
        raise GenerationValidationFailure(
            [{"expected_generated_token_ids": expected, "actual": sample["generated_token_ids"]}]
        )


def _summarize(samples):
    if len(samples) < 3:
        raise ValueError("at least three paired samples are required")
    metrics = ("total_wall_seconds", "time_to_first_token_seconds", "decode_wall_seconds")
    expected = samples[0]["MLX"]["generated_token_ids"]
    if len(expected) < 2 or any(type(token) is not int or token < 0 for token in expected):
        raise ValueError("generation requires at least two valid token IDs")
    for index, sample in enumerate(samples):
        order = ["MLX", "meTile"] if index % 2 == 0 else ["meTile", "MLX"]
        if sample["order"] != order:
            raise ValueError("paired samples must alternate measurement order")
        for backend in ("MLX", "meTile"):
            trial = sample[backend]
            _check_generated(trial, expected)
            for metric in metrics:
                duration = trial[metric]
                if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("sample durations must be positive and finite")
            if not math.isclose(
                trial["total_wall_seconds"],
                trial["time_to_first_token_seconds"] + trial["decode_wall_seconds"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("TTFT plus decode time must equal total generation time")
    summary = {}
    for metric in metrics:
        summary[metric] = {
            "mlx": statistics.median(sample["MLX"][metric] for sample in samples),
            "metile": statistics.median(sample["meTile"][metric] for sample in samples),
            "paired_speedup": statistics.median(
                sample["MLX"][metric] / sample["meTile"][metric] for sample in samples
            ),
        }
    summary["decode_tokens_per_second"] = {
        backend: statistics.median(
            (len(expected) - 1) / sample[label]["decode_wall_seconds"] for sample in samples
        )
        for backend, label in (("mlx", "MLX"), ("metile", "meTile"))
    }
    return summary


def _measure_workload(model, candidate, prompt, output_tokens, trials, expected):
    import mlx.core as mx

    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    operations = (
        ("MLX", lambda: _native_generation(model, prompt, output_tokens)),
        ("meTile", lambda: _candidate_generation(candidate, prompt, output_tokens)),
    )
    for _ in range(2):
        for _, operation in operations:
            mx.synchronize()
            device.sync()
            _check_generated(operation(), expected)
    samples = []
    for trial in range(trials):
        ordered = operations if trial % 2 == 0 else operations[::-1]
        sample = {"trial": trial, "order": [name for name, _ in ordered]}
        for name, operation in ordered:
            mx.synchronize()
            device.sync()
            sample[name] = operation()
            _check_generated(sample[name], expected)
        samples.append(sample)
    return {"samples": samples, "medians": _summarize(samples)}


def _workload_results(arguments, model, candidate, prompts):
    results = []
    for prompt in prompts:
        for output_tokens in arguments.output_lengths:
            result = {
                "prompt_tokens": len(prompt),
                "prompt_token_ids": prompt,
                "requested_output_tokens": output_tokens,
                "status": "validated",
            }
            results.append(result)
            print(f"Validating {len(prompt)} prompt / {output_tokens} output tokens...", flush=True)
            try:
                result["correctness"] = _validate_workload(
                    model, candidate, prompt, output_tokens, arguments.dtype
                )
            except GenerationValidationFailure as failure:
                result.update(status="validation_failed", correctness={"checks": failure.checks})
                return "validation_failed", results
    for result in results:
        try:
            measured = _measure_workload(
                model,
                candidate,
                result["prompt_token_ids"],
                result["requested_output_tokens"],
                arguments.trials,
                result["correctness"]["generated_token_ids"],
            )
        except GenerationValidationFailure as failure:
            for previous in results:
                previous.pop("samples", None)
                previous.pop("medians", None)
                previous["status"] = "validated"
            result.update(status="timing_validation_failed", timing_failure=failure.checks)
            return "validation_failed", results
        result.update(measured, status="ok")
        total = measured["medians"]["total_wall_seconds"]
        print(
            f"prompt={result['prompt_tokens']}, output={result['requested_output_tokens']}: "
            f"MLX {total['mlx'] * 1000:.1f} ms, meTile {total['metile'] * 1000:.1f} ms; "
            f"paired speedup {total['paired_speedup']:.4f}x",
            flush=True,
        )
    return "ok", results


def run(arguments):
    _require_precision_environment(arguments.dtype)
    root = _benchmark_checkout()

    import mlx.core as mx
    from mlx_lm.utils import load_model

    from benchmarks.common.checkout import implementation_hash
    from benchmarks.megakernels import qwen3, qwen3_runtime, qwen3_staged_runtime
    from metile.runtime.metal_device import MetalDevice

    compiler_hash = implementation_hash(root)
    source_paths = [
        Path(__file__),
        Path(qwen3.__file__),
        Path(qwen3_runtime.__file__),
        Path(qwen3_staged_runtime.__file__),
        root / "benchmarks/common/checkout.py",
    ]
    source_hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths}
    snapshot = _local_snapshot(arguments.model, arguments.revision)
    checkpoint_config = json.loads((snapshot / "config.json").read_text())
    _validate_checkpoint_config(checkpoint_config)
    model, _ = load_model(snapshot)
    model.set_dtype(getattr(mx, arguments.dtype))
    mx.eval(model.parameters())
    capacity = max(arguments.prompt_lengths) + max(arguments.output_lengths) - 1
    candidate = qwen3_staged_runtime.Qwen3Staged(model, capacity, threads=arguments.threads)
    candidate.prepare()
    generator = np.random.default_rng(arguments.seed)
    tokens = generator.integers(
        1, model.args.vocab_size, size=max(arguments.prompt_lengths)
    ).tolist()
    prompts = [tokens[:length] for length in arguments.prompt_lengths]
    status, results = _workload_results(arguments, model, candidate, prompts)
    for result in results:
        result["candidate_dispatches_per_generation"] = (
            result["prompt_tokens"] - 1
        ) * candidate.dispatches_per_forward(project=False, include_greedy=False) + result[
            "requested_output_tokens"
        ] * candidate.dispatches_per_forward(project=True, include_greedy=True)
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
            "policy": "Identical converted checkpoint weights; complete logits and all valid KV entries pass recorded dtype-specific tolerances, and all greedy output IDs agree exactly.",
        },
        "weights": {
            "source_dtype": checkpoint_config.get("torch_dtype") or checkpoint_config.get("dtype"),
            "both_backends": arguments.dtype,
            "same_values": True,
            "quantized": False,
        },
        "workload": {
            "batch": 1,
            "prompt_lengths": arguments.prompt_lengths,
            "output_lengths": arguments.output_lengths,
            "capacity": capacity,
            "seed": arguments.seed,
            "prompt_kind": "fixed synthetic token IDs; shorter prompts share the longer prefix",
            "trials": arguments.trials,
            "warmup_generations_per_backend_per_workload": 2,
            "sampling": "greedy GPU argmax; EOS ignored; exact requested output count",
            "tokenizer": "not used; benchmark starts at token IDs and ends at token IDs",
        },
        "execution": {
            "candidate_backend": "GPU-wide DSL kernels separated by dispatch boundaries",
            "native_backend": "unmodified MLX-LM Qwen3 with batched prefill",
            "candidate_prefill": "sequential tokens; vocabulary projection only on the last prompt token",
            "native_prefill": "batched prompt except final token; evaluate only KV state, allowing MLX to prune unused final-layer work after its KV update; candidate sequential prefill computes the full decoder body",
            "scope": "embedding, every decoder layer, final RMSNorm, full vocabulary logits, GPU argmax",
            "intergroup_ordering": "dispatch boundaries; no unsupported grid barrier or spin waiting",
            "single_dispatch": False,
            "maximum_projection_threadgroups": candidate.max_projection_threadgroups,
            "candidate_dispatches_per_prefill_token": candidate.dispatches_per_forward(
                project=False, include_greedy=False
            ),
            "candidate_dispatches_per_generated_token": candidate.dispatches_per_forward(
                project=True, include_greedy=True
            ),
        },
        "timing": {
            "metric": "synchronized token-ID-to-token-ID generation wall seconds",
            "included": [
                "fresh native cache construction or reusable candidate cache reset",
                "all prefill and autoregressive decode model work and KV updates",
                "input token allocations/updates, graph/dispatch submission and synchronization",
                "full vocabulary projection for each returned output token",
                "GPU greedy selection and host token retrieval for each returned token",
            ],
            "excluded": [
                "checkpoint loading, weight conversion/packing, rotary-table preparation and compilation",
                "candidate persistent buffer allocation and pipeline preparation",
                "tokenization, text decoding and correctness comparisons",
                "two complete warmup generations per backend per workload",
            ],
            "ttft": "start of cache reset/construction through first returned token ID, including prefill",
            "decode": "first returned token ID through final returned token ID; output_count - 1 forwards",
            "allocation_difference": "candidate reuses persistent buffers and zeros KV inside timing; native MLX manages intermediate, output and growing KV allocations inside timing",
            "aggregation": "median of within-trial MLX/meTile ratios, separately for each metric",
            "correctness_gate": "all workloads must pass complete logits/KV checks before any timing; every timed and warmup generated sequence must exactly match validation",
            "limitations": "synthetic fixed prompts, batch one, short contexts, steady-state warmed pipelines; not text-serving or cold-start latency",
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
        print("Correctness failed; no timings saved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
