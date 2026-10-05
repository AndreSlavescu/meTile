"""Tune Qwen3 prefill chunks, then compare long generation with a native MLX model loop.

The primary document is held out from automatic chunk-size selection, not
from implementation development. It was exercised during 4096-token
correctness diagnostics. The reference is an explicit native-model greedy
loop, not the pipelined mlx_lm.stream_generate API.
"""

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
    _array_fidelity,
    _cache_fidelity,
    _fidelity_tolerances,
    _local_snapshot,
    _logit_fidelity,
    _require_precision_environment,
    _sha256,
    _validate_checkpoint_config,
)
from benchmarks.megakernels.qwen3_end_to_end import (
    GenerationValidationFailure,
    _candidate_generation,
    _check_generated,
    _generation_sample,
    _native_generation,
)

METRICS = ("total_wall_seconds", "time_to_first_token_seconds", "decode_wall_seconds")
DEFAULT_PROMPT_SOURCES = (
    "docs/guide/language.rst",
    "docs/guide/tensor-memory.rst",
    "docs/guide/tile-ops.rst",
    "docs/guide/memory.rst",
)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--tuning-prompt-tokens", type=int, default=2048)
    parser.add_argument("--tuning-output-tokens", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tuning-trials", type=int, default=5)
    parser.add_argument("--threads", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--attention-backend", choices=("tiled", "matrix"), default="tiled")
    parser.add_argument(
        "--projection-backend", choices=("simdgroup", "tensor_ops"), default="simdgroup"
    )
    parser.add_argument("--projection-tile", type=int, nargs=3, default=[64, 64, 32])
    parser.add_argument("--cache-check-interval", type=int, default=64)
    parser.add_argument("--sequential-baseline", action="store_true")
    parser.add_argument(
        "--no-lossless-decode-weights",
        dest="lossless_decode_weights",
        action="store_false",
        help="retain original projection storage even when FP32 weights pack losslessly",
    )
    parser.add_argument("--prompt-sources", nargs="+", default=list(DEFAULT_PROMPT_SOURCES))
    parser.add_argument(
        "--tuning-prompt-sources",
        nargs="+",
        default=["docs/guide/architecture.rst", "docs/guide/graph-fusion.rst"],
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output is None:
        label = "matrix" if arguments.attention_backend == "matrix" else "chunked"
        arguments.output = Path(f"benchmarks/results/m5-qwen3-{label}-prefill-end-to-end.json")
    if any(size < 8 or size % 8 for size in arguments.projection_tile):
        parser.error("projection tile dimensions must be positive multiples of eight")
    for name in ("prompt_tokens", "tuning_prompt_tokens", "cache_check_interval"):
        if getattr(arguments, name) < 1:
            parser.error(f"{name.replace('_', ' ')} must be positive")
    for name in ("output_tokens", "tuning_output_tokens"):
        if getattr(arguments, name) < 2:
            parser.error(f"{name.replace('_', ' ')} must be at least two")
    if arguments.trials < 5 or arguments.tuning_trials < 5:
        parser.error("at least five trials are required for tuning and held-out measurement")
    if any(size < 1 for size in arguments.chunk_sizes):
        parser.error("chunk sizes must be positive")
    if len(set(arguments.chunk_sizes)) != len(arguments.chunk_sizes):
        parser.error("chunk sizes must be distinct")
    return arguments


def _benchmark_checkout():
    from benchmarks.common.checkout import load_kernel

    root = Path(__file__).resolve().parents[2]
    load_kernel(root, "megakernels.qwen3_prefill")
    return root


def _document_prompt(root, tokenizer, sources, length):
    paths = [(root / source).resolve() for source in sources]
    if len(set(paths)) != len(paths) or any(not path.is_relative_to(root) for path in paths):
        raise ValueError("prompt sources must be distinct files inside the selected checkout")
    text = "\n\n".join(path.read_text() for path in paths)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < length:
        raise ValueError(f"document has {len(tokens)} tokens but {length} were requested")
    selected = tokens[:length]
    if any(type(token) is not int or token < 0 for token in selected):
        raise ValueError("tokenizer must return nonnegative integer token IDs")
    return {
        "kind": "contiguous token prefix of local technical documentation",
        "sources_sha256": {path.relative_to(root).as_posix(): _sha256(path) for path in paths},
        "source_token_count": len(tokens),
        "token_offset": 0,
        "token_ids": selected,
        "actual_prompt_tokens": len(selected),
        "chat_template": False,
        "special_tokens_added": False,
    }


def _require_disjoint_prompts(tuning, held_out):
    if tuning["sources_sha256"].keys() & held_out["sources_sha256"].keys():
        raise ValueError("tuning and held-out prompts must use disjoint source files")
    if set(tuning["sources_sha256"].values()) & set(held_out["sources_sha256"].values()):
        raise ValueError("tuning and held-out sources must not duplicate the same file content")
    if tuning["token_ids"] == held_out["token_ids"]:
        raise ValueError("tuning and held-out token sequences must differ")


def _cache_slot_fidelity(native_cache, candidate, position, tolerance):
    storage = candidate.cache.numpy()
    maximum = 0.0
    passed = True
    for layer_index, layer in enumerate(native_cache):
        for kind, native in enumerate((layer.keys, layer.values)):
            fidelity = _array_fidelity(
                np.asarray(native[:, :, position : position + 1])[0],
                storage[kind, layer_index, :, position : position + 1],
                tolerance,
            )
            maximum = max(maximum, fidelity["max_absolute_error"])
            passed = passed and fidelity["passed"]
    return {
        "passed": passed,
        "max_absolute_error": maximum,
        "layers_checked": len(native_cache),
        "valid_tokens_checked": 1,
        "positions_checked": [position],
        "keys_and_values_checked": True,
        "full_prefix_checked": False,
        **tolerance,
    }


def _validate_workload(
    model, candidate, prompt, output_tokens, dtype, chunk_size, interval, sequential=False
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    logit_tolerance, cache_tolerance = _fidelity_tolerances(dtype)
    cache = make_prompt_cache(model)
    if len(prompt) > 1:
        model(mx.array([prompt[:-1]], dtype=mx.int32), cache=cache)
        mx.eval([layer.state for layer in cache])
    native = model(mx.array([[prompt[-1]]], dtype=mx.int32), cache=cache)
    mx.eval(native, [layer.state for layer in cache])
    prefix = prompt[:-1] if sequential else prompt
    candidate.reset()
    checks = []
    for start in range(0, len(prefix), chunk_size):
        chunk = prefix[start : start + chunk_size]
        if sequential:
            for offset, token in enumerate(chunk):
                candidate.forward(token, start + offset, project=False)
        else:
            candidate.prefill(chunk)
        valid = start + len(chunk)
        try:
            fidelity = _cache_fidelity(cache, candidate, valid, cache_tolerance)
            check = {
                "stage": "prefill",
                "valid_tokens": valid,
                "cache": {**fidelity, "full_prefix_checked": True},
                "passed": fidelity["passed"],
            }
        except ValueError as error:
            check = {
                "stage": "prefill",
                "valid_tokens": valid,
                "passed": False,
                "error": str(error),
            }
        checks.append(check)
        if not check["passed"]:
            raise GenerationValidationFailure(checks)
    boundary_checks = len(checks)
    if not sequential:
        candidate.reset()
        candidate.prefill(prompt)
    generated = []
    token = prompt[-1]
    for offset in range(output_tokens):
        position = len(prompt) - 1 + offset
        if offset:
            native = model(mx.array([[token]], dtype=mx.int32), cache=cache)
            mx.eval(native, [layer.state for layer in cache])
        if offset or sequential:
            candidate.forward(token, position, project=True)
        else:
            candidate.project_last_prefill()
        try:
            logits = _logit_fidelity(np.asarray(native), candidate.logits.numpy(), logit_tolerance)
            full_prefix = offset == 0 or (offset + 1) % interval == 0 or offset == output_tokens - 1
            if full_prefix:
                fidelity = {
                    **_cache_fidelity(cache, candidate, position + 1, cache_tolerance),
                    "full_prefix_checked": True,
                }
            else:
                fidelity = _cache_slot_fidelity(cache, candidate, position, cache_tolerance)
            selected = candidate.greedy()
            check = {
                "stage": "generation",
                "position": position,
                "input_token": token,
                "logits": logits,
                "cache": fidelity,
                "gpu_greedy_token": selected,
                "passed": logits["passed"]
                and fidelity["passed"]
                and selected == logits["mlx_next_token"],
            }
        except ValueError as error:
            check = {
                "stage": "generation",
                "position": position,
                "passed": False,
                "error": str(error),
            }
        if offset == 0 and not sequential:
            check["prefill_mode"] = "single_whole_prompt_call"
        checks.append(check)
        if not check["passed"]:
            raise GenerationValidationFailure(checks)
        generated.append(selected)
        token = selected
    result = {
        "passed": True,
        "checks": checks,
        "generated_token_ids": generated,
        "actual_output_tokens": len(generated),
        "validation_counts": {
            "chunk_boundary_cache_checks": boundary_checks,
            "whole_prompt_prefill_calls": int(not sequential),
            "generation_full_logit_checks": len(generated),
            "generation_cache_checks": len(generated),
            "generation_gpu_greedy_checks": len(generated),
            "post_first_token_decode_forwards": len(generated) - 1,
        },
    }
    if not sequential:
        result["whole_prompt_replay"] = {
            "passed": True,
            "prefill_calls": 1,
            "actual_prompt_tokens": len(prompt),
            "generation_check_index": boundary_checks,
            "shares_first_generation_check": True,
        }
    return result


def _chunked_generation(candidate, prompt, output_tokens):
    start = time.perf_counter_ns()
    candidate.reset()
    candidate.prefill(prompt)
    candidate.project_last_prefill()
    token = candidate.greedy()
    generated = [token]
    first_token = time.perf_counter_ns()
    for offset in range(output_tokens - 1):
        candidate.forward(token, len(prompt) + offset, project=True)
        token = candidate.greedy()
        generated.append(token)
    return _generation_sample(start, first_token, time.perf_counter_ns(), generated)


def _trial_order(trial, sequential=False):
    order = ["MLX", "chunked"] if trial % 2 == 0 else ["chunked", "MLX"]
    if sequential:
        order.insert(trial % 3, "sequential")
    return order


def _summarize(samples):
    if len(samples) < 5:
        raise ValueError("at least five paired trials are required")
    labels = ["MLX", "chunked"] + (["sequential"] if "sequential" in samples[0] else [])
    expected = samples[0]["MLX"]["generated_token_ids"]
    if len(expected) < 2:
        raise ValueError("at least two generated tokens are required")
    for index, sample in enumerate(samples):
        if sample["order"] != _trial_order(index, "sequential" in labels):
            raise ValueError("paired trials must alternate native/chunked order")
        for label in labels:
            observation = sample[label]
            _check_generated(observation, expected)
            for metric in METRICS:
                duration = observation[metric]
                if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("durations must be positive and finite")
            if not math.isclose(
                observation["total_wall_seconds"],
                observation["time_to_first_token_seconds"] + observation["decode_wall_seconds"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("TTFT plus decode must equal total duration")
    medians = {
        label: {
            metric: statistics.median(sample[label][metric] for sample in samples)
            for metric in METRICS
        }
        for label in labels
    }
    for label in labels:
        medians[label]["decode_tokens_per_second"] = statistics.median(
            (len(expected) - 1) / sample[label]["decode_wall_seconds"] for sample in samples
        )
    ratios = {
        "chunked_over_mlx": ("MLX", "chunked"),
        **(
            {
                "chunked_over_sequential": ("sequential", "chunked"),
                "sequential_over_mlx": ("MLX", "sequential"),
            }
            if "sequential" in labels
            else {}
        ),
    }
    return {
        "medians": medians,
        "paired_speedups": {
            name: {
                metric: statistics.median(
                    sample[numerator][metric] / sample[denominator][metric] for sample in samples
                )
                for metric in METRICS
            }
            for name, (numerator, denominator) in ratios.items()
        },
    }


def _measure(model, candidate, prompt, output_tokens, trials, expected, sequential=None):
    import mlx.core as mx

    from metile.runtime.metal_device import MetalDevice

    device = MetalDevice.get()
    operations = {
        "MLX": lambda: _native_generation(model, prompt, output_tokens),
        "chunked": lambda: _chunked_generation(candidate, prompt, output_tokens),
    }
    if sequential is not None:
        operations["sequential"] = lambda: _candidate_generation(sequential, prompt, output_tokens)
    for _ in range(2):
        for operation in operations.values():
            mx.synchronize()
            device.sync()
            _check_generated(operation(), expected)
    samples = []
    for trial in range(trials):
        order = _trial_order(trial, sequential is not None)
        sample = {"trial": trial, "order": order}
        for label in order:
            mx.synchronize()
            device.sync()
            sample[label] = operations[label]()
            _check_generated(sample[label], expected)
        samples.append(sample)
        print(
            f"  trial {trial + 1}/{trials}: native {sample['MLX']['total_wall_seconds']:.3f}s, chunked {sample['chunked']['total_wall_seconds']:.3f}s",
            flush=True,
        )
    return {"samples": samples, **_summarize(samples)}


def _select_chunk(tuning):
    if not tuning or any(result.get("status") != "ok" for result in tuning):
        raise ValueError("chunk selection requires a complete validated tuning sweep")
    return min(
        tuning,
        key=lambda result: (
            result["medians"]["chunked"]["time_to_first_token_seconds"],
            result["chunk_size"],
        ),
    )["chunk_size"]


def _discard_timings(report, failure):
    for phase in ("tuning", "held_out"):
        for result in report[phase]:
            for key in ("samples", "medians", "paired_speedups"):
                result.pop(key, None)
            result["status"] = "validated"
    report["status"] = "validation_failed"
    report["validation_failure"] = failure.checks
    report.pop("selected_chunk_size", None)


def _run_phases(arguments, model, factory, sequential_factory, tuning_prompt, held_out_prompt):
    report = {"status": "ok", "tuning": [], "held_out": []}
    capacity = max(
        len(tuning_prompt) + arguments.tuning_output_tokens - 1,
        len(held_out_prompt) + arguments.output_tokens - 1,
    )
    candidate = factory(capacity, arguments.chunk_sizes[0])
    report["candidate_weight_packing"] = candidate.config.get("decode_weight_packing")
    try:
        for chunk_size in arguments.chunk_sizes:
            candidate.set_chunk_size(chunk_size)
            candidate.prepare()
            print(f"Validating tuning chunk={chunk_size}...", flush=True)
            correctness = _validate_workload(
                model,
                candidate,
                tuning_prompt,
                arguments.tuning_output_tokens,
                arguments.dtype,
                chunk_size,
                arguments.cache_check_interval,
            )
            report["tuning"].append(
                {"chunk_size": chunk_size, "status": "validated", "correctness": correctness}
            )
        for result in report["tuning"]:
            candidate.set_chunk_size(result["chunk_size"])
            candidate.prepare()
            print(f"Timing tuning chunk={result['chunk_size']}...", flush=True)
            result.update(
                _measure(
                    model,
                    candidate,
                    tuning_prompt,
                    arguments.tuning_output_tokens,
                    arguments.tuning_trials,
                    result["correctness"]["generated_token_ids"],
                ),
                status="ok",
            )
        selected = _select_chunk(report["tuning"])
        report["selected_chunk_size"] = selected
        candidate.set_chunk_size(selected)
        candidate.prepare()
        sequential = sequential_factory(capacity) if arguments.sequential_baseline else None
        print(
            f"Validating held-out chunk={selected}, prompt={len(held_out_prompt)}, output={arguments.output_tokens}...",
            flush=True,
        )
        correctness = _validate_workload(
            model,
            candidate,
            held_out_prompt,
            arguments.output_tokens,
            arguments.dtype,
            selected,
            arguments.cache_check_interval,
        )
        result = {
            "chunk_size": selected,
            "status": "validated",
            "correctness": correctness,
            "geometry": dict(candidate.config),
            "parameter_count": candidate.parameter_count,
            "prefill_weight_bytes": candidate.prefill_weight_bytes,
        }
        report["held_out"].append(result)
        if sequential is not None:
            sequential.prepare()
            result["sequential_correctness"] = _validate_workload(
                model,
                sequential,
                held_out_prompt,
                arguments.output_tokens,
                arguments.dtype,
                selected,
                arguments.cache_check_interval,
                sequential=True,
            )
            if (
                result["sequential_correctness"]["generated_token_ids"]
                != correctness["generated_token_ids"]
            ):
                raise GenerationValidationFailure(
                    [{"error": "sequential and chunked validation token trajectories differ"}]
                )
        print("Timing held-out generation...", flush=True)
        result.update(
            _measure(
                model,
                candidate,
                held_out_prompt,
                arguments.output_tokens,
                arguments.trials,
                correctness["generated_token_ids"],
                sequential,
            ),
            status="ok",
        )
    except GenerationValidationFailure as failure:
        _discard_timings(report, failure)
    return report


def run(arguments):
    _require_precision_environment(arguments.dtype)
    root = _benchmark_checkout()

    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer

    from benchmarks.common.checkout import implementation_hash
    from benchmarks.megakernels import qwen3_prefill_runtime, qwen3_staged_runtime
    from metile.runtime.metal_device import MetalDevice

    source_paths = [path for path in Path(__file__).parent.glob("qwen3*.py")] + [
        root / "benchmarks/common/checkout.py"
    ]
    source_hashes = {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths}
    compiler_hash = implementation_hash(root)
    snapshot = _local_snapshot(arguments.model, arguments.revision)
    checkpoint_config = json.loads((snapshot / "config.json").read_text())
    _validate_checkpoint_config(checkpoint_config)
    tokenizer = load_tokenizer(snapshot, {"local_files_only": True, "trust_remote_code": False})
    tuning_prompt = _document_prompt(
        root, tokenizer, arguments.tuning_prompt_sources, arguments.tuning_prompt_tokens
    )
    held_out_prompt = _document_prompt(
        root, tokenizer, arguments.prompt_sources, arguments.prompt_tokens
    )
    _require_disjoint_prompts(tuning_prompt, held_out_prompt)
    model, _ = load_model(snapshot)
    model.set_dtype(getattr(mx, arguments.dtype))
    mx.eval(model.parameters())

    def factory(capacity, chunk_size):
        return qwen3_prefill_runtime.Qwen3ChunkedPrefill(
            model,
            capacity,
            threads=arguments.threads,
            chunk_size=chunk_size,
            attention_backend=arguments.attention_backend,
            projection_backend=arguments.projection_backend,
            projection_tile=tuple(arguments.projection_tile),
            lossless_decode_weights=arguments.lossless_decode_weights,
        )

    def sequential_factory(capacity):
        return qwen3_staged_runtime.Qwen3Staged(model, capacity, threads=arguments.threads)

    report = _run_phases(
        arguments,
        model,
        factory,
        sequential_factory,
        tuning_prompt["token_ids"],
        held_out_prompt["token_ids"],
    )
    if compiler_hash != implementation_hash(root) or source_hashes != {
        path.relative_to(root).as_posix(): _sha256(path) for path in source_paths
    }:
        raise RuntimeError(
            "compiler, kernel, benchmark or runtime sources changed during measurement"
        )
    for prompt in (tuning_prompt, held_out_prompt):
        if prompt["sources_sha256"] != {
            relative: _sha256(root / relative) for relative in prompt["sources_sha256"]
        }:
            raise RuntimeError("prompt document sources changed during measurement")
    packed_weights = bool((report.get("candidate_weight_packing") or {}).get("enabled"))
    report.update(
        schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        model=arguments.model,
        checkpoint_revision=snapshot.name,
        checkpoint_sha256={
            path.name: _sha256(path)
            for path in sorted([snapshot / "config.json", *snapshot.glob("model*.safetensors")])
        },
        tokenizer_sha256={
            path.name: _sha256(path)
            for path in sorted(snapshot.glob("tokenizer*"))
            if path.is_file()
        },
        hardware=mx.device_info(),
        software={
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "platform": platform.platform(),
            "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
            "candidate_environment": {
                name: os.environ.get(name)
                for name in (
                    "METILE_SCHEDULE",
                    "METILE_ONLINE_SOFTMAX",
                    "METILE_LOW_LATENCY_SPIN_US",
                )
            },
        },
        metal_compiler=MetalDevice.get().metal_compiler_version,
        source_sha256=source_hashes,
        compiler_and_kernels_sha256=compiler_hash,
        weights={
            "source_dtype": checkpoint_config.get("torch_dtype") or checkpoint_config.get("dtype"),
            "both_backends": arguments.dtype,
            "dtype_scope": "decoded_weight_values",
            "same_values": True,
            "quantized": False,
            "native_projection_storage": arguments.dtype,
            "candidate_prefill_projection_storage": arguments.dtype,
            "candidate_decode_projection_storage": (
                "packed_uint32_bfloat16_pairs" if packed_weights else arguments.dtype
            ),
        },
        precision_comparison={
            "class": "lossless_weight_storage" if packed_weights else "same_storage_precision",
            "same_weight_representation": not packed_weights,
            "same_weight_values": True,
            "bitwise_exact": False,
            "storage_dtype": arguments.dtype,
            "storage_dtype_scope": "activations_and_kv_cache",
            "accumulation_dtype": "float32",
        },
        selection={
            "candidate_chunk_sizes": arguments.chunk_sizes,
            "criterion": "minimum tuning median candidate TTFT; smaller chunk breaks exact ties",
            "held_out_used_for_selection": False,
            "held_out_scope": "held out from automatic chunk-size selection only; the 4096-token document was exercised during implementation correctness development, so it is not unseen development data",
            "tuning_prompt": tuning_prompt,
            "tuning_output_tokens": arguments.tuning_output_tokens,
            "tuning_trials": arguments.tuning_trials,
        },
        workload={
            "batch": 1,
            "prompt": held_out_prompt,
            "requested_output_tokens": arguments.output_tokens,
            "trials": arguments.trials,
            "sampling": "greedy GPU argmax; EOS ignored; exact requested output count",
            "sequential_baseline_included": arguments.sequential_baseline,
            "warmup_generations_per_backend": 2,
            "candidate_cache_capacity": max(
                arguments.prompt_tokens + arguments.output_tokens - 1,
                arguments.tuning_prompt_tokens + arguments.tuning_output_tokens - 1,
            ),
            "candidate_storage_reuse": "one weight/cache allocation for the whole tuning and held-out session; chunk reconfiguration and pipeline preparation are untimed",
        },
        correctness_policy={
            "prefill": "native full-prefix batched reference; every candidate chunk boundary checks complete valid K and V across all layers",
            "whole_prompt_replay": "after chunk-boundary checks, reset and call candidate.prefill(prompt) once as in timing; its full valid KV cache, final-row vocabulary logits and GPU greedy token are checked by the first generation entry, then every decode check continues from this whole-call state; the replay references that entry rather than adding a duplicate comparison",
            "generation": "every generated step checks full vocabulary logits, greedy token and newly written KV slot in every layer",
            "full_prefix_check_interval_generated_tokens": arguments.cache_check_interval,
            "full_prefix_also_checked": "first and final generated step",
            "timing_gate": "all tuning candidates validate before tuning timings; selected held-out model validates before held-out timings; every warmup/timed token sequence must exactly match validation; any failure removes all saved timings",
        },
        execution={
            "candidate": "multi-dispatch DSL batched prefill over the entire prompt, final prompt row vocabulary projection, then GPU-wide single-token decode",
            "candidate_attention_backend": arguments.attention_backend,
            "candidate_projection_backend": arguments.projection_backend,
            "candidate_projection_tile": arguments.projection_tile,
            "candidate_strict_math": True,
            "candidate_projection_relaxed_precision": False,
            "native": "native MLX Qwen3 model in an explicit greedy loop, not mlx_lm.stream_generate; full batched prefix except final prompt token, then final-prompt single-token forward",
            "prefill_difference": "native prefix evaluates only KV state and may prune final-layer work after its cache update; candidate stops the final layer after its KV write for nonterminal internal chunks, runs the complete terminal chunk and projects only its final row",
            "not_a_single_dispatch_megakernel": True,
            "optional_sequential_baseline": "separate contemporary run of the earlier tokenwise-prefill staged implementation, including its earlier decode kernels and unpacked projection weights; not an isolated prefill-only ablation",
        },
        timing={
            "metric": "synchronized token-ID-to-token-ID generation wall seconds",
            "included": [
                "fresh native cache construction or candidate cache reset",
                "all prefill and autoregressive decode work and KV updates",
                "GPU greedy selection and host observation of each output ID",
                "token allocation/control updates and graph/dispatch submissions",
            ],
            "excluded": [
                "checkpoint loading, dtype conversion, weight packing and rotary-table preparation",
                "compilation, persistent candidate buffer allocation and pipeline preparation",
                "document loading, tokenization, text decoding and correctness checks",
                "two complete warmup generations per backend",
            ],
            "ttft": "cache initialization through the first returned token ID, including all prefill",
            "decode": "first through final returned token ID; exactly output_tokens - 1 decode forwards",
            "allocation_difference": "candidate reuses persistent storage and zeros cache in timing; native MLX manages intermediate/output and growing-cache allocation in timing",
            "aggregation": "median of within-trial ratios; separate median latency and median N-1 decode throughput",
            "limitations": "one held-out technical-document prompt, batch one, forced fixed output length, warmed pipelines; not chat quality or serving latency",
        },
    )
    return report


def main(argv=None):
    arguments = _arguments(argv)
    if arguments.output.exists():
        raise FileExistsError(f"Preserving {arguments.output}; choose a new --output path")
    report = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {arguments.output}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
