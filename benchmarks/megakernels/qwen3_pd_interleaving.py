"""Compare serialized prefill/decode scheduling between two independent requests.

Request A has completed prefill and returned its first token before timing.
At time zero, request B arrives while A is ready to decode. This is a
steady-state interference experiment, not a serving system, hardware-overlap
measurement, or native-MLX speed comparison.
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
    _cache_fidelity,
    _fidelity_tolerances,
    _local_snapshot,
    _logit_fidelity,
    _require_precision_environment,
    _sha256,
    _validate_checkpoint_config,
)
from benchmarks.megakernels.qwen3_chunked_prefill import (
    DEFAULT_PROMPT_SOURCES,
    _benchmark_checkout,
    _cache_slot_fidelity,
    _document_prompt,
    _require_disjoint_prompts,
)
from benchmarks.megakernels.qwen3_end_to_end import GenerationValidationFailure

POLICIES = ("fifo", "eager", "chunked")
REQUESTS = ("A", "B")


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["eager", "chunked"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cache-check-interval", type=int, default=64)
    parser.add_argument("--stall-threshold-ms", type=float, default=100.0)
    parser.add_argument("--no-lossless-decode-weights", action="store_true")
    parser.add_argument(
        "--logical-cache-reset",
        action="store_true",
        help="invalidate cache length without clearing physical K/V storage, for every policy",
    )
    parser.add_argument(
        "--request-a-sources",
        nargs="+",
        default=["docs/api/reference.rst", "docs/guide/execution-schedules.rst"],
    )
    parser.add_argument("--request-b-sources", nargs="+", default=list(DEFAULT_PROMPT_SOURCES))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/m5-qwen3-pd-interleaving.json")
    )
    arguments = parser.parse_args(argv)
    for name in ("prompt_tokens", "trials", "warmups", "cache_check_interval"):
        if getattr(arguments, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if arguments.output_tokens < 2:
        parser.error("output-tokens must be at least two")
    if not 1 <= arguments.chunk_size <= 4096:
        parser.error("chunk-size must be in [1, 4096]")
    if len(arguments.policies) < 2 or len(set(arguments.policies)) != len(arguments.policies):
        parser.error("at least two distinct policies are required")
    if not math.isfinite(arguments.stall_threshold_ms) or arguments.stall_threshold_ms <= 0:
        parser.error("stall-threshold-ms must be finite and positive")
    return arguments


def _order(policies, trial):
    start = trial % len(policies)
    return list(policies[start:]) + list(policies[:start])


def _distribution(values, threshold):
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "p50_seconds": float(np.quantile(values, 0.5)) if values else None,
        "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
        "p99_seconds": float(np.quantile(values, 0.99)) if values else None,
        "max_seconds": max(values) if values else None,
        "stall_count": sum(value > threshold for value in values),
        "stall_excess_seconds": sum(max(value - threshold, 0.0) for value in values),
    }


def _sample_metrics(sample, threshold):
    requests = sample["requests"]
    times = {name: requests[name]["emission_times_seconds"] for name in REQUESTS}
    if times["A"][0] != 0 or any(
        len(times[name]) < 2
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in times[name])
        or any(right <= left for left, right in zip(times[name], times[name][1:]))
        for name in REQUESTS
    ):
        raise ValueError("emission timestamps must be finite and strictly increasing")
    if times["B"][0] <= 0:
        raise ValueError("request B's first token must follow its time-zero arrival")
    intervals = {name: list(zip(times[name][:-1], times[name][1:])) for name in REQUESTS}
    gaps = {name: [end - start for start, end in intervals[name]] for name in REQUESTS}
    prefill_events = [event for event in sample["events"] if event["kind"] == "prefill"]
    prefill_start = min(event["started_seconds"] for event in prefill_events)
    if not 0 <= prefill_start < times["B"][0]:
        raise ValueError("B prefill must occur between arrival and its first emitted token")
    summaries = {name: _distribution(gaps[name], threshold) for name in REQUESTS}
    for name in REQUESTS:
        requests[name]["inter_token_gap_seconds"] = gaps[name]
        requests[name]["gap_distribution"] = summaries[name]
        requests[name]["completion_since_arrival_seconds"] = times[name][-1]
    active = {}
    for name, end in (("b_prefill", times["B"][0]), ("b_in_service", times["B"][-1])):
        values = [
            right - left for left, right in intervals["A"] if left < end and right > prefill_start
        ]
        active[name] = {
            "window_seconds": [prefill_start, end],
            "a_gap_distribution": _distribution(values, threshold),
        }
    sample["active_interference"] = active
    makespan = max(times[name][-1] for name in REQUESTS)
    timed_tokens = sum(len(times[name]) for name in REQUESTS) - 1
    return {
        "makespan_seconds": makespan,
        "timed_output_tokens": timed_tokens,
        "output_tokens_per_second": timed_tokens / makespan,
        "b_ttft_seconds": times["B"][0],
        "a_initial_resumption_gap_seconds": gaps["A"][0],
        "a_tpot_seconds": summaries["A"]["mean_seconds"],
        "b_tpot_seconds": summaries["B"]["mean_seconds"],
        **{
            f"a_gap_{name}": summaries["A"][name]
            for name in ("p50_seconds", "p95_seconds", "p99_seconds", "max_seconds")
        },
        "a_stall_count": summaries["A"]["stall_count"],
        "a_stall_excess_seconds": summaries["A"]["stall_excess_seconds"],
    }


def _prime(candidate, prompt, synchronize, observe=None, *, clear_cache=True):
    candidate.reset(clear_cache=clear_cache)
    for start in range(0, len(prompt), candidate.chunk_size):
        chunk = prompt[start : start + candidate.chunk_size]
        candidate.prefill_chunk(chunk, final_chunk=start + len(chunk) == len(prompt))
        synchronize()
        if observe is not None:
            observe("A", "prefill", valid_tokens=start + len(chunk))
    candidate.project_last_prefill()
    token = candidate.greedy()
    if observe is not None:
        observe("A", "generation", offset=0, input_token=prompt[-1], token=token)
    return token


def _execute(
    policy,
    candidates,
    prompts,
    output_tokens,
    initial_token,
    synchronize,
    threshold,
    observe=None,
    clock=time.perf_counter_ns,
    *,
    clear_cache=True,
):
    if policy not in POLICIES:
        raise ValueError("unknown scheduling policy")
    start_ns = clock()
    generated = {"A": [initial_token], "B": []}
    emitted = {"A": [0.0], "B": []}
    events = []
    prefilled = 0

    def now():
        return (clock() - start_ns) * 1e-9

    reset_start = now()
    candidates["B"].reset(clear_cache=clear_cache)
    synchronize()
    events.append(
        {
            "request": "B",
            "kind": "reset",
            "started_seconds": reset_start,
            "completed_seconds": now(),
        }
    )

    def emit(name, token, input_token):
        offset = len(generated[name])
        generated[name].append(token)
        emitted[name].append(now())
        if observe is not None:
            observe(name, "generation", offset=offset, input_token=input_token, token=token)

    def decode(name):
        token = generated[name][-1]
        position = len(prompts[name]) + len(generated[name]) - 1
        began = now()
        candidates[name].forward(token, position, project=True)
        emit(name, candidates[name].greedy(), token)
        events.append(
            {
                "request": name,
                "kind": "decode",
                "position": position,
                "started_seconds": began,
                "completed_seconds": emitted[name][-1],
            }
        )

    def prefill():
        nonlocal prefilled
        candidate = candidates["B"]
        chunk = prompts["B"][prefilled : prefilled + candidate.chunk_size]
        final = prefilled + len(chunk) == len(prompts["B"])
        began = now()
        candidate.prefill_chunk(chunk, final_chunk=final)
        synchronize()
        if observe is not None:
            observe("B", "prefill", valid_tokens=prefilled + len(chunk))
        if final:
            candidate.project_last_prefill()
            emit("B", candidate.greedy(), prompts["B"][-1])
        ended = emitted["B"][-1] if final else now()
        events.append(
            {
                "request": "B",
                "kind": "prefill",
                "prompt_start": prefilled,
                "prompt_tokens": len(chunk),
                "final_chunk": final,
                "started_seconds": began,
                "completed_seconds": ended,
            }
        )
        prefilled += len(chunk)

    if policy == "fifo":
        while len(generated["A"]) < output_tokens:
            decode("A")
    if policy in ("fifo", "eager"):
        while prefilled < len(prompts["B"]):
            prefill()
    while any(len(generated[name]) < output_tokens for name in REQUESTS):
        ready = [name for name in REQUESTS if 0 < len(generated[name]) < output_tokens]
        for name in ready:
            decode(name)
        if prefilled < len(prompts["B"]):
            prefill()
    sample = {
        "policy": policy,
        "requests": {
            name: {
                "generated_token_ids": generated[name],
                "actual_output_tokens": len(generated[name]),
                "timed_output_tokens": len(generated[name]) - int(name == "A"),
                "initial_token_timed": name != "A",
                "emission_times_seconds": emitted[name],
            }
            for name in REQUESTS
        },
        "events": events,
    }
    sample["metrics"] = _sample_metrics(sample, threshold)
    return sample


def _check_sample(sample, expected):
    for name in REQUESTS:
        observed = sample["requests"][name]
        if (
            observed["generated_token_ids"] != expected[name]
            or observed["actual_output_tokens"] != len(expected[name])
            or observed["timed_output_tokens"] != len(expected[name]) - int(name == "A")
        ):
            raise GenerationValidationFailure(
                [{"request": name, "error": "scheduled token trajectory or count differs"}]
            )


def _validate_policy(model, candidates, prompts, arguments, policy, synchronize):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    logit_tolerance, cache_tolerance = _fidelity_tolerances("float32")
    native = {}
    checks = {name: [] for name in REQUESTS}
    expected = {name: [] for name in REQUESTS}
    for name in REQUESTS:
        cache = make_prompt_cache(model)
        prompt = prompts[name]
        if len(prompt) > 1:
            model(mx.array([prompt[:-1]], dtype=mx.int32), cache=cache)
            mx.eval([layer.state for layer in cache])
        logits = model(mx.array([[prompt[-1]]], dtype=mx.int32), cache=cache)
        mx.eval(logits, [layer.state for layer in cache])
        native[name] = {"cache": cache, "logits": logits}

    def compare(name, stage, **event):
        candidate = candidates[name]
        cache = native[name]["cache"]
        if stage == "prefill":
            fidelity = {
                **_cache_fidelity(cache, candidate, event["valid_tokens"], cache_tolerance),
                "full_prefix_checked": True,
            }
            check = {"stage": stage, **event, "cache": fidelity, "passed": fidelity["passed"]}
        else:
            offset = event["offset"]
            position = len(prompts[name]) - 1 + offset
            if offset:
                native[name]["logits"] = model(
                    mx.array([[expected[name][-1]]], dtype=mx.int32), cache=cache
                )
                mx.eval(native[name]["logits"], [layer.state for layer in cache])
            logits = _logit_fidelity(
                np.asarray(native[name]["logits"]), candidate.logits.numpy(), logit_tolerance
            )
            full = (
                offset == 0
                or (offset + 1) % arguments.cache_check_interval == 0
                or offset == arguments.output_tokens - 1
            )
            fidelity = (
                {
                    **_cache_fidelity(cache, candidate, position + 1, cache_tolerance),
                    "full_prefix_checked": True,
                }
                if full
                else _cache_slot_fidelity(cache, candidate, position, cache_tolerance)
            )
            expected_input = expected[name][-1] if offset else prompts[name][-1]
            check = {
                "stage": stage,
                **event,
                "position": position,
                "logits": logits,
                "cache": fidelity,
                "passed": logits["passed"]
                and fidelity["passed"]
                and event["token"] == logits["mlx_next_token"]
                and event["input_token"] == expected_input,
            }
            expected[name].append(logits["mlx_next_token"])
        checks[name].append(check)
        if not check["passed"]:
            raise GenerationValidationFailure([{"policy": policy, "request": name, **check}])

    def observe(name, stage, **event):
        try:
            compare(name, stage, **event)
        except ValueError as error:
            raise GenerationValidationFailure(
                [
                    {
                        "policy": policy,
                        "request": name,
                        "stage": stage,
                        "passed": False,
                        "error": str(error),
                    }
                ]
            ) from error

    initial = _prime(
        candidates["A"],
        prompts["A"],
        synchronize,
        observe,
        clear_cache=not arguments.logical_cache_reset,
    )
    sample = _execute(
        policy,
        candidates,
        prompts,
        arguments.output_tokens,
        initial,
        synchronize,
        arguments.stall_threshold_ms / 1000,
        observe,
        clear_cache=not arguments.logical_cache_reset,
    )
    _check_sample(sample, expected)
    return {
        "passed": True,
        "requests": {
            name: {
                "passed": True,
                "checks": checks[name],
                "generated_token_ids": expected[name],
                "actual_output_tokens": len(expected[name]),
                "chunk_boundary_cache_checks": sum(
                    check["stage"] == "prefill" for check in checks[name]
                ),
                "generation_logit_cache_greedy_checks": len(expected[name]),
            }
            for name in REQUESTS
        },
    }


def _run_policies(arguments, candidates, prompts, validate, synchronize):
    report = {"status": "ok", "validation": {}, "samples": []}
    expected = None
    try:
        for policy in arguments.policies:
            print(f"Validating policy={policy}...", flush=True)
            validation = validate(policy)
            report["validation"][policy] = validation
            if validation.get("passed") is not True or any(
                validation["requests"][name].get("passed") is not True
                or len(validation["requests"][name]["generated_token_ids"])
                != arguments.output_tokens
                for name in REQUESTS
            ):
                raise GenerationValidationFailure([{"error": "incomplete policy validation"}])
            current = {
                name: validation["requests"][name]["generated_token_ids"] for name in REQUESTS
            }
            if expected is not None and current != expected:
                raise GenerationValidationFailure(
                    [{"error": "policy reference trajectories differ"}]
                )
            expected = current

        def measure(policy):
            initial = _prime(
                candidates["A"],
                prompts["A"],
                synchronize,
                clear_cache=not arguments.logical_cache_reset,
            )
            sample = _execute(
                policy,
                candidates,
                prompts,
                arguments.output_tokens,
                initial,
                synchronize,
                arguments.stall_threshold_ms / 1000,
                clear_cache=not arguments.logical_cache_reset,
            )
            _check_sample(sample, expected)
            return sample

        for _ in range(arguments.warmups):
            for policy in arguments.policies:
                measure(policy)
        for trial in range(arguments.trials):
            order = _order(arguments.policies, trial)
            samples = {policy: measure(policy) for policy in order}
            report["samples"].append({"trial": trial, "order": order, "policies": samples})
            print(
                f"  trial {trial + 1}/{arguments.trials}: "
                + ", ".join(
                    f"{policy} {samples[policy]['metrics']['makespan_seconds']:.3f}s"
                    for policy in order
                ),
                flush=True,
            )
    except GenerationValidationFailure as failure:
        report.update(status="validation_failed", validation_failure=failure.checks)
        report.pop("samples", None)
        return report
    report["medians"] = {
        policy: {
            metric: statistics.median(
                sample["policies"][policy]["metrics"][metric] for sample in report["samples"]
            )
            for metric in report["samples"][0]["policies"][policy]["metrics"]
        }
        for policy in arguments.policies
    }
    baseline = arguments.policies[0]
    report["paired_ratios"] = {
        policy: {
            metric: statistics.median(
                sample["policies"][baseline]["metrics"][metric]
                / sample["policies"][policy]["metrics"][metric]
                for sample in report["samples"]
            )
            for metric in ("makespan_seconds", "b_ttft_seconds", "a_gap_max_seconds")
        }
        for policy in arguments.policies[1:]
    }
    return report


def run(arguments):
    _require_precision_environment("float32")
    root = _benchmark_checkout()
    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer

    from benchmarks.common.checkout import implementation_hash
    from benchmarks.megakernels.qwen3_prefill_runtime import Qwen3ChunkedPrefill
    from metile.runtime.metal_device import MetalDevice

    source_paths = [
        *sorted(Path(__file__).parent.glob("qwen3*.py")),
        root / "benchmarks/common/checkout.py",
    ]
    sources = {path.relative_to(root).as_posix(): _sha256(path) for path in source_paths}
    compiler_hash = implementation_hash(root)
    snapshot = _local_snapshot(arguments.model, arguments.revision)
    config = json.loads((snapshot / "config.json").read_text())
    _validate_checkpoint_config(config)
    tokenizer = load_tokenizer(snapshot, {"local_files_only": True, "trust_remote_code": False})
    documents = {
        name: _document_prompt(
            root,
            tokenizer,
            getattr(arguments, f"request_{name.lower()}_sources"),
            arguments.prompt_tokens,
        )
        for name in REQUESTS
    }
    _require_disjoint_prompts(documents["A"], documents["B"])
    prompts = {name: documents[name]["token_ids"] for name in REQUESTS}
    model, _ = load_model(snapshot)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    first = Qwen3ChunkedPrefill(
        model,
        arguments.prompt_tokens + arguments.output_tokens - 1,
        chunk_size=arguments.chunk_size,
        attention_backend="matrix",
        projection_backend="tensor_ops",
        projection_tile=(64, 64, 64),
        lossless_decode_weights=not arguments.no_lossless_decode_weights,
    ).prepare()
    candidates = {"A": first, "B": first.fork_request().prepare()}
    device = MetalDevice.get()
    report = _run_policies(
        arguments,
        candidates,
        prompts,
        lambda policy: _validate_policy(model, candidates, prompts, arguments, policy, device.sync),
        device.sync,
    )
    if compiler_hash != implementation_hash(root) or any(
        _sha256(root / path) != digest for path, digest in sources.items()
    ):
        raise RuntimeError(
            "compiler, kernel, benchmark or runtime sources changed during measurement"
        )
    if any(
        _sha256(root / path) != digest
        for document in documents.values()
        for path, digest in document["sources_sha256"].items()
    ):
        raise RuntimeError("prompt sources changed during measurement")
    report.update(
        schema_version=1,
        benchmark="qwen3_pd_interleaving",
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
        source_sha256=sources,
        compiler_and_kernels_sha256=compiler_hash,
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
        metal_compiler=device.metal_compiler_version,
        geometry=dict(first.config),
        parameter_count=first.parameter_count,
        prefill_weight_bytes=first.prefill_weight_bytes,
        workload={
            "requests": documents,
            "output_tokens_per_request": arguments.output_tokens,
            "timed_output_tokens": 2 * arguments.output_tokens - 1,
            "chunk_size": arguments.chunk_size,
            "chunk_selection": "fixed before this experiment; no timing-based selection",
            "policies": arguments.policies,
            "trials": arguments.trials,
            "warmups_per_policy": arguments.warmups,
            "sampling": "GPU greedy, EOS ignored, exact output counts",
            "arrival": "A has returned token zero before timing; at t0 B arrives and A is ready for its next decode",
        },
        precision={
            "storage_dtype": "float32",
            "accumulation_dtype": "float32",
            "candidate_weight_packing": first.config.get("decode_weight_packing"),
            "same_backend_and_weight_storage_across_policies": True,
            "native_role": "independent correctness reference only; no native speed comparison",
        },
        precision_comparison={
            "class": "same_backend_scheduling",
            "same_weight_representation": True,
            "same_weight_values": True,
            "storage_dtype": "float32",
            "storage_dtype_scope": "activations_and_kv_cache",
            "accumulation_dtype": "float32",
            "bitwise_exact": False,
            "scope": "identical candidate implementation and physical weight storage across scheduling policies; native MLX is correctness-only",
        },
        execution={
            "cooperative_serial_interleaving": True,
            "cache_reset_policy": "logical" if arguments.logical_cache_reset else "zero_fill",
            "hardware_overlap_claimed": False,
            "self_request_prefill_decode_overlap": False,
            "shared_immutable_weights": True,
            "independent_request_cache_and_workspace": True,
            "fifo": "finish A, then prefill and generate B",
            "eager": "finish every B prefill chunk, then alternate ready A/B decodes",
            "chunked": "one decode per request ready at round start, then at most one B prefill chunk",
            "terminal_prefill": "only the true final prompt chunk computes final-layer output and immediately projects/selects B's first token",
            "quantum_completion": "host synchronization after each prefill chunk; GPU greedy token read synchronizes each decode",
        },
        correctness={
            "cache_check_interval": arguments.cache_check_interval,
            "tolerances": _fidelity_tolerances("float32"),
            "policy_gate": "every policy validates all A/B prefill boundaries and output steps before any timings",
            "generation": "full vocabulary logits and exact greedy/input tokens every step; new K/V slot in every layer; full valid prefix at first output, interval checkpoints and final output",
            "timed_sequences": "every warmup and timed sequence must match independent native IDs; any failure discards all timings",
        },
        measurement={
            "included": "B cache reset, scheduler/control work, remaining A decode, all B prefill/decode, GPU greedy and host token observation",
            "excluded": "A cache reset/prefill/first token, checkpoint load, conversion/packing, compilation, persistent allocation, tokenization, correctness checks and warmups",
            "request_a_latency": "remaining completion latency since resumption; not its end-to-end request latency or TTFT",
            "throughput": "(2 * output_tokens - 1) / makespan; includes exactly all timed output IDs",
            "a_initial_gap": "time from fixed t0 resumption/B arrival to A's next observed token",
            "tpot": "mean within-request inter-token gap; individual p50/p95/p99/max gaps are also recorded",
            "active_interference": "full A inter-token intervals intersecting B's prefill/service windows; empty intersections have null quantiles, not zero latency",
            "quantile_method": "linear interpolation; within-request intervals, not independent trials",
            "stall_threshold_seconds": arguments.stall_threshold_ms / 1000,
            "stall_definition": "gap strictly above preset threshold; excess sums max(gap - threshold, 0), not a hardware stall counter",
            "aggregation": "per-policy trial medians; paired baseline/policy ratios for lower-is-better times; all samples retained, no confidence intervals",
            "paired_baseline": arguments.policies[0],
            "limitations": "two fixed requests and few repetitions; descriptive latency distributions, not serving capacity or statistically established tail guarantees",
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
