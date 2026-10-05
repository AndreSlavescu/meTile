"""Validate and plot same-backend cooperative prefill/decode scheduling trials."""

import argparse
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

from benchmarks.megakernels.qwen3 import _fidelity_tolerances
from benchmarks.megakernels.qwen3_chunked_prefill import _require_disjoint_prompts
from benchmarks.megakernels.qwen3_pd_interleaving import POLICIES, REQUESTS, _order, _sample_metrics
from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_chunked_prefill import (
    _cache,
    _execution_metadata,
    _fidelity,
    _finite,
    _integer,
    _mapping,
    _prompt,
    _tokens,
)

DEFAULT_INPUT = Path("benchmarks/results/m5-qwen3-pd-interleaving.json")
DEFAULT_OUTPUT = Path("docs/_static/qwen3-pd-interleaving.png")
LABELS = {"fifo": "Request FIFO", "eager": "Eager prefill", "chunked": "Chunked, decode first"}
COLORS = {"fifo": style.ACCENT, "eager": style.PREFILL, "chunked": style.DECODE}
METRICS = (
    ("a_gap_max_seconds", "A · longest gap between tokens", "milliseconds", 1000),
    ("b_ttft_seconds", "B · time to first token", "seconds", 1),
    ("makespan_seconds", "Complete the remaining work", "seconds", 1),
)


def _same(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(left, right) for left, right in zip(actual, expected, strict=True))
        )
    if type(expected) is float:
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def _hashes(values, name):
    values = _mapping(values, name)
    if not values or any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in values.items()
    ):
        raise ValueError(f"{name} requires file identities and SHA-256 hashes")


def _precision(report, geometry):
    comparison = _mapping(report.get("precision_comparison"), "precision comparison")
    expected = {
        "class": "same_backend_scheduling",
        "same_weight_representation": True,
        "same_weight_values": True,
        "storage_dtype": "float32",
        "storage_dtype_scope": "activations_and_kv_cache",
        "accumulation_dtype": "float32",
        "bitwise_exact": False,
    }
    if any(not _same(comparison.get(name), value) for name, value in expected.items()):
        raise ValueError("policies must use the same physical weights and FP32 arithmetic")
    precision = _mapping(report.get("precision"), "precision")
    if (
        precision.get("storage_dtype") != "float32"
        or precision.get("accumulation_dtype") != "float32"
        or precision.get("same_backend_and_weight_storage_across_policies") is not True
        or report.get("software", {}).get("MLX_ENABLE_TF32") != "0"
    ):
        raise ValueError("same-backend FP32 comparison and strict native validation are required")
    packing = precision.get("candidate_weight_packing")
    configured = geometry.get("decode_weight_packing")
    if not _same(packing, configured):
        raise ValueError("weight packing must match the shared model geometry")
    if packing is None:
        return False
    packing = _mapping(packing, "weight packing")
    fields = {
        "enabled": True,
        "format": "fp32_high16x2_u32_lossless",
        "decoded_dtype": "float32",
        "storage_dtype": "uint32",
        "values_per_word": 2,
        "scopes": ["layer_weights", "embedding"],
        "consumers": ["decode_projections", "first_token_vocabulary_projection"],
        "original_buffers_retained": True,
        "verification": {
            "finite": True,
            "even_elements": True,
            "zero_low16_bits": True,
            "roundtrip_bitwise": True,
        },
    }
    count = packing.get("element_count")
    if (
        any(not _same(packing.get(name), value) for name, value in fields.items())
        or not _integer(count)
        or count % 2
        or not _integer(report.get("parameter_count"))
        or not _integer(geometry.get("hidden_size"))
        or count != report["parameter_count"] - geometry["hidden_size"]
        or not _same(packing.get("packed_bytes"), count * 2)
        or not _same(packing.get("unpacked_bytes"), count * 4)
    ):
        raise ValueError(
            "lossless weight packing requires exact counts and all verification checks"
        )
    return True


def _validation(evidence, prompt, outputs, chunk_size, layers, vocabulary, interval):
    evidence = _mapping(evidence, "request validation")
    generated = evidence.get("generated_token_ids")
    boundaries = [*range(chunk_size, len(prompt), chunk_size), len(prompt)]
    if (
        evidence.get("passed") is not True
        or not _tokens(generated, outputs, vocabulary)
        or not _same(evidence.get("actual_output_tokens"), outputs)
        or not _same(evidence.get("chunk_boundary_cache_checks"), len(boundaries))
        or not _same(evidence.get("generation_logit_cache_greedy_checks"), outputs)
        or not isinstance(evidence.get("checks"), list)
        or len(evidence["checks"]) != len(boundaries) + outputs
    ):
        raise ValueError("complete per-request validation counts and token IDs are required")
    logits_tolerance, cache_tolerance = _fidelity_tolerances("float32")
    checks = evidence["checks"]
    for check, count in zip(checks[: len(boundaries)], boundaries, strict=True):
        check = _mapping(check, "prefill validation")
        if (
            check.get("stage") != "prefill"
            or check.get("passed") is not True
            or not _same(check.get("valid_tokens"), count)
        ):
            raise ValueError("every prompt chunk must validate in order")
        _cache(check.get("cache"), count, layers, True, cache_tolerance)
    for offset, check in enumerate(checks[len(boundaries) :]):
        check = _mapping(check, "generation validation")
        position = len(prompt) - 1 + offset
        token = generated[offset]
        if (
            check.get("stage") != "generation"
            or check.get("passed") is not True
            or not _same(check.get("offset"), offset)
            or not _same(check.get("position"), position)
            or not _same(check.get("token"), token)
            or not _same(check.get("input_token"), generated[offset - 1] if offset else prompt[-1])
        ):
            raise ValueError("each generated step must advance its own validated trajectory")
        logits = _fidelity(check.get("logits"), logits_tolerance, "logit validation")
        if (
            logits.get("greedy_token_equal") is not True
            or not _same(logits.get("mlx_next_token"), token)
            or not _same(logits.get("metile_next_token"), token)
        ):
            raise ValueError("full-logit validation and GPU greedy selection must agree")
        full = offset == 0 or (offset + 1) % interval == 0 or offset == outputs - 1
        _cache(
            check.get("cache"), position + 1 if full else 1, layers, full, cache_tolerance, position
        )
    return generated


def _expected_events(policy, prompt_lengths, outputs, chunk):
    events = [{"request": "B", "kind": "reset"}]
    generated = {"A": 1, "B": 0}
    prefilled = 0

    def decode(name):
        events.append(
            {
                "request": name,
                "kind": "decode",
                "position": prompt_lengths[name] + generated[name] - 1,
            }
        )
        generated[name] += 1

    def prefill():
        nonlocal prefilled
        count = min(chunk, prompt_lengths["B"] - prefilled)
        final = prefilled + count == prompt_lengths["B"]
        events.append(
            {
                "request": "B",
                "kind": "prefill",
                "prompt_start": prefilled,
                "prompt_tokens": count,
                "final_chunk": final,
            }
        )
        prefilled += count
        if final:
            generated["B"] += 1

    if policy == "fifo":
        while generated["A"] < outputs:
            decode("A")
    if policy in ("fifo", "eager"):
        while prefilled < prompt_lengths["B"]:
            prefill()
    while any(count < outputs for count in generated.values()):
        for name in REQUESTS:
            if 0 < generated[name] < outputs:
                decode(name)
        if prefilled < prompt_lengths["B"]:
            prefill()
    return events


def _sample(sample, policy, expected_tokens, prompts, outputs, chunk, threshold):
    sample = _mapping(sample, "sample")
    if sample.get("policy") != policy or set(sample.get("requests", {})) != set(REQUESTS):
        raise ValueError("sample needs exactly its declared policy and requests A/B")
    for name in REQUESTS:
        request = _mapping(sample["requests"][name], "timed request")
        if (
            not _same(request.get("generated_token_ids"), expected_tokens[name])
            or not _same(request.get("actual_output_tokens"), outputs)
            or not _same(request.get("timed_output_tokens"), outputs - int(name == "A"))
            or request.get("initial_token_timed") is not (name != "A")
            or not isinstance(request.get("emission_times_seconds"), list)
            or len(request["emission_times_seconds"]) != outputs
        ):
            raise ValueError("every timed trajectory/count must match its independent validation")
    expected_events = _expected_events(
        policy, {name: len(prompts[name]) for name in REQUESTS}, outputs, chunk
    )
    events = sample.get("events")
    if not isinstance(events, list) or len(events) != len(expected_events):
        raise ValueError("complete policy quantum events are required")
    previous_end = 0
    emitted = {"A": 1, "B": 0}
    for actual, expected in zip(events, expected_events, strict=True):
        actual = _mapping(actual, "quantum event")
        start, end = actual.get("started_seconds"), actual.get("completed_seconds")
        if (
            actual.keys() != expected.keys() | {"started_seconds", "completed_seconds"}
            or any(not _same(actual.get(name), value) for name, value in expected.items())
            or not _finite(start)
            or not _finite(end)
            or start < previous_end
            or end < start
        ):
            raise ValueError(
                "quantum events must match serialized policy order and legal cache positions"
            )
        if actual["kind"] == "decode" or actual.get("final_chunk"):
            name = actual["request"]
            if end != sample["requests"][name]["emission_times_seconds"][emitted[name]]:
                raise ValueError("token emission must coincide with its completed quantum")
            emitted[name] += 1
        previous_end = end
    computed = deepcopy(sample)
    metrics = _sample_metrics(computed, threshold)
    if not _same(sample.get("metrics"), metrics):
        raise ValueError("sample metrics must match raw observed emission timestamps")
    if not _same(sample.get("active_interference"), computed["active_interference"]):
        raise ValueError("active-interference scopes must match raw quantum windows")
    for name in REQUESTS:
        for field in (
            "inter_token_gap_seconds",
            "gap_distribution",
            "completion_since_arrival_seconds",
        ):
            if not _same(sample["requests"][name].get(field), computed["requests"][name][field]):
                raise ValueError("request latency and gap summaries must match raw emissions")
    return metrics


def chart_data(report):
    report = _mapping(report, "report")
    if (
        report.get("status") != "ok"
        or not _same(report.get("schema_version"), 1)
        or report.get("benchmark") != "qwen3_pd_interleaving"
    ):
        raise ValueError("only successful PD-interleaving schema-1 reports can be plotted")
    if not isinstance(report.get("model"), str) or not report["model"]:
        raise ValueError("checkpoint identity is required")
    for name in ("source_sha256", "checkpoint_sha256", "tokenizer_sha256"):
        _hashes(report.get(name), name)
    _hashes(
        {"implementation": report.get("compiler_and_kernels_sha256")}, "compiler/kernel fingerprint"
    )
    geometry = _mapping(report.get("geometry"), "geometry")
    if (
        geometry.get("storage_dtype") != "float32"
        or not _integer(geometry.get("num_hidden_layers"))
        or not _integer(geometry.get("vocab_size"))
    ):
        raise ValueError("model geometry needs FP32 storage, layer and vocabulary counts")
    fields = (
        "projection_backend",
        "projection_tile",
        "strict_math",
        "projection_relaxed_precision",
    )
    _execution_metadata(
        {
            "candidate_attention_backend": "matrix",
            **{f"candidate_{name}": geometry.get(f"prefill_{name}") for name in fields},
        },
        geometry,
    )
    packed = _precision(report, geometry)
    execution = _mapping(report.get("execution"), "execution")
    if execution.get("cache_reset_policy") not in ("zero_fill", "logical"):
        raise ValueError("an explicit zero_fill or logical cache reset policy is required")
    for name, value in (
        ("cooperative_serial_interleaving", True),
        ("hardware_overlap_claimed", False),
        ("self_request_prefill_decode_overlap", False),
        ("shared_immutable_weights", True),
        ("independent_request_cache_and_workspace", True),
    ):
        if execution.get(name) is not value:
            raise ValueError(
                "chart requires independent requests with cooperative serial execution"
            )
    workload = _mapping(report.get("workload"), "workload")
    policies = workload.get("policies")
    if (
        not isinstance(policies, list)
        or len(policies) < 2
        or any(not isinstance(policy, str) or policy not in POLICIES for policy in policies)
        or len(set(policies)) != len(policies)
    ):
        raise ValueError("distinct recognized scheduling policies are required")
    outputs, chunk, trials = (
        workload.get(name) for name in ("output_tokens_per_request", "chunk_size", "trials")
    )
    if (
        not _integer(outputs, 2)
        or not _integer(chunk)
        or chunk > 4096
        or not _integer(trials)
        or not _integer(workload.get("warmups_per_policy"))
        or not _same(workload.get("timed_output_tokens"), 2 * outputs - 1)
        or not _same(geometry.get("prefill_chunk_size"), chunk)
    ):
        raise ValueError("workload counts must exclude A's untimed first token exactly once")
    documents = _mapping(workload.get("requests"), "request documents")
    if set(documents) != set(REQUESTS):
        raise ValueError("exactly two independent request documents are required")
    prompts = {name: _prompt(documents[name], geometry["vocab_size"]) for name in REQUESTS}
    _require_disjoint_prompts(documents["A"], documents["B"])
    if not _same(
        geometry.get("capacity"), max(len(prompt) for prompt in prompts.values()) + outputs - 1
    ):
        raise ValueError("request cache capacity must cover each complete trajectory")
    correctness = _mapping(report.get("correctness"), "correctness policy")
    interval = correctness.get("cache_check_interval")
    if not _integer(interval) or not _same(
        correctness.get("tolerances"), list(_fidelity_tolerances("float32"))
    ):
        raise ValueError("fixed FP32 logit/cache tolerances and checkpoint interval are required")
    validations = _mapping(report.get("validation"), "policy validations")
    if set(validations) != set(policies):
        raise ValueError("every measured policy needs independent full validation")
    expected = None
    for policy in policies:
        validation = _mapping(validations[policy], "policy validation")
        requests = _mapping(validation.get("requests"), "validated requests")
        if validation.get("passed") is not True or set(requests) != set(REQUESTS):
            raise ValueError("every policy must validate both requests")
        current = {
            name: _validation(
                requests[name],
                prompts[name],
                outputs,
                chunk,
                geometry["num_hidden_layers"],
                geometry["vocab_size"],
                interval,
            )
            for name in REQUESTS
        }
        if expected is not None and current != expected:
            raise ValueError("policies must produce identical validated request trajectories")
        expected = current
    measurement = _mapping(report.get("measurement"), "measurement")
    threshold = measurement.get("stall_threshold_seconds")
    if (
        not _finite(threshold)
        or threshold == 0
        or measurement.get("paired_baseline") != policies[0]
    ):
        raise ValueError(
            "a positive preset stall threshold and declared paired baseline are required"
        )
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != trials:
        raise ValueError("all configured paired trials must be present")
    observations = {policy: [] for policy in policies}
    for trial, sample in enumerate(samples):
        sample = _mapping(sample, "paired trial")
        arms = _mapping(sample.get("policies"), "timed policies")
        if (
            not _same(sample.get("trial"), trial)
            or sample.get("order") != _order(policies, trial)
            or set(arms) != set(policies)
        ):
            raise ValueError(
                "paired trials must have every policy in the prescribed rotating order"
            )
        for policy in policies:
            observations[policy].append(
                _sample(arms[policy], policy, expected, prompts, outputs, chunk, threshold)
            )
    medians = {
        policy: {metric: statistics.median(row[metric] for row in rows) for metric in rows[0]}
        for policy, rows in observations.items()
    }
    if not _same(report.get("medians"), medians):
        raise ValueError("saved medians must match recomputed raw-trial medians")
    ratios = {
        policy: {
            metric: statistics.median(
                baseline[metric] / candidate[metric]
                for baseline, candidate in zip(
                    observations[policies[0]], observations[policy], strict=True
                )
            )
            for metric in ("makespan_seconds", "b_ttft_seconds", "a_gap_max_seconds")
        }
        for policy in policies[1:]
    }
    if not _same(report.get("paired_ratios"), ratios):
        raise ValueError("saved ratios must match medians of within-trial policy ratios")
    return {
        "policies": policies,
        "observations": observations,
        "medians": medians,
        "packed_weights": packed,
    }


def render(report, output, source_name=DEFAULT_INPUT.name):
    data = chart_data(report)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    figure = pyplot.figure(figsize=(style.WIDTH, 11.6), dpi=style.DPI)
    for panel, (metric, title, units, scale) in enumerate(METRICS):
        axis = figure.add_axes((0.29, 0.72 - panel * 0.23, 0.52, 0.12))
        transform = blended_transform_factory(figure.transFigure, axis.transData)
        all_values = []
        for position, policy in enumerate(data["policies"]):
            values = [row[metric] * scale for row in data["observations"][policy]]
            all_values.extend(values)
            jitter = [
                0.16 * (index / max(len(values) - 1, 1) - 0.5) for index in range(len(values))
            ]
            dots = axis.scatter(
                values,
                [position + offset for offset in jitter],
                color=COLORS[policy],
                s=28,
                zorder=3,
            )
            dots.set_gid("recorded-trials")
            median = statistics.median(values)
            bar = axis.vlines(
                median,
                position - 0.18,
                position + 0.18,
                color=COLORS[policy],
                linewidth=2.2,
                zorder=4,
            )
            bar.set_gid("sample-median")
            axis.text(
                0.045,
                position,
                LABELS[policy],
                transform=transform,
                color=COLORS[policy],
                fontsize=10,
                va="center",
            )
            axis.text(
                0.95,
                position,
                f"{median:,.2f}",
                transform=transform,
                color=COLORS[policy],
                fontsize=10,
                va="center",
                ha="right",
            )
        axis.set_xlim(0, max(all_values) * 1.1)
        axis.set_ylim(len(data["policies"]) - 0.5, -0.5)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=18)
        axis.set_xlabel(f"{units} · lower is better", fontsize=10)
        axis.text(
            0.95,
            1.12,
            "MEDIAN",
            transform=blended_transform_factory(figure.transFigure, axis.transAxes),
            color=style.INK_MUTED,
            fontsize=9,
            ha="right",
            va="bottom",
        )
        style.frame(axis)
    workload = report["workload"]
    count = workload["output_tokens_per_request"]
    prompt_count = workload["requests"]["B"]["actual_prompt_tokens"]
    packing = (
        "lossless-packed decode/head weights" if data["packed_weights"] else "unpacked FP32 weights"
    )
    cache_reset = (
        "logical invalidation"
        if report["execution"]["cache_reset_policy"] == "logical"
        else "zero fill"
    )
    style.headings(
        figure,
        "When a new prompt meets an active decode",
        f"{report['model']} · same meTile backend · chunk {workload['chunk_size']}\nFP32 arithmetic · {packing}",
        f"{report.get('hardware', {}).get('device_name', 'Apple silicon')} · source: {source_name}\n"
        f"B: {prompt_count:,} prompt / {count:,} output tokens. A already returned its first token at t0.\n"
        f"Timed work: {count - 1:,} remaining A tokens + {count:,} B tokens = {2 * count - 1:,} output IDs.\n"
        f"Dots: all {workload['trials']} paired trials per policy; bars: medians. No confidence intervals or invented densities.\n"
        "Cooperative serial interleaving; no hardware overlap or SM reservation. Native MLX validates accuracy only.\n"
        "This is not an MLX speed comparison. Shorter A stalls may trade off against B's first-token latency.\n"
        f"A maximum gap includes initial resumption. Stall threshold: {report['measurement']['stall_threshold_seconds'] * 1000:g} ms.\n"
        f"Configured warmups per policy: {workload['warmups_per_policy']}. Cache reset: {cache_reset}.\n"
        "Setup and A's initial prefill are excluded; B's reset and all scheduling work are timed.",
    )
    style.save(figure, output)
    pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    render(json.loads(arguments.input.read_text()), arguments.output, arguments.input.name)


if __name__ == "__main__":
    main()
