"""Render validated Qwen3 generation trials without loading weights or running a GPU.

Each panel uses one common, zero-based scale across workloads. Dots preserve
every measured trial; bars mark sample medians. Relative speeds are medians
of within-trial MLX / meTile latency ratios, not ratios of reported medians.
The renderer never reconstructs samples or estimates density from five trials.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.megakernels.qwen3_end_to_end import _summarize
from benchmarks.plots import chartstyle as style

DEFAULT_INPUT = Path("benchmarks/results/m5-qwen3-gpu-wide-end-to-end.json")
DEFAULT_OUTPUT = Path("docs/_static/qwen3-gpu-wide-end-to-end.png")
ARMS = (("MLX", "Native MLX", style.PREFILL), ("meTile", "meTile GPU-wide", style.DECODE))
METRICS = (
    (
        "ttft_ms",
        "Time to first token",
        "milliseconds · lower is faster",
        "time_to_first_token_seconds",
    ),
    (
        "decode_tokens_per_second",
        "Decode after the first token",
        "tokens / second · higher is faster",
        "decode_wall_seconds",
    ),
    ("total_ms", "Complete generation", "milliseconds · lower is faster", "total_wall_seconds"),
)


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _lengths(values, minimum):
    return (
        isinstance(values, list)
        and bool(values)
        and all(_integer(value, minimum) for value in values)
        and len(set(values)) == len(values)
    )


def _tokens(values, count, vocabulary):
    return (
        isinstance(values, list)
        and len(values) == count
        and all(_integer(value, 0) and value < vocabulary for value in values)
    )


def _correctness(result, prompt, output_count, geometry):
    evidence = result.get("correctness", {})
    if not isinstance(evidence, dict):
        raise ValueError("complete correctness evidence must be a mapping")
    vocabulary = geometry["vocab_size"]
    generated = evidence.get("generated_token_ids")
    checks = evidence.get("checks")
    if (
        evidence.get("passed") is not True
        or type(evidence.get("actual_output_tokens")) is not int
        or evidence["actual_output_tokens"] != output_count
        or not _tokens(generated, output_count, vocabulary)
        or not isinstance(checks, list)
        or len(checks) != len(prompt) + output_count - 1
    ):
        raise ValueError(
            "complete prompt/decode correctness checks and generated tokens are required"
        )
    inputs = prompt + generated[:-1]
    for position, (check, token) in enumerate(zip(checks, inputs, strict=True)):
        if not isinstance(check, dict):
            raise ValueError("every correctness check must be a mapping")
        logits, cache = check.get("logits", {}), check.get("cache", {})
        selection = check.get("gpu_greedy_selection", {})
        if any(not isinstance(value, dict) for value in (logits, cache, selection)):
            raise ValueError("logits, cache and GPU selection checks must be mappings")
        selected = selection.get("token")
        if (
            check.get("passed") is not True
            or type(check.get("position")) is not int
            or check["position"] != position
            or type(check.get("input_token")) is not int
            or check["input_token"] != token
            or logits.get("passed") is not True
            or logits.get("greedy_token_equal") is not True
            or not _tokens([selected], 1, vocabulary)
            or not _tokens(
                [logits.get("mlx_next_token"), logits.get("metile_next_token")], 2, vocabulary
            )
            or logits.get("mlx_next_token") != selected
            or logits.get("metile_next_token") != selected
            or cache.get("passed") is not True
            or not _integer(cache.get("layers_checked"))
            or not _integer(cache.get("valid_tokens_checked"))
            or cache.get("layers_checked") != geometry["num_hidden_layers"]
            or cache.get("valid_tokens_checked") != position + 1
            or cache.get("keys_and_values_checked") is not True
            or selection.get("passed") is not True
        ):
            raise ValueError("every forward must pass logits, GPU selection and complete KV checks")
        if position >= len(prompt) - 1 and selected != generated[position - len(prompt) + 1]:
            raise ValueError("validated generated tokens must match GPU selections")
    return generated


def chart_data(report):
    """Fail closed on incomplete evidence and recompute all displayed statistics."""
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("status") != "ok"
    ):
        raise ValueError("only successful schema-1 generation reports can be plotted")
    weights, geometry = report.get("weights", {}), report.get("geometry", {})
    if any(not isinstance(value, dict) for value in (weights, geometry)):
        raise ValueError("weight and geometry metadata must be mappings")
    dtype = weights.get("both_backends")
    if (
        dtype not in ("float16", "float32")
        or geometry.get("dtype") != dtype
        or weights.get("same_values") is not True
        or weights.get("quantized") is not False
    ):
        raise ValueError("both backends require identical unquantized weights and storage dtype")
    if dtype == "float32" and report.get("software", {}).get("MLX_ENABLE_TF32") != "0":
        raise ValueError("FP32 comparisons must disable MLX TF32")
    if not isinstance(report.get("model"), str) or not report["model"]:
        raise ValueError("a model checkpoint name is required")
    if not _integer(geometry.get("num_hidden_layers")) or not _integer(geometry.get("vocab_size")):
        raise ValueError("model layer and vocabulary counts are required")
    execution, workload = report.get("execution", {}), report.get("workload", {})
    if any(not isinstance(value, dict) for value in (execution, workload)):
        raise ValueError("execution and workload metadata must be mappings")
    layers = geometry["num_hidden_layers"]
    if (
        execution.get("single_dispatch") is not False
        or execution.get("candidate_dispatches_per_prefill_token") != 1 + 8 * layers
        or execution.get("candidate_dispatches_per_generated_token") != 5 + 8 * layers
        or not str(execution.get("candidate_prefill", "")).startswith("sequential tokens;")
        or not str(execution.get("native_prefill", "")).startswith(
            "batched prompt except final token;"
        )
    ):
        raise ValueError(
            "the chart requires staged DSL decode, sequential DSL and batched native prefill"
        )
    prompt_lengths, output_lengths = workload.get("prompt_lengths"), workload.get("output_lengths")
    if not _lengths(prompt_lengths, 1) or not _lengths(output_lengths, 2):
        raise ValueError("prompt/output lengths must be distinct positive integer lists")
    if (
        type(workload.get("batch")) is not int
        or workload["batch"] != 1
        or not _integer(workload.get("trials"), 3)
    ):
        raise ValueError("batch one and at least three paired trials are required")
    if workload.get("capacity") != max(prompt_lengths) + max(output_lengths) - 1:
        raise ValueError("cache capacity must cover the full requested generation grid")
    expected_workloads = [
        (prompt, output) for prompt in prompt_lengths for output in output_lengths
    ]
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(expected_workloads):
        raise ValueError("results must cover the complete prompt/output grid")
    rows, known_prompts = [], {}
    for (prompt_count, output_count), result in zip(expected_workloads, results, strict=True):
        if (
            not isinstance(result, dict)
            or result.get("status") != "ok"
            or result.get("prompt_tokens") != prompt_count
            or result.get("requested_output_tokens") != output_count
        ):
            raise ValueError("each requested workload requires one successful ordered result")
        prompt = result.get("prompt_token_ids")
        if not _tokens(prompt, prompt_count, geometry["vocab_size"]):
            raise ValueError(
                "prompt token IDs must match the declared prompt length and vocabulary"
            )
        if prompt_count in known_prompts and known_prompts[prompt_count] != prompt:
            raise ValueError("a prompt must be identical across its output-length comparisons")
        known_prompts[prompt_count] = prompt
        generated = _correctness(result, prompt, output_count, geometry)
        samples = result.get("samples")
        if not isinstance(samples, list) or len(samples) != workload["trials"]:
            raise ValueError("raw sample count must match workload.trials")
        for index, sample in enumerate(samples):
            if (
                not isinstance(sample, dict)
                or type(sample.get("trial")) is not int
                or sample["trial"] != index
            ):
                raise ValueError("raw trials must have distinct consecutive trial identifiers")
            for backend, _, _ in ARMS:
                measured = sample.get(backend, {})
                if not isinstance(measured, dict):
                    raise ValueError("every measured arm must be a mapping")
                if (
                    type(measured.get("actual_output_tokens")) is not int
                    or measured["actual_output_tokens"] != output_count
                    or measured.get("generated_token_ids") != generated
                    or not _tokens(
                        measured.get("generated_token_ids"), output_count, geometry["vocab_size"]
                    )
                ):
                    raise ValueError(
                        "every timed sequence must match the validated output count and tokens"
                    )
                for _, _, _, key in METRICS:
                    value = measured.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                        or not math.isfinite(value * 1000)
                        or not math.isfinite((output_count - 1) / value)
                    ):
                        raise ValueError(
                            "raw durations and derived rates must be finite positive numbers"
                        )
            for _, _, _, source in METRICS:
                ratio = sample["MLX"][source] / sample["meTile"][source]
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError("paired speed ratios must be finite positive numbers")
        summary = _summarize(samples)
        metrics = {}
        for key, _, _, source in METRICS:
            metrics[key] = {
                "paired_speedup": summary[source]["paired_speedup"],
                "arms": [
                    {
                        "label": label,
                        "color": color,
                        "values": [
                            (output_count - 1) / sample[backend][source]
                            if key == "decode_tokens_per_second"
                            else sample[backend][source] * 1000
                            for sample in samples
                        ],
                    }
                    for backend, label, color in ARMS
                ],
            }
        rows.append(
            {"prompt_tokens": prompt_count, "output_tokens": output_count, "metrics": metrics}
        )
    return rows


def render(report, output, source_name=DEFAULT_INPUT.name):
    rows = chart_data(report)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    height = 4.3 + 1.65 * len(rows)
    figure, axes = pyplot.subplots(3, 1, figsize=(style.WIDTH, height), dpi=style.DPI)
    for axis, (key, title, units, _) in zip(axes, METRICS, strict=True):
        transform = blended_transform_factory(figure.transFigure, axis.transData)
        values = [
            value for row in rows for arm in row["metrics"][key]["arms"] for value in arm["values"]
        ]
        axis.set_xlim(0, max(values) * 1.08)
        axis.set_ylim(len(rows) - 0.48, -0.55)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=17)
        axis.set_xlabel(units, fontsize=10)
        for column, text in ((0.83, "MEDIAN"), (0.97, "VS MLX")):
            axis.text(
                column,
                1.08,
                text,
                transform=blended_transform_factory(figure.transFigure, axis.transAxes),
                fontsize=9,
                ha="right",
                va="bottom",
                color=style.INK_MUTED,
            )
        for position, row in enumerate(rows):
            metric = row["metrics"][key]
            axis.text(
                0.045,
                position,
                f"{row['prompt_tokens']} prompt\n{row['output_tokens']} output",
                transform=transform,
                va="center",
                fontsize=10,
                color=style.INK,
            )
            axis.text(
                0.97,
                position,
                f"{metric['paired_speedup']:.2f}x",
                transform=transform,
                va="center",
                ha="right",
                fontsize=10,
                color=style.INK_SOFT,
            )
            for arm_index, arm in enumerate(metric["arms"]):
                center = position + (arm_index - 0.5) * 0.46
                observations = arm["values"]
                jitter = [
                    0.1 * (index / (len(observations) - 1) - 0.5)
                    for index in range(len(observations))
                ]
                dots = axis.scatter(
                    observations,
                    [center + offset for offset in jitter],
                    s=23,
                    color=arm["color"],
                    edgecolors=style.SURFACE,
                    linewidths=0.5,
                    zorder=3,
                )
                dots.set_gid("recorded-trials")
                median = statistics.median(observations)
                bar = axis.vlines(
                    median, center - 0.10, center + 0.10, color=arm["color"], linewidth=2, zorder=4
                )
                bar.set_gid("sample-median")
                axis.text(
                    0.83,
                    center,
                    f"{median:,.1f}",
                    transform=transform,
                    va="center",
                    ha="right",
                    fontsize=10,
                    color=arm["color"],
                )
        style.frame(axis)
    hardware, software = report.get("hardware", {}), report.get("software", {})
    chip = hardware.get("device_name", hardware.get("chip", "Apple silicon"))
    dtype = report["weights"]["both_backends"].replace("float", "FP")
    precision = f"identical {dtype} weights"
    if dtype == "FP32":
        precision += " · MLX TF32 disabled"
    style.headings(
        figure,
        "Qwen3 generation, end to end",
        f"{report['model']} · {precision} · batch one\n"
        "Fixed synthetic prompts · GPU greedy selection · exact output lengths",
        f"{chip} · MLX {software.get('mlx', 'unknown')} · source: {source_name}\n"
        f"Dots: all {report['workload']['trials']} paired trials per arm. Bars: medians. No confidence intervals.\n"
        "VS MLX: median paired latency ratio; 1.00x is parity, above 1.00x favors meTile.\n"
        "Decode rate counts output tokens after the first: (output count - 1) / decode time.\n"
        "Prefill: sequential DSL tokens versus batched native MLX. Persistent meTile buffers; native allocation.\n"
        "Includes cache reset, host submission and token retrieval. Excludes compilation and tokenization.",
    )
    handles = [
        Line2D([], [], marker="o", color=color, linestyle="none", label=label, markersize=6)
        for _, label, color in ARMS
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 1 - 1.08 / height),
        ncol=2,
        borderaxespad=0,
    )
    figure.subplots_adjust(
        left=0.23, right=0.72, top=1 - 1.85 / height, bottom=1.65 / height, hspace=0.8
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
