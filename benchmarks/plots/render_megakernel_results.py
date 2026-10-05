"""Plot validated full-token megakernel trials without running a model or GPU."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.megakernels.qwen3 import _summarize
from benchmarks.plots import chartstyle as style

DEFAULT_INPUT = Path("benchmarks/results/m5-qwen3-megakernel-fp32.json")
DEFAULT_OUTPUT = Path("docs/_static/qwen3-megakernel-latency.png")
ARMS = (
    ("mlx_wall_seconds", "Native MLX", style.PREFILL),
    ("metile_wall_seconds", "meTile megakernel", style.DECODE),
)


def chart_data(report):
    """Require complete correctness evidence and retain every paired observation."""
    if report.get("schema_version") != 1 or report.get("status") != "ok":
        raise ValueError("only successful schema-1 benchmark reports can be plotted")
    weights, geometry = report.get("weights", {}), report.get("geometry", {})
    dtype = weights.get("both_backends")
    if (
        dtype not in ("float16", "float32")
        or geometry.get("dtype") != dtype
        or weights.get("same_values") is not True
        or weights.get("quantized") is not False
    ):
        raise ValueError("both backends must use identical, unquantized weight values and dtype")
    if dtype == "float32" and report.get("software", {}).get("MLX_ENABLE_TF32") != "0":
        raise ValueError("float32 comparisons must disable MLX TF32")
    if not isinstance(report.get("model"), str) or not report["model"]:
        raise ValueError("a checkpoint name is required")
    execution = report.get("execution", {})
    if (
        execution.get("metile_dispatches_per_forward") != 1
        or execution.get("metile_threadgroups") != 1
    ):
        raise ValueError("this chart requires a one-dispatch, one-threadgroup forward")
    workload = report.get("workload", {})
    if workload.get("batch") != 1 or workload.get("query_tokens") != 1:
        raise ValueError("this chart requires batch one and one query token")
    trials = workload.get("trials")
    steps = workload.get("validation_steps_per_context")
    layers = geometry.get("num_hidden_layers")
    if type(trials) is not int or trials < 3 or type(steps) is not int or steps < 3:
        raise ValueError("at least three trials and validation steps are required")
    if type(layers) is not int or layers < 1:
        raise ValueError("model layer count is required")
    contexts = workload.get("contexts")
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(type(context) is not int or context < 0 for context in contexts)
        or len(set(contexts)) != len(contexts)
    ):
        raise ValueError("prefix lengths must be distinct nonnegative integers")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(contexts):
        raise ValueError("results must cover every requested prefix length")
    rows = []
    for context, result in zip(contexts, results, strict=True):
        if result.get("context_tokens") != context or result.get("status") != "ok":
            raise ValueError("every prefix must have a successful matching result")
        checks = result.get("correctness")
        if not isinstance(checks, list) or len(checks) != steps:
            raise ValueError("complete correctness checks are required before plotting")
        for step, check in enumerate(checks):
            logits, cache = check.get("logits", {}), check.get("cache", {})
            if (
                check.get("position") != context + step
                or logits.get("passed") is not True
                or logits.get("greedy_token_equal") is not True
                or cache.get("passed") is not True
                or cache.get("layers_checked") != layers
                or cache.get("valid_tokens_checked") != context + step + 1
                or cache.get("keys_and_values_checked") is not True
            ):
                raise ValueError("all logits, greedy tokens and complete KV caches must pass")
        samples = result.get("samples")
        if not isinstance(samples, list) or len(samples) != trials:
            raise ValueError("raw sample count must match workload.trials")
        for sample in samples:
            for key, _, _ in ARMS:
                value = sample.get(key) if isinstance(sample, dict) else None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    or not math.isfinite(value * 1000)
                ):
                    raise ValueError("raw durations must be finite, positive seconds")
        summary = _summarize(samples)
        rows.append(
            {
                "context": context,
                "paired_speedup": summary["paired_speedup"],
                "arms": [
                    {
                        "label": label,
                        "color": color,
                        "values_ms": [sample[key] * 1000 for sample in samples],
                    }
                    for key, label, color in ARMS
                ],
            }
        )
    return rows


def render(report, output, source_name=DEFAULT_INPUT.name):
    rows = chart_data(report)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
    from matplotlib.transforms import blended_transform_factory

    height = 1.6 * len(rows) + 3.0
    figure, axis = pyplot.subplots(figsize=(style.WIDTH, height), dpi=style.DPI)
    transform = blended_transform_factory(figure.transFigure, axis.transData)
    all_values = [value for row in rows for arm in row["arms"] for value in arm["values_ms"]]
    logarithmic = max(all_values) / min(all_values) >= 10
    if logarithmic:
        axis.set_xscale("log")
        axis.set_xlim(min(all_values) / 1.6, max(all_values) * 1.6)
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
    else:
        axis.set_xlim(0, max(all_values) * 1.12)
        axis.xaxis.set_major_locator(MaxNLocator(4))
    for row_index, row in enumerate(rows):
        base = row_index * 3.0
        axis.text(
            0.045,
            base - 0.7,
            f"{row['context']} prefix tokens",
            transform=transform,
            fontsize=11,
            fontweight="bold",
            color=style.INK,
            va="center",
        )
        speedup = row["paired_speedup"]
        comparison = (
            f"meTile {1 / speedup:.1f}x slower" if speedup < 1 else f"meTile {speedup:.1f}x faster"
        )
        axis.text(
            0.81,
            base - 0.7,
            comparison,
            transform=transform,
            fontsize=10,
            color=style.INK_SOFT,
            ha="right",
            va="center",
        )
        for arm_index, arm in enumerate(row["arms"]):
            position = base + arm_index
            values, color = arm["values_ms"], arm["color"]
            offsets = [0.38 * (index / (len(values) - 1) - 0.5) for index in range(len(values))]
            dots = axis.scatter(
                values,
                [position + offset for offset in offsets],
                s=27,
                color=color,
                edgecolors=style.SURFACE,
                linewidths=0.6,
                zorder=3,
            )
            dots.set_gid("recorded-trials")
            median = statistics.median(values)
            marker = axis.vlines(
                median, position - 0.29, position + 0.29, color=color, linewidth=2.2, zorder=4
            )
            marker.set_gid("sample-median")
            axis.text(
                0.045,
                position,
                arm["label"],
                transform=transform,
                fontsize=10.5,
                color=color,
                va="center",
            )
            axis.text(
                0.97,
                position,
                f"{median:,.2f} ms",
                transform=transform,
                fontsize=10.5,
                color=color,
                va="center",
                ha="right",
            )
    axis.text(
        0.97,
        -0.7,
        "MEDIAN",
        transform=transform,
        fontsize=9,
        color=style.INK_MUTED,
        va="center",
        ha="right",
    )
    axis.set_ylim((len(rows) - 1) * 3 + 1.65, -1.1)
    axis.set_yticks([])
    scale_note = "log scale" if logarithmic else "linear scale"
    axis.set_xlabel(f"Wall-clock milliseconds · {scale_note} · lower is faster", fontsize=10)
    style.frame(axis)
    hardware, software = report.get("hardware", {}), report.get("software", {})
    chip = hardware.get("device_name", hardware.get("chip", "Apple silicon"))
    dtype = report["weights"]["both_backends"].replace("float", "FP")
    style.headings(
        figure,
        "One-token full-model forward",
        f"{report['model']} · identical {dtype} weights · batch one\n"
        "meTile: every decoder layer and full logits in one dispatch / one threadgroup",
        f"{chip} · MLX {software.get('mlx', 'unknown')} · source: {source_name}\n"
        f"Dots: all {report['workload']['trials']} paired trials per arm. Bars: sample medians. No confidence intervals.\n"
        "Relative speed: median of paired MLX / meTile time ratios.\n"
        "Prepared meTile buffers; native MLX manages intermediate/output allocation.\n"
        "Fixed-position replay. Excludes prefill, compilation and sampling; not generation throughput.",
    )
    figure.subplots_adjust(left=0.34, right=0.81, top=1 - 1.5 / height, bottom=1.75 / height)
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
