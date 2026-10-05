"""Plot recorded MLX-LM first-token trials, without running models or a GPU.

Each distribution contains one model, one selected plan, and one measured arm
from a single suite. The harness records wall-clock time from stream generation
to its first response, not isolated kernel or GPU time. Shared native fallback
samples are drawn once. Neither tuning trials nor summary medians are samples.

The violin is a descriptive Scott-bandwidth KDE clipped to the observed range.
Eight samples and three distinct values are minimum rendering safeguards, not
a statistical power claim. Smaller or effectively constant groups show only
their observed dots, range, and median. No confidence intervals are estimated.
"""

import argparse
import inspect
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_mlx_lm_results import _model_label, _precision_class, _validate_suite

DEFAULT_INPUT = Path("benchmarks/results/m5-mlx-lm-models.json")
DEFAULT_OUTPUT = Path("docs/_static/mlx-model-ttft-distribution.png")
SAMPLE_METRIC = "time_to_first_token_seconds"
MIN_DENSITY_SAMPLES = 8
ARMS = ("MLX", "MLX + meTile")


def chart_data(suite):
    """Extract every final trial, converting seconds to milliseconds exactly once."""
    _validate_suite(suite)
    rows = []
    for model in suite["models"]:
        if (
            _precision_class(model) != "same_precision"
            or model.get("precision_comparison", {}).get("same_weight_representation") is not True
        ):
            raise ValueError("trial distributions require the same weight representation")
        samples = model.get("samples")
        if not isinstance(samples, dict) or set(samples) != set(ARMS):
            raise ValueError("recorded final samples are required for both measured arms")
        expected = model["workload"].get("trials")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
            raise ValueError("workload.trials must be a positive integer")
        extracted = []
        for arm in ARMS:
            records = samples[arm]
            if not isinstance(records, list) or len(records) != expected:
                raise ValueError("sample count must match workload.trials for each arm")
            values = []
            for record in records:
                value = record.get(SAMPLE_METRIC) if isinstance(record, dict) else None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    or not math.isfinite(value * 1000)
                ):
                    raise ValueError("every recorded first-token time must be finite and positive")
                values.append(value * 1000)
            extracted.append(
                {"key": arm, "label": "Native MLX" if arm == "MLX" else arm, "values_ms": values}
            )
        comparison = model.get("comparison_mode", "alternating")
        if comparison == "shared_native_fallback":
            if samples[ARMS[0]] != samples[ARMS[1]]:
                raise ValueError("shared native fallback must contain one identical sample set")
            extracted = [dict(extracted[0], label="Shared native fallback")]
        rows.append(
            {
                "model": model["model"],
                "label": _model_label(model["model"]).replace("\n", " "),
                "comparison_mode": comparison,
                "arms": extracted,
            }
        )
    return rows


def supports_density(values):
    """Avoid inventing a smooth distribution from tiny or degenerate groups."""
    return (
        len(values) >= MIN_DENSITY_SAMPLES
        and len(set(values)) >= 3
        and max(values) - min(values) > max(abs(value) for value in values) * 1e-9
    )


def _draw_arm(axis, values, position, color):
    if supports_density(values):
        orientation = (
            {"orientation": "horizontal"}
            if "orientation" in inspect.signature(axis.violinplot).parameters
            else {"vert": False}
        )
        violin = axis.violinplot(
            [values],
            positions=[position],
            widths=0.62,
            points=100,
            bw_method="scott",
            showextrema=False,
            **orientation,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.16)
            body.set_linewidth(0.8)
            body.set_gid("observed-range-density")
    observed = axis.hlines(position, min(values), max(values), color=color, alpha=0.5, linewidth=1)
    observed.set_gid("observed-range")
    offsets = [
        0.5 * (index - (len(values) - 1) / 2) / max(1, len(values) - 1)
        for index in range(len(values))
    ]
    dots = axis.scatter(
        values,
        [position + offset for offset in offsets],
        s=22,
        color=color,
        edgecolors=style.SURFACE,
        linewidths=0.5,
        zorder=4,
    )
    dots.set_gid("recorded-trials")
    median = statistics.median(values)
    marker = axis.vlines(
        median, position - 0.31, position + 0.31, color=color, linewidth=2, zorder=3
    )
    marker.set_gid("sample-median")


def render(suite, output, source_name=DEFAULT_INPUT.name):
    rows = chart_data(suite)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    height = 2.15 * len(rows) + 2.45
    figure, axes = pyplot.subplots(
        len(rows), 1, figsize=(style.WIDTH, height), dpi=style.DPI, squeeze=False
    )
    all_values = [value for row in rows for arm in row["arms"] for value in arm["values_ms"]]
    padding = max((max(all_values) - min(all_values)) * 0.06, 1.0)
    limits = (max(0, min(all_values) - padding), max(all_values) + padding)
    for axis, row in zip(axes[:, 0], rows, strict=True):
        transform = blended_transform_factory(figure.transFigure, axis.transData)
        for index, arm in enumerate(row["arms"]):
            position = index if len(row["arms"]) == 2 else 0.5
            color = style.PREFILL if arm["key"] == "MLX" else style.DECODE
            values = arm["values_ms"]
            _draw_arm(axis, values, position, color)
            density_note = "" if supports_density(values) else " · dots only"
            axis.text(
                0.045,
                position,
                f"{arm['label']} · n = {len(values)}\n"
                f"{statistics.median(values):.1f} ms median\n"
                f"{min(values):.1f}-{max(values):.1f} ms range{density_note}",
                transform=transform,
                va="center",
                fontsize=9.5,
                color=color,
            )
        axis.set_title(row["label"], loc="left", fontsize=12, fontweight="bold", pad=16)
        if row["comparison_mode"] == "shared_native_fallback":
            axis.text(
                0.98,
                1.04,
                "One shared sample set, drawn once",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=style.INK_MUTED,
            )
        axis.set_ylim(1.5, -0.55)
        axis.set_xlim(*limits)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_yticks([])
        style.frame(axis)
    axes[-1, 0].set_xlabel(
        "Time to first token · wall-clock milliseconds · lower is faster", fontsize=10.5
    )
    first = suite["models"][0]
    workload = first["workload"]
    hardware, software = first.get("hardware", {}), first.get("software", {})
    style.headings(
        figure,
        "First-token latency, trial by trial",
        f"Same weight representation · {workload['prompt_tokens']} prompt / "
        f"{workload['generation_tokens']} requested output tokens · seed {workload['seed']}\n"
        "Fixed synthetic prompt per model · all source models · common time scale",
        f"{hardware.get('chip', 'Apple silicon')} · MLX {software.get('mlx', 'unknown')} · "
        f"MLX-LM {software.get('mlx_lm', 'unknown')} · source: {source_name}\n"
        "Dots: recorded trials, with vertical jitter. Line: observed range. Bar: median. No confidence intervals.\n"
        "Density is descriptive (Scott bandwidth); fewer than 8 or degenerate trials use dots only.",
    )
    figure.subplots_adjust(
        left=0.34, right=0.97, top=1 - 1.65 / height, bottom=1.35 / height, hspace=0.72
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
