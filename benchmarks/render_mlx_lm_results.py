"""Render reproducible PNG bar charts from an MLX-LM benchmark suite result."""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

MLX_COLOR = "#ffb000"
METILE_COLOR = "#3f7ee8"
_CHART_METRICS = (
    "mlx_decode_tokens_per_second",
    "metile_decode_tokens_per_second",
    "mlx_time_to_first_token_seconds",
    "metile_time_to_first_token_seconds",
    "mlx_elapsed_seconds",
    "metile_elapsed_seconds",
)
_WORKLOAD_KEYS = ("prompt_tokens", "generation_tokens", "trials", "seed")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("docs/_static/mlx-model-throughput.png"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        default=Path("docs/_static/mlx-model-latency.png"),
    )
    return parser.parse_args()


def _model_label(model):
    name = model.rsplit("/", 1)[-1]
    for suffix in ("-Instruct-4bit", "-4bit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    family, size = name.rsplit("-", 1)
    family = family.replace("Llama-", "Llama ").replace("Qwen2.5", "Qwen 2.5")
    return f"{family}\n{size} 4-bit"


def _suite_context(suite):
    first = suite["models"][0]
    workload = first["workload"]
    hardware = first.get("hardware", {})
    software = first.get("software", {})
    chip = hardware.get("chip") or hardware.get("processor") or hardware.get("machine", "unknown")
    comparison_modes = {model.get("comparison_mode", "alternating") for model in suite["models"]}
    trial_kind = (
        "native-fallback trials"
        if comparison_modes == {"shared_native_fallback"}
        else "alternating trials"
    )
    subtitle = (
        f"{chip} · {workload['prompt_tokens']} prompt tokens · "
        f"{workload['generation_tokens']} generated · "
        f"median of {workload['trials']} {trial_kind}"
    )
    footer = (
        f"MLX {software.get('mlx', 'unknown')} · MLX-LM {software.get('mlx_lm', 'unknown')} · "
        f"seed {workload.get('seed', 0)} · rev {first.get('revision', 'unknown')[:7]}"
    )
    return subtitle, footer


def _validate_suite(suite):
    models = suite.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("benchmark suite contains no model results")

    reference = models[0]
    reference_workload = reference.get("workload", {})
    context = {
        "hardware": reference.get("hardware", {}),
        "software": reference.get("software", {}),
        "workload": {key: reference_workload.get(key) for key in _WORKLOAD_KEYS},
    }
    comparison_modes = set()
    model_names = set()
    for result in models:
        model = result.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("every benchmark result requires a model name")
        if model in model_names:
            raise ValueError(f"duplicate benchmark model: {model}")
        model_names.add(model)

        workload = result.get("workload", {})
        result_context = {
            "hardware": result.get("hardware", {}),
            "software": result.get("software", {}),
            "workload": {key: workload.get(key) for key in _WORKLOAD_KEYS},
        }
        if result_context != context:
            raise ValueError("all charted models must share hardware, software, and workload")
        comparison_modes.add(result.get("comparison_mode", "alternating"))

        medians = result.get("medians", {})
        for metric in _CHART_METRICS:
            value = medians.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"benchmark metric {metric!r} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"benchmark metric {metric!r} must be finite and positive")

    if len(comparison_modes) != 1:
        raise ValueError("all charted models must use the same comparison mode")


def _chart_data(suite):
    models = suite["models"]
    return {
        "labels": [_model_label(result["model"]) for result in models],
        "mlx": [result["medians"]["mlx_decode_tokens_per_second"] for result in models],
        "metile": [result["medians"]["metile_decode_tokens_per_second"] for result in models],
        "mlx_ttft_ms": [
            result["medians"]["mlx_time_to_first_token_seconds"] * 1e3 for result in models
        ],
        "metile_ttft_ms": [
            result["medians"]["metile_time_to_first_token_seconds"] * 1e3 for result in models
        ],
        "mlx_total_seconds": [result["medians"]["mlx_elapsed_seconds"] for result in models],
        "metile_total_seconds": [result["medians"]["metile_elapsed_seconds"] for result in models],
    }


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError(
            "Rendering benchmark charts requires the 'benchmarks' extra: "
            "pip install -e '.[benchmarks]'"
        ) from error
    return pyplot


def _style_axes(axes, subtitle, footer):
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.text(
        0,
        1.015,
        subtitle,
        transform=axes.transAxes,
        color="#666666",
        fontsize=9,
        va="bottom",
    )
    axes.text(
        1,
        -0.19,
        footer,
        transform=axes.transAxes,
        color="#777777",
        fontsize=8,
        ha="right",
        va="top",
    )


def _validate_text_layout(figure):
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [
        artist
        for artist in figure.findobj(Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    bounds = [(artist, artist.get_window_extent(renderer).padded(2)) for artist in texts]
    canvas = figure.bbox
    for artist, box in bounds:
        if box.x0 < canvas.x0 or box.y0 < canvas.y0 or box.x1 > canvas.x1 or box.y1 > canvas.y1:
            raise RuntimeError(f"chart text leaves the canvas: {artist.get_text()!r}")
    for (left, left_box), (right, right_box) in combinations(bounds, 2):
        if left_box.overlaps(right_box):
            raise RuntimeError(f"chart text overlaps: {left.get_text()!r} and {right.get_text()!r}")


def _render_throughput(suite, output):
    _validate_suite(suite)
    pyplot = _matplotlib()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    positions = list(range(len(data["labels"])))
    width = 0.34

    figure_width = max(10.0, 2.1 * len(data["labels"]) + 1.6)
    figure, axes = pyplot.subplots(figsize=(figure_width, 5.6), dpi=180)
    mlx_bars = axes.bar(
        [position - width / 2 for position in positions],
        data["mlx"],
        width,
        label="Native MLX",
        color=MLX_COLOR,
    )
    metile_bars = axes.bar(
        [position + width / 2 for position in positions],
        data["metile"],
        width,
        label="MLX + meTile",
        color=METILE_COLOR,
    )
    axes.bar_label(mlx_bars, fmt="%.1f", padding=3, fontsize=8, color="#444444")
    axes.bar_label(metile_bars, fmt="%.1f", padding=3, fontsize=8, color="#444444")
    axes.set_title("Decode throughput by model", loc="left", fontsize=18, pad=28)
    axes.set_ylabel("Tokens / second")
    axes.set_xticks(positions, data["labels"])
    axes.set_ylim(0, max(data["mlx"] + data["metile"]) * 1.18)
    axes.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=2)
    _style_axes(axes, subtitle, footer)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.24)
    _validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", metadata={"Software": "meTile benchmark renderer"})
    pyplot.close(figure)


def _render_latency(suite, output):
    _validate_suite(suite)
    pyplot = _matplotlib()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    positions = list(range(len(data["labels"])))
    width = 0.34
    panels = (
        (
            "Time to first token",
            "Milliseconds",
            data["mlx_ttft_ms"],
            data["metile_ttft_ms"],
            "%.1f",
        ),
        (
            "End-to-end generation",
            "Seconds",
            data["mlx_total_seconds"],
            data["metile_total_seconds"],
            "%.2f",
        ),
    )

    figure_width = max(12.0, 2.2 * len(data["labels"]) + 3.0)
    figure, axes = pyplot.subplots(1, 2, figsize=(figure_width, 5.8), dpi=180)
    for axes_index, (title, ylabel, mlx_values, metile_values, label_format) in enumerate(panels):
        axis = axes[axes_index]
        mlx_bars = axis.bar(
            [position - width / 2 for position in positions],
            mlx_values,
            width,
            label="Native MLX",
            color=MLX_COLOR,
        )
        metile_bars = axis.bar(
            [position + width / 2 for position in positions],
            metile_values,
            width,
            label="MLX + meTile",
            color=METILE_COLOR,
        )
        axis.bar_label(mlx_bars, fmt=label_format, padding=3, fontsize=7, color="#444444")
        axis.bar_label(metile_bars, fmt=label_format, padding=3, fontsize=7, color="#444444")
        axis.set_title(title, loc="left", fontsize=12, pad=10)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, data["labels"], fontsize=8)
        axis.set_ylim(0, max(mlx_values + metile_values) * 1.18)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Latency by model (lower is better)", x=0.07, y=0.965, ha="left", fontsize=18)
    figure.text(0.07, 0.88, subtitle, color="#666666", fontsize=9, va="bottom")
    figure.text(0.98, 0.035, footer, color="#777777", fontsize=8, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.965), ncol=2
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.24)
    _validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", metadata={"Software": "meTile benchmark renderer"})
    pyplot.close(figure)


def main():
    arguments = _arguments()
    suite = json.loads(arguments.input.read_text())
    _render_throughput(suite, arguments.throughput_output)
    _render_latency(suite, arguments.latency_output)
    print(f"Wrote {arguments.throughput_output}")
    print(f"Wrote {arguments.latency_output}")


if __name__ == "__main__":
    main()
