"""Render reproducible PNG bar charts from an MLX-LM benchmark suite result."""

import argparse
import json
from itertools import combinations
from pathlib import Path

MLX_COLOR = "#ffb000"
METILE_COLOR = "#3f7ee8"
END_TO_END_COLOR = "#8b5cf6"
TTFT_COLOR = "#16a085"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("docs/_static/mlx-model-throughput.png"),
    )
    parser.add_argument(
        "--speedup-output",
        type=Path,
        default=Path("docs/_static/mlx-model-speedups.png"),
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
    subtitle = (
        f"{chip} · {workload['prompt_tokens']} prompt tokens · "
        f"{workload['generation_tokens']} generated · "
        f"median of {workload['trials']} alternating trials"
    )
    footer = (
        f"MLX {software.get('mlx', 'unknown')} · MLX-LM {software.get('mlx_lm', 'unknown')} · "
        f"seed {workload.get('seed', 0)} · rev {first.get('revision', 'unknown')[:7]}"
    )
    return subtitle, footer


def _chart_data(suite):
    models = suite["models"]
    return {
        "labels": [_model_label(result["model"]) for result in models],
        "mlx": [result["medians"]["mlx_decode_tokens_per_second"] for result in models],
        "metile": [result["medians"]["metile_decode_tokens_per_second"] for result in models],
        "decode_change": [(result["medians"]["decode_speedup"] - 1.0) * 100.0 for result in models],
        "ttft_change": [
            (result["medians"]["ttft_speedup"] - 1.0) * 100.0
            if "ttft_speedup" in result["medians"]
            else None
            for result in models
        ],
        "end_to_end_change": [
            (result["medians"]["end_to_end_speedup"] - 1.0) * 100.0 for result in models
        ],
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
    pyplot = _matplotlib()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    positions = list(range(len(data["labels"])))
    width = 0.34

    figure, axes = pyplot.subplots(figsize=(10, 5.6), dpi=180)
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


def _render_speedups(suite, output):
    pyplot = _matplotlib()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    positions = list(range(len(data["labels"])))
    series = [
        ("Decode", data["decode_change"], METILE_COLOR),
        ("End-to-end", data["end_to_end_change"], END_TO_END_COLOR),
    ]
    if all(value is not None for value in data["ttft_change"]):
        series.insert(1, ("TTFT", data["ttft_change"], TTFT_COLOR))
    width = 0.72 / len(series)

    figure, axes = pyplot.subplots(figsize=(10, 5.6), dpi=180)
    for index, (label, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = axes.bar(
            [position + offset for position in positions],
            values,
            width,
            label=label,
            color=color,
        )
        axes.bar_label(bars, fmt="%+.1f%%", padding=3, fontsize=8, color="#444444")
    axes.axhline(0, color="#555555", linewidth=1.1)
    axes.set_title("Latency and throughput change vs native MLX", loc="left", fontsize=18, pad=28)
    axes.set_ylabel("Improvement vs MLX (%)")
    axes.set_xticks(positions, data["labels"])
    largest = max(abs(value) for _, values, _ in series for value in values)
    limit = max(2.0, largest * 1.5)
    axes.set_ylim(-limit, limit)
    axes.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.14),
        ncol=len(series),
    )
    _style_axes(axes, subtitle, footer)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.24)
    _validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", metadata={"Software": "meTile benchmark renderer"})
    pyplot.close(figure)


def main():
    arguments = _arguments()
    suite = json.loads(arguments.input.read_text())
    if not suite.get("models"):
        raise ValueError("benchmark suite contains no model results")
    _render_throughput(suite, arguments.throughput_output)
    _render_speedups(suite, arguments.speedup_output)
    print(f"Wrote {arguments.throughput_output}")
    print(f"Wrote {arguments.speedup_output}")


if __name__ == "__main__":
    main()
