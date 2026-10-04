"""Render measured projection and MLP-block speedups at model-shaped dimensions."""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _root)

from benchmarks.plots import chartstyle as style

SERIES = (
    ("prefill_down_speedup", style.DECODE, "Prefill\ndown proj."),
    ("row_1", style.ACCENT, "Decode\n1 row"),
    ("row_16", style.PREFILL, "Decode\n16 rows"),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results/m5-model-shape-matrix.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/mlx-model-shape-speedup.png")
    )
    return parser.parse_args()


def _label(record):
    name = record["model"].replace("-Instruct", "").replace("-4bit", "")
    name = name.replace("-", " ")
    return f"{name}\nwidth {record['hidden']}"


def render(payload, output):
    if payload.get("scope") != "model_shape_matrix":
        raise ValueError("input is not a model shape matrix result")

    pyplot = style.matplotlib_pyplot()
    records = [
        dict(record, row_1=record["block_speedup"]["1"], row_16=record["block_speedup"]["16"])
        for record in sorted(payload["models"], key=lambda record: record["hidden"])
    ]

    from matplotlib.ticker import FuncFormatter, MultipleLocator

    height = 0.65 * len(records) + 2.9
    figure, axis = pyplot.subplots(figsize=(style.WIDTH, height), dpi=style.DPI)
    style.parity_rule(axis, "vertical")
    series = [([record[key] for record in records], colour, name) for key, colour, name in SERIES]
    columns = (0.755, 0.865, 0.975)
    style.comparison_rows(axis, [_label(record) for record in records], series, columns=columns)
    style.value_headers(axis, series, columns=columns, label="MODEL SHAPE")
    lowest = min(min(record[key] for record in records) for key, _, _ in SERIES)
    highest = max(max(record[key] for record in records) for key, _, _ in SERIES)
    axis.set_xlim(min(0.9, lowest - 0.06), highest + 0.12)
    axis.set_ylim(len(records) - 0.45, -0.65)
    axis.xaxis.set_major_locator(MultipleLocator(0.5))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: style.multiplier(value)))
    axis.set_xlabel("Speedup vs native MLX →", fontsize=10.5)

    style.headings(
        figure,
        "Layer workloads, across model shapes",
        "INT4 group 64 · prefill down projection and decode MLP blocks\n"
        "Synthetic layers, not end-to-end generation",
        f"Apple M5 · identical weights on both sides · {payload['prompt_rows']} prompt rows · "
        f"{payload['rounds']} rounds\n1.00x = parity with native MLX; higher is faster",
    )
    figure.subplots_adjust(left=0.305, right=0.65, top=1 - 1.65 / height, bottom=1.25 / height)
    style.save(figure, output)
    pyplot.close(figure)


def main():
    arguments = _arguments()
    render(json.loads(arguments.input.read_text()), arguments.output)


if __name__ == "__main__":
    main()
