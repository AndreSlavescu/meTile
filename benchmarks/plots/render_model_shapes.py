"""Render measured projection and MLP-block speedups at model-shaped dimensions."""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _root)

from benchmarks.plots import chartstyle as style

SERIES = (
    ("prefill_down_speedup", style.DECODE, "prefill, down projection"),
    ("row_1", style.ACCENT, "decode, 1 row"),
    ("row_16", style.PREFILL, "decode, 16 rows"),
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
    return f"{name}   ·   width {record['hidden']}"


def main():
    arguments = _arguments()
    payload = json.loads(arguments.input.read_text())
    if payload.get("scope") != "model_shape_matrix":
        raise ValueError("input is not a model shape matrix result")

    pyplot = style.matplotlib_pyplot()
    records = sorted(payload["models"], key=lambda record: record["hidden"])
    for record in records:
        record["row_1"] = record["block_speedup"]["1"]
        record["row_16"] = record["block_speedup"]["16"]

    height = 0.62 * len(records) + 2.6
    figure, axis = pyplot.subplots(figsize=(10.6, height), dpi=180)
    figure.patch.set_facecolor(style.SURFACE)
    style.parity_rule(axis, "vertical")

    slots = [float(index) for index in range(len(records))]
    offsets = (0.22, 0.0, -0.22)
    for (key, colour, name), offset in zip(SERIES, offsets):
        values = [record[key] for record in records]
        positions = [slot + offset for slot in slots]
        axis.scatter(
            values,
            positions,
            s=54,
            color=colour,
            edgecolor=style.SURFACE,
            linewidth=1.1,
            zorder=3,
            label=name,
        )
        for value, position in zip(values, positions):
            axis.annotate(
                style.multiplier(value),
                (value, position),
                textcoords="offset points",
                xytext=(9, 0),
                ha="left",
                va="center",
                fontsize=8,
                color=style.INK_SOFT,
            )

    lowest = min(min(record[key] for record in records) for key, _, _ in SERIES)
    highest = max(max(record[key] for record in records) for key, _, _ in SERIES)
    axis.set_xlim(min(0.92, lowest - 0.06), highest + 0.34)
    axis.set_ylim(len(records) - 0.45, -0.75)
    axis.set_yticks(slots)
    axis.set_yticklabels([_label(record) for record in records], fontsize=9.5)
    axis.set_xlabel("speedup vs native MLX", fontsize=9.5, color=style.INK_SOFT)
    style.frame(axis, grid_axis="x")
    axis.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=style.INK_SOFT)

    style.headings(
        figure,
        "Projection and block speedups at model-shaped dimensions",
        "INT4 group 64 · synthetic layer workloads, not end-to-end generation",
        f"Apple M5 · int4 group 64 · identical weights on both sides · "
        f"{payload['prompt_rows']} prompt rows · {payload['rounds']} rounds",
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, arguments.output)
    pyplot.close(figure)


if __name__ == "__main__":
    main()
