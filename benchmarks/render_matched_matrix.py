"""Render the matched-representation matrix as one speedup chart, in multiplier units.

Identical weights and format on both sides at every point, so each line is a kernel
comparison rather than a representation change.
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import chartstyle as style  # noqa: E402

FORMATS = (
    ("bf16", style.DECODE, "BF16"),
    ("int4", style.PREFILL, "INT4 g64"),
    ("int8", style.ACCENT, "INT8 g64"),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results/m5-matched-representation-matrix.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/mlx-matched-speedup.png")
    )
    return parser.parse_args()


def _series(measurements, format_name):
    rows = sorted(
        (record for record in measurements if record["format"] == format_name),
        key=lambda record: record["rows"],
    )
    return [record["rows"] for record in rows], [record["speedup"] for record in rows]


def render(payload, output):
    pyplot = style.matplotlib_pyplot()
    measurements = payload["measurements"]

    figure, axis = pyplot.subplots(figsize=(10.0, 5.2), dpi=180)
    figure.patch.set_facecolor(style.SURFACE)
    style.parity_rule(axis, "horizontal")

    peak, floor = 1.0, 1.0
    for format_name, colour, label in FORMATS:
        rows, speedups = _series(measurements, format_name)
        if not rows:
            continue
        peak, floor = max(peak, max(speedups)), min(floor, min(speedups))
        axis.plot(
            rows,
            speedups,
            color=colour,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            markeredgecolor=style.SURFACE,
            markeredgewidth=1.1,
            label=label,
            zorder=3,
        )

    # Label only the peak of the one line that leaves the parity band: a number on every
    # point would just be a table.
    rows, speedups = _series(measurements, "bf16")
    best = max(range(len(speedups)), key=speedups.__getitem__)
    axis.annotate(
        f"{style.multiplier(speedups[best])} at {rows[best]} rows",
        (rows[best], speedups[best]),
        textcoords="offset points",
        xytext=(10, 6),
        fontsize=9,
        color=style.INK_SOFT,
    )

    ticker = __import__("matplotlib").ticker
    axis.set_xscale("log", base=2)
    axis.set_xticks(rows)
    axis.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{int(value)}")
    )
    axis.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: style.multiplier(value))
    )
    axis.set_xlim(0.9, max(rows) * 1.25)
    axis.set_ylim(min(0.92, floor - 0.06), peak + 0.10)
    axis.set_xlabel(
        "rows per dispatch  (1 = single-token decode)", fontsize=9.5, color=style.INK_SOFT
    )
    axis.set_ylabel("speedup vs native MLX", fontsize=9.5, color=style.INK_SOFT)
    style.frame(axis, grid_axis="y")

    legend = axis.legend(
        loc="upper right", frameon=False, fontsize=9.5, labelcolor=style.INK_SOFT
    )
    legend.set_zorder(5)

    hardware = payload.get("hardware", {})
    software = payload.get("software", {})
    shape = payload.get("shape", {})
    style.headings(
        figure,
        "Speedup by batch size, at matched weight representation",
        "Same weights, same format, both sides.",
        f"{hardware.get('chip', '')} · {hardware.get('memory', '')} · "
        f"MLX {software.get('mlx', '')} · {shape.get('label', '')} "
        f"{shape.get('hidden')}->{shape.get('intermediate')}->{shape.get('hidden')} · "
        f"{payload.get('rounds')} interleaved rounds",
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, output)
    pyplot.close(figure)


def main():
    arguments = _arguments()
    payload = json.loads(arguments.input.read_text())
    if payload.get("scope") != "matched_representation_matrix":
        raise ValueError("input is not a matched-representation matrix result")
    render(payload, arguments.output)


if __name__ == "__main__":
    main()
