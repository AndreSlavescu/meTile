"""Render the two shape-sensitivity charts.

  width   where the 4-bit prefill win comes from, and why it depends on the model
  batch   where weight-once bandwidth is being lost, and which formats still lose it
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import chartstyle as style  # noqa: E402

# Measured with a hand-written streaming-read kernel on the same machine: the most any
# kernel can move, so a line below it is bandwidth left unused.
STREAMING_CEILING = 120.6


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results/m5-shape-sensitivity.json"),
    )
    parser.add_argument(
        "--width-output", type=Path, default=Path("docs/_static/mlx-width-cliff.png")
    )
    parser.add_argument(
        "--batch-output", type=Path, default=Path("docs/_static/mlx-batch-efficiency.png")
    )
    return parser.parse_args()


def _footer(payload):
    hardware = payload.get("hardware", {})
    software = payload.get("software", {})
    return (
        f"{hardware.get('chip', '')} · {hardware.get('memory', '')} · "
        f"MLX {software.get('mlx', '')} · identical weights and format on both sides · "
        f"{payload.get('rounds')} rounds"
    )


def render_width(payload, output):
    pyplot = style.matplotlib_pyplot()
    records = sorted(payload["width_sweep"], key=lambda record: record["output_features"])
    widths = [record["output_features"] for record in records]
    speedups = [record["speedup"] for record in records]

    figure, axis = pyplot.subplots(figsize=(10.0, 5.2), dpi=180)
    figure.patch.set_facecolor(style.SURFACE)
    style.parity_rule(axis, "horizontal")
    axis.plot(
        widths, speedups, color=style.DECODE, linewidth=2.0, marker="o", markersize=6,
        markeredgecolor=style.SURFACE, markeredgewidth=1.2, zorder=3, label="meTile INT4",
    )

    for record in records:
        if not record["model"]:
            continue
        axis.annotate(
            f"{record['model']}\n{style.multiplier(record['speedup'])}",
            (record["output_features"], record["speedup"]),
            textcoords="offset points",
            xytext=(0, 14 if record["speedup"] > 1.5 else 16),
            ha="center",
            fontsize=8.5,
            color=style.INK_SOFT,
        )

    ticker = __import__("matplotlib").ticker
    axis.set_xscale("log", base=2)
    axis.set_xticks(widths)
    axis.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: style.multiplier(value))
    )
    axis.set_ylim(0.85, max(speedups) + 0.55)
    axis.set_xlabel(
        "output width of the projection", fontsize=9.5, color=style.INK_SOFT
    )
    axis.set_ylabel("speedup vs native MLX", fontsize=9.5, color=style.INK_SOFT)
    style.frame(axis, grid_axis="y")
    axis.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=style.INK_SOFT)

    style.headings(
        figure,
        "The 4-bit prefill win depends on how wide the projection is",
        "MLX changes kernel above 2048 wide.",
        _footer(payload),
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, output)
    pyplot.close(figure)


def render_batch(payload, output):
    pyplot = style.matplotlib_pyplot()
    records = payload["batch_sweep"]

    def series(format_name, field):
        rows = sorted(
            (record for record in records if record["format"] == format_name),
            key=lambda record: record["rows"],
        )
        return (
            [record["rows"] for record in rows],
            [record[field] for record in rows if record[field] is not None],
        )

    figure, axis = pyplot.subplots(figsize=(10.0, 5.4), dpi=180)
    figure.patch.set_facecolor(style.SURFACE)

    axis.axhline(
        STREAMING_CEILING, color=style.RULE, linewidth=1.3, linestyle=(0, (5, 4)), zorder=1,
        label=f"most this machine can move ({STREAMING_CEILING:.0f} GB/s)",
    )

    def tracks_mlx(format_name, tolerance=0.08):
        """True when meTile's line would sit on MLX's, because it defers to it."""
        rows = [record for record in records if record["format"] == format_name]
        if any(record["metile_bandwidth"] is None for record in rows):
            return False
        return all(
            abs(record["metile_bandwidth"] - record["mlx_bandwidth"])
            <= tolerance * record["mlx_bandwidth"]
            for record in rows
        )

    # For the quantized formats meTile has no kernel of its own and calls MLX's, so one
    # line describes both backends. That is not two implementations tying, it is the same
    # code measured twice, so the label says "both" rather than drawing a duplicate.
    lines = [("bf16", "metile_bandwidth", style.DECODE, "BF16, meTile"),
             ("bf16", "mlx_bandwidth", style.PREFILL, "BF16, MLX")]
    for format_name, colour in (("int8", style.ACCENT), ("int4", style.FOURTH)):
        shared = tracks_mlx(format_name)
        label = f"{format_name.upper()}, {'both' if shared else 'MLX'}"
        lines.append((format_name, "mlx_bandwidth", colour, label))
    lines = tuple(lines)
    endpoints = []
    for format_name, field, colour, label in lines:
        rows, values = series(format_name, field)
        if not values:
            continue
        axis.plot(
            rows[: len(values)], values, color=colour, linewidth=2.0, marker="o",
            markersize=5.5, markeredgecolor=style.SURFACE, markeredgewidth=1.1,
            zorder=3, label=label,
        )
        endpoints.append((rows[len(values) - 1], values[-1], label, colour))

    # Direct labels: two palette slots sit under 3:1 on a light surface, so identity must
    # not rest on colour alone.
    for x, y, label, _ in endpoints:
        axis.annotate(
            label, (x, y), textcoords="offset points", xytext=(9, 0), va="center",
            fontsize=8.5, color=style.INK_SOFT, annotation_clip=False,
        )

    ticker = __import__("matplotlib").ticker
    axis.set_xscale("log", base=2)
    axis.set_xticks([record["rows"] for record in records if record["format"] == "bf16"])
    axis.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_xlim(0.9, max(record["rows"] for record in records) * 3.4)
    axis.set_ylim(0, STREAMING_CEILING * 1.18)
    axis.set_xlabel(
        "rows per dispatch  (1 = one token, 2 to 32 = speculative decoding and batching)",
        fontsize=9.5, color=style.INK_SOFT,
    )
    axis.set_ylabel("weight bandwidth achieved, GB/s", fontsize=9.5, color=style.INK_SOFT)
    style.frame(axis, grid_axis="y")
    axis.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=style.INK_SOFT)

    style.headings(
        figure,
        "Batching should be free, and for quantized weights it is not",
        "Weights are read once, so these should stay flat.",
        _footer(payload),
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, output)
    pyplot.close(figure)


def main():
    arguments = _arguments()
    payload = json.loads(arguments.input.read_text())
    if payload.get("scope") != "shape_sensitivity":
        raise ValueError("input is not a shape-sensitivity result")
    render_width(payload, arguments.width_output)
    render_batch(payload, arguments.batch_output)


if __name__ == "__main__":
    main()
