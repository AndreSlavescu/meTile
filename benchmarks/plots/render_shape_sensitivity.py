"""Render recorded width and effective weight-bandwidth sweeps without new measurements."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.plots import chartstyle as style


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
        f"MLX {software.get('mlx', '')} · same weight representation · "
        f"{payload.get('rounds')} rounds"
    )


def _bandwidth_series(records, format_name, field):
    selected = sorted(
        (
            record
            for record in records
            if record["format"] == format_name and record[field] is not None
        ),
        key=lambda record: record["rows"],
    )
    return [record["rows"] for record in selected], [record[field] for record in selected]


def render_width(payload, output):
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import FuncFormatter

    records = sorted(payload["width_sweep"], key=lambda record: record["output_features"])
    widths = [record["output_features"] for record in records]
    speedups = [record["speedup"] for record in records]
    figure, axis = pyplot.subplots(figsize=(style.WIDTH, 5.8), dpi=style.DPI)
    style.parity_rule(axis, "horizontal")
    axis.plot(
        widths,
        speedups,
        color=style.DECODE,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=style.SURFACE,
        markeredgewidth=1.2,
        zorder=3,
        label="meTile INT4",
    )
    for record in records:
        if record["model"]:
            axis.annotate(
                f"{record['model']}\n{style.multiplier(record['speedup'])}",
                (record["output_features"], record["speedup"]),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                fontsize=10.5,
                color=style.INK_SOFT,
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(widths)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: style.multiplier(value)))
    axis.set_ylim(0.8, max(speedups) + 0.6)
    axis.set_xlabel("Projection output width · logarithmic scale (base 2)")
    axis.set_ylabel("Speedup vs native MLX")
    style.frame(axis, grid_axis="y")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        fontsize=10,
        borderaxespad=0,
    )
    style.headings(
        figure,
        "INT4 prefill: width changes the result",
        f"{payload['prefill_rows']} input rows · reduction width {payload['reduction']} · INT4 group 64\n"
        "Same weight representation · higher is faster",
        _footer(payload),
    )
    figure.subplots_adjust(left=0.105, right=0.975, top=0.71, bottom=0.19)
    style.save(figure, output)
    pyplot.close(figure)


def render_batch(payload, output):
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import FuncFormatter

    records = payload["batch_sweep"]
    figure, axes = pyplot.subplots(3, 1, figsize=(style.WIDTH, 10.8), dpi=style.DPI, sharey=True)
    maximum = max(
        record[field]
        for record in records
        for field in ("metile_bandwidth", "mlx_bandwidth")
        if record[field] is not None
    )
    for axis, format_name in zip(axes, ("bf16", "int4", "int8")):
        for field, color, label, marker, dash in (
            ("metile_bandwidth", style.DECODE, "meTile", "o", "solid"),
            ("mlx_bandwidth", style.PREFILL, "Native MLX", "s", (0, (4, 3))),
        ):
            rows, values = _bandwidth_series(records, format_name, field)
            axis.plot(
                rows,
                values,
                color=color,
                label=label,
                marker=marker,
                linestyle=dash,
                linewidth=2,
                markersize=8 if field == "metile_bandwidth" else 4.5,
                markerfacecolor=style.SURFACE if field == "metile_bandwidth" else color,
                markeredgecolor=color,
                markeredgewidth=1.5 if field == "metile_bandwidth" else 0.7,
                zorder=3,
            )
        axis.set_title(format_name.upper(), loc="left", fontsize=12, fontweight="bold", pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({record["rows"] for record in records}))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        axis.set_ylim(0, maximum * 1.12)
        axis.set_yticks([0, 40, 80, 120])
        axis.set_xlabel("Rows per dispatch · logarithmic scale (base 2)", fontsize=10.5)
        axis.set_ylabel("Effective GB/s", fontsize=10.5)
        style.frame(axis, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.89),
        ncol=2,
        handlelength=2.5,
    )
    style.headings(
        figure,
        "Weight bandwidth, across batch sizes",
        "Weight bytes / measured latency — not a hardware traffic counter\n"
        "Same y-axis scale in every panel; both backend series remain visible",
        _footer(payload),
    )
    figure.subplots_adjust(left=0.105, right=0.975, top=0.825, bottom=0.105, hspace=0.68)
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
