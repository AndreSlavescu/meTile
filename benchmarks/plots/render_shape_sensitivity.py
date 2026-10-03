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
    figure, axis = pyplot.subplots(figsize=(10.6, 5.4), dpi=180)
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
                fontsize=9,
                color=style.INK_SOFT,
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(widths)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: style.multiplier(value)))
    axis.set_ylim(0.85, max(speedups) + 0.6)
    axis.set_xlabel("projection output width")
    axis.set_ylabel("speedup vs native MLX · higher is faster")
    style.frame(axis, grid_axis="y")
    axis.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=style.INK_SOFT)
    style.headings(
        figure,
        "INT4 prefill speedup varies with projection width",
        f"{payload['prefill_rows']} input rows · reduction width {payload['reduction']} · INT4 group 64",
        _footer(payload),
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, output)
    pyplot.close(figure)


def render_batch(payload, output):
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import FuncFormatter

    records = payload["batch_sweep"]
    figure, axes = pyplot.subplots(1, 3, figsize=(11.8, 5.4), dpi=180, sharey=True)
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
                markersize=5,
                markeredgecolor=style.SURFACE,
                zorder=3,
            )
        axis.set_title(format_name.upper(), loc="left", fontsize=12, fontweight="bold")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({record["rows"] for record in records}))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        axis.set_ylim(0, maximum * 1.12)
        axis.set_xlabel("rows per dispatch")
        style.frame(axis, grid_axis="y")
    axes[0].set_ylabel("effective weight bandwidth (GB/s)")
    axes[-1].legend(loc="lower left", frameon=False, fontsize=9)
    style.headings(
        figure,
        "Effective weight bandwidth across batch sizes",
        "Weight bytes / measured latency · all recorded backend series · not a hardware traffic counter",
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
