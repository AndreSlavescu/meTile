"""Render every benchmarked model as one speedup chart, in multiplier units.

Only matched-representation runs are graphed: both sides execute identical weights in
identical formats, so every point is a kernel comparison. The BF16 capacity suite, where
meTile runs affine-INT8 decode projections against MLX BF16, is deliberately not plotted
- comparing two different weight representations is not a speedup, and drawing it beside
matched results invites exactly that misreading. Those numbers stay in the committed
JSON and in the README table. Pass --include-mixed to graph them anyway.
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import chartstyle as style

_MATCHED = "matched representation  ·  identical weights and format on both sides"
_MIXED = "mixed precision  ·  meTile affine-INT8 decode vs MLX BF16"


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            Path("benchmarks/results/m5-mlx-lm-models.json"),
            Path("benchmarks/results/m5-mlx-lm-bf16-dense-qwen15.json"),
            Path("benchmarks/results/m5-mlx-lm-bf16-models.json"),
        ],
    )
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("docs/_static/mlx-model-speedup.png"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        default=Path("docs/_static/mlx-model-latency-speedup.png"),
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Also graph the BF16 suite, where meTile runs INT8 decode against MLX BF16.",
    )
    return parser.parse_args()


def _short_name(model):
    # Keep the weight format in the label: the same model appears in more than one
    # suite, and only the format tells the reader which comparison it belongs to.
    name = model.split("/")[-1].replace("-Instruct", "")
    name = name.replace("-bf16", " BF16").replace("-4bit", " 4-bit")
    return name.replace("-", " ")


def _collect(paths, include_mixed=False):
    rows = []
    context = {}
    for path in paths:
        if not path.exists():
            print(f"skipping missing {path}")
            continue
        payload = json.loads(path.read_text())
        suite = payload.get("suite")
        # The bf16 capacity suite swaps decode projections to affine INT8 while MLX keeps
        # BF16. Everything else compares identical weights on both sides.
        section = _MIXED if suite == "bf16" else _MATCHED
        if section == _MIXED and not include_mixed:
            continue
        for model in payload["models"]:
            medians = model["medians"]
            label = _short_name(model["model"])
            if model.get("comparison_mode") == "shared_native_fallback":
                label += "  (native fallback)"
            rows.append(
                {
                    "label": label,
                    "section": section,
                    "decode": medians["decode_speedup"],
                    "prefill": medians["prefill_speedup"],
                    "ttft": medians["ttft_speedup"],
                    "end_to_end": medians["end_to_end_speedup"],
                }
            )
        hardware = payload.get("hardware") or (payload["models"][0].get("hardware", {}))
        software = payload.get("software", {})
        context.setdefault("chip", hardware.get("chip", "Apple silicon"))
        context.setdefault("memory", hardware.get("memory", ""))
        context.setdefault("mlx", software.get("mlx", "0.32.0"))
    if not rows:
        raise SystemExit("no benchmark results found")
    ordered = [row for row in rows if row["section"] == _MATCHED]
    ordered += [row for row in rows if row["section"] == _MIXED]
    return ordered, context


def _render(rows, context, output, title, subtitle, series):
    pyplot = style.matplotlib_pyplot()

    height = 0.44 * len(rows) + 0.42 * 2 + 2.6
    figure, axis = pyplot.subplots(figsize=(10.6, height), dpi=180)
    figure.patch.set_facecolor(style.SURFACE)

    # One y slot per model, plus a slot for each section heading.
    slots, ticks, tick_labels, heading_slots = [], [], [], []
    current_section, cursor = None, 0.0
    for row in rows:
        if row["section"] != current_section:
            if current_section is not None:
                cursor += 0.55
            heading_slots.append((cursor, row["section"]))
            cursor += 0.85
            current_section = row["section"]
        slots.append(cursor)
        ticks.append(cursor)
        tick_labels.append(row["label"])
        cursor += 1.0

    style.parity_rule(axis, "vertical")

    offset = 0.19
    for index, (key, colour, name) in enumerate(series):
        shift = offset if index == 0 else -offset
        positions = [slot + shift for slot in slots]
        values = [row[key] for row in rows]
        axis.scatter(
            positions and values,
            positions,
            s=52,
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
                xytext=(9 if value >= 1.0 else -9, 0),
                ha="left" if value >= 1.0 else "right",
                va="center",
                fontsize=8,
                color=style.INK_SOFT,
            )

    lowest = min(min(row[key] for row in rows) for key, _, _ in series)
    highest = max(max(row[key] for row in rows) for key, _, _ in series)
    axis.set_xlim(min(0.93, lowest - 0.10), highest + 0.16)
    axis.set_ylim(cursor - 0.45, -0.9)
    axis.set_yticks(ticks)
    axis.set_yticklabels(tick_labels, fontsize=9.5)
    axis.set_xlabel("speedup vs native MLX", fontsize=9.5, color=style.INK_SOFT)
    style.frame(axis, grid_axis="x")

    # A single group needs no heading: the subtitle already states the property.
    for slot, heading in heading_slots if len(heading_slots) > 1 else ():
        axis.annotate(
            heading,
            (0.0, slot),
            xycoords=("axes fraction", "data"),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=style.INK_MUTED,
            fontweight="bold",
        )

    # Upper right: the matched-representation rows all sit near parity, so the top of
    # the fast side is the one region no marker occupies.
    legend = axis.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=style.INK_SOFT)
    legend.set_zorder(5)

    style.headings(
        figure,
        title,
        subtitle,
        f"{context['chip']} · {context['memory']} · MLX {context['mlx']} · "
        f"128 prompt tokens · median of paired alternating trials",
    )
    figure.tight_layout(rect=style.layout_rect(figure))
    style.save(figure, output)
    pyplot.close(figure)


def main():
    arguments = _arguments()
    rows, context = _collect(arguments.inputs, arguments.include_mixed)
    _render(
        rows,
        context,
        arguments.throughput_output,
        "Decode and prefill speedup by model",
        "Same weights, same format, both sides.",
        (("decode", style.DECODE, "decode"), ("prefill", style.PREFILL, "prefill")),
    )
    _render(
        rows,
        context,
        arguments.latency_output,
        "Time-to-first-token and end-to-end speedup by model",
        "Same runs, latency side.",
        (
            ("ttft", style.DECODE, "time to first token"),
            ("end_to_end", style.PREFILL, "end to end"),
        ),
    )


if __name__ == "__main__":
    main()
