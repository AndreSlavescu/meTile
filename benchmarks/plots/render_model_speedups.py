"""Render recorded model speedups, separating weight-representation categories.

The default plots only same-representation results. Pass --include-mixed to show
compression-assisted results in their own labeled section as well.
"""

import argparse
import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _root)

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_mlx_lm_results import _precision_class, _validate_suite

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
        _validate_suite(payload)
        for model in payload["models"]:
            precision_class = _precision_class(model)
            section = _MATCHED if precision_class == "same_precision" else _MIXED
            if section == _MIXED and not include_mixed:
                continue
            if precision_class == "mixed_precision_mxfp8_decode":
                section = "mixed precision  ·  meTile MXFP8 decode vs source weights"
            elif precision_class == "mixed_precision_hybrid_decode":
                section = "mixed precision  ·  multiple decode weight formats"
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
        software = payload.get("software") or payload["models"][0].get("software", {})
        context.setdefault("chip", hardware.get("chip", "Apple silicon"))
        context.setdefault("memory", hardware.get("memory", ""))
        context.setdefault("mlx", software.get("mlx", "unknown"))
    if not rows:
        raise SystemExit("no benchmark results found")
    ordered = [row for row in rows if row["section"] == _MATCHED]
    ordered += sorted(
        (row for row in rows if row["section"] != _MATCHED), key=lambda row: row["section"]
    )
    return ordered, context


def _render(rows, context, output, title, subtitle, series):
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.transforms import blended_transform_factory

    sections = list(dict.fromkeys(row["section"] for row in rows))
    height = 0.57 * len(rows) + 0.75 * len(sections) + 2.6
    figure, axis = pyplot.subplots(figsize=(style.WIDTH, height), dpi=style.DPI)
    slots, labels, heading_slots = [], [], []
    current_section, cursor = None, 0.0
    for row in rows:
        if row["section"] != current_section:
            if current_section is not None:
                cursor += 0.65
            heading_slots.append((cursor, row["section"]))
            cursor += 1.05
            current_section = row["section"]
        slots.append(cursor)
        labels.append(row["label"].replace("  (native fallback)", "\n(native fallback)"))
        cursor += 1.0

    style.parity_rule(axis, "vertical")
    plotted = [([row[key] for row in rows], colour, name) for key, colour, name in series]
    style.comparison_rows(axis, labels, plotted, slots)
    style.value_headers(axis, plotted)
    lowest = min(min(row[key] for row in rows) for key, _, _ in series)
    highest = max(max(row[key] for row in rows) for key, _, _ in series)
    axis.set_xlim(min(0.93, lowest - 0.05), highest + 0.06)
    axis.set_ylim(cursor - 0.4, -0.5)
    axis.xaxis.set_major_locator(MultipleLocator(0.1 if highest - lowest < 0.5 else 0.25))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: style.multiplier(value)))
    axis.set_xlabel("Speedup vs native MLX →", fontsize=10.5)
    transform = blended_transform_factory(figure.transFigure, axis.transData)
    section_labels = {
        _MATCHED: "Same representation · identical weights and format",
        _MIXED: "Mixed precision · meTile affine INT8 decode vs MLX BF16",
    }
    for slot, heading in heading_slots:
        axis.text(
            0.045,
            slot,
            section_labels.get(heading, heading),
            transform=transform,
            fontsize=10,
            color=style.INK_SOFT,
            fontweight="bold",
            va="center",
            bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 4},
        )

    style.headings(
        figure,
        title,
        subtitle,
        f"{context['chip']} · {context['memory']} · MLX {context['mlx']} · 128 prompt tokens\n"
        "Median of paired alternating trials · 1.00x = parity; higher is faster",
    )
    figure.subplots_adjust(left=0.34, right=0.73, top=1 - 1.5 / height, bottom=1.25 / height)
    style.save(figure, output)
    pyplot.close(figure)


def main():
    arguments = _arguments()
    rows, context = _collect(arguments.inputs, arguments.include_mixed)
    _render(
        rows,
        context,
        arguments.throughput_output,
        "Model throughput",
        "Same-weight and compression-assisted results are separate comparisons."
        if arguments.include_mixed
        else "Same weight representation · native fallbacks remain in the results.",
        (("decode", style.DECODE, "Decode"), ("prefill", style.PREFILL, "Prefill")),
    )
    _render(
        rows,
        context,
        arguments.latency_output,
        "Model latency",
        "Time to first token and complete generation, from the same paired runs.",
        (
            ("ttft", style.DECODE, "First token"),
            ("end_to_end", style.PREFILL, "End to end"),
        ),
    )


if __name__ == "__main__":
    main()
