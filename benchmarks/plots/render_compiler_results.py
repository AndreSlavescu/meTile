"""Plot every frozen-policy RMSNorm holdout case from existing JSON, without a GPU."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.plots import chartstyle as style

DEFAULT_INPUTS = (
    Path("benchmarks/results/m5-rmsnorm-tiling-heldout.json"),
    Path("benchmarks/results/m5-rmsnorm-tiling-heldout-repeat.json"),
)
PANELS = (
    ("frozen_register4", "gpu_us", "GPU time vs prior register4"),
    ("frozen_register4", "wall_us", "Wall time vs prior register4"),
    ("mlx_matched", "wall_us", "Wall time vs matched MLX graph"),
)


def chart_data(reports):
    """Preserve source ratios, order, case identity, and all independently recorded runs."""
    if not reports:
        raise ValueError("at least one heldout report is required")
    expected = {
        f"{dtype}_{batches}x{width}"
        for dtype in ("float16", "float32")
        for width in (1009, 1024)
        for batches in (1, 32, 256)
    }
    first = reports[0]
    identities = (
        "device",
        "mlx_version",
        "manifest_sha256",
        "baseline_export_sha256",
        "compiler_implementation_sha256",
        "gate_policy",
        "precision_comparison",
    )
    rows = []
    for run_index, report in enumerate(reports):
        if report.get("kind") != "register_tiling_validate":
            raise ValueError("charts require heldout validation reports, not tuning results")
        if any(report.get(key) != first.get(key) for key in identities):
            raise ValueError(
                "runs must share the device, MLX version, baseline, and frozen selection"
            )
        precision = report["precision_comparison"]
        if precision["class"] != "same_storage_precision" or precision["relaxed_precision"]:
            raise ValueError("charts require the matched FP32-compute precision policy")
        cases = report["cases"]
        if len(cases) != len(expected) or {case["name"] for case in cases} != expected:
            raise ValueError("every report must contain the complete 12-case heldout matrix")
        baseline = report["frozen_baseline"]
        if baseline["kind"] != "frozen_static_width_register4_baseline":
            raise ValueError("baseline must be the frozen static-width register4 shader")
        for case in cases:
            if (
                case["status"] != "measured"
                or not case["validated_before_timing"]
                or not {"candidate", "frozen_register4", "mlx_matched"}.issubset(
                    case["correctness"]
                )
                or not all(check["passed"] for check in case["correctness"].values())
            ):
                raise ValueError("every case must pass correctness before timing")
            if case["variant"] != report["frozen_manifest"]["selection"]["selected"]:
                raise ValueError("every case must use the frozen policy")
            if case["name"] != f"{case['dtype']}_{case['batches']}x{case['width']}":
                raise ValueError("case name and dimensions disagree")
            for comparator, metric, _ in PANELS:
                summary = case["summary"][comparator][metric]
                for field in ("speedup", "candidate", "control"):
                    value = summary[field]
                    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                        raise ValueError("timings and ratios must be finite and positive")
                rows.append(
                    {
                        "run": run_index,
                        "name": case["name"],
                        "dtype": case["dtype"],
                        "batches": case["batches"],
                        "width": case["width"],
                        "comparator": comparator,
                        "metric": metric,
                        "speedup": summary["speedup"],
                        "candidate_us": summary["candidate"],
                        "control_us": summary["control"],
                    }
                )
    return rows


def render(reports, output):
    rows = chart_data(reports)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.ticker import FuncFormatter

    figure, axes = pyplot.subplots(3, 1, figsize=(style.WIDTH, 16.8), dpi=style.DPI)
    names = [case["name"] for case in reports[0]["cases"]]
    labels = [
        f"{'FP16' if case['dtype'] == 'float16' else 'FP32'}  {case['batches']:>3} x {case['width']}"
        + (" *" if case["width"] == 1024 and case["batches"] in (32, 256) else "")
        for case in reports[0]["cases"]
    ]
    maximum = max(1.15, max(row["speedup"] for row in rows) + 0.015)
    minimum = min(0.98, min(row["speedup"] for row in rows) - 0.015)
    for axis, (comparator, metric, title) in zip(axes, PANELS):
        style.parity_rule(axis, reference="comparator")
        series = []
        for run_index, _ in enumerate(reports):
            selected = {
                row["name"]: row
                for row in rows
                if row["run"] == run_index
                and row["comparator"] == comparator
                and row["metric"] == metric
            }
            values = [selected[name]["speedup"] for name in names]
            color = style.SERIES[run_index % len(style.SERIES)]
            series.append((values, color, f"Heldout run {run_index + 1}"))
        columns = tuple(
            0.85 + 0.12 * index / max(1, len(reports) - 1) for index in range(len(reports))
        )
        style.comparison_rows(axis, labels, series, columns=columns, digits=3)
        style.value_headers(
            axis,
            [
                (values, color, f"Run {index + 1}")
                for index, (values, color, _) in enumerate(series)
            ],
            columns=columns,
            label="STORAGE  /  ROWS x WIDTH",
        )
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=38)
        axis.text(
            1.0,
            1.025,
            "1.00x parity",
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontsize=9.5,
            color=style.INK_MUTED,
        )
        axis.set_xlim(minimum, maximum)
        axis.set_xticks([1.0, 1.05, 1.10, 1.15])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: style.multiplier(value)))
        axis.set_xlabel("Comparator time / candidate time →", fontsize=10.5)
        axis.set_ylim(len(names) - 0.5, -0.7)
    target = reports[0]["gate_policy"]["aligned_throughput_gpu_speedup_minimum"]
    axes[0].axvline(target, color=style.ACCENT, linewidth=1.2, linestyle="dotted")
    axes[0].text(
        target,
        1.025,
        f"{target:.2f}x target *",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        fontsize=9.5,
        color=style.ACCENT,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles[1:],
        legend_labels[1:],
        loc="upper left",
        bbox_to_anchor=(0.035, 0.923),
        ncol=len(reports),
        frameon=False,
        fontsize=10.5,
    )
    passed = all(report["promotion_gate"]["passed"] for report in reports)
    style.headings(
        figure,
        "RMSNorm register tiling",
        (
            "Promotion gate passed"
            if passed
            else "Promotion gate not met · default remains unpromoted"
        )
        + "\nAll 12 heldout cases · frozen static-N register4 baseline\n"
        + f"{len(reports)} fresh-process runs, not confidence intervals",
        f"{reports[0]['device']} · MLX {reports[0]['mlx_version']} · FP32 arithmetic, final output cast\n"
        "Matched MLX means the FP32 graph, not mx.fast.rms_norm · higher is faster",
    )
    figure.text(
        0.045,
        0.050,
        "* Target applies to aligned 32- and 256-row GPU cases",
        ha="left",
        fontsize=10,
        color=style.INK_SOFT,
    )
    figure.subplots_adjust(left=0.28, right=0.74, top=0.865, bottom=0.095, hspace=0.39)
    style.save(figure, output)
    pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/compiler-rmsnorm-heldout.png")
    )
    arguments = parser.parse_args()
    render([json.loads(path.read_text()) for path in arguments.inputs], arguments.output)


if __name__ == "__main__":
    main()
