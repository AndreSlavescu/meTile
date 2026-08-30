"""Render reproducible PNG bar charts from an MLX-LM benchmark suite result."""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

from benchmarks import chartstyle as style

MLX_COLOR = "#ffb000"
METILE_COLOR = "#3f7ee8"
_CHART_METRICS = (
    "mlx_decode_tokens_per_second",
    "metile_decode_tokens_per_second",
    "mlx_prefill_tokens_per_second",
    "metile_prefill_tokens_per_second",
    "mlx_time_to_first_token_seconds",
    "metile_time_to_first_token_seconds",
    "mlx_elapsed_seconds",
    "metile_elapsed_seconds",
)
_WORKLOAD_KEYS = (
    "prompt_tokens",
    "generation_tokens",
    "trials",
    "prefill_step_size",
    "delay_seconds",
    "plan_decode_steps",
    "plan_trials",
    "confirmation_trials",
    "seed",
)
_COMPRESSION_FEATURES = (
    "compressed_down",
    "compressed_gate_up",
    "compressed_vocab",
    "compressed_attention",
)
_PRECISION_CLASSES = {
    "same_precision",
    "mixed_precision_affine_int8_decode",
    "mixed_precision_mxfp8_decode",
    "mixed_precision_hybrid_decode",
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("docs/_static/mlx-model-throughput.png"),
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        default=Path("docs/_static/mlx-model-latency.png"),
    )
    return parser.parse_args()


def _model_label(model):
    name = model.rsplit("/", 1)[-1]
    precision = "unknown"
    for suffix, label in (
        ("-Instruct-4bit", "4-bit"),
        ("-4bit", "4-bit"),
        ("-Instruct-bf16", "BF16"),
        ("-bf16", "BF16"),
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            precision = label
            break
    family, size = name.rsplit("-", 1)
    family = (
        family.replace("Llama-", "Llama ").replace("Qwen2.5", "Qwen 2.5").replace("Qwen3", "Qwen 3")
    )
    return f"{family}\n{size} {precision}"


def _selected_feature(result, feature):
    selected_plan = result.get("selected_plan")
    if isinstance(selected_plan, dict):
        return bool(selected_plan.get(feature, False))
    return result.get(feature) is not None


def _inferred_precision_class(result):
    selected = {feature for feature in _COMPRESSION_FEATURES if _selected_feature(result, feature)}
    if not selected:
        return "same_precision"
    formats = {"affine8" for feature in selected if feature != "compressed_down"}
    if "compressed_down" in selected:
        compressed_down = result.get("compressed_down")
        if isinstance(compressed_down, dict):
            formats.add(compressed_down.get("format", "unknown"))
        else:
            formats.add("unknown")
    if formats == {"affine8"}:
        return "mixed_precision_affine_int8_decode"
    if formats == {"mxfp8"}:
        return "mixed_precision_mxfp8_decode"
    return "mixed_precision_hybrid_decode"


def _precision_class(result):
    comparison = result.get("precision_comparison")
    if isinstance(comparison, dict) and comparison.get("class") in _PRECISION_CLASSES:
        return comparison["class"]
    return _inferred_precision_class(result)


def _precision_labels(suite):
    classes = {_precision_class(result) for result in suite["models"]}
    if classes == {"same_precision"}:
        return "Native MLX", "MLX + meTile (same precision)", "same weight representation"
    if classes == {"mixed_precision_affine_int8_decode"}:
        return (
            "Native MLX (source precision)",
            "meTile plan (affine INT8 decode)",
            "mixed precision; not BF16-vs-BF16",
        )
    if classes == {"mixed_precision_mxfp8_decode"}:
        return (
            "Native MLX (source precision)",
            "meTile plan (MXFP8 decode)",
            "mixed precision; not same-format",
        )
    return (
        "Native MLX (source precision)",
        "meTile plan (mixed decode precision)",
        "mixed precision classes; not same-format",
    )


def _suite_context(suite):
    first = suite["models"][0]
    workload = first["workload"]
    hardware = first.get("hardware", {})
    software = first.get("software", {})
    chip = hardware.get("chip") or hardware.get("processor") or hardware.get("machine", "unknown")
    comparison_modes = {model.get("comparison_mode", "alternating") for model in suite["models"]}
    if comparison_modes == {"shared_native_fallback"}:
        trial_kind = "native-fallback trials"
    elif comparison_modes == {"alternating"}:
        trial_kind = "alternating trials"
    else:
        trial_kind = "guarded paired/fallback trials"
    compressed_formats = {
        result["compressed_down"]["format"]
        for result in suite["models"]
        if _selected_feature(result, "compressed_down")
        and result.get("compressed_down") is not None
    }
    has_compressed_gate_up = any(
        _selected_feature(result, "compressed_gate_up") for result in suite["models"]
    )
    has_compressed_vocab = any(
        _selected_feature(result, "compressed_vocab") for result in suite["models"]
    )
    has_compressed_attention = any(
        _selected_feature(result, "compressed_attention") for result in suite["models"]
    )
    strategy = ""
    if compressed_formats == {"affine8"} and has_compressed_gate_up:
        families = ["MLP"]
        if has_compressed_attention:
            families.append("attention")
        if has_compressed_vocab:
            families.append("vocab")
        strategy = " · composable affine INT8 " + " + ".join(families)
    elif compressed_formats == {"affine8"}:
        strategy = " · guarded affine INT8 down"
    elif has_compressed_gate_up:
        strategy = " · guarded affine INT8 gate/up"
    elif has_compressed_attention:
        strategy = " · guarded affine INT8 attention"
    elif compressed_formats:
        strategy = " · guarded " + "/".join(sorted(compressed_formats)) + " down"
    model_delay = suite.get("model_delay_seconds", 0)
    if isinstance(model_delay, (int, float)) and model_delay > 0:
        strategy += f" · {model_delay:g}s model cooldown"
    _, _, precision_note = _precision_labels(suite)
    subtitle = (
        f"{chip} · {workload['prompt_tokens']} prompt tokens · "
        f"{workload['generation_tokens']} generated · "
        f"median of {workload['trials']} {trial_kind}{strategy} · {precision_note}"
    )
    footer = (
        f"MLX {software.get('mlx', 'unknown')} · MLX-LM {software.get('mlx_lm', 'unknown')} · "
        f"seed {workload.get('seed', 0)} · rev {first.get('revision', 'unknown')[:7]}"
    )
    return subtitle, footer


def _validate_suite(suite):
    models = suite.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("benchmark suite contains no model results")

    reference = models[0]
    reference_workload = reference.get("workload", {})
    context = {
        "hardware": reference.get("hardware", {}),
        "software": reference.get("software", {}),
        "workload": {key: reference_workload.get(key) for key in _WORKLOAD_KEYS},
    }
    comparison_modes = set()
    precision_classes = set()
    model_names = set()
    for result in models:
        model = result.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("every benchmark result requires a model name")
        if model in model_names:
            raise ValueError(f"duplicate benchmark model: {model}")
        model_names.add(model)

        workload = result.get("workload", {})
        result_context = {
            "hardware": result.get("hardware", {}),
            "software": result.get("software", {}),
            "workload": {key: workload.get(key) for key in _WORKLOAD_KEYS},
        }
        if result_context != context:
            raise ValueError("all charted models must share hardware, software, and workload")
        comparison_modes.add(result.get("comparison_mode", "alternating"))
        precision_class = _precision_class(result)
        precision_classes.add(precision_class)
        if result.get("schema_version", 0) >= 19:
            comparison = result.get("precision_comparison")
            if (
                not isinstance(comparison, dict)
                or comparison.get("class") not in _PRECISION_CLASSES
            ):
                raise ValueError("schema 19 benchmark requires a recognized precision comparison")
            if precision_class != _inferred_precision_class(result):
                raise ValueError("benchmark precision comparison disagrees with selected plan")
            same_weight_representation = comparison.get("same_weight_representation")
            if not isinstance(same_weight_representation, bool) or (
                same_weight_representation != (precision_class == "same_precision")
            ):
                raise ValueError("benchmark precision comparison has inconsistent weight metadata")
            optimized_weights = comparison.get("optimized_decode_weights")
            if (
                not isinstance(optimized_weights, list)
                or not optimized_weights
                or any(not isinstance(value, str) or not value for value in optimized_weights)
            ):
                raise ValueError("benchmark precision comparison requires decode weight formats")
            for key in ("baseline_weights", "prefill_weights"):
                value = comparison.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"benchmark precision comparison requires {key}")
            if comparison["prefill_weights"] != comparison["baseline_weights"]:
                raise ValueError("benchmark prefill and baseline weight formats must match")
            if precision_class == "same_precision" and optimized_weights != [
                comparison["baseline_weights"]
            ]:
                raise ValueError("same-precision benchmark must preserve native decode weights")
            if comparison.get("native_weights_preserved") is not True:
                raise ValueError("benchmark must record whether native weights are preserved")

        medians = result.get("medians", {})
        for metric in _CHART_METRICS:
            value = medians.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"benchmark metric {metric!r} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"benchmark metric {metric!r} must be finite and positive")

    if not comparison_modes <= {"alternating", "shared_native_fallback"}:
        raise ValueError("benchmark comparison mode is not recognized")
    if not precision_classes <= _PRECISION_CLASSES:
        raise ValueError("benchmark precision class is not recognized")


def _chart_data(suite):
    models = suite["models"]
    return {
        "labels": [_model_label(result["model"]) for result in models],
        "mlx": [result["medians"]["mlx_decode_tokens_per_second"] for result in models],
        "metile": [result["medians"]["metile_decode_tokens_per_second"] for result in models],
        "mlx_prefill": [result["medians"]["mlx_prefill_tokens_per_second"] for result in models],
        "metile_prefill": [
            result["medians"]["metile_prefill_tokens_per_second"] for result in models
        ],
        "mlx_ttft_ms": [
            result["medians"]["mlx_time_to_first_token_seconds"] * 1e3 for result in models
        ],
        "metile_ttft_ms": [
            result["medians"]["metile_time_to_first_token_seconds"] * 1e3 for result in models
        ],
        "mlx_total_seconds": [result["medians"]["mlx_elapsed_seconds"] for result in models],
        "metile_total_seconds": [result["medians"]["metile_elapsed_seconds"] for result in models],
    }


def _validate_text_layout(figure):
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [
        artist
        for artist in figure.findobj(Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    bounds = [
        (
            artist,
            artist.get_window_extent(renderer).padded(6 if artist.get_gid() == "bar-value" else 2),
        )
        for artist in texts
    ]
    canvas = figure.bbox
    for artist, box in bounds:
        if box.x0 < canvas.x0 or box.y0 < canvas.y0 or box.x1 > canvas.x1 or box.y1 > canvas.y1:
            raise RuntimeError(f"chart text leaves the canvas: {artist.get_text()!r}")
    for (left, left_box), (right, right_box) in combinations(bounds, 2):
        if left_box.overlaps(right_box):
            raise RuntimeError(f"chart text overlaps: {left.get_text()!r} and {right.get_text()!r}")


def _label_paired_bars(axis, mlx_bars, metile_bars, label_format):
    maximum = max(bar.get_height() for bar in (*mlx_bars, *metile_bars))
    labels = []
    for mlx_bar, metile_bar in zip(mlx_bars, metile_bars, strict=True):
        mlx_height = mlx_bar.get_height()
        metile_height = metile_bar.get_height()
        near_equal = abs(mlx_height - metile_height) <= maximum * 0.02
        for bar, padding in ((mlx_bar, 3), (metile_bar, 12 if near_equal else 3)):
            label = axis.annotate(
                label_format % bar.get_height(),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, padding),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )
            label.set_gid("bar-value")
            labels.append(label)
    return tuple(labels)


def _render_throughput(suite, output):
    _validate_suite(suite)
    pyplot = style.matplotlib_pyplot()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    native_label, optimized_label, _ = _precision_labels(suite)
    positions = list(range(len(data["labels"])))
    width = 0.34

    panels = (
        ("Prefill throughput", data["mlx_prefill"], data["metile_prefill"], "%.0f"),
        ("Decode throughput", data["mlx"], data["metile"], "%.1f"),
    )
    figure_width = max(12.0, 2.2 * len(data["labels"]) + 3.0)
    figure, axes = pyplot.subplots(1, 2, figsize=(figure_width, 5.8), dpi=180)
    for axis, (title, mlx_values, metile_values, label_format) in zip(axes, panels, strict=True):
        mlx_bars = axis.bar(
            [position - width / 2 for position in positions],
            mlx_values,
            width,
            label=native_label,
            color=MLX_COLOR,
        )
        metile_bars = axis.bar(
            [position + width / 2 for position in positions],
            metile_values,
            width,
            label=optimized_label,
            color=METILE_COLOR,
        )
        _label_paired_bars(axis, mlx_bars, metile_bars, label_format)
        axis.set_title(title, loc="left", fontsize=12, pad=10)
        axis.set_ylabel("Tokens / second")
        axis.set_xticks(positions, data["labels"], fontsize=8)
        axis.set_ylim(0, max(mlx_values + metile_values) * 1.18)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Throughput by model (higher is better)", x=0.07, y=0.965, ha="left", fontsize=18
    )
    figure.text(0.07, 0.88, subtitle, color="#666666", fontsize=9, va="bottom")
    figure.text(0.98, 0.035, footer, color="#777777", fontsize=8, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.965), ncol=2
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.20, wspace=0.24)
    _validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", metadata={"Software": "meTile benchmark renderer"})
    pyplot.close(figure)


def _render_latency(suite, output):
    _validate_suite(suite)
    pyplot = style.matplotlib_pyplot()
    data = _chart_data(suite)
    subtitle, footer = _suite_context(suite)
    native_label, optimized_label, _ = _precision_labels(suite)
    positions = list(range(len(data["labels"])))
    width = 0.34
    panels = (
        (
            "Time to first token",
            "Milliseconds",
            data["mlx_ttft_ms"],
            data["metile_ttft_ms"],
            "%.1f",
        ),
        (
            "End-to-end generation",
            "Seconds",
            data["mlx_total_seconds"],
            data["metile_total_seconds"],
            "%.2f",
        ),
    )

    figure_width = max(12.0, 2.2 * len(data["labels"]) + 3.0)
    figure, axes = pyplot.subplots(1, 2, figsize=(figure_width, 5.8), dpi=180)
    for axes_index, (title, ylabel, mlx_values, metile_values, label_format) in enumerate(panels):
        axis = axes[axes_index]
        mlx_bars = axis.bar(
            [position - width / 2 for position in positions],
            mlx_values,
            width,
            label=native_label,
            color=MLX_COLOR,
        )
        metile_bars = axis.bar(
            [position + width / 2 for position in positions],
            metile_values,
            width,
            label=optimized_label,
            color=METILE_COLOR,
        )
        _label_paired_bars(axis, mlx_bars, metile_bars, label_format)
        axis.set_title(title, loc="left", fontsize=12, pad=10)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, data["labels"], fontsize=8)
        axis.set_ylim(0, max(mlx_values + metile_values) * 1.18)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Latency by model (lower is better)", x=0.07, y=0.965, ha="left", fontsize=18)
    figure.text(0.07, 0.88, subtitle, color="#666666", fontsize=9, va="bottom")
    figure.text(0.98, 0.035, footer, color="#777777", fontsize=8, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.965), ncol=2
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.20, wspace=0.24)
    _validate_text_layout(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white", metadata={"Software": "meTile benchmark renderer"})
    pyplot.close(figure)


def main():
    arguments = _arguments()
    suite = json.loads(arguments.input.read_text())
    _render_throughput(suite, arguments.throughput_output)
    _render_latency(suite, arguments.latency_output)
    print(f"Wrote {arguments.throughput_output}")
    print(f"Wrote {arguments.latency_output}")


if __name__ == "__main__":
    main()
