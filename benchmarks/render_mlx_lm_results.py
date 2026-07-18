"""Render reproducible SVG figures from an MLX-LM benchmark suite result."""

import argparse
import json
import math
from html import escape
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("docs/_static/mlx-model-throughput.svg"),
    )
    parser.add_argument(
        "--speedup-output",
        type=Path,
        default=Path("docs/_static/mlx-model-speedups.svg"),
    )
    return parser.parse_args()


def _model_label(model):
    name = model.rsplit("/", 1)[-1]
    for suffix in ("-Instruct-4bit", "-4bit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("Llama-", "Llama ").replace("Qwen2.5-", "Qwen 2.5 ").replace("-", " ")


def _model_detail(result):
    config = result.get("model_config", {})
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    details = []
    if layers is not None:
        details.append(f"{layers} layers")
    if hidden is not None:
        details.append(f"hidden {hidden}")
    details.append("4-bit")
    return " · ".join(details)


def _suite_context(suite):
    first = suite["models"][0]
    workload = first["workload"]
    hardware = first.get("hardware", {})
    software = first.get("software", {})
    chip = hardware.get("chip") or hardware.get("processor") or hardware.get("machine", "unknown")
    memory = hardware.get("memory", "unknown memory")
    subtitle = (
        f"{workload['prompt_tokens']}-token prompt · {workload['generation_tokens']} generated "
        f"· median of {workload['trials']} alternating trials"
    )
    recorded_date = suite.get("recorded_at", first.get("recorded_at", ""))[:10]
    revision = first.get("revision", "unknown")[:7]
    footer = (
        f"{chip} · {memory} · MLX {software.get('mlx', 'unknown')} · "
        f"MLX-LM {software.get('mlx_lm', 'unknown')} · seed {workload.get('seed', 0)} · "
        f"rev {revision} · {recorded_date}"
    )
    return subtitle, footer


def _svg_prelude(width, height, title, description):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'  <title id="title">{escape(title)}</title>',
        f'  <desc id="description">{escape(description)}</desc>',
        "  <defs>",
        "    <style>",
        "      text { font-family: Arial, Helvetica, sans-serif; fill: #20242b; }",
        "      .heading { font-size: 27px; font-weight: 700; }",
        "      .subtitle { font-size: 13px; fill: #505866; }",
        "      .model { font-size: 15px; font-weight: 700; }",
        "      .detail { font-size: 11px; fill: #596273; }",
        "      .label { font-size: 12px; font-weight: 700; }",
        "      .value { font-size: 11px; fill: #384252; }",
        "      .axis { font-size: 10px; fill: #667085; }",
        "      .footer { font-size: 10px; fill: #667085; }",
        "    </style>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def _render_throughput(suite):
    models = suite["models"]
    subtitle, footer = _suite_context(suite)
    width = 1200
    height = 190 + 116 * len(models)
    plot_x = 275
    plot_width = 730
    maximum = max(
        max(
            result["medians"]["mlx_decode_tokens_per_second"],
            result["medians"]["metile_decode_tokens_per_second"],
        )
        for result in models
    )
    axis_max = max(50, math.ceil(maximum / 25) * 25)
    lines = _svg_prelude(
        width,
        height,
        "MLX-LM decode throughput across models",
        "Paired horizontal bars compare native MLX and MLX with the guarded meTile backend across four locally cached 4-bit language models.",
    )
    lines.extend(
        [
            '  <text x="40" y="48" class="heading">MLX-LM decode throughput across models</text>',
            f'  <text x="40" y="70" class="subtitle">{escape(subtitle)}</text>',
            '  <line x1="40" y1="88" x2="1160" y2="88" stroke="#b4bcc7" stroke-width="1"/>',
        ]
    )
    for index in range(5):
        value = axis_max * index / 4
        x = plot_x + plot_width * index / 4
        lines.append(
            f'  <line x1="{x:.1f}" y1="112" x2="{x:.1f}" y2="{height - 62}" stroke="#e0e4ea" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="108" text-anchor="middle" class="axis">{value:.0f}</text>'
        )
    lines.append(
        f'  <text x="{plot_x + plot_width / 2:.1f}" y="126" text-anchor="middle" class="axis">median decode tokens / second</text>'
    )
    for index, result in enumerate(models):
        y = 154 + 116 * index
        medians = result["medians"]
        baseline = medians["mlx_decode_tokens_per_second"]
        patched = medians["metile_decode_tokens_per_second"]
        baseline_width = plot_width * baseline / axis_max
        patched_width = plot_width * patched / axis_max
        lines.extend(
            [
                f'  <text x="40" y="{y + 12}" class="model">{escape(_model_label(result["model"]))}</text>',
                f'  <text x="40" y="{y + 31}" class="detail">{escape(_model_detail(result))}</text>',
                f'  <text x="260" y="{y + 16}" text-anchor="end" class="label">MLX</text>',
                f'  <rect x="{plot_x}" y="{y}" width="{baseline_width:.2f}" height="24" rx="2" fill="#dcecf8" stroke="#526071" stroke-width="1"/>',
                f'  <text x="{plot_x + baseline_width + 9:.2f}" y="{y + 16}" class="value">{baseline:.2f}</text>',
                f'  <text x="260" y="{y + 53}" text-anchor="end" class="label">MLX + meTile</text>',
                f'  <rect x="{plot_x}" y="{y + 37}" width="{patched_width:.2f}" height="24" rx="2" fill="#dcefdc" stroke="#526071" stroke-width="1"/>',
                f'  <text x="{plot_x + patched_width + 9:.2f}" y="{y + 53}" class="value">{patched:.2f}</text>',
                f'  <text x="1140" y="{y + 35}" text-anchor="end" class="label">{medians["decode_speedup"]:.3f}&#215; decode</text>',
                f'  <line x1="40" y1="{y + 82}" x2="1160" y2="{y + 82}" stroke="#edf0f3" stroke-width="1"/>',
            ]
        )
    lines.extend(
        [
            f'  <rect x="40" y="{height - 43}" width="12" height="12" fill="#dcecf8" stroke="#526071"/>',
            f'  <text x="60" y="{height - 33}" class="footer">native MLX</text>',
            f'  <rect x="142" y="{height - 43}" width="12" height="12" fill="#dcefdc" stroke="#526071"/>',
            f'  <text x="162" y="{height - 33}" class="footer">MLX + guarded meTile</text>',
            f'  <text x="1160" y="{height - 33}" text-anchor="end" class="footer">{escape(footer)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_speedups(suite):
    models = suite["models"]
    subtitle, footer = _suite_context(suite)
    width = 1200
    height = 190 + 96 * len(models)
    plot_x = 275
    plot_width = 690
    values = [1.0]
    for result in models:
        values.extend(
            [
                result["medians"]["decode_speedup"],
                result["medians"]["end_to_end_speedup"],
            ]
        )
    axis_min = math.floor((min(values) - 0.015) * 20) / 20
    axis_max = math.ceil((max(values) + 0.015) * 20) / 20
    if axis_max - axis_min < 0.1:
        axis_min -= 0.05
        axis_max += 0.05

    def position(value):
        return plot_x + plot_width * (value - axis_min) / (axis_max - axis_min)

    lines = _svg_prelude(
        width,
        height,
        "meTile speedup relative to native MLX",
        "A dot plot compares decode throughput speedup and end-to-end generation speedup for each model. The vertical reference line at one indicates parity with native MLX.",
    )
    lines.extend(
        [
            '  <text x="40" y="48" class="heading">meTile speedup relative to native MLX</text>',
            f'  <text x="40" y="70" class="subtitle">{escape(subtitle)} · native MLX = 1.000&#215;</text>',
            '  <line x1="40" y1="88" x2="1160" y2="88" stroke="#b4bcc7" stroke-width="1"/>',
        ]
    )
    tick = math.ceil(axis_min * 20) / 20
    while tick <= axis_max + 1e-9:
        x = position(tick)
        stroke = "#5b6574" if abs(tick - 1.0) < 1e-9 else "#e0e4ea"
        stroke_width = 1.6 if abs(tick - 1.0) < 1e-9 else 1
        lines.append(
            f'  <line x1="{x:.2f}" y1="116" x2="{x:.2f}" y2="{height - 64}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'  <text x="{x:.2f}" y="108" text-anchor="middle" class="axis">{tick:.2f}&#215;</text>'
        )
        tick += 0.05
    for index, result in enumerate(models):
        y = 150 + 96 * index
        medians = result["medians"]
        decode = medians["decode_speedup"]
        end_to_end = medians["end_to_end_speedup"]
        lines.extend(
            [
                f'  <text x="40" y="{y + 12}" class="model">{escape(_model_label(result["model"]))}</text>',
                f'  <text x="40" y="{y + 31}" class="detail">{escape(_model_detail(result))}</text>',
                f'  <line x1="{plot_x}" y1="{y + 10}" x2="{plot_x + plot_width}" y2="{y + 10}" stroke="#d8dde5" stroke-width="2"/>',
                f'  <circle cx="{position(decode):.2f}" cy="{y}" r="8" fill="#6fa87d" stroke="#2f5f3d" stroke-width="1.4"/>',
                f'  <circle cx="{position(end_to_end):.2f}" cy="{y + 22}" r="8" fill="#a88ac6" stroke="#5f4778" stroke-width="1.4"/>',
                f'  <text x="1000" y="{y + 4}" class="label">decode {decode:.3f}&#215;</text>',
                f'  <text x="1000" y="{y + 26}" class="label">end-to-end {end_to_end:.3f}&#215;</text>',
                f'  <line x1="40" y1="{y + 58}" x2="1160" y2="{y + 58}" stroke="#edf0f3" stroke-width="1"/>',
            ]
        )
    lines.extend(
        [
            f'  <circle cx="48" cy="{height - 37}" r="6" fill="#6fa87d" stroke="#2f5f3d"/>',
            f'  <text x="62" y="{height - 33}" class="footer">decode throughput</text>',
            f'  <circle cx="183" cy="{height - 37}" r="6" fill="#a88ac6" stroke="#5f4778"/>',
            f'  <text x="197" y="{height - 33}" class="footer">end-to-end generation</text>',
            f'  <text x="1160" y="{height - 33}" text-anchor="end" class="footer">{escape(footer)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    arguments = _arguments()
    suite = json.loads(arguments.input.read_text())
    if not suite.get("models"):
        raise ValueError("benchmark suite contains no model results")
    arguments.throughput_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.speedup_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.throughput_output.write_text(_render_throughput(suite))
    arguments.speedup_output.write_text(_render_speedups(suite))
    print(f"Wrote {arguments.throughput_output}")
    print(f"Wrote {arguments.speedup_output}")


if __name__ == "__main__":
    main()
