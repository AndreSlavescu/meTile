import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.mlx_lm_suite import DEFAULT_BF16_MODELS, _backend_command
from benchmarks.render_mlx_lm_results import (
    _chart_data,
    _label_paired_bars,
    _model_label,
    _render_latency,
    _render_throughput,
    _suite_context,
    _validate_suite,
    _validate_text_layout,
)


def _suite():
    common = {
        "hardware": {"chip": "Apple M5", "memory": "32 GB"},
        "software": {"mlx": "0.32.0", "mlx_lm": "0.31.3"},
        "workload": {
            "prompt_tokens": 128,
            "generation_tokens": 256,
            "trials": 5,
            "prefill_step_size": 2048,
            "delay_seconds": 2.0,
            "plan_decode_steps": 8,
            "plan_trials": 5,
            "confirmation_trials": 3,
            "seed": 0,
        },
    }
    return {
        "models": [
            {
                **common,
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "medians": {
                    "mlx_decode_tokens_per_second": 100.0,
                    "metile_decode_tokens_per_second": 105.0,
                    "mlx_prefill_tokens_per_second": 1000.0,
                    "metile_prefill_tokens_per_second": 1250.0,
                    "mlx_time_to_first_token_seconds": 0.12,
                    "metile_time_to_first_token_seconds": 0.10,
                    "mlx_elapsed_seconds": 1.5,
                    "metile_elapsed_seconds": 1.4,
                    "decode_speedup": 1.05,
                    "ttft_speedup": 1.02,
                    "end_to_end_speedup": 1.03,
                },
            },
            {
                **common,
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "medians": {
                    "mlx_decode_tokens_per_second": 200.0,
                    "metile_decode_tokens_per_second": 198.0,
                    "mlx_prefill_tokens_per_second": 2000.0,
                    "metile_prefill_tokens_per_second": 2400.0,
                    "mlx_time_to_first_token_seconds": 0.09,
                    "metile_time_to_first_token_seconds": 0.08,
                    "mlx_elapsed_seconds": 0.8,
                    "metile_elapsed_seconds": 0.75,
                    "decode_speedup": 0.99,
                    "ttft_speedup": 1.04,
                    "end_to_end_speedup": 1.01,
                },
            },
        ]
    }


def test_chart_data_preserves_absolute_and_relative_results():
    data = _chart_data(_suite())

    assert data["labels"] == ["Llama 3.2\n1B 4-bit", "Qwen 2.5\n0.5B 4-bit"]
    assert data["mlx"] == [100.0, 200.0]
    assert data["metile"] == [105.0, 198.0]
    assert data["mlx_prefill"] == [1000.0, 2000.0]
    assert data["metile_prefill"] == [1250.0, 2400.0]
    assert data["mlx_ttft_ms"] == pytest.approx([120.0, 90.0])
    assert data["metile_ttft_ms"] == pytest.approx([100.0, 80.0])
    assert data["mlx_total_seconds"] == pytest.approx([1.5, 0.8])
    assert data["metile_total_seconds"] == pytest.approx([1.4, 0.75])


def test_model_label_keeps_family_and_quantization_readable():
    assert _model_label("mlx-community/Qwen2.5-1.5B-Instruct-4bit") == ("Qwen 2.5\n1.5B 4-bit")
    assert _model_label("mlx-community/Llama-3.2-3B-Instruct-bf16") == ("Llama 3.2\n3B BF16")


def test_bf16_suite_spans_dense_models_through_seven_billion_parameters():
    assert DEFAULT_BF16_MODELS[0].endswith("0.5B-Instruct-bf16")
    assert DEFAULT_BF16_MODELS[-1].endswith("7B-Instruct-bf16")
    assert len(DEFAULT_BF16_MODELS) == 6


def test_bf16_suite_disables_quantized_only_backends(tmp_path):
    arguments = SimpleNamespace(
        suite="bf16",
        prompt_tokens=128,
        generation_tokens=64,
        trials=5,
        prefill_step_size=2048,
        delay=0.1,
        seed=0,
        plan_decode_steps=8,
        plan_trials=5,
        confirmation_trials=3,
        skip_verify=False,
        disable_attention=False,
        disable_rmsnorm=False,
        disable_graph_fusion=False,
        disable_quantized_mlp=False,
        disable_affine_prefill=False,
        disable_dense_mlp=False,
        disable_model_autotune=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    assert "--disable-quantized-mlp" in command
    assert "--disable-affine-prefill" in command
    assert "--disable-dense-mlp" not in command
    assert "--disable-attention" not in command


def test_4bit_suite_disables_dense_only_backend(tmp_path):
    arguments = SimpleNamespace(
        suite="4bit",
        prompt_tokens=128,
        generation_tokens=64,
        trials=5,
        prefill_step_size=2048,
        delay=0.1,
        seed=0,
        plan_decode_steps=8,
        plan_trials=5,
        confirmation_trials=3,
        skip_verify=False,
        disable_attention=False,
        disable_rmsnorm=False,
        disable_graph_fusion=False,
        disable_quantized_mlp=False,
        disable_affine_prefill=False,
        disable_dense_mlp=False,
        disable_model_autotune=False,
    )

    command = _backend_command(arguments, "example-4bit", tmp_path / "result.json")

    assert "--disable-dense-mlp" in command
    assert "--disable-quantized-mlp" not in command


def test_suite_context_labels_shared_native_fallback_trials():
    suite = _suite()
    for model in suite["models"]:
        model["comparison_mode"] = "shared_native_fallback"

    subtitle, _ = _suite_context(suite)

    assert "median of 5 native-fallback trials" in subtitle


def test_suite_context_labels_mixed_guarded_trials():
    suite = _suite()
    suite["models"][0]["comparison_mode"] = "alternating"
    suite["models"][1]["comparison_mode"] = "shared_native_fallback"

    subtitle, _ = _suite_context(suite)

    assert "median of 5 guarded paired/fallback trials" in subtitle


def test_suite_validation_rejects_mixed_workloads():
    suite = _suite()
    suite["models"][1]["workload"] = {**suite["models"][1]["workload"], "prompt_tokens": 64}

    with pytest.raises(ValueError, match="share hardware, software, and workload"):
        _validate_suite(suite)


def test_layout_validator_rejects_overlapping_text():
    pyplot = pytest.importorskip("matplotlib.pyplot")
    figure = pyplot.figure()
    figure.text(0.5, 0.5, "left")
    figure.text(0.5, 0.5, "right")

    with pytest.raises(RuntimeError, match="chart text overlaps"):
        _validate_text_layout(figure)
    pyplot.close(figure)


def test_paired_bar_labels_stagger_near_equal_values():
    pyplot = pytest.importorskip("matplotlib.pyplot")
    figure, axis = pyplot.subplots()
    mlx_bars = axis.bar([-0.17], [4.34], 0.34)
    metile_bars = axis.bar([0.17], [4.34], 0.34)

    labels = _label_paired_bars(axis, mlx_bars, metile_bars, "%.2f")

    assert labels[0].get_position()[1] == 3
    assert labels[1].get_position()[1] == 12
    pyplot.close(figure)


@pytest.mark.parametrize("renderer", (_render_throughput, _render_latency))
def test_renderer_writes_high_resolution_png(renderer, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "chart.png"

    renderer(_suite(), output)

    payload = output.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert width >= 1600
    assert height >= 900
    assert len(payload) > 20_000


@pytest.mark.parametrize("renderer", (_render_throughput, _render_latency))
@pytest.mark.parametrize(
    "result_name",
    (
        "m5-mlx-lm-models.json",
        "m5-mlx-lm-bf16-models.json",
        "m5-mlx-lm-bf16-dense-qwen15.json",
    ),
)
def test_published_suite_renders_without_text_overlap(renderer, result_name, tmp_path):
    pytest.importorskip("matplotlib")
    root = Path(__file__).parents[1]
    suite = json.loads((root / f"benchmarks/results/{result_name}").read_text())

    renderer(suite, tmp_path / "published.png")
