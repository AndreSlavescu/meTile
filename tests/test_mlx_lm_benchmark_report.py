import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.mlx_lm_backend import _precision_comparison
from benchmarks.mlx_lm_suite import DEFAULT_BF16_MODELS, _backend_command
from benchmarks.render_mlx_lm_results import (
    _chart_data,
    _label_paired_bars,
    _model_label,
    _precision_class,
    _precision_labels,
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
    assert _model_label("mlx-community/Qwen3-8B-bf16") == "Qwen 3\n8B BF16"


def test_bf16_suite_spans_dense_models_through_eight_billion_parameters():
    assert DEFAULT_BF16_MODELS[0].endswith("0.5B-Instruct-bf16")
    assert DEFAULT_BF16_MODELS[-1].endswith("Qwen3-8B-bf16")
    assert len(DEFAULT_BF16_MODELS) == 7


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


def test_bf16_suite_propagates_compressed_down_group_size(tmp_path):
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
        disable_dense_mlp=True,
        disable_model_autotune=False,
        compressed_down_format="affine8",
        compressed_down_group_size=32,
        allow_approximate_compressed_down=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    index = command.index("--compressed-down-group-size")
    assert command[index + 1] == "32"


def test_bf16_suite_defaults_to_device_group_autotuning(tmp_path):
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
        disable_dense_mlp=True,
        disable_model_autotune=False,
        compressed_down_format="affine8",
        allow_approximate_compressed_down=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    index = command.index("--compressed-down-group-size")
    assert command[index + 1] == "auto"


def test_bf16_suite_propagates_compressed_gate_up(tmp_path):
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
        disable_dense_mlp=True,
        disable_model_autotune=False,
        compressed_down_format="none",
        compressed_gate_up=True,
        compressed_gate_up_group_size=128,
        allow_approximate_compressed_down=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    assert "--compressed-gate-up" in command
    index = command.index("--compressed-gate-up-group-size")
    assert command[index + 1] == "128"


def test_bf16_suite_propagates_compressed_vocab(tmp_path):
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
        disable_dense_mlp=True,
        disable_model_autotune=False,
        compressed_down_format="none",
        compressed_gate_up=False,
        compressed_vocab=True,
        compressed_vocab_group_size=64,
        allow_approximate_compressed_down=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    assert "--compressed-vocab" in command
    index = command.index("--compressed-vocab-group-size")
    assert command[index + 1] == "64"


def test_bf16_suite_propagates_compressed_attention(tmp_path):
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
        disable_dense_mlp=True,
        disable_model_autotune=False,
        compressed_down_format="none",
        compressed_gate_up=False,
        compressed_vocab=False,
        compressed_attention=True,
        compressed_attention_group_size=128,
        allow_approximate_compressed_down=False,
    )

    command = _backend_command(arguments, DEFAULT_BF16_MODELS[0], tmp_path / "result.json")

    assert "--compressed-attention" in command
    index = command.index("--compressed-attention-group-size")
    assert command[index + 1] == "128"


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


def test_suite_context_labels_compressed_down_strategy():
    suite = _suite()
    for model in suite["models"]:
        model["compressed_down"] = {"format": "affine8"}

    subtitle, _ = _suite_context(suite)

    assert "guarded affine INT8 down" in subtitle


def test_suite_context_labels_composable_compressed_mlp_strategy():
    suite = _suite()
    for model in suite["models"]:
        model["compressed_down"] = {"format": "affine8"}
        model["compressed_gate_up"] = {"group_size": 128}

    subtitle, _ = _suite_context(suite)

    assert "composable affine INT8 MLP" in subtitle


def test_suite_context_labels_composable_vocab_strategy():
    suite = _suite()
    for model in suite["models"]:
        model["compressed_down"] = {"format": "affine8"}
        model["compressed_gate_up"] = {"group_size": 128}
        model["compressed_vocab"] = {"group_size": 64}

    subtitle, _ = _suite_context(suite)

    assert "composable affine INT8 MLP + vocab" in subtitle


def test_suite_context_labels_composable_attention_strategy():
    suite = _suite()
    for model in suite["models"]:
        model["compressed_down"] = {"format": "affine8"}
        model["compressed_gate_up"] = {"group_size": 128}
        model["compressed_vocab"] = {"group_size": 64}
        model["compressed_attention"] = {"group_size": 32}

    subtitle, _ = _suite_context(suite)

    assert "composable affine INT8 MLP + attention + vocab" in subtitle


def test_precision_comparison_records_native_weight_representation():
    from metile.integrations.mlx_lm import MLXLMPlan

    comparison = _precision_comparison(
        MLXLMPlan(False, False, False, False), None, {"torch_dtype": "bfloat16"}
    )

    assert comparison == {
        "class": "same_precision",
        "same_weight_representation": True,
        "baseline_weights": "bfloat16",
        "optimized_decode_weights": ["bfloat16"],
        "prefill_weights": "bfloat16",
        "native_weights_preserved": True,
    }


def test_precision_comparison_records_selected_decode_compression():
    from metile.integrations.mlx_lm import MLXLMPlan

    plan = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_vocab=True,
    )

    comparison = _precision_comparison(plan, SimpleNamespace(format="affine8"))

    assert comparison["class"] == "mixed_precision_affine_int8_decode"
    assert not comparison["same_weight_representation"]
    assert comparison["optimized_decode_weights"] == ["affine8"]
    assert comparison["prefill_weights"] == "model_native"
    assert comparison["native_weights_preserved"]


def test_precision_class_uses_selected_plan_not_prepared_candidates():
    result = {
        "selected_plan": {
            feature: False
            for feature in (
                "compressed_down",
                "compressed_gate_up",
                "compressed_vocab",
                "compressed_attention",
            )
        },
        "compressed_down": {"format": "affine8"},
    }

    assert _precision_class(result) == "same_precision"


def test_precision_labels_call_out_mixed_precision_comparison():
    suite = _suite()
    for result in suite["models"]:
        result["selected_plan"] = {"compressed_vocab": True}
        result["precision_comparison"] = {
            "class": "mixed_precision_affine_int8_decode",
            "same_weight_representation": False,
        }

    native, optimized, note = _precision_labels(suite)

    assert native == "Native MLX (source precision)"
    assert optimized == "meTile plan (affine INT8 decode)"
    assert note == "mixed precision; not BF16-vs-BF16"


def test_schema19_suite_requires_precision_comparison():
    suite = _suite()
    for result in suite["models"]:
        result["schema_version"] = 19
        result["selected_plan"] = {}

    with pytest.raises(ValueError, match="recognized precision comparison"):
        _validate_suite(suite)


def test_schema19_suite_rejects_precision_class_that_disagrees_with_plan():
    suite = _suite()
    for result in suite["models"]:
        result["schema_version"] = 19
        result["selected_plan"] = {"compressed_vocab": True}
        result["precision_comparison"] = {
            "class": "same_precision",
            "same_weight_representation": True,
            "baseline_weights": "model_native",
            "optimized_decode_weights": ["model_native"],
            "prefill_weights": "model_native",
            "native_weights_preserved": True,
        }

    with pytest.raises(ValueError, match="disagrees with selected plan"):
        _validate_suite(suite)


def test_schema19_suite_accepts_complete_same_precision_metadata():
    suite = _suite()
    for result in suite["models"]:
        result["schema_version"] = 19
        result["selected_plan"] = {}
        result["precision_comparison"] = {
            "class": "same_precision",
            "same_weight_representation": True,
            "baseline_weights": "model_native",
            "optimized_decode_weights": ["model_native"],
            "prefill_weights": "model_native",
            "native_weights_preserved": True,
        }

    _validate_suite(suite)


@pytest.mark.parametrize(
    ("result_name", "precision_class", "exact"),
    (
        ("m5-dense-bf16-swiglu-qwen05.json", "same_precision", True),
        ("m5-affine8-swiglu-qwen05.json", "matched_precision_affine_int8", False),
    ),
)
def test_published_primitive_results_are_matched_representation(
    result_name, precision_class, exact
):
    root = Path(__file__).parents[1]
    result = json.loads((root / f"benchmarks/results/{result_name}").read_text())

    assert result["scope"] == "isolated_primitive"
    assert result["precision_comparison"]["class"] == precision_class
    assert result["precision_comparison"]["same_weight_representation"]
    assert result["timings"]["speedup"] > 1.0
    assert result["fidelity"].get("bitwise_exact", False) is exact


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
