"""Validate chart provenance using synthetic unit-test data, never benchmark results."""

import statistics
from copy import deepcopy

import pytest

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_megakernel_results import ARMS, chart_data, render


@pytest.fixture
def report():
    contexts = [0, 16]
    return {
        "schema_version": 1,
        "status": "ok",
        "model": "Qwen/Qwen3-0.6B",
        "weights": {"both_backends": "float32", "same_values": True, "quantized": False},
        "geometry": {"dtype": "float32", "num_hidden_layers": 2},
        "software": {"MLX_ENABLE_TF32": "0", "mlx": "test"},
        "hardware": {"device_name": "Test fixture"},
        "execution": {"metile_dispatches_per_forward": 1, "metile_threadgroups": 1},
        "workload": {
            "batch": 1,
            "query_tokens": 1,
            "trials": 9,
            "validation_steps_per_context": 3,
            "contexts": contexts,
        },
        "results": [
            {
                "context_tokens": context,
                "status": "ok",
                "medians": {"paired_speedup": -123},
                "correctness": [
                    {
                        "position": context + step,
                        "logits": {"passed": True, "greedy_token_equal": True},
                        "cache": {
                            "passed": True,
                            "layers_checked": 2,
                            "valid_tokens_checked": context + step + 1,
                            "keys_and_values_checked": True,
                        },
                    }
                    for step in range(3)
                ],
                "samples": [
                    {
                        "order": ["MLX", "meTile"] if index % 2 == 0 else ["meTile", "MLX"],
                        "mlx_wall_seconds": 0.002 + index * 0.0001,
                        "metile_wall_seconds": 0.25 + index * 0.01,
                    }
                    for index in range(9)
                ],
            }
            for context in contexts
        ],
    }


def test_every_raw_value_is_preserved_and_medians_are_recomputed(report):
    original = deepcopy(report)
    rows = chart_data(report)
    assert sum(len(arm["values_ms"]) for row in rows for arm in row["arms"]) == 36
    for row, result in zip(rows, report["results"], strict=True):
        assert row["context"] == result["context_tokens"]
        assert row["paired_speedup"] == statistics.median(
            sample["mlx_wall_seconds"] / sample["metile_wall_seconds"]
            for sample in result["samples"]
        )
        for arm, (key, _, _) in zip(row["arms"], ARMS, strict=True):
            assert arm["values_ms"] == [sample[key] * 1000 for sample in result["samples"]]
    assert report == original


@pytest.mark.parametrize("status", [None, "validation_failed", "validated"])
def test_failed_or_unmeasured_reports_are_rejected(report, status):
    report["status"] = status
    with pytest.raises(ValueError, match="successful"):
        chart_data(report)


@pytest.mark.parametrize("value", [None, True, "0.1", 0, -1, float("inf"), float("nan"), 1e308])
def test_invalid_raw_times_cannot_be_plotted(report, value):
    report["results"][0]["samples"][0]["mlx_wall_seconds"] = value
    with pytest.raises(ValueError, match="finite, positive"):
        chart_data(report)


def test_missing_or_extra_samples_are_rejected(report):
    report["results"][0]["samples"].pop()
    with pytest.raises(ValueError, match="sample count"):
        chart_data(report)


def test_all_contexts_must_be_present(report):
    report["results"].pop()
    with pytest.raises(ValueError, match="every requested prefix"):
        chart_data(report)


@pytest.mark.parametrize("field", ["passed", "greedy_token_equal"])
def test_failed_logits_or_changed_token_prevent_a_performance_chart(report, field):
    report["results"][0]["correctness"][1]["logits"][field] = False
    with pytest.raises(ValueError, match="must pass"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("layers_checked", 1),
        ("valid_tokens_checked", 0),
        ("keys_and_values_checked", False),
    ],
)
def test_incomplete_cache_correctness_is_rejected(report, field, value):
    report["results"][1]["correctness"][0]["cache"][field] = value
    with pytest.raises(ValueError, match="must pass"):
        chart_data(report)


def test_missing_correctness_is_rejected(report):
    del report["results"][0]["correctness"]
    with pytest.raises(ValueError, match="correctness checks"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value", [("same_values", False), ("quantized", True), ("both_backends", "bfloat16")]
)
def test_compression_or_mixed_precision_cannot_be_misrepresented(report, field, value):
    report["weights"][field] = value
    with pytest.raises(ValueError, match="identical"):
        chart_data(report)


def test_float32_with_tf32_enabled_is_rejected(report):
    report["software"]["MLX_ENABLE_TF32"] = "1"
    with pytest.raises(ValueError, match="disable MLX TF32"):
        chart_data(report)


@pytest.mark.parametrize("wide", [True, False])
def test_plot_preserves_coordinates_and_labels_common_axis(report, monkeypatch, tmp_path, wide):
    pytest.importorskip("matplotlib")
    if not wide:
        for result in report["results"]:
            for sample in result["samples"]:
                sample["metile_wall_seconds"] /= 100
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    render(report, tmp_path / "fixture.png", "unit-test-fixture.json")
    axis = figures[0].axes[0]
    assert axis.get_xscale() == ("log" if wide else "linear")
    dots = [
        collection for collection in axis.collections if collection.get_gid() == "recorded-trials"
    ]
    medians = [
        collection for collection in axis.collections if collection.get_gid() == "sample-median"
    ]
    arms = [arm for row in chart_data(report) for arm in row["arms"]]
    for points, median, arm in zip(dots, medians, arms, strict=True):
        assert points.get_offsets()[:, 0].tolist() == arm["values_ms"]
        assert median.get_segments()[0][:, 0].tolist() == [statistics.median(arm["values_ms"])] * 2
    texts = "\n".join(text.get_text() for text in [*figures[0].texts, *axis.texts])
    assert "0 prefix tokens" in texts and "16 prefix tokens" in texts
    assert "one dispatch / one threadgroup" in texts
    assert "not generation throughput" in texts and "identical FP32 weights" in texts


def test_svg_and_png_exports_include_provenance_and_no_fake_confidence_intervals(report, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "fixture.png"
    render(report, output, "unit-test-fixture.json")
    assert output.read_bytes().startswith(b"\x89PNG")
    vector = output.with_suffix(".svg").read_text()
    assert "<text" in vector
    assert "unit-test-fixture.json" in vector
    assert "No confidence intervals" in vector and "all 9 paired trials" in vector
    assert "log scale" in vector and "lower is faster" in vector
