"""Synthetic chart fixtures test rendering contracts, never publish benchmark data."""

import statistics
from copy import deepcopy

import pytest

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_qwen3_end_to_end import ARMS, METRICS, chart_data, render


@pytest.fixture
def report():
    results = []
    for prompt_count in (8, 32):
        for output_count in (8, 16):
            prompt = list(range(1, prompt_count + 1))
            generated = list(range(40, 40 + output_count))
            checks = []
            for position, token in enumerate(prompt + generated[:-1]):
                selected = generated[max(0, position - prompt_count + 1)]
                checks.append(
                    {
                        "passed": True,
                        "position": position,
                        "input_token": token,
                        "logits": {
                            "passed": True,
                            "greedy_token_equal": True,
                            "mlx_next_token": selected,
                            "metile_next_token": selected,
                        },
                        "cache": {
                            "passed": True,
                            "layers_checked": 2,
                            "valid_tokens_checked": position + 1,
                            "keys_and_values_checked": True,
                        },
                        "gpu_greedy_selection": {"passed": True, "token": selected},
                    }
                )
            samples = []
            for trial in range(5):
                sample = {
                    "trial": trial,
                    "order": ["MLX", "meTile"] if trial % 2 == 0 else ["meTile", "MLX"],
                }
                for backend, _, _ in ARMS:
                    ttft = (prompt_count / 1000 + trial * 0.0001) * (
                        3 if backend == "meTile" else 1
                    )
                    decode = (output_count - 1) / (40 + trial) / (1.1 if backend == "meTile" else 1)
                    sample[backend] = {
                        "total_wall_seconds": ttft + decode,
                        "time_to_first_token_seconds": ttft,
                        "decode_wall_seconds": decode,
                        "actual_output_tokens": output_count,
                        "generated_token_ids": list(generated),
                    }
                samples.append(sample)
            results.append(
                {
                    "status": "ok",
                    "prompt_tokens": prompt_count,
                    "requested_output_tokens": output_count,
                    "prompt_token_ids": prompt,
                    "medians": {"paired_speedup": -123},
                    "correctness": {
                        "passed": True,
                        "checks": checks,
                        "actual_output_tokens": output_count,
                        "generated_token_ids": generated,
                    },
                    "samples": samples,
                }
            )
    return {
        "schema_version": 1,
        "status": "ok",
        "model": "Qwen/Qwen3-0.6B",
        "weights": {"both_backends": "float32", "same_values": True, "quantized": False},
        "geometry": {"dtype": "float32", "num_hidden_layers": 2, "vocab_size": 67},
        "software": {"MLX_ENABLE_TF32": "0", "mlx": "test"},
        "hardware": {"device_name": "Synthetic test fixture"},
        "execution": {
            "single_dispatch": False,
            "candidate_dispatches_per_prefill_token": 17,
            "candidate_dispatches_per_generated_token": 21,
            "candidate_prefill": "sequential tokens; test fixture",
            "native_prefill": "batched prompt except final token; test fixture",
        },
        "workload": {
            "batch": 1,
            "trials": 5,
            "capacity": 47,
            "prompt_lengths": [8, 32],
            "output_lengths": [8, 16],
        },
        "results": results,
    }


def test_raw_trials_preserved_units_converted_once_and_summaries_recomputed(report):
    original = deepcopy(report)
    rows = chart_data(report)
    assert [(row["prompt_tokens"], row["output_tokens"]) for row in rows] == [
        (8, 8),
        (8, 16),
        (32, 8),
        (32, 16),
    ]
    observations = 0
    for row, result in zip(rows, report["results"], strict=True):
        for key, _, _, source in METRICS:
            metric = row["metrics"][key]
            assert metric["paired_speedup"] == statistics.median(
                sample["MLX"][source] / sample["meTile"][source] for sample in result["samples"]
            )
            for arm, (backend, _, _) in zip(metric["arms"], ARMS, strict=True):
                expected = [
                    (result["requested_output_tokens"] - 1) / sample[backend][source]
                    if key == "decode_tokens_per_second"
                    else sample[backend][source] * 1000
                    for sample in result["samples"]
                ]
                assert arm["values"] == expected
                observations += len(arm["values"])
    assert observations == 120
    assert report == original


@pytest.mark.parametrize("status", [None, "validation_failed", "validated"])
def test_failed_or_unmeasured_reports_are_rejected(report, status):
    report["status"] = status
    with pytest.raises(ValueError, match="successful"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value", [("same_values", False), ("quantized", True), ("both_backends", "bfloat16")]
)
def test_precision_or_representation_mismatch_is_rejected(report, field, value):
    report["weights"][field] = value
    with pytest.raises(ValueError, match="identical"):
        chart_data(report)


def test_fp32_requires_explicit_tf32_disable(report):
    report["software"].pop("MLX_ENABLE_TF32")
    with pytest.raises(ValueError, match="disable MLX TF32"):
        chart_data(report)


@pytest.mark.parametrize(
    "field",
    [
        "single_dispatch",
        "candidate_dispatches_per_prefill_token",
        "candidate_dispatches_per_generated_token",
        "candidate_prefill",
        "native_prefill",
    ],
)
def test_execution_metadata_must_support_chart_claims(report, field):
    report["execution"].pop(field)
    with pytest.raises(ValueError, match="staged DSL"):
        chart_data(report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed"])
def test_workload_grid_requires_each_successful_result_once(report, mutation):
    if mutation == "missing":
        report["results"].pop()
    elif mutation == "duplicate":
        report["results"][1] = deepcopy(report["results"][0])
    else:
        report["results"][1]["status"] = "validation_failed"
    with pytest.raises(ValueError, match=r"grid|ordered result"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_lengths", [8, 8]),
        ("output_lengths", [1, 16]),
        ("capacity", 46),
        ("batch", True),
        ("trials", 2),
    ],
)
def test_invalid_workload_metadata_is_rejected(report, field, value):
    report["workload"][field] = value
    with pytest.raises(ValueError):
        chart_data(report)


def test_correctness_requires_every_prompt_and_decode_forward(report):
    report["results"][0]["correctness"]["checks"].pop()
    with pytest.raises(ValueError, match="complete prompt/decode"):
        chart_data(report)


@pytest.mark.parametrize("field,value", [("passed", False), ("position", 5), ("input_token", -1)])
def test_failed_or_wrong_position_check_is_rejected(report, field, value):
    report["results"][0]["correctness"]["checks"][2][field] = value
    with pytest.raises(ValueError, match="every forward"):
        chart_data(report)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("logits", "passed", False),
        ("logits", "greedy_token_equal", False),
        ("logits", "mlx_next_token", 0),
        ("cache", "passed", False),
        ("cache", "layers_checked", 1),
        ("cache", "valid_tokens_checked", 1),
        ("cache", "keys_and_values_checked", False),
        ("gpu_greedy_selection", "passed", False),
        ("gpu_greedy_selection", "token", 0),
    ],
)
def test_logits_full_cache_and_gpu_selection_all_must_pass(report, section, field, value):
    report["results"][0]["correctness"]["checks"][2][section][field] = value
    with pytest.raises(ValueError, match="every forward"):
        chart_data(report)


def test_changed_prompt_across_output_lengths_is_rejected(report):
    report["results"][1]["prompt_token_ids"][0] = 3
    with pytest.raises(ValueError, match="prompt must be identical"):
        chart_data(report)


@pytest.mark.parametrize("field", ["actual_output_tokens", "generated_token_ids"])
def test_timed_generation_must_match_validated_tokens_and_count(report, field):
    report["results"][0]["samples"][0]["meTile"][field] = (
        7 if field == "actual_output_tokens" else [40] * 8
    )
    with pytest.raises(ValueError, match="every timed sequence"):
        chart_data(report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate_id", "wrong_order"])
def test_raw_trial_count_identifiers_and_alternation_are_checked(report, mutation):
    samples = report["results"][0]["samples"]
    if mutation == "missing":
        samples.pop()
    elif mutation == "duplicate_id":
        samples[1]["trial"] = 0
    else:
        samples[1]["order"] = ["MLX", "meTile"]
    with pytest.raises(ValueError, match=r"sample count|identifiers|alternate"):
        chart_data(report)


@pytest.mark.parametrize("source", [metric[3] for metric in METRICS])
@pytest.mark.parametrize(
    "duration", [None, True, "0.1", 0, -1, float("inf"), float("nan"), 1e308, 1e-320]
)
def test_invalid_times_or_derived_rates_cannot_be_plotted(report, source, duration):
    report["results"][0]["samples"][0]["meTile"][source] = duration
    with pytest.raises(ValueError, match="finite positive"):
        chart_data(report)


def test_total_time_must_equal_ttft_plus_decode(report):
    report["results"][0]["samples"][0]["MLX"]["total_wall_seconds"] += 0.1
    with pytest.raises(ValueError, match="TTFT plus decode"):
        chart_data(report)


def test_plot_coordinates_common_scales_medians_and_caveats(report, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    render(report, tmp_path / "fixture.png", "unit-test-fixture.json")
    assert len(figures[0].axes) == 3
    rows = chart_data(report)
    for axis, (key, title, units, _) in zip(figures[0].axes, METRICS, strict=True):
        assert axis.get_title(loc="left") == title
        assert axis.get_xlabel() == units
        assert axis.get_xlim()[0] == 0
        assert axis.get_xscale() == "linear"
        dots = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "recorded-trials"
        ]
        medians = [
            collection for collection in axis.collections if collection.get_gid() == "sample-median"
        ]
        arms = [arm for row in rows for arm in row["metrics"][key]["arms"]]
        assert len(dots) == len(medians) == 8
        for points, marker, arm in zip(dots, medians, arms, strict=True):
            assert points.get_offsets()[:, 0].tolist() == arm["values"]
            assert marker.get_segments()[0][:, 0].tolist() == [statistics.median(arm["values"])] * 2
            assert max(arm["values"]) < axis.get_xlim()[1]
    prose = "\n".join(text.get_text() for text in figures[0].texts)
    assert "identical FP32 weights" in prose and "MLX TF32 disabled" in prose
    assert "sequential DSL tokens versus batched native MLX" in prose
    assert "No confidence intervals" in prose and "output count - 1" in prose
    assert "1.00x is parity" in prose and "above 1.00x favors meTile" in prose


def test_vector_and_raster_exports_include_provenance_without_density(report, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "synthetic-test-fixture.png"
    render(report, output, "unit-test-fixture.json")
    assert output.read_bytes().startswith(b"\x89PNG")
    vector = output.with_suffix(".svg").read_text()
    assert "<text" in vector and "unit-test-fixture.json" in vector
    assert "all 5 paired trials" in vector and "No confidence intervals" in vector
    assert "observed-range-density" not in vector
