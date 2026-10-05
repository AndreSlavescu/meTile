"""Raw trial provenance and honest density rendering, without MLX or a GPU."""

import json
import statistics
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_trial_distributions import (
    ARMS,
    DEFAULT_INPUT,
    SAMPLE_METRIC,
    chart_data,
    render,
    supports_density,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def suite():
    return json.loads((ROOT / DEFAULT_INPUT).read_text())


def test_extracts_final_trials_without_pooling_models_or_mutating_source(suite):
    original = deepcopy(suite)
    rows = chart_data(suite)
    assert [row["model"] for row in rows] == [model["model"] for model in suite["models"]]
    assert len(rows) == 4
    assert sum(len(arm["values_ms"]) for row in rows for arm in row["arms"]) == 63
    for row, model in zip(rows, suite["models"], strict=True):
        for arm in row["arms"]:
            assert arm["values_ms"] == [
                sample[SAMPLE_METRIC] * 1000 for sample in model["samples"][arm["key"]]
            ]
            assert len(arm["values_ms"]) == model["workload"]["trials"] == 9
    assert suite == original


def test_shared_native_fallback_is_one_sample_set_not_two_independent_arms(suite):
    fallback = chart_data(suite)[1]
    assert fallback["comparison_mode"] == "shared_native_fallback"
    assert len(fallback["arms"]) == 1
    assert fallback["arms"][0]["label"] == "Shared native fallback"
    assert suite["models"][1]["samples"][ARMS[0]] == suite["models"][1]["samples"][ARMS[1]]


def test_inconsistent_shared_fallback_is_rejected(suite):
    suite["models"][1]["samples"][ARMS[1]][0][SAMPLE_METRIC] += 0.01
    with pytest.raises(ValueError, match="identical sample set"):
        chart_data(suite)


@pytest.mark.parametrize("value", (None, True, "0.1", 0, -0.1, float("nan"), float("inf"), 1e308))
def test_invalid_or_overflowing_trial_times_are_rejected(suite, value):
    suite["models"][0]["samples"][ARMS[0]][0][SAMPLE_METRIC] = value
    with pytest.raises(ValueError, match="finite and positive"):
        chart_data(suite)


def test_aggregates_are_not_substituted_for_missing_trials(suite):
    del suite["models"][0]["samples"]
    with pytest.raises(ValueError, match="recorded final samples"):
        chart_data(suite)


def test_missing_trial_is_rejected_instead_of_silently_changing_sample_size(suite):
    suite["models"][0]["samples"][ARMS[0]].pop()
    with pytest.raises(ValueError, match="sample count"):
        chart_data(suite)


def test_mixed_precision_suite_cannot_be_presented_as_same_weight_distributions():
    path = ROOT / "benchmarks/results/m5-mlx-lm-bf16-models.json"
    with pytest.raises(ValueError, match="same weight representation"):
        chart_data(json.loads(path.read_text()))


def test_incompatible_workloads_cannot_be_pooled(suite):
    suite["models"][0]["workload"]["generation_tokens"] = 64
    with pytest.raises(ValueError, match="share hardware, software, and workload"):
        chart_data(suite)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ([], False),
        ([1.0], False),
        (list(range(1, 8)), False),
        (list(range(1, 9)), True),
        ([1.0] * 9, False),
        ([1.0, 2.0] * 4, False),
        ([1.0 + index * 1e-12 for index in range(9)], False),
    ),
)
def test_density_guard_is_explicit_for_sparse_and_degenerate_samples(values, expected):
    assert supports_density(values) is expected


@pytest.fixture
def plotted(monkeypatch):
    pytest.importorskip("matplotlib")
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    return figures


def test_plot_retains_every_trial_and_observed_range_with_one_common_time_scale(
    suite, plotted, tmp_path
):
    render(suite, tmp_path / "trials.png")
    figure = plotted[0]
    rows = chart_data(suite)
    assert len(figure.axes) == 4
    assert len({axis.get_xlim() for axis in figure.axes}) == 1
    for axis, row in zip(figure.axes, rows, strict=True):
        dots = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "recorded-trials"
        ]
        densities = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "observed-range-density"
        ]
        medians = [
            collection for collection in axis.collections if collection.get_gid() == "sample-median"
        ]
        assert len(dots) == len(densities) == len(medians) == len(row["arms"])
        for points, density, median, arm in zip(dots, densities, medians, row["arms"], strict=True):
            values = arm["values_ms"]
            assert points.get_offsets()[:, 0].tolist() == values
            vertices = density.get_paths()[0].vertices[:, 0]
            assert min(vertices) == min(values)
            assert max(vertices) == max(values)
            assert median.get_segments()[0][:, 0].tolist() == [statistics.median(values)] * 2


@pytest.mark.parametrize("mode", ("small", "constant"))
def test_renderer_uses_raw_dots_without_kde_for_unsupported_densities(
    suite, plotted, tmp_path, mode
):
    for model in suite["models"]:
        if mode == "small":
            model["workload"]["trials"] = 3
            model["samples"] = {key: samples[:3] for key, samples in model["samples"].items()}
        else:
            for samples in model["samples"].values():
                for sample in samples:
                    sample[SAMPLE_METRIC] = 0.1
    render(suite, tmp_path / "fallback.png")
    figure = plotted[0]
    assert not any(
        collection.get_gid() == "observed-range-density"
        for axis in figure.axes
        for collection in axis.collections
    )
    assert sum(
        len(collection.get_offsets())
        for axis in figure.axes
        for collection in axis.collections
        if collection.get_gid() == "recorded-trials"
    ) == (21 if mode == "small" else 63)


def test_exports_searchable_svg_and_png_with_sample_and_method_caveats(suite, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "trials.png"
    render(suite, output)
    assert output.read_bytes().startswith(b"\x89PNG")
    vector = output.with_suffix(".svg").read_text()
    assert "<text" in vector and "sans-serif" in vector
    assert "No confidence intervals" in vector
    assert "Scott bandwidth" in vector
    assert "common time scale" in vector
    assert "One shared sample set, drawn once" in vector
    assert "wall-clock milliseconds" in vector
    assert vector.count("n = 9") == 7
    assert "m5-mlx-lm-models.json" in vector
    assert all(line == line.rstrip() for line in vector.splitlines())
