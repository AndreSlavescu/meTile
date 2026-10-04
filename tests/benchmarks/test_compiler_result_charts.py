import json
import xml.etree.ElementTree as element_tree
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_compiler_results import DEFAULT_INPUTS, PANELS, chart_data, render
from benchmarks.plots.render_matched_matrix import FORMATS
from benchmarks.plots.render_matched_matrix import render as render_matched
from benchmarks.plots.render_model_shapes import render as render_shapes
from benchmarks.plots.render_model_speedups import _MATCHED, _MIXED, _collect
from benchmarks.plots.render_model_speedups import _render as render_models
from benchmarks.plots.render_shape_sensitivity import _bandwidth_series, render_batch, render_width

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reports():
    return [json.loads((ROOT / path).read_text()) for path in DEFAULT_INPUTS]


def test_every_heldout_point_preserves_its_source_metric_and_baseline(reports):
    rows = chart_data(reports)
    assert len(rows) == 2 * 12 * len(PANELS)
    for row in rows:
        case = next(case for case in reports[row["run"]]["cases"] if case["name"] == row["name"])
        source = case["summary"][row["comparator"]][row["metric"]]
        assert row["speedup"] == source["speedup"]
        assert row["candidate_us"] == source["candidate"]
        assert row["control_us"] == source["control"]
    assert {row["comparator"] for row in rows} == {"frozen_register4", "mlx_matched"}
    assert len({row["name"] for row in rows}) == 12
    assert all(not report["promotion_gate"]["passed"] for report in reports)


@pytest.mark.parametrize(
    "replacement",
    (
        "missing",
        "duplicate",
        "tuning",
        "wrong_baseline",
        "failed_correctness",
        "wrong_policy",
        "invalid_ratio",
    ),
)
def test_renderer_rejects_incomplete_or_incompatible_evidence(reports, replacement):
    report = deepcopy(reports[0])
    if replacement == "missing":
        report["cases"].pop()
    elif replacement == "duplicate":
        report["cases"][-1] = report["cases"][0]
    elif replacement == "tuning":
        report["kind"] = "register_tiling_tune"
    elif replacement == "wrong_baseline":
        report["frozen_baseline"]["kind"] = "dynamic_loop_baseline"
    elif replacement == "failed_correctness":
        report["cases"][0]["correctness"]["mlx_matched"]["passed"] = False
    elif replacement == "wrong_policy":
        report["cases"][0]["variant"] = {"name": "register32_striped"}
    else:
        report["cases"][0]["summary"]["frozen_register4"]["gpu_us"]["speedup"] = float("nan")
    with pytest.raises(ValueError):
        chart_data([report])


def test_renderer_rejects_repeats_with_different_frozen_selections(reports):
    reports[1]["manifest_sha256"] = "different policy"
    with pytest.raises(ValueError, match="frozen selection"):
        chart_data(reports)


def test_model_categories_come_from_precision_metadata_not_suite_name(tmp_path):
    source = ROOT / "benchmarks/results/m5-mlx-lm-bf16-models.json"
    suite = json.loads(source.read_text())
    suite["suite"] = "same_precision_in_name_only"
    path = tmp_path / "models.json"
    path.write_text(json.dumps(suite))
    rows, _ = _collect([path], include_mixed=True)
    assert len(rows) == len(suite["models"])
    assert all(row["section"] == _MIXED for row in rows)
    assert [row["decode"] for row in rows] == [
        model["medians"]["decode_speedup"] for model in suite["models"]
    ]
    source = ROOT / "benchmarks/results/m5-mlx-lm-bf16-dense-qwen15.json"
    rows, _ = _collect([source])
    assert all(row["section"] == _MATCHED for row in rows)


def test_bandwidth_keeps_both_near_equal_series_and_pairs_sparse_coordinates():
    payload = json.loads((ROOT / "benchmarks/results/m5-shape-sensitivity.json").read_text())
    records = payload["batch_sweep"]
    native_rows, native_values = _bandwidth_series(records, "int8", "mlx_bandwidth")
    metile_rows, metile_values = _bandwidth_series(records, "int8", "metile_bandwidth")
    assert native_rows == metile_rows == [1, 2, 4, 8, 16, 32]
    assert native_values != metile_values
    records = deepcopy(records)
    next(record for record in records if record["format"] == "int8" and record["rows"] == 4)[
        "metile_bandwidth"
    ] = None
    rows, values = _bandwidth_series(records, "int8", "metile_bandwidth")
    assert rows == [1, 2, 8, 16, 32]
    assert values == [value for index, value in enumerate(metile_values) if index != 2]


def test_compiler_chart_exports_vector_and_raster_without_losing_labels(reports, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "rmsnorm.png"
    render(reports, output)
    vector = output.with_suffix(".svg").read_text()
    assert output.read_bytes().startswith(b"\x89PNG")
    assert "1.10x target" in vector
    assert "not confidence intervals" in vector
    assert "Wall time vs matched MLX graph" in vector
    assert "Heldout run 1" in vector and "Heldout run 2" in vector
    assert "mlx_fast" not in vector
    assert "<text" in vector
    assert "'DejaVu Sans', 'Arial', 'Helvetica', sans-serif" in vector
    assert all(line == line.rstrip() for line in vector.splitlines())
    document = element_tree.fromstring(vector)
    viewbox = [float(value) for value in document.attrib["viewBox"].split()]
    assert viewbox[3] > viewbox[2]
    labels = [element.text for element in document.iter("{http://www.w3.org/2000/svg}text")]
    assert labels.count("FP16    1 x 1009") == 3
    assert labels.count("1.10x target *") == 1


@pytest.fixture
def plotted(monkeypatch):
    pytest.importorskip("matplotlib")
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        assert figure.get_size_inches()[0] == style.WIDTH
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    return figures


def test_all_72_compiler_points_reach_the_canvas_unchanged(reports, plotted, tmp_path):
    render(reports, tmp_path / "rmsnorm.png")
    figure = plotted[0]
    for axis, (comparator, metric, _) in zip(figure.axes, PANELS, strict=True):
        assert len(axis.collections) == len(reports)
        for points, report in zip(axis.collections, reports, strict=True):
            assert points.get_offsets()[:, 0].tolist() == [
                case["summary"][comparator][metric]["speedup"] for case in report["cases"]
            ]
        target_rules = [line for line in axis.lines if line.get_color() == style.ACCENT]
        assert len(target_rules) == (metric == "gpu_us")
    assert len({axis.get_xlim() for axis in figure.axes}) == 1


@pytest.mark.parametrize("include_mixed", (False, True))
@pytest.mark.parametrize("metrics", (("decode", "prefill"), ("ttft", "end_to_end")))
def test_model_charts_draw_every_collected_ratio(include_mixed, metrics, plotted, tmp_path):
    paths = [
        ROOT / "benchmarks/results" / name
        for name in (
            "m5-mlx-lm-models.json",
            "m5-mlx-lm-bf16-dense-qwen15.json",
            "m5-mlx-lm-bf16-models.json",
        )
    ]
    rows, context = _collect(paths, include_mixed)
    series = [(key, color, key) for key, color in zip(metrics, style.SERIES)]
    render_models(
        rows, context, tmp_path / "models.png", "Model comparisons", "Recorded results", series
    )
    axis = plotted[0].axes[0]
    assert 1.0 in axis.get_xticks()
    for points, metric in zip(axis.collections, metrics, strict=True):
        assert points.get_offsets()[:, 0].tolist() == [row[metric] for row in rows]
    assert len(rows) == (12 if include_mixed else 5)
    assert any("native fallback" in label.get_text() for label in axis.texts)
    if include_mixed:
        assert any("Mixed precision" in label.get_text() for label in axis.texts)


def test_model_shape_plot_preserves_payload_and_all_three_series(plotted, tmp_path):
    payload = json.loads((ROOT / "benchmarks/results/m5-model-shape-matrix.json").read_text())
    original = deepcopy(payload)
    render_shapes(payload, tmp_path / "shapes.png")
    records = sorted(payload["models"], key=lambda record: record["hidden"])
    expected = [
        [record["prefill_down_speedup"] for record in records],
        [record["block_speedup"]["1"] for record in records],
        [record["block_speedup"]["16"] for record in records],
    ]
    for points, values in zip(plotted[0].axes[0].collections, expected, strict=True):
        assert points.get_offsets()[:, 0].tolist() == values
    assert payload == original


def test_matched_matrix_draws_every_format_and_batch(plotted, tmp_path):
    payload = json.loads(
        (ROOT / "benchmarks/results/m5-matched-representation-matrix.json").read_text()
    )
    render_matched(payload, tmp_path / "matched.png")
    axis = plotted[0].axes[0]
    for line, (format_name, _, _) in zip(axis.lines[1:], FORMATS, strict=True):
        records = sorted(
            (record for record in payload["measurements"] if record["format"] == format_name),
            key=lambda record: record["rows"],
        )
        assert line.get_xdata().tolist() == [record["rows"] for record in records]
        assert line.get_ydata().tolist() == [record["speedup"] for record in records]
    assert axis.get_xscale() == "log"


def test_shape_sweeps_preserve_every_point_and_share_zero_based_bandwidth_scale(plotted, tmp_path):
    payload = json.loads((ROOT / "benchmarks/results/m5-shape-sensitivity.json").read_text())
    render_width(payload, tmp_path / "width.png")
    records = sorted(payload["width_sweep"], key=lambda record: record["output_features"])
    line = plotted[0].axes[0].lines[1]
    assert line.get_xdata().tolist() == [record["output_features"] for record in records]
    assert line.get_ydata().tolist() == [record["speedup"] for record in records]
    render_batch(payload, tmp_path / "batch.png")
    for axis, format_name in zip(plotted[1].axes, ("bf16", "int4", "int8"), strict=True):
        for line, field in zip(axis.lines, ("metile_bandwidth", "mlx_bandwidth"), strict=True):
            rows, values = _bandwidth_series(payload["batch_sweep"], format_name, field)
            assert line.get_xdata().tolist() == rows
            assert line.get_ydata().tolist() == values
        assert axis.get_ylim()[0] == 0
    assert len({axis.get_ylim() for axis in plotted[1].axes}) == 1
    assert len({axis.get_position().x0 for axis in plotted[1].axes}) == 1


def test_export_refuses_clipped_text_before_writing_files(tmp_path):
    pytest.importorskip("matplotlib")
    pyplot = style.matplotlib_pyplot()
    figure = pyplot.figure()
    figure.text(1.2, 0.5, "clipped label")
    output = tmp_path / "clipped.svg"
    with pytest.raises(RuntimeError, match="leaves the canvas"):
        style.save(figure, output)
    assert not output.exists()
    pyplot.close(figure)
