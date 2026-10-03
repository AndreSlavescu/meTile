import json
import xml.etree.ElementTree as element_tree
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.plots.render_compiler_results import DEFAULT_INPUTS, PANELS, chart_data, render
from benchmarks.plots.render_model_speedups import _MATCHED, _MIXED, _collect
from benchmarks.plots.render_shape_sensitivity import _bandwidth_series

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
    document = element_tree.fromstring(vector)
    viewbox = [float(value) for value in document.attrib["viewBox"].split()]
    assert viewbox[3] > viewbox[2]
    labels = [element.text for element in document.iter("{http://www.w3.org/2000/svg}text")]
    assert labels.count("FP16    1 x 1009") == 3
    assert labels.count("1.10x target *") == 1
