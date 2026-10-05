"""Every published result must say what it compared, and a quantized win must show its accuracy.

A speedup is meaningless without the precision it was measured at, and the failure mode is not that
someone lies about it — it is that a number gets quoted without the label, because the label lives in
a field nobody printed. `benchmarks/results/m5-mlx-lm-bf16-models.json` reported decode speedups of
1.37x to 1.75x with no precision metadata at all, and those runs had meTile quantizing the down
projection to affine8 while MLX ran bf16. The comparison was a representation change, not a kernel
win, and nothing in the artifact said so.

The suite writer already refuses to emit an unlabelled result at schema 19, and `_validate_suite`
even rejects a label that disagrees with the recorded plan. What it could not do is police artifacts
written at older schemas, which is exactly where the unlabelled ones were. This audit covers the
directory instead of the writer, so age is no longer an exemption.

The accuracy requirement follows MLPerf's rule for quantized submissions: arbitrary reproducible
quantization is allowed, but it has to be described and it has to meet an accuracy target. A mixed
precision speedup with no accuracy evidence beside it is not a result anyone can act on.
"""

import json
import math
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results"

# Substrings that mark a comparison as running different numeric representations on the two sides.
MIXED_MARKERS = ("mixed_precision", "mixed_representation")

# Fields any of which counts as evidence that a lossy comparison was checked for accuracy.
ACCURACY_FIELDS = ("kl_divergence", "max_logit_error", "mean_logit_error", "next_token")


def _published():
    return sorted(RESULTS.glob("*.json"))


def _labels(document):
    """Every precision label in a document, whether recorded once or per model."""
    found = []
    top = document.get("precision_comparison")
    if isinstance(top, dict):
        found.append(top)
    models = document.get("models")
    entries = []
    if isinstance(models, list):
        entries = [m for m in models if isinstance(m, dict)]
    elif isinstance(models, dict):
        entries = [m for m in models.values() if isinstance(m, dict)]
    for entry in entries:
        label = entry.get("precision_comparison")
        if isinstance(label, dict):
            found.append(label)
    return found


def _accuracy_evidence(node):
    """Whether an accuracy metric appears anywhere in this record."""
    if isinstance(node, dict):
        if any(field in node for field in ACCURACY_FIELDS):
            return True
        if _structured_logits_pass(node.get("logits")):
            return True
        return any(_accuracy_evidence(value) for value in node.values())
    if isinstance(node, list):
        return any(_accuracy_evidence(value) for value in node)
    return False


def _structured_logits_pass(logits):
    if not isinstance(logits, dict):
        return False
    metrics = ("max_absolute_error", "mean_absolute_error", "rtol", "atol")
    if any(
        type(logits.get(field)) not in (int, float)
        or not math.isfinite(logits[field])
        or logits[field] < 0
        for field in metrics
    ):
        return False
    return (
        logits.get("passed") is True
        and logits.get("greedy_token_equal") is True
        and logits["mean_absolute_error"] <= logits["max_absolute_error"]
        and type(logits.get("mlx_next_token")) is int
        and type(logits.get("metile_next_token")) is int
        and logits["mlx_next_token"] >= 0
        and logits["mlx_next_token"] == logits["metile_next_token"]
    )


def _logit_records(node):
    if isinstance(node, dict):
        if "logits" in node:
            yield node["logits"]
        for value in node.values():
            yield from _logit_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from _logit_records(value)


def _require_accuracy_evidence(document, name):
    if any(label.get("class") == "lossless_weight_storage" for label in _labels(document)):
        from benchmarks.plots.render_chunked_prefill import chart_data

        chart_data(document)
        records = list(_logit_records(document))
        assert records and all(_structured_logits_pass(record) for record in records), (
            f"{name} requires passed structured logits with finite nonnegative errors and "
            "tolerances, and matching greedy token IDs at every checked step"
        )
    assert _accuracy_evidence(document), (
        f"{name} reports a representation change with no accuracy metric anywhere in it. "
        f"One of {ACCURACY_FIELDS} or passed structured logit metrics must accompany it."
    )


def test_there_are_published_results_to_audit():
    """A vacuous audit is worse than none, so fail if the directory moved."""
    assert _published(), f"no published results under {RESULTS}"


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.name)
def test_every_published_result_states_what_it_compared(path):
    """No artifact may report a speedup without saying at what precision.

    Checked over the directory rather than at write time because the unlabelled files were the ones
    written at older schema versions, which the writer's own validation never sees again.
    """
    document = json.loads(path.read_text())
    labels = _labels(document)
    assert labels, (
        f"{path.name} publishes results with no precision_comparison. A speedup without the "
        f"precision it was measured at will be quoted as a like-for-like win."
    )
    for label in labels:
        assert label.get("class"), f"{path.name} has a precision_comparison with no class"
        assert "same_weight_representation" in label, (
            f"{path.name} does not say whether both sides ran the same weight representation"
        )


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.name)
def test_a_mixed_precision_result_carries_accuracy_evidence(path):
    """Following MLPerf: quantization is allowed, but it must meet a stated accuracy target.

    Without this a representation change reads as a kernel win. The bf16 suite's 1.37x-1.75x decode
    figures come from quantizing the down projection to affine8, and the calibration fidelity that
    justifies it was already recorded -- it simply was not connected to the claim.
    """
    document = json.loads(path.read_text())
    mixed = [
        label
        for label in _labels(document)
        if any(marker in str(label.get("class", "")) for marker in MIXED_MARKERS)
        or label.get("same_weight_representation") is False
        or label.get("class") == "lossless_weight_storage"
    ]
    if not mixed:
        pytest.skip(f"{path.name} is a same-representation comparison")
    _require_accuracy_evidence(document, path.name)


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.name)
def test_a_label_agrees_with_the_plan_it_was_recorded_beside(path):
    """A label that contradicts the recorded plan is worse than a missing one.

    The plan says which lossy features ran, so it decides the class rather than merely accompanying
    it. This catches a backfilled or hand-edited label drifting from the measurement.
    """
    document = json.loads(path.read_text())
    models = document.get("models")
    entries = []
    if isinstance(models, list):
        entries = [m for m in models if isinstance(m, dict)]
    elif isinstance(models, dict):
        entries = [m for m in models.values() if isinstance(m, dict)]

    lossy_features = ("compressed_down", "compressed_gate_up", "compressed_vocab")
    for entry in entries:
        label = entry.get("precision_comparison")
        plan = entry.get("selected_plan")
        if not isinstance(label, dict) or not isinstance(plan, dict):
            continue
        lossy = any(plan.get(feature) for feature in lossy_features)
        same = label.get("same_weight_representation")
        assert same is not lossy or same is None, (
            f"{path.name}: plan runs {[f for f in lossy_features if plan.get(f)]} but the label "
            f"claims same_weight_representation={same}"
        )


@pytest.fixture
def structured_logits():
    return {
        "passed": True,
        "max_absolute_error": 0.0001,
        "mean_absolute_error": 0.00002,
        "rtol": 0.001,
        "atol": 0.001,
        "greedy_token_equal": True,
        "mlx_next_token": 37,
        "metile_next_token": 37,
    }


def test_passed_structured_logit_metrics_are_accuracy_evidence(structured_logits):
    assert _accuracy_evidence({"correctness": {"checks": [{"logits": structured_logits}]}})


@pytest.mark.parametrize(
    "field",
    [
        "passed",
        "max_absolute_error",
        "mean_absolute_error",
        "rtol",
        "atol",
        "greedy_token_equal",
        "mlx_next_token",
        "metile_next_token",
    ],
)
def test_structured_logit_evidence_requires_every_metric_and_greedy_check(structured_logits, field):
    structured_logits.pop(field)
    assert not _accuracy_evidence({"logits": structured_logits})


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("passed", 1),
        ("max_absolute_error", float("nan")),
        ("max_absolute_error", float("inf")),
        ("max_absolute_error", -1.0),
        ("max_absolute_error", True),
        ("mean_absolute_error", float("inf")),
        ("mean_absolute_error", -0.1),
        ("mean_absolute_error", "0"),
        ("mean_absolute_error", 0.1),
        ("rtol", float("nan")),
        ("rtol", -0.001),
        ("rtol", True),
        ("atol", float("inf")),
        ("atol", -0.001),
        ("greedy_token_equal", False),
        ("greedy_token_equal", 1),
        ("mlx_next_token", True),
        ("mlx_next_token", -1),
        ("metile_next_token", 38),
        ("metile_next_token", 37.0),
    ],
)
def test_invalid_structured_logit_metrics_do_not_count_as_accuracy_evidence(
    structured_logits, field, value
):
    structured_logits[field] = value
    assert not _accuracy_evidence({"logits": structured_logits})


@pytest.mark.parametrize("field", ACCURACY_FIELDS)
def test_legacy_mixed_precision_accuracy_fields_remain_recognized(field):
    document = {
        "precision_comparison": {"class": "mixed_precision", "same_weight_representation": False},
        "checks": [{field: 0.0}],
    }
    _require_accuracy_evidence(document, "legacy")


@pytest.fixture
def lossless_report():
    return json.loads((RESULTS / "m5-qwen3-direct-packed-end-to-end.json").read_text())


def test_published_lossless_report_passes_complete_accuracy_validation(lossless_report):
    _require_accuracy_evidence(lossless_report, "lossless")


def test_lossless_label_cannot_skip_validation_by_claiming_same_storage(lossless_report, tmp_path):
    lossless_report["precision_comparison"]["same_weight_representation"] = True
    path = tmp_path / "contradictory-lossless.json"
    path.write_text(json.dumps(lossless_report))
    with pytest.raises(ValueError):
        test_a_mixed_precision_result_carries_accuracy_evidence(path)


@pytest.mark.parametrize(
    "case",
    ["all_logits", "one_logits", "cache", "checks", "mean_error", "tolerance", "greedy", "packing"],
)
def test_stripping_published_lossless_evidence_cannot_leave_an_accepted_result(
    lossless_report, case
):
    if case == "all_logits":
        for phase in ("tuning", "held_out"):
            for result in lossless_report[phase]:
                for check in result["correctness"]["checks"]:
                    check.pop("logits", None)
    elif case == "packing":
        lossless_report["candidate_weight_packing"]["verification"].pop("roundtrip_bitwise")
    else:
        checks = lossless_report["held_out"][0]["correctness"]["checks"]
        generation = next(check for check in checks if check["stage"] == "generation")
        if case == "one_logits":
            generation.pop("logits")
        elif case == "cache":
            generation.pop("cache")
        elif case == "checks":
            checks.pop()
        elif case == "mean_error":
            generation["logits"].pop("mean_absolute_error")
        elif case == "tolerance":
            generation["logits"].pop("rtol")
        else:
            generation["logits"].pop("greedy_token_equal")
    with pytest.raises((AssertionError, ValueError)):
        _require_accuracy_evidence(lossless_report, "stripped lossless")
