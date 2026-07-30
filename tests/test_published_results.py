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
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

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
        return any(_accuracy_evidence(value) for value in node.values())
    if isinstance(node, list):
        return any(_accuracy_evidence(value) for value in node)
    return False


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
    ]
    if not mixed:
        pytest.skip(f"{path.name} is a same-representation comparison")
    assert _accuracy_evidence(document), (
        f"{path.name} reports a mixed-precision speedup with no accuracy metric anywhere in it. "
        f"One of {ACCURACY_FIELDS} must accompany a result measured at a different precision."
    )


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
