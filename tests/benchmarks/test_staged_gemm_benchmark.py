from copy import deepcopy

import pytest

from benchmarks.compiler.staged_gemm import cases, summarize


def _samples():
    baseline_case = {
        "name": "float16_64x64x64",
        "shape": [64, 64, 64],
        "dtype": "float16",
        "status": "ok",
        "input_sha256": "input",
        "source_sha256": "baseline-source",
        "schedule": {"backend": "simdgroup", "double_buffer": True},
        "timing": {"gpu_us": 10.0, "wall_us": 100.0},
    }
    current_case = deepcopy(baseline_case)
    current_case.update(source_sha256="current-source", timing={"gpu_us": 5.0, "wall_us": 80.0})
    return {
        "baseline": [
            {"compiler_implementation_sha256": "baseline", "cases": [baseline_case]}
            for _ in range(2)
        ],
        "current": [
            {"compiler_implementation_sha256": "current", "cases": [current_case]} for _ in range(2)
        ],
    }


def test_cases_cover_both_storage_types_and_aligned_and_ragged_shapes():
    selected = cases()
    assert len(selected) == 8
    assert {case["dtype"] for case in selected} == {"float16", "float32"}
    assert {tuple(case["shape"]) for case in selected} == {
        (64, 64, 64),
        (256, 256, 256),
        (1024, 1024, 1024),
        (257, 255, 259),
    }


def test_summary_uses_identical_contracts_and_independent_timing_metrics():
    result = summarize(_samples())[0]

    assert result["gpu_us"]["speedup"] == pytest.approx(2.0)
    assert result["wall_us"]["speedup"] == pytest.approx(1.25)
    assert result["source_sha256"] == {"baseline": "baseline-source", "current": "current-source"}


@pytest.mark.parametrize("status", ["incorrect", "unavailable"])
def test_baseline_failures_are_reported_not_removed_or_timed(status):
    samples = _samples()
    samples["baseline"][0]["cases"][0] = {
        "name": "float16_64x64x64",
        "status": status,
        "error": "failed baseline",
    }

    result = summarize(samples)[0]

    assert result["status"] == "incomparable"
    assert result["failures"]["baseline"] == [{"status": status, "error": "failed baseline"}]
    assert "gpu_us" not in result


@pytest.mark.parametrize(
    "field", ["shape", "dtype", "input_sha256", "schedule", "source_sha256", "gpu_us", "compiler"]
)
def test_mismatched_or_unstable_samples_are_rejected(field):
    samples = deepcopy(_samples())
    current = samples["current"][0]["cases"][0]
    if field == "shape":
        current["shape"] = [32, 64, 64]
    elif field == "dtype":
        current["dtype"] = "float32"
    elif field == "input_sha256":
        current[field] = "different-input"
    elif field == "schedule":
        current[field] = {"double_buffer": False}
    elif field == "source_sha256":
        samples["current"][0]["cases"][0] = dict(current, source_sha256="changed-source")
    elif field == "gpu_us":
        current["timing"][field] = float("nan")
    elif field == "compiler":
        samples["current"][0]["compiler_implementation_sha256"] = "changed-compiler"

    with pytest.raises(ValueError):
        summarize(samples)


def test_case_sets_cannot_silently_drop_unavailable_results():
    samples = _samples()
    samples["baseline"][0]["cases"] = []

    with pytest.raises(ValueError, match="same ordered cases"):
        summarize(samples)


def test_summary_requires_results_from_both_compilers():
    with pytest.raises(ValueError, match="both baseline and current"):
        summarize({"baseline": [], "current": []})
