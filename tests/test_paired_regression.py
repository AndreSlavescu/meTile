import pytest

from benchmarks.paired_regression import aggregate_results, regression_rows


def test_aggregate_results_uses_geometric_mean():
    assert aggregate_results([{"kernel": 1.0}, {"kernel": 4.0}]) == {"kernel": 2.0}


def test_aggregate_results_uses_supported_intersection():
    assert aggregate_results([{"shared": 1.0, "first": 1.0}, {"shared": 4.0, "second": 1.0}]) == {
        "shared": 2.0
    }


def test_aggregate_results_requires_common_kernel():
    with pytest.raises(ValueError, match="no kernels in common"):
        aggregate_results([{"first": 1.0}, {"second": 1.0}])


def test_regression_rows_apply_relative_threshold():
    rows, regressions = regression_rows(
        {"fast": 1.1, "slow": 1.2}, {"fast": 1.0, "slow": 1.0}, threshold=0.15
    )

    assert [row[0] for row in rows] == ["fast", "slow"]
    assert regressions == ["slow"]
