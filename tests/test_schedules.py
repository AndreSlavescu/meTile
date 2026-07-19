from itertools import pairwise

import pytest

from metile.compiler.schedules import resolve_schedule, schedule_coordinates, validate_schedule


@pytest.mark.parametrize("grid_m,grid_n", [(1, 1), (3, 5), (4, 4), (8, 12), (6, 10)])
@pytest.mark.parametrize(
    "pattern",
    [
        "auto",
        "grouped2",
        "grouped4",
        "grouped8",
        "hilbert",
        "morton",
        "diagonal",
        "linear",
    ],
)
def test_schedules_visit_every_tile_once(grid_m, grid_n, pattern):
    coordinates = list(schedule_coordinates(grid_m, grid_n, pattern))
    assert len(coordinates) == grid_m * grid_n
    assert len(set(coordinates)) == len(coordinates)
    assert set(coordinates) == {(m, n) for m in range(grid_m) for n in range(grid_n)}


def test_hilbert_panel_has_unit_locality():
    coordinates = list(schedule_coordinates(4, 4, "hilbert"))
    distances = [abs(m1 - m0) + abs(n1 - n0) for (m0, n0), (m1, n1) in pairwise(coordinates)]
    assert distances == [1] * 15


def test_auto_falls_back_without_duplicate_edge_tiles():
    assert resolve_schedule("auto", 8, 8) == "hilbert"
    assert resolve_schedule("auto", 6, 10) == "morton"
    assert resolve_schedule("auto", 3, 5) == "diagonal"


def test_schedule_validation():
    assert validate_schedule("") == "linear"
    with pytest.raises(ValueError, match="Unknown tile schedule"):
        validate_schedule("random")
    with pytest.raises(ValueError, match="block_size=4"):
        validate_schedule("hilbert", 2)
