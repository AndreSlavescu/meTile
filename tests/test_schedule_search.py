from metile.compiler.schedule_search import (
    FinitePermutation,
    GridSymmetryGroup,
    optimize_tile_schedules,
    rank_schedules,
)
from metile.ir import metal_ir as mir


def test_permutations_form_a_group_under_composition():
    permutation = FinitePermutation((1, 2, 0, 3))
    identity = FinitePermutation.identity(4)
    assert permutation.compose(identity) == permutation
    assert permutation.compose(permutation.inverse()) == identity
    assert permutation.cycles == ((0, 1, 2), (3,))


def test_grid_symmetry_fundamental_domain_covers_orbits():
    group = GridSymmetryGroup(4, 6)
    domain = group.fundamental_domain
    orbit_union = {
        symmetry.images[representative] for representative in domain for symmetry in group.elements
    }
    assert orbit_union == set(range(24))
    assert len(domain) < 24


def test_schedule_search_prefers_locality_and_removes_equivalents():
    ranked = rank_schedules(8, 8)
    assert ranked[0].locality_cost <= ranked[-1].locality_cost
    assert len({cost.canonical_signature for cost in ranked}) == len(ranked)


def test_schedule_pass_resolves_auto_to_static_ir():
    function = mir.MFunction("schedule_test", kernel_type="tensor_ops_gemm")
    schedule = mir.MTileSchedule(pattern="auto", grid_m=8, grid_n=8)
    function.add_op(schedule)
    optimize_tile_schedules(function)
    assert schedule.is_static
    assert schedule.pattern in {"hilbert", "morton", "diagonal", "linear"}
