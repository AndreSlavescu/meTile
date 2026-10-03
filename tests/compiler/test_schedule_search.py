from metile.codegen.msl_emitter import emit
from metile.compiler.schedule_expr import schedule_programs, select_schedule_program
from metile.compiler.schedule_search import (
    FinitePermutation,
    FinitePermutationGroup,
    GridSymmetryGroup,
    choose_mdl_tie,
    compressed_description_bits,
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


def test_generated_group_closure_contains_inverses():
    cycle = FinitePermutation((1, 2, 3, 0))
    reflection = FinitePermutation((0, 3, 2, 1))
    group = FinitePermutationGroup(4, (cycle, reflection))
    assert len(group.elements) == 8
    assert all(element.inverse() in group.elements for element in group.elements)


def test_grid_symmetry_fundamental_domain_covers_orbits():
    group = GridSymmetryGroup(4, 6)
    domain = group.fundamental_domain
    orbit_union = {
        symmetry.images[representative] for representative in domain for symmetry in group.elements
    }
    assert orbit_union == set(range(24))
    assert len(domain) < 24
    for representative in domain:
        assert len(group.orbit(representative)) * len(group.stabilizer(representative)) == len(
            group.elements
        )


def test_square_grid_uses_full_dihedral_symmetry_group():
    assert len(GridSymmetryGroup(4, 4).elements) == 8
    assert len(GridSymmetryGroup(4, 6).elements) == 4
    assert GridSymmetryGroup(4, 4).name == "D4"
    assert GridSymmetryGroup(4, 4, block_m=64, block_n=128).name == "D2"
    assert GridSymmetryGroup(4, 4, axis_interchangeable=False).name == "D2"


def test_schedule_expression_encodings_are_equivalent_and_bijective():
    cases = [
        ("linear", 3, 5),
        ("diagonal", 3, 5),
        ("grouped2", 4, 6),
        ("grouped4", 8, 6),
        ("morton", 6, 10),
        ("hilbert", 8, 8),
    ]
    for pattern, grid_m, grid_n in cases:
        programs = schedule_programs(pattern, grid_m, grid_n)
        results = [
            tuple(program.evaluate(m, n) for m in range(grid_m) for n in range(grid_n))
            for program in programs
        ]
        assert all(result == results[0] for result in results)
        assert set(results[0]) == {(m, n) for m in range(grid_m) for n in range(grid_n)}


def test_program_extraction_strength_reduces_power_of_two_division():
    selected = select_schedule_program("hilbert", 8, 8)
    arithmetic = select_schedule_program("hilbert", 8, 8, "arithmetic")
    assert selected.encoding == "bitwise"
    assert selected.target_cost == arithmetic.target_cost
    assert selected.description_bits < arithmetic.description_bits
    assert ">>" in "\n".join(selected.emit_lines())


def test_static_schedule_codegen_emits_extracted_expression_program():
    function = mir.MFunction("schedule_codegen", kernel_type="tensor_ops_gemm")
    schedule = mir.MTileSchedule(pattern="hilbert", block_m=64, block_n=64, grid_m=8, grid_n=8)
    function.add_op(schedule)
    optimize_tile_schedules(function)
    source = emit(function)
    program = select_schedule_program("hilbert", 8, 8, schedule.encoding)
    assert all(line in source for line in program.emit_lines())


def test_schedule_pass_honors_an_explicit_equivalent_encoding():
    function = mir.MFunction("schedule_encoding", kernel_type="tensor_ops_gemm")
    schedule = mir.MTileSchedule(
        pattern="hilbert",
        block_m=64,
        block_n=64,
        grid_m=8,
        grid_n=8,
        encoding="arithmetic",
    )
    function.add_op(schedule)
    optimize_tile_schedules(function)
    assert schedule.encoding == "arithmetic"
    assert "/ 16u" in emit(function)
    assert select_schedule_program("linear", 3, 5, "bitwise").encoding == "bitwise"


def test_mdl_breaks_only_measurement_scale_latency_ties():
    candidates = [(1.0, 100, "fast"), (1.002, 50, "compact")]
    assert choose_mdl_tie(candidates) == "compact"
    assert choose_mdl_tie([(1.0, 100, "fast"), (1.004, 50, "compact")]) == "fast"
    assert compressed_description_bits("repeat " * 100) < compressed_description_bits(
        "".join(chr(33 + index % 90) for index in range(700))
    )


def test_schedule_search_prefers_locality_and_removes_equivalents():
    ranked = rank_schedules(8, 8)
    assert ranked[0].locality_cost <= ranked[-1].locality_cost
    assert len({cost.canonical_signature for cost in ranked}) == len(ranked)


def test_schedule_pass_resolves_auto_to_static_ir():
    function = mir.MFunction("schedule_test", kernel_type="tensor_ops_gemm")
    schedule = mir.MTileSchedule(
        pattern="auto",
        block_m=64,
        block_n=64,
        grid_m=8,
        grid_n=8,
        axis_interchangeable=True,
    )
    function.add_op(schedule)
    optimize_tile_schedules(function)
    assert schedule.is_static
    assert schedule.encoding in {"arithmetic", "bitwise"}
    assert schedule.description_bits > 0
    assert schedule.symmetry_group == "D4"
    assert schedule.pattern in {
        "grouped2",
        "grouped4",
        "grouped8",
        "hilbert",
        "morton",
        "diagonal",
        "linear",
    }
