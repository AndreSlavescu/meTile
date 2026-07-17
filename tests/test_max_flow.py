from itertools import combinations
from random import Random

import pytest

from metile.compiler.max_flow import FlowNetwork, clear_flow_solver_cache


def test_dinic_returns_max_flow_and_source_side_minimum_cut():
    network = FlowNetwork()
    network.add_edge("source", "left", 3)
    network.add_edge("source", "right", 2)
    network.add_edge("left", "right", 1)
    network.add_edge("left", "sink", 2)
    network.add_edge("right", "sink", 3)

    flow, source_side = network.minimum_cut("source", "sink")

    assert flow == 5
    assert source_side == frozenset({"source"})


def test_flow_network_rejects_negative_capacities():
    network = FlowNetwork()

    with pytest.raises(ValueError, match="non-negative"):
        network.add_edge("source", "sink", -1)


def test_flow_network_rejects_non_finite_capacities():
    network = FlowNetwork()

    with pytest.raises(ValueError, match="finite"):
        network.add_edge("source", "sink", float("inf"))


def test_max_flow_matches_exhaustive_cuts_on_small_graphs():
    vertices = ("source", "a", "b", "c", "sink")
    edges = (
        ("source", "a", 7),
        ("source", "b", 4),
        ("a", "b", 3),
        ("a", "c", 5),
        ("b", "c", 2),
        ("b", "sink", 4),
        ("c", "sink", 8),
    )
    network = FlowNetwork()
    for edge in edges:
        network.add_edge(*edge)

    flow, source_side = network.minimum_cut("source", "sink")
    internal = vertices[1:-1]
    cuts = []
    for count in range(len(internal) + 1):
        for selected in combinations(internal, count):
            selected = {"source", *selected}
            cost = sum(
                capacity
                for source, target, capacity in edges
                if source in selected and target not in selected
            )
            cuts.append((cost, selected))

    expected = min(cost for cost, _ in cuts)
    actual_cut = sum(
        capacity
        for source, target, capacity in edges
        if source in source_side and target not in source_side
    )
    assert flow == expected
    assert actual_cut == expected


def test_exact_solvers_are_reusable_and_agree_on_random_graphs():
    random = Random(7)
    for vertex_count in range(2, 11):
        for _ in range(20):
            network = FlowNetwork()
            vertices = list(range(vertex_count))
            edges = []
            for source in vertices:
                for target in vertices:
                    if source != target and random.random() < 0.25:
                        edge = (source, target, random.randint(1, 20))
                        edges.append(edge)
                        network.add_edge(*edge)

            dinic = network.minimum_cut(0, vertex_count - 1, solver="dinic")
            push_relabel = network.minimum_cut(0, vertex_count - 1, solver="push_relabel")
            automatic = network.minimum_cut(0, vertex_count - 1)
            repeated = network.minimum_cut(0, vertex_count - 1, solver="dinic")

            assert push_relabel[0] == pytest.approx(dinic[0])
            assert automatic[0] == pytest.approx(dinic[0])
            assert repeated == dinic
            assert _cut_capacity(edges, dinic[1]) == pytest.approx(dinic[0])
            assert _cut_capacity(edges, push_relabel[1]) == pytest.approx(push_relabel[0])
            assert _cut_capacity(edges, automatic[1]) == pytest.approx(automatic[0])


def test_auto_solver_keeps_small_graphs_on_dinic():
    network = FlowNetwork()
    network.add_edge("source", "sink", 1)

    assert network.select_solver() == "dinic"


def test_auto_solver_dispatches_large_dense_graphs_to_push_relabel():
    clear_flow_solver_cache()
    network = FlowNetwork()
    for source in range(32):
        for offset in range(1, 12):
            network.add_edge(source, (source + offset) % 32, 1)

    assert network.select_solver() == "autotune"

    automatic = network.minimum_cut(0, 31)
    reference = network.minimum_cut(0, 31, solver="dinic")

    selected = network.select_solver(source=0, sink=31)
    assert selected in {"dinic", "push_relabel"}
    assert automatic[0] == pytest.approx(reference[0])

    equivalent = FlowNetwork()
    for source in range(32):
        for offset in range(1, 12):
            equivalent.add_edge(source, (source + offset) % 32, 1)
    assert equivalent.select_solver(source=0, sink=31) == selected

    different_capacities = FlowNetwork()
    for source in range(32):
        for offset in range(1, 12):
            different_capacities.add_edge(source, (source + offset) % 32, 2)
    assert different_capacities.select_solver(source=0, sink=31) == "autotune"


def test_auto_solver_tunes_large_sparse_graphs():
    clear_flow_solver_cache()
    network = FlowNetwork()
    for source in range(128):
        for offset in range(1, 4):
            network.add_edge(source, (source + offset) % 128, 1)

    assert network.select_solver() == "autotune"

    automatic = network.minimum_cut(0, 127)
    reference = network.minimum_cut(0, 127, solver="dinic")

    assert network.select_solver(source=0, sink=127) in {"dinic", "push_relabel"}
    assert automatic[0] == pytest.approx(reference[0])


def test_flow_network_rejects_unknown_solver():
    network = FlowNetwork()

    with pytest.raises(ValueError, match="unknown flow solver"):
        network.select_solver("not-a-solver")


def test_dinic_handles_graphs_deeper_than_the_python_recursion_limit():
    network = FlowNetwork()
    for vertex in range(1_500):
        network.add_edge(vertex, vertex + 1, 1)

    flow, source_side = network.minimum_cut(0, 1_500, solver="dinic")

    assert flow == 1
    assert 0 in source_side
    assert 1_500 not in source_side


def _cut_capacity(edges, source_side):
    return sum(
        capacity
        for source, target, capacity in edges
        if source in source_side and target not in source_side
    )
