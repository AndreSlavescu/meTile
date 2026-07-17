from itertools import combinations

import pytest

from metile.compiler.max_flow import FlowNetwork


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
