from itertools import combinations
from random import Random

import pytest

from metile.compiler import graph_fusion
from metile.compiler.graph_fusion import (
    FusionRegion,
    FusionTarget,
    ProducerConsumerRule,
    plan_graph_fusion,
)
from metile.ir.graph_ir import GraphBuilder, GraphNode, TensorSpec


def _residual_rms_graph(*, return_residual=True):
    builder = GraphBuilder()
    spec = TensorSpec((2, 1, 2048), "f16")
    values = builder.input("values", spec)
    residual = builder.input("residual", spec)
    weight = builder.input("weight", TensorSpec((2048,), "f16"))
    summed = builder.add(values, residual, name="residual_add")
    normalized = builder.rms_norm(summed, weight, 1e-5, name="rms_norm")
    outputs = (summed, normalized) if return_residual else normalized
    return builder.build(outputs)


def _swiglu_mlp_graph(*, return_gate=False):
    builder = GraphBuilder()
    values = builder.input("values", TensorSpec((2, 32, 64), "f16"))
    gate_weight = builder.input("gate_weight", TensorSpec((2, 64, 128), "f16"))
    up_weight = builder.input("up_weight", TensorSpec((2, 64, 128), "f16"))
    down_weight = builder.input("down_weight", TensorSpec((2, 128, 64), "f16"))
    gate = builder.matmul(values, gate_weight, name="gate")
    up = builder.matmul(values, up_weight, name="up")
    activated = builder.silu(gate, name="silu")
    hidden = builder.multiply(activated, up, name="gated")
    output = builder.matmul(hidden, down_weight, name="down")
    return builder.build((gate, output) if return_gate else output)


def _overlapping_scale_graph():
    builder = GraphBuilder()
    values = builder.input("values", TensorSpec((64,), "f16"))
    first = builder.scale(values, 2, name="z0")
    second = builder.scale(first, 3, name="a1")
    third = builder.scale(second, 4, name="z2")
    output = builder.scale(third, 5, name="z3")
    return builder.build(output)


def test_graph_fusion_discovers_multi_output_residual_rms_norm():
    graph = _residual_rms_graph()
    plan = plan_graph_fusion(graph)

    assert len(plan.regions) == 1
    region = plan.regions[0]
    assert region.rule.name == "residual_add_rms_norm"
    assert [node.op for node in region.nodes] == ["add", "rms_norm"]
    assert region.outputs == graph.outputs
    assert region.benefit_ns > 0


def test_graph_fusion_counts_dead_intermediate_memory_savings():
    graph = _residual_rms_graph(return_residual=False)
    target = FusionTarget(launch_overhead_ns=0, memory_bandwidth_bytes_per_ns=1)
    plan = plan_graph_fusion(graph, target=target)

    assert len(plan.regions) == 1
    assert plan.regions[0].benefit_ns == graph.nodes[0].outputs[0].spec.nbytes * 2


def test_graph_fusion_respects_resource_limits():
    graph = _residual_rms_graph()
    plan = plan_graph_fusion(graph, target=FusionTarget(max_register_values=4))

    assert plan.regions == ()


def test_graph_fusion_discovers_parallel_swiglu_epilogue_pipeline():
    graph = _swiglu_mlp_graph()
    plan = plan_graph_fusion(graph)

    assert len(plan.regions) == 1
    region = plan.regions[0]
    assert region.rule.name == "parallel_matmul_swiglu_down"
    assert [node.op for node in region.nodes] == [
        "matmul",
        "matmul",
        "silu",
        "multiply",
        "matmul",
    ]
    assert region.outputs == graph.outputs
    assert region.benefit_ns > 0


def test_parallel_swiglu_pipeline_rejects_escaping_intermediate():
    graph = _swiglu_mlp_graph(return_gate=True)

    assert plan_graph_fusion(graph).regions == ()


def test_parallel_swiglu_pipeline_respects_threadgroup_memory():
    graph = _swiglu_mlp_graph()
    plan = plan_graph_fusion(graph, target=FusionTarget(max_threadgroup_bytes=2048))

    assert plan.regions == ()


def test_graph_cut_keeps_materialized_producer_separate_without_launch_savings():
    graph = _residual_rms_graph()
    plan = plan_graph_fusion(graph, target=FusionTarget(launch_overhead_ns=0))

    assert plan.regions == ()


def test_global_graph_cut_beats_greedy_region_selection():
    graph = _overlapping_scale_graph()
    rule = ProducerConsumerRule(
        name="scale_chain",
        producer_op="scale",
        consumer_op="scale",
        consumer_input=0,
        register_values=1,
    )

    plan = plan_graph_fusion(graph, rules=(rule,))

    assert [tuple(node.name for node in region.nodes) for region in plan.regions] == [
        ("z0", "a1"),
        ("z2", "z3"),
    ]


def test_bipartite_graph_cut_matches_exhaustive_weighted_selection():
    random = Random(11)
    rule = ProducerConsumerRule("test", "add", "add", 0, 1)
    for _ in range(100):
        left_count = random.randint(1, 4)
        right_count = random.randint(1, 4)
        edge_nodes = {}
        for left in range(left_count):
            for right in range(right_count):
                if random.random() < 0.45:
                    edge_nodes[left, right] = GraphNode(f"edge_{left}_{right}", "add", (), {}, ())
        candidates = []
        for side, count in ((0, left_count), (1, right_count)):
            for vertex in range(count):
                nodes = tuple(
                    node
                    for (left, right), node in edge_nodes.items()
                    if (left if side == 0 else right) == vertex
                )
                if not nodes:
                    nodes = (GraphNode(f"private_{side}_{vertex}", "add", (), {}, ()),)
                candidates.append(
                    FusionRegion(
                        rule,
                        nodes,
                        (),
                        (),
                        float(random.randint(1, 20)),
                    )
                )
        candidates.sort(key=lambda region: -region.benefit_ns)

        selected = graph_fusion._select_non_overlapping_regions(candidates)
        selected_nodes = [set(region.nodes) for region in selected]
        actual = sum(region.benefit_ns for region in selected)
        expected = 0.0
        for size in range(len(candidates) + 1):
            for subset in combinations(candidates, size):
                node_sets = [set(region.nodes) for region in subset]
                if all(
                    node_sets[left].isdisjoint(node_sets[right])
                    for left in range(len(node_sets))
                    for right in range(left + 1, len(node_sets))
                ):
                    expected = max(expected, sum(region.benefit_ns for region in subset))

        assert all(
            selected_nodes[left].isdisjoint(selected_nodes[right])
            for left in range(len(selected_nodes))
            for right in range(left + 1, len(selected_nodes))
        )
        assert actual == expected


def test_graph_builder_rejects_incompatible_iteration_domains():
    builder = GraphBuilder()
    left = builder.input("left", TensorSpec((64,), "f16"))
    right = builder.input("right", TensorSpec((32,), "f16"))

    with pytest.raises(ValueError, match="identical"):
        builder.add(left, right)
