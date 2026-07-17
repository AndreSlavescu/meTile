from __future__ import annotations

from dataclasses import dataclass

from metile.compiler.max_flow import FlowNetwork
from metile.ir.graph_ir import ComputeGraph, GraphNode, GraphValue


@dataclass(frozen=True)
class FusionTarget:
    """Target costs and resource limits used during region selection."""

    launch_overhead_ns: float = 12_000.0
    memory_bandwidth_bytes_per_ns: float = 0.12
    max_register_values: int = 64
    max_threadgroup_bytes: int = 32 * 1024


@dataclass(frozen=True)
class ProducerConsumerRule:
    """A declarative two-operation fusion rule."""

    name: str
    producer_op: str
    consumer_op: str
    consumer_input: int
    register_values: int
    threadgroup_bytes_per_element: int = 0
    materialize_producer: bool = False


@dataclass(frozen=True)
class FusionRegion:
    """A legal, selected graph region that may lower as one kernel."""

    rule: ProducerConsumerRule
    nodes: tuple[GraphNode, ...]
    inputs: tuple[GraphValue, ...]
    outputs: tuple[GraphValue, ...]
    benefit_ns: float


@dataclass(frozen=True)
class FusionPlan:
    graph: ComputeGraph
    regions: tuple[FusionRegion, ...]

    def region_for(self, node: GraphNode) -> FusionRegion | None:
        return next((region for region in self.regions if node in region.nodes), None)


DEFAULT_FUSION_RULES = (
    ProducerConsumerRule(
        name="residual_add_rms_norm",
        producer_op="add",
        consumer_op="rms_norm",
        consumer_input=0,
        register_values=8,
        materialize_producer=True,
    ),
)


def plan_graph_fusion(
    graph: ComputeGraph,
    *,
    target: FusionTarget | None = None,
    rules: tuple[ProducerConsumerRule, ...] = DEFAULT_FUSION_RULES,
) -> FusionPlan:
    """Select non-overlapping profitable fusion regions from a compute DAG.

    Legality is structural and target-aware. Profitability estimates launch and
    intermediate-memory savings, but final framework dispatch still measures a
    fused kernel against its unfused implementation.
    """

    target = target or FusionTarget()
    consumers = graph.consumers()
    graph_outputs = set(graph.outputs)
    candidates = []
    for consumer in graph.nodes:
        for rule in rules:
            region = _match_rule(graph, consumer, rule, consumers, graph_outputs, target)
            if region is not None and region.benefit_ns > 0:
                candidates.append(region)

    candidates.sort(key=lambda region: (-region.benefit_ns, region.nodes[0].name))
    selected = []
    occupied = set()
    for region in candidates:
        if any(node in occupied for node in region.nodes):
            continue
        selected.append(region)
        occupied.update(region.nodes)

    order = {node: index for index, node in enumerate(graph.nodes)}
    selected.sort(key=lambda region: order[region.nodes[0]])
    return FusionPlan(graph, tuple(selected))


def _match_rule(graph, consumer, rule, consumers, graph_outputs, target):
    if consumer.op != rule.consumer_op or consumer.side_effect:
        return None
    if rule.consumer_input >= len(consumer.inputs):
        return None
    produced_value = consumer.inputs[rule.consumer_input]
    producer = produced_value.producer
    if producer is None or producer.op != rule.producer_op or producer.side_effect:
        return None
    if len(producer.outputs) != 1 or produced_value is not producer.outputs[0]:
        return None
    if produced_value.spec != consumer.outputs[0].spec:
        return None
    trailing_extent = produced_value.spec.shape[-1]
    exceeds_resources = (
        rule.register_values > target.max_register_values
        or rule.threadgroup_bytes_per_element * trailing_extent > target.max_threadgroup_bytes
    )

    external_consumers = tuple(
        node for node in consumers.get(produced_value, ()) if node is not consumer
    )
    producer_escapes = produced_value in graph_outputs or bool(external_consumers)
    if producer_escapes and not rule.materialize_producer:
        return None

    nodes = (producer, consumer)
    node_set = set(nodes)
    inputs = tuple(
        value for node in nodes for value in node.inputs if value.producer not in node_set
    )
    inputs = tuple(dict.fromkeys(inputs))
    outputs = tuple(
        output
        for node in nodes
        for output in node.outputs
        if output in graph_outputs
        or any(user not in node_set for user in consumers.get(output, ()))
    )
    if not outputs:
        outputs = consumer.outputs

    saved_bytes = 0 if producer_escapes else produced_value.spec.nbytes * 2
    materialization_cost = saved_bytes / max(target.memory_bandwidth_bytes_per_ns, 1e-9)
    separate_cost = target.launch_overhead_ns + materialization_cost
    fusion_cost = 0.0
    infinity = separate_cost + target.launch_overhead_ns + 1.0

    source = ("terminal", "fused")
    sink = ("terminal", "separate")
    network = FlowNetwork()
    network.add_edge(source, consumer, infinity)
    network.add_edge(consumer, producer, separate_cost)
    network.add_edge(producer, sink, infinity if exceeds_resources else fusion_cost)
    cut_cost, fused_side = network.minimum_cut(source, sink)
    if producer not in fused_side:
        return None

    benefit = separate_cost - cut_cost
    return FusionRegion(rule, nodes, inputs, outputs, benefit)


__all__ = [
    "DEFAULT_FUSION_RULES",
    "FusionPlan",
    "FusionRegion",
    "FusionTarget",
    "ProducerConsumerRule",
    "plan_graph_fusion",
]
