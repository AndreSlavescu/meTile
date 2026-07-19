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
class ParallelEpilogueRule:
    """A parallel pair of producers joined by an elementwise epilogue."""

    name: str
    producer_op: str
    activation_op: str
    combine_op: str
    consumer_op: str
    register_values: int
    threadgroup_bytes: int
    kernel_count: int = 2


@dataclass(frozen=True)
class FusionRegion:
    """A legal, selected graph region that may lower as one kernel."""

    rule: ProducerConsumerRule | ParallelEpilogueRule
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
    ParallelEpilogueRule(
        name="parallel_matmul_swiglu_down",
        producer_op="matmul",
        activation_op="silu",
        combine_op="multiply",
        consumer_op="matmul",
        register_values=16,
        threadgroup_bytes=4 * 1024,
    ),
)


def plan_graph_fusion(
    graph: ComputeGraph,
    *,
    target: FusionTarget | None = None,
    rules: tuple[ProducerConsumerRule | ParallelEpilogueRule, ...] = DEFAULT_FUSION_RULES,
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
    selected = _select_non_overlapping_regions(candidates)

    order = {node: index for index, node in enumerate(graph.nodes)}
    selected.sort(key=lambda region: order[region.nodes[0]])
    return FusionPlan(graph, tuple(selected))


def _select_non_overlapping_regions(candidates):
    """Select globally optimal regions for cut-representable conflict components."""
    if not candidates:
        return []
    conflicts = _region_conflicts(candidates)
    selected = []
    for component in _conflict_components(conflicts):
        colors = _bipartite_colors(component, conflicts)
        if colors is None:
            selected.extend(_select_greedy_component(candidates, component, conflicts))
        else:
            selected.extend(_select_bipartite_component(candidates, component, conflicts, colors))
    return selected


def _region_conflicts(candidates):
    conflicts = [set() for _ in candidates]
    owners = {}
    for candidate_index, region in enumerate(candidates):
        for node in set(region.nodes):
            for owner in owners.setdefault(node, []):
                conflicts[candidate_index].add(owner)
                conflicts[owner].add(candidate_index)
            owners[node].append(candidate_index)
    return tuple(frozenset(neighbors) for neighbors in conflicts)


def _conflict_components(conflicts):
    remaining = set(range(len(conflicts)))
    components = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = []
        stack = [root]
        while stack:
            candidate = stack.pop()
            component.append(candidate)
            neighbors = conflicts[candidate] & remaining
            remaining.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _bipartite_colors(component, conflicts):
    colors = {}
    for root in component:
        if root in colors:
            continue
        colors[root] = 0
        stack = [root]
        while stack:
            candidate = stack.pop()
            color = colors[candidate]
            for neighbor in conflicts[candidate]:
                expected = 1 - color
                known = colors.get(neighbor)
                if known is not None and known != expected:
                    return None
                if known is None:
                    colors[neighbor] = expected
                    stack.append(neighbor)
    return colors


def _select_bipartite_component(candidates, component, conflicts, colors):
    source = ("terminal", "select")
    sink = ("terminal", "reject")
    vertices = {candidate: ("candidate", candidate) for candidate in component}
    total_benefit = sum(candidates[candidate].benefit_ns for candidate in component)
    infinity = total_benefit + 1.0
    network = FlowNetwork()
    for candidate in component:
        vertex = vertices[candidate]
        benefit = candidates[candidate].benefit_ns
        if colors[candidate] == 0:
            network.add_edge(source, vertex, benefit)
            for neighbor in conflicts[candidate]:
                network.add_edge(vertex, vertices[neighbor], infinity)
        else:
            network.add_edge(vertex, sink, benefit)
    _, source_side = network.minimum_cut(source, sink)
    return [
        candidates[candidate]
        for candidate in component
        if (colors[candidate] == 0) == (vertices[candidate] in source_side)
    ]


def _select_greedy_component(candidates, component, conflicts):
    selected = []
    occupied = set()
    for candidate in component:
        if conflicts[candidate] & occupied:
            continue
        selected.append(candidates[candidate])
        occupied.add(candidate)
    return selected


def _match_rule(graph, consumer, rule, consumers, graph_outputs, target):
    if isinstance(rule, ParallelEpilogueRule):
        return _match_parallel_epilogue(
            graph,
            consumer,
            rule,
            consumers,
            graph_outputs,
            target,
        )
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


def _match_parallel_epilogue(graph, consumer, rule, consumers, graph_outputs, target):
    if consumer.op != rule.consumer_op or consumer.side_effect or not consumer.inputs:
        return None
    combined = consumer.inputs[0]
    combine = combined.producer
    if combine is None or combine.op != rule.combine_op or combine.side_effect:
        return None
    if len(combine.inputs) != 2 or len(combine.outputs) != 1:
        return None

    activated = None
    parallel = None
    activation = None
    for candidate, other in (combine.inputs, reversed(combine.inputs)):
        candidate_producer = candidate.producer
        if candidate_producer is not None and candidate_producer.op == rule.activation_op:
            activated = candidate
            parallel = other
            activation = candidate_producer
            break
    if activation is None or parallel is None or len(activation.inputs) != 1:
        return None

    primary = activation.inputs[0]
    primary_producer = primary.producer
    parallel_producer = parallel.producer
    if (
        primary_producer is None
        or parallel_producer is None
        or primary_producer.op != rule.producer_op
        or parallel_producer.op != rule.producer_op
        or primary_producer.side_effect
        or parallel_producer.side_effect
        or primary_producer is parallel_producer
        or len(primary_producer.inputs) < 2
        or len(parallel_producer.inputs) < 2
        or primary_producer.inputs[0] is not parallel_producer.inputs[0]
        or primary.spec != parallel.spec
        or activated.spec != parallel.spec
        or combined.spec != parallel.spec
    ):
        return None

    nodes = (primary_producer, parallel_producer, activation, combine, consumer)
    node_set = set(nodes)
    intermediates = (primary, parallel, activated, combined)
    if any(
        value in graph_outputs or any(user not in node_set for user in consumers.get(value, ()))
        for value in intermediates
    ):
        return None
    if (
        rule.register_values > target.max_register_values
        or rule.threadgroup_bytes > target.max_threadgroup_bytes
    ):
        return None

    inputs = tuple(
        dict.fromkeys(
            value for node in nodes for value in node.inputs if value.producer not in node_set
        )
    )
    outputs = tuple(
        output
        for node in nodes
        for output in node.outputs
        if output in graph_outputs
        or any(user not in node_set for user in consumers.get(output, ()))
    )
    if not outputs:
        outputs = consumer.outputs

    saved_launches = max(0, len(nodes) - rule.kernel_count)
    elided_values = (primary, parallel, activated)
    saved_bytes = sum(value.spec.nbytes * 2 for value in elided_values)
    benefit = saved_launches * target.launch_overhead_ns + saved_bytes / max(
        target.memory_bandwidth_bytes_per_ns,
        1e-9,
    )
    return FusionRegion(rule, nodes, inputs, outputs, benefit)


__all__ = [
    "DEFAULT_FUSION_RULES",
    "FusionPlan",
    "FusionRegion",
    "FusionTarget",
    "ParallelEpilogueRule",
    "ProducerConsumerRule",
    "plan_graph_fusion",
]
