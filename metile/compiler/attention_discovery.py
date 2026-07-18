from __future__ import annotations

from dataclasses import dataclass

from metile.compiler.reduction_algebra import (
    ReductionCertificate,
    prove_reduction,
    weighted_softmax_reduction,
)
from metile.ir.graph_ir import ComputeGraph, GraphNode, GraphValue


@dataclass(frozen=True)
class FlashAttentionMatch:
    """One exact attention subgraph and its reduction proof."""

    query: GraphValue
    key: GraphValue
    value: GraphValue
    output_node: GraphNode
    nodes: tuple[GraphNode, ...]
    scale: float
    causal: bool
    certificate: ReductionCertificate


def find_flash_attention(graph: ComputeGraph) -> tuple[FlashAttentionMatch, ...]:
    """Find exact `softmax(scale(Q K^T)) V` regions with private intermediates."""
    consumers = graph.consumers()
    graph_outputs = set(graph.outputs)
    certificate = prove_reduction(weighted_softmax_reduction())
    if not certificate.verified:
        return ()

    matches = []
    occupied = set()
    for output_node in graph.nodes:
        match = _match_attention(
            output_node,
            consumers,
            graph_outputs,
            certificate,
        )
        if match is None or any(node in occupied for node in match.nodes):
            continue
        matches.append(match)
        occupied.update(match.nodes)
    return tuple(matches)


def discover_flash_attention(graph: ComputeGraph) -> ComputeGraph:
    """Replace verified exact-attention regions with one proof-carrying graph op."""
    matches = find_flash_attention(graph)
    if not matches:
        return graph

    final_matches = {match.output_node: match for match in matches}
    removed = {node for match in matches for node in match.nodes if node is not match.output_node}
    values = {value: value for value in graph.inputs}
    nodes = []
    for node in graph.nodes:
        if node in removed:
            continue
        if node in final_matches:
            match = final_matches[node]
            mapped_inputs = tuple(values[value] for value in (match.query, match.key, match.value))
            attrs = {
                "causal": match.causal,
                "reduction_certificate": match.certificate,
                "scale": match.scale,
            }
            replacement = _clone_node(node, "flash_attention", mapped_inputs, attrs)
        else:
            replacement = _clone_node(
                node,
                node.op,
                tuple(values[value] for value in node.inputs),
                node.attrs,
            )
        nodes.append(replacement)
        values.update(zip(node.outputs, replacement.outputs, strict=True))
    return ComputeGraph(
        graph.inputs,
        tuple(nodes),
        tuple(values[output] for output in graph.outputs),
    )


def _match_attention(output_node, consumers, graph_outputs, certificate):
    if output_node.op != "matmul" or output_node.attrs.get("transpose_right", False):
        return None
    if len(output_node.inputs) != 2:
        return None
    probabilities, value = output_node.inputs
    softmax = probabilities.producer
    if softmax is None or softmax.op != "softmax" or softmax.attrs.get("axis") != -1:
        return None

    current = softmax.inputs[0]
    causal = False
    mask = current.producer
    chain = [softmax]
    if mask is not None and mask.op == "causal_mask":
        causal = True
        chain.append(mask)
        current = mask.inputs[0]

    scale = 1.0
    scale_node = current.producer
    if scale_node is not None and scale_node.op == "scale":
        scale = float(scale_node.attrs["factor"])
        chain.append(scale_node)
        current = scale_node.inputs[0]

    scores = current.producer
    if scores is None or scores.op != "matmul" or not scores.attrs.get("transpose_right", False):
        return None
    query, key = scores.inputs
    chain.append(scores)
    chain.reverse()
    chain.append(output_node)

    if query.spec.dtype != key.spec.dtype or query.spec.dtype != value.spec.dtype:
        return None
    if (
        query.spec.shape[:-2] != key.spec.shape[:-2]
        or query.spec.shape[:-2] != value.spec.shape[:-2]
    ):
        return None
    if query.spec.shape[-1] != key.spec.shape[-1]:
        return None
    if key.spec.shape[-2] != value.spec.shape[-2]:
        return None
    if output_node.outputs[0].spec.shape != (*query.spec.shape[:-1], value.spec.shape[-1]):
        return None

    private_values = tuple(node.outputs[0] for node in chain[:-1])
    if any(
        intermediate in graph_outputs or consumers.get(intermediate, ()) != (consumer,)
        for intermediate, consumer in zip(private_values, chain[1:], strict=True)
    ):
        return None
    return FlashAttentionMatch(
        query,
        key,
        value,
        output_node,
        tuple(chain),
        scale,
        causal,
        certificate,
    )


def _clone_node(node, operation, inputs, attrs):
    outputs = tuple(
        GraphValue(output.name, output.spec, output_index=output.output_index)
        for output in node.outputs
    )
    replacement = GraphNode(
        node.name,
        operation,
        inputs,
        dict(attrs),
        outputs,
        side_effect=node.side_effect,
    )
    for output in outputs:
        output.producer = replacement
    return replacement


__all__ = ["FlashAttentionMatch", "discover_flash_attention", "find_flash_attention"]
