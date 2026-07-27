"""Algorithmic discovery: propose, select, rewrite.

Every algorithmic rewrite meTile can discover lives here, and all of them run the
same three stages regardless of which IR they target:

    propose   a law from :mod:`metile.compiler.reduction_algebra` says what is
              legal; a matcher finds regions obeying it, each carrying a
              machine-checked certificate and a benefit estimate
    select    candidates that overlap are mutually exclusive, so an exact s-t
              min-cut picks the maximum-benefit conflict-free set
    rewrite   the survivors are applied to the IR

Registered rewrites:

    online softmax    Tile IR         3-pass softmax -> 2-pass, one fewer read of X
    flash attention   ComputeGraph    softmax(scale(Q K^T)) V -> one fused op

Both are licensed by the *same* law. ``weighted_softmax_reduction`` is the
``(maximum, normalizer, numerator)`` monoid; online softmax is its
``(maximum, normalizer)`` projection with the value component fixed at 1. Adding a
rewrite therefore means adding a matcher, not a new proof.

This module is the single entry point: the reduction laws and certificate types are
re-exported, so composing discovery only ever requires importing
:mod:`metile.compiler.algo_discovery`. It builds on two libraries that are
deliberately kept separate because they are not discovery — the symbolic proof
theory in :mod:`metile.compiler.reduction_algebra` and the exact min-cut solver in
:mod:`metile.compiler.max_flow`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from metile.compiler.max_flow import FlowNetwork
from metile.compiler.reduction_algebra import (
    ProofObligation,
    ReductionCertificate,
    ReductionLaw,
    max_reduction,
    prove_reduction,
    sum_reduction,
    weighted_softmax_reduction,
)
from metile.ir import tile_ir as tir
from metile.ir.graph_ir import ComputeGraph, GraphNode, GraphValue
from metile.ir.types import ScalarType, TileType

__all__ = [
    "Candidate",
    "FlashAttentionMatch",
    "ProofObligation",
    "ReductionCertificate",
    "ReductionLaw",
    "attention_monoid_certificate",
    "discover",
    "discover_flash_attention",
    "discover_online_softmax",
    "find_flash_attention",
    "find_online_softmax",
    "max_reduction",
    "prove_reduction",
    "select_candidates",
    "sum_reduction",
    "weighted_softmax_reduction",
]


# --------------------------------------------------------------------------- law


_ATTENTION_CERTIFICATE: ReductionCertificate | None = None


def attention_monoid_certificate() -> ReductionCertificate:
    """Prove the stable weighted-softmax monoid once and reuse it.

    Both registered rewrites depend on this single law, so the obligations are
    discharged once per process rather than per candidate.
    """
    global _ATTENTION_CERTIFICATE
    if _ATTENTION_CERTIFICATE is None:
        _ATTENTION_CERTIFICATE = prove_reduction(weighted_softmax_reduction())
    return _ATTENTION_CERTIFICATE


# -------------------------------------------------------------------- candidates


@dataclass(eq=False)
class Candidate:
    """One proposed rewrite, its region, and what applying it is worth.

    ``region`` holds the IR members the rewrite consumes. Two candidates conflict
    when their regions intersect, which is what makes selection a real choice rather
    than an iteration order. ``benefit`` is only ever compared against other
    candidates from the same IR, so its unit is per-matcher: bytes not materialized
    for graph rewrites, passes eliminated for Tile IR rewrites.
    """

    name: str
    region: tuple[object, ...]
    benefit: float
    certificate: ReductionCertificate
    payload: object = None

    @property
    def verified(self) -> bool:
        return self.certificate.verified


def select_candidates(candidates) -> tuple[Candidate, ...]:
    """Return the maximum-benefit conflict-free subset of verified candidates.

    Candidates sharing a region member cannot both be applied, so this is
    maximum-weight independent set. An exact s-t min-cut solves it whenever the
    conflict component is bipartite (the project-selection reduction used by
    :mod:`metile.compiler.graph_fusion`); other components fall back to
    benefit-ordered greedy, because the general problem is NP-hard.
    """
    verified = [candidate for candidate in candidates if candidate.verified]
    if len(verified) <= 1:
        return tuple(verified)

    conflicts = _conflicts(verified)
    if not any(conflicts.values()):
        return tuple(verified)

    selected: list[Candidate] = []
    for component in _components(verified, conflicts):
        colors = _bipartite_colors(component, conflicts)
        if colors is None:
            selected.extend(_greedy_component(component, conflicts))
        else:
            selected.extend(_min_cut_component(component, conflicts, colors))
    order = {candidate: index for index, candidate in enumerate(verified)}
    return tuple(sorted(selected, key=order.__getitem__))


def _conflicts(candidates):
    owners: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        for member in candidate.region:
            owners.setdefault(id(member), []).append(candidate)
    conflicts = {candidate: set() for candidate in candidates}
    for sharing in owners.values():
        for left in sharing:
            for right in sharing:
                if left is not right:
                    conflicts[left].add(right)
    return conflicts


def _components(candidates, conflicts):
    seen: set[int] = set()
    components = []
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        stack, component = [candidate], []
        seen.add(id(candidate))
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in conflicts[current]:
                if id(neighbor) not in seen:
                    seen.add(id(neighbor))
                    stack.append(neighbor)
        components.append(component)
    return components


def _bipartite_colors(component, conflicts):
    colors: dict[Candidate, int] = {}
    for start in component:
        if start in colors:
            continue
        colors[start] = 0
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in conflicts[current]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[current]
                    stack.append(neighbor)
                elif colors[neighbor] == colors[current]:
                    return None
    return colors


def _min_cut_component(component, conflicts, colors):
    source = ("terminal", "select")
    sink = ("terminal", "reject")
    vertices = {candidate: ("candidate", index) for index, candidate in enumerate(component)}
    infinity = sum(candidate.benefit for candidate in component) + 1.0

    network = FlowNetwork()
    for candidate in component:
        vertex = vertices[candidate]
        if colors[candidate] == 0:
            network.add_edge(source, vertex, candidate.benefit)
            for neighbor in conflicts[candidate]:
                network.add_edge(vertex, vertices[neighbor], infinity)
        else:
            network.add_edge(vertex, sink, candidate.benefit)

    _, source_side = network.minimum_cut(source, sink)
    return [
        candidate
        for candidate in component
        if (colors[candidate] == 0) == (vertices[candidate] in source_side)
    ]


def _greedy_component(component, conflicts):
    selected, occupied = [], set()
    for candidate in sorted(component, key=lambda item: -item.benefit):
        if any(id(neighbor) in occupied for neighbor in conflicts[candidate]):
            continue
        selected.append(candidate)
        occupied.add(id(candidate))
    return selected


# ------------------------------------------------------------------- entry point


def discover(ir):
    """Run every rewrite registered for this IR and return the rewritten IR."""
    if isinstance(ir, ComputeGraph):
        return discover_flash_attention(ir)
    if isinstance(ir, tir.Function):
        return discover_online_softmax(ir)
    raise TypeError(f"no algorithmic discovery registered for {type(ir).__name__}")


# ------------------------------------------------------- flash attention (graph)


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
    certificate = attention_monoid_certificate()
    if not certificate.verified:
        return ()

    consumers = graph.consumers()
    graph_outputs = set(graph.outputs)
    candidates = []
    for output_node in graph.nodes:
        match = _match_attention(output_node, consumers, graph_outputs, certificate)
        if match is None:
            continue
        # Not materializing the probability tensor is what the rewrite buys.
        saved = float(match.output_node.inputs[0].spec.nbytes)
        candidates.append(
            Candidate(
                name="flash_attention",
                region=match.nodes,
                benefit=saved,
                certificate=certificate,
                payload=match,
            )
        )
    return tuple(candidate.payload for candidate in select_candidates(candidates))


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


# -------------------------------------------------------- online softmax (tile)


@dataclass(frozen=True)
class _SoftmaxPattern:
    """The three loops and two reductions of a textbook 3-pass softmax."""

    max_loop: tir.ForRange
    sum_loop: tir.ForRange
    norm_loop: tir.ForRange
    max_reduce: tir.Reduce
    sum_reduce: tir.Reduce
    loop_indices: tuple[int, ...] = field(default=())


def find_online_softmax(func: tir.Function) -> tuple[Candidate, ...]:
    """Propose the online-softmax rewrite wherever a 3-pass softmax appears."""
    pattern = _detect_softmax_pattern(func.ops)
    if pattern is None:
        return ()
    certificate = attention_monoid_certificate()
    if not certificate.verified:
        return ()
    max_loop, sum_loop, norm_loop, max_reduce, sum_reduce, indices = pattern
    return (
        Candidate(
            name="online_softmax",
            region=(max_loop, sum_loop, norm_loop, max_reduce, sum_reduce),
            # One of the three passes over X disappears.
            benefit=1.0,
            certificate=certificate,
            payload=_SoftmaxPattern(
                max_loop, sum_loop, norm_loop, max_reduce, sum_reduce, tuple(indices)
            ),
        ),
    )


def discover_online_softmax(func: tir.Function) -> tir.Function:
    """Fuse a 3-pass softmax into a 2-pass online softmax.

    Detects::

        loop 1: m_tile = max(m_tile, load(X));  m = reduce_max(m_tile)
        loop 2: s_tile = s_tile + exp(load(X) - m);  s = reduce_sum(s_tile)
        loop 3: store(exp(load(X) - m) / s)

    and rewrites the first two into one pass carrying a running maximum and a
    running normalizer, rescaling the normalizer by ``exp(m_prev - m_new)`` whenever
    the maximum moves. Saves one full read of the input.

    Known limitation: the threadgroup reduction now sits inside the loop, and the
    ragged tail loop wraps its body in a mask branch, so not every thread reaches the
    barrier when the length is not a multiple of the block. Correct for aligned
    lengths only, which is why the pass stays behind ``METILE_ONLINE_SOFTMAX``.
    """
    selected = select_candidates(find_online_softmax(func))
    if not selected:
        return func
    return _apply_online_softmax(func, selected[0].payload)


def _apply_online_softmax(func: tir.Function, pattern: _SoftmaxPattern) -> tir.Function:
    max_loop = pattern.max_loop
    max_reduce, sum_reduce = pattern.max_reduce, pattern.sum_reduce

    load_in_max = _find_load(max_loop.body)
    if load_in_max is None or not isinstance(load_in_max.result.type, TileType):
        return func
    tile_shape = load_in_max.result.type.shape

    # Keep the addressing prefix of the max loop (arange, mask, pointer arithmetic,
    # load) and drop its element-wise max chain. That chain seeds a *tile* accumulator,
    # so its Constant is promoted to a tile-typed variable by the lowering; reading it
    # as the running scalar maximum is what produced `float x = <float4>`.
    fused_body = copy.deepcopy(max_loop.body)
    chunk_load = _find_load(fused_body)
    tile_max = _find_binop(fused_body, "max")
    if chunk_load is None or tile_max is None:
        return func
    tile_seed = (
        tile_max.lhs.defining_op if isinstance(tile_max.lhs.defining_op, tir.Constant) else None
    )
    fused_body = [op for op in fused_body if op is not tile_max and op is not tile_seed]

    scalar = ScalarType("f32")
    tile = TileType(tile_shape, "f32")
    name = _NameGen()
    online: list[tir.Op] = []

    def emit(op, result_name, result_type):
        value = _bind(op, result_name, result_type)
        online.append(op)
        return value

    chunk_max = emit(tir.Reduce(op="max", operand=chunk_load.result), name("cmax"), scalar)

    # Running maximum and running normalizer each get their own Constant seed, so the
    # lowering promotes exactly one scalar accumulator per chain and reads inside the
    # body observe the previous iteration's value.
    previous_max = _bind(
        tir.Constant(value=-1e38, dtype="f32", explicit_scalar=True), name("mprev"), scalar
    )
    online.append(previous_max.defining_op)
    running_max = emit(
        tir.BinOp(op="max", lhs=previous_max, rhs=chunk_max), max_reduce.result.name, scalar
    )

    shift = emit(tir.BinOp(op="sub", lhs=previous_max, rhs=running_max), name("mdiff"), scalar)
    correction = emit(tir.Unary(op="exp", operand=shift), name("corr"), scalar)

    centered = emit(
        tir.BinOp(op="sub", lhs=chunk_load.result, rhs=running_max), name("shift"), tile
    )
    weights = emit(tir.Unary(op="exp", operand=centered), name("cexp"), tile)
    chunk_sum = emit(tir.Reduce(op="sum", operand=weights), name("csum"), scalar)

    previous_sum = _bind(
        tir.Constant(value=0.0, dtype="f32", explicit_scalar=True), name("sprev"), scalar
    )
    online.append(previous_sum.defining_op)
    rescaled = emit(
        tir.BinOp(op="mul", lhs=previous_sum, rhs=correction), name("srescale"), scalar
    )
    emit(tir.BinOp(op="add", lhs=rescaled, rhs=chunk_sum), sum_reduce.result.name, scalar)

    fused_body.extend(online)
    fused_loop = tir.ForRange(
        start=max_loop.start,
        end=max_loop.end,
        step=max_loop.step,
        iv=copy.deepcopy(max_loop.iv),
        body=fused_body,
        num_stages=max_loop.num_stages,
        # The monoid's identity is (-inf, 0, 0), and -inf supplies all of it here: it is
        # the identity of the maximum, and exp(-inf - m) = 0 is the identity of the
        # normalizer. Masked lanes therefore cannot change either reduction, so the
        # ragged tail can predicate the load instead of branching around the barrier.
        masked_identity=-1e38,
    )

    # The running maximum and normalizer carry the names the post-loop reductions used,
    # so the normalize loop resolves them to the loop accumulators. Both reductions and
    # the whole sum loop then drop out. Keeping the reductions would re-reduce a value
    # that is already threadgroup-wide and inflate the normalizer by the thread count.
    discarded = {id(max_reduce), id(sum_reduce), id(pattern.sum_loop)}
    first = pattern.loop_indices[0]
    new_ops = list(func.ops[:first])
    new_ops.append(fused_loop)
    new_ops.extend(op for op in func.ops[first + 1 :] if id(op) not in discarded)

    func.ops = new_ops
    func._online_softmax = True
    return func


class _NameGen:
    def __init__(self):
        self._count = 0

    def __call__(self, prefix):
        self._count += 1
        return f"_osm_{prefix}_{self._count}"


def _bind(op, name, result_type):
    """Assign a result Value to an op and return the Value."""
    value = tir.Value(name, result_type)
    value.defining_op = op
    op.result = value
    return value


def _find_load(body):
    for op in body:
        if isinstance(op, tir.Load) and op.result is not None:
            return op
    return None


def _find_binop(body, op_name):
    for op in body:
        if isinstance(op, tir.BinOp) and op.op == op_name:
            return op
    return None


def _detect_softmax_pattern(ops):
    """Detect the 3-loop softmax pattern.

    Returns (max_loop, sum_loop, norm_loop, max_reduce, sum_reduce, loop_indices)
    or None.
    """
    for_ranges = []
    reduces = []
    for_indices = []

    for index, op in enumerate(ops):
        if isinstance(op, tir.ForRange):
            for_ranges.append(op)
            for_indices.append(index)
        elif isinstance(op, tir.Reduce):
            reduces.append(op)

    if len(for_ranges) != 3 or len(reduces) < 2:
        return None

    max_reduce = sum_reduce = None
    for reduce_op in reduces:
        if reduce_op.op == "max" and max_reduce is None:
            max_reduce = reduce_op
        elif reduce_op.op == "sum" and sum_reduce is None:
            sum_reduce = reduce_op

    if max_reduce is None or sum_reduce is None:
        return None

    max_reduce_index = ops.index(max_reduce)
    sum_reduce_index = ops.index(sum_reduce)

    if not (for_indices[0] < max_reduce_index < for_indices[1]):
        return None
    if not (for_indices[1] < sum_reduce_index < for_indices[2]):
        return None

    max_loop, sum_loop, norm_loop = for_ranges
    if max_loop.end.name != sum_loop.end.name or sum_loop.end.name != norm_loop.end.name:
        return None
    if max_loop.step != sum_loop.step or sum_loop.step != norm_loop.step:
        return None

    if not any(isinstance(op, tir.BinOp) and op.op == "max" for op in max_loop.body):
        return None

    has_exp = any(isinstance(op, tir.Unary) and op.op == "exp" for op in sum_loop.body)
    has_add = any(isinstance(op, tir.BinOp) and op.op == "add" for op in sum_loop.body)
    if not (has_exp and has_add):
        return None

    return max_loop, sum_loop, norm_loop, max_reduce, sum_reduce, for_indices
