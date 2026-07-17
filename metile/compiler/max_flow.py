from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

FlowSolver = Literal["auto", "dinic", "push_relabel"]
ResolvedFlowSolver = Literal["dinic", "push_relabel", "autotune"]
_EPSILON = 1e-12
_AUTOTUNE_ROUNDS = 3
_AUTOTUNE_SWITCH_MARGIN = 0.10
_AUTOTUNE_CACHE_LIMIT = 128
_autotune_cache = {}
_autotune_cache_lock = threading.Lock()


@dataclass(frozen=True)
class _InputEdge:
    source: int
    target: int
    capacity: float


@dataclass
class _ResidualEdge:
    target: int
    reverse: int
    capacity: float


class FlowNetwork:
    """Reusable exact s-t min-cut problem with automatic solver dispatch.

    Dinic is the low-overhead reference solver for the small sparse networks
    emitted by current fusion rules. Highest-label push-relabel participates
    when a measured tournament proves it faster for a larger graph family.
    Both consume a fresh residual graph, so a network can be solved repeatedly
    while tuning the dispatch policy.
    """

    def __init__(self):
        self._indices = {}
        self._vertices = []
        self._input_edges: list[_InputEdge] = []
        self._edge_hash = None

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_count(self) -> int:
        return len(self._input_edges)

    def add_edge(self, source, target, capacity: float):
        capacity = float(capacity)
        if capacity < 0 or not math.isfinite(capacity):
            raise ValueError("flow capacities must be finite and non-negative")
        source_index = self._index(source)
        target_index = self._index(target)
        if source_index != target_index and capacity > _EPSILON:
            self._input_edges.append(_InputEdge(source_index, target_index, capacity))
            self._edge_hash = None

    def select_solver(
        self,
        solver: FlowSolver = "auto",
        *,
        source=None,
        sink=None,
    ) -> ResolvedFlowSolver:
        """Resolve an explicit solver, static policy, or cached tournament winner."""
        if solver not in {"auto", "dinic", "push_relabel"}:
            raise ValueError(f"unknown flow solver: {solver}")
        if solver != "auto":
            return solver

        if self.vertex_count < 32:
            return "dinic"
        if source is None or sink is None:
            return "autotune"
        source_index = self._indices.get(source)
        sink_index = self._indices.get(sink)
        if source_index is None or sink_index is None:
            return "autotune"
        return _cached_solver(self._signature(source_index, sink_index)) or "autotune"

    def minimum_cut(
        self,
        source,
        sink,
        *,
        solver: FlowSolver = "auto",
    ) -> tuple[float, frozenset[object]]:
        """Return max-flow value and source-side vertices of the minimum cut."""
        if source == sink:
            raise ValueError("flow source and sink must differ")
        source_index = self._index(source)
        sink_index = self._index(sink)
        selected = self.select_solver(solver, source=source, sink=sink)
        adjacency = self._residual_graph()
        if selected == "dinic":
            total = _dinic(adjacency, source_index, sink_index)
        elif selected == "push_relabel":
            total = _push_relabel(adjacency, source_index, sink_index)
        else:
            total, adjacency = self._autotune(source_index, sink_index)

        reachable_indices = _reachable(adjacency, source_index)
        source_side = frozenset(self._vertices[index] for index in reachable_indices)
        return total, source_side

    def _index(self, vertex) -> int:
        index = self._indices.get(vertex)
        if index is None:
            index = len(self._vertices)
            self._indices[vertex] = index
            self._vertices.append(vertex)
        return index

    def _residual_graph(self) -> list[list[_ResidualEdge]]:
        adjacency: list[list[_ResidualEdge]] = [[] for _ in self._vertices]
        for input_edge in self._input_edges:
            source_edges = adjacency[input_edge.source]
            target_edges = adjacency[input_edge.target]
            forward = _ResidualEdge(input_edge.target, len(target_edges), input_edge.capacity)
            reverse = _ResidualEdge(input_edge.source, len(source_edges), 0.0)
            source_edges.append(forward)
            target_edges.append(reverse)
        return adjacency

    def _signature(self, source: int, sink: int):
        if self._edge_hash is None:
            self._edge_hash = hash(tuple(self._input_edges))
        return (
            self.vertex_count,
            self.edge_count,
            source,
            sink,
            self._edge_hash,
        )

    def _autotune(self, source: int, sink: int):
        timings = {"dinic": [], "push_relabel": []}
        results = {}
        for round_index in range(_AUTOTUNE_ROUNDS):
            solvers = ("dinic", "push_relabel")
            if round_index % 2:
                solvers = tuple(reversed(solvers))
            for solver in solvers:
                adjacency = self._residual_graph()
                start = time.perf_counter_ns()
                if solver == "dinic":
                    total = _dinic(adjacency, source, sink)
                else:
                    total = _push_relabel(adjacency, source, sink)
                timings[solver].append(time.perf_counter_ns() - start)
                results[solver] = (total, adjacency)

        dinic_time = statistics.median(timings["dinic"])
        push_relabel_time = statistics.median(timings["push_relabel"])
        if not math.isclose(
            results["dinic"][0],
            results["push_relabel"][0],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise RuntimeError("exact flow solvers produced different maximum-flow values")
        selected = "push_relabel" if push_relabel_time < 0.90 * dinic_time else "dinic"
        _cache_solver(self._signature(source, sink), selected)
        return results[selected]


def _dinic(adjacency, source: int, sink: int) -> float:
    total = 0.0
    levels = _levels(adjacency, source)
    while levels[sink] >= 0:
        cursors = [0] * len(adjacency)
        while True:
            pushed = _dinic_push(adjacency, source, sink, math.inf, levels, cursors)
            if pushed <= _EPSILON:
                break
            total += pushed
        levels = _levels(adjacency, source)
    return total


def _levels(adjacency, source: int) -> list[int]:
    levels = [-1] * len(adjacency)
    levels[source] = 0
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        for edge in adjacency[vertex]:
            if edge.capacity > _EPSILON and levels[edge.target] < 0:
                levels[edge.target] = levels[vertex] + 1
                queue.append(edge.target)
    return levels


def _dinic_push(adjacency, vertex, sink, flow, levels, cursors):
    path = []
    path_flows = [flow]
    while True:
        if vertex == sink:
            pushed = path_flows[-1]
            for parent, edge_index in path:
                edge = adjacency[parent][edge_index]
                edge.capacity -= pushed
                adjacency[edge.target][edge.reverse].capacity += pushed
            return pushed

        edges = adjacency[vertex]
        while cursors[vertex] < len(edges):
            edge = edges[cursors[vertex]]
            if edge.capacity > _EPSILON and levels[edge.target] == levels[vertex] + 1:
                path.append((vertex, cursors[vertex]))
                path_flows.append(min(path_flows[-1], edge.capacity))
                vertex = edge.target
                break
            cursors[vertex] += 1
        else:
            if not path:
                return 0.0
            vertex, _ = path.pop()
            path_flows.pop()
            cursors[vertex] += 1


def _push_relabel(adjacency, source: int, sink: int) -> float:
    vertex_count = len(adjacency)
    if vertex_count <= 1:
        return 0.0

    heights = [0] * vertex_count
    excess = [0.0] * vertex_count
    cursors = [0] * vertex_count
    heights[source] = vertex_count

    for edge in adjacency[source]:
        if edge.capacity <= _EPSILON:
            continue
        pushed = edge.capacity
        edge.capacity = 0.0
        adjacency[edge.target][edge.reverse].capacity += pushed
        excess[source] -= pushed
        excess[edge.target] += pushed

    heights = _distance_labels(adjacency, source, sink)
    bucket_count = 2 * vertex_count + 2
    buckets = [deque() for _ in range(bucket_count)]
    active = [False] * vertex_count
    height_counts = [0] * bucket_count
    for vertex, height in enumerate(heights):
        if vertex not in {source, sink}:
            height_counts[height] += 1
    highest = -1
    work = 0
    global_relabel_interval = max(1, 4 * sum(len(edges) for edges in adjacency))

    def activate(vertex: int):
        nonlocal highest
        if vertex in {source, sink} or active[vertex] or excess[vertex] <= _EPSILON:
            return
        bucket = min(heights[vertex], bucket_count - 1)
        buckets[bucket].append(vertex)
        active[vertex] = True
        highest = max(highest, bucket)

    for vertex in range(vertex_count):
        activate(vertex)

    while highest >= 0:
        while highest >= 0 and not buckets[highest]:
            highest -= 1
        if highest < 0:
            break
        vertex = buckets[highest].popleft()
        active[vertex] = False
        if heights[vertex] != highest:
            activate(vertex)
            continue

        while excess[vertex] > _EPSILON:
            edges = adjacency[vertex]
            if cursors[vertex] >= len(edges):
                residual_heights = [
                    heights[edge.target] for edge in edges if edge.capacity > _EPSILON
                ]
                if not residual_heights:
                    break
                old_height = heights[vertex]
                new_height = min(min(residual_heights) + 1, bucket_count - 1)
                heights[vertex] = new_height
                cursors[vertex] = 0
                height_counts[old_height] -= 1
                height_counts[new_height] += 1
                if old_height < vertex_count and height_counts[old_height] == 0:
                    for other in range(vertex_count):
                        other_height = heights[other]
                        if other in {source, sink} or not old_height < other_height < vertex_count:
                            continue
                        height_counts[other_height] -= 1
                        heights[other] = vertex_count + 1
                        height_counts[vertex_count + 1] += 1
                        cursors[other] = 0
                work += 1
                continue

            edge = edges[cursors[vertex]]
            if edge.capacity > _EPSILON and heights[vertex] == heights[edge.target] + 1:
                pushed = min(excess[vertex], edge.capacity)
                edge.capacity -= pushed
                adjacency[edge.target][edge.reverse].capacity += pushed
                excess[vertex] -= pushed
                excess[edge.target] += pushed
                activate(edge.target)
            else:
                cursors[vertex] += 1
                work += 1
        activate(vertex)

        if work >= global_relabel_interval:
            heights = _distance_labels(adjacency, source, sink)
            cursors = [0] * vertex_count
            buckets = [deque() for _ in range(bucket_count)]
            active = [False] * vertex_count
            height_counts = [0] * bucket_count
            for other, height in enumerate(heights):
                if other not in {source, sink}:
                    height_counts[height] += 1
            highest = -1
            for other in range(vertex_count):
                activate(other)
            work = 0

    return excess[sink]


def _distance_labels(adjacency, source: int, sink: int) -> list[int]:
    vertex_count = len(adjacency)
    unreachable = vertex_count + 1
    heights = [unreachable] * vertex_count
    heights[sink] = 0
    queue = deque([sink])
    while queue:
        vertex = queue.popleft()
        for edge in adjacency[vertex]:
            predecessor = edge.target
            reverse = adjacency[predecessor][edge.reverse]
            if reverse.capacity > _EPSILON and heights[predecessor] == unreachable:
                heights[predecessor] = heights[vertex] + 1
                queue.append(predecessor)
    heights[source] = vertex_count
    return heights


def _reachable(adjacency, source: int) -> set[int]:
    reachable = {source}
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        for edge in adjacency[vertex]:
            if edge.capacity > _EPSILON and edge.target not in reachable:
                reachable.add(edge.target)
                queue.append(edge.target)
    return reachable


def _cached_solver(signature):
    with _autotune_cache_lock:
        return _autotune_cache.get(signature)


def _cache_solver(signature, solver):
    with _autotune_cache_lock:
        if len(_autotune_cache) >= _AUTOTUNE_CACHE_LIMIT:
            _autotune_cache.pop(next(iter(_autotune_cache)))
        _autotune_cache[signature] = solver


def clear_flow_solver_cache():
    """Clear measured flow dispatch choices, primarily for benchmarking."""
    with _autotune_cache_lock:
        _autotune_cache.clear()


__all__ = ["FlowNetwork", "FlowSolver", "ResolvedFlowSolver", "clear_flow_solver_cache"]
