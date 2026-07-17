from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: float


class FlowNetwork:
    """Deterministic Dinic max-flow solver for small compiler graphs."""

    def __init__(self):
        self._indices = {}
        self._vertices = []
        self._edges: list[list[_Edge]] = []

    def add_edge(self, source, target, capacity: float):
        if capacity < 0:
            raise ValueError("flow capacities must be non-negative")
        source_index = self._index(source)
        target_index = self._index(target)
        forward = _Edge(target_index, len(self._edges[target_index]), float(capacity))
        reverse = _Edge(source_index, len(self._edges[source_index]), 0.0)
        self._edges[source_index].append(forward)
        self._edges[target_index].append(reverse)

    def minimum_cut(self, source, sink) -> tuple[float, frozenset[object]]:
        """Return max-flow value and source-side vertices of the minimum cut."""
        if source == sink:
            raise ValueError("flow source and sink must differ")
        source_index = self._index(source)
        sink_index = self._index(sink)
        total = 0.0
        levels = self._levels(source_index)
        while levels[sink_index] >= 0:
            cursors = [0] * len(self._vertices)
            while True:
                pushed = self._push(source_index, sink_index, float("inf"), levels, cursors)
                if pushed <= 1e-12:
                    break
                total += pushed
            levels = self._levels(source_index)

        reachable_indices = self._reachable(source_index)
        return total, frozenset(self._vertices[index] for index in reachable_indices)

    def _index(self, vertex) -> int:
        index = self._indices.get(vertex)
        if index is None:
            index = len(self._vertices)
            self._indices[vertex] = index
            self._vertices.append(vertex)
            self._edges.append([])
        return index

    def _levels(self, source: int) -> list[int]:
        levels = [-1] * len(self._vertices)
        levels[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for edge in self._edges[vertex]:
                if edge.capacity > 1e-12 and levels[edge.target] < 0:
                    levels[edge.target] = levels[vertex] + 1
                    queue.append(edge.target)
        return levels

    def _push(self, vertex, sink, flow, levels, cursors):
        if vertex == sink:
            return flow
        edges = self._edges[vertex]
        while cursors[vertex] < len(edges):
            edge = edges[cursors[vertex]]
            if edge.capacity > 1e-12 and levels[edge.target] == levels[vertex] + 1:
                pushed = self._push(
                    edge.target,
                    sink,
                    min(flow, edge.capacity),
                    levels,
                    cursors,
                )
                if pushed > 1e-12:
                    edge.capacity -= pushed
                    self._edges[edge.target][edge.reverse].capacity += pushed
                    return pushed
            cursors[vertex] += 1
        return 0.0

    def _reachable(self, source: int) -> set[int]:
        reachable = {source}
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for edge in self._edges[vertex]:
                if edge.capacity > 1e-12 and edge.target not in reachable:
                    reachable.add(edge.target)
                    queue.append(edge.target)
        return reachable


__all__ = ["FlowNetwork"]
