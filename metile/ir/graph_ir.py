from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, prod

_DTYPE_BYTES = {
    "bool": 1,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
}


@dataclass(frozen=True)
class TensorSpec:
    """Static tensor metadata used by the compute-graph optimizer."""

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor shapes must contain positive dimensions")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported graph dtype: {self.dtype}")

    @property
    def elements(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.elements * _DTYPE_BYTES[self.dtype]


@dataclass(eq=False)
class GraphValue:
    """One SSA value in a high-level compute graph."""

    name: str
    spec: TensorSpec
    producer: GraphNode | None = field(default=None, repr=False)
    output_index: int = 0


@dataclass(eq=False)
class GraphNode:
    """A pure or effectful operation over graph values."""

    name: str
    op: str
    inputs: tuple[GraphValue, ...]
    attrs: dict[str, object]
    outputs: tuple[GraphValue, ...]
    side_effect: bool = False


@dataclass
class ComputeGraph:
    """Topologically ordered high-level compute graph."""

    inputs: tuple[GraphValue, ...]
    nodes: tuple[GraphNode, ...]
    outputs: tuple[GraphValue, ...]

    def __post_init__(self):
        available = set(self.inputs)
        for node in self.nodes:
            if any(value not in available for value in node.inputs):
                raise ValueError(f"graph node {node.name} is not topologically ordered")
            if any(output.producer is not node for output in node.outputs):
                raise ValueError(f"graph node {node.name} has an invalid output producer")
            available.update(node.outputs)
        if any(value not in available for value in self.outputs):
            raise ValueError("graph output is not produced by this graph")

    def consumers(self) -> dict[GraphValue, tuple[GraphNode, ...]]:
        consumers: dict[GraphValue, list[GraphNode]] = {}
        for node in self.nodes:
            for value in node.inputs:
                consumers.setdefault(value, []).append(node)
        return {value: tuple(nodes) for value, nodes in consumers.items()}


class GraphBuilder:
    """Build a typed compute DAG without prescribing kernel boundaries."""

    def __init__(self):
        self._inputs: list[GraphValue] = []
        self._nodes: list[GraphNode] = []
        self._names: set[str] = set()

    def input(self, name: str, spec: TensorSpec) -> GraphValue:
        self._claim_name(name)
        value = GraphValue(name, spec)
        self._inputs.append(value)
        return value

    def add(self, left: GraphValue, right: GraphValue, *, name: str | None = None) -> GraphValue:
        if left.spec != right.spec:
            raise ValueError("add inputs must have identical tensor specifications")
        return self._node("add", (left, right), {}, (left.spec,), name)[0]

    def multiply(
        self,
        left: GraphValue,
        right: GraphValue,
        *,
        name: str | None = None,
    ) -> GraphValue:
        if left.spec != right.spec:
            raise ValueError("multiply inputs must have identical tensor specifications")
        return self._node("multiply", (left, right), {}, (left.spec,), name)[0]

    def silu(self, values: GraphValue, *, name: str | None = None) -> GraphValue:
        return self._node("silu", (values,), {}, (values.spec,), name)[0]

    def rms_norm(
        self,
        values: GraphValue,
        weight: GraphValue,
        eps: float,
        *,
        name: str | None = None,
    ) -> GraphValue:
        if weight.spec.shape != (values.spec.shape[-1],):
            raise ValueError("RMSNorm weight must match the final input dimension")
        if weight.spec.dtype != values.spec.dtype:
            raise ValueError("RMSNorm values and weight must have the same dtype")
        if eps <= 0:
            raise ValueError("RMSNorm epsilon must be positive")
        return self._node(
            "rms_norm",
            (values, weight),
            {"eps": float(eps)},
            (values.spec,),
            name,
        )[0]

    def matmul(
        self,
        left: GraphValue,
        right: GraphValue,
        *,
        transpose_right: bool = False,
        name: str | None = None,
    ) -> GraphValue:
        if len(left.spec.shape) < 2 or len(left.spec.shape) != len(right.spec.shape):
            raise ValueError("matmul inputs must have equal rank of at least two")
        if left.spec.shape[:-2] != right.spec.shape[:-2]:
            raise ValueError("matmul batch dimensions must match exactly")
        if left.spec.dtype != right.spec.dtype:
            raise ValueError("matmul inputs must have the same dtype")
        reduction = right.spec.shape[-1] if transpose_right else right.spec.shape[-2]
        if left.spec.shape[-1] != reduction:
            raise ValueError("matmul reduction dimensions must match")
        columns = right.spec.shape[-2] if transpose_right else right.spec.shape[-1]
        output = TensorSpec((*left.spec.shape[:-2], left.spec.shape[-2], columns), left.spec.dtype)
        return self._node(
            "matmul",
            (left, right),
            {"transpose_right": bool(transpose_right)},
            (output,),
            name,
        )[0]

    def scale(
        self,
        values: GraphValue,
        factor: float,
        *,
        name: str | None = None,
    ) -> GraphValue:
        if not isfinite(factor):
            raise ValueError("scale factor must be finite")
        return self._node("scale", (values,), {"factor": float(factor)}, (values.spec,), name)[0]

    def causal_mask(self, values: GraphValue, *, name: str | None = None) -> GraphValue:
        if len(values.spec.shape) < 2:
            raise ValueError("causal masking requires a rank-two or higher tensor")
        return self._node("causal_mask", (values,), {}, (values.spec,), name)[0]

    def softmax(
        self,
        values: GraphValue,
        axis: int = -1,
        *,
        name: str | None = None,
    ) -> GraphValue:
        rank = len(values.spec.shape)
        normalized_axis = axis + rank if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= rank:
            raise ValueError("softmax axis is outside the input rank")
        canonical_axis = normalized_axis - rank if normalized_axis == rank - 1 else normalized_axis
        return self._node(
            "softmax",
            (values,),
            {"axis": canonical_axis},
            (values.spec,),
            name,
        )[0]

    def build(self, outputs: GraphValue | tuple[GraphValue, ...]) -> ComputeGraph:
        if isinstance(outputs, GraphValue):
            outputs = (outputs,)
        if not outputs:
            raise ValueError("a compute graph must have at least one output")
        return ComputeGraph(tuple(self._inputs), tuple(self._nodes), tuple(outputs))

    def _node(
        self,
        op: str,
        inputs: tuple[GraphValue, ...],
        attrs: dict[str, object],
        output_specs: tuple[TensorSpec, ...],
        name: str | None,
    ) -> tuple[GraphValue, ...]:
        node_name = name or f"{op}_{len(self._nodes)}"
        self._claim_name(node_name)
        outputs = tuple(
            GraphValue(
                node_name if len(output_specs) == 1 else f"{node_name}_{index}",
                spec,
                output_index=index,
            )
            for index, spec in enumerate(output_specs)
        )
        node = GraphNode(node_name, op, inputs, dict(attrs), outputs)
        for output in outputs:
            output.producer = node
        self._nodes.append(node)
        return outputs

    def _claim_name(self, name: str):
        if not name or name in self._names:
            raise ValueError(f"duplicate or empty graph name: {name!r}")
        self._names.add(name)


__all__ = [
    "ComputeGraph",
    "GraphBuilder",
    "GraphNode",
    "GraphValue",
    "TensorSpec",
]
