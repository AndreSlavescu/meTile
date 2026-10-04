"""Execute only explicit native attention graph operators.

No softmax-pattern rewrite, MLX adapter, model identification, or hidden
recurrent cache is performed. Each recurrent node consumes an initial state
and produces a distinct final state. Explicit reverse-mode VJPs cover exactly
the four supported operators, including all shared-input contributions.

NumPy inputs are snapshotted when recording a tape. Device inputs, saved state,
and output buffers must remain unchanged until backward finishes; Buffer has
no mutation version counter. The graph structure/metadata is checked against
the tape. Backward never writes caller inputs, seeds, or saved buffers.
Gradients are FP32, treating floating casts as real-valued conversions rather
than differentiating rounding. Backend imports are deferred until execution.
"""

from dataclasses import dataclass
from math import prod

import numpy as np

from metile.ir.attention_graph import validate_attention_node
from metile.ir.graph_ir import ComputeGraph
from metile.runtime.buffer import MtileBuffer

_DTYPES = {
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "bool": np.dtype(np.bool_),
    "u8": np.dtype(np.uint8),
}


def _validate_graph(graph):
    if not isinstance(graph, ComputeGraph):
        raise TypeError("native attention compilation requires a ComputeGraph")
    graph.__post_init__()
    for node in graph.nodes:
        validate_attention_node(node)


def _validate_value(value, actual):
    if not isinstance(actual, (np.ndarray, MtileBuffer)):
        raise TypeError(f"graph value {value.name} must be a NumPy array or meTile Buffer")
    if tuple(actual.shape) != value.spec.shape or np.dtype(actual.dtype) != _DTYPES.get(
        value.spec.dtype
    ):
        raise ValueError(f"graph value {value.name} expects {value.spec.shape}/{value.spec.dtype}")
    if isinstance(actual, np.ndarray) and not actual.flags.c_contiguous:
        raise ValueError(f"graph value {value.name} must be C-contiguous")
    if isinstance(actual, MtileBuffer) and actual.dtype == np.dtype(np.bool_):
        raise ValueError("device graph masks must use uint8 rather than bool")


def _execute_node(node, environment, *, save_context):
    inputs = [environment[value] for value in node.inputs]
    original_inputs = tuple(inputs)
    options = dict(node.attrs)
    if node.op in {"stable_attention", "dual_chunk_attention"}:
        has_mask = options.pop("has_mask")
        options["mask"] = inputs.pop() if has_mask else None
    if node.op == "stable_attention":
        from metile.backends.attention import attention_forward

        output, context = attention_forward(*inputs, **options)
        outputs = (output,)
    elif node.op in {"gated_delta_attention", "kimi_delta_attention"}:
        from metile.backends.gated_delta import gated_delta_forward

        result = gated_delta_forward(*inputs, save_states=save_context, **options)
        context = result
        outputs = (result.output, result.final_state)
    elif node.op == "dual_chunk_attention":
        from metile.backends.dual_chunk_attention import dual_chunk_attention_forward

        result = dual_chunk_attention_forward(*inputs, **options)
        context = result
        outputs = (result.output,)
    else:
        raise ValueError(f"unsupported native attention graph operation: {node.op}")
    for value, actual in zip(node.outputs, outputs, strict=True):
        _validate_value(value, actual)
        environment[value] = actual
    return _NodeContext(node, original_inputs, context) if save_context else None


@dataclass(frozen=True)
class _NodeContext:
    node: object
    inputs: tuple
    saved: object


@dataclass(frozen=True)
class NativeAttentionGraphTape:
    executable: object
    signature: tuple
    inputs: tuple
    nodes: tuple


def _signature(graph):
    def value_signature(value):
        return id(value), value.name, value.spec, value.output_index

    return (
        tuple(value_signature(value) for value in graph.inputs),
        tuple(
            (
                id(node),
                node.op,
                tuple(id(value) for value in node.inputs),
                tuple(sorted(node.attrs.items())),
                node.side_effect,
                tuple(value_signature(value) for value in node.outputs),
            )
            for node in graph.nodes
        ),
        tuple(id(value) for value in graph.outputs),
    )


def _validate_seed(value, seed):
    if seed is None:
        return
    if value.spec.dtype not in {"f16", "f32"}:
        raise ValueError(f"nondifferentiable graph output {value.name} requires a None cotangent")
    if not isinstance(seed, (np.ndarray, MtileBuffer)):
        raise TypeError("output cotangents must be NumPy arrays, Buffers, or None")
    if tuple(seed.shape) != value.spec.shape or np.dtype(seed.dtype) not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
    ):
        raise ValueError(f"cotangent for {value.name} requires {value.spec.shape} and FP16/FP32")
    if isinstance(seed, np.ndarray) and not seed.flags.c_contiguous:
        raise ValueError("output cotangents must be C-contiguous")


def _add(left, right=None):
    from metile_kernels.attention_graph import attention_graph_gradient_add

    output = MtileBuffer.empty(left.shape)
    elements = prod(left.shape)
    attention_graph_gradient_add[((elements + 255) // 256,)](
        left,
        left if right is None else right,
        output,
        elements,
        ADD_RIGHT=right is not None,
        BLOCK=256,
        STRICT_MATH=True,
    )
    return output


def _fp32(value):
    if isinstance(value, np.ndarray):
        return MtileBuffer.from_numpy(value.astype(np.float32))
    return value if value.dtype == np.dtype(np.float32) else _add(value)


def _accumulate(gradients, value, contribution):
    if contribution is None:
        return
    contribution = _fp32(contribution)
    previous = gradients.get(value)
    gradients[value] = contribution if previous is None else _add(previous, contribution)


def _reverse_node(record, output_gradients):
    node = record.node
    if node.op == "stable_attention":
        from metile.backends.attention import attention_backward

        contributions = attention_backward(record.saved, output_gradients[0])
    elif node.op == "dual_chunk_attention":
        from metile.backends.dual_chunk_attention import dual_chunk_attention_backward

        result = dual_chunk_attention_backward(
            record.saved, output_gradients[0], max_workspace_bytes=node.attrs["max_workspace_bytes"]
        )
        contributions = (
            result.query_intra,
            result.query_successive,
            result.query_inter,
            result.key,
            result.value,
        )
    else:
        from metile.backends.gated_delta import gated_delta_backward

        result = gated_delta_backward(
            *record.inputs[:5], record.saved.states, *output_gradients, **node.attrs
        )
        contributions = (
            result.query,
            result.key,
            result.value,
            result.log_decay,
            result.beta,
            result.initial_state,
        )
    if node.op in {"stable_attention", "dual_chunk_attention"} and node.attrs["has_mask"]:
        contributions = (*contributions, None)
    return contributions


@dataclass(frozen=True)
class NativeAttentionGraphExecutable:
    graph: ComputeGraph

    def __call__(self, *inputs):
        return self._forward(inputs, save_context=False)[0]

    def forward_with_context(self, *inputs):
        """Return ``(outputs, tape)``; recurrent intermediates are saved for VJPs."""
        return self._forward(inputs, save_context=True)

    def _forward(self, inputs, *, save_context):
        _validate_graph(self.graph)
        if len(inputs) != len(self.graph.inputs):
            raise ValueError(
                f"expected {len(self.graph.inputs)} graph inputs, received {len(inputs)}"
            )
        environment = {}
        for value, actual in zip(self.graph.inputs, inputs, strict=True):
            _validate_value(value, actual)
            environment[value] = actual
        if save_context:
            environment = {
                value: (
                    actual.copy()
                    if actual.dtype == np.dtype(np.bool_)
                    else MtileBuffer.from_numpy(actual)
                )
                if isinstance(actual, np.ndarray)
                else actual
                for value, actual in environment.items()
            }
        captured_inputs = tuple(environment[value] for value in self.graph.inputs)
        records = []
        for node in self.graph.nodes:
            records.append(_execute_node(node, environment, save_context=save_context))
        outputs = tuple(environment[value] for value in self.graph.outputs)
        tape = (
            NativeAttentionGraphTape(self, _signature(self.graph), captured_inputs, tuple(records))
            if save_context
            else None
        )
        return (outputs[0] if len(outputs) == 1 else outputs), tape

    def backward(self, tape, output_cotangents=None):
        """Return FP32 input gradients in input order; masks have gradient None.

        For multiple outputs, pass a tuple/list of cotangents in output order.
        None means a zero cotangent, including an unused recurrent final state.
        Disconnected floating inputs receive explicit zeros. Tapes may be reused
        if their device inputs and saved buffers have not changed.
        """
        if not isinstance(tape, NativeAttentionGraphTape) or tape.executable is not self:
            raise TypeError("backward requires a tape recorded by this executable")
        _validate_graph(self.graph)
        if tape.signature != _signature(self.graph) or len(tape.nodes) != len(self.graph.nodes):
            raise ValueError("graph structure or metadata changed after recording the tape")
        if any(
            record.node is not node
            for record, node in zip(tape.nodes, self.graph.nodes, strict=True)
        ):
            raise ValueError("tape node contexts do not match the executable graph")
        if output_cotangents is None:
            seeds = (None,) * len(self.graph.outputs)
        elif len(self.graph.outputs) == 1 and not isinstance(output_cotangents, (tuple, list)):
            seeds = (output_cotangents,)
        elif isinstance(output_cotangents, (tuple, list)) and len(output_cotangents) == len(
            self.graph.outputs
        ):
            seeds = tuple(output_cotangents)
        else:
            raise ValueError("cotangent count must match the graph outputs")
        for value, seed in zip(self.graph.outputs, seeds, strict=True):
            _validate_seed(value, seed)
        gradients = {}
        for value, seed in zip(self.graph.outputs, seeds, strict=True):
            _accumulate(gradients, value, seed)
        for record in reversed(tape.nodes):
            if not any(value in gradients for value in record.node.outputs):
                continue
            output_gradients = [gradients.get(value) for value in record.node.outputs]
            output_gradients = [
                gradient if gradient is not None else MtileBuffer.zeros(value.spec.shape)
                for value, gradient in zip(record.node.outputs, output_gradients, strict=True)
            ]
            contributions = _reverse_node(record, output_gradients)
            for value, contribution in zip(record.node.inputs, contributions, strict=True):
                _accumulate(gradients, value, contribution)
        return tuple(
            (gradients[value] if value in gradients else MtileBuffer.zeros(value.spec.shape))
            if value.spec.dtype in {"f16", "f32"}
            else None
            for value in self.graph.inputs
        )


def compile_native_attention_graph(graph):
    """Validate explicit attention semantics and create a native-only executable."""
    _validate_graph(graph)
    return NativeAttentionGraphExecutable(graph)


__all__ = [
    "NativeAttentionGraphExecutable",
    "NativeAttentionGraphTape",
    "compile_native_attention_graph",
]
