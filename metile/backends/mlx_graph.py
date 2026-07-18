from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from metile.backends.mlx import mlx_add_rms_norm, mlx_attention_decode
from metile.backends.mlx_attention import mlx_flash_attention
from metile.compiler.attention_discovery import discover_flash_attention
from metile.compiler.graph_fusion import FusionPlan, FusionRegion, plan_graph_fusion
from metile.ir.graph_ir import ComputeGraph, GraphNode, GraphValue

_SUPPORTED_OPS = {
    "add",
    "causal_mask",
    "flash_attention",
    "matmul",
    "multiply",
    "rms_norm",
    "scale",
    "silu",
    "softmax",
}


@dataclass(frozen=True)
class MLXGraphExecutable:
    """An MLX graph executable with measured fused-region dispatch."""

    plan: FusionPlan
    autotune: bool = True

    def __call__(self, *inputs):
        graph = self.plan.graph
        if len(inputs) != len(graph.inputs):
            raise ValueError(f"expected {len(graph.inputs)} graph inputs, received {len(inputs)}")
        environment = {}
        for value, runtime_value in zip(graph.inputs, inputs):
            _validate_runtime_value(value, runtime_value)
            environment[value] = runtime_value

        completed_regions = set()
        for node in graph.nodes:
            region = self.plan.region_for(node)
            if region is None:
                _execute_node(node, environment, autotune=self.autotune)
                continue
            if region in completed_regions:
                continue
            _execute_region(region, environment, autotune=self.autotune)
            completed_regions.add(region)

        outputs = tuple(environment[value] for value in graph.outputs)
        return outputs[0] if len(outputs) == 1 else outputs


def compile_mlx_graph(graph: ComputeGraph, *, autotune: bool = True) -> MLXGraphExecutable:
    """Plan legal graph fusion and create an MLX-backed executable."""
    graph = discover_flash_attention(graph)
    unsupported = sorted({node.op for node in graph.nodes if node.op not in _SUPPORTED_OPS})
    if unsupported:
        raise ValueError(f"unsupported MLX graph operations: {', '.join(unsupported)}")
    return MLXGraphExecutable(plan_graph_fusion(graph), autotune=autotune)


def _execute_node(node: GraphNode, environment, *, autotune):
    if node.op == "add":
        environment[node.outputs[0]] = environment[node.inputs[0]] + environment[node.inputs[1]]
    elif node.op == "multiply":
        environment[node.outputs[0]] = environment[node.inputs[0]] * environment[node.inputs[1]]
    elif node.op == "matmul":
        import mlx.core as mx

        right = environment[node.inputs[1]]
        if node.attrs.get("transpose_right", False):
            right = mx.swapaxes(right, -1, -2)
        environment[node.outputs[0]] = mx.matmul(environment[node.inputs[0]], right)
    elif node.op == "scale":
        environment[node.outputs[0]] = environment[node.inputs[0]] * node.attrs["factor"]
    elif node.op == "silu":
        import mlx.nn as nn

        environment[node.outputs[0]] = nn.silu(environment[node.inputs[0]])
    elif node.op == "causal_mask":
        environment[node.outputs[0]] = _causal_mask(environment[node.inputs[0]])
    elif node.op == "softmax":
        import mlx.core as mx

        environment[node.outputs[0]] = mx.softmax(
            environment[node.inputs[0]],
            axis=node.attrs["axis"],
        )
    elif node.op == "flash_attention":
        certificate = node.attrs["reduction_certificate"]
        if not certificate.verified:
            raise ValueError("refusing to execute an unverified attention rewrite")
        query, key, value = (environment[input_value] for input_value in node.inputs)
        if query.shape[-2] == 1 and not node.attrs["causal"]:
            result = mlx_attention_decode(
                query,
                key,
                value,
                scale=node.attrs["scale"],
                autotune=autotune,
            )
        else:
            result = mlx_flash_attention(
                query,
                key,
                value,
                scale=node.attrs["scale"],
                causal=node.attrs["causal"],
                autotune=autotune,
            )
        environment[node.outputs[0]] = result
    elif node.op == "rms_norm":
        import mlx.core as mx

        environment[node.outputs[0]] = mx.fast.rms_norm(
            environment[node.inputs[0]],
            environment[node.inputs[1]],
            node.attrs["eps"],
        )
    else:
        raise ValueError(f"unsupported MLX graph operation: {node.op}")


def _causal_mask(values):
    import mlx.core as mx

    rows, columns = values.shape[-2:]
    row = mx.arange(rows)[:, None]
    column = mx.arange(columns)[None, :]
    allowed = column <= row + columns - rows
    return mx.where(allowed, values, mx.array(float("-inf"), dtype=values.dtype))


def _execute_region(region: FusionRegion, environment, *, autotune):
    if region.rule.name == "residual_add_rms_norm":
        add_node, rms_node = region.nodes
        summed, normalized = mlx_add_rms_norm(
            environment[add_node.inputs[0]],
            environment[add_node.inputs[1]],
            environment[rms_node.inputs[1]],
            rms_node.attrs["eps"],
            autotune=autotune,
        )
        environment[add_node.outputs[0]] = summed
        environment[rms_node.outputs[0]] = normalized
        return
    if region.rule.name == "parallel_matmul_swiglu_down":
        gate_node, up_node, activation_node, combine_node, down_node = region.nodes
        transpose = tuple(
            bool(node.attrs.get("transpose_right", False))
            for node in (gate_node, up_node, down_node)
        )
        operation = _compiled_swiglu_mlp(transpose)
        result = operation(
            environment[gate_node.inputs[0]],
            environment[gate_node.inputs[1]],
            environment[up_node.inputs[1]],
            environment[down_node.inputs[1]],
        )
        environment[gate_node.outputs[0]] = None
        environment[up_node.outputs[0]] = None
        environment[activation_node.outputs[0]] = None
        environment[combine_node.outputs[0]] = None
        environment[down_node.outputs[0]] = result
        return
    raise ValueError(f"unsupported MLX fusion rule: {region.rule.name}")


@lru_cache(maxsize=8)
def _compiled_swiglu_mlp(transpose: tuple[bool, bool, bool]):
    import mlx.core as mx
    import mlx.nn as nn

    def operation(values, gate_weight, up_weight, down_weight):
        weights = (gate_weight, up_weight, down_weight)
        right = tuple(
            mx.swapaxes(weight, -1, -2) if transpose_right else weight
            for weight, transpose_right in zip(weights, transpose, strict=True)
        )
        gate = mx.matmul(values, right[0])
        up = mx.matmul(values, right[1])
        return mx.matmul(nn.silu(gate) * up, right[2])

    return mx.compile(operation)


def _validate_runtime_value(value: GraphValue, runtime_value):
    dtype = {
        "mlx.core.float16": "f16",
        "mlx.core.float32": "f32",
    }.get(str(runtime_value.dtype))
    if tuple(runtime_value.shape) != value.spec.shape or dtype != value.spec.dtype:
        raise ValueError(
            f"graph input {value.name} expects {value.spec.shape}/{value.spec.dtype}, "
            f"received {tuple(runtime_value.shape)}/{runtime_value.dtype}"
        )


__all__ = ["MLXGraphExecutable", "compile_mlx_graph"]
