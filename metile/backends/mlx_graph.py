from __future__ import annotations

from dataclasses import dataclass

from metile.backends.mlx import mlx_add_rms_norm
from metile.compiler.graph_fusion import FusionPlan, FusionRegion, plan_graph_fusion
from metile.ir.graph_ir import ComputeGraph, GraphNode, GraphValue

_SUPPORTED_OPS = {"add", "rms_norm"}


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
                _execute_node(node, environment)
                continue
            if region in completed_regions:
                continue
            _execute_region(region, environment, autotune=self.autotune)
            completed_regions.add(region)

        outputs = tuple(environment[value] for value in graph.outputs)
        return outputs[0] if len(outputs) == 1 else outputs


def compile_mlx_graph(graph: ComputeGraph, *, autotune: bool = True) -> MLXGraphExecutable:
    """Plan legal graph fusion and create an MLX-backed executable."""
    unsupported = sorted({node.op for node in graph.nodes if node.op not in _SUPPORTED_OPS})
    if unsupported:
        raise ValueError(f"unsupported MLX graph operations: {', '.join(unsupported)}")
    return MLXGraphExecutable(plan_graph_fusion(graph), autotune=autotune)


def _execute_node(node: GraphNode, environment):
    if node.op == "add":
        environment[node.outputs[0]] = environment[node.inputs[0]] + environment[node.inputs[1]]
    elif node.op == "rms_norm":
        import mlx.core as mx

        environment[node.outputs[0]] = mx.fast.rms_norm(
            environment[node.inputs[0]],
            environment[node.inputs[1]],
            node.attrs["eps"],
        )
    else:
        raise ValueError(f"unsupported MLX graph operation: {node.op}")


def _execute_region(region: FusionRegion, environment, *, autotune):
    if region.rule.name != "residual_add_rms_norm":
        raise ValueError(f"unsupported MLX fusion rule: {region.rule.name}")
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
