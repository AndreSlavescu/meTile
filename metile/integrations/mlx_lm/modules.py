"""modules layer of the mlx_lm integration."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

from metile.backends.mlx_affine import (
    MLXAffineWeight,
    mlx_affine_matmul,
)
from metile.backends.mlx_dense import (
    MLXDenseWeight,
)
from metile.backends.mlx_dense_residual import (
    mlx_dense_residual_qmv,
)
from metile.backends.mlx_dense_swiglu import (
    mlx_dense_swiglu,
    mlx_dense_swiglu_projected,
)
from metile.backends.mlx_graph import compile_mlx_graph
from metile.backends.mlx_quantized import (
    mlx_affine_mlp_executor,
    mlx_affine_swiglu,
)
from metile.integrations.mlx_lm._state import (
    _graph_executable_cache,
    _quantized_mlp_executor_cache,
)
from metile.integrations.mlx_lm.core import (
    _model_layers,
    _supports_metile_rms_norm,
    _tensor_spec,
)
from metile.ir.graph_ir import GraphBuilder


@dataclass
class MLXAffinePrefill:
    """AOT-repacked affine projections for one MLX-LM model."""

    model: object
    weights: dict[int, tuple[object, MLXAffineWeight]]
    min_rows: int = 32
    patched_classes: dict[int, type] = field(default_factory=dict)

    def weight_for(self, module):
        entry = self.weights.get(id(module))
        return entry[1] if entry is not None and entry[0] is module else None

    @property
    def projection_count(self):
        return len(self.weights)

    def patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(module)
        original_call = original_class.__call__
        weight = self.weight_for(module)
        min_rows = self.min_rows

        class MLXAffinePrefillLinear(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows < min_rows:
                    object.__setattr__(self, "__class__", original_class)
                    return original_call(self, values)
                output = mlx_affine_matmul(values, weight)
                if "bias" in self:
                    output = output + self["bias"]
                return output

        MLXAffinePrefillLinear.__name__ = f"MeTile{original_class.__name__}"
        MLXAffinePrefillLinear.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXAffinePrefillLinear
        return MLXAffinePrefillLinear


@dataclass
class MLXDenseMLP:
    """AOT K-major and optional interleaved weights for dense SwiGLU blocks."""

    model: object
    weights: dict[int, tuple[object, MLXDenseWeight, MLXDenseWeight, object]]
    min_rows: int = 1
    repack_bytes: int = 0
    implementation: str = "projected"
    patched_classes: dict[int, type] = field(default_factory=dict)
    paired_weights: dict[int, tuple[object, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.implementation not in {"fused", "native", "projected"}:
            raise ValueError("dense MLP implementation must be fused, projected, or native")

    def weights_for(self, module):
        entry = self.weights.get(id(module))
        return entry[1:] if entry is not None and entry[0] is module else None

    def paired_weight_for(self, module):
        entry = self.paired_weights.get(id(module))
        return entry[1] if entry is not None and entry[0] is module else None

    @property
    def mlp_count(self):
        return len(self.weights)

    def patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(module)
        original_call = original_class.__call__
        gate_weight, up_weight, _ = self.weights_for(module)
        paired_weight = self.paired_weight_for(module)
        min_rows = self.min_rows
        prepared = self

        class MLXDenseMLPBlock(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows < min_rows:
                    object.__setattr__(self, "__class__", original_class)
                    return original_call(self, values)
                if prepared.implementation == "native":
                    return original_call(self, values)
                if prepared.implementation == "fused":
                    hidden = (
                        mlx_dense_swiglu(values, gate_weight, up_weight)
                        if paired_weight is None
                        else mlx_dense_swiglu(
                            values,
                            gate_weight,
                            up_weight,
                            paired_weight=paired_weight,
                        )
                    )
                else:
                    hidden = mlx_dense_swiglu_projected(values, gate_weight, up_weight)
                return self.down_proj(hidden)

        MLXDenseMLPBlock.__name__ = f"MeTile{original_class.__name__}"
        MLXDenseMLPBlock.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXDenseMLPBlock
        return MLXDenseMLPBlock


def _select_model_affine8_group(total_layers, layer_counts, timings, native_timing):
    if total_layers < 1:
        raise ValueError("model affine8 group selection requires at least one layer")
    groups = tuple(sorted(layer_counts))
    if not groups or set(groups) != set(timings):
        raise ValueError("model affine8 group candidates must have matching timings")
    if native_timing <= 0 or any(
        count < 0 or count > total_layers for count in layer_counts.values()
    ):
        raise ValueError("model affine8 group measurements must be positive and bounded")
    estimates = {
        group: layer_counts[group] * timings[group]
        + (total_layers - layer_counts[group]) * native_timing
        for group in groups
    }
    selected = min(groups, key=lambda group: (estimates[group], -layer_counts[group], group))
    return selected, estimates


def _supports_metile_residual_rms_norm(values, residual, norm):
    weight = norm["weight"]
    return (
        values.shape == residual.shape
        and values.dtype == residual.dtype
        and _supports_metile_rms_norm(values, weight)
    )


def _execute_residual_rms_graph(values, residual, norm):
    weight = norm["weight"]
    key = (tuple(values.shape), str(values.dtype), float(norm.eps))
    executable = _graph_executable_cache.get(key)
    if executable is None:
        builder = GraphBuilder()
        values_input = builder.input("values", _tensor_spec(values))
        residual_input = builder.input("residual", _tensor_spec(residual))
        weight_input = builder.input("weight", _tensor_spec(weight))
        summed = builder.add(values_input, residual_input, name="residual_add")
        normalized = builder.rms_norm(
            summed,
            weight_input,
            norm.eps,
            name="post_attention_rms_norm",
        )
        executable = compile_mlx_graph(builder.build((summed, normalized)))
        _graph_executable_cache[key] = executable
    return executable(values, residual, weight)


def _supports_dense_residual_mlp(module, values, residual, dense_mlp):
    weights = dense_mlp.weights_for(module) if dense_mlp is not None else None
    if weights is None:
        return False
    gate_weight, _, down_weight = weights
    rows = values.size // values.shape[-1]
    return (
        rows == 1
        and values.shape[-1] == gate_weight.shape[0]
        and residual.shape == (*values.shape[:-1], down_weight.shape[0])
        and values.dtype == down_weight.dtype
        and residual.dtype == values.dtype
    )


def _execute_dense_swiglu(module, values, dense_mlp, use_generated_swiglu):
    if not use_generated_swiglu:
        import mlx.nn as nn

        return nn.silu(module.gate_proj(values)) * module.up_proj(values)
    gate_weight, up_weight, _ = dense_mlp.weights_for(module)
    paired_weight = dense_mlp.paired_weight_for(module)
    if dense_mlp.implementation == "fused":
        return mlx_dense_swiglu(
            values,
            gate_weight,
            up_weight,
            paired_weight=paired_weight,
        )
    return mlx_dense_swiglu_projected(values, gate_weight, up_weight)


def _execute_dense_mlp(module, values, residual, dense_mlp, use_generated_swiglu=True):
    hidden = _execute_dense_swiglu(module, values, dense_mlp, use_generated_swiglu)
    down_weight = dense_mlp.weights_for(module)[2]
    return mlx_dense_residual_qmv(hidden, down_weight, residual)


def _supports_quantized_mlp(module, values, quantized_linear):
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    return (
        isinstance(gate, quantized_linear)
        and isinstance(up, quantized_linear)
        and gate.mode == up.mode == "affine"
        and gate.group_size == up.group_size == 64
        and gate.bits == up.bits == 4
        and gate.get("biases") is not None
        and up.get("biases") is not None
        and "bias" not in gate
        and "bias" not in up
        and str(values.dtype) == "mlx.core.float16"
    )


def _supports_quantized_residual_mlp(module, values, residual, quantized_linear):
    down = getattr(module, "down_proj", None)
    return (
        _supports_quantized_mlp(module, values, quantized_linear)
        and isinstance(down, quantized_linear)
        and down.mode == "affine"
        and down.group_size == 64
        and down.bits == 4
        and down.get("biases") is not None
        and "bias" not in down
        and residual.shape == (*values.shape[:-1], down.weight.shape[0])
        and residual.dtype == values.dtype
    )


def _execute_quantized_mlp(module, values, residual=None):
    gate = module.gate_proj
    up = module.up_proj
    if residual is None:
        hidden = mlx_affine_swiglu(
            values,
            gate["weight"],
            gate["scales"],
            gate.get("biases"),
            up["weight"],
            up["scales"],
            up.get("biases"),
            group_size=gate.group_size,
            bits=gate.bits,
        )
        return module.down_proj(hidden)
    down = module.down_proj
    cache_key = id(module)
    cached = _quantized_mlp_executor_cache.get(cache_key)
    if cached is None or cached[0]() is not module:
        executor = mlx_affine_mlp_executor(
            values,
            gate["weight"],
            gate["scales"],
            gate.get("biases"),
            up["weight"],
            up["scales"],
            up.get("biases"),
            down["weight"],
            down["scales"],
            down.get("biases"),
            residual,
            group_size=down.group_size,
            bits=down.bits,
        )

        def discard(reference, key=cache_key):
            if _quantized_mlp_executor_cache.get(key, (None,))[0] is reference:
                del _quantized_mlp_executor_cache[key]

        try:
            module_reference = weakref.ref(module, discard)
        except TypeError:

            def module_reference():
                return module

        cached = (module_reference, executor)
        _quantized_mlp_executor_cache[cache_key] = cached
    return cached[1](values, residual)


def prepare_mlx_lm_affine_prefill(
    model,
    *,
    projections=("down_proj",),
    min_rows=32,
):
    """AOT-repack exact affine weights for generated prefill matmuls."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if not projections or not all(isinstance(name, str) and name for name in projections):
        raise ValueError("projections must contain at least one attribute name")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    try:
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Affine prefill preparation requires the optional 'mlx' package"
        ) from error

    weights = {}
    for layer in _model_layers(model):
        mlp = getattr(layer, "mlp", None)
        for name in projections:
            module = getattr(mlp, name, None)
            if (
                not isinstance(module, nn.QuantizedLinear)
                or module.mode != "affine"
                or module.group_size != 64
                or module.bits != 4
                or module.get("biases") is None
                or module.weight.shape[0] % 32
            ):
                continue
            weight = MLXAffineWeight.from_mlx(
                module.weight,
                module.scales,
                module.biases,
                group_size=module.group_size,
                bits=module.bits,
            )
            weights[id(module)] = (module, weight)
    if not weights:
        raise ValueError("model contains no supported affine prefill projections")
    return MLXAffinePrefill(model, weights, min_rows)


def prepare_mlx_lm_dense_mlp(
    model,
    *,
    min_rows=1,
    max_working_set_fraction=0.8,
):
    """AOT-prepare dense gate/up layouts for generated prefill and decode."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    if not 0.0 < max_working_set_fraction <= 1.0:
        raise ValueError("max_working_set_fraction must be in (0, 1]")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError("Dense MLP preparation requires the optional 'mlx' package") from error

    supported = []
    for layer in _model_layers(model):
        module = getattr(layer, "mlp", None)
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        down = getattr(module, "down_proj", None)
        if (
            not isinstance(gate, nn.Linear)
            or not isinstance(up, nn.Linear)
            or not isinstance(down, nn.Linear)
            or "bias" in gate
            or "bias" in up
            or "bias" in down
            or gate.weight.shape != up.weight.shape
            or down.weight.shape != (gate.weight.shape[1], gate.weight.shape[0])
            or gate.weight.shape[0] % 64
            or gate.weight.shape[1] % 32
            or str(gate.weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16")
            or gate.weight.dtype != up.weight.dtype
            or gate.weight.dtype != down.weight.dtype
        ):
            continue
        supported.append((module, gate.weight, up.weight, down.weight))
    if not supported:
        raise ValueError("model contains no supported dense SwiGLU blocks")
    repack_bytes = sum(gate.nbytes + up.nbytes for _, gate, up, _ in supported)
    recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    budget = int(recommended * max_working_set_fraction)
    active = int(mx.get_active_memory())
    if recommended and active + repack_bytes > budget:
        raise ValueError(
            f"dense AOT repack needs {repack_bytes / 2**30:.2f} GiB with "
            f"{active / 2**30:.2f} GiB active, exceeding the "
            f"{budget / 2**30:.2f} GiB working-set budget"
        )

    paired_bytes = repack_bytes if not recommended or active + 2 * repack_bytes <= budget else 0
    weights = {}
    paired_weights = {}
    for module, gate, up, down in supported:
        weights[id(module)] = (
            module,
            MLXDenseWeight.from_mlx(gate),
            MLXDenseWeight.from_mlx(up),
            down,
        )
        if paired_bytes:
            paired = mx.stack((gate, up), axis=-1)
            mx.eval(paired)
            paired_weights[id(module)] = (module, paired)
    return MLXDenseMLP(
        model,
        weights,
        min_rows,
        repack_bytes=repack_bytes + paired_bytes,
        paired_weights=paired_weights,
    )
