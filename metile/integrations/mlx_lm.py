from __future__ import annotations

import copy
import gc
import inspect
import os
import statistics
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from itertools import combinations

from metile.backends.mlx import (
    _mlx_attention_decode_unchecked,
    mlx_add_rms_norm_dispatches,
    mlx_add_rms_norm_selection,
    mlx_attention_dispatches,
    mlx_rms_norm,
    mlx_rms_norm_dispatches,
)
from metile.backends.mlx_affine import (
    MLXAffineWeight,
    mlx_affine_backend_signature,
    mlx_affine_matmul,
    mlx_affine_matmul_dispatches,
)
from metile.backends.mlx_compressed_down import (
    MLXCompressedDownWeight,
    mlx_compressed_down_backend_signature,
    tune_mlx_affine8_group_size,
)
from metile.backends.mlx_dense import (
    MLXDenseWeight,
    mlx_dense_backend_signature,
    mlx_dense_matmul_dispatches,
)
from metile.backends.mlx_dense_residual import (
    mlx_dense_residual_backend_signature,
    mlx_dense_residual_dispatches,
    mlx_dense_residual_qmv,
)
from metile.backends.mlx_dense_swiglu import (
    mlx_dense_swiglu,
    mlx_dense_swiglu_backend_signature,
    mlx_dense_swiglu_dispatches,
    mlx_dense_swiglu_projected,
)
from metile.backends.mlx_graph import compile_mlx_graph
from metile.backends.mlx_quantized import (
    mlx_affine_mlp_executor,
    mlx_affine_residual_qmv_dispatches,
    mlx_affine_swiglu,
    mlx_affine_swiglu_backend_signature,
    mlx_affine_swiglu_dispatches,
    mlx_affine_swiglu_executor,
)
from metile.compiler.schedule_search import choose_mdl_tie
from metile.ir.graph_ir import GraphBuilder, TensorSpec
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest

_graph_executable_cache = {}
_quantized_mlp_executor_cache = {}
_mlx_lm_plan_cache = {}
_mlx_lm_plan_lock = threading.RLock()
_compressed_down_calibration_lock = threading.RLock()
_compressed_gate_up_calibration_lock = threading.RLock()
_compressed_gate_up_group_lock = threading.RLock()
_compressed_gate_up_implementation_lock = threading.RLock()
_compressed_attention_calibration_lock = threading.RLock()
_compressed_attention_group_lock = threading.RLock()
_compressed_vocab_calibration_lock = threading.RLock()
_compressed_down_calibration_cache_path = cache_root() / "mlx-compressed-down-calibration-v4.json"
_compressed_gate_up_calibration_cache_path = (
    cache_root() / "mlx-compressed-gate-up-calibration-v4.json"
)
_compressed_gate_up_group_cache_path = cache_root() / "mlx-compressed-gate-up-group-v1.json"
_compressed_gate_up_implementation_cache_path = (
    cache_root() / "mlx-compressed-gate-up-implementation-v1.json"
)
_compressed_attention_calibration_cache_path = (
    cache_root() / "mlx-compressed-attention-calibration-v2.json"
)
_compressed_attention_group_cache_path = cache_root() / "mlx-compressed-attention-group-v1.json"
_compressed_vocab_calibration_cache_path = cache_root() / "mlx-compressed-vocab-calibration-v2.json"
_mlx_lm_plan_cache_path = cache_root() / "mlx-lm-plan-autotune-v47.json"
_MODEL_SWITCH_MARGIN = 0.01
_MODEL_DECODE_SWITCH_MARGIN = 0.01
_MODEL_STRONG_DECODE_SWITCH_MARGIN = 0.05
_MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN = 0.015
_MODEL_TTFT_SWITCH_MARGIN = 0.02
_MODEL_REGRESSION_MARGIN = 0.005
_MODEL_VALIDATION_MIN_DECODE_STEPS = 32
_MODEL_VALIDATION_MIN_TRIALS = 7
_MODEL_VALIDATION_ATTEMPTS = 3
_MODEL_VALIDATION_MAX_SURVIVORS = 3
_MODEL_VALIDATION_SCREEN_TRIALS = 3
_MODEL_SCREEN_MAX_FINALISTS = 8
_MODEL_SCREEN_RELATIVE_MARGIN = 0.08
_MODEL_SCREEN_ROUNDS = 1
_MODEL_PROVISIONAL_MAX_FINALISTS = 10
_MODEL_PROVISIONAL_RELATIVE_MARGIN = 0.03
_MODEL_PROVISIONAL_ROUNDS = 3
_MODEL_SEARCH_MIN_DECODE_STEPS = 16
_MODEL_KL_LIMIT = 1e-3
_MODEL_MEAN_LOGIT_ERROR_LIMIT = 0.02
_MODEL_MAX_LOGIT_ERROR_LIMIT = 0.25
_MODEL_BF16_MEAN_LOGIT_ERROR_LIMIT = 0.04
_MODEL_BF16_MAX_LOGIT_ERROR_LIMIT = 0.5
_COMPRESSED_AFFINE8_KL_LIMIT = 1e-3
_COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT = 0.05
_COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT = 0.5
_COMPRESSED_APPROXIMATE_KL_LIMIT = 0.05
_COMPRESSED_APPROXIMATE_MEAN_LOGIT_ERROR_LIMIT = 0.3
_COMPRESSED_APPROXIMATE_MAX_LOGIT_ERROR_LIMIT = 2.0
_COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS = 8
_COMPRESSED_GATE_UP_FUSION_MARGIN = 0.01
_COMPRESSED_INTERVAL_DIRECTION_BUDGET = 13
_COMPRESSED_WORKING_SET_FRACTION = 0.9
_COMPRESSED_SUBSET_AUGMENTATION_BUDGET = 16
_QUANTIZED_MLP_MIN_ROWS = 32
# The exact classes meTile will replace, by module and name. A set of modules plus a hardcoded
# class name was not enough: every architecture here computes the same gated MLP, but they do not
# all call the class MLP. Qwen3.5 and Qwen3.6 reach it as Qwen3NextMLP from a third module, so a
# name check silently excluded three of the newest models, and their equivalence tests skipped
# with "patches nothing" rather than failing. A skipped test looks like a passing one in a summary.
#
# Membership is a claim about the implementation, not the name. Every class here has a __call__ of
# `down_proj(swiglu(gate_proj(x), up_proj(x)))`, and `swiglu(gate, x)` is `nn.silu(gate) * x`,
# which is what `_execute_quantized_mlp` computes.
_GATED_MLP_CLASSES = frozenset(
    {
        ("mlx_lm.models.llama", "MLP"),
        ("mlx_lm.models.qwen2", "MLP"),
        ("mlx_lm.models.qwen3", "MLP"),
        ("mlx_lm.models.qwen3_next", "Qwen3NextMLP"),
    }
)

# Blocks whose residual structure the fusion pass reproduces, which is a stricter requirement than
# carrying a gated MLP. Every class here has a __call__ of `r = attention(input_layernorm(x));
# h = x + r; out = h + mlp(post_attention_layernorm(h))`, and for the first three that text is
# character-for-character identical.
#
# Qwen3.5's DecoderLayer is included, but it is the reason `_attention_module` exists. It differs
# from the others in one respect: on every layer that is not a multiple of full_attention_interval
# the attention is a GatedDeltaNet bound to `linear_attn`, and `self_attn` is not present at all.
# The residual structure around it is the same, so resolving the attention by attribute rather
# than by name is the whole adaptation needed. Excluding it instead left the equivalence tests for
# three of the newest models skipping rather than passing.
_FUSED_BLOCK_CLASSES = frozenset(
    {
        ("mlx_lm.models.llama", "TransformerBlock"),
        ("mlx_lm.models.qwen2", "TransformerBlock"),
        ("mlx_lm.models.qwen3", "TransformerBlock"),
        ("mlx_lm.models.qwen3_5", "DecoderLayer"),
    }
)

# Attribute names a block may bind its attention to, in the order to look. Hybrid architectures
# alternate: Qwen3.5 uses `linear_attn` on most layers and `self_attn` on the rest, so this is
# resolved per call rather than once per class.
_ATTENTION_ATTRIBUTES = ("self_attn", "linear_attn")


def _attention_module(block):
    """The attention a block will call, or None if it binds none this pass understands."""
    for name in _ATTENTION_ATTRIBUTES:
        found = getattr(block, name, None)
        if found is not None:
            return found
    return None


# Attributes a class must carry to be a candidate for each replacement. Structure rather than name,
# because a name list only ever covers the architectures someone remembered: Qwen3.5, Qwen3.6 and
# Qwen3-VL were all excluded by one until it was noticed, and their equivalence tests reported skips that
# read like passes.
#
# Structure is a weaker claim than the name list it supplements. `gate_proj`/`up_proj`/`down_proj` says a
# class has the parts of a gated MLP, not that its __call__ combines them the way meTile's replacement
# does -- a model scaling the product, or applying a different activation, presents identically. So a
# structural match admits a candidate and nothing more; `metile.compile` runs the model and compares
# against the unpatched result before keeping any of it.
_GATED_MLP_ATTRIBUTES = ("gate_proj", "up_proj", "down_proj")
_FUSED_BLOCK_ATTRIBUTES = ("input_layernorm", "post_attention_layernorm", "mlp")

_STRUCTURE = {
    id(_GATED_MLP_CLASSES): _GATED_MLP_ATTRIBUTES,
    id(_FUSED_BLOCK_CLASSES): _FUSED_BLOCK_ATTRIBUTES,
}


def _structurally_matches(cls, registry):
    """Whether a class carries the parts the registry's replacement needs.

    Read off the class, so it works for architectures nobody enumerated. Requires __call__ to be defined
    on the class itself rather than inherited, because a class that does not define one has nothing for
    meTile to replace and `getattr` would hand back a fresh method-wrapper from the metaclass.
    """
    required = _STRUCTURE.get(id(registry))
    if required is None:
        return False
    if not any("__call__" in vars(klass) for klass in cls.__mro__):
        return False
    if not all(
        hasattr(cls, name) or name in getattr(cls, "__annotations__", {}) for name in required
    ):
        # Attributes are usually set in __init__ rather than declared, so fall back to the source of
        # __call__: a replacement only works if the body actually reaches those names.
        source = _call_source(cls)
        return source is not None and all(name in source for name in required)
    return True


def _call_source(cls):
    import inspect as _inspect

    for klass in cls.__mro__:
        if "__call__" in vars(klass):
            try:
                return _inspect.getsource(vars(klass)["__call__"])
            except (OSError, TypeError):
                return None
    return None


def _recognised(cls, registry, structural=True):
    """Whether meTile is allowed to replace this class's __call__.

    The named pairs are the combinations whose arithmetic has been checked against MLX in the model
    matrix. Structural matches are candidates that `metile.compile` verifies at patch time.
    """
    if (cls.__module__, cls.__name__) in registry:
        return True
    return structural and _structurally_matches(cls, registry)


def _registry_classes(registry):
    """Import and return the classes in a registry, skipping any this mlx-lm does not have.

    Used when patching without a model in hand. Skipping rather than raising because the
    registry spans several mlx-lm versions and a missing architecture is not an error.
    """
    import importlib

    found = []
    for module_name, class_name in sorted(registry):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            found.append(cls)
    return found


@dataclass(frozen=True)
class MLXLMPlan:
    """A measured MLX-LM feature combination."""

    attention: bool = True
    rms_norm: bool = True
    graph_fusion: bool = True
    quantized_mlp: bool = True
    affine_prefill: bool = False
    dense_mlp: bool = False
    dense_residual: bool = False
    compressed_down: bool = False
    compressed_gate_up: bool = False
    compressed_vocab: bool = False
    compressed_attention: bool = False

    @property
    def feature_count(self):
        return sum(vars(self).values())

    @property
    def is_decode_only_compression(self):
        return any(
            (
                self.compressed_down,
                self.compressed_gate_up,
                self.compressed_vocab,
                self.compressed_attention,
            )
        ) and not any(
            (
                self.attention,
                self.rms_norm,
                self.graph_fusion,
                self.quantized_mlp,
                self.affine_prefill,
                self.dense_mlp,
                self.dense_residual,
            )
        )

    def as_dict(self):
        return dict(vars(self))


@dataclass
class MLXPatch:
    """A reversible set of MLX-LM module patches."""

    replacements: list[tuple[object, str, object]]
    replacement: object | None = None
    original: object | None = None

    def restore(self):
        if self.replacement is not None:
            for module in tuple(sys.modules.values()):
                if (
                    module is not None
                    and getattr(module, "__name__", "").startswith("mlx_lm.models")
                    and getattr(module, "scaled_dot_product_attention", None) is self.replacement
                ):
                    module.scaled_dot_product_attention = self.original
        for module, name, original in reversed(self.replacements):
            if name == "__class__":
                object.__setattr__(module, name, original)
            else:
                setattr(module, name, original)
        self.replacements.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


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


@dataclass
class MLXCompressedDown:
    """AOT compressed down-projection weights for decode-only dispatch."""

    model: object
    weights: dict[int, tuple[object, MLXCompressedDownWeight]]
    format: str
    repack_bytes: int
    allow_approximate: bool = False
    group_size: int = 64
    group_tuning: dict | None = None
    patched_classes: dict[int, type] = field(default_factory=dict)
    calibrated: bool = False
    selection: str = "all"
    layer_indices: tuple[int, ...] = ()
    calibration_fidelity: dict | None = None

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

        class MLXCompressedDownLinear(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows != 1:
                    return original_call(self, values)
                return weight(values)

        MLXCompressedDownLinear.__name__ = f"MeTile{original_class.__name__}"
        MLXCompressedDownLinear.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXCompressedDownLinear
        return MLXCompressedDownLinear

    def fidelity_compatible(self, fidelity):
        if fidelity["next_token"] != fidelity["actual_next_token"]:
            return False
        if self.allow_approximate:
            return (
                fidelity["kl_divergence"] <= _COMPRESSED_APPROXIMATE_KL_LIMIT
                and fidelity["mean_logit_error"] <= _COMPRESSED_APPROXIMATE_MEAN_LOGIT_ERROR_LIMIT
                and fidelity["max_logit_error"] <= _COMPRESSED_APPROXIMATE_MAX_LOGIT_ERROR_LIMIT
            )
        return (
            self.format == "affine8"
            and fidelity["kl_divergence"] <= _COMPRESSED_AFFINE8_KL_LIMIT
            and fidelity["mean_logit_error"] <= _COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT
            and fidelity["max_logit_error"] <= _COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT
        )


@dataclass
class MLXCompressedGateUp:
    """Layer-grouped affine-INT8 gate/up weights for one-row decode."""

    model: object
    layers: dict[
        int, tuple[object, object, MLXCompressedDownWeight, object, MLXCompressedDownWeight]
    ]
    repack_bytes: int
    group_size: int = 64
    group_tuning: dict | None = None
    patched_classes: dict[int, type] = field(default_factory=dict)
    calibrated: bool = False
    selection: str = "all"
    layer_indices: tuple[int, ...] = ()
    calibration_fidelity: dict | None = None
    executors: dict[tuple, object] = field(default_factory=dict)
    implementation: str = "projected"
    implementation_tuning: dict | None = None
    source_layers: dict[int, tuple[object, object, object, object, object]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.implementation not in {"fused", "projected"}:
            raise ValueError("compressed gate/up implementation must be fused or projected")

    def weight_for(self, module):
        for _, gate, gate_weight, up, up_weight in self.layers.values():
            if gate is module:
                return gate_weight
            if up is module:
                return up_weight
        return None

    @property
    def layer_count(self):
        return len(self.layers)

    @property
    def projection_count(self):
        return 2 * self.layer_count

    def patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(module)
        original_call = original_class.__call__
        weight = self.weight_for(module)

        class MLXCompressedGateUpLinear(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows != 1:
                    return original_call(self, values)
                return weight(values)

        MLXCompressedGateUpLinear.__name__ = f"MeTile{original_class.__name__}"
        MLXCompressedGateUpLinear.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXCompressedGateUpLinear
        return MLXCompressedGateUpLinear

    def fused_patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        entry = self.layers.get(key)
        if entry is None or entry[0] is not module:
            raise ValueError("compressed gate/up layer is not active")
        _, _, gate_weight, _, up_weight = entry
        original_class = type(module)
        original_call = original_class.__call__
        group_size = self.group_size
        prepared = self

        class MLXCompressedGateUpMLP(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows != 1:
                    return original_call(self, values)
                executor_key = (
                    key,
                    id(gate_weight.values),
                    id(up_weight.values),
                    str(values.dtype),
                )
                executor = prepared.executors.get(executor_key)
                if executor is None:
                    executor = mlx_affine_swiglu_executor(
                        values,
                        gate_weight.values,
                        gate_weight.scales,
                        gate_weight.biases,
                        up_weight.values,
                        up_weight.scales,
                        up_weight.biases,
                        group_size=group_size,
                        bits=8,
                    )
                    prepared.executors[executor_key] = executor
                hidden = executor(values)
                return self.down_proj(hidden)

        MLXCompressedGateUpMLP.__name__ = f"MeTile{original_class.__name__}"
        MLXCompressedGateUpMLP.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXCompressedGateUpMLP
        return MLXCompressedGateUpMLP

    def fidelity_compatible(self, fidelity):
        return (
            fidelity["next_token"] == fidelity["actual_next_token"]
            and fidelity["kl_divergence"] <= _COMPRESSED_AFFINE8_KL_LIMIT
            and fidelity["mean_logit_error"] <= _COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT
            and fidelity["max_logit_error"] <= _COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT
        )


@dataclass
class MLXCompressedAttention:
    """Layer-grouped affine-INT8 attention projections for one-row decode."""

    model: object
    layers: dict[
        int,
        tuple[
            object,
            tuple[tuple[object, MLXCompressedDownWeight], ...],
        ],
    ]
    repack_bytes: int
    group_size: int = 64
    group_tuning: dict | None = None
    patched_classes: dict[int, type] = field(default_factory=dict)
    calibrated: bool = False
    selection: str = "all"
    layer_indices: tuple[int, ...] = ()
    calibration_fidelity: dict | None = None
    source_layers: dict[
        int,
        tuple[object, tuple[tuple[object, object], ...]],
    ] = field(default_factory=dict)

    def weight_for(self, module):
        for _, projections in self.layers.values():
            for projection, weight in projections:
                if projection is module:
                    return weight
        return None

    @property
    def layer_count(self):
        return len(self.layers)

    @property
    def projection_count(self):
        return sum(len(projections) for _, projections in self.layers.values())

    def patched_class(self, module):
        key = id(module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(module)
        original_call = original_class.__call__
        weight = self.weight_for(module)

        class MLXCompressedAttentionLinear(original_class):
            def __call__(self, values):
                rows = values.size // values.shape[-1]
                if rows != 1:
                    return original_call(self, values)
                output = weight(values)
                if "bias" in self:
                    output = output + self["bias"]
                return output

        MLXCompressedAttentionLinear.__name__ = f"MeTile{original_class.__name__}"
        MLXCompressedAttentionLinear.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = MLXCompressedAttentionLinear
        return MLXCompressedAttentionLinear

    def fidelity_compatible(self, fidelity):
        return (
            fidelity["next_token"] == fidelity["actual_next_token"]
            and fidelity["kl_divergence"] <= _COMPRESSED_AFFINE8_KL_LIMIT
            and fidelity["mean_logit_error"] <= _COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT
            and fidelity["max_logit_error"] <= _COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT
        )


@dataclass
class MLXCompressedVocab:
    """Affine-INT8 vocabulary projection for one-row decode."""

    model: object
    module: object
    weight: MLXCompressedDownWeight | None
    tied: bool
    repack_bytes: int
    group_size: int = 64
    group_tuning: dict | None = None
    patched_classes: dict[int, type] = field(default_factory=dict)
    calibrated: bool = False
    calibration_fidelity: dict | None = None

    @property
    def projection_count(self):
        return int(self.weight is not None)

    def patched_class(self):
        key = id(self.module)
        patched = self.patched_classes.get(key)
        if patched is not None:
            return patched
        original_class = type(self.module)
        weight = self.weight
        if weight is None:
            raise ValueError("compressed vocabulary projection is disabled")

        if self.tied:
            original_as_linear = original_class.as_linear

            class MLXCompressedVocabEmbedding(original_class):
                def as_linear(self, values):
                    rows = values.size // values.shape[-1]
                    if rows != 1:
                        return original_as_linear(self, values)
                    return weight(values)

            patched = MLXCompressedVocabEmbedding
        else:
            original_call = original_class.__call__

            class MLXCompressedVocabLinear(original_class):
                def __call__(self, values):
                    rows = values.size // values.shape[-1]
                    if rows != 1:
                        return original_call(self, values)
                    return weight(values)

            patched = MLXCompressedVocabLinear

        patched.__name__ = f"MeTile{original_class.__name__}"
        patched.__qualname__ = f"MeTile{original_class.__qualname__}"
        self.patched_classes[key] = patched
        return patched

    def fidelity_compatible(self, fidelity):
        return (
            fidelity["next_token"] == fidelity["actual_next_token"]
            and fidelity["kl_divergence"] <= _COMPRESSED_AFFINE8_KL_LIMIT
            and fidelity["mean_logit_error"] <= _COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT
            and fidelity["max_logit_error"] <= _COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT
        )


def _mlx_lm_plan_candidates(requested):
    requested_names = tuple(name for name, enabled in requested.as_dict().items() if enabled)
    compression_names = tuple(
        name
        for name in (
            "compressed_down",
            "compressed_gate_up",
            "compressed_vocab",
            "compressed_attention",
        )
        if name in requested_names
    )
    structural_names = tuple(name for name in requested_names if name not in compression_names)
    enabled_sets = {
        frozenset(name for index, name in enumerate(compression_names) if mask & (1 << index))
        for mask in range(1 << len(compression_names))
    }
    maximum_structural_order = len(structural_names) if len(structural_names) <= 3 else 2
    structural_sets = {
        frozenset(names)
        for order in range(maximum_structural_order + 1)
        for names in combinations(structural_names, order)
    }
    full_compression = frozenset(compression_names)
    for structural in structural_sets:
        enabled_sets.add(structural)
        enabled_sets.add(structural | full_compression)
    enabled_sets.add(frozenset(requested_names))

    candidates = []
    for enabled in enabled_sets:
        if "compressed_down" in enabled and "dense_residual" in enabled:
            continue
        if "compressed_gate_up" in enabled and "dense_mlp" in enabled:
            continue
        candidates.append(
            MLXLMPlan(
                attention="attention" in enabled,
                rms_norm="rms_norm" in enabled,
                graph_fusion="graph_fusion" in enabled,
                quantized_mlp="quantized_mlp" in enabled,
                affine_prefill="affine_prefill" in enabled,
                dense_mlp="dense_mlp" in enabled,
                dense_residual="dense_residual" in enabled,
                compressed_down="compressed_down" in enabled,
                compressed_gate_up="compressed_gate_up" in enabled,
                compressed_vocab="compressed_vocab" in enabled,
                compressed_attention="compressed_attention" in enabled,
            )
        )
    return tuple(
        sorted(candidates, key=lambda plan: (plan.feature_count, tuple(vars(plan).values())))
    )


def _mlx_lm_warmup_plans(candidates):
    """Select the linear-sized plans needed to populate primitive dispatch caches."""
    compile_features = (
        "attention",
        "rms_norm",
        "graph_fusion",
        "quantized_mlp",
        "affine_prefill",
        "dense_mlp",
        "dense_residual",
    )
    available = {name for plan in candidates for name, enabled in plan.as_dict().items() if enabled}
    required = {frozenset()}
    required.update(frozenset((name,)) for name in compile_features if name in available)
    for interaction in (
        frozenset(("graph_fusion", "quantized_mlp")),
        frozenset(("dense_mlp", "dense_residual")),
    ):
        if interaction <= available:
            required.add(interaction)
    return tuple(
        plan
        for plan in candidates
        if frozenset(name for name, enabled in plan.as_dict().items() if enabled) in required
    )


def _effective_mlx_lm_plan(
    plan,
    affine_prefill=None,
    dense_mlp=None,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    return MLXLMPlan(
        attention=plan.attention
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_attention_dispatches()),
        rms_norm=plan.rms_norm
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_rms_norm_dispatches()),
        graph_fusion=plan.graph_fusion
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_add_rms_norm_dispatches()),
        quantized_mlp=plan.quantized_mlp
        and (
            any(dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_swiglu_dispatches())
            or any(
                dispatch["algorithm"] != "mlx" for dispatch in mlx_affine_residual_qmv_dispatches()
            )
        ),
        affine_prefill=plan.affine_prefill
        and affine_prefill is not None
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_affine_matmul_dispatches()),
        dense_mlp=plan.dense_mlp
        and dense_mlp is not None
        and (
            (
                dense_mlp.implementation == "fused"
                and any(
                    dispatch["algorithm"] == "metile" for dispatch in mlx_dense_swiglu_dispatches()
                )
            )
            or (
                dense_mlp.implementation == "projected"
                and any(
                    dispatch["algorithm"] == "metile" for dispatch in mlx_dense_matmul_dispatches()
                )
            )
        ),
        dense_residual=plan.dense_residual
        and dense_mlp is not None
        and any(dispatch["algorithm"] == "metile" for dispatch in mlx_dense_residual_dispatches()),
        compressed_down=plan.compressed_down
        and compressed_down is not None
        and compressed_down.projection_count > 0,
        compressed_gate_up=plan.compressed_gate_up
        and compressed_gate_up is not None
        and compressed_gate_up.projection_count > 0,
        compressed_vocab=plan.compressed_vocab
        and compressed_vocab is not None
        and compressed_vocab.projection_count > 0,
        compressed_attention=plan.compressed_attention
        and compressed_attention is not None
        and compressed_attention.projection_count > 0,
    )


def _mlx_lm_model_signature(model):
    layers = tuple(_model_layers(model))
    first_layer = layers[0] if layers else None
    attention = getattr(first_layer, "self_attn", None)
    norm = getattr(first_layer, "input_layernorm", None)
    weight = getattr(norm, "weight", None)
    return {
        "attention_class": (
            f"{type(attention).__module__}.{type(attention).__qualname__}"
            if attention is not None
            else None
        ),
        "head_dim": getattr(attention, "head_dim", None),
        "hidden": weight.shape[0] if weight is not None else None,
        "layers": len(layers),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "n_heads": getattr(attention, "n_heads", None),
        "n_kv_heads": getattr(attention, "n_kv_heads", None),
        "scale": getattr(attention, "scale", None),
    }


def _compressed_down_calibration_key(
    model,
    sample_tokens,
    compressed_down,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "allow_approximate": compressed_down.allow_approximate,
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "format": compressed_down.format,
            "group_size": compressed_down.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_down),
                    "class": inspect.getsource(MLXCompressedDown.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedDown.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_down),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_down_calibration),
                    "write": inspect.getsource(_write_compressed_down_calibration),
                }
            ),
            "weights": tuple(
                (weight.shape, weight.format, weight.group_size)
                for _, weight in compressed_down.weights.values()
            ),
        }
    )


def _mlx_lm_plan_key(
    model,
    sample_tokens,
    requested,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "affine_prefill": (
                {
                    "min_rows": affine_prefill.min_rows,
                    "projections": affine_prefill.projection_count,
                }
                if affine_prefill is not None
                else None
            ),
            "decode_steps": decode_steps,
            "compressed_down": (
                {
                    "allow_approximate": compressed_down.allow_approximate,
                    "format": compressed_down.format,
                    "group_size": compressed_down.group_size,
                    "layer_indices": compressed_down.layer_indices,
                    "projections": compressed_down.projection_count,
                    "repack_bytes": compressed_down.repack_bytes,
                    "selection": compressed_down.selection,
                }
                if compressed_down is not None
                else None
            ),
            "compressed_gate_up": (
                {
                    "group_size": compressed_gate_up.group_size,
                    "implementation": compressed_gate_up.implementation,
                    "layer_indices": compressed_gate_up.layer_indices,
                    "layers": compressed_gate_up.layer_count,
                    "projections": compressed_gate_up.projection_count,
                    "repack_bytes": compressed_gate_up.repack_bytes,
                    "selection": compressed_gate_up.selection,
                }
                if compressed_gate_up is not None
                else None
            ),
            "compressed_vocab": (
                {
                    "group_size": compressed_vocab.group_size,
                    "projections": compressed_vocab.projection_count,
                    "repack_bytes": compressed_vocab.repack_bytes,
                    "tied": compressed_vocab.tied,
                }
                if compressed_vocab is not None
                else None
            ),
            "compressed_attention": (
                {
                    "group_size": compressed_attention.group_size,
                    "layer_indices": compressed_attention.layer_indices,
                    "layers": compressed_attention.layer_count,
                    "projections": compressed_attention.projection_count,
                    "repack_bytes": compressed_attention.repack_bytes,
                    "selection": compressed_attention.selection,
                }
                if compressed_attention is not None
                else None
            ),
            "dense_mlp": (
                {
                    "min_rows": dense_mlp.min_rows,
                    "mlps": dense_mlp.mlp_count,
                    "repack_bytes": dense_mlp.repack_bytes,
                }
                if dense_mlp is not None
                else None
            ),
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt_bucket": 1 << max(sample_tokens.shape[1] - 1, 0).bit_length(),
            "requested": requested.as_dict(),
            "source": stable_digest(
                {
                    "apply": inspect.getsource(apply_metile_to_mlx_lm),
                    "affine_backend": mlx_affine_backend_signature(),
                    "affine_prefill_class": inspect.getsource(MLXAffinePrefill.patched_class),
                    "affine_prefill": inspect.getsource(_patch_affine_prefill),
                    "affine_swiglu_backend": mlx_affine_swiglu_backend_signature(),
                    "choose": inspect.getsource(_choose_mlx_lm_plan),
                    "plan_candidates": inspect.getsource(_mlx_lm_plan_candidates),
                    "compressed_down_backend": mlx_compressed_down_backend_signature(),
                    "compressed_down_class": inspect.getsource(MLXCompressedDown.patched_class),
                    "compressed_down_calibration": inspect.getsource(_calibrate_compressed_down),
                    "compressed_down_candidates": inspect.getsource(
                        _compressed_down_subset_candidates
                    ),
                    "compressed_down_patch": inspect.getsource(_patch_compressed_down),
                    "compressed_gate_up_class": inspect.getsource(
                        MLXCompressedGateUp.patched_class
                    ),
                    "compressed_gate_up_fused_class": inspect.getsource(
                        MLXCompressedGateUp.fused_patched_class
                    ),
                    "compressed_gate_up_fusion_guard": inspect.getsource(
                        _supports_compressed_gate_up_fusion
                    ),
                    "compressed_gate_up_calibration": inspect.getsource(
                        _calibrate_compressed_gate_up
                    ),
                    "compressed_gate_up_group": inspect.getsource(
                        _autotune_compressed_gate_up_group
                    ),
                    "compressed_gate_up_implementation": inspect.getsource(
                        _select_compressed_gate_up_implementation
                    ),
                    "compressed_gate_up_patch": inspect.getsource(_patch_compressed_gate_up),
                    "compressed_attention_class": inspect.getsource(
                        MLXCompressedAttention.patched_class
                    ),
                    "compressed_attention_calibration": inspect.getsource(
                        _calibrate_compressed_attention
                    ),
                    "compressed_attention_group": inspect.getsource(
                        _autotune_compressed_attention_group
                    ),
                    "compressed_attention_patch": inspect.getsource(_patch_compressed_attention),
                    "compressed_region_policy": _compressed_region_policy_signature(),
                    "compressed_vocab_calibration": inspect.getsource(_calibrate_compressed_vocab),
                    "compressed_vocab_class": inspect.getsource(MLXCompressedVocab.patched_class),
                    "compressed_vocab_patch": inspect.getsource(_patch_compressed_vocab),
                    "dense_backend": mlx_dense_swiglu_backend_signature(),
                    "dense_matmul_backend": mlx_dense_backend_signature(),
                    "dense_residual_backend": mlx_dense_residual_backend_signature(),
                    "dense_class": inspect.getsource(MLXDenseMLP.patched_class),
                    "dense_execute": inspect.getsource(_execute_dense_mlp),
                    "dense_patch": inspect.getsource(_patch_dense_mlp),
                    "dense_residual_support": inspect.getsource(_supports_dense_residual_mlp),
                    "dense_selection": inspect.getsource(_select_dense_mlp_implementation),
                    "decode_only_plan": inspect.getsource(_is_decode_only_compression_plan),
                    "effective": inspect.getsource(_effective_mlx_lm_plan),
                    "fidelity": inspect.getsource(_plan_preserves_logits),
                    "finalists": inspect.getsource(_provisional_mlx_lm_finalists),
                    "plan": inspect.getsource(MLXLMPlan),
                    "prompt": inspect.getsource(_prepare_mlx_lm_prompt),
                    "warmups": inspect.getsource(_mlx_lm_warmup_plans),
                    "graph_patch": inspect.getsource(_patch_graph_fusion),
                    "quantized_mlp_patch": inspect.getsource(_patch_quantized_mlp),
                    "rank": inspect.getsource(_rank_mlx_lm_plans),
                    "timing": inspect.getsource(_time_mlx_lm_plan),
                    "validation": inspect.getsource(_validate_mlx_lm_plan),
                    "validation_finalists": inspect.getsource(_mlx_lm_validation_finalists),
                    "validation_joint": inspect.getsource(_validate_mlx_lm_finalists_repeated),
                    "validation_retry": inspect.getsource(_validate_mlx_lm_plan_repeated),
                }
            ),
            "regression_margin": _MODEL_REGRESSION_MARGIN,
            "decode_switch_margin": _MODEL_DECODE_SWITCH_MARGIN,
            "strong_decode_switch_margin": _MODEL_STRONG_DECODE_SWITCH_MARGIN,
            "strong_decode_ttft_regression_margin": (_MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN),
            "dense_mlp_implementation": (
                dense_mlp.implementation if dense_mlp is not None else None
            ),
            "quantized_mlp_min_rows": _QUANTIZED_MLP_MIN_ROWS,
            "switch_margin": _MODEL_SWITCH_MARGIN,
            "ttft_switch_margin": _MODEL_TTFT_SWITCH_MARGIN,
            "trials": trials,
            "screen_max_finalists": _MODEL_SCREEN_MAX_FINALISTS,
            "screen_relative_margin": _MODEL_SCREEN_RELATIVE_MARGIN,
            "screen_rounds": _MODEL_SCREEN_ROUNDS,
            "provisional_max_finalists": _MODEL_PROVISIONAL_MAX_FINALISTS,
            "provisional_relative_margin": _MODEL_PROVISIONAL_RELATIVE_MARGIN,
            "provisional_rounds": _MODEL_PROVISIONAL_ROUNDS,
            "search_decode_steps": _MODEL_SEARCH_MIN_DECODE_STEPS,
            "validation_decode_steps": _MODEL_VALIDATION_MIN_DECODE_STEPS,
            "validation_attempts": _MODEL_VALIDATION_ATTEMPTS,
            "validation_max_survivors": _MODEL_VALIDATION_MAX_SURVIVORS,
            "validation_screen_trials": _MODEL_VALIDATION_SCREEN_TRIALS,
            "validation_trials": _MODEL_VALIDATION_MIN_TRIALS,
            "tuner": 47,
        }
    )


def _read_mlx_lm_plan(key):
    cached = _mlx_lm_plan_cache.get(key)
    if cached is not None:
        return cached
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_mlx_lm_plan_cache_path, {}).get(key)
    if not isinstance(payload, dict):
        return None
    try:
        return MLXLMPlan(**{name: bool(payload[name]) for name in vars(MLXLMPlan())})
    except KeyError:
        return None


def _write_mlx_lm_plan(key, plan):
    _mlx_lm_plan_cache[key] = plan
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_mlx_lm_plan_cache_path, {})
    payload[key] = plan.as_dict()
    atomic_write_json(_mlx_lm_plan_cache_path, payload)


def _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if decode_steps < 1:
        raise ValueError("prompt preparation requires positive decode steps")
    cache = make_prompt_cache(model)
    start = time.perf_counter_ns()
    logits = model(sample_tokens, cache=cache)
    mx.eval(logits)
    elapsed = (time.perf_counter_ns() - start) * 1e-9
    trajectory_cache = copy.deepcopy(cache)
    decode_tokens = []
    for step in range(decode_steps):
        token = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(token)
        decode_tokens.append(token)
        if step + 1 < decode_steps:
            logits = model(token, cache=trajectory_cache)
            mx.eval(logits)
    return cache, elapsed, tuple(decode_tokens)


def _time_mlx_lm_plan(
    model,
    sample_tokens,
    plan,
    affine_prefill,
    dense_mlp,
    decode_steps,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    prepared_prompt=None,
    decode_tokens=None,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    decode_token = sample_tokens[:, -1:]
    if prepared_prompt is not None:
        cache = copy.deepcopy(prepared_prompt[0])
        ttft = prepared_prompt[1]
    else:
        cache = make_prompt_cache(model)
    with apply_metile_to_mlx_lm(
        model=model,
        plan=plan,
        affine_prefill=affine_prefill,
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
    ):
        if prepared_prompt is None:
            total_start = time.perf_counter_ns()
            logits = model(sample_tokens, cache=cache)
            mx.eval(logits)
            ttft = (time.perf_counter_ns() - total_start) * 1e-9
        if decode_tokens is None:
            decode_tokens = (decode_token,) * decode_steps
        elif len(decode_tokens) != decode_steps:
            raise ValueError("decode trajectory must match decode steps")
        decode_start = time.perf_counter_ns()
        for token in decode_tokens:
            logits = model(token, cache=cache)
            mx.eval(logits)
        decode_elapsed = (time.perf_counter_ns() - decode_start) * 1e-9
        decode = decode_elapsed / decode_steps
        total = ttft + decode_elapsed
    next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
    return (ttft, decode, total), next_token


def _logit_fidelity(reference, actual):
    import mlx.core as mx

    reference_dtype = str(reference.dtype)
    reference = reference[:, -1].astype(mx.float32)
    actual = actual[:, -1].astype(mx.float32)
    difference = mx.abs(reference - actual)
    reference_log_probs = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    actual_log_probs = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
    divergence = mx.sum(
        mx.exp(reference_log_probs) * (reference_log_probs - actual_log_probs),
        axis=-1,
    )
    mx.eval(difference, divergence)
    return {
        "reference_dtype": reference_dtype,
        "next_token": int(mx.argmax(reference, axis=-1).item()),
        "actual_next_token": int(mx.argmax(actual, axis=-1).item()),
        "kl_divergence": max(0.0, float(mx.max(divergence).item())),
        "mean_logit_error": float(mx.mean(difference).item()),
        "max_logit_error": float(mx.max(difference).item()),
    }


def _fidelity_compatible(fidelity):
    is_bfloat16 = fidelity.get("reference_dtype") == "mlx.core.bfloat16"
    mean_limit = (
        _MODEL_BF16_MEAN_LOGIT_ERROR_LIMIT if is_bfloat16 else _MODEL_MEAN_LOGIT_ERROR_LIMIT
    )
    maximum_limit = (
        _MODEL_BF16_MAX_LOGIT_ERROR_LIMIT if is_bfloat16 else _MODEL_MAX_LOGIT_ERROR_LIMIT
    )
    return (
        fidelity["next_token"] == fidelity["actual_next_token"]
        and fidelity["kl_divergence"] <= _MODEL_KL_LIMIT
        and fidelity["mean_logit_error"] <= mean_limit
        and fidelity["max_logit_error"] <= maximum_limit
    )


def _compressed_down_subset_candidates(count):
    if count < 1:
        return
    yield "all", tuple(range(count))
    for size in range(count - 1, 0, -1):
        yield f"suffix:{size}", tuple(range(count - size, count))
        yield f"prefix:{size}", tuple(range(size))


def _augment_compressed_subset(count, evaluate, selected, *, budget):
    """Grow a compatible interval into a bounded non-contiguous layer subset."""
    selected_name, selected_indices, selected_fidelity = selected
    selected_indices = tuple(sorted(selected_indices))
    attempts = 0
    while attempts < budget and len(selected_indices) < count:
        selected_set = set(selected_indices)
        if selected_name.startswith("prefix"):
            excluded = (index for index in range(count) if index not in selected_set)
        else:
            excluded = (index for index in range(count - 1, -1, -1) if index not in selected_set)
        grew = False
        for index in excluded:
            candidate = tuple(sorted((*selected_indices, index)))
            name = "subset:" + ",".join(map(str, candidate))
            compatible, fidelity = evaluate(name, candidate)
            attempts += 1
            if compatible:
                selected_name = name
                selected_indices = candidate
                selected_fidelity = fidelity
                grew = True
                break
            if attempts >= budget:
                break
        if not grew:
            break
    return selected_name, selected_indices, selected_fidelity


def _select_compressed_region(
    count,
    evaluate,
    *,
    augmentation_budget=_COMPRESSED_SUBSET_AUGMENTATION_BUDGET,
):
    """Find a large compatible layer mask with logarithmic intervals and bounded augmentation."""
    if count < 1:
        return "native", (), None
    all_indices = tuple(range(count))
    compatible, fidelity = evaluate("all", all_indices)
    if compatible:
        return "all", all_indices, fidelity

    def indices_for(direction, size):
        if direction == "suffix":
            return tuple(range(count - size, count))
        return tuple(range(size))

    def search(direction):
        results = {}

        def check(size):
            cached = results.get(size)
            if cached is not None:
                return cached
            indices = indices_for(direction, size)
            if len(results) >= _COMPRESSED_INTERVAL_DIRECTION_BUDGET:
                return False, None, indices
            result = evaluate(f"{direction}:{size}", indices)
            results[size] = (*result, indices)
            return results[size]

        upper_failure = count
        step = 1
        while True:
            size = max(1, count - step)
            is_compatible, _, _ = check(size)
            if is_compatible:
                lower = size
                break
            upper_failure = size
            if size == 1:
                return None
            step *= 2

        upper = upper_failure - 1
        while lower < upper:
            middle = (lower + upper + 1) // 2
            is_compatible, _, _ = check(middle)
            if is_compatible:
                lower = middle
            else:
                upper = middle - 1
        audit_sizes = {
            failed + offset
            for failed, result in results.items()
            if not result[0]
            for offset in (-2, -1, 1, 2)
            if lower < failed + offset < count
        }
        for size in sorted(audit_sizes, reverse=True):
            audit_compatible, _, _ = check(size)
            if audit_compatible:
                lower = max(lower, size)
        while lower < count - 1:
            boundary_compatible, _, _ = check(lower + 1)
            if not boundary_compatible:
                break
            lower += 1
        is_compatible, selected_fidelity, selected_indices = check(lower)
        if not is_compatible:
            return None
        return f"{direction}:{lower}", selected_indices, selected_fidelity

    candidates = tuple(
        candidate for candidate in (search("suffix"), search("prefix")) if candidate is not None
    )
    selected = (
        max(
            candidates,
            key=lambda candidate: (
                len(candidate[1]),
                candidate[0].startswith("suffix"),
            ),
        )
        if candidates
        else ("native", (), None)
    )
    return _augment_compressed_subset(
        count,
        evaluate,
        selected,
        budget=augmentation_budget,
    )


def _audit_larger_compressed_regions(
    count,
    evaluate,
    selected,
    *,
    selected_compatible=True,
    local_window=4,
):
    """Refine a short-horizon frontier with bounded full-horizon evaluations."""
    target_size = len(selected[1])
    best = selected if selected_compatible else ("native", (), None)
    candidates = {("all", tuple(range(count)))}
    selected_set = set(selected[1])
    prefix_run = next(
        (index for index in range(count) if index not in selected_set),
        count,
    )
    suffix_run = next(
        (count - index - 1 for index in range(count - 1, -1, -1) if index not in selected_set),
        count,
    )
    preferred = "suffix" if suffix_run >= prefix_run else "prefix"
    opposite = "prefix" if preferred == "suffix" else "suffix"
    sizes = set(range(max(1, count - 4), count))
    sizes.update(
        range(
            max(1, target_size - local_window),
            min(count - 1, target_size + 2) + 1,
        )
    )
    for size in sizes:
        indices = tuple(range(count - size, count)) if preferred == "suffix" else tuple(range(size))
        candidates.add((f"{preferred}:{size}", indices))
    opposite_size = count - 1
    opposite_indices = (
        tuple(range(opposite_size)) if opposite == "prefix" else tuple(range(1, count))
    )
    candidates.add((f"{opposite}:{opposite_size}", opposite_indices))
    for name, indices in sorted(
        candidates,
        key=lambda candidate: (
            -len(candidate[1]),
            not candidate[0].startswith("suffix"),
        ),
    ):
        if indices == selected[1] or len(indices) <= len(best[1]):
            continue
        compatible, fidelity = evaluate(name, indices)
        if compatible:
            return name, indices, fidelity
    if best[1]:
        return best
    return _select_compressed_region(count, evaluate, augmentation_budget=0)


@dataclass(frozen=True)
class _CompressedCalibrationReference:
    decode_token: object
    prompt_cache: object | None
    search_steps: int
    search_reference: object
    full_reference: object


def _prepare_compressed_calibration_reference(model, sample_tokens, decode_steps):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if decode_steps < 1:
        raise ValueError("compressed calibration requires positive decode steps")
    decode_token = sample_tokens[:, -1:]
    reference_cache = make_prompt_cache(model)
    reference = model(sample_tokens, cache=reference_cache)
    mx.eval(reference)
    prompt_cache = copy.deepcopy(reference_cache) if sample_tokens.shape[1] > 1 else None
    search_steps = min(_COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS, decode_steps)
    search_reference = reference
    for step in range(decode_steps):
        reference = model(decode_token, cache=reference_cache)
        mx.eval(reference)
        if step + 1 == search_steps:
            search_reference = reference
    return _CompressedCalibrationReference(
        decode_token,
        prompt_cache,
        search_steps,
        search_reference,
        reference,
    )


def _run_compressed_calibration_candidate(
    model,
    sample_tokens,
    reference,
    steps,
    plan,
    **patches,
):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if steps < 1:
        raise ValueError("compressed calibration requires positive decode steps")
    actual_cache = (
        copy.deepcopy(reference.prompt_cache)
        if reference.prompt_cache is not None
        else make_prompt_cache(model)
    )
    with apply_metile_to_mlx_lm(model=model, plan=plan, **patches):
        if reference.prompt_cache is None:
            actual = model(sample_tokens, cache=actual_cache)
            mx.eval(actual)
        for _ in range(steps):
            actual = model(reference.decode_token, cache=actual_cache)
            mx.eval(actual)
    return actual


def _compressed_region_policy_signature():
    return stable_digest(
        {
            "candidate_trajectory": inspect.getsource(_run_compressed_calibration_candidate),
            "full_horizon": inspect.getsource(_audit_larger_compressed_regions),
            "interval": inspect.getsource(_select_compressed_region),
            "interval_direction_budget": _COMPRESSED_INTERVAL_DIRECTION_BUDGET,
            "reference_trajectory": inspect.getsource(_prepare_compressed_calibration_reference),
            "subset": inspect.getsource(_augment_compressed_subset),
            "subset_budget": _COMPRESSED_SUBSET_AUGMENTATION_BUDGET,
        }
    )


def _restore_compressed_down_calibration(compressed_down, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return False
    record = read_json(_compressed_down_calibration_cache_path, {}).get(key)
    if not isinstance(record, dict):
        return False
    selection = record.get("selection")
    indices = record.get("layer_indices")
    fidelity = record.get("fidelity")
    entries = tuple(compressed_down.weights.items())
    if (
        not isinstance(selection, str)
        or not isinstance(indices, list)
        or not all(
            not isinstance(index, bool) and isinstance(index, int) and 0 <= index < len(entries)
            for index in indices
        )
        or len(indices) != len(set(indices))
        or (fidelity is not None and not isinstance(fidelity, dict))
    ):
        return False
    compressed_down.weights = {entries[index][0]: entries[index][1] for index in indices}
    compressed_down.repack_bytes = sum(
        weight.nbytes for _, weight in compressed_down.weights.values()
    )
    compressed_down.calibrated = True
    compressed_down.selection = selection
    compressed_down.layer_indices = tuple(indices)
    compressed_down.calibration_fidelity = fidelity
    return True


def _write_compressed_down_calibration(compressed_down, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_compressed_down_calibration_cache_path, {})
    payload[key] = {
        "fidelity": compressed_down.calibration_fidelity,
        "layer_indices": compressed_down.layer_indices,
        "selection": compressed_down.selection,
    }
    atomic_write_json(_compressed_down_calibration_cache_path, payload)


def _calibrate_compressed_down(model, sample_tokens, compressed_down, decode_steps):
    if compressed_down.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_down.weights.items())
    if not entries:
        compressed_down.calibrated = True
        compressed_down.selection = "native"
        return
    key = _compressed_down_calibration_key(
        model,
        sample_tokens,
        compressed_down,
        decode_steps,
    )
    with _compressed_down_calibration_lock:
        if _restore_compressed_down_calibration(compressed_down, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_down=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_down.patched_classes.clear()
            compressed_down.weights = {entries[index][0]: entries[index][1] for index in indices}
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_down=compressed_down,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_down.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_down.patched_classes.clear()
    compressed_down.weights = {entries[index][0]: entries[index][1] for index in selected_indices}
    compressed_down.repack_bytes = sum(
        weight.nbytes for _, weight in compressed_down.weights.values()
    )
    compressed_down.calibrated = True
    compressed_down.selection = selected_name
    compressed_down.layer_indices = selected_indices
    compressed_down.calibration_fidelity = selected_fidelity
    with _compressed_down_calibration_lock:
        _write_compressed_down_calibration(compressed_down, key)
    gc.collect()
    mx.clear_cache()


def _compressed_gate_up_repack_bytes(layers):
    return sum(
        gate_weight.nbytes + up_weight.nbytes for _, _, gate_weight, _, up_weight in layers.values()
    )


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


def _repack_compressed_gate_up_group(compressed_gate_up, group_size):
    import mlx.core as mx

    compressed_gate_up.layers = {}
    compressed_gate_up.patched_classes.clear()
    compressed_gate_up.executors.clear()
    compressed_gate_up.calibrated = False
    compressed_gate_up.implementation = "projected"
    compressed_gate_up.implementation_tuning = None
    compressed_gate_up.selection = "all"
    compressed_gate_up.layer_indices = ()
    compressed_gate_up.calibration_fidelity = None
    gc.collect()
    mx.clear_cache()
    layers = {}
    for key, (module, gate, gate_weight, up, up_weight) in compressed_gate_up.source_layers.items():
        layers[key] = (
            module,
            gate,
            MLXCompressedDownWeight.quantize(
                gate_weight,
                format="affine8",
                group_size=group_size,
            ),
            up,
            MLXCompressedDownWeight.quantize(
                up_weight,
                format="affine8",
                group_size=group_size,
            ),
        )
    compressed_gate_up.layers = layers
    compressed_gate_up.group_size = group_size
    compressed_gate_up.repack_bytes = _compressed_gate_up_repack_bytes(layers)


def _compressed_gate_up_group_key(model, sample_tokens, compressed_gate_up, decode_steps):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_tuning": {
                name: value
                for name, value in compressed_gate_up.group_tuning.items()
                if name != "cached"
            },
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "calibrate": inspect.getsource(_calibrate_compressed_gate_up),
                    "region_policy": _compressed_region_policy_signature(),
                    "repack": inspect.getsource(_repack_compressed_gate_up_group),
                    "select": inspect.getsource(_select_model_affine8_group),
                    "tune": inspect.getsource(_autotune_compressed_gate_up_group),
                }
            ),
            "weights": tuple(
                (gate_weight.shape, str(gate_weight.dtype), up_weight.shape)
                for _, _, gate_weight, _, up_weight in compressed_gate_up.source_layers.values()
            ),
        }
    )


def _autotune_compressed_gate_up_group(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
):
    tuning = compressed_gate_up.group_tuning
    if tuning is None or tuning.get("model_calibrated") or not compressed_gate_up.source_layers:
        return
    timings_payload = tuning.get("median_nanoseconds")
    native_timing = tuning.get("native_median_nanoseconds")
    if not isinstance(timings_payload, dict) or not isinstance(native_timing, int):
        return
    timings = {int(group): int(value) for group, value in timings_payload.items()}
    key = _compressed_gate_up_group_key(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
    )
    cached = None
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_gate_up_group_lock:
            cached = read_json(_compressed_gate_up_group_cache_path, {}).get(key)
    if isinstance(cached, dict) and cached.get("group_size") in timings:
        selected = cached["group_size"]
        _repack_compressed_gate_up_group(compressed_gate_up, selected)
        _calibrate_compressed_gate_up(
            model,
            sample_tokens,
            compressed_gate_up,
            decode_steps,
        )
        compressed_gate_up.group_tuning = {
            **tuning,
            **cached,
            "cached": True,
            "group_size": selected,
            "micro_group_size": tuning["group_size"],
            "model_calibrated": True,
        }
        return

    total_layers = len(compressed_gate_up.source_layers)
    candidates = {}
    layer_counts = {}
    for group in sorted(timings):
        _repack_compressed_gate_up_group(compressed_gate_up, group)
        _calibrate_compressed_gate_up(
            model,
            sample_tokens,
            compressed_gate_up,
            decode_steps,
        )
        layer_counts[group] = compressed_gate_up.layer_count
        candidates[str(group)] = {
            "fidelity": compressed_gate_up.calibration_fidelity,
            "layers": compressed_gate_up.layer_count,
            "selection": compressed_gate_up.selection,
        }
    selected, estimates = _select_model_affine8_group(
        total_layers,
        layer_counts,
        timings,
        native_timing,
    )
    if compressed_gate_up.group_size != selected:
        _repack_compressed_gate_up_group(compressed_gate_up, selected)
        _calibrate_compressed_gate_up(
            model,
            sample_tokens,
            compressed_gate_up,
            decode_steps,
        )
    record = {
        "group_size": selected,
        "model_calibrated": True,
        "model_candidates": candidates,
        "predicted_nanoseconds": {str(group): round(value) for group, value in estimates.items()},
    }
    compressed_gate_up.group_tuning = {
        **tuning,
        **record,
        "cached": False,
        "micro_group_size": tuning["group_size"],
    }
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_gate_up_group_lock:
            payload = read_json(_compressed_gate_up_group_cache_path, {})
            payload[key] = record
            atomic_write_json(_compressed_gate_up_group_cache_path, payload)


def _compressed_gate_up_calibration_key(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_gate_up.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_gate_up),
                    "class": inspect.getsource(MLXCompressedGateUp.patched_class),
                    "fused_backend": mlx_affine_swiglu_backend_signature(),
                    "fused_class": inspect.getsource(MLXCompressedGateUp.fused_patched_class),
                    "fused_guard": inspect.getsource(_supports_compressed_gate_up_fusion),
                    "fidelity": inspect.getsource(MLXCompressedGateUp.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_gate_up),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_gate_up_calibration),
                    "write": inspect.getsource(_write_compressed_gate_up_calibration),
                }
            ),
            "weights": tuple(
                (
                    gate_weight.shape,
                    gate_weight.group_size,
                    up_weight.shape,
                    up_weight.group_size,
                )
                for _, _, gate_weight, _, up_weight in compressed_gate_up.layers.values()
            ),
        }
    )


def _restore_compressed_gate_up_calibration(compressed_gate_up, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return False
    record = read_json(_compressed_gate_up_calibration_cache_path, {}).get(key)
    if not isinstance(record, dict):
        return False
    selection = record.get("selection")
    indices = record.get("layer_indices")
    fidelity = record.get("fidelity")
    entries = tuple(compressed_gate_up.layers.items())
    if (
        not isinstance(selection, str)
        or not isinstance(indices, list)
        or not all(
            not isinstance(index, bool) and isinstance(index, int) and 0 <= index < len(entries)
            for index in indices
        )
        or len(indices) != len(set(indices))
        or (fidelity is not None and not isinstance(fidelity, dict))
    ):
        return False
    compressed_gate_up.layers = {entries[index][0]: entries[index][1] for index in indices}
    compressed_gate_up.repack_bytes = _compressed_gate_up_repack_bytes(compressed_gate_up.layers)
    compressed_gate_up.calibrated = True
    compressed_gate_up.selection = selection
    compressed_gate_up.layer_indices = tuple(indices)
    compressed_gate_up.calibration_fidelity = fidelity
    return True


def _write_compressed_gate_up_calibration(compressed_gate_up, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_compressed_gate_up_calibration_cache_path, {})
    payload[key] = {
        "fidelity": compressed_gate_up.calibration_fidelity,
        "layer_indices": compressed_gate_up.layer_indices,
        "selection": compressed_gate_up.selection,
    }
    atomic_write_json(_compressed_gate_up_calibration_cache_path, payload)


def _calibrate_compressed_gate_up(model, sample_tokens, compressed_gate_up, decode_steps):
    if compressed_gate_up.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_gate_up.layers.items())
    if not entries:
        compressed_gate_up.calibrated = True
        compressed_gate_up.selection = "native"
        return
    key = _compressed_gate_up_calibration_key(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
    )
    with _compressed_gate_up_calibration_lock:
        if _restore_compressed_gate_up_calibration(compressed_gate_up, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_gate_up=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_gate_up.patched_classes.clear()
            compressed_gate_up.layers = {entries[index][0]: entries[index][1] for index in indices}
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_gate_up=compressed_gate_up,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_gate_up.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_gate_up.patched_classes.clear()
    compressed_gate_up.layers = {entries[index][0]: entries[index][1] for index in selected_indices}
    compressed_gate_up.repack_bytes = _compressed_gate_up_repack_bytes(compressed_gate_up.layers)
    compressed_gate_up.calibrated = True
    compressed_gate_up.selection = selected_name
    compressed_gate_up.layer_indices = selected_indices
    compressed_gate_up.calibration_fidelity = selected_fidelity
    with _compressed_gate_up_calibration_lock:
        _write_compressed_gate_up_calibration(compressed_gate_up, key)
    gc.collect()
    mx.clear_cache()


def _compressed_gate_up_implementation_key(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_gate_up.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "selection": compressed_gate_up.selection,
            "source": stable_digest(
                {
                    "backend": mlx_affine_swiglu_backend_signature(),
                    "class": inspect.getsource(MLXCompressedGateUp.fused_patched_class),
                    "fidelity": inspect.getsource(MLXCompressedGateUp.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_gate_up),
                    "select": inspect.getsource(_select_compressed_gate_up_implementation),
                    "switch_margin": _COMPRESSED_GATE_UP_FUSION_MARGIN,
                }
            ),
            "weights": tuple(
                (
                    gate_weight.shape,
                    gate_weight.group_size,
                    up_weight.shape,
                    up_weight.group_size,
                )
                for _, _, gate_weight, _, up_weight in compressed_gate_up.layers.values()
            ),
        }
    )


def _write_compressed_gate_up_implementation(key, record):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    with _compressed_gate_up_implementation_lock:
        payload = read_json(_compressed_gate_up_implementation_cache_path, {})
        payload[key] = record
        atomic_write_json(_compressed_gate_up_implementation_cache_path, payload)


def _select_compressed_gate_up_implementation(
    model,
    sample_tokens,
    compressed_gate_up,
    decode_steps,
    trials,
):
    compressed_gate_up.implementation = "projected"
    if not compressed_gate_up.layers or not any(
        _supports_compressed_gate_up_fusion(module)
        for module, *_ in compressed_gate_up.layers.values()
    ):
        compressed_gate_up.implementation_tuning = {
            "implementation": "projected",
            "reason": "no_supported_fusion",
        }
        return
    key = _compressed_gate_up_implementation_key(
        model,
        sample_tokens,
        compressed_gate_up,
        decode_steps,
    )
    cached = None
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_gate_up_implementation_lock:
            cached = read_json(_compressed_gate_up_implementation_cache_path, {}).get(key)
    if isinstance(cached, dict) and cached.get("implementation") in {"fused", "projected"}:
        compressed_gate_up.implementation = cached["implementation"]
        compressed_gate_up.implementation_tuning = {**cached, "cached": True}
        return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )
    plan = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    compressed_gate_up.implementation = "fused"
    actual = _run_compressed_calibration_candidate(
        model,
        sample_tokens,
        reference,
        decode_steps,
        plan,
        compressed_gate_up=compressed_gate_up,
    )
    fidelity = _logit_fidelity(reference.full_reference, actual)
    if not compressed_gate_up.fidelity_compatible(fidelity):
        record = {
            "cached": False,
            "fidelity": fidelity,
            "implementation": "projected",
            "reason": "fidelity",
        }
        compressed_gate_up.implementation = "projected"
        compressed_gate_up.implementation_tuning = record
        _write_compressed_gate_up_implementation(key, record)
        return

    prepared_prompt = _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps)
    decode_tokens = prepared_prompt[2]
    implementations = ("projected", "fused")
    samples = {implementation: [] for implementation in implementations}
    for round_index in range(max(3, trials)):
        order = implementations if round_index % 2 == 0 else tuple(reversed(implementations))
        for implementation in order:
            compressed_gate_up.implementation = implementation
            measurement, _ = _time_mlx_lm_plan(
                model,
                sample_tokens,
                plan,
                None,
                None,
                decode_steps,
                compressed_gate_up=compressed_gate_up,
                prepared_prompt=prepared_prompt,
                decode_tokens=decode_tokens,
            )
            samples[implementation].append(measurement[1])
    medians = {
        implementation: statistics.median(values) for implementation, values in samples.items()
    }
    selected = (
        "fused"
        if medians["fused"] < medians["projected"] * (1.0 - _COMPRESSED_GATE_UP_FUSION_MARGIN)
        else "projected"
    )
    record = {
        "cached": False,
        "fidelity": fidelity,
        "implementation": selected,
        "median_nanoseconds": {
            implementation: round(value * 1e9) for implementation, value in medians.items()
        },
        "reason": "timing",
    }
    compressed_gate_up.implementation = selected
    compressed_gate_up.implementation_tuning = record
    _write_compressed_gate_up_implementation(key, record)


def _compressed_attention_repack_bytes(layers):
    return sum(weight.nbytes for _, projections in layers.values() for _, weight in projections)


def _repack_compressed_attention_group(compressed_attention, group_size):
    import mlx.core as mx

    compressed_attention.layers = {}
    compressed_attention.patched_classes.clear()
    compressed_attention.calibrated = False
    compressed_attention.selection = "all"
    compressed_attention.layer_indices = ()
    compressed_attention.calibration_fidelity = None
    gc.collect()
    mx.clear_cache()
    layers = {
        key: (
            attention,
            tuple(
                (
                    module,
                    MLXCompressedDownWeight.quantize(
                        weight,
                        format="affine8",
                        group_size=group_size,
                    ),
                )
                for module, weight in projections
            ),
        )
        for key, (attention, projections) in compressed_attention.source_layers.items()
    }
    compressed_attention.layers = layers
    compressed_attention.group_size = group_size
    compressed_attention.repack_bytes = _compressed_attention_repack_bytes(layers)


def _compressed_attention_group_key(model, sample_tokens, compressed_attention, decode_steps):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_tuning": {
                name: value
                for name, value in compressed_attention.group_tuning.items()
                if name != "cached"
            },
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "calibrate": inspect.getsource(_calibrate_compressed_attention),
                    "region_policy": _compressed_region_policy_signature(),
                    "repack": inspect.getsource(_repack_compressed_attention_group),
                    "select": inspect.getsource(_select_model_affine8_group),
                    "tune": inspect.getsource(_autotune_compressed_attention_group),
                }
            ),
            "weights": tuple(
                tuple((weight.shape, str(weight.dtype)) for _, weight in projections)
                for _, projections in compressed_attention.source_layers.values()
            ),
        }
    )


def _autotune_compressed_attention_group(
    model,
    sample_tokens,
    compressed_attention,
    decode_steps,
):
    tuning = compressed_attention.group_tuning
    if tuning is None or tuning.get("model_calibrated") or not compressed_attention.source_layers:
        return
    timings_payload = tuning.get("median_nanoseconds")
    native_timing = tuning.get("native_median_nanoseconds")
    if not isinstance(timings_payload, dict) or not isinstance(native_timing, int):
        return
    timings = {int(group): int(value) for group, value in timings_payload.items()}
    key = _compressed_attention_group_key(
        model,
        sample_tokens,
        compressed_attention,
        decode_steps,
    )
    cached = None
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_attention_group_lock:
            cached = read_json(_compressed_attention_group_cache_path, {}).get(key)
    if isinstance(cached, dict) and cached.get("group_size") in timings:
        selected = cached["group_size"]
        _repack_compressed_attention_group(compressed_attention, selected)
        _calibrate_compressed_attention(
            model,
            sample_tokens,
            compressed_attention,
            decode_steps,
        )
        compressed_attention.group_tuning = {
            **tuning,
            **cached,
            "cached": True,
            "group_size": selected,
            "micro_group_size": tuning["group_size"],
            "model_calibrated": True,
        }
        return

    total_layers = len(compressed_attention.source_layers)
    candidates = {}
    layer_counts = {}
    for group in sorted(timings):
        _repack_compressed_attention_group(compressed_attention, group)
        _calibrate_compressed_attention(
            model,
            sample_tokens,
            compressed_attention,
            decode_steps,
        )
        layer_counts[group] = compressed_attention.layer_count
        candidates[str(group)] = {
            "fidelity": compressed_attention.calibration_fidelity,
            "layers": compressed_attention.layer_count,
            "selection": compressed_attention.selection,
        }
    selected, estimates = _select_model_affine8_group(
        total_layers,
        layer_counts,
        timings,
        native_timing,
    )
    if compressed_attention.group_size != selected:
        _repack_compressed_attention_group(compressed_attention, selected)
        _calibrate_compressed_attention(
            model,
            sample_tokens,
            compressed_attention,
            decode_steps,
        )
    record = {
        "group_size": selected,
        "model_calibrated": True,
        "model_candidates": candidates,
        "predicted_nanoseconds": {str(group): round(value) for group, value in estimates.items()},
    }
    compressed_attention.group_tuning = {
        **tuning,
        **record,
        "cached": False,
        "micro_group_size": tuning["group_size"],
    }
    if os.environ.get("METILE_DISABLE_DISK_CACHE") != "1":
        with _compressed_attention_group_lock:
            payload = read_json(_compressed_attention_group_cache_path, {})
            payload[key] = record
            atomic_write_json(_compressed_attention_group_cache_path, payload)


def _compressed_attention_calibration_key(
    model,
    sample_tokens,
    compressed_attention,
    decode_steps,
):
    import mlx.core as mx

    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_attention.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "search_decode_steps": _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_attention),
                    "class": inspect.getsource(MLXCompressedAttention.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedAttention.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_attention),
                    "region_policy": _compressed_region_policy_signature(),
                    "restore": inspect.getsource(_restore_compressed_attention_calibration),
                    "write": inspect.getsource(_write_compressed_attention_calibration),
                }
            ),
            "weights": tuple(
                tuple((weight.shape, weight.group_size) for _, weight in projections)
                for _, projections in compressed_attention.layers.values()
            ),
        }
    )


def _restore_compressed_attention_calibration(compressed_attention, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return False
    record = read_json(_compressed_attention_calibration_cache_path, {}).get(key)
    if not isinstance(record, dict):
        return False
    selection = record.get("selection")
    indices = record.get("layer_indices")
    fidelity = record.get("fidelity")
    entries = tuple(compressed_attention.layers.items())
    if (
        not isinstance(selection, str)
        or not isinstance(indices, list)
        or not all(
            not isinstance(index, bool) and isinstance(index, int) and 0 <= index < len(entries)
            for index in indices
        )
        or len(indices) != len(set(indices))
        or (fidelity is not None and not isinstance(fidelity, dict))
    ):
        return False
    compressed_attention.layers = {entries[index][0]: entries[index][1] for index in indices}
    compressed_attention.repack_bytes = _compressed_attention_repack_bytes(
        compressed_attention.layers
    )
    compressed_attention.calibrated = True
    compressed_attention.selection = selection
    compressed_attention.layer_indices = tuple(indices)
    compressed_attention.calibration_fidelity = fidelity
    return True


def _write_compressed_attention_calibration(compressed_attention, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_compressed_attention_calibration_cache_path, {})
    payload[key] = {
        "fidelity": compressed_attention.calibration_fidelity,
        "layer_indices": compressed_attention.layer_indices,
        "selection": compressed_attention.selection,
    }
    atomic_write_json(_compressed_attention_calibration_cache_path, payload)


def _calibrate_compressed_attention(model, sample_tokens, compressed_attention, decode_steps):
    if compressed_attention.calibrated:
        return
    import mlx.core as mx

    entries = tuple(compressed_attention.layers.items())
    if not entries:
        compressed_attention.calibrated = True
        compressed_attention.selection = "native"
        return
    key = _compressed_attention_calibration_key(
        model,
        sample_tokens,
        compressed_attention,
        decode_steps,
    )
    with _compressed_attention_calibration_lock:
        if _restore_compressed_attention_calibration(compressed_attention, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )

    plan = MLXLMPlan(False, False, False, False, compressed_attention=True)

    def make_evaluator(expected, steps):
        evaluations = {}

        def evaluate(_name, indices):
            cached = evaluations.get(indices)
            if cached is not None:
                return cached
            compressed_attention.patched_classes.clear()
            compressed_attention.layers = {
                entries[index][0]: entries[index][1] for index in indices
            }
            actual = _run_compressed_calibration_candidate(
                model,
                sample_tokens,
                reference,
                steps,
                plan,
                compressed_attention=compressed_attention,
            )
            fidelity = _logit_fidelity(expected, actual)
            result = compressed_attention.fidelity_compatible(fidelity), fidelity
            evaluations[indices] = result
            return result

        return evaluate

    search_evaluate = make_evaluator(reference.search_reference, reference.search_steps)
    selected_name, selected_indices, selected_fidelity = _select_compressed_region(
        len(entries),
        search_evaluate,
        augmentation_budget=0,
    )
    if decode_steps > reference.search_steps:
        full_evaluate = make_evaluator(reference.full_reference, decode_steps)
        compatible = False
        if selected_indices:
            compatible, selected_fidelity = full_evaluate(selected_name, selected_indices)
        selected_name, selected_indices, selected_fidelity = _audit_larger_compressed_regions(
            len(entries),
            full_evaluate,
            (selected_name, selected_indices, selected_fidelity),
            selected_compatible=compatible,
        )

    compressed_attention.patched_classes.clear()
    compressed_attention.layers = {
        entries[index][0]: entries[index][1] for index in selected_indices
    }
    compressed_attention.repack_bytes = _compressed_attention_repack_bytes(
        compressed_attention.layers
    )
    compressed_attention.calibrated = True
    compressed_attention.selection = selected_name
    compressed_attention.layer_indices = selected_indices
    compressed_attention.calibration_fidelity = selected_fidelity
    with _compressed_attention_calibration_lock:
        _write_compressed_attention_calibration(compressed_attention, key)
    gc.collect()
    mx.clear_cache()


def _compressed_vocab_calibration_key(
    model,
    sample_tokens,
    compressed_vocab,
    decode_steps,
):
    import mlx.core as mx

    weight = compressed_vocab.weight
    return stable_digest(
        {
            "architecture": mx.device_info().get("architecture"),
            "decode_steps": decode_steps,
            "group_size": compressed_vocab.group_size,
            "mlx": mx.__version__,
            "model": _mlx_lm_model_signature(model),
            "prompt": sample_tokens.tolist(),
            "source": stable_digest(
                {
                    "backend": mlx_compressed_down_backend_signature(),
                    "calibrate": inspect.getsource(_calibrate_compressed_vocab),
                    "class": inspect.getsource(MLXCompressedVocab.patched_class),
                    "fidelity": inspect.getsource(MLXCompressedVocab.fidelity_compatible),
                    "patch": inspect.getsource(_patch_compressed_vocab),
                }
            ),
            "tied": compressed_vocab.tied,
            "weight": (weight.shape, weight.format, weight.group_size),
        }
    )


def _restore_compressed_vocab_calibration(compressed_vocab, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return False
    record = read_json(_compressed_vocab_calibration_cache_path, {}).get(key)
    if not isinstance(record, dict):
        return False
    enabled = record.get("enabled")
    fidelity = record.get("fidelity")
    if not isinstance(enabled, bool) or (fidelity is not None and not isinstance(fidelity, dict)):
        return False
    if not enabled:
        compressed_vocab.weight = None
        compressed_vocab.repack_bytes = 0
    compressed_vocab.calibrated = True
    compressed_vocab.calibration_fidelity = fidelity
    return True


def _write_compressed_vocab_calibration(compressed_vocab, key):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_compressed_vocab_calibration_cache_path, {})
    payload[key] = {
        "enabled": compressed_vocab.projection_count > 0,
        "fidelity": compressed_vocab.calibration_fidelity,
    }
    atomic_write_json(_compressed_vocab_calibration_cache_path, payload)


def _calibrate_compressed_vocab(model, sample_tokens, compressed_vocab, decode_steps):
    if compressed_vocab.calibrated or compressed_vocab.weight is None:
        return
    import mlx.core as mx

    key = _compressed_vocab_calibration_key(
        model,
        sample_tokens,
        compressed_vocab,
        decode_steps,
    )
    with _compressed_vocab_calibration_lock:
        if _restore_compressed_vocab_calibration(compressed_vocab, key):
            return

    reference = _prepare_compressed_calibration_reference(
        model,
        sample_tokens,
        decode_steps,
    )
    plan = MLXLMPlan(False, False, False, False, compressed_vocab=True)
    actual = _run_compressed_calibration_candidate(
        model,
        sample_tokens,
        reference,
        decode_steps,
        plan,
        compressed_vocab=compressed_vocab,
    )
    fidelity = _logit_fidelity(reference.full_reference, actual)
    if not compressed_vocab.fidelity_compatible(fidelity):
        compressed_vocab.weight = None
        compressed_vocab.repack_bytes = 0
    compressed_vocab.patched_classes.clear()
    compressed_vocab.calibrated = True
    compressed_vocab.calibration_fidelity = fidelity
    with _compressed_vocab_calibration_lock:
        _write_compressed_vocab_calibration(compressed_vocab, key)
    gc.collect()
    mx.clear_cache()


def _cache_aware_dense_fidelity(model, sample_tokens, dense_mlp, implementation):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    dense_mlp.implementation = implementation
    reference_cache = make_prompt_cache(model)
    actual_cache = make_prompt_cache(model)
    reference_prefix = model(sample_tokens[:, :-1], cache=reference_cache)
    mx.eval(reference_prefix)
    reference = model(sample_tokens[:, -1:], cache=reference_cache)
    mx.eval(reference)
    dense_plan = MLXLMPlan(False, False, False, False, False, True)
    with apply_metile_to_mlx_lm(model=model, plan=dense_plan, dense_mlp=dense_mlp):
        actual_prefix = model(sample_tokens[:, :-1], cache=actual_cache)
        mx.eval(actual_prefix)
        actual = model(sample_tokens[:, -1:], cache=actual_cache)
        mx.eval(actual)
    return _logit_fidelity(reference, actual)


def _time_dense_mlp_implementation(model, sample_tokens, dense_mlp, implementation):
    import mlx.core as mx

    dense_mlp.implementation = implementation
    dense_plan = MLXLMPlan(False, False, False, False, False, True)
    start = time.perf_counter_ns()
    with apply_metile_to_mlx_lm(model=model, plan=dense_plan, dense_mlp=dense_mlp):
        output = model(sample_tokens)
        mx.eval(output)
    return (time.perf_counter_ns() - start) * 1e-9


def _select_dense_mlp_implementation(model, sample_tokens, dense_mlp, trials):
    compatible = []
    for implementation in ("fused", "projected"):
        try:
            fidelity = _cache_aware_dense_fidelity(
                model,
                sample_tokens,
                dense_mlp,
                implementation,
            )
        except (RuntimeError, TypeError, ValueError):
            continue
        exact_fusion = implementation != "fused" or (
            fidelity["kl_divergence"] == 0.0
            and fidelity["mean_logit_error"] == 0.0
            and fidelity["max_logit_error"] == 0.0
        )
        if _fidelity_compatible(fidelity) and exact_fusion:
            compatible.append(implementation)
    if not compatible:
        dense_mlp.implementation = "native"
        return

    for implementation in compatible:
        _time_dense_mlp_implementation(model, sample_tokens, dense_mlp, implementation)
    samples = {implementation: [] for implementation in compatible}
    for round_index in range(max(3, min(trials, 7))):
        ordered = compatible if round_index % 2 == 0 else tuple(reversed(compatible))
        for implementation in ordered:
            samples[implementation].append(
                _time_dense_mlp_implementation(
                    model,
                    sample_tokens,
                    dense_mlp,
                    implementation,
                )
            )
    dense_mlp.implementation = min(
        compatible,
        key=lambda implementation: statistics.median(samples[implementation]),
    )


def _plan_preserves_logits(
    model,
    sample_tokens,
    plan,
    affine_prefill,
    dense_mlp,
    reference,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    import mlx.core as mx

    decode_compression = any(
        (
            plan.compressed_down,
            plan.compressed_gate_up,
            plan.compressed_vocab,
            plan.compressed_attention,
        )
    )
    if decode_compression and sample_tokens.shape[1] > 1:
        from mlx_lm.models.cache import make_prompt_cache

        reference_cache = make_prompt_cache(model)
        reference_prefix = model(sample_tokens[:, :-1], cache=reference_cache)
        mx.eval(reference_prefix)
        reference = model(sample_tokens[:, -1:], cache=reference_cache)
        mx.eval(reference)

    with apply_metile_to_mlx_lm(
        model=model,
        plan=plan,
        affine_prefill=affine_prefill,
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
    ):
        if decode_compression and sample_tokens.shape[1] > 1:
            actual_cache = make_prompt_cache(model)
            actual_prefix = model(sample_tokens[:, :-1], cache=actual_cache)
            mx.eval(actual_prefix)
            actual = model(sample_tokens[:, -1:], cache=actual_cache)
        else:
            actual = model(sample_tokens)
        mx.eval(actual)
    fidelity = _logit_fidelity(reference, actual)
    policies = []
    if plan.compressed_down and compressed_down is not None:
        policies.append(compressed_down.fidelity_compatible)
    if plan.compressed_gate_up and compressed_gate_up is not None:
        policies.append(compressed_gate_up.fidelity_compatible)
    if plan.compressed_vocab and compressed_vocab is not None:
        policies.append(compressed_vocab.fidelity_compatible)
    if plan.compressed_attention and compressed_attention is not None:
        policies.append(compressed_attention.fidelity_compatible)
    return (
        all(policy(fidelity) for policy in policies) if policies else _fidelity_compatible(fidelity)
    )


def _measure_mlx_lm_plans(
    model,
    sample_tokens,
    candidates,
    affine_prefill,
    dense_mlp,
    decode_steps,
    rounds,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    *,
    prepared_prompt=None,
    validate_fidelity=True,
):
    import mlx.core as mx

    samples = {plan: [] for plan in candidates}
    expected_token = None
    compatible = set(candidates)
    if validate_fidelity:
        reference = model(sample_tokens)
        mx.eval(reference)
        for plan in candidates:
            if not plan.feature_count:
                continue
            try:
                if not _plan_preserves_logits(
                    model,
                    sample_tokens,
                    plan,
                    affine_prefill,
                    dense_mlp,
                    reference,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                ):
                    compatible.remove(plan)
            except (RuntimeError, TypeError, ValueError):
                compatible.remove(plan)
    if prepared_prompt is None:
        prepared_prompt = (
            _prepare_mlx_lm_prompt(model, sample_tokens, decode_steps)
            if sample_tokens.shape[1] > 1
            and any(
                not plan.feature_count or _is_decode_only_compression_plan(plan)
                for plan in compatible
            )
            else None
        )
    for round_index in range(rounds):
        shift = round_index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if round_index & 1:
            ordered = tuple(reversed(ordered))
        for plan in ordered:
            if plan not in compatible:
                continue
            try:
                measurement, next_token = _time_mlx_lm_plan(
                    model,
                    sample_tokens,
                    plan,
                    affine_prefill,
                    dense_mlp,
                    decode_steps,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                    prepared_prompt=(
                        prepared_prompt
                        if not plan.feature_count or _is_decode_only_compression_plan(plan)
                        else None
                    ),
                    decode_tokens=(prepared_prompt[2] if prepared_prompt is not None else None),
                )
            except (RuntimeError, TypeError, ValueError):
                if plan.feature_count == 0:
                    raise
                compatible.remove(plan)
                continue
            if expected_token is None:
                expected_token = next_token
            if next_token != expected_token:
                compatible.remove(plan)
                continue
            samples[plan].append(measurement)
    return {plan: values for plan, values in samples.items() if values and plan in compatible}


def _extend_mlx_lm_measurements(
    model,
    sample_tokens,
    measured,
    candidates,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    *,
    prepared_prompt=None,
):
    existing_trials = min(len(measured[plan]) for plan in candidates)
    remaining_trials = max(0, trials - existing_trials)
    if not remaining_trials:
        return {plan: measured[plan] for plan in candidates}
    additional = _measure_mlx_lm_plans(
        model,
        sample_tokens,
        candidates,
        affine_prefill,
        dense_mlp,
        decode_steps,
        remaining_trials,
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
        prepared_prompt=prepared_prompt,
        validate_fidelity=False,
    )
    return {plan: measured[plan] + additional[plan] for plan in candidates if plan in additional}


def _rank_mlx_lm_plans(samples):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = samples[native]
    generated = []
    for plan, measurements in samples.items():
        if not plan.feature_count:
            continue
        ttft_ratios = _paired_plan_ratios(measurements, native_measurements, 0)
        decode_ratios = _paired_plan_ratios(measurements, native_measurements, 1)
        total_ratios = _paired_plan_ratios(measurements, native_measurements, 2)
        required_wins = max(1, (len(total_ratios) * 2 + 2) // 3)
        ttft_median = statistics.median(ttft_ratios)
        decode_median = statistics.median(decode_ratios)
        total_median = statistics.median(total_ratios)
        improves_total = total_median < 1.0 - _MODEL_SWITCH_MARGIN
        improves_decode = decode_median < 1.0 - _MODEL_DECODE_SWITCH_MARGIN
        improves_ttft = ttft_median < 1.0 - _MODEL_TTFT_SWITCH_MARGIN
        decode_only = _is_decode_only_compression_plan(plan)
        decode_sensitive = any(
            (
                plan.attention,
                plan.rms_norm,
                plan.graph_fusion,
                plan.quantized_mlp,
                plan.dense_mlp,
                plan.dense_residual,
                plan.compressed_down,
                plan.compressed_gate_up,
                plan.compressed_vocab,
                plan.compressed_attention,
            )
        )
        decode_limit = 1.0 + (_MODEL_REGRESSION_MARGIN if decode_sensitive else 0.05)
        strong_decode_win = (
            decode_sensitive
            and decode_median <= 1.0 - _MODEL_STRONG_DECODE_SWITCH_MARGIN
            and sum(ratio < 1.0 for ratio in total_ratios) >= required_wins
        )
        stable_ttft = decode_only or (
            sum(ratio <= 1.01 for ratio in ttft_ratios) >= required_wins or strong_decode_win
        )
        if strong_decode_win:
            ttft_margin = _MODEL_STRONG_DECODE_TTFT_REGRESSION_MARGIN
        else:
            ttft_margin = _MODEL_REGRESSION_MARGIN
        ttft_limit = 1.0 + ttft_margin
        if (
            (decode_only or ttft_median <= ttft_limit)
            and decode_median <= decode_limit
            and total_median <= 1.0 + _MODEL_REGRESSION_MARGIN
            and (improves_total or improves_decode or improves_ttft)
            and stable_ttft
            and sum(ratio <= 1.05 for ratio in decode_ratios) >= required_wins
            and any(
                (
                    improves_total and sum(ratio < 1.0 for ratio in total_ratios) >= required_wins,
                    improves_decode
                    and sum(ratio < 1.0 for ratio in decode_ratios) >= required_wins,
                    improves_ttft and sum(ratio < 0.98 for ratio in ttft_ratios) >= required_wins,
                )
            )
        ):
            objective = (
                min(total_median, decode_median)
                if decode_only
                else min(total_median, decode_median, ttft_median)
            )
            generated.append((objective, plan.feature_count * 64, plan))
    ranked = []
    while generated:
        selected = choose_mdl_tie(generated)
        ranked.append(selected)
        generated = [candidate for candidate in generated if candidate[2] != selected]
    return tuple(ranked)


def _choose_mlx_lm_plan(samples):
    ranked = _rank_mlx_lm_plans(samples)
    return ranked[0] if ranked else MLXLMPlan(False, False, False, False)


def _median_plan_measurement(measurements):
    return tuple(statistics.median(values) for values in zip(*measurements))


def _paired_plan_ratios(measurements, native_measurements, metric):
    return tuple(
        measurement[metric] / native[metric]
        for measurement, native in zip(measurements, native_measurements)
    )


def _is_decode_only_compression_plan(plan):
    return plan.is_decode_only_compression


def _compression_ladder(plans, decode_ratios):
    singleton_names = []
    for plan in sorted(
        (
            plan
            for plan in plans
            if _is_decode_only_compression_plan(plan) and plan.feature_count == 1
        ),
        key=decode_ratios.__getitem__,
    ):
        singleton_names.extend(name for name, enabled in plan.as_dict().items() if enabled)
    selected = []
    enabled = set()
    available = set(plans)
    for name in singleton_names:
        enabled.add(name)
        candidate = MLXLMPlan(**{feature: feature in enabled for feature in MLXLMPlan().as_dict()})
        if candidate in available:
            selected.append(candidate)
    return tuple(selected)


def _validate_mlx_lm_plan(
    model,
    sample_tokens,
    selected,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    if not selected.feature_count:
        return selected
    native = MLXLMPlan(False, False, False, False)
    measured = _measure_mlx_lm_plans(
        model,
        sample_tokens,
        (native, selected),
        affine_prefill,
        dense_mlp,
        max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        max(_MODEL_VALIDATION_MIN_TRIALS, trials),
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
    )
    return _choose_mlx_lm_plan(measured)


def _validate_mlx_lm_plan_repeated(
    model,
    sample_tokens,
    selected,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    native = MLXLMPlan(False, False, False, False)
    for _ in range(_MODEL_VALIDATION_ATTEMPTS):
        validated = _validate_mlx_lm_plan(
            model,
            sample_tokens,
            selected,
            affine_prefill,
            dense_mlp,
            decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )
        if validated.feature_count:
            return validated
    return native


def _mlx_lm_validation_finalists(measured):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = measured[native]
    ranked = _rank_mlx_lm_plans(measured)
    decode_order = sorted(
        (plan for plan in measured if plan.feature_count),
        key=lambda plan: statistics.median(
            _paired_plan_ratios(measured[plan], native_measurements, 1)
        ),
    )
    compressed_order = sorted(
        (plan for plan in measured if _is_decode_only_compression_plan(plan)),
        key=lambda plan: (
            -plan.feature_count,
            statistics.median(_paired_plan_ratios(measured[plan], native_measurements, 1)),
        ),
    )
    decode_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 1))
        for plan, samples in measured.items()
    }
    return tuple(
        dict.fromkeys(
            (
                native,
                *ranked[:2],
                *decode_order[:3],
                *compressed_order[:1],
                *_compression_ladder(measured, decode_ratios),
            )
        )
    )


def _mlx_lm_validation_survivors(measured):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = measured[native]
    ranked = _rank_mlx_lm_plans(measured)
    metric_leaders = tuple(
        min(
            measured,
            key=lambda plan: statistics.median(
                _paired_plan_ratios(measured[plan], native_measurements, metric)
            ),
        )
        for metric in (1, 2, 0)
    )
    ordered = tuple(dict.fromkeys((native, *ranked[:2], *metric_leaders)))
    return ordered[:_MODEL_VALIDATION_MAX_SURVIVORS]


def _validate_mlx_lm_finalists_repeated(
    model,
    sample_tokens,
    finalists,
    affine_prefill,
    dense_mlp,
    decode_steps,
    trials,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
    native = MLXLMPlan(False, False, False, False)
    validation_decode_steps = max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4)
    validation_trials = max(_MODEL_VALIDATION_MIN_TRIALS, trials)
    prepared_prompt = (
        _prepare_mlx_lm_prompt(model, sample_tokens, validation_decode_steps)
        if getattr(sample_tokens, "ndim", None) == 2 and sample_tokens.shape[1] > 1
        else None
    )
    for _ in range(_MODEL_VALIDATION_ATTEMPTS):
        screening_trials = (
            min(_MODEL_VALIDATION_SCREEN_TRIALS, validation_trials)
            if len(finalists) > _MODEL_VALIDATION_MAX_SURVIVORS
            else validation_trials
        )
        screening = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            finalists,
            affine_prefill,
            dense_mlp,
            validation_decode_steps,
            screening_trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        survivors = (
            _mlx_lm_validation_survivors(screening)
            if screening_trials < validation_trials
            else tuple(screening)
        )
        measured = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            screening,
            survivors,
            affine_prefill,
            dense_mlp,
            validation_decode_steps,
            validation_trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        selected = _choose_mlx_lm_plan(measured)
        if selected.feature_count:
            return selected
    return native


def _provisional_mlx_lm_finalists(
    provisional,
    candidates,
    *,
    max_finalists=_MODEL_PROVISIONAL_MAX_FINALISTS,
    relative_margin=_MODEL_PROVISIONAL_RELATIVE_MARGIN,
):
    native = MLXLMPlan(False, False, False, False)
    native_measurements = provisional[native]
    total_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 2))
        for plan, samples in provisional.items()
    }
    ttft_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 0))
        for plan, samples in provisional.items()
    }
    decode_ratios = {
        plan: statistics.median(_paired_plan_ratios(samples, native_measurements, 1))
        for plan, samples in provisional.items()
    }
    best_total = min(total_ratios.values())
    best_ttft = min(ttft_ratios.values())
    best_decode = min(decode_ratios.values())
    fastest_total = min(total_ratios, key=total_ratios.__getitem__)
    fastest_ttft = min(ttft_ratios, key=ttft_ratios.__getitem__)
    fastest_decode = min(decode_ratios, key=decode_ratios.__getitem__)
    compressed = tuple(
        plan
        for plan in candidates
        if plan in total_ratios and _is_decode_only_compression_plan(plan)
    )
    maximal_compression = (
        min(
            compressed,
            key=lambda plan: (-plan.feature_count, decode_ratios[plan]),
        )
        if compressed
        else native
    )
    required = {
        native,
        fastest_total,
        fastest_ttft,
        fastest_decode,
        maximal_compression,
        *_compression_ladder(provisional, decode_ratios),
        *(
            plan
            for plan in candidates
            if plan in total_ratios
            and plan.feature_count == 1
            and min(total_ratios[plan], ttft_ratios[plan], decode_ratios[plan]) < 0.99
        ),
    }
    eligible = sorted(
        (
            plan
            for plan in candidates
            if plan in total_ratios
            and (
                total_ratios[plan] <= best_total * (1.0 + relative_margin)
                or ttft_ratios[plan] <= best_ttft * (1.0 + relative_margin)
                or decode_ratios[plan] <= best_decode * (1.0 + relative_margin)
            )
        ),
        key=lambda plan: min(
            total_ratios[plan] / best_total,
            ttft_ratios[plan] / best_ttft,
            decode_ratios[plan] / best_decode,
        ),
    )
    selected = set(required)
    for plan in eligible:
        if len(selected) >= max_finalists:
            break
        selected.add(plan)
    return tuple(plan for plan in candidates if plan in selected and plan in total_ratios)


def autotune_metile_for_mlx_lm(
    model,
    sample_tokens,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
    affine_prefill=None,
    dense_mlp=None,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    decode_steps=8,
    trials=5,
):
    """Choose a persistent feature plan by timing the real MLX-LM decode graph."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if sample_tokens.ndim != 2 or sample_tokens.shape[1] < 1:
        raise ValueError("sample_tokens must have shape [batch, sequence]")
    if decode_steps < 1 or trials < 1:
        raise ValueError("decode_steps and trials must be positive")
    if affine_prefill is not None and not isinstance(affine_prefill, MLXAffinePrefill):
        raise TypeError("affine_prefill must be an MLXAffinePrefill")
    if affine_prefill is not None and affine_prefill.model is not model:
        raise ValueError("affine_prefill was prepared for a different model")
    if dense_mlp is not None and not isinstance(dense_mlp, MLXDenseMLP):
        raise TypeError("dense_mlp must be an MLXDenseMLP")
    if dense_mlp is not None and dense_mlp.model is not model:
        raise ValueError("dense_mlp was prepared for a different model")
    if compressed_down is not None and not isinstance(compressed_down, MLXCompressedDown):
        raise TypeError("compressed_down must be an MLXCompressedDown")
    if compressed_down is not None and compressed_down.model is not model:
        raise ValueError("compressed_down was prepared for a different model")
    if compressed_gate_up is not None and not isinstance(compressed_gate_up, MLXCompressedGateUp):
        raise TypeError("compressed_gate_up must be an MLXCompressedGateUp")
    if compressed_gate_up is not None and compressed_gate_up.model is not model:
        raise ValueError("compressed_gate_up was prepared for a different model")
    if compressed_vocab is not None and not isinstance(compressed_vocab, MLXCompressedVocab):
        raise TypeError("compressed_vocab must be an MLXCompressedVocab")
    if compressed_vocab is not None and compressed_vocab.model is not model:
        raise ValueError("compressed_vocab was prepared for a different model")
    if compressed_attention is not None and not isinstance(
        compressed_attention, MLXCompressedAttention
    ):
        raise TypeError("compressed_attention must be an MLXCompressedAttention")
    if compressed_attention is not None and compressed_attention.model is not model:
        raise ValueError("compressed_attention was prepared for a different model")
    if compressed_down is not None:
        _calibrate_compressed_down(
            model,
            sample_tokens,
            compressed_down,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if compressed_gate_up is not None:
        _autotune_compressed_gate_up_group(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _calibrate_compressed_gate_up(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _select_compressed_gate_up_implementation(
            model,
            sample_tokens,
            compressed_gate_up,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
            trials,
        )
    if compressed_vocab is not None:
        _calibrate_compressed_vocab(
            model,
            sample_tokens,
            compressed_vocab,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if compressed_attention is not None:
        _autotune_compressed_attention_group(
            model,
            sample_tokens,
            compressed_attention,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
        _calibrate_compressed_attention(
            model,
            sample_tokens,
            compressed_attention,
            max(_MODEL_VALIDATION_MIN_DECODE_STEPS, decode_steps * 4),
        )
    if dense_mlp is not None and sample_tokens.shape[1] >= dense_mlp.min_rows:
        _select_dense_mlp_implementation(model, sample_tokens, dense_mlp, trials)
    requested = MLXLMPlan(
        attention=attention,
        rms_norm=rms_norm,
        graph_fusion=graph_fusion,
        quantized_mlp=quantized_mlp,
        affine_prefill=affine_prefill is not None,
        dense_mlp=dense_mlp is not None,
        dense_residual=dense_mlp is not None,
        compressed_down=compressed_down is not None and compressed_down.projection_count > 0,
        compressed_gate_up=compressed_gate_up is not None
        and compressed_gate_up.projection_count > 0,
        compressed_vocab=compressed_vocab is not None and compressed_vocab.projection_count > 0,
        compressed_attention=compressed_attention is not None
        and compressed_attention.projection_count > 0,
    )
    key = _mlx_lm_plan_key(
        model,
        sample_tokens,
        requested,
        affine_prefill,
        dense_mlp,
        decode_steps,
        trials,
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
    )
    with _mlx_lm_plan_lock:
        cached = _read_mlx_lm_plan(key)
        if cached is not None:
            return cached

        search_decode_steps = max(_MODEL_SEARCH_MIN_DECODE_STEPS, decode_steps)
        candidates = _mlx_lm_plan_candidates(requested)
        prepared_prompt = (
            _prepare_mlx_lm_prompt(model, sample_tokens, search_decode_steps)
            if sample_tokens.shape[1] > 1
            and any(
                not plan.feature_count or _is_decode_only_compression_plan(plan)
                for plan in candidates
            )
            else None
        )
        _measure_mlx_lm_plans(
            model,
            sample_tokens,
            _mlx_lm_warmup_plans(candidates),
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            1,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        candidates = tuple(
            dict.fromkeys(
                _effective_mlx_lm_plan(
                    plan,
                    affine_prefill,
                    dense_mlp,
                    compressed_down,
                    compressed_gate_up,
                    compressed_vocab,
                    compressed_attention,
                )
                for plan in candidates
            )
        )
        screening = _measure_mlx_lm_plans(
            model,
            sample_tokens,
            candidates,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            _MODEL_SCREEN_ROUNDS,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        screened = _provisional_mlx_lm_finalists(
            screening,
            candidates,
            max_finalists=_MODEL_SCREEN_MAX_FINALISTS,
            relative_margin=_MODEL_SCREEN_RELATIVE_MARGIN,
        )
        provisional = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            screening,
            screened,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            _MODEL_PROVISIONAL_ROUNDS,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        finalists = _provisional_mlx_lm_finalists(provisional, screened)
        measured = _extend_mlx_lm_measurements(
            model,
            sample_tokens,
            provisional,
            finalists,
            affine_prefill,
            dense_mlp,
            search_decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
            prepared_prompt=prepared_prompt,
        )
        selected = _validate_mlx_lm_finalists_repeated(
            model,
            sample_tokens,
            _mlx_lm_validation_finalists(measured),
            affine_prefill,
            dense_mlp,
            decode_steps,
            trials,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )
        _write_mlx_lm_plan(key, selected)
        return selected


# Shapes whose decode kernel cannot be built on this device, learned by trying once.
#
# The shape gate below cannot express every hardware constraint. A head dimension of 256,
# which Qwen3.5, Qwen3.6 and Qwen3-VL all use, satisfies every condition it checks and then
# needs 40960 bytes of threadgroup memory against a 32768-byte limit, so the kernel raises at
# build time. Before this cache that RuntimeError escaped into the caller's generate loop:
# enabling meTile attention on those models crashed instead of falling back.
#
# Deriving a head-dimension bound arithmetically was the alternative and is worse: it would
# hardcode the current kernel's allocation formula into the gate, and drift silently the first
# time the kernel's tiling changes. Recording the failure per shape is self-calibrating and
# costs one build attempt for each shape that cannot work.
#
# Head dimension 256 was the case that motivated this and no longer needs it. The kernel was
# always fine there; the tuner offered a block size that could not fit and failed the whole shape
# instead of pruning it, so attention fell back for a reason that had nothing to do with the
# shape being unsupported. Tuners now prune on OutOfResources. This set stays as the backstop for
# shapes that genuinely cannot run, which is what it was for.
_unsupported_decode_shapes = set()


def _decode_shape_key(queries, keys):
    return (queries.shape[1], keys.shape[1], queries.shape[-1], str(queries.dtype))


def _supports_metile_decode(queries, keys, values, cache, mask, sinks):
    return (
        not hasattr(cache, "bits")
        and mask is None
        and sinks is None
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[2] == 1
        and keys.shape == values.shape
        and queries.shape[0] == keys.shape[0]
        and queries.shape[1] % keys.shape[1] == 0
        and queries.shape[-1] == keys.shape[-1]
        and queries.shape[-1] % 32 == 0
        and queries.dtype == keys.dtype == values.dtype
        and str(queries.dtype) in ("mlx.core.bfloat16", "mlx.core.float16", "mlx.core.float32")
        and _decode_shape_key(queries, keys) not in _unsupported_decode_shapes
    )


def _supports_metile_rms_norm(values, weight):
    return (
        values.ndim >= 1
        and weight.ndim == 1
        and values.shape[-1] == weight.shape[0]
        and values.dtype == weight.dtype
        and str(values.dtype) in ("mlx.core.bfloat16", "mlx.core.float16", "mlx.core.float32")
    )


def _supports_metile_residual_rms_norm(values, residual, norm):
    weight = norm["weight"]
    return (
        values.shape == residual.shape
        and values.dtype == residual.dtype
        and _supports_metile_rms_norm(values, weight)
    )


def _tensor_spec(value):
    dtype = {
        "mlx.core.bfloat16": "bf16",
        "mlx.core.float16": "f16",
        "mlx.core.float32": "f32",
    }[str(value.dtype)]
    return TensorSpec(tuple(value.shape), dtype)


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


def _patch_graph_fusion(
    model,
    replacements,
    quantized_linear=None,
    dense_mlp=None,
    dense_swiglu=False,
    *,
    quantized_mlp_min_rows=1,
    quantized_mlp_max_rows=None,
    fuse_residual_rms=True,
):
    classes = []
    if model is not None:
        classes.extend(type(layer) for layer in _model_layers(model))
    else:
        classes.extend(_registry_classes(_FUSED_BLOCK_CLASSES))

    for block_class in dict.fromkeys(classes):
        if not _recognised(block_class, _FUSED_BLOCK_CLASSES):
            continue
        original = block_class.__call__
        if getattr(original, "_metile_original", None) is not None:
            continue
        quantized_support_cache = {}

        def make_replacement(original_call, support_cache):
            def replacement(self, values, mask=None, cache=None):
                supports_quantized_residual = False
                supports_dense_residual = False
                if (
                    quantized_linear is not None
                    and hasattr(values, "size")
                    and getattr(values, "shape", ())
                    and hasattr(self, "mlp")
                ):
                    rows = values.size // values.shape[-1]
                    support_key = id(self.mlp)
                    support = support_cache.get(support_key)
                    if (
                        support is None
                        or support[0] is not self.mlp
                        or support[1] != values.shape[-1]
                        or support[2] != values.dtype
                    ):
                        support = (
                            self.mlp,
                            values.shape[-1],
                            values.dtype,
                            _supports_quantized_residual_mlp(
                                self.mlp,
                                values,
                                values,
                                quantized_linear,
                            ),
                        )
                        support_cache[support_key] = support
                    supports_quantized_residual = (
                        rows >= quantized_mlp_min_rows
                        and (quantized_mlp_max_rows is None or rows <= quantized_mlp_max_rows)
                        and support[3]
                    )
                if (
                    dense_mlp is not None
                    and hasattr(values, "size")
                    and getattr(values, "shape", ())
                    and hasattr(self, "mlp")
                ):
                    supports_dense_residual = _supports_dense_residual_mlp(
                        self.mlp,
                        values,
                        values,
                        dense_mlp,
                    )
                selected = (
                    mlx_add_rms_norm_selection(values, self.post_attention_layernorm.eps)
                    if fuse_residual_rms
                    else None
                )
                if (
                    not fuse_residual_rms
                    and not supports_quantized_residual
                    and not supports_dense_residual
                ):
                    return original_call(self, values, mask, cache)
                if (
                    selected is not None
                    and selected.algorithm == "mlx"
                    and not supports_quantized_residual
                    and not supports_dense_residual
                ):
                    return original_call(self, values, mask, cache)

                # A block binding its attention to neither name is one this replacement cannot
                # reproduce, so hand it back rather than guess. Checked here rather than at patch
                # time because a hybrid architecture binds different names on different layers.
                attention = _attention_module(self)
                if attention is None:
                    return original_call(self, values, mask, cache)
                attention_output = attention(self.input_layernorm(values), mask, cache)
                if (
                    fuse_residual_rms
                    and (selected is None or selected.algorithm != "mlx")
                    and _supports_metile_residual_rms_norm(
                        values, attention_output, self.post_attention_layernorm
                    )
                ):
                    hidden, normalized = _execute_residual_rms_graph(
                        values, attention_output, self.post_attention_layernorm
                    )
                else:
                    hidden = values + attention_output
                    normalized = self.post_attention_layernorm(hidden)
                if supports_quantized_residual:
                    return _execute_quantized_mlp(self.mlp, normalized, hidden)
                if supports_dense_residual:
                    return _execute_dense_mlp(
                        self.mlp,
                        normalized,
                        hidden,
                        dense_mlp,
                        dense_swiglu,
                    )
                return hidden + self.mlp(normalized)

            return replacement

        metile_transformer_block = make_replacement(original, quantized_support_cache)

        metile_transformer_block._metile_original = original
        replacements.append((block_class, "__call__", original))
        block_class.__call__ = metile_transformer_block


def _model_layers(model):
    if model is None:
        return ()
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", ())
    return layers


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


def _patch_quantized_mlp(
    model,
    replacements,
    quantized_linear,
    *,
    min_rows=1,
    max_rows=None,
):
    if min_rows < 1:
        raise ValueError("quantized MLP minimum rows must be positive")
    if max_rows is not None and max_rows < min_rows:
        raise ValueError("quantized MLP maximum rows must not be smaller than its minimum")
    classes = [type(layer.mlp) for layer in _model_layers(model) if hasattr(layer, "mlp")]
    if model is None:
        classes.extend(_registry_classes(_GATED_MLP_CLASSES))

    for mlp_class in dict.fromkeys(classes):
        if not _recognised(mlp_class, _GATED_MLP_CLASSES):
            continue
        original = mlp_class.__call__
        if getattr(original, "_metile_original", None) is not None:
            continue

        def make_replacement(original_call):
            def replacement(self, values):
                rows = values.size // values.shape[-1]
                if rows < min_rows:
                    type(self).__call__ = original_call
                    return original_call(self, values)
                if max_rows is not None and rows > max_rows:
                    return original_call(self, values)
                if not _supports_quantized_mlp(self, values, quantized_linear):
                    return original_call(self, values)
                return _execute_quantized_mlp(self, values)

            return replacement

        metile_mlp = make_replacement(original)
        metile_mlp._metile_original = original
        replacements.append((mlp_class, "__call__", original))
        mlp_class.__call__ = metile_mlp


def _patch_affine_prefill(affine_prefill, replacements):
    if affine_prefill is None:
        return
    for module, _ in affine_prefill.weights.values():
        patched_class = affine_prefill.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", affine_prefill.patched_class(module))


def _patch_dense_mlp(dense_mlp, replacements):
    if dense_mlp is None:
        return
    for module, *_ in dense_mlp.weights.values():
        patched_class = dense_mlp.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", dense_mlp.patched_class(module))


def _patch_compressed_down(compressed_down, replacements):
    if compressed_down is None:
        return
    for module, _ in compressed_down.weights.values():
        patched_class = compressed_down.patched_classes.get(id(module))
        original_class = type(module)
        if original_class is patched_class:
            continue
        replacements.append((module, "__class__", original_class))
        object.__setattr__(module, "__class__", compressed_down.patched_class(module))


def _supports_compressed_gate_up_fusion(module):
    return _recognised(type(module), _GATED_MLP_CLASSES) and callable(
        getattr(module, "down_proj", None)
    )


def _patch_compressed_gate_up(compressed_gate_up, replacements):
    if compressed_gate_up is None:
        return
    for module, gate, _, up, _ in compressed_gate_up.layers.values():
        if compressed_gate_up.implementation == "fused" and _supports_compressed_gate_up_fusion(
            module
        ):
            patched_class = compressed_gate_up.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(
                module,
                "__class__",
                compressed_gate_up.fused_patched_class(module),
            )
            continue
        for module in (gate, up):
            patched_class = compressed_gate_up.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(module, "__class__", compressed_gate_up.patched_class(module))


def _patch_compressed_attention(compressed_attention, replacements):
    if compressed_attention is None:
        return
    for _, projections in compressed_attention.layers.values():
        for module, _ in projections:
            patched_class = compressed_attention.patched_classes.get(id(module))
            original_class = type(module)
            if original_class is patched_class:
                continue
            replacements.append((module, "__class__", original_class))
            object.__setattr__(module, "__class__", compressed_attention.patched_class(module))


def _patch_compressed_vocab(compressed_vocab, replacements):
    if compressed_vocab is None or compressed_vocab.weight is None:
        return
    module = compressed_vocab.module
    patched_class = compressed_vocab.patched_classes.get(id(module))
    original_class = type(module)
    if original_class is patched_class:
        return
    replacements.append((module, "__class__", original_class))
    object.__setattr__(module, "__class__", compressed_vocab.patched_class())


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


def prepare_mlx_lm_compressed_gate_up(
    model,
    *,
    group_size="auto",
    max_working_set_fraction=_COMPRESSED_WORKING_SET_FRACTION,
):
    """AOT-compress dense gate/up pairs for guarded one-row decode."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if group_size not in {"auto", 32, 64, 128}:
        raise ValueError("affine8 group size must be auto, 32, 64, or 128")
    if not 0.0 < max_working_set_fraction <= 1.0:
        raise ValueError("max_working_set_fraction must be in (0, 1]")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Compressed gate/up preparation requires the optional 'mlx' package"
        ) from error

    supported = []
    for layer in _model_layers(model):
        module = getattr(layer, "mlp", None)
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        if (
            not isinstance(gate, nn.Linear)
            or not isinstance(up, nn.Linear)
            or "bias" in gate
            or "bias" in up
            or gate.weight.shape != up.weight.shape
            or gate.weight.shape[1] % 64
            or str(gate.weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16")
            or gate.weight.dtype != up.weight.dtype
        ):
            continue
        supported.append((module, gate, gate.weight, up, up.weight))
    if not supported:
        raise ValueError("model contains no supported dense gate/up pairs")

    group_tuning = None
    if group_size == "auto":
        group_size, group_tuning = tune_mlx_affine8_group_size(
            (
                weight
                for _, _, gate_weight, _, up_weight in supported
                for weight in (gate_weight, up_weight)
            ),
            objective="throughput",
        )

    dense_bytes = sum(
        gate_weight.nbytes + up_weight.nbytes for _, _, gate_weight, _, up_weight in supported
    )
    estimated_bytes = int(dense_bytes * 0.55)
    recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    budget = int(recommended * max_working_set_fraction)
    active = int(mx.get_active_memory())
    if recommended and active + estimated_bytes > budget:
        raise ValueError(
            "compressed gate/up AOT repack needs approximately "
            f"{estimated_bytes / 2**30:.2f} GiB with {active / 2**30:.2f} GiB active, "
            f"exceeding the {budget / 2**30:.2f} GiB working-set budget"
        )

    source_layers = {
        id(module): (module, gate, gate_weight, up, up_weight)
        for module, gate, gate_weight, up, up_weight in supported
    }
    layers = {
        key: (
            module,
            gate,
            MLXCompressedDownWeight.quantize(
                gate_weight,
                format="affine8",
                group_size=group_size,
            ),
            up,
            MLXCompressedDownWeight.quantize(
                up_weight,
                format="affine8",
                group_size=group_size,
            ),
        )
        for key, (module, gate, gate_weight, up, up_weight) in source_layers.items()
    }
    repack_bytes = _compressed_gate_up_repack_bytes(layers)
    return MLXCompressedGateUp(
        model,
        layers,
        repack_bytes,
        group_size,
        group_tuning,
        source_layers=source_layers,
    )


def prepare_mlx_lm_compressed_attention(
    model,
    *,
    group_size="auto",
    max_working_set_fraction=_COMPRESSED_WORKING_SET_FRACTION,
):
    """AOT-compress Q/K/V/output projections for guarded one-row decode."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if group_size not in {"auto", 32, 64, 128}:
        raise ValueError("affine8 group size must be auto, 32, 64, or 128")
    if not 0.0 < max_working_set_fraction <= 1.0:
        raise ValueError("max_working_set_fraction must be in (0, 1]")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Compressed attention preparation requires the optional 'mlx' package"
        ) from error

    projection_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    supported = []
    for layer in _model_layers(model):
        attention = getattr(layer, "self_attn", None)
        modules = tuple(getattr(attention, name, None) for name in projection_names)
        if (
            not all(isinstance(module, nn.Linear) for module in modules)
            or any(module.weight.shape[1] % 64 for module in modules)
            or any(
                str(module.weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16")
                for module in modules
            )
            or len({module.weight.dtype for module in modules}) != 1
        ):
            continue
        supported.append(
            (
                attention,
                tuple((module, module.weight) for module in modules),
            )
        )
    if not supported:
        raise ValueError("model contains no supported dense attention projections")

    dense_weights = tuple(weight for _, projections in supported for _, weight in projections)
    group_tuning = None
    if group_size == "auto":
        group_size, group_tuning = tune_mlx_affine8_group_size(
            dense_weights,
            objective="balanced",
        )

    dense_bytes = sum(weight.nbytes for weight in dense_weights)
    estimated_bytes = int(dense_bytes * 0.55)
    recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    budget = int(recommended * max_working_set_fraction)
    active = int(mx.get_active_memory())
    if recommended and active + estimated_bytes > budget:
        raise ValueError(
            "compressed attention AOT repack needs approximately "
            f"{estimated_bytes / 2**30:.2f} GiB with {active / 2**30:.2f} GiB active, "
            f"exceeding the {budget / 2**30:.2f} GiB working-set budget"
        )

    source_layers = {
        id(attention): (attention, projections) for attention, projections in supported
    }
    layers = {
        key: (
            attention,
            tuple(
                (
                    module,
                    MLXCompressedDownWeight.quantize(
                        weight,
                        format="affine8",
                        group_size=group_size,
                    ),
                )
                for module, weight in projections
            ),
        )
        for key, (attention, projections) in source_layers.items()
    }
    repack_bytes = _compressed_attention_repack_bytes(layers)
    return MLXCompressedAttention(
        model,
        layers,
        repack_bytes,
        group_size,
        group_tuning,
        source_layers=source_layers,
    )


def prepare_mlx_lm_compressed_vocab(
    model,
    *,
    group_size="auto",
    max_working_set_fraction=_COMPRESSED_WORKING_SET_FRACTION,
):
    """AOT-compress a tied embedding or LM head for guarded one-row decode."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if group_size not in {"auto", 32, 64, 128}:
        raise ValueError("affine8 group size must be auto, 32, 64, or 128")
    if not 0.0 < max_working_set_fraction <= 1.0:
        raise ValueError("max_working_set_fraction must be in (0, 1]")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Compressed vocabulary preparation requires the optional 'mlx' package"
        ) from error

    tied = bool(getattr(getattr(model, "args", None), "tie_word_embeddings", False))
    if tied:
        module = getattr(getattr(model, "model", None), "embed_tokens", None)
        supported = isinstance(module, nn.Embedding)
    else:
        module = getattr(model, "lm_head", None)
        supported = isinstance(module, nn.Linear) and "bias" not in module
    weight = getattr(module, "weight", None)
    if (
        not supported
        or weight is None
        or weight.ndim != 2
        or weight.shape[-1] % 64
        or str(weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16")
    ):
        raise ValueError("model contains no supported dense vocabulary projection")

    group_tuning = None
    if group_size == "auto":
        group_size, group_tuning = tune_mlx_affine8_group_size(
            (weight,),
            objective="throughput",
        )
    estimated_bytes = int(weight.nbytes * 0.55)
    recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    budget = int(recommended * max_working_set_fraction)
    active = int(mx.get_active_memory())
    if recommended and active + estimated_bytes > budget:
        raise ValueError(
            "compressed vocabulary AOT repack needs approximately "
            f"{estimated_bytes / 2**30:.2f} GiB with {active / 2**30:.2f} GiB active, "
            f"exceeding the {budget / 2**30:.2f} GiB working-set budget"
        )

    compressed = MLXCompressedDownWeight.quantize(
        weight,
        format="affine8",
        group_size=group_size,
    )
    return MLXCompressedVocab(
        model,
        module,
        compressed,
        tied,
        compressed.nbytes,
        group_size,
        group_tuning,
    )


def prepare_mlx_lm_compressed_down(
    model,
    *,
    format="affine8",
    group_size="auto",
    allow_approximate=False,
    max_working_set_fraction=_COMPRESSED_WORKING_SET_FRACTION,
):
    """AOT-compress dense down projections for guarded decode-only dispatch."""
    if not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    if format not in {"affine8", "mxfp8"}:
        raise ValueError("compressed down format must be affine8 or mxfp8")
    if format == "affine8" and group_size not in {"auto", 32, 64, 128}:
        raise ValueError("affine8 group size must be auto, 32, 64, or 128")
    if format == "mxfp8" and not allow_approximate:
        raise ValueError("mxfp8 down projection requires allow_approximate=True")
    if not 0.0 < max_working_set_fraction <= 1.0:
        raise ValueError("max_working_set_fraction must be in (0, 1]")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "Compressed down preparation requires the optional 'mlx' package"
        ) from error

    supported = []
    for layer in _model_layers(model):
        module = getattr(layer, "mlp", None)
        down = getattr(module, "down_proj", None)
        if (
            not isinstance(down, nn.Linear)
            or "bias" in down
            or down.weight.shape[1] % 64
            or str(down.weight.dtype) not in ("mlx.core.bfloat16", "mlx.core.float16")
        ):
            continue
        supported.append((down, down.weight))
    if not supported:
        raise ValueError("model contains no supported dense down projections")

    group_tuning = None
    if format == "affine8" and group_size == "auto":
        group_size, group_tuning = tune_mlx_affine8_group_size(
            (weight for _, weight in supported),
            objective="throughput",
        )
    elif format == "mxfp8":
        group_size = 32

    dense_bytes = sum(weight.nbytes for _, weight in supported)
    estimated_bytes = int(dense_bytes * (0.55 if format == "affine8" else 0.52))
    recommended = int(mx.device_info().get("max_recommended_working_set_size", 0))
    budget = int(recommended * max_working_set_fraction)
    active = int(mx.get_active_memory())
    if recommended and active + estimated_bytes > budget:
        raise ValueError(
            f"compressed down AOT repack needs approximately {estimated_bytes / 2**30:.2f} GiB "
            f"with {active / 2**30:.2f} GiB active, exceeding the "
            f"{budget / 2**30:.2f} GiB working-set budget"
        )

    weights = {}
    for module, weight in supported:
        compressed = MLXCompressedDownWeight.quantize(
            weight,
            format=format,
            group_size=group_size,
        )
        weights[id(module)] = (module, compressed)
    repack_bytes = sum(weight.nbytes for _, weight in weights.values())
    return MLXCompressedDown(
        model,
        weights,
        format,
        repack_bytes,
        allow_approximate,
        32 if format == "mxfp8" else group_size,
        group_tuning,
    )


def apply_metile_to_mlx_lm(
    model=None,
    *,
    attention=True,
    rms_norm=True,
    graph_fusion=True,
    quantized_mlp=True,
    affine_prefill=None,
    dense_mlp=None,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
    plan=None,
):
    """Patch MLX-LM with zero-copy, autotuned meTile primitives.

    Decode attention, RMSNorm, dense/quantized SwiGLU, and compute-graph fusion are independently
    selectable. Unsupported calls preserve MLX-LM's original implementation.
    The returned handle can restore every changed module or be used as a context
    manager.
    """
    if model is not None and not callable(model):
        raise TypeError("model must be an MLX-LM callable")
    dense_swiglu = dense_mlp
    dense_residual = dense_mlp
    active_compressed_down = compressed_down
    active_compressed_gate_up = compressed_gate_up
    active_compressed_vocab = compressed_vocab
    active_compressed_attention = compressed_attention
    if plan is not None:
        if not isinstance(plan, MLXLMPlan):
            raise TypeError("plan must be an MLXLMPlan")
        attention = attention and plan.attention
        rms_norm = rms_norm and plan.rms_norm
        graph_fusion = graph_fusion and plan.graph_fusion
        quantized_mlp = quantized_mlp and plan.quantized_mlp
        if not plan.affine_prefill:
            affine_prefill = None
        if not plan.dense_mlp:
            dense_swiglu = None
        if not plan.dense_residual:
            dense_residual = None
        if not plan.compressed_down:
            active_compressed_down = None
        if not plan.compressed_gate_up:
            active_compressed_gate_up = None
        if not plan.compressed_vocab:
            active_compressed_vocab = None
        if not plan.compressed_attention:
            active_compressed_attention = None
    if affine_prefill is not None:
        if not isinstance(affine_prefill, MLXAffinePrefill):
            raise TypeError("affine_prefill must be an MLXAffinePrefill")
        if model is not affine_prefill.model:
            raise ValueError("affine_prefill was prepared for a different model")
    if dense_mlp is not None:
        if not isinstance(dense_mlp, MLXDenseMLP):
            raise TypeError("dense_mlp must be an MLXDenseMLP")
        if model is not dense_mlp.model:
            raise ValueError("dense_mlp was prepared for a different model")
    if compressed_down is not None:
        if not isinstance(compressed_down, MLXCompressedDown):
            raise TypeError("compressed_down must be an MLXCompressedDown")
        if model is not compressed_down.model:
            raise ValueError("compressed_down was prepared for a different model")
    if compressed_gate_up is not None:
        if not isinstance(compressed_gate_up, MLXCompressedGateUp):
            raise TypeError("compressed_gate_up must be an MLXCompressedGateUp")
        if model is not compressed_gate_up.model:
            raise ValueError("compressed_gate_up was prepared for a different model")
    if compressed_vocab is not None:
        if not isinstance(compressed_vocab, MLXCompressedVocab):
            raise TypeError("compressed_vocab must be an MLXCompressedVocab")
        if model is not compressed_vocab.model:
            raise ValueError("compressed_vocab was prepared for a different model")
    if compressed_attention is not None:
        if not isinstance(compressed_attention, MLXCompressedAttention):
            raise TypeError("compressed_attention must be an MLXCompressedAttention")
        if model is not compressed_attention.model:
            raise ValueError("compressed_attention was prepared for a different model")
    if (
        not attention
        and not rms_norm
        and not graph_fusion
        and not quantized_mlp
        and affine_prefill is None
        and dense_swiglu is None
        and dense_residual is None
        and active_compressed_down is None
        and active_compressed_gate_up is None
        and active_compressed_vocab is None
        and active_compressed_attention is None
    ):
        return MLXPatch([])
    try:
        import mlx.nn as nn
        from mlx_lm.models import base
    except ImportError as error:
        raise ImportError(
            "The MLX-LM integration requires the optional 'mlx-lm' package"
        ) from error

    replacements = []
    attention_replacement = None
    attention_original = None
    if attention:
        attention_original = base.scaled_dot_product_attention
        if getattr(attention_original, "_metile_original", None) is None:

            def metile_scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache,
                scale,
                mask,
                sinks=None,
            ):
                if _supports_metile_decode(queries, keys, values, cache, mask, sinks):
                    try:
                        return _mlx_attention_decode_unchecked(
                            queries,
                            keys,
                            values,
                            scale,
                        )
                    except RuntimeError:
                        # The kernel cannot be built for this shape on this device, most
                        # often because its threadgroup memory exceeds the limit. Record the
                        # shape so later tokens skip the attempt, and serve this one from MLX.
                        # Falling back is the whole point: a shape meTile cannot handle must
                        # cost speed, never correctness or a crash.
                        _unsupported_decode_shapes.add(_decode_shape_key(queries, keys))
                return attention_original(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )

            metile_scaled_dot_product_attention._metile_original = attention_original
            attention_replacement = metile_scaled_dot_product_attention
            for module in tuple(sys.modules.values()):
                if module is None or not getattr(module, "__name__", "").startswith(
                    "mlx_lm.models"
                ):
                    continue
                if getattr(module, "scaled_dot_product_attention", None) is attention_original:
                    replacements.append(
                        (module, "scaled_dot_product_attention", attention_original)
                    )
                    module.scaled_dot_product_attention = attention_replacement

    if rms_norm:
        original_rms_norm = nn.RMSNorm.__call__
        if getattr(original_rms_norm, "_metile_original", None) is None:

            def metile_rms_norm(self, values):
                weight = self["weight"]
                if _supports_metile_rms_norm(values, weight):
                    return mlx_rms_norm(values, weight, self.eps)
                return original_rms_norm(self, values)

            metile_rms_norm._metile_original = original_rms_norm
            replacements.append((nn.RMSNorm, "__call__", original_rms_norm))
            nn.RMSNorm.__call__ = metile_rms_norm

    quantized_mlp_prefill_min_rows = _QUANTIZED_MLP_MIN_ROWS if affine_prefill is not None else 1
    quantized_mlp_prefill_max_rows = None if affine_prefill is not None else 1
    if graph_fusion or quantized_mlp or dense_residual is not None:
        _patch_graph_fusion(
            model,
            replacements,
            nn.QuantizedLinear if quantized_mlp else None,
            dense_mlp=dense_residual,
            dense_swiglu=dense_swiglu is not None,
            quantized_mlp_min_rows=1,
            quantized_mlp_max_rows=1,
            fuse_residual_rms=graph_fusion,
        )
    if quantized_mlp:
        _patch_quantized_mlp(
            model,
            replacements,
            nn.QuantizedLinear,
            min_rows=quantized_mlp_prefill_min_rows,
            max_rows=quantized_mlp_prefill_max_rows,
        )
    _patch_affine_prefill(affine_prefill, replacements)
    _patch_dense_mlp(dense_swiglu, replacements)
    _patch_compressed_down(active_compressed_down, replacements)
    _patch_compressed_gate_up(active_compressed_gate_up, replacements)
    _patch_compressed_vocab(active_compressed_vocab, replacements)
    _patch_compressed_attention(active_compressed_attention, replacements)

    return MLXPatch(replacements, attention_replacement, attention_original)


__all__ = [
    "MLXAffinePrefill",
    "MLXCompressedAttention",
    "MLXCompressedDown",
    "MLXCompressedGateUp",
    "MLXCompressedVocab",
    "MLXDenseMLP",
    "MLXLMPlan",
    "MLXPatch",
    "apply_metile_to_mlx_lm",
    "autotune_metile_for_mlx_lm",
    "prepare_mlx_lm_affine_prefill",
    "prepare_mlx_lm_compressed_attention",
    "prepare_mlx_lm_compressed_down",
    "prepare_mlx_lm_compressed_gate_up",
    "prepare_mlx_lm_compressed_vocab",
    "prepare_mlx_lm_dense_mlp",
]
