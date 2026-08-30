"""Tunables, caches and locks shared across the mlx_lm integration.

Split out so every module can reach the same objects. The caches are mutated in place and
never rebound, so importing the name here is importing the one shared instance.
"""

from __future__ import annotations

import threading

from metile.runtime.cache import cache_root

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
