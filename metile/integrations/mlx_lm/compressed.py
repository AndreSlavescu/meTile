"""compressed layer of the mlx_lm integration."""

from __future__ import annotations

import copy
import gc
import os
from dataclasses import dataclass, field

from metile.backends.mlx_compressed_down import (
    MLXCompressedDownWeight,
    tune_mlx_affine8_group_size,
)
from metile.backends.mlx_quantized import (
    mlx_affine_swiglu_executor,
)
from metile.integrations.mlx_lm._state import (
    _COMPRESSED_AFFINE8_KL_LIMIT,
    _COMPRESSED_AFFINE8_MAX_LOGIT_ERROR_LIMIT,
    _COMPRESSED_AFFINE8_MEAN_LOGIT_ERROR_LIMIT,
    _COMPRESSED_APPROXIMATE_KL_LIMIT,
    _COMPRESSED_APPROXIMATE_MAX_LOGIT_ERROR_LIMIT,
    _COMPRESSED_APPROXIMATE_MEAN_LOGIT_ERROR_LIMIT,
    _COMPRESSED_CALIBRATION_SEARCH_DECODE_STEPS,
    _COMPRESSED_INTERVAL_DIRECTION_BUDGET,
    _COMPRESSED_SUBSET_AUGMENTATION_BUDGET,
    _COMPRESSED_WORKING_SET_FRACTION,
    _GATED_MLP_CLASSES,
    _compressed_attention_calibration_cache_path,
    _compressed_down_calibration_cache_path,
    _compressed_gate_up_calibration_cache_path,
    _compressed_gate_up_implementation_cache_path,
    _compressed_gate_up_implementation_lock,
    _compressed_vocab_calibration_cache_path,
)
from metile.integrations.mlx_lm.core import (
    _model_layers,
    _recognised,
)
from metile.runtime.cache import (
    atomic_write_json,
    read_cached_selection,
    read_json,
    write_cached_selection,
)


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


def _write_compressed_layer_calibration(path, region, key):
    """Persist which layers a compressed region settled on, and how well it scored."""
    write_cached_selection(
        path,
        key,
        {
            "fidelity": region.calibration_fidelity,
            "layer_indices": region.layer_indices,
            "selection": region.selection,
        },
    )


def _restore_compressed_layer_calibration(path, region, key, repack_bytes):
    """Reapply a persisted layer selection to `region`, reporting whether it was usable.

    The record is rejected rather than trusted when it does not describe a valid selection of
    the layers this model actually has, so a cache written for a different model or a changed
    layer count re-calibrates instead of silently compressing the wrong layers.
    """
    record = read_cached_selection(path, key)
    if record is None:
        return False
    selection = record.get("selection")
    indices = record.get("layer_indices")
    fidelity = record.get("fidelity")
    entries = tuple(region.layers.items())
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
    region.layers = {entries[index][0]: entries[index][1] for index in indices}
    region.repack_bytes = repack_bytes(region.layers)
    region.calibrated = True
    region.selection = selection
    region.layer_indices = tuple(indices)
    region.calibration_fidelity = fidelity
    return True


def _write_compressed_down_calibration(compressed_down, key):
    _write_compressed_layer_calibration(
        _compressed_down_calibration_cache_path, compressed_down, key
    )


def _write_compressed_gate_up_calibration(compressed_gate_up, key):
    _write_compressed_layer_calibration(
        _compressed_gate_up_calibration_cache_path, compressed_gate_up, key
    )


def _write_compressed_attention_calibration(compressed_attention, key):
    _write_compressed_layer_calibration(
        _compressed_attention_calibration_cache_path, compressed_attention, key
    )


def _write_compressed_vocab_calibration(compressed_vocab, key):
    write_cached_selection(
        _compressed_vocab_calibration_cache_path,
        key,
        {
            "enabled": compressed_vocab.projection_count > 0,
            "fidelity": compressed_vocab.calibration_fidelity,
        },
    )


def _restore_compressed_gate_up_calibration(compressed_gate_up, key):
    return _restore_compressed_layer_calibration(
        _compressed_gate_up_calibration_cache_path,
        compressed_gate_up,
        key,
        _compressed_gate_up_repack_bytes,
    )


def _restore_compressed_attention_calibration(compressed_attention, key):
    return _restore_compressed_layer_calibration(
        _compressed_attention_calibration_cache_path,
        compressed_attention,
        key,
        _compressed_attention_repack_bytes,
    )


def _compressed_gate_up_repack_bytes(layers):
    return sum(
        gate_weight.nbytes + up_weight.nbytes for _, _, gate_weight, _, up_weight in layers.values()
    )


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


def _write_compressed_gate_up_implementation(key, record):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    with _compressed_gate_up_implementation_lock:
        payload = read_json(_compressed_gate_up_implementation_cache_path, {})
        payload[key] = record
        atomic_write_json(_compressed_gate_up_implementation_cache_path, payload)


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


def _supports_compressed_gate_up_fusion(module):
    return _recognised(type(module), _GATED_MLP_CLASSES) and callable(
        getattr(module, "down_proj", None)
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
