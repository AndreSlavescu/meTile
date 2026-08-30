"""MLX backend tests: compressed."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tests.module_patching import _patch_mlx_lm


@pytest.mark.parametrize(
    ("format", "mean_limit", "maximum_limit"),
    (("affine8", 0.1, 0.3), ("mxfp8", 0.8, 2.1)),
)
def test_mlx_compressed_down_residual_matches_quantized_reference(
    format,
    mean_limit,
    maximum_limit,
):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_compressed_down import (
        MLXCompressedDownWeight,
        mlx_compressed_down_residual,
    )

    random = np.random.default_rng(2041)
    values = mx.array(random.normal(size=(1, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    dense = mx.array(random.normal(size=(64, 64)).astype(np.float32)).astype(mx.bfloat16)
    residual = mx.array(random.normal(size=(1, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
    weight = MLXCompressedDownWeight.quantize(dense, format=format)

    actual = mlx_compressed_down_residual(values, weight, residual)
    expected = values @ dense.T + residual
    mx.eval(actual, expected)

    assert actual.dtype == mx.bfloat16
    assert weight.nbytes < dense.nbytes
    error = np.abs(np.array(actual.astype(mx.float32)) - np.array(expected.astype(mx.float32)))
    assert float(error.mean()) < mean_limit
    assert float(error.max()) < maximum_limit


@pytest.mark.parametrize("group_size", (32, 64, 128))
def test_mlx_compressed_down_affine_group_sizes(group_size):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_compressed_down import MLXCompressedDownWeight

    dense = mx.ones((64, 128), dtype=mx.bfloat16)
    weight = MLXCompressedDownWeight.quantize(
        dense,
        format="affine8",
        group_size=group_size,
    )

    assert weight.group_size == group_size
    assert weight.shape == (64, 128)


def test_mlx_compressed_down_autotunes_strict_affine_groups(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_compressed_down

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_compressed_down._affine8_group_cache.clear()
    dense = mx.ones((64, 128), dtype=mx.bfloat16)

    group_size, tuning = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert group_size in {32, 64, 128}
    assert tuning["group_size"] == group_size
    assert not tuning["cached"]
    assert tuning["objective"] == "balanced"
    assert set(tuning["median_nanoseconds"]) == {"32", "64", "128"}
    assert all(value > 0 for value in tuning["median_nanoseconds"].values())
    assert tuning["native_median_nanoseconds"] > 0
    assert set(tuning["mean_absolute_error"]) == {"32", "64", "128"}
    assert all(value >= 0 for value in tuning["mean_absolute_error"].values())

    cached_group_size, cached = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert cached_group_size == group_size
    assert cached["cached"]


def test_mlx_compressed_down_autotune_omits_invalid_group128(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_compressed_down

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_compressed_down._affine8_group_cache.clear()
    dense = mx.ones((64, 192), dtype=mx.bfloat16)

    group_size, tuning = mlx_compressed_down.tune_mlx_affine8_group_size(
        (dense,),
        trials=3,
    )

    assert group_size in {32, 64}
    assert set(tuning["median_nanoseconds"]) == {"32", "64"}
    assert set(tuning["mean_absolute_error"]) == {"32", "64"}
    assert tuning["native_median_nanoseconds"] > 0


def test_compressed_calibration_candidate_reuses_native_prompt_cache(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    _patch_mlx_lm(monkeypatch, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    prompt_cache = SimpleNamespace(marker="native-prompt")
    reference = mlx_lm._CompressedCalibrationReference(
        mx.array([[9]]), prompt_cache, 2, object(), object()
    )

    mlx_lm._run_compressed_calibration_candidate(
        Model(),
        mx.array([[1, 2, 3]]),
        reference,
        2,
        mlx_lm.MLXLMPlan(False, False, False, False),
    )

    assert [values for values, _ in calls] == [(9,), (9,)]
    assert all(cache is calls[0][1] for _, cache in calls)
    assert calls[0][1] is not prompt_cache
    assert calls[0][1].marker == "native-prompt"


def test_compressed_calibration_candidate_replays_single_token_prompt(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import cache as cache_module

    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    _patch_mlx_lm(monkeypatch, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    monkeypatch.setattr(
        cache_module,
        "make_prompt_cache",
        lambda _model: SimpleNamespace(marker="fresh"),
    )
    reference = mlx_lm._CompressedCalibrationReference(mx.array([[9]]), None, 1, object(), object())

    mlx_lm._run_compressed_calibration_candidate(
        Model(),
        mx.array([[3]]),
        reference,
        1,
        mlx_lm.MLXLMPlan(False, False, False, False),
    )

    assert [values for values, _ in calls] == [(3,), (9,)]
    assert all(cache is calls[0][1] for _, cache in calls)
    assert calls[0][1].marker == "fresh"


def test_mlx_lm_plan_candidates_compose_compressed_projection_families():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            False,
            False,
            False,
            False,
            compressed_down=True,
            compressed_gate_up=True,
            compressed_vocab=True,
            compressed_attention=True,
        )
    )

    assert len(candidates) == 16
    assert any(
        plan.compressed_down
        and plan.compressed_gate_up
        and plan.compressed_vocab
        and plan.compressed_attention
        for plan in candidates
    )


def test_mlx_lm_compressed_subset_candidates_prefer_largest_simple_regions():
    from metile.integrations.mlx_lm import _compressed_down_subset_candidates

    assert tuple(_compressed_down_subset_candidates(4)) == (
        ("all", (0, 1, 2, 3)),
        ("suffix:3", (1, 2, 3)),
        ("prefix:3", (0, 1, 2)),
        ("suffix:2", (2, 3)),
        ("prefix:2", (0, 1)),
        ("suffix:1", (3,)),
        ("prefix:1", (0,)),
    )


def test_mlx_lm_compressed_region_search_finds_largest_boundary_logarithmically():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        limit = 21 if name.startswith("suffix") else 18
        return len(indices) <= limit, {"name": name}

    name, indices, fidelity = _select_compressed_region(28, evaluate)

    assert name == "suffix:21"
    assert indices == tuple(range(7, 28))
    assert fidelity == {"name": "suffix:21"}
    assert len(calls) <= 2 * 28


def test_mlx_lm_compressed_region_search_audits_nonmonotonic_islands():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        size = len(indices)
        compatible = (name.startswith("suffix") and (size <= 9 or size == 21)) or (
            name.startswith("prefix") and size <= 5
        )
        return compatible, {"name": name}

    name, indices, fidelity = _select_compressed_region(28, evaluate)

    assert name == "suffix:21"
    assert indices == tuple(range(7, 28))
    assert fidelity == {"name": "suffix:21"}
    assert len(calls) <= 2 * 28


def test_mlx_lm_compressed_region_search_short_circuits_full_model():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return True, {"name": name}

    assert _select_compressed_region(4, evaluate) == (
        "all",
        (0, 1, 2, 3),
        {"name": "all"},
    )
    assert len(calls) == 1


def test_mlx_lm_compressed_region_search_augments_noncontiguous_subset():
    from metile.integrations.mlx_lm import _select_compressed_region

    compatible = {
        (4,),
        (3, 4),
        (0, 3, 4),
    }

    def evaluate(name, indices):
        return indices in compatible, {"name": name}

    name, indices, fidelity = _select_compressed_region(5, evaluate)

    assert name == "subset:0,3,4"
    assert indices == (0, 3, 4)
    assert fidelity == {"name": "subset:0,3,4"}


def test_mlx_lm_compressed_region_search_can_preserve_interval_mask():
    from metile.integrations.mlx_lm import _select_compressed_region

    compatible = {
        (4,),
        (3, 4),
        (0, 3, 4),
    }

    def evaluate(name, indices):
        return indices in compatible, {"name": name}

    assert _select_compressed_region(5, evaluate, augmentation_budget=0) == (
        "suffix:2",
        (3, 4),
        {"name": "suffix:2"},
    )


def test_mlx_lm_compressed_region_search_bounds_subset_evaluations():
    from metile.integrations.mlx_lm import _augment_compressed_subset

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return False, None

    selected = ("suffix:2", (62, 63), {"name": "suffix:2"})

    assert _augment_compressed_subset(64, evaluate, selected, budget=7) == selected
    assert len(calls) == 7


def test_mlx_lm_compressed_region_search_bounds_interval_directions():
    from metile.integrations.mlx_lm import _select_compressed_region

    calls = []

    def evaluate(name, indices):
        calls.append((name, indices))
        return False, None

    assert _select_compressed_region(128, evaluate) == ("native", (), None)
    assert len(calls) <= 43


def test_mlx_lm_compressed_region_policy_signature_tracks_budgets(monkeypatch):
    from metile.integrations import mlx_lm

    first = mlx_lm._compressed_region_policy_signature()
    _patch_mlx_lm(
        monkeypatch,
        "_COMPRESSED_INTERVAL_DIRECTION_BUDGET",
        mlx_lm._COMPRESSED_INTERVAL_DIRECTION_BUDGET + 1,
    )

    assert mlx_lm._compressed_region_policy_signature() != first


def test_mlx_lm_compressed_region_full_audit_recovers_late_horizon_island():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        return name == "suffix:35", {"name": name}

    selected = ("suffix:18", tuple(range(18, 36)), {"name": "suffix:18"})
    name, indices, fidelity = _audit_larger_compressed_regions(36, evaluate, selected)

    assert name == "suffix:35"
    assert indices == tuple(range(1, 36))
    assert fidelity == {"name": "suffix:35"}
    assert len(calls) <= 9


def test_mlx_lm_compressed_region_full_audit_checks_opposite_edge_escape():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    def evaluate(name, indices):
        return name == "prefix:23", {"name": name}

    selected = ("suffix:7", tuple(range(17, 24)), {"name": "suffix:7"})

    assert _audit_larger_compressed_regions(24, evaluate, selected) == (
        "prefix:23",
        tuple(range(23)),
        {"name": "prefix:23"},
    )


def test_mlx_lm_compressed_region_full_audit_refines_failed_frontier_locally():
    from metile.integrations.mlx_lm import _audit_larger_compressed_regions

    calls = []

    def evaluate(name, indices):
        calls.append(name)
        return name == "suffix:19", {"name": name}

    selected = (
        "subset:3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23",
        (3, *range(6, 24)),
        {"name": "short-horizon"},
    )
    name, indices, fidelity = _audit_larger_compressed_regions(
        24,
        evaluate,
        selected,
        selected_compatible=False,
    )

    assert name == "suffix:19"
    assert indices == tuple(range(5, 24))
    assert fidelity == {"name": "suffix:19"}
    assert len(calls) <= 12


def test_mlx_lm_compressed_calibration_cache_restores_layer_mask(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    weights = {id(module): (module, SimpleNamespace(nbytes=100)) for module in modules}
    prepared = mlx_lm.MLXCompressedDown(object(), dict(weights), "affine8", 300)
    prepared.weights = dict(tuple(weights.items())[1:])
    prepared.repack_bytes = 200
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    _patch_mlx_lm(
        monkeypatch,
        "_compressed_down_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_down_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedDown(object(), dict(weights), "affine8", 300)

    assert mlx_lm._restore_compressed_down_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.projection_count == 2
    assert restored.repack_bytes == 200


def test_mlx_lm_compressed_gate_up_cache_restores_layer_pairs(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    layers = {
        id(module): (
            module,
            object(),
            SimpleNamespace(nbytes=100),
            object(),
            SimpleNamespace(nbytes=100),
        )
        for module in modules
    }
    prepared = mlx_lm.MLXCompressedGateUp(object(), dict(layers), 600)
    prepared.layers = dict(tuple(layers.items())[1:])
    prepared.repack_bytes = 400
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    _patch_mlx_lm(
        monkeypatch,
        "_compressed_gate_up_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_gate_up_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedGateUp(object(), dict(layers), 600)

    assert mlx_lm._restore_compressed_gate_up_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.layer_count == 2
    assert restored.projection_count == 4
    assert restored.repack_bytes == 400


def test_mlx_lm_compressed_attention_cache_restores_layer_groups(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    modules = tuple(object() for _ in range(3))
    layers = {
        id(module): (
            module,
            tuple((object(), SimpleNamespace(nbytes=100)) for _ in range(4)),
        )
        for module in modules
    }
    prepared = mlx_lm.MLXCompressedAttention(object(), dict(layers), 1200)
    prepared.layers = dict(tuple(layers.items())[1:])
    prepared.repack_bytes = 800
    prepared.calibrated = True
    prepared.selection = "suffix:2"
    prepared.layer_indices = (1, 2)
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 7}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    _patch_mlx_lm(
        monkeypatch,
        "_compressed_attention_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_attention_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedAttention(object(), dict(layers), 1200)

    assert mlx_lm._restore_compressed_attention_calibration(restored, "key")
    assert restored.selection == "suffix:2"
    assert restored.layer_indices == (1, 2)
    assert restored.layer_count == 2
    assert restored.projection_count == 8
    assert restored.repack_bytes == 800


def test_mlx_lm_compressed_vocab_cache_restores_rejection(tmp_path, monkeypatch):
    from metile.integrations import mlx_lm

    prepared = mlx_lm.MLXCompressedVocab(object(), object(), object(), True, 100)
    prepared.weight = None
    prepared.repack_bytes = 0
    prepared.calibrated = True
    prepared.calibration_fidelity = {"next_token": 7, "actual_next_token": 8}
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    _patch_mlx_lm(
        monkeypatch,
        "_compressed_vocab_calibration_cache_path",
        tmp_path / "calibration.json",
    )

    mlx_lm._write_compressed_vocab_calibration(prepared, "key")
    restored = mlx_lm.MLXCompressedVocab(object(), object(), object(), True, 100)

    assert mlx_lm._restore_compressed_vocab_calibration(restored, "key")
    assert restored.calibrated
    assert restored.projection_count == 0
    assert restored.repack_bytes == 0


def test_mlx_lm_compressed_down_patch_is_decode_only():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")

    from metile.integrations import mlx_lm

    class DenseProjection:
        def __call__(self, values):
            return values

    projection = DenseProjection()

    class Model:
        def __call__(self):
            pass

    model = Model()
    calls = []

    class CompressedWeight:
        shape = (64, 64)

        def __call__(self, values):
            calls.append(values)
            return "compressed"

    compressed = mlx_lm.MLXCompressedDown(
        model,
        {id(projection): (projection, CompressedWeight())},
        "affine8",
        1024,
    )

    values = mx.ones((1, 1, 64), dtype=mx.bfloat16)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_down=compressed,
        plan=mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            compressed_down=True,
        ),
    )
    try:
        decode = projection(values)
        prefill = projection(mx.ones((1, 2, 64), dtype=mx.bfloat16))
    finally:
        patch.restore()

    assert decode == "compressed"
    assert prefill.shape == (1, 2, 64)
    assert calls == [values]
    assert type(projection) is DenseProjection


def test_prepare_mlx_lm_compressed_down_quantizes_supported_projection():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations.mlx_lm import prepare_mlx_lm_compressed_down

    down_proj = nn.Linear(64, 64, bias=False)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)
    mlp = SimpleNamespace(down_proj=down_proj)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    model = Model()
    prepared = prepare_mlx_lm_compressed_down(model, group_size=32)

    assert prepared.model is model
    assert prepared.format == "affine8"
    assert prepared.group_size == 32
    assert prepared.projection_count == 1
    assert prepared.weight_for(down_proj).shape == (64, 64)
    assert prepared.repack_bytes < down_proj.weight.nbytes


def test_prepare_mlx_lm_compressed_down_autotunes_group(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    down_proj = nn.Linear(128, 64, bias=False)
    down_proj.weight = down_proj.weight.astype(mx.bfloat16)

    class Model:
        layers = (SimpleNamespace(mlp=SimpleNamespace(down_proj=down_proj)),)

        def __call__(self):
            pass

    model = Model()
    tuning = {
        "cached": False,
        "group_size": 128,
        "median_nanoseconds": {"32": 120, "64": 110, "128": 100},
    }
    _patch_mlx_lm(
        monkeypatch,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (128, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_down(model)

    assert prepared.group_size == 128
    assert prepared.group_tuning == tuning
    assert prepared.weight_for(down_proj).group_size == 128


def test_prepare_mlx_lm_compressed_gate_up_preserves_layer_pairs(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    gate = nn.Linear(128, 64, bias=False)
    up = nn.Linear(128, 64, bias=False)
    gate.weight = gate.weight.astype(mx.bfloat16)
    up.weight = up.weight.astype(mx.bfloat16)
    mlp = SimpleNamespace(gate_proj=gate, up_proj=up)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100, "128": 110},
    }
    _patch_mlx_lm(
        monkeypatch,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_gate_up(Model())

    assert prepared.layer_count == 1
    assert len(prepared.source_layers) == 1
    assert prepared.projection_count == 2
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert prepared.weight_for(gate).shape == gate.weight.shape
    assert prepared.weight_for(up).shape == up.weight.shape
    assert prepared.repack_bytes < gate.weight.nbytes + up.weight.nbytes


def test_mlx_lm_compressed_gate_up_patch_is_reversible_and_decode_only(monkeypatch):
    pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Linear:
        def __call__(self, values):
            calls.append(("native", self, values))
            return "native"

    gate = Linear()
    up = Linear()

    gate_weight = SimpleNamespace(values="gate", scales="gate-scale", biases="gate-bias")
    up_weight = SimpleNamespace(values="up", scales="up-scale", biases="up-bias")

    class MLP:
        gate_proj = gate
        up_proj = up

        def __call__(self, values):
            calls.append(("native-mlp", values))
            return "native-mlp"

        def down_proj(self, hidden):
            calls.append(("down", hidden))
            return "projected", hidden

    mlp = MLP()

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(mlp): (mlp, gate, gate_weight, up, up_weight)},
        200,
        calibrated=True,
        implementation="fused",
    )
    _patch_mlx_lm(monkeypatch, "_supports_compressed_gate_up_fusion", lambda _module: True)

    def fused(values, *weights, **options):
        calls.append(("fused", values, weights, options))
        return "fused-hidden"

    _patch_mlx_lm(
        monkeypatch,
        "mlx_affine_swiglu_executor",
        lambda *args, **options: lambda values: fused(values, *args[1:], **options),
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_gate_up=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64), dtype="bf16")
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64), dtype="bf16")

    assert mlp(decode) == ("projected", "fused-hidden")
    assert mlp(prefill) == "native-mlp"
    assert type(gate) is Linear
    assert type(up) is Linear

    patch.restore()

    assert type(mlp) is MLP
    assert mlp(decode) == "native-mlp"
    fused_call = next(call for call in calls if call[0] == "fused")
    assert fused_call[2] == (
        "gate",
        "gate-scale",
        "gate-bias",
        "up",
        "up-scale",
        "up-bias",
    )
    assert fused_call[3] == {"group_size": 64, "bits": 8}


def test_mlx_lm_compressed_gate_up_falls_back_to_projection_patches(monkeypatch):
    pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    class Linear:
        def __call__(self, _values):
            return "native"

    gate = Linear()
    up = Linear()

    def gate_weight(values):
        return "compressed-gate", values

    def up_weight(values):
        return "compressed-up", values

    mlp = SimpleNamespace(gate_proj=gate, up_proj=up)

    class Model:
        layers = (SimpleNamespace(mlp=mlp),)

        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(mlp): (mlp, gate, gate_weight, up, up_weight)},
        200,
        calibrated=True,
    )
    _patch_mlx_lm(monkeypatch, "_supports_compressed_gate_up_fusion", lambda _module: False)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_gate_up=prepared,
        plan=mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True),
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))

    assert gate(decode) == ("compressed-gate", decode)
    assert up(decode) == ("compressed-up", decode)
    patch.restore()
    assert type(gate) is Linear
    assert type(up) is Linear


def test_mlx_lm_compressed_gate_up_selects_faster_fused_model_path(monkeypatch):
    from metile.integrations import mlx_lm

    class Model:
        def __call__(self):
            pass

    module = object()
    weight = SimpleNamespace(shape=(64, 64), group_size=64)
    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(module): (module, object(), weight, object(), weight)},
        200,
    )
    reference = SimpleNamespace(full_reference=object())
    fidelity = {
        "next_token": 7,
        "actual_next_token": 7,
        "kl_divergence": 0.0,
        "mean_logit_error": 0.0,
        "max_logit_error": 0.0,
    }
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    _patch_mlx_lm(monkeypatch, "_supports_compressed_gate_up_fusion", lambda _module: True)
    _patch_mlx_lm(monkeypatch, "_compressed_gate_up_implementation_key", lambda *_args: "key")
    _patch_mlx_lm(
        monkeypatch,
        "_prepare_compressed_calibration_reference",
        lambda *_args: reference,
    )
    _patch_mlx_lm(
        monkeypatch,
        "_run_compressed_calibration_candidate",
        lambda *_args, **_options: object(),
    )
    _patch_mlx_lm(monkeypatch, "_logit_fidelity", lambda *_args: fidelity)
    _patch_mlx_lm(
        monkeypatch,
        "_prepare_mlx_lm_prompt",
        lambda *_args: (object(), 0.1, (object(), object())),
    )

    def time_plan(*_args, compressed_gate_up=None, **_options):
        decode = 0.8 if compressed_gate_up.implementation == "fused" else 1.0
        return (0.1, decode, 0.1 + decode), 7

    _patch_mlx_lm(monkeypatch, "_time_mlx_lm_plan", time_plan)

    mlx_lm._select_compressed_gate_up_implementation(
        prepared.model,
        SimpleNamespace(),
        prepared,
        2,
        3,
    )

    assert prepared.implementation == "fused"
    assert prepared.implementation_tuning["reason"] == "timing"
    assert prepared.implementation_tuning["median_nanoseconds"] == {
        "projected": 1_000_000_000,
        "fused": 800_000_000,
    }


def test_mlx_lm_compressed_gate_up_rejects_inexact_fusion(monkeypatch):
    from metile.integrations import mlx_lm

    class Model:
        def __call__(self):
            pass

    module = object()
    weight = SimpleNamespace(shape=(64, 64), group_size=64)
    prepared = mlx_lm.MLXCompressedGateUp(
        Model(),
        {id(module): (module, object(), weight, object(), weight)},
        200,
    )
    reference = SimpleNamespace(full_reference=object())
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    _patch_mlx_lm(monkeypatch, "_supports_compressed_gate_up_fusion", lambda _module: True)
    _patch_mlx_lm(monkeypatch, "_compressed_gate_up_implementation_key", lambda *_args: "key")
    _patch_mlx_lm(
        monkeypatch,
        "_prepare_compressed_calibration_reference",
        lambda *_args: reference,
    )
    _patch_mlx_lm(
        monkeypatch,
        "_run_compressed_calibration_candidate",
        lambda *_args, **_options: object(),
    )
    _patch_mlx_lm(
        monkeypatch,
        "_logit_fidelity",
        lambda *_args: {
            "next_token": 7,
            "actual_next_token": 8,
            "kl_divergence": 0.0,
            "mean_logit_error": 0.0,
            "max_logit_error": 0.0,
        },
    )
    _patch_mlx_lm(
        monkeypatch,
        "_time_mlx_lm_plan",
        lambda *_args, **_options: pytest.fail("inexact fusion must not be timed"),
    )

    mlx_lm._select_compressed_gate_up_implementation(
        prepared.model,
        SimpleNamespace(),
        prepared,
        2,
        3,
    )

    assert prepared.implementation == "projected"
    assert prepared.implementation_tuning["reason"] == "fidelity"


def test_prepare_mlx_lm_compressed_attention_preserves_layer_groups(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    projections = tuple(nn.Linear(128, 64, bias=index == 0) for index in range(4))
    for projection in projections:
        projection.weight = projection.weight.astype(mx.bfloat16)
        if "bias" in projection:
            projection.bias = projection.bias.astype(mx.bfloat16)
    attention = SimpleNamespace(
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        o_proj=projections[3],
    )

    class Model:
        layers = (SimpleNamespace(self_attn=attention),)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100, "128": 110},
    }
    _patch_mlx_lm(
        monkeypatch,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    prepared = mlx_lm.prepare_mlx_lm_compressed_attention(Model())

    assert prepared.layer_count == 1
    assert prepared.projection_count == 4
    assert len(prepared.source_layers) == 1
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert all(
        prepared.weight_for(projection).shape == projection.weight.shape
        for projection in projections
    )
    assert prepared.repack_bytes < sum(projection.weight.nbytes for projection in projections)


def test_mlx_lm_compressed_attention_patch_is_reversible_decode_only_and_biased():
    pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Linear(dict):
        def __call__(self, values):
            calls.append(("native", self, values))
            return "native"

    projections = tuple(Linear() for _ in range(4))
    projections[0]["bias"] = 2
    attention = SimpleNamespace(
        q_proj=projections[0],
        k_proj=projections[1],
        v_proj=projections[2],
        o_proj=projections[3],
    )
    weights = tuple(lambda _values, result=result: result for result in (5, 6, 7, 8))

    class Model:
        layers = (SimpleNamespace(self_attn=attention),)

        def __call__(self):
            pass

    model = Model()
    prepared = mlx_lm.MLXCompressedAttention(
        model,
        {id(attention): (attention, tuple(zip(projections, weights)))},
        400,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_attention=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_attention=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64))

    assert projections[0](decode) == 7
    assert projections[1](decode) == 6
    assert projections[0](prefill) == "native"

    patch.restore()

    assert all(type(projection) is Linear for projection in projections)
    assert projections[0](decode) == "native"


def test_prepare_mlx_lm_compressed_vocab_supports_tied_embedding(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn

    from metile.integrations import mlx_lm

    embedding = nn.Embedding(128, 64)
    embedding.weight = embedding.weight.astype(mx.bfloat16)

    class Model:
        args = SimpleNamespace(tie_word_embeddings=True)
        model = SimpleNamespace(embed_tokens=embedding)

        def __call__(self):
            pass

    tuning = {
        "cached": False,
        "group_size": 64,
        "median_nanoseconds": {"32": 120, "64": 100},
    }
    _patch_mlx_lm(
        monkeypatch,
        "tune_mlx_affine8_group_size",
        lambda weights, **_options: (64, tuning),
    )

    model = Model()
    prepared = mlx_lm.prepare_mlx_lm_compressed_vocab(model)

    assert prepared.model is model
    assert prepared.module is embedding
    assert prepared.tied
    assert prepared.group_size == 64
    assert prepared.group_tuning == tuning
    assert prepared.projection_count == 1
    assert prepared.weight.shape == embedding.weight.shape
    assert prepared.repack_bytes < embedding.weight.nbytes


def test_mlx_lm_compressed_vocab_tied_patch_is_reversible_and_decode_only():
    pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Embedding:
        def as_linear(self, values):
            calls.append(("native", values))
            return "native"

    module = Embedding()

    def weight(values):
        return "compressed", values

    class Model:
        def __call__(self):
            pass

    model = Model()
    prepared = mlx_lm.MLXCompressedVocab(
        model,
        module,
        weight,
        True,
        100,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    patch = mlx_lm.apply_metile_to_mlx_lm(
        model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_vocab=prepared,
        plan=plan,
    )
    decode = SimpleNamespace(size=64, shape=(1, 1, 64))
    prefill = SimpleNamespace(size=128, shape=(1, 2, 64))

    assert module.as_linear(decode) == ("compressed", decode)
    assert module.as_linear(prefill) == "native"

    patch.restore()

    assert type(module) is Embedding
    assert module.as_linear(decode) == "native"


def test_mlx_lm_compressed_vocab_untied_patch_uses_linear_call():
    pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    class Linear:
        def __call__(self, _values):
            return "native"

    module = Linear()

    class Model:
        def __call__(self):
            pass

    prepared = mlx_lm.MLXCompressedVocab(
        Model(),
        module,
        lambda values: ("compressed", values),
        False,
        100,
        calibrated=True,
    )
    plan = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    decode = SimpleNamespace(size=64, shape=(1, 64))

    with mlx_lm.apply_metile_to_mlx_lm(
        prepared.model,
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_vocab=prepared,
        plan=plan,
    ):
        assert module(decode) == ("compressed", decode)

    assert module(decode) == "native"


def test_mlx_lm_compressed_down_fidelity_policy_distinguishes_approximation():
    from metile.integrations.mlx_lm import MLXCompressedDown

    fidelity = {
        "next_token": 7,
        "actual_next_token": 7,
        "kl_divergence": 0.02,
        "mean_logit_error": 0.2,
        "max_logit_error": 1.0,
    }
    strict = MLXCompressedDown(object(), {}, "affine8", 0)
    approximate = MLXCompressedDown(object(), {}, "mxfp8", 0, True)

    assert not strict.fidelity_compatible(fidelity)
    assert approximate.fidelity_compatible(fidelity)
    assert not approximate.fidelity_compatible({**fidelity, "actual_next_token": 8})


def _group_tuning_region():
    return SimpleNamespace(
        group_tuning={
            "group_size": 32,
            "median_nanoseconds": {"64": 1000, "128": 800},
            "native_median_nanoseconds": 1500,
        },
        source_layers=list(range(8)),
        layer_count=4,
        group_size=32,
        calibration_fidelity={"kl": 0.0},
        selection="all",
    )


def _group_tuning_collaborators(calls):
    def repack(region, group):
        calls.append(("repack", group))
        region.group_size = group
        region.layer_count = group // 16

    def calibrate(_model, _tokens, region, _steps):
        calls.append(("calibrate", region.group_size))

    return {
        "group_key": lambda *_: "K",
        "lock": threading.Lock(),
        "repack": repack,
        "calibrate": calibrate,
    }


def test_autotune_compressed_group_tunes_then_reuses_the_cached_group(tmp_path):
    """The gate-up and attention group tuners share one implementation.

    Both had 91 identical lines before they were merged, and nothing in the suite reached
    either of them, so this covers the shared body directly.
    """
    from metile.integrations.mlx_lm.tuning import _autotune_compressed_group

    path = tmp_path / "group.json"
    calls = []
    parts = _group_tuning_collaborators(calls)

    cold = _group_tuning_region()
    _autotune_compressed_group(object(), object(), cold, 3, cache_path=path, **parts)

    assert cold.group_tuning["model_calibrated"] is True
    assert cold.group_tuning["cached"] is False
    assert cold.group_tuning["group_size"] in (64, 128)
    assert sorted(cold.group_tuning["model_candidates"]) == ["128", "64"]
    assert path.exists(), "a cold run must persist the chosen group size"
    cold_calls = len(calls)

    calls.clear()
    warm = _group_tuning_region()
    _autotune_compressed_group(object(), object(), warm, 3, cache_path=path, **parts)

    assert warm.group_tuning["cached"] is True
    assert warm.group_tuning["group_size"] == cold.group_tuning["group_size"]
    assert len(calls) < cold_calls, "a cached run must not re-time every candidate"


def test_autotune_compressed_group_skips_the_cache_when_disabled(tmp_path, monkeypatch):
    from metile.integrations.mlx_lm.tuning import _autotune_compressed_group

    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    path = tmp_path / "group.json"
    parts = _group_tuning_collaborators([])

    region = _group_tuning_region()
    _autotune_compressed_group(object(), object(), region, 3, cache_path=path, **parts)

    assert region.group_tuning["cached"] is False
    assert not path.exists(), "the disk cache must not be written when disabled"
