"""MLX backend tests: planning."""

from types import SimpleNamespace

import pytest

from tests.module_patching import _patch_mlx_lm


def test_mlx_lm_model_group_selection_values_fidelity_coverage():
    from metile.integrations.mlx_lm import _select_model_affine8_group

    selected, estimates = _select_model_affine8_group(
        28,
        {32: 26, 64: 14, 128: 1},
        {32: 435_000, 64: 419_000, 128: 405_000},
        620_000,
    )

    assert selected == 32
    assert estimates[32] < estimates[64] < estimates[128]


def test_mlx_affine_swiglu_fidelity_uses_scale_aware_error():
    mx = pytest.importorskip("mlx.core")
    from metile.backends import mlx_quantized

    reference = mx.array([0.0, 1000.0, -2000.0], dtype=mx.float16)
    reduction_drift = mx.array([0.2, 1001.0, -1998.0], dtype=mx.float16)
    material_error = mx.array([0.0, 1015.0, -1960.0], dtype=mx.float16)

    assert mlx_quantized._affine_swiglu_compatible(reduction_drift, reference)
    assert not mlx_quantized._affine_swiglu_compatible(material_error, reference)


def test_mlx_lm_plan_key_tracks_affine_backend(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    class Model:
        layers = ()

        def __call__(self, values):
            return values

    tokens = mx.zeros((1, 8), dtype=mx.int32)
    requested = mlx_lm.MLXLMPlan(False, False, False, False, True)
    _patch_mlx_lm(monkeypatch, "mlx_affine_backend_signature", lambda: "backend-a")
    first = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, None, 8, 5)
    _patch_mlx_lm(monkeypatch, "mlx_affine_backend_signature", lambda: "backend-b")
    second = mlx_lm._mlx_lm_plan_key(Model(), tokens, requested, None, None, 8, 5)

    assert first != second


def test_mlx_lm_plan_candidates_cover_requested_feature_lattice():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(attention=True, rms_norm=True, graph_fusion=False, quantized_mlp=False)
    )

    assert len(candidates) == 4
    assert MLXLMPlan(False, False, False, False) in candidates
    assert MLXLMPlan(True, True, False, False) in candidates
    assert all(not plan.graph_fusion and not plan.quantized_mlp for plan in candidates)


def test_mlx_lm_plan_timing_reuses_prepared_prompt(monkeypatch):
    from contextlib import nullcontext

    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            calls.append((tuple(tokens.shape), tuple(tokens.flatten().tolist()), cache))
            return mx.array([[[0.0, 1.0]]])

    _patch_mlx_lm(monkeypatch, "apply_metile_to_mlx_lm", lambda **_options: nullcontext())
    tokens = mx.array([[1, 2, 3]])
    prepared_cache = SimpleNamespace(marker="prepared")
    decode_tokens = (mx.array([[7]]), mx.array([[8]]))

    measurement, next_token = mlx_lm._time_mlx_lm_plan(
        Model(),
        tokens,
        mlx_lm.MLXLMPlan(False, False, False, False),
        None,
        None,
        2,
        prepared_prompt=(prepared_cache, 0.25, decode_tokens),
        decode_tokens=decode_tokens,
    )

    assert next_token == 1
    assert len(calls) == 2
    assert [values for _, values, _ in calls] == [(7,), (8,)]
    assert all(shape == (1, 1) for shape, _, _ in calls)
    assert all(cache is not prepared_cache and cache.marker == "prepared" for _, _, cache in calls)
    assert measurement[0] == 0.25
    assert measurement[2] >= 0.25


def test_mlx_lm_measurement_shares_prompt_only_with_decode_only_plans(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    structural = mlx_lm.MLXLMPlan(True, False, False, False)
    decode_tokens = (object(), object())
    prepared = (object(), 0.25, decode_tokens)
    seen = []

    _patch_mlx_lm(monkeypatch, "_plan_preserves_logits", lambda *_args, **_kwargs: True)
    _patch_mlx_lm(monkeypatch, "_prepare_mlx_lm_prompt", lambda *_args: prepared)

    def time_plan(*arguments, prepared_prompt=None, **_options):
        seen.append((arguments[2], prepared_prompt, _options["decode_tokens"]))
        return (0.25, 0.01, 0.27), 1

    _patch_mlx_lm(monkeypatch, "_time_mlx_lm_plan", time_plan)

    class Model:
        def __call__(self, _tokens):
            return mx.array([[[0.0, 1.0]]])

    mlx_lm._measure_mlx_lm_plans(
        Model(),
        mx.array([[1, 2, 3]]),
        (native, compressed, structural),
        None,
        None,
        2,
        1,
    )

    assert seen == [
        (native, prepared, decode_tokens),
        (compressed, prepared, decode_tokens),
        (structural, None, decode_tokens),
    ]


def test_mlx_lm_plan_candidates_split_dense_swiglu_and_residual():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            attention=False,
            rms_norm=False,
            graph_fusion=False,
            quantized_mlp=False,
            dense_mlp=True,
            dense_residual=True,
        )
    )

    assert len(candidates) == 4
    assert (
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=True,
            dense_residual=False,
        )
        in candidates
    )
    assert (
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=False,
            dense_residual=True,
        )
        in candidates
    )


def test_mlx_lm_plan_candidates_exclude_competing_down_rewrites():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            attention=False,
            rms_norm=False,
            graph_fusion=False,
            quantized_mlp=False,
            dense_residual=True,
            compressed_down=True,
        )
    )

    assert len(candidates) == 3
    assert any(plan.dense_residual for plan in candidates)
    assert any(plan.compressed_down for plan in candidates)
    assert all(not (plan.dense_residual and plan.compressed_down) for plan in candidates)


def test_mlx_lm_plan_candidates_bound_structural_cross_product():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    requested = MLXLMPlan(
        attention=True,
        rms_norm=True,
        graph_fusion=True,
        quantized_mlp=False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
        compressed_attention=True,
    )
    candidates = _mlx_lm_plan_candidates(requested)

    assert len(candidates) == 30
    assert requested in candidates
    assert sum(plan.feature_count == 1 for plan in candidates) == 7


def test_mlx_lm_plan_candidates_exclude_bypassed_gate_up_projection():
    from metile.integrations.mlx_lm import MLXLMPlan, _mlx_lm_plan_candidates

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(
            False,
            False,
            False,
            False,
            dense_mlp=True,
            compressed_gate_up=True,
        )
    )

    assert len(candidates) == 3
    assert all(not (plan.dense_mlp and plan.compressed_gate_up) for plan in candidates)


def test_mlx_lm_warmup_plans_scale_with_primitives_not_plan_lattice():
    from metile.integrations.mlx_lm import (
        MLXLMPlan,
        _mlx_lm_plan_candidates,
        _mlx_lm_warmup_plans,
    )

    candidates = _mlx_lm_plan_candidates(
        MLXLMPlan(**{name: True for name in MLXLMPlan().as_dict()})
    )
    warmups = _mlx_lm_warmup_plans(candidates)
    enabled = {
        frozenset(name for name, active in plan.as_dict().items() if active) for plan in warmups
    }

    assert len(warmups) <= 10
    assert len(warmups) < len(candidates) // 4
    assert frozenset() in enabled
    assert frozenset(("attention",)) in enabled
    assert frozenset(("graph_fusion", "quantized_mlp")) in enabled
    assert frozenset(("dense_mlp", "dense_residual")) in enabled
    assert all("compressed_vocab" not in features for features in enabled)
    assert all("compressed_attention" not in features for features in enabled)


def test_mlx_lm_effective_plan_splits_dense_swiglu_and_residual(monkeypatch):
    from metile.integrations import mlx_lm

    requested = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=True,
        dense_residual=True,
    )
    prepared = SimpleNamespace(implementation="fused")
    _patch_mlx_lm(
        monkeypatch,
        "mlx_dense_swiglu_dispatches",
        lambda: ({"algorithm": "metile"},),
    )
    _patch_mlx_lm(
        monkeypatch,
        "mlx_dense_residual_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(requested, dense_mlp=prepared) == mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=True,
        dense_residual=False,
    )

    _patch_mlx_lm(
        monkeypatch,
        "mlx_dense_swiglu_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )
    _patch_mlx_lm(
        monkeypatch,
        "mlx_dense_residual_dispatches",
        lambda: ({"algorithm": "metile"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(requested, dense_mlp=prepared) == mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        dense_mlp=False,
        dense_residual=True,
    )


def test_mlx_lm_effective_plan_prunes_native_wrappers(monkeypatch):
    from metile.integrations import mlx_lm

    _patch_mlx_lm(monkeypatch, "mlx_attention_dispatches", lambda: ({"algorithm": "metile"},))
    _patch_mlx_lm(monkeypatch, "mlx_rms_norm_dispatches", lambda: ({"algorithm": "mlx"},))
    _patch_mlx_lm(monkeypatch, "mlx_add_rms_norm_dispatches", lambda: ())
    _patch_mlx_lm(monkeypatch, "mlx_affine_swiglu_dispatches", lambda: ({"algorithm": "mlx"},))
    _patch_mlx_lm(
        monkeypatch,
        "mlx_affine_residual_qmv_dispatches",
        lambda: ({"algorithm": "mlx"},),
    )

    assert mlx_lm._effective_mlx_lm_plan(mlx_lm.MLXLMPlan()) == mlx_lm.MLXLMPlan(
        attention=True,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
    )


def test_mlx_lm_effective_plan_keeps_compiled_quantized_mlp(monkeypatch):
    from metile.integrations import mlx_lm

    _patch_mlx_lm(monkeypatch, "mlx_attention_dispatches", lambda: ())
    _patch_mlx_lm(monkeypatch, "mlx_rms_norm_dispatches", lambda: ())
    _patch_mlx_lm(monkeypatch, "mlx_add_rms_norm_dispatches", lambda: ())
    _patch_mlx_lm(
        monkeypatch,
        "mlx_affine_swiglu_dispatches",
        lambda: ({"algorithm": "mlx_compiled"},),
    )
    _patch_mlx_lm(monkeypatch, "mlx_affine_residual_qmv_dispatches", lambda: ())

    assert mlx_lm._effective_mlx_lm_plan(mlx_lm.MLXLMPlan()) == mlx_lm.MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=True,
    )


def test_mlx_lm_plan_requires_decode_ttft_and_total_headroom():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    safe = MLXLMPlan(attention=True, rms_norm=False, graph_fusion=False, quantized_mlp=False)
    ttft_regression = MLXLMPlan(
        attention=False,
        rms_norm=True,
        graph_fusion=False,
        quantized_mlp=False,
    )
    selected = _choose_mlx_lm_plan(
        {
            native: [(0.100, 0.0100, 0.180)] * 5,
            safe: [(0.099, 0.0095, 0.174)] * 5,
            ttft_regression: [(0.102, 0.0090, 0.172)] * 5,
        }
    )

    assert selected == safe


def test_mlx_lm_plan_falls_back_when_total_win_is_too_small():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    close = MLXLMPlan(attention=True, rms_norm=False, graph_fusion=False, quantized_mlp=False)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                close: [(0.100, 0.0099, 0.179)] * 5,
            }
        )
        == native
    )


def test_mlx_lm_plan_accepts_sustained_decode_win_without_latency_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    dense = MLXLMPlan(False, False, False, False, False, True)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                dense: [(0.100, 0.0098, 0.1795)] * 5,
            }
        )
        == dense
    )


def test_mlx_lm_plan_ranking_retains_simpler_validated_fallback():
    from metile.integrations.mlx_lm import MLXLMPlan, _rank_mlx_lm_plans

    native = MLXLMPlan(False, False, False, False)
    combined = MLXLMPlan(
        attention=True,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        compressed_down=True,
    )
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )

    assert _rank_mlx_lm_plans(
        {
            native: [(0.100, 0.0100, 0.180)] * 5,
            combined: [(0.098, 0.0085, 0.155)] * 5,
            compressed: [(0.099, 0.0090, 0.165)] * 5,
        }
    ) == (combined, compressed)


def test_mlx_lm_plan_accepts_strong_decode_with_stable_median_ttft():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    ttft_ratios = (0.980, 1.008, 1.011, 1.012, 1.014, 0.978, 0.994)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * len(ttft_ratios),
                compressed: [(0.100 * ttft_ratio, 0.0090, 0.167) for ttft_ratio in ttft_ratios],
            }
        )
        == compressed
    )


def test_mlx_lm_plan_accepts_decode_only_compression_with_noisy_ttft():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    compressed = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 7,
                compressed: [(0.102, 0.0090, 0.167)] * 7,
            }
        )
        == compressed
    )


def test_mlx_lm_plan_rejects_primitive_rewrite_above_ttft_regression_bound():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    rms_norm = MLXLMPlan(False, True, False, False)

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 7,
                rms_norm: [(0.102, 0.0090, 0.167)] * 7,
            }
        )
        == native
    )


def test_mlx_lm_plan_validation_uses_longer_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    dense = mlx_lm.MLXLMPlan(False, False, False, False, False, True)
    calls = []

    def measure(
        _model,
        _tokens,
        candidates,
        _affine_prefill,
        _dense_mlp,
        decode_steps,
        rounds,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
        **_options,
    ):
        calls.append((candidates, decode_steps, rounds))
        return {native: [(1.0, 1.0, 1.0)], dense: [(1.0, 0.98, 0.99)]}

    _patch_mlx_lm(monkeypatch, "_measure_mlx_lm_plans", measure)
    _patch_mlx_lm(monkeypatch, "_choose_mlx_lm_plan", lambda measured: dense)

    assert mlx_lm._validate_mlx_lm_plan(object(), object(), dense, None, None, 8, 5) == dense
    assert calls == [((native, dense), 32, 7)]


def test_mlx_lm_plan_validation_retries_one_noisy_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    results = iter((native, compressed))
    calls = []

    def validate(*arguments):
        calls.append(arguments[2])
        return next(results)

    _patch_mlx_lm(monkeypatch, "_validate_mlx_lm_plan", validate)

    assert (
        mlx_lm._validate_mlx_lm_plan_repeated(
            object(),
            object(),
            compressed,
            None,
            None,
            8,
            5,
        )
        == compressed
    )
    assert calls == [compressed, compressed]


def test_mlx_lm_joint_validation_keeps_decode_and_composition_leaders():
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_gate_up=True,
        compressed_vocab=True,
        compressed_attention=True,
    )
    measured = {
        native: [(0.100, 0.010, 0.180)] * 5,
        vocab: [(0.090, 0.009, 0.165)] * 5,
        composite: [(0.106, 0.006, 0.145)] * 5,
    }

    finalists = mlx_lm._mlx_lm_validation_finalists(measured)

    assert finalists[0] == native
    assert vocab in finalists
    assert composite in finalists


def test_mlx_lm_joint_validation_preserves_compression_ladder():
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    down = mlx_lm.MLXLMPlan(False, False, False, False, compressed_down=True)
    gate = mlx_lm.MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    down_gate = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
    )
    down_gate_vocab = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
    )
    measured = {
        native: [(0.100, 0.0100, 0.180)] * 5,
        down: [(0.110, 0.0060, 0.160)] * 5,
        gate: [(0.110, 0.0065, 0.162)] * 5,
        vocab: [(0.090, 0.0080, 0.150)] * 5,
        down_gate: [(0.110, 0.0075, 0.165)] * 5,
        down_gate_vocab: [(0.110, 0.0070, 0.160)] * 5,
    }

    finalists = mlx_lm._mlx_lm_validation_finalists(measured)

    assert down_gate in finalists
    assert down_gate_vocab in finalists


def test_mlx_lm_joint_validation_compares_finalists_on_one_holdout(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_gate_up=True,
        compressed_attention=True,
    )
    finalists = (native, vocab, composite)
    calls = []

    def measure(
        _model,
        _tokens,
        candidates,
        _affine_prefill,
        _dense_mlp,
        decode_steps,
        rounds,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
        **_options,
    ):
        calls.append((candidates, decode_steps, rounds))
        return {
            native: [(0.100, 0.010, 0.180)] * 7,
            vocab: [(0.099, 0.009, 0.165)] * 7,
            composite: [(0.100, 0.006, 0.140)] * 7,
        }

    _patch_mlx_lm(monkeypatch, "_measure_mlx_lm_plans", measure)

    selected = mlx_lm._validate_mlx_lm_finalists_repeated(
        object(),
        object(),
        finalists,
        None,
        None,
        8,
        5,
    )

    assert selected == composite
    assert calls == [(finalists, 32, 7)]


def test_mlx_lm_joint_validation_halves_large_holdout_before_full_trials(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    down = mlx_lm.MLXLMPlan(False, False, False, False, compressed_down=True)
    vocab = mlx_lm.MLXLMPlan(False, False, False, False, compressed_vocab=True)
    composite = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        compressed_down=True,
        compressed_vocab=True,
    )
    finalists = (native, down, vocab, composite)
    calls = []

    def measure(*arguments, **options):
        active = arguments[2]
        rounds = arguments[6]
        calls.append((active, rounds, options.get("validate_fidelity", True)))
        decode = {native: 1.0, down: 0.82, vocab: 0.90, composite: 0.72}
        return {plan: [(1.0, decode[plan], 0.5 + decode[plan])] * rounds for plan in active}

    _patch_mlx_lm(monkeypatch, "_measure_mlx_lm_plans", measure)

    selected = mlx_lm._validate_mlx_lm_finalists_repeated(
        object(),
        object(),
        finalists,
        None,
        None,
        8,
        5,
    )

    assert selected == composite
    assert calls[0] == (finalists, 3, True)
    assert len(calls[1][0]) == 3
    assert calls[1][1:] == (4, False)


def test_mlx_lm_plan_accepts_strong_ttft_win_without_total_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    prefill = MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        affine_prefill=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.180)] * 5,
                prefill: [(0.090, 0.0100, 0.1795)] * 5,
            }
        )
        == prefill
    )


def test_mlx_lm_plan_accepts_two_percent_ttft_win_without_total_regression():
    from metile.integrations.mlx_lm import MLXLMPlan, _choose_mlx_lm_plan

    native = MLXLMPlan(False, False, False, False)
    prefill = MLXLMPlan(
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        dense_mlp=True,
    )

    assert (
        _choose_mlx_lm_plan(
            {
                native: [(0.100, 0.0100, 0.1800)] * 5,
                prefill: [(0.0975, 0.0100, 0.1795)] * 5,
            }
        )
        == prefill
    )


def test_mlx_lm_provisional_finalists_preserve_fastest_ttft_plan():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False, False)
    total = MLXLMPlan(True, False, False, False, False)
    ttft = MLXLMPlan(False, False, False, False, True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        total: [(0.099, 0.010, 0.160)] * 3,
        ttft: [(0.080, 0.010, 0.190)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, (native, total, ttft))

    assert finalists == (native, total, ttft)


def test_mlx_lm_provisional_finalists_preserve_fastest_decode_plan():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False, False)
    total = MLXLMPlan(True, False, False, False, False)
    decode = MLXLMPlan(False, False, False, False, compressed_down=True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        total: [(0.099, 0.0098, 0.160)] * 3,
        decode: [(0.130, 0.0070, 0.190)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, (native, total, decode))

    assert finalists == (native, total, decode)


def test_mlx_lm_provisional_finalists_preserve_promising_single_feature_fallbacks():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False)
    attention = MLXLMPlan(True, False, False, False)
    compressed = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    combined = MLXLMPlan(True, False, False, False, compressed_gate_up=True)
    provisional = {
        native: [(0.100, 0.010, 0.180)] * 3,
        attention: [(0.130, 0.009, 0.200)] * 3,
        compressed: [(0.095, 0.011, 0.200)] * 3,
        combined: [(0.090, 0.008, 0.150)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(
        provisional,
        (native, attention, compressed, combined),
    )

    assert finalists == (native, attention, compressed, combined)


def test_mlx_lm_provisional_finalists_preserve_compression_ladder():
    from metile.integrations.mlx_lm import MLXLMPlan, _provisional_mlx_lm_finalists

    native = MLXLMPlan(False, False, False, False)
    down = MLXLMPlan(False, False, False, False, compressed_down=True)
    gate = MLXLMPlan(False, False, False, False, compressed_gate_up=True)
    vocab = MLXLMPlan(False, False, False, False, compressed_vocab=True)
    down_gate = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
    )
    down_gate_vocab = MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
        compressed_gate_up=True,
        compressed_vocab=True,
    )
    candidates = (native, down, gate, vocab, down_gate, down_gate_vocab)
    provisional = {
        native: [(0.100, 0.0100, 0.180)] * 3,
        down: [(0.110, 0.0060, 0.160)] * 3,
        gate: [(0.110, 0.0065, 0.162)] * 3,
        vocab: [(0.090, 0.0080, 0.150)] * 3,
        down_gate: [(0.110, 0.0075, 0.165)] * 3,
        down_gate_vocab: [(0.110, 0.0070, 0.160)] * 3,
    }

    finalists = _provisional_mlx_lm_finalists(provisional, candidates)

    assert down_gate in finalists
    assert down_gate_vocab in finalists


def test_mlx_lm_bfloat16_fidelity_uses_precision_aware_limits():
    from metile.integrations import mlx_lm

    fidelity = {
        "reference_dtype": "mlx.core.bfloat16",
        "next_token": 42,
        "actual_next_token": 42,
        "kl_divergence": 7e-4,
        "mean_logit_error": 0.03,
        "max_logit_error": 0.4,
    }

    assert mlx_lm._fidelity_compatible(fidelity)
    assert not mlx_lm._fidelity_compatible({**fidelity, "reference_dtype": "mlx.core.float16"})
