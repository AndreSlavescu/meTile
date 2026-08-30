"""MLX backend tests: core."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends import mlx as mlx_backend
from tests.module_patching import _patch_mlx_lm


def test_mlx_lm_benchmark_helpers_import_without_optional_mlx_packages():
    script = """
import builtins

original_import = builtins.__import__

def import_without_mlx(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm" or name.startswith("mlx_lm."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_mlx
from benchmarks import mlx_lm_backend
assert callable(mlx_lm_backend._confirm_plan)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_mlx_kernel_body_rebinds_metal_thread_attributes():
    source = """
#include <metal_stdlib>
using namespace metal;
[[kernel]] void example(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint tgp_id_x [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]]) {
    Y[tgp_id_x + lid] = X[slid];
}
"""

    body = mlx_backend._mlx_kernel_body(source)

    assert "threadgroup_position_in_grid.x" in body
    assert "thread_index_in_threadgroup" in body
    assert "thread_index_in_simdgroup" in body
    assert "tgp_id_x" not in body
    assert " lid" not in body
    assert "slid" not in body


def test_mlx_kernel_body_accepts_kernel_attribute_lists():
    source = """
[[kernel, max_total_threads_per_threadgroup(32)]] void example(
    device half* X [[buffer(0)]],
    uint3 tgp_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    X[tgp_id.x] = half(sgid);
}
"""

    body = mlx_backend._mlx_kernel_body(source)

    assert "threadgroup_position_in_grid.x" in body
    assert "simdgroup_index_in_threadgroup" in body


def test_mlx_bfloat16_source_specialization_uses_native_metal_types_and_stores():
    source = """
device const half4* X;
device half* Out;
Out[thread_index_in_threadgroup] = float(X[0].x);
*((device half4*)(&Out[4])) = half4(values[0], values[1], values[2], values[3]);
"""

    specialized = mlx_backend._specialize_mlx_source(source, "mlx.core.bfloat16")

    assert "device const bfloat4* X" in specialized
    assert "device bfloat* Out" in specialized
    assert "Out[thread_index_in_threadgroup] = bfloat(float(X[0].x));" in specialized
    assert (
        "bfloat4(bfloat(values[0]), bfloat(values[1]), bfloat(values[2]), bfloat(values[3]))"
        in specialized
    )
    assert "half" not in specialized


def test_mlx_source_specialization_preserves_non_bfloat16_source():
    source = "device half* Out;"

    assert mlx_backend._specialize_mlx_source(source, "mlx.core.float16") == source


def test_mlx_lm_model_group_selection_balances_speed_with_broad_coverage():
    from metile.integrations.mlx_lm import _select_model_affine8_group

    selected, estimates = _select_model_affine8_group(
        36,
        {32: 36, 64: 35, 128: 33},
        {32: 421_000, 64: 398_000, 128: 391_000},
        620_000,
    )

    assert selected == 64
    assert estimates[64] < estimates[32]


def test_confirm_pairwise_times_each_candidate_alone_against_the_baseline():
    """Candidates must be timed two at a time, and ranked on the ratio to the baseline.

    How long a candidate measures depends on how many others share the round-robin, so
    ranking from one crowded rotation can prefer a kernel that loses head to head. Ranking
    on the ratio is what makes separate pairings comparable when the baseline drifts.
    """
    from metile.tuning import confirm_pairwise

    # The baseline reads 1.0 in one pairing and 2.0 in the other, so raw times would rank
    # `fast` (0.5) and `slow` (1.6) by that drift rather than by merit.
    timings = {"fast": {"base": 1.0, "fast": 0.5}, "slow": {"base": 2.0, "slow": 1.6}}
    group_sizes = []

    def measure(thunk):
        return thunk()

    def thunk_for(subject, key):
        return lambda: timings[subject][key]

    def spy_round_robin(candidates, rounds, measure_fn):
        group_sizes.append(len(candidates))
        return {key: [measure_fn(thunk)] for key, thunk in candidates}

    import metile.tuning.tournament as tournament

    original = tournament.round_robin
    tournament.round_robin = spy_round_robin
    try:
        results = {}
        for subject in ("fast", "slow"):
            candidates = [
                ("base", thunk_for(subject, "base")),
                (subject, thunk_for(subject, subject)),
            ]
            results.update(confirm_pairwise(candidates, "base", 3, measure))
    finally:
        tournament.round_robin = original

    assert group_sizes == [2, 2]
    # fast is 0.5x its baseline, slow is 0.8x its own, so fast must win despite 0.5 < 1.6
    # having been measured against different baseline readings.
    assert results["fast"] / results["base"] < results["slow"] / results["base"]


def test_mlx_lm_prompt_preparation_builds_autoregressive_trajectory(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import cache as cache_module

    from metile.integrations import mlx_lm

    calls = []

    class Model:
        def __call__(self, tokens, cache=None):
            values = tuple(tokens.flatten().tolist())
            calls.append((values, cache))
            next_token = values[-1] + 1
            logits = [0.0] * 16
            logits[next_token] = 1.0
            return mx.array([[logits]])

    monkeypatch.setattr(
        cache_module,
        "make_prompt_cache",
        lambda _model: SimpleNamespace(marker="prompt"),
    )

    cache, elapsed, decode_tokens = mlx_lm._prepare_mlx_lm_prompt(Model(), mx.array([[1, 2, 3]]), 3)

    assert cache.marker == "prompt"
    assert elapsed >= 0.0
    assert [tuple(token.flatten().tolist()) for token in decode_tokens] == [(4,), (5,), (6,)]
    assert [values for values, _ in calls] == [(1, 2, 3), (4,), (5,)]
    assert all(call_cache is not cache for _, call_cache in calls[1:])


def test_mlx_lm_measurement_extension_reuses_provisional_trials(monkeypatch):
    from metile.integrations import mlx_lm

    native = mlx_lm.MLXLMPlan(False, False, False, False)
    compressed = mlx_lm.MLXLMPlan(
        False,
        False,
        False,
        False,
        compressed_down=True,
    )
    provisional = {
        native: [(1.0, 1.0, 1.0)] * 3,
        compressed: [(1.0, 0.8, 0.9)] * 3,
    }
    calls = []

    prepared_prompt = object()

    def measure(*arguments, **options):
        calls.append((arguments[2], arguments[6], options))
        return {
            native: [(1.0, 1.0, 1.0)] * arguments[6],
            compressed: [(1.0, 0.8, 0.9)] * arguments[6],
        }

    _patch_mlx_lm(monkeypatch, "_measure_mlx_lm_plans", measure)

    measured = mlx_lm._extend_mlx_lm_measurements(
        object(),
        object(),
        provisional,
        (native, compressed),
        None,
        None,
        16,
        7,
        prepared_prompt=prepared_prompt,
    )

    assert calls == [
        (
            (native, compressed),
            4,
            {"prepared_prompt": prepared_prompt, "validate_fidelity": False},
        )
    ]
    assert all(len(samples) == 7 for samples in measured.values())


def test_mlx_lm_model_search_screens_full_lattice_before_successive_halving(monkeypatch):
    from metile.integrations import mlx_lm

    compression_names = (
        "compressed_down",
        "compressed_gate_up",
        "compressed_vocab",
        "compressed_attention",
    )
    candidates = tuple(
        mlx_lm.MLXLMPlan(
            False,
            False,
            False,
            False,
            **{name: bool(mask & (1 << index)) for index, name in enumerate(compression_names)},
        )
        for mask in range(1 << len(compression_names))
    )
    native = candidates[0]
    requested = candidates[-1]
    prepared_prompt = object()
    calls = []

    class Tokens:
        ndim = 2
        shape = (1, 128)

    def measure(
        _model,
        _sample_tokens,
        active,
        _affine_prefill,
        _dense_mlp,
        _decode_steps,
        rounds,
        _compressed_down=None,
        _compressed_gate_up=None,
        _compressed_vocab=None,
        _compressed_attention=None,
        *,
        prepared_prompt=None,
        validate_fidelity=True,
    ):
        calls.append((active, rounds, prepared_prompt, validate_fidelity))
        return {
            plan: [
                (
                    1.0,
                    1.0 - 0.03 * plan.feature_count,
                    1.0 - 0.02 * plan.feature_count,
                )
            ]
            * rounds
            for plan in active
        }

    _patch_mlx_lm(monkeypatch, "_mlx_lm_plan_candidates", lambda _requested: candidates)
    _patch_mlx_lm(monkeypatch, "_mlx_lm_warmup_plans", lambda _candidates: (native,))
    _patch_mlx_lm(monkeypatch, "_effective_mlx_lm_plan", lambda plan, *_args: plan)
    _patch_mlx_lm(monkeypatch, "_mlx_lm_plan_key", lambda *_args: "key")
    _patch_mlx_lm(monkeypatch, "_read_mlx_lm_plan", lambda _key: None)
    _patch_mlx_lm(monkeypatch, "_write_mlx_lm_plan", lambda *_args: None)
    _patch_mlx_lm(
        monkeypatch,
        "_prepare_mlx_lm_prompt",
        lambda *_args: prepared_prompt,
    )
    _patch_mlx_lm(monkeypatch, "_measure_mlx_lm_plans", measure)
    _patch_mlx_lm(
        monkeypatch,
        "_mlx_lm_validation_finalists",
        lambda measured: tuple(measured),
    )
    _patch_mlx_lm(
        monkeypatch,
        "_validate_mlx_lm_finalists_repeated",
        lambda *_args: requested,
    )

    selected = mlx_lm.autotune_metile_for_mlx_lm(
        lambda _tokens: None,
        Tokens(),
        attention=False,
        rms_norm=False,
        graph_fusion=False,
        quantized_mlp=False,
        decode_steps=8,
        trials=5,
    )

    assert selected == requested
    assert [len(active) for active, *_ in calls] == [1, 16, 8, 8]
    assert [rounds for _, rounds, *_ in calls] == [1, 1, 2, 2]
    assert all(prompt is prepared_prompt for _, _, prompt, _ in calls)
    assert [validate for *_, validate in calls] == [True, True, False, False]


@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_mlx_rms_norm_matches_mlx_low_precision(dtype_name, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_rms_kernel_cache.clear()
    mlx_backend._mlx_rms_schedule_cache.clear()
    random = np.random.default_rng(17)
    dtype = getattr(mx, dtype_name)
    values = mx.array(random.standard_normal((2, 3, 2048)).astype(np.float32)).astype(dtype)
    weight = mx.array(random.standard_normal((2048,)).astype(np.float32)).astype(dtype)

    actual = mlx_backend.mlx_rms_norm(values, weight, 1e-5, autotune=False)
    expected = mx.fast.rms_norm(values, weight, 1e-5)
    mx.eval(actual, expected)

    tolerance = 4e-2 if dtype_name == "bfloat16" else 2e-3
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_mlx_fused_add_rms_norm_preserves_both_outputs(dtype_name, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.setenv("METILE_DISABLE_DISK_CACHE", "1")
    mlx_backend._mlx_add_rms_kernel_cache.clear()
    mlx_backend._mlx_add_rms_schedule_cache.clear()
    random = np.random.default_rng(29)
    dtype = getattr(mx, dtype_name)
    values = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float32)).astype(dtype)
    residual = mx.array(random.standard_normal((2, 1, 2048)).astype(np.float32)).astype(dtype)
    weight = mx.array(random.standard_normal((2048,)).astype(np.float32)).astype(dtype)

    actual_sum, actual_norm = mlx_backend.mlx_add_rms_norm(
        values, residual, weight, 1e-5, autotune=False
    )
    expected_sum = values + residual
    expected_norm = mx.fast.rms_norm(expected_sum, weight, 1e-5)
    mx.eval(actual_sum, actual_norm, expected_sum, expected_norm)

    np.testing.assert_array_equal(
        np.array(actual_sum.astype(mx.float32)),
        np.array(expected_sum.astype(mx.float32)),
    )
    tolerance = 4e-2 if dtype_name == "bfloat16" else 3e-3
    np.testing.assert_allclose(
        np.array(actual_norm.astype(mx.float32)),
        np.array(expected_norm.astype(mx.float32)),
        rtol=tolerance,
        atol=tolerance,
    )


def test_prepare_mlx_lm_mxfp8_down_requires_explicit_approximation():
    pytest.importorskip("mlx.core")

    from metile.integrations.mlx_lm import prepare_mlx_lm_compressed_down

    with pytest.raises(ValueError, match="allow_approximate"):
        prepare_mlx_lm_compressed_down(lambda: None, format="mxfp8")


@pytest.mark.parametrize(
    ("generated_elapsed", "expected_accepted"),
    ((0.95, True), (1.05, False)),
)
def test_mlx_lm_generation_confirmation_requires_end_to_end_safety(
    generated_elapsed,
    expected_accepted,
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=3, delay=0)
    candidate = MLXLMPlan(False, False, False, False, True)

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(generation_tps=100.0, prompt_tps=1000.0)
        return response, generated_elapsed if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(),
        object(),
        [1, 2, 3],
        arguments,
        candidate,
        None,
        None,
    )

    assert confirmation["accepted"] is expected_accepted
    assert bool(selected.feature_count) is expected_accepted


@pytest.mark.parametrize(
    ("candidate", "expected_accepted"),
    (
        ("prefill", True),
        ("decode", False),
    ),
)
def test_mlx_lm_generation_confirmation_uses_prefill_noise_floor(
    candidate,
    expected_accepted,
    monkeypatch,
):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=3, delay=0)
    plan = (
        MLXLMPlan(False, False, False, False, True)
        if candidate == "prefill"
        else MLXLMPlan(True, False, False, False, False)
    )

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=99.2 if patched else 100.0,
            prompt_tps=1100.0 if patched else 1000.0,
        )
        return response, 0.98 if patched else 1.0, 0.08 if patched else 0.1

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert confirmation["accepted"] is expected_accepted
    assert bool(selected.feature_count) is expected_accepted


def test_mlx_lm_generation_confirmation_rejects_unstable_ttft(monkeypatch):
    from types import SimpleNamespace

    from benchmarks import mlx_lm_backend
    from metile.integrations.mlx_lm import MLXLMPlan

    arguments = SimpleNamespace(confirmation_trials=5, delay=0)
    plan = MLXLMPlan(False, False, False, False, False, True)
    generated_ttft = iter((0.1, 0.1, 0.1, 0.2, 0.2))

    def generate(
        _model,
        _tokenizer,
        _prompt,
        _arguments,
        patched,
        _plan,
        _affine_prefill,
        _dense_mlp,
        _compressed_down,
        _compressed_gate_up,
        _compressed_vocab,
        _compressed_attention,
    ):
        response = SimpleNamespace(
            generation_tps=102.0 if patched else 100.0,
            prompt_tps=1000.0,
        )
        ttft = next(generated_ttft) if patched else 0.1
        return response, 0.98 if patched else 1.0, ttft

    monkeypatch.setattr(mlx_lm_backend, "_generate", generate)

    selected, confirmation = mlx_lm_backend._confirm_plan(
        object(), object(), [1, 2, 3], arguments, plan, None, None
    )

    assert not confirmation["accepted"]
    assert not selected.feature_count


def test_pessimistic_penalises_an_inconsistent_candidate():
    """Ranking must prefer dependable over occasionally-brilliant.

    Measured on a 17408-wide affine matmul, a generated kernel ran 1643us at its fastest and
    2874us typically while native ran 2047 and 2073. The median understates that gap and the
    minimum inverts it, which is how a kernel measuring 0.85x in steady state got selected
    and then cached. A high quantile asks how the kernel behaves when conditions are ordinary.
    """
    from metile.tuning import pessimistic

    steady = [2.05, 2.06, 2.07, 2.07, 2.08, 2.09]
    spiky = [1.64, 1.65, 2.60, 2.80, 2.87, 3.30]

    # The minimum would pick the spiky kernel, and the median is much closer than it should be.
    assert min(spiky) < min(steady)
    # The pessimistic summary puts the steady kernel ahead, which is the decision we want.
    assert pessimistic(steady) < pessimistic(spiky)


def test_pessimistic_agrees_with_the_median_when_spread_is_equal():
    """It must only change decisions in the case it exists to catch."""
    from metile.tuning import pessimistic

    tight = [1.00, 1.01, 1.02, 1.03]
    assert pessimistic(tight) == pytest.approx(1.025, abs=0.01)
    assert pessimistic([2.0]) == 2.0
