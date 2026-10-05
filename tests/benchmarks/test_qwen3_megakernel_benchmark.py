import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.megakernels.qwen3 import (
    CACHE_TOLERANCE,
    DEFAULT_MODEL,
    FP32_TOLERANCE,
    LOGIT_TOLERANCE,
    VALIDATION_STEPS,
    ValidationFailure,
    _arguments,
    _array_fidelity,
    _benchmark_checkout,
    _cache_fidelity,
    _context_results,
    _fidelity_tolerances,
    _logit_fidelity,
    _padded_cache,
    _require_precision_environment,
    _sha256,
    _summarize,
    _validate_checkpoint_config,
)


def _samples():
    return [
        {"order": ["MLX", "meTile"], "mlx_wall_seconds": 1.0, "metile_wall_seconds": 2.0},
        {"order": ["meTile", "MLX"], "mlx_wall_seconds": 6.0, "metile_wall_seconds": 3.0},
        {"order": ["MLX", "meTile"], "mlx_wall_seconds": 10.0, "metile_wall_seconds": 1.0},
    ]


def test_defaults_identify_real_small_model_and_fixed_context_workload():
    arguments = _arguments([])
    assert arguments.model == DEFAULT_MODEL == "Qwen/Qwen3-0.6B"
    assert arguments.contexts == [0, 16]
    assert arguments.trials == 9
    assert arguments.dtype == "float32"
    assert VALIDATION_STEPS >= 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["--contexts", "-1"],
        ["--contexts", "16", "16"],
        ["--trials", "2"],
        ["--threads", "0"],
        ["--threads", "33"],
        ["--threads", "96"],
        ["--threads", "1024"],
    ],
)
def test_invalid_workload_is_rejected(arguments):
    with pytest.raises(SystemExit):
        _arguments(arguments)


def test_summary_uses_median_of_paired_ratios_not_ratio_of_medians():
    summary = _summarize(_samples())
    assert summary["mlx_wall_seconds"] == 6.0
    assert summary["metile_wall_seconds"] == 2.0
    assert summary["paired_speedup"] == 2.0
    assert summary["paired_speedup"] != summary["mlx_wall_seconds"] / summary["metile_wall_seconds"]


@pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, math.nan, True])
def test_summary_rejects_invalid_durations(duration):
    samples = _samples()
    samples[0]["metile_wall_seconds"] = duration
    with pytest.raises(ValueError, match="positive and finite"):
        _summarize(samples)


def test_summary_requires_multiple_trials_and_alternating_order():
    with pytest.raises(ValueError, match="three"):
        _summarize(_samples()[:2])
    samples = _samples()
    samples[1]["order"] = ["MLX", "meTile"]
    with pytest.raises(ValueError, match="alternate"):
        _summarize(samples)


def test_fidelity_checks_complete_arrays_not_only_argmax():
    reference = np.array([0.0, 5.0, 1.0], dtype=np.float16)
    actual = np.array([0.0, 5.0, 2.0], dtype=np.float16)
    result = _logit_fidelity(reference, actual)
    assert result["greedy_token_equal"]
    assert not result["passed"]
    assert result["max_absolute_error"] == 1.0


def test_fidelity_rejects_changed_greedy_token_even_inside_tolerance():
    result = _logit_fidelity([1.0, 1.001], [1.002, 1.001])
    assert not result["greedy_token_equal"]
    assert not result["passed"]


def test_fidelity_accepts_small_bounded_roundoff_with_same_token():
    result = _logit_fidelity([0.0, 2.0, -1.0], [0.001, 2.001, -1.001])
    assert result["passed"]
    assert result["mlx_next_token"] == result["metile_next_token"] == 1


@pytest.mark.parametrize("values", [[math.nan], [math.inf], []])
def test_nonfinite_or_empty_correctness_arrays_are_rejected(values):
    with pytest.raises(ValueError, match="finite arrays"):
        _array_fidelity(values, values, CACHE_TOLERANCE)


def test_array_shape_mismatch_is_rejected_without_broadcasting():
    with pytest.raises(ValueError, match="shape mismatch"):
        _array_fidelity(np.ones((2, 1)), np.ones((2,)), CACHE_TOLERANCE)


def test_file_fingerprint_covers_contents(tmp_path):
    source = tmp_path / "weights"
    source.write_bytes(b"abc")
    assert _sha256(source) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_snapshot_resolution_forces_offline_cache(monkeypatch, tmp_path):
    from benchmarks.megakernels.qwen3 import _local_snapshot

    calls = []

    def snapshot_download(model, **options):
        calls.append((model, options))
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    assert _local_snapshot(DEFAULT_MODEL, "exact-revision") == tmp_path / "snapshot"
    assert calls == [(DEFAULT_MODEL, {"revision": "exact-revision", "local_files_only": True})]


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("token", [0, 2])
def test_cache_fidelity_checks_all_layers_past_tokens_and_both_cache_kinds(kind, token):
    storage = np.ones((2, 2, 2, 4, 3), dtype=np.float16)
    storage[kind, 1, 0, token, 0] += 1
    candidate = SimpleNamespace(cache=SimpleNamespace(numpy=lambda: storage))
    native = [
        SimpleNamespace(
            keys=np.ones((1, 2, 4, 3), dtype=np.float16),
            values=np.ones((1, 2, 4, 3), dtype=np.float16),
        )
        for _ in range(2)
    ]
    result = _cache_fidelity(native, candidate, 3)
    assert not result["passed"]
    assert result["layers_checked"] == 2
    assert result["valid_tokens_checked"] == 3
    assert result["max_absolute_error"] == 1.0


def test_cache_fidelity_excludes_unused_capacity():
    storage = np.ones((2, 1, 2, 4, 3), dtype=np.float16)
    storage[:, :, :, 3] = np.nan
    candidate = SimpleNamespace(cache=SimpleNamespace(numpy=lambda: storage))
    native = [
        SimpleNamespace(
            keys=np.ones((1, 2, 4, 3), dtype=np.float16),
            values=np.ones((1, 2, 4, 3), dtype=np.float16),
        )
    ]
    assert _cache_fidelity(native, candidate, 3)["passed"]


def test_existing_snapshot_path_does_not_call_hub(monkeypatch, tmp_path):
    from benchmarks.megakernels.qwen3 import _local_snapshot

    def unexpected_download(*args, **kwargs):
        pytest.fail("local snapshots must not use the hub")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected_download)
    )
    assert _local_snapshot(str(tmp_path), None) == tmp_path


def test_benchmark_checkout_verifies_the_actual_kernel_import(monkeypatch):
    from benchmarks.common import checkout

    calls = []
    monkeypatch.setattr(checkout, "load_kernel", lambda root, module: calls.append((root, module)))
    root = _benchmark_checkout()
    assert calls == [(root, "megakernels.qwen3")]
    assert (root / "metile" / "__init__.py").is_file()


@pytest.mark.parametrize("package", ["metile", "metile_kernels"])
def test_benchmark_checkout_rejects_previously_loaded_packages_from_elsewhere(
    monkeypatch, tmp_path, package
):
    external_module = SimpleNamespace(__file__=str(tmp_path / package / "__init__.py"))
    monkeypatch.setitem(sys.modules, package, external_module)
    with pytest.raises(RuntimeError, match="outside the selected checkout"):
        _benchmark_checkout()


@pytest.mark.parametrize("config", [{}, {"model_type": "qwen3_5"}, [], None])
def test_checkpoint_guard_rejects_unexpected_architectures(config):
    with pytest.raises(ValueError, match="dense Qwen3"):
        _validate_checkpoint_config(config)


@pytest.mark.parametrize("field", ["quantization", "quantization_config"])
def test_checkpoint_guard_rejects_both_quantization_config_formats(field):
    with pytest.raises(ValueError, match="unquantized"):
        _validate_checkpoint_config({"model_type": "qwen3", field: {"bits": 4}})


@pytest.mark.parametrize("model_file", ["custom.py", None])
def test_checkpoint_guard_rejects_custom_executable_model_configuration(model_file):
    with pytest.raises(ValueError, match="executable model"):
        _validate_checkpoint_config({"model_type": "qwen3", "model_file": model_file})


def test_checkpoint_guard_accepts_the_standard_dense_architecture():
    assert _validate_checkpoint_config({"model_type": "qwen3", "torch_dtype": "bfloat16"}) is None


def test_precision_tolerances_preserve_fp16_and_require_tighter_fp32_agreement():
    assert _fidelity_tolerances("float16") == (LOGIT_TOLERANCE, CACHE_TOLERANCE)
    assert _fidelity_tolerances("float32") == (FP32_TOLERANCE, FP32_TOLERANCE)
    assert FP32_TOLERANCE == {"rtol": 0.001, "atol": 0.001}
    assert not _logit_fidelity([0.0, 1.0], [0.01, 1.0], FP32_TOLERANCE)["passed"]
    assert _logit_fidelity([0.0, 1.0], [0.01, 1.0], LOGIT_TOLERANCE)["passed"]
    with pytest.raises(ValueError, match="comparison dtypes"):
        _fidelity_tolerances("bfloat16")


@pytest.mark.parametrize("setting", [None, "1"])
def test_fp32_requires_explicit_tf32_disable_before_mlx_import(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", setting)
    with pytest.raises(RuntimeError, match="before importing MLX"):
        _require_precision_environment("float32")
    assert _require_precision_environment("float16") is None


def test_fp32_accepts_tf32_disabled_environment(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    assert _require_precision_environment("float32") is None


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_native_cache_padding_preserves_storage_dtype(monkeypatch, dtype):
    fake_mx = SimpleNamespace(array=np.array, eval=lambda *arrays: None)
    fake_cache = SimpleNamespace(KVCache=lambda: SimpleNamespace(keys=None, values=None, offset=0))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)
    previous = SimpleNamespace(
        keys=np.ones((1, 2, 3, 4), dtype=dtype),
        values=np.full((1, 2, 3, 4), 2, dtype=dtype),
        offset=2,
    )
    padded = _padded_cache([previous], 6)[0]
    assert padded.keys.dtype == padded.values.dtype == dtype
    assert padded.offset == 2
    np.testing.assert_array_equal(padded.keys[:, :, :2], previous.keys[:, :, :2])
    np.testing.assert_array_equal(padded.values[:, :, :2], previous.values[:, :, :2])
    assert not padded.keys[:, :, 2:].any()


def test_later_context_validation_failure_prevents_all_timings(monkeypatch):
    from benchmarks.megakernels import qwen3

    failed_checks = [{"position": 1, "logits": {"passed": True}, "cache": {"passed": False}}]

    def validate(model, candidate, cache, token, context, capacity, dtype):
        if context:
            raise ValidationFailure(context, failed_checks)
        return [{"position": 0, "logits": {"passed": True}, "cache": {"passed": True}}]

    monkeypatch.setattr(qwen3, "_prefix_cache", lambda model, prefix: object())
    monkeypatch.setattr(qwen3, "_validate_context", validate)
    monkeypatch.setattr(
        qwen3, "_measure_context", lambda *args: pytest.fail("must not time failures")
    )
    arguments = SimpleNamespace(contexts=[0, 1], dtype="float32", trials=9)
    status, results = _context_results(arguments, object(), object(), 4, [5, 6, 7, 8])
    assert status == "validation_failed"
    assert results[0]["status"] == "validated"
    assert results[1]["correctness"] == failed_checks
    assert all("samples" not in result and "medians" not in result for result in results)


def test_main_saves_structured_failure_before_exiting_nonzero(monkeypatch, tmp_path):
    from benchmarks.megakernels import qwen3

    report = {
        "status": "validation_failed",
        "geometry": {"dtype": "float16"},
        "results": [{"correctness": [{"position": 1, "cache": {"passed": False}}]}],
    }
    monkeypatch.setattr(qwen3, "run", lambda arguments: report)
    destination = tmp_path / "failure.json"
    with pytest.raises(SystemExit) as error:
        qwen3.main(["--dtype", "float16", "--output", str(destination)])
    assert error.value.code == 1
    assert json.loads(destination.read_text()) == report
