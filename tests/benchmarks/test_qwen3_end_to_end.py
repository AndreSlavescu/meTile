import copy
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.megakernels import qwen3_end_to_end as benchmark


def _generation(duration, tokens=None):
    tokens = [4, 5, 6] if tokens is None else tokens
    return {
        "total_wall_seconds": duration,
        "time_to_first_token_seconds": duration / 4,
        "decode_wall_seconds": duration * 3 / 4,
        "actual_output_tokens": len(tokens),
        "generated_token_ids": list(tokens),
    }


def _samples():
    return [
        {"order": ["MLX", "meTile"], "MLX": _generation(1), "meTile": _generation(2)},
        {"order": ["meTile", "MLX"], "MLX": _generation(6), "meTile": _generation(3)},
        {"order": ["MLX", "meTile"], "MLX": _generation(10), "meTile": _generation(1)},
    ]


class FakeCache:
    def __init__(self):
        self.keys = np.zeros((1, 2, 64, 3), dtype=np.float32)
        self.values = np.zeros_like(self.keys)
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values


def _logits(token, position):
    output = np.arange(13, dtype=np.float32) * 0.01
    output[(token + position + 1) % 13] = 4
    return output


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, cache):
        self.calls.append(inputs.tolist()[0])
        outputs = []
        for token in inputs[0]:
            position = cache[0].offset
            for layer_index, layer in enumerate(cache):
                layer.keys[:, :, position] = token + position + layer_index
                layer.values[:, :, position] = token - position - layer_index
                layer.offset += 1
            outputs.append(_logits(token, position))
        return np.asarray([outputs])


class FakeCandidate:
    def __init__(self):
        self.storage = np.zeros((2, 2, 2, 64, 3), dtype=np.float32)
        self.output = np.zeros(13, dtype=np.float32)
        self.cache = SimpleNamespace(numpy=lambda: self.storage)
        self.logits = SimpleNamespace(numpy=lambda: self.output)
        self.calls = []
        self.resets = 0
        self.greedy_calls = 0

    def reset(self):
        self.storage.fill(0)
        self.resets += 1

    def forward(self, token, position, project=True):
        self.calls.append((token, position, project))
        for layer_index in range(2):
            self.storage[0, layer_index, :, position] = token + position + layer_index
            self.storage[1, layer_index, :, position] = token - position - layer_index
        if project:
            self.output[:] = _logits(token, position)

    def greedy(self):
        self.greedy_calls += 1
        return int(np.argmax(self.output))


@pytest.fixture
def fake_mlx(monkeypatch):
    events = []
    fake_mx = SimpleNamespace(
        array=np.array,
        int32=np.int32,
        eval=lambda *arrays: events.append("eval"),
        argmax=np.argmax,
        synchronize=lambda: events.append("synchronize"),
    )
    fake_cache = SimpleNamespace(make_prompt_cache=lambda model: [FakeCache(), FakeCache()])
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)
    return events


def test_defaults_are_real_autoregressive_generations():
    arguments = benchmark._arguments([])
    assert arguments.model == "Qwen/Qwen3-0.6B"
    assert arguments.dtype == "float32"
    assert arguments.prompt_lengths == [8, 32]
    assert arguments.output_lengths == [8, 16]
    assert arguments.trials == 5


@pytest.mark.parametrize(
    "arguments",
    [
        ["--prompt-lengths", "0"],
        ["--prompt-lengths", "8", "8"],
        ["--output-lengths", "1"],
        ["--output-lengths", "8", "8"],
        ["--trials", "2"],
        ["--threads", "33"],
        ["--seed", "-1"],
    ],
)
def test_invalid_generation_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._arguments(arguments)


def test_checkout_verifies_gpu_wide_kernels(monkeypatch):
    from benchmarks.common import checkout

    calls = []
    monkeypatch.setattr(checkout, "load_kernel", lambda root, name: calls.append((root, name)))
    root = benchmark._benchmark_checkout()
    assert calls == [(root, "megakernels.qwen3_staged")]


def test_summary_uses_paired_ratios_and_n_minus_one_decode_tokens():
    summary = benchmark._summarize(_samples())
    total = summary["total_wall_seconds"]
    assert total == {"mlx": 6, "metile": 2, "paired_speedup": 2}
    assert total["paired_speedup"] != total["mlx"] / total["metile"]
    assert summary["decode_tokens_per_second"]["mlx"] == pytest.approx(2 / 4.5)
    assert summary["decode_tokens_per_second"]["metile"] == pytest.approx(2 / 1.5)


@pytest.mark.parametrize(
    "metric", ["total_wall_seconds", "time_to_first_token_seconds", "decode_wall_seconds"]
)
@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan, True])
def test_summary_rejects_invalid_durations(metric, duration):
    samples = _samples()
    samples[0]["meTile"][metric] = duration
    with pytest.raises(ValueError, match="positive and finite"):
        benchmark._summarize(samples)


def test_summary_rejects_inconsistent_total():
    samples = _samples()
    samples[0]["MLX"]["decode_wall_seconds"] += 1
    with pytest.raises(ValueError, match="TTFT plus decode"):
        benchmark._summarize(samples)


def test_summary_requires_alternation_and_repeated_trials():
    with pytest.raises(ValueError, match="three"):
        benchmark._summarize(_samples()[:2])
    samples = _samples()
    samples[1]["order"] = ["MLX", "meTile"]
    with pytest.raises(ValueError, match="alternate"):
        benchmark._summarize(samples)


@pytest.mark.parametrize("field", ["generated_token_ids", "actual_output_tokens"])
def test_summary_rejects_changed_actual_tokens_or_count(field):
    samples = _samples()
    samples[2]["meTile"][field] = [4, 5, 7] if field == "generated_token_ids" else 2
    with pytest.raises(benchmark.GenerationValidationFailure):
        benchmark._summarize(samples)


def test_generation_sample_has_exact_observed_boundaries():
    sample = benchmark._generation_sample(10, 110, 510, [4, 5, 6])
    assert sample["total_wall_seconds"] == pytest.approx(500e-9)
    assert sample["time_to_first_token_seconds"] == pytest.approx(100e-9)
    assert sample["decode_wall_seconds"] == pytest.approx(400e-9)
    assert sample["actual_output_tokens"] == 3


def test_native_generation_batches_prefill_then_feeds_back_generated_ids(fake_mlx):
    model = FakeModel()
    result = benchmark._native_generation(model, [3, 5, 2], 3)
    assert model.calls == [[3, 5], [2], [5], [9]]
    assert result["generated_token_ids"] == [5, 9, 1]
    assert result["actual_output_tokens"] == 3
    assert fake_mlx.count("synchronize") == 3


def test_one_token_native_prompt_does_not_call_empty_prefill(fake_mlx):
    model = FakeModel()
    benchmark._native_generation(model, [3], 2)
    assert model.calls == [[3], [4]]


def test_candidate_generation_resets_and_advances_cache_without_prefix_logits():
    candidate = FakeCandidate()
    result = benchmark._candidate_generation(candidate, [3, 5, 2], 3)
    assert candidate.resets == 1
    assert candidate.calls == [
        (3, 0, False),
        (5, 1, False),
        (2, 2, True),
        (5, 3, True),
        (9, 4, True),
    ]
    assert candidate.greedy_calls == 3
    assert result["generated_token_ids"] == [5, 9, 1]
    assert result["actual_output_tokens"] == 3


def test_validation_checks_batched_prompt_and_each_advanced_decode_step(fake_mlx):
    model = FakeModel()
    candidate = FakeCandidate()
    result = benchmark._validate_workload(model, candidate, [3, 5, 2], 3, "float32")
    assert model.calls == [[3, 5, 2], [5], [9]]
    assert candidate.calls == [(3, 0, True), (5, 1, True), (2, 2, True), (5, 3, True), (9, 4, True)]
    assert result["passed"]
    assert result["generated_token_ids"] == [5, 9, 1]
    assert len(result["checks"]) == 5
    for position, check in enumerate(result["checks"]):
        assert check["passed"]
        assert check["cache"]["valid_tokens_checked"] == position + 1
        assert check["cache"]["layers_checked"] == 2
        assert check["cache"]["keys_and_values_checked"]


@pytest.mark.parametrize("kind", [0, 1])
def test_validation_rejects_old_cache_corruption_even_when_tokens_match(fake_mlx, kind):
    candidate = FakeCandidate()
    forward = candidate.forward

    def corrupt(token, position, project=True):
        forward(token, position, project)
        if position == 3:
            candidate.storage[kind, 1, :, 0] += 1

    candidate.forward = corrupt
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 3, "float32")
    failed = error.value.checks[-1]
    assert failed["position"] == 3
    assert failed["logits"]["passed"]
    assert not failed["cache"]["passed"]


def test_validation_rejects_broken_gpu_argmax_independently_of_logits(fake_mlx):
    candidate = FakeCandidate()
    candidate.greedy = lambda: 12
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3], 2, "float32")
    failed = error.value.checks[-1]
    assert failed["logits"]["passed"]
    assert failed["cache"]["passed"]
    assert not failed["gpu_greedy_selection"]["passed"]


def test_validation_handles_nonfinite_arrays_as_structured_failure(fake_mlx):
    candidate = FakeCandidate()
    candidate.logits = SimpleNamespace(numpy=lambda: np.full(13, np.nan))
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3], 2, "float32")
    assert "finite" in error.value.checks[-1]["error"]


def test_later_validation_failure_prevents_all_timing(monkeypatch):
    calls = []

    def validate(model, candidate, prompt, output_tokens, dtype):
        calls.append((len(prompt), output_tokens))
        if len(prompt) > 1:
            raise benchmark.GenerationValidationFailure([{"passed": False}])
        return {"passed": True, "generated_token_ids": [4, 5]}

    monkeypatch.setattr(benchmark, "_validate_workload", validate)
    monkeypatch.setattr(benchmark, "_measure_workload", lambda *args: pytest.fail("must not time"))
    arguments = SimpleNamespace(output_lengths=[2], dtype="float32", trials=3)
    status, results = benchmark._workload_results(arguments, object(), object(), [[3], [3, 5]])
    assert status == "validation_failed"
    assert calls == [(1, 2), (2, 2)]
    assert all("samples" not in result and "medians" not in result for result in results)


def test_all_workloads_validate_before_timing_and_late_mismatch_discards_all_samples(monkeypatch):
    events = []

    def validate(model, candidate, prompt, output_tokens, dtype):
        events.append(("validate", len(prompt)))
        return {"passed": True, "generated_token_ids": [4, 5, 6]}

    def measure(model, candidate, prompt, output_tokens, trials, expected):
        events.append(("measure", len(prompt)))
        if len(prompt) == 2:
            raise benchmark.GenerationValidationFailure([{"expected": expected, "actual": [4]}])
        return {"samples": _samples(), "medians": benchmark._summarize(_samples())}

    monkeypatch.setattr(benchmark, "_validate_workload", validate)
    monkeypatch.setattr(benchmark, "_measure_workload", measure)
    arguments = SimpleNamespace(output_lengths=[3], dtype="float32", trials=3)
    status, results = benchmark._workload_results(arguments, object(), object(), [[3], [3, 5]])
    assert events == [("validate", 1), ("validate", 2), ("measure", 1), ("measure", 2)]
    assert status == "validation_failed"
    assert results[-1]["status"] == "timing_validation_failed"
    assert all("samples" not in result and "medians" not in result for result in results)


def test_measurement_has_untimed_warmups_alternating_pairs_and_syncs(monkeypatch, fake_mlx):
    events = []
    device = SimpleNamespace(sync=lambda: events.append("sync"))
    monkeypatch.setitem(
        sys.modules,
        "metile.runtime.metal_device",
        SimpleNamespace(MetalDevice=SimpleNamespace(get=lambda: device)),
    )

    def operation(label):
        events.append(label)
        return _generation(2)

    monkeypatch.setattr(benchmark, "_native_generation", lambda *args: operation("MLX"))
    monkeypatch.setattr(benchmark, "_candidate_generation", lambda *args: operation("meTile"))
    result = benchmark._measure_workload(object(), object(), [3], 3, 3, [4, 5, 6])
    labels = [event for event in events if event != "sync"]
    assert labels == ["MLX", "meTile"] * 3 + ["meTile", "MLX", "MLX", "meTile"]
    assert events.count("sync") == fake_mlx.count("synchronize") == 10
    assert len(result["samples"]) == 3
    assert result["medians"]["total_wall_seconds"]["paired_speedup"] == 1


def test_summary_does_not_mutate_raw_observations():
    samples = _samples()
    before = copy.deepcopy(samples)
    benchmark._summarize(samples)
    assert samples == before


def test_main_saves_failure_artifact_and_exits_nonzero(monkeypatch, tmp_path):
    report = {"status": "validation_failed", "results": [{"correctness": {"passed": False}}]}
    monkeypatch.setattr(benchmark, "run", lambda arguments: report)
    output = tmp_path / "failure.json"
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(output)])
    assert error.value.code == 1
    assert json.loads(output.read_text()) == report
