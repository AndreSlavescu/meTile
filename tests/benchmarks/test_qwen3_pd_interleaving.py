import copy
import json
import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.megakernels import qwen3_pd_interleaving as benchmark
from tests.benchmarks.test_qwen3_end_to_end import FakeCandidate, FakeModel, _logits
from tests.benchmarks.test_qwen3_end_to_end import fake_mlx as fake_mlx


class FakeRequest(FakeCandidate):
    def __init__(self, name, events, chunk_size=2):
        super().__init__()
        self.name = name
        self.events = events
        self.chunk_size = chunk_size
        self.cached_tokens = 0
        self.final = False

    def reset(self, *, clear_cache=True):
        super().reset()
        self.clear_cache = clear_cache
        self.cached_tokens = 0
        self.final = False
        self.events.append((self.name, "reset"))

    def prefill_chunk(self, tokens, *, final_chunk):
        assert 0 < len(tokens) <= self.chunk_size
        self.events.append((self.name, "prefill", self.cached_tokens, len(tokens), final_chunk))
        for token in tokens:
            super().forward(token, self.cached_tokens, project=False)
            self.cached_tokens += 1
            self.last_token = token
        self.final = final_chunk

    def project_last_prefill(self):
        assert self.final
        self.events.append((self.name, "project"))
        self.output[:] = _logits(self.last_token, self.cached_tokens - 1)

    def forward(self, token, position, project=True):
        assert position == self.cached_tokens
        self.events.append((self.name, "decode", token, position))
        super().forward(token, position, project)
        self.cached_tokens += 1


def _setup(chunk_size=2):
    events = []
    candidates = {name: FakeRequest(name, events, chunk_size) for name in benchmark.REQUESTS}
    prompts = {"A": [3, 5, 2, 6, 4], "B": [9, 1, 2, 3, 5]}
    return candidates, prompts, events


class Clock:
    def __init__(self):
        self.nanoseconds = 0

    def __call__(self):
        self.nanoseconds += 10_000_000
        return self.nanoseconds


def _run(policy, output_tokens=4, chunk_size=2):
    candidates, prompts, events = _setup(chunk_size)

    def sync():
        events.append(("sync",))

    initial = benchmark._prime(candidates["A"], prompts["A"], sync)
    events.clear()
    sample = benchmark._execute(
        policy, candidates, prompts, output_tokens, initial, sync, 0.1, clock=Clock()
    )
    return sample, candidates, prompts, events


def _args(**updates):
    arguments = benchmark._arguments([])
    arguments.output_tokens = 4
    arguments.policies = list(benchmark.POLICIES)
    arguments.trials = 3
    arguments.cache_check_interval = 2
    for name, value in updates.items():
        setattr(arguments, name, value)
    return arguments


def test_defaults_describe_fixed_two_request_interference_not_a_model_speed_comparison():
    args = benchmark._arguments([])
    assert args.prompt_tokens == 4096 and args.output_tokens == 1024
    assert args.chunk_size == 128 and args.trials == 3 and args.warmups == 1
    assert args.policies == ["eager", "chunked"]
    assert args.stall_threshold_ms == 100
    assert args.cache_check_interval == 64
    assert args.logical_cache_reset is False
    assert set(args.request_a_sources).isdisjoint(args.request_b_sources)


def test_logical_reset_is_explicit_and_applies_to_both_requests():
    arguments = benchmark._arguments(["--logical-cache-reset"])
    assert arguments.logical_cache_reset is True
    candidates, prompts, _ = _setup()
    initial = benchmark._prime(candidates["A"], prompts["A"], lambda: None, clear_cache=False)
    assert candidates["A"].clear_cache is False
    benchmark._execute(
        "chunked",
        candidates,
        prompts,
        4,
        initial,
        lambda: None,
        0.1,
        clock=Clock(),
        clear_cache=False,
    )
    assert candidates["B"].clear_cache is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--prompt-tokens", "0"],
        ["--output-tokens", "1"],
        ["--chunk-size", "0"],
        ["--chunk-size", "4097"],
        ["--trials", "0"],
        ["--warmups", "0"],
        ["--cache-check-interval", "0"],
        ["--stall-threshold-ms", "0"],
        ["--stall-threshold-ms", "nan"],
        ["--stall-threshold-ms", "inf"],
        ["--policies", "fifo"],
        ["--policies", "eager", "eager"],
    ],
)
def test_invalid_experiment_contract_is_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._arguments(arguments)


@pytest.mark.parametrize("policy", benchmark.POLICIES)
@pytest.mark.parametrize("output_tokens", [2, 4, 10])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_every_policy_produces_identical_work_and_exact_advancing_trajectories(
    policy, output_tokens, chunk_size
):
    sample, candidates, prompts, events = _run(policy, output_tokens, chunk_size)
    for name in benchmark.REQUESTS:
        expected = []
        token = prompts[name][-1]
        for position in range(len(prompts[name]) - 1, len(prompts[name]) + output_tokens - 1):
            token = int(np.argmax(_logits(token, position)))
            expected.append(token)
        observed = sample["requests"][name]
        assert observed["generated_token_ids"] == expected
        assert observed["actual_output_tokens"] == output_tokens
        assert observed["timed_output_tokens"] == output_tokens - int(name == "A")
        assert candidates[name].cached_tokens == len(prompts[name]) + output_tokens - 1
    assert sample["metrics"]["timed_output_tokens"] == 2 * output_tokens - 1
    assert (
        sample["metrics"]["output_tokens_per_second"]
        == (2 * output_tokens - 1) / sample["metrics"]["makespan_seconds"]
    )
    assert sample["requests"]["A"]["emission_times_seconds"][0] == 0
    assert not sample["requests"]["A"]["initial_token_timed"]
    assert sample["requests"]["B"]["initial_token_timed"]
    prefill = [event for event in events if event[:2] == ("B", "prefill")]
    assert sum(event[3] for event in prefill) == len(prompts["B"])
    assert [event[4] for event in prefill] == [False] * (len(prefill) - 1) + [True]
    assert events.count(("B", "project")) == 1
    assert not any(event[:2] == ("A", "reset") for event in events)


def test_policy_quantum_orders_are_explicit_and_have_no_same_request_phase_overlap():
    expected = {
        "fifo": [("A", "decode")] * 3 + [("B", "prefill")] * 3 + [("B", "decode")] * 3,
        "eager": [("B", "prefill")] * 3 + [("A", "decode"), ("B", "decode")] * 3,
        "chunked": [("A", "decode"), ("B", "prefill")] * 3 + [("B", "decode")] * 3,
    }
    for policy in benchmark.POLICIES:
        sample, _, _, _ = _run(policy)
        events = sample["events"]
        assert [(event["request"], event["kind"]) for event in events[1:]] == expected[policy]
        assert events[0]["kind"] == "reset"
        assert all(event["completed_seconds"] >= event["started_seconds"] for event in events)
        assert all(
            right["started_seconds"] > left["completed_seconds"] for left, right in pairwise(events)
        )


def test_chunked_serves_each_ready_decode_before_one_prefill_chunk():
    sample, _, _, _ = _run("chunked", output_tokens=6)
    events = [(event["request"], event["kind"]) for event in sample["events"]]
    assert events[:7] == [
        ("B", "reset"),
        ("A", "decode"),
        ("B", "prefill"),
        ("A", "decode"),
        ("B", "prefill"),
        ("A", "decode"),
        ("B", "prefill"),
    ]
    assert events[7:11] == [("A", "decode"), ("B", "decode"), ("A", "decode"), ("B", "decode")]


def test_initial_resume_and_overlap_scopes_do_not_hide_an_eager_prefill_stall():
    eager, _, _, _ = _run("eager")
    chunked, _, _, _ = _run("chunked")
    fifo, _, _, _ = _run("fifo")
    assert (
        eager["metrics"]["a_initial_resumption_gap_seconds"]
        > chunked["metrics"]["a_initial_resumption_gap_seconds"]
    )
    assert eager["active_interference"]["b_prefill"]["a_gap_distribution"]["count"] == 1
    assert chunked["active_interference"]["b_prefill"]["a_gap_distribution"]["count"] == 2
    assert fifo["active_interference"]["b_prefill"]["a_gap_distribution"]["count"] == 0
    assert fifo["active_interference"]["b_prefill"]["a_gap_distribution"]["p95_seconds"] is None


def test_stall_threshold_is_strict_and_quantiles_are_linear():
    result = benchmark._distribution([0.01, 0.1, 0.2, 0.5], 0.1)
    assert result["stall_count"] == 2
    assert result["stall_excess_seconds"] == pytest.approx(0.5)
    assert result["mean_seconds"] == pytest.approx(0.2025)
    assert result["p50_seconds"] == pytest.approx(0.15)
    assert result["p95_seconds"] == pytest.approx(0.455)
    assert result["p99_seconds"] == pytest.approx(0.491)
    assert result["max_seconds"] == 0.5


@pytest.mark.parametrize("value", [math.nan, math.inf, True, -1, 0])
def test_invalid_emission_timestamps_cannot_be_summarized(value):
    sample, _, _, _ = _run("eager")
    sample["requests"]["A"]["emission_times_seconds"][1] = value
    with pytest.raises(ValueError, match="emission timestamps"):
        benchmark._sample_metrics(sample, 0.1)


@pytest.mark.parametrize("policy", benchmark.POLICIES)
def test_actual_schedule_validates_all_prompt_boundaries_logits_cache_and_ids(fake_mlx, policy):
    candidates, prompts, _ = _setup()
    model = FakeModel()
    result = benchmark._validate_policy(model, candidates, prompts, _args(), policy, lambda: None)
    assert result["passed"]
    assert model.calls[:4] == [
        prompts["A"][:-1],
        [prompts["A"][-1]],
        prompts["B"][:-1],
        [prompts["B"][-1]],
    ]
    for name in benchmark.REQUESTS:
        validation = result["requests"][name]
        assert validation["chunk_boundary_cache_checks"] == 3
        assert validation["generation_logit_cache_greedy_checks"] == 4
        assert validation["actual_output_tokens"] == 4
        checks = validation["checks"]
        assert len(checks) == 7
        assert [check["valid_tokens"] for check in checks[:3]] == [2, 4, 5]
        for check in checks:
            assert check["passed"] and check["cache"]["layers_checked"] == 2
            assert check["cache"]["keys_and_values_checked"]
            assert check["cache"]["rtol"] == check["cache"]["atol"] == 0.001
        generation = checks[3:]
        assert [check["cache"]["full_prefix_checked"] for check in generation] == [
            True,
            True,
            False,
            True,
        ]
        assert generation[2]["cache"]["positions_checked"] == [6]


@pytest.mark.parametrize("corruption", ["old_cache", "new_cache", "logits", "greedy", "nonfinite"])
def test_validation_rejects_errors_even_if_other_requests_succeed(fake_mlx, corruption):
    candidates, prompts, _ = _setup()
    candidate = candidates["B"]
    if corruption == "greedy":
        candidate.greedy = lambda: 12
    elif corruption == "nonfinite":
        candidate.logits = SimpleNamespace(numpy=lambda: np.full(13, np.nan))
    else:
        original = candidate.forward

        def corrupt(token, position, project=True):
            original(token, position, project)
            if corruption == "old_cache":
                candidate.storage[0, 1, :, 0] += 1
            elif corruption == "new_cache":
                candidate.storage[1, 0, :, position] += 1
            else:
                candidate.output[0] += 1

        candidate.forward = corrupt
    with pytest.raises(benchmark.GenerationValidationFailure) as failure:
        benchmark._validate_policy(
            FakeModel(), candidates, prompts, _args(), "chunked", lambda: None
        )
    assert failure.value.checks[-1]["request"] == "B"


def test_validation_catches_shared_request_cache_contamination(fake_mlx):
    candidates, prompts, _ = _setup()
    candidates["B"].storage = candidates["A"].storage
    with pytest.raises(benchmark.GenerationValidationFailure):
        benchmark._validate_policy(
            FakeModel(), candidates, prompts, _args(), "chunked", lambda: None
        )


def test_all_policies_validate_before_warmup_and_rotating_trials(fake_mlx, monkeypatch):
    candidates, prompts, _ = _setup()
    args = _args()
    validations = {}

    def validate(policy):
        validations[policy] = benchmark._validate_policy(
            FakeModel(), candidates, prompts, args, policy, lambda: None
        )
        return validations[policy]

    original = benchmark._prime
    measurements = []

    def prime(*args, **kwargs):
        if len(validations) == 3:
            measurements.append("prime")
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_prime", prime)
    result = benchmark._run_policies(args, candidates, prompts, validate, lambda: None)
    assert result["status"] == "ok"
    assert len(result["samples"]) == 3
    assert len(measurements) == 12
    assert [sample["order"] for sample in result["samples"]] == [
        ["fifo", "eager", "chunked"],
        ["eager", "chunked", "fifo"],
        ["chunked", "fifo", "eager"],
    ]
    assert set(result["paired_ratios"]) == {"eager", "chunked"}
    assert all(
        result["medians"][policy]["timed_output_tokens"] == 7 for policy in benchmark.POLICIES
    )
    json.dumps(result, allow_nan=False)


def test_validation_failure_prevents_all_measurements(monkeypatch):
    calls = []

    def validate(policy):
        calls.append(policy)
        raise benchmark.GenerationValidationFailure([{"passed": False}])

    monkeypatch.setattr(benchmark, "_prime", lambda *args: pytest.fail("must not warm or measure"))
    result = benchmark._run_policies(_args(), {}, {}, validate, lambda: None)
    assert calls == ["fifo"]
    assert result["status"] == "validation_failed"
    assert "samples" not in result and "medians" not in result


def test_late_token_failure_discards_previously_completed_trials(fake_mlx, monkeypatch):
    candidates, prompts, _ = _setup()
    args = _args()
    validation = benchmark._validate_policy(
        FakeModel(), candidates, prompts, args, "fifo", lambda: None
    )
    original = benchmark._execute
    calls = []

    def execute(*args, **kwargs):
        sample = original(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == 8:
            sample["requests"]["B"]["generated_token_ids"][-1] ^= 1
        return sample

    monkeypatch.setattr(benchmark, "_execute", execute)
    result = benchmark._run_policies(
        args, candidates, prompts, lambda policy: copy.deepcopy(validation), lambda: None
    )
    assert len(calls) == 8
    assert result["status"] == "validation_failed"
    assert "samples" not in result and "medians" not in result and "paired_ratios" not in result


def test_existing_artifact_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "preserve.json"
    output.write_text("original")
    monkeypatch.setattr(benchmark, "run", lambda args: pytest.fail("must not run"))
    with pytest.raises(FileExistsError):
        benchmark.main(["--output", str(output)])
    assert output.read_text() == "original"


def test_failed_artifact_is_written_without_timings_and_exits_nonzero(tmp_path, monkeypatch):
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda args: {"status": "validation_failed", "validation_failure": [{"passed": False}]},
    )
    with pytest.raises(SystemExit) as failure:
        benchmark.main(["--output", str(output)])
    assert failure.value.code == 1
    assert json.loads(output.read_text())["status"] == "validation_failed"
