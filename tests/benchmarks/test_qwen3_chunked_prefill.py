import copy
import json
import math
import sys
from types import SimpleNamespace

import pytest

from benchmarks.megakernels import qwen3_chunked_prefill as benchmark
from tests.benchmarks.test_qwen3_end_to_end import FakeCandidate, FakeModel, _generation, _logits
from tests.benchmarks.test_qwen3_end_to_end import fake_mlx as fake_mlx


class FakeChunked(FakeCandidate):
    def __init__(self):
        super().__init__()
        self.cached_tokens = 0
        self.chunks = []
        self.projections = 0
        self.events = []
        self.chunk_size = 2
        self.config = {"chunk_size": 2}
        self.parameter_count = 100
        self.prefill_weight_bytes = 64

    def reset(self):
        super().reset()
        self.cached_tokens = 0

    def forward(self, token, position, project=True):
        super().forward(token, position, project)
        self.cached_tokens = position + 1
        self.last_token = token

    def prefill(self, tokens):
        self.chunks.append(list(tokens))
        for token in tokens:
            self.forward(token, self.cached_tokens, project=False)

    def project_last_prefill(self):
        self.projections += 1
        self.output[:] = _logits(self.last_token, self.cached_tokens - 1)

    def set_chunk_size(self, size):
        self.events.append(("configure", size))
        self.chunk_size = size
        self.config["chunk_size"] = size

    def prepare(self):
        self.events.append(("prepare", self.chunk_size))
        self.reset()


def _samples(sequential=False, duration=2):
    samples = []
    for index, native_time in enumerate((1, 6, 10, 2, 9)):
        sample = {
            "trial": index,
            "order": benchmark._trial_order(index, sequential),
            "MLX": _generation(native_time),
            "chunked": _generation(duration),
        }
        if sequential:
            sample["sequential"] = _generation(duration * 3)
        samples.append(sample)
    return samples


def _phase_arguments():
    return SimpleNamespace(
        chunk_sizes=[2, 4],
        tuning_output_tokens=3,
        output_tokens=3,
        tuning_trials=5,
        trials=5,
        dtype="float32",
        cache_check_interval=2,
        sequential_baseline=False,
    )


def test_default_primary_is_actual_long_document_generation():
    arguments = benchmark._arguments([])
    assert arguments.model == "Qwen/Qwen3-0.6B"
    assert arguments.prompt_tokens == 4096
    assert arguments.output_tokens == 1024
    assert arguments.chunk_sizes == [128, 256, 512, 1024]
    assert arguments.trials == arguments.tuning_trials == 5
    assert arguments.tuning_prompt_tokens == 2048
    assert arguments.tuning_output_tokens == 8
    assert not arguments.sequential_baseline
    assert arguments.lossless_decode_weights
    assert arguments.attention_backend == "tiled"
    assert arguments.projection_backend == "simdgroup"
    assert arguments.projection_tile == [64, 64, 32]


def test_lossless_weight_packing_can_be_disabled_for_storage_matched_ablation():
    assert not benchmark._arguments(["--no-lossless-decode-weights"]).lossless_decode_weights


def test_matrix_attention_report_does_not_overwrite_the_recorded_scalar_baseline():
    baseline = benchmark._arguments([])
    matrix = benchmark._arguments(["--attention-backend", "matrix"])
    assert matrix.output != baseline.output
    assert matrix.output.name == "m5-qwen3-matrix-prefill-end-to-end.json"


@pytest.mark.parametrize("tile", [(0, 64, 32), (64, 31, 32), (-8, 64, 32)])
def test_projection_tiles_reject_invalid_shapes(tile):
    with pytest.raises(SystemExit):
        benchmark._arguments(["--projection-tile", *(str(size) for size in tile)])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--chunk-sizes", "0"],
        ["--chunk-sizes", "64", "64"],
        ["--prompt-tokens", "0"],
        ["--output-tokens", "1"],
        ["--tuning-output-tokens", "1"],
        ["--tuning-prompt-tokens", "0"],
        ["--trials", "4"],
        ["--tuning-trials", "4"],
        ["--cache-check-interval", "0"],
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._arguments(arguments)


def test_prompt_is_a_real_contiguous_document_prefix_with_hashes(tmp_path):
    first = tmp_path / "first.rst"
    second = tmp_path / "second.rst"
    first.write_text("hello")
    second.write_text("world")
    calls = []

    def encode(text, **options):
        calls.append((text, options))
        return [ord(character) for character in text]

    result = benchmark._document_prompt(
        tmp_path, SimpleNamespace(encode=encode), ["first.rst", "second.rst"], 8
    )
    assert result["token_ids"] == [ord(character) for character in "hello\n\nw"]
    assert result["source_token_count"] == 12
    assert result["actual_prompt_tokens"] == 8
    assert result["sources_sha256"] == {
        "first.rst": benchmark._sha256(first),
        "second.rst": benchmark._sha256(second),
    }
    assert calls == [("hello\n\nworld", {"add_special_tokens": False})]
    assert not result["chat_template"]


def test_prompt_rejects_short_documents_without_padding_or_repeating(tmp_path):
    (tmp_path / "doc.rst").write_text("short")
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [4, 5])
    with pytest.raises(ValueError, match="2 tokens but 3"):
        benchmark._document_prompt(tmp_path, tokenizer, ["doc.rst"], 3)


@pytest.mark.parametrize("sources", [["doc.rst", "doc.rst"], ["../outside.rst"]])
def test_prompt_requires_distinct_checkout_local_sources(tmp_path, sources):
    with pytest.raises(ValueError, match="distinct files inside"):
        benchmark._document_prompt(tmp_path, object(), sources, 3)


@pytest.mark.parametrize(
    "held_out",
    [
        {"sources_sha256": {"tune.rst": "different"}, "token_ids": [2]},
        {"sources_sha256": {"held.rst": "abc"}, "token_ids": [2]},
        {"sources_sha256": {"held.rst": "def"}, "token_ids": [1]},
    ],
)
def test_tuning_and_held_out_must_not_share_files_content_or_token_sequence(held_out):
    tuning = {"sources_sha256": {"tune.rst": "abc"}, "token_ids": [1]}
    with pytest.raises(ValueError):
        benchmark._require_disjoint_prompts(tuning, held_out)


def test_disjoint_prompt_sources_are_accepted():
    benchmark._require_disjoint_prompts(
        {"sources_sha256": {"tune.rst": "abc"}, "token_ids": [1]},
        {"sources_sha256": {"held.rst": "def"}, "token_ids": [2]},
    )


def test_chunked_generation_prefills_all_tokens_then_projects_only_final_row():
    candidate = FakeChunked()
    result = benchmark._chunked_generation(candidate, [3, 5, 2], 3)
    assert candidate.resets == 1
    assert candidate.chunks == [[3, 5, 2]]
    assert candidate.projections == 1
    assert candidate.calls == [
        (3, 0, False),
        (5, 1, False),
        (2, 2, False),
        (5, 3, True),
        (9, 4, True),
    ]
    assert candidate.greedy_calls == 3
    assert result["generated_token_ids"] == [5, 9, 1]
    assert result["actual_output_tokens"] == 3


def test_validation_uses_native_batched_prefix_and_checks_each_chunk_boundary(fake_mlx):
    model = FakeModel()
    candidate = FakeChunked()
    result = benchmark._validate_workload(model, candidate, [3, 5, 2], 5, "float32", 2, 3)
    assert model.calls[:2] == [[3, 5], [2]]
    assert candidate.chunks == [[3, 5], [2], [3, 5, 2]]
    assert candidate.resets == 2
    assert candidate.projections == 1
    prefix_checks = [check for check in result["checks"] if check["stage"] == "prefill"]
    assert [check["valid_tokens"] for check in prefix_checks] == [2, 3]
    assert all(check["cache"]["full_prefix_checked"] for check in prefix_checks)
    generated = [check for check in result["checks"] if check["stage"] == "generation"]
    assert [check["cache"]["full_prefix_checked"] for check in generated] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert [check["position"] for check in generated] == [2, 3, 4, 5, 6]
    assert generated[1]["cache"]["positions_checked"] == [3]
    assert all(check["cache"]["layers_checked"] == 2 for check in generated)
    assert len(result["generated_token_ids"]) == 5
    assert result["whole_prompt_replay"] == {
        "passed": True,
        "prefill_calls": 1,
        "actual_prompt_tokens": 3,
        "generation_check_index": 2,
        "shares_first_generation_check": True,
    }
    assert result["checks"][result["whole_prompt_replay"]["generation_check_index"]] is generated[0]
    assert generated[0]["prefill_mode"] == "single_whole_prompt_call"
    assert result["validation_counts"] == {
        "chunk_boundary_cache_checks": 2,
        "whole_prompt_prefill_calls": 1,
        "generation_full_logit_checks": 5,
        "generation_cache_checks": 5,
        "generation_gpu_greedy_checks": 5,
        "post_first_token_decode_forwards": 4,
    }


@pytest.mark.parametrize("corruption", ["key", "value", "logits", "greedy"])
def test_whole_prompt_only_corruption_fails_after_successful_chunk_boundaries(fake_mlx, corruption):
    candidate = FakeChunked()
    prefill = candidate.prefill
    project = candidate.project_last_prefill
    greedy = candidate.greedy
    whole_prompt = False

    def fill(tokens):
        nonlocal whole_prompt
        whole_prompt = len(tokens) == 3
        prefill(tokens)
        if whole_prompt and corruption in {"key", "value"}:
            candidate.storage[int(corruption == "value"), 1, :, 0] += 1

    def project_last():
        project()
        if whole_prompt and corruption == "logits":
            candidate.output[0] += 0.1

    def select():
        selected = greedy()
        return (selected + 1) % 13 if whole_prompt and corruption == "greedy" else selected

    candidate.prefill = fill
    candidate.project_last_prefill = project_last
    candidate.greedy = select
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 3, "float32", 2, 3)
    checks = error.value.checks
    assert [check["stage"] for check in checks] == ["prefill", "prefill", "generation"]
    assert all(check["passed"] for check in checks[:-1])
    assert not checks[-1]["passed"]
    assert checks[-1]["prefill_mode"] == "single_whole_prompt_call"
    assert candidate.chunks == [[3, 5], [2], [3, 5, 2]]


def test_decode_validation_continues_from_whole_prompt_replay_not_boundary_state(fake_mlx):
    candidate = FakeChunked()
    prefill = candidate.prefill
    forward = candidate.forward
    whole_prompt = False

    def fill(tokens):
        nonlocal whole_prompt
        whole_prompt = len(tokens) == 3
        prefill(tokens)

    def decode(token, position, project=True):
        forward(token, position, project)
        if whole_prompt and position == 3:
            candidate.storage[0, 1, :, position] += 1

    candidate.prefill = fill
    candidate.forward = decode
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 3, "float32", 2, 3)
    checks = error.value.checks
    assert all(check["passed"] for check in checks[:-1])
    assert checks[-2]["prefill_mode"] == "single_whole_prompt_call"
    assert checks[-1]["position"] == 3
    assert not checks[-1]["cache"]["passed"]
    assert not checks[-1]["cache"]["full_prefix_checked"]


def test_sequential_validation_does_not_require_chunked_api(fake_mlx):
    candidate = FakeCandidate()
    result = benchmark._validate_workload(
        FakeModel(), candidate, [3, 5, 2], 3, "float32", 2, 3, sequential=True
    )
    assert candidate.calls == [
        (3, 0, False),
        (5, 1, False),
        (2, 2, True),
        (5, 3, True),
        (9, 4, True),
    ]
    assert result["generated_token_ids"] == [5, 9, 1]
    assert "whole_prompt_replay" not in result
    assert result["validation_counts"]["whole_prompt_prefill_calls"] == 0
    assert candidate.resets == 1


@pytest.mark.parametrize("kind", [0, 1])
def test_new_decode_slot_is_checked_between_full_prefix_milestones(fake_mlx, kind):
    candidate = FakeChunked()
    forward = candidate.forward

    def corrupt(token, position, project=True):
        forward(token, position, project)
        if position == 3:
            candidate.storage[kind, 1, :, position] += 1

    candidate.forward = corrupt
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 5, "float32", 2, 3)
    assert error.value.checks[-1]["position"] == 3
    assert not error.value.checks[-1]["cache"]["full_prefix_checked"]
    assert not error.value.checks[-1]["cache"]["passed"]


def test_old_cache_corruption_is_caught_at_next_full_prefix_milestone(fake_mlx):
    candidate = FakeChunked()
    forward = candidate.forward

    def corrupt(token, position, project=True):
        forward(token, position, project)
        if position == 3:
            candidate.storage[1, 1, :, 0] += 1

    candidate.forward = corrupt
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 5, "float32", 2, 3)
    assert error.value.checks[-2]["position"] == 3
    assert error.value.checks[-2]["passed"]
    assert error.value.checks[-1]["position"] == 4
    assert error.value.checks[-1]["cache"]["full_prefix_checked"]
    assert not error.value.checks[-1]["cache"]["passed"]


def test_bad_logits_fail_even_when_the_greedy_token_still_matches(fake_mlx):
    candidate = FakeChunked()
    project = candidate.project_last_prefill

    def corrupt():
        project()
        candidate.output[0] += 0.1

    candidate.project_last_prefill = corrupt
    with pytest.raises(benchmark.GenerationValidationFailure) as error:
        benchmark._validate_workload(FakeModel(), candidate, [3, 5, 2], 3, "float32", 2, 3)
    check = error.value.checks[-1]
    assert check["logits"]["greedy_token_equal"]
    assert not check["logits"]["passed"]


def test_pair_order_alternates_while_optional_third_arm_rotates():
    orders = [benchmark._trial_order(trial, True) for trial in range(6)]
    assert [order.index("sequential") for order in orders] == [0, 1, 2, 0, 1, 2]
    for index, order in enumerate(orders):
        assert [label for label in order if label != "sequential"] == benchmark._trial_order(index)


def test_summary_uses_within_trial_ratios_and_n_minus_one_decode_throughput():
    samples = _samples(sequential=True)
    result = benchmark._summarize(samples)
    assert result["medians"]["MLX"]["total_wall_seconds"] == 6
    assert result["medians"]["chunked"]["decode_tokens_per_second"] == pytest.approx(2 / 1.5)
    assert result["paired_speedups"]["chunked_over_sequential"]["total_wall_seconds"] == 3
    samples[0]["chunked"] = _generation(1)
    samples[1]["chunked"] = _generation(3)
    samples[2]["chunked"] = _generation(10)
    changed = benchmark._summarize(samples)
    assert changed["paired_speedups"]["chunked_over_mlx"]["total_wall_seconds"] == 1
    assert (
        changed["medians"]["MLX"]["total_wall_seconds"]
        / changed["medians"]["chunked"]["total_wall_seconds"]
        == 3
    )


@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan, True])
def test_invalid_durations_are_rejected(duration):
    samples = _samples()
    samples[0]["chunked"]["decode_wall_seconds"] = duration
    with pytest.raises(ValueError, match="positive and finite"):
        benchmark._summarize(samples)


def test_inconsistent_totals_short_trials_and_bad_order_are_rejected():
    samples = _samples()
    with pytest.raises(ValueError, match="five"):
        benchmark._summarize(samples[:4])
    samples[0]["order"].reverse()
    with pytest.raises(ValueError, match="alternate"):
        benchmark._summarize(samples)
    samples[0]["order"].reverse()
    samples[0]["chunked"]["total_wall_seconds"] += 1
    with pytest.raises(ValueError, match="TTFT plus decode"):
        benchmark._summarize(samples)


def test_wrong_timed_token_sequence_is_rejected():
    samples = _samples()
    samples[2]["chunked"]["generated_token_ids"][-1] += 1
    with pytest.raises(benchmark.GenerationValidationFailure):
        benchmark._summarize(samples)


def test_chunk_selection_uses_tuning_ttft_not_decode_or_native_ratio():
    tuning = [
        {
            "status": "ok",
            "chunk_size": 64,
            "medians": {"chunked": {"time_to_first_token_seconds": 2, "total_wall_seconds": 3}},
        },
        {
            "status": "ok",
            "chunk_size": 128,
            "medians": {"chunked": {"time_to_first_token_seconds": 1, "total_wall_seconds": 99}},
        },
        {
            "status": "ok",
            "chunk_size": 256,
            "medians": {"chunked": {"time_to_first_token_seconds": 1, "total_wall_seconds": 1}},
        },
    ]
    assert benchmark._select_chunk(tuning) == 128
    tuning[-1]["status"] = "validation_failed"
    with pytest.raises(ValueError, match="complete validated"):
        benchmark._select_chunk(tuning)


def test_phases_reuse_one_allocation_and_never_tune_on_held_out(monkeypatch):
    events = []
    allocations = []
    candidate = FakeChunked()

    def factory(capacity, chunk_size):
        allocations.append((capacity, chunk_size))
        return candidate

    def validate(model, selected, prompt, outputs, dtype, chunk_size, interval):
        events.append(("validate", len(prompt), chunk_size))
        return {"passed": True, "generated_token_ids": [4, 5, 6]}

    def measure(model, selected, prompt, outputs, trials, expected, sequential=None):
        events.append(("measure", len(prompt), selected.chunk_size))
        samples = _samples(duration=6 / selected.chunk_size)
        return {"samples": samples, **benchmark._summarize(samples)}

    monkeypatch.setattr(benchmark, "_validate_workload", validate)
    monkeypatch.setattr(benchmark, "_measure", measure)
    result = benchmark._run_phases(
        _phase_arguments(),
        object(),
        factory,
        lambda capacity: pytest.fail("optional baseline disabled"),
        [3, 4, 5],
        [6] * 7,
    )
    assert allocations == [(9, 2)]
    assert events == [
        ("validate", 3, 2),
        ("validate", 3, 4),
        ("measure", 3, 2),
        ("measure", 3, 4),
        ("validate", 7, 4),
        ("measure", 7, 4),
    ]
    assert result["selected_chunk_size"] == 4
    assert len(result["tuning"]) == 2
    assert len(result["held_out"]) == 1
    assert result["held_out"][0]["geometry"]["chunk_size"] == 4


def test_late_held_out_failure_removes_even_previously_collected_tuning_measurements(monkeypatch):
    def validate(model, candidate, prompt, outputs, dtype, chunk_size, interval):
        if len(prompt) == 7:
            raise benchmark.GenerationValidationFailure([{"error": "held-out cache mismatch"}])
        return {"passed": True, "generated_token_ids": [4, 5, 6]}

    monkeypatch.setattr(benchmark, "_validate_workload", validate)
    monkeypatch.setattr(
        benchmark,
        "_measure",
        lambda *args, **kwargs: {"samples": _samples(), **benchmark._summarize(_samples())},
    )
    result = benchmark._run_phases(
        _phase_arguments(), object(), lambda *args: FakeChunked(), object(), [3, 4, 5], [6] * 7
    )
    assert result["status"] == "validation_failed"
    assert "selected_chunk_size" not in result
    assert result["validation_failure"] == [{"error": "held-out cache mismatch"}]
    assert all(entry["status"] == "validated" for entry in result["tuning"])
    assert all(
        "samples" not in entry and "medians" not in entry and "paired_speedups" not in entry
        for entry in result["tuning"]
    )


def test_measurement_checks_warmups_and_alternates_fresh_three_arm_trials(monkeypatch, fake_mlx):
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
    monkeypatch.setattr(benchmark, "_chunked_generation", lambda *args: operation("chunked"))
    monkeypatch.setattr(benchmark, "_candidate_generation", lambda *args: operation("sequential"))
    result = benchmark._measure(object(), object(), [3], 3, 5, [4, 5, 6], object())
    labels = [event for event in events if event != "sync"]
    expected = ["MLX", "chunked", "sequential"] * 2
    for trial in range(5):
        expected.extend(benchmark._trial_order(trial, True))
    assert labels == expected
    assert events.count("sync") == fake_mlx.count("synchronize") == 21
    assert len(result["samples"]) == 5


def test_summary_does_not_modify_raw_samples():
    samples = _samples(True)
    before = copy.deepcopy(samples)
    benchmark._summarize(samples)
    assert samples == before


def test_main_writes_structured_failure_without_timings(monkeypatch, tmp_path):
    report = {
        "status": "validation_failed",
        "tuning": [],
        "held_out": [],
        "validation_failure": [{"error": "mismatch"}],
    }
    monkeypatch.setattr(benchmark, "run", lambda arguments: report)
    output = tmp_path / "failure.json"
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(output)])
    assert error.value.code == 1
    assert json.loads(output.read_text()) == report


def test_main_preserves_existing_evidence_before_running(monkeypatch, tmp_path):
    output = tmp_path / "measured.json"
    output.write_text("recorded evidence\n")
    monkeypatch.setattr(benchmark, "run", lambda arguments: pytest.fail("must not run"))
    with pytest.raises(FileExistsError, match="choose a new --output"):
        benchmark.main(["--output", str(output)])
    assert output.read_text() == "recorded evidence\n"
