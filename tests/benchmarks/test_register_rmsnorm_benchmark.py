import hashlib
import itertools
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.compiler import register_rmsnorm as benchmark


def _gate_cases(gpu=1.11, wall=1.0):
    return [
        {
            **case,
            "validated_before_timing": True,
            "correctness": {"candidate": {"passed": True}, "frozen_baseline": {"passed": True}},
            "summary": {
                name: {"gpu_us": {"speedup": gpu}, "wall_us": {"speedup": wall}}
                for name in ("frozen_baseline", "current_control")
            },
        }
        for case in benchmark.benchmark_cases()
    ]


def _export():
    case = benchmark.benchmark_cases()[0]
    source = "frozen Metal shader source"
    return {
        **case,
        "input_sha256": "inputs",
        "correctness": {"passed": True},
        "shader": {
            "msl_source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "argument_indices": [0, 1, 2, 3, 4],
            "output_indices": [2],
            "is_gemm": False,
            "threadgroup_size": [256, 1, 1],
        },
    }


def test_predeclared_matrix_covers_aligned_ragged_throughput_and_latency_cases():
    cases = benchmark.benchmark_cases()
    assert len(cases) == 12
    assert len({case["name"] for case in cases}) == 12
    assert {case["width"] for case in cases} == {1009, 1024}
    assert {case["batches"] for case in cases} == {1, 32, 256}
    assert {case["dtype"] for case in cases} == {"float16", "float32"}


def test_case_inputs_are_repeatable_independent_of_case_order():
    case = benchmark.benchmark_cases()[0]
    first = benchmark.case_inputs(case, 619)
    second = benchmark.case_inputs(case, 619)
    different = benchmark.case_inputs(case, 620)
    for original, repeated, changed in zip(first, second, different, strict=True):
        np.testing.assert_array_equal(original, repeated)
        assert not np.array_equal(original, changed)
    assert first[0].shape == (case["batches"], case["width"])
    assert first[0].dtype == np.dtype(case["dtype"])


def test_float16_mlx_fast_rounding_reference_is_not_mislabeled_as_matched_policy():
    source, weight = benchmark.case_inputs(benchmark.benchmark_cases()[0], 619)

    matched, fast = benchmark.references(source, weight)

    assert matched.dtype == fast.dtype == np.float16
    assert np.count_nonzero(matched != fast) > 0
    values = source.astype(np.float32)
    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True) + np.float32(1e-5)
    )
    expected = (normalized * weight.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(matched, expected, rtol=1e-3, atol=1e-3)


def test_float32_references_have_the_same_rounding_boundaries():
    source, weight = benchmark.case_inputs(benchmark.benchmark_cases()[-1], 619)

    matched, fast = benchmark.references(source, weight)

    np.testing.assert_array_equal(matched, fast)


@pytest.mark.parametrize("change", ["shape", "dtype", "nan", "wrong_values"])
def test_correctness_checker_rejects_invalid_outputs(change):
    expected = np.ones((1, 1009), dtype=np.float16)
    if change == "shape":
        actual = expected.ravel()
    elif change == "dtype":
        actual = expected.astype(np.float32)
    elif change == "nan":
        actual = np.full_like(expected, np.nan)
    else:
        actual = expected * 2

    with pytest.raises(AssertionError):
        benchmark.assert_correct(actual, expected, "candidate")


def test_frozen_shader_import_checks_case_bytes_source_and_abi():
    exported = _export()

    benchmark.validate_baseline_export(exported, benchmark.benchmark_cases()[0], "inputs")


@pytest.mark.parametrize(
    "change", ["case", "inputs", "source", "binding", "output", "geometry", "gemm", "correctness"]
)
def test_frozen_shader_import_rejects_incompatible_or_unchecked_exports(change):
    exported = _export()
    if change == "case":
        exported["width"] = 1024
    elif change == "inputs":
        exported["input_sha256"] = "other"
    elif change == "source":
        exported["shader"]["msl_source"] += "changed"
    elif change == "binding":
        exported["shader"]["argument_indices"] = [0, 1, 2, 4, 3]
    elif change == "output":
        exported["shader"]["output_indices"] = [1]
    elif change == "geometry":
        exported["shader"]["threadgroup_size"] = [128, 1, 1]
    elif change == "gemm":
        exported["shader"]["is_gemm"] = True
    else:
        exported["correctness"]["passed"] = False

    with pytest.raises(ValueError):
        benchmark.validate_baseline_export(exported, benchmark.benchmark_cases()[0], "inputs")


def test_pair_timing_alternates_order_and_measures_the_same_gpu_dispatches(monkeypatch):
    functions = (object(), object())
    sequence = []
    ticks = itertools.count()
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    def measure(function, synchronize, gpu_elapsed):
        sequence.append(functions.index(function))
        scale = 1 if function is functions[0] else 2
        return {"wall_us": 10 * scale, "gpu_us": scale}

    monkeypatch.setattr(benchmark, "_timed_call", measure)

    result = benchmark.paired_measure(*functions, lambda: None, 0, 0.01)

    assert sequence == [0, 1, 1, 0] * 10
    assert result["pairs"] == 20
    assert result["gpu_us"]["speedup"] == result["wall_us"]["speedup"] == 2


def test_round_summary_uses_pair_ratios_not_unrelated_baseline_timings():
    rounds = [
        {
            "frozen_baseline": {
                "gpu_us": {"candidate": 1, "control": 2},
                "wall_us": {"candidate": 10, "control": 10},
            }
        },
        {
            "frozen_baseline": {
                "gpu_us": {"candidate": 4, "control": 2},
                "wall_us": {"candidate": 10, "control": 10},
            }
        },
    ]

    summary = benchmark.summarize_pairs(rounds)

    assert summary["frozen_baseline"]["gpu_us"]["speedup"] == 1
    assert summary["frozen_baseline"]["gpu_us"]["round_speedups"] == [2, 0.5]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_durations_cannot_become_a_speedup(duration):
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.summarize_pairs(
            [{"control": {"gpu_us": {"candidate": duration, "control": duration}}}]
        )


def test_promotion_gate_passes_only_with_both_control_comparisons():
    gate = benchmark.promotion_gate(_gate_cases())

    assert gate["passed"]
    assert gate["gpu_speedup_minimum"] == 1.10
    assert gate["wall_latency_ratio_maximum"] == 1.03
    assert set(gate["comparisons"]) == {"frozen_baseline", "current_control"}


@pytest.mark.parametrize("comparator", ["frozen_baseline", "current_control"])
def test_one_underperforming_throughput_case_fails_promotion(comparator):
    cases = _gate_cases()
    case = next(case for case in cases if case["batches"] == 32)
    case["summary"][comparator]["gpu_us"]["speedup"] = 1.09

    gate = benchmark.promotion_gate(cases)

    assert not gate["passed"]
    assert gate["comparisons"][comparator]["failed_gpu_cases"] == [case["name"]]


def test_batch_one_does_not_need_throughput_win_but_still_has_wall_guard():
    cases = _gate_cases()
    case = next(case for case in cases if case["batches"] == 1)
    case["summary"]["frozen_baseline"]["gpu_us"]["speedup"] = 0.99
    assert benchmark.promotion_gate(cases)["passed"]
    case["summary"]["frozen_baseline"]["wall_us"]["speedup"] = 1 / 1.031

    gate = benchmark.promotion_gate(cases)

    assert not gate["passed"]
    assert gate["comparisons"]["frozen_baseline"]["failed_wall_cases"] == [case["name"]]


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_case", "incorrect", "unchecked"])
def test_gate_cannot_pass_by_dropping_or_mislabeling_difficult_cases(change):
    cases = deepcopy(_gate_cases())
    if change == "missing":
        cases.pop()
    elif change == "duplicate":
        cases[-1] = cases[0]
    elif change == "wrong_case":
        cases[1]["batches"] = 1
        with pytest.raises(ValueError, match="case identities"):
            benchmark.promotion_gate(cases)
        return
    elif change == "incorrect":
        cases[1]["correctness"]["candidate"]["passed"] = False
    else:
        cases[1]["validated_before_timing"] = False

    assert not benchmark.promotion_gate(cases)["passed"]


class _CompilerDevice:
    def __init__(self, available, precompiled=True):
        self.has_metal_compiler = available
        self.precompiled = precompiled
        self.metal_compiler_version = "test Metal toolchain"
        self.calls = []

    def compile_msl(self, source, name):
        self.calls.append(("jit", source, name))
        return "jit_pipeline"

    def compile_msl_precompiled(self, source, name):
        self.calls.append(("offline", source, name))
        return "offline_pipeline", self.precompiled


@pytest.mark.parametrize(
    ("available", "precompiled", "called", "reported"),
    [
        (True, True, "offline", "offline"),
        (True, False, "offline", "runtime_jit"),
        (False, False, "jit", "runtime_jit"),
    ],
)
def test_all_shader_imports_use_the_frontend_compile_path_and_record_actual_fallback(
    available, precompiled, called, reported
):
    device = _CompilerDevice(available, precompiled)
    shader = {"msl_source": "shader", "function_name": "rmsnorm"}

    _, metadata = benchmark._compile_shader(device, shader)

    assert device.calls == [(called, "shader", "rmsnorm")]
    assert metadata["path"] == reported
    assert metadata["precompiled"] is precompiled
    assert metadata["flags"] == (["-O2", "-ffast-math"] if precompiled else [])
    assert metadata["metal_compiler"] == "test Metal toolchain"


def _compilations():
    _, metadata = benchmark._compile_shader(
        _CompilerDevice(True), {"msl_source": "shader", "function_name": "norm"}
    )
    return {
        name: deepcopy(metadata) for name in ("candidate", "current_control", "frozen_baseline")
    }


def test_identical_compile_paths_are_a_required_benchmark_invariant():
    benchmark._validate_compile_paths(_compilations())


@pytest.mark.parametrize(
    "field", ["path", "precompiled", "metal_compiler", "flags", "metal_standard", "missing"]
)
def test_frozen_baseline_cannot_use_a_different_shader_compilation_policy(field):
    metadata = _compilations()
    if field == "missing":
        metadata.pop("frozen_baseline")
    elif field == "flags":
        metadata["frozen_baseline"][field] = ["-O0"]
    elif field == "precompiled":
        metadata["frozen_baseline"][field] = False
    else:
        metadata["frozen_baseline"][field] = "different"

    with pytest.raises(RuntimeError):
        benchmark._validate_compile_paths(metadata)
