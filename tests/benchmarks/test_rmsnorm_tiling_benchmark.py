import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.compiler import rmsnorm_tiling as benchmark


def gate_cases(variant=None, gpu=1.11, wall=1.0):
    return [
        {
            **case,
            "variant": variant or benchmark.variants()[0],
            "status": "measured",
            "validated_before_timing": True,
            "correctness": {
                "candidate": {"passed": True},
                "frozen_register4": {"passed": True},
            },
            "summary": {
                "frozen_register4": {
                    "gpu_us": {"speedup": gpu},
                    "wall_us": {"speedup": wall},
                }
            },
        }
        for case in benchmark.shared.benchmark_cases()
    ]


def frozen_shader():
    source = "static-width four-register baseline"
    return {
        **benchmark.shared.benchmark_cases()[0],
        "variant": benchmark.variants()[0],
        "correctness": {"passed": True},
        "shader": {
            "msl_source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "argument_indices": [0, 1, 2, 3],
            "output_indices": [2],
            "is_gemm": False,
            "threadgroup_size": [256, 1, 1],
        },
    }


def compilation():
    return {
        "path": "offline",
        "precompiled": True,
        "flags": ["-O2", "-ffast-math"],
        "metal_compiler": "test-toolchain",
        "metal_standard": "compiler_default",
    }


def test_finite_candidates_include_every_declared_count_and_valid_layout():
    variants = benchmark.variants()
    assert len(variants) == 9
    assert len({variant["name"] for variant in variants}) == 9
    assert variants[0] == {
        "name": "register4_striped",
        "elements_per_thread": 4,
        "layout": "striped",
    }
    assert {variant["elements_per_thread"] for variant in variants} == {2, 4, 8, 16, 32}
    assert not any(
        variant["elements_per_thread"] == 2 and variant["layout"] == "blocked4"
        for variant in variants
    )


@pytest.mark.parametrize("variant", benchmark.variants(), ids=lambda variant: variant["name"])
def test_layout_bits_preserve_bijection_and_blocked_four_owned_values(variant):
    from metile import ThreadLayout

    layout = ThreadLayout(
        benchmark.layout_bits(variant), elements_per_thread=variant["elements_per_thread"]
    )
    indices = [
        layout.logical_index(thread, register)
        for thread in range(layout.thread_count)
        for register in range(layout.elements_per_thread)
    ]
    assert sorted(indices) == list(range(1024))
    if variant["layout"] == "blocked4":
        for thread in range(layout.thread_count):
            for register in range(0, layout.elements_per_thread, 4):
                start = layout.logical_index(thread, register)
                assert start % 4 == 0
                assert [
                    layout.logical_index(thread, register + offset) for offset in range(4)
                ] == list(range(start, start + 4))
    else:
        assert layout.bit_order == tuple(range(10))


def test_layout_selection_rejects_undeclared_candidates_and_tile_sizes():
    with pytest.raises(ValueError, match="finite candidate"):
        benchmark.layout_bits({"name": "unknown", "elements_per_thread": 4, "layout": "striped"})
    with pytest.raises(ValueError, match="1024"):
        benchmark.layout_bits(benchmark.variants()[0], block=512)


def test_train_and_validation_inputs_are_distinct_and_pairwise_deterministic():
    assert benchmark.TRAINING_SEED != benchmark.VALIDATION_SEED
    for case in benchmark.shared.benchmark_cases():
        training = benchmark.shared.case_inputs(case, benchmark.TRAINING_SEED)
        validation = benchmark.shared.case_inputs(case, benchmark.VALIDATION_SEED)
        repeated = benchmark.shared.case_inputs(case, benchmark.VALIDATION_SEED)
        assert benchmark.input_hash(*training) != benchmark.input_hash(*validation)
        assert benchmark.input_hash(*repeated) == benchmark.input_hash(*validation)
        np.testing.assert_array_equal(repeated[0], validation[0])


def test_frozen_register4_export_uses_static_width_abi():
    benchmark.validate_shader(frozen_shader(), benchmark.shared.benchmark_cases()[0])


@pytest.mark.parametrize(
    "change", ["identity", "source", "abi", "output", "geometry", "variant", "correctness", "gemm"]
)
def test_frozen_import_rejects_wrong_source_abi_launch_or_unverified_kernel(change):
    exported = frozen_shader()
    if change == "identity":
        exported["width"] = 1024
    elif change == "source":
        exported["shader"]["msl_source"] += "changed"
    elif change == "abi":
        exported["shader"]["argument_indices"] = [0, 1, 2, 3, 4]
    elif change == "output":
        exported["shader"]["output_indices"] = [0]
    elif change == "geometry":
        exported["shader"]["threadgroup_size"] = [128, 1, 1]
    elif change == "variant":
        exported["variant"] = benchmark.variants()[1]
    elif change == "gemm":
        exported["shader"]["is_gemm"] = True
    else:
        exported["correctness"]["passed"] = False
    with pytest.raises(ValueError):
        benchmark.validate_shader(exported, benchmark.shared.benchmark_cases()[0])


def test_compile_policy_requires_offline_matching_flags_and_toolchain():
    benchmark.validate_compilations({"candidate": compilation(), "frozen_register4": compilation()})


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "runtime_jit"),
        ("precompiled", False),
        ("flags", []),
        ("metal_compiler", "other"),
        ("metal_standard", "metal3.1"),
    ],
)
def test_compile_policy_rejects_mismatched_or_nonoffline_compilation(field, value):
    other = compilation()
    other[field] = value
    with pytest.raises(ValueError):
        benchmark.validate_compilations({"candidate": compilation(), "frozen_register4": other})


def test_compile_policy_requires_both_records():
    with pytest.raises(ValueError):
        benchmark.validate_compilations({"candidate": compilation()})


def test_gate_requires_aligned_throughput_not_aligned_single_row_speedup():
    cases = gate_cases()
    for case in cases:
        if case["width"] == 1024 and case["batches"] == 1:
            case["summary"]["frozen_register4"]["gpu_us"]["speedup"] = 0.9
    assert benchmark.promotion_gate(cases)["passed"]


def test_gate_reports_all_three_guard_failures_without_hiding_ragged_losses():
    cases = gate_cases()
    cases[0]["summary"]["frozen_register4"]["gpu_us"]["speedup"] = 0.96
    cases[1]["summary"]["frozen_register4"]["wall_us"]["speedup"] = 0.96
    aligned = next(case for case in cases if case["width"] == 1024 and case["batches"] == 32)
    aligned["summary"]["frozen_register4"]["gpu_us"]["speedup"] = 1.09
    gate = benchmark.promotion_gate(cases)
    assert not gate["passed"]
    assert gate["failed_ragged_gpu_cases"] == [cases[0]["name"]]
    assert gate["failed_wall_cases"] == [cases[1]["name"]]
    assert gate["failed_aligned_throughput_gpu_cases"] == [aligned["name"]]


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "unchecked", "failed", "missing_control"]
)
def test_gate_rejects_incomplete_or_incorrect_matrix(change):
    cases = gate_cases()
    if change == "missing":
        cases.pop()
    elif change == "duplicate":
        cases[-1] = deepcopy(cases[0])
    elif change == "unchecked":
        cases[0]["validated_before_timing"] = False
    elif change == "missing_control":
        del cases[0]["correctness"]["frozen_register4"]
    else:
        cases[0]["correctness"]["candidate"]["passed"] = False
    assert not benchmark.promotion_gate(cases)["passed"]


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), 0.0, -1.0])
def test_gate_rejects_nonfinite_or_nonpositive_measurements(ratio):
    cases = gate_cases(gpu=ratio)
    with pytest.raises(ValueError, match="positive finite"):
        benchmark.promotion_gate(cases)


def test_chooser_selects_one_policy_with_regression_guards_before_peak_speed():
    slower, faster = benchmark.variants()[:2]
    measurements = gate_cases(slower, gpu=1.15) + gate_cases(faster, gpu=1.5)
    measurements[-1]["summary"]["frozen_register4"]["wall_us"]["speedup"] = 0.95
    selection = benchmark.choose_policy(measurements)
    assert selection["selected"] == slower
    assert selection["heldout_results_used"] is False
    assert len(selection["scores"]) == len(benchmark.variants())


def test_chooser_uses_worst_aligned_case_before_geometric_mean():
    uniform, uneven = benchmark.variants()[:2]
    measurements = gate_cases(uniform, gpu=1.15) + gate_cases(uneven, gpu=2.0)
    measurements[-1]["summary"]["frozen_register4"]["gpu_us"]["speedup"] = 1.12
    assert benchmark.choose_policy(measurements)["selected"] == uniform


def test_chooser_rejects_incomplete_or_failed_candidate_matrix():
    cases = gate_cases()
    cases[0]["status"] = "failed"
    with pytest.raises(ValueError, match="no candidate"):
        benchmark.choose_policy(cases)


def manifest_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.shared, "_implementation_hash", lambda root: "compiler")
    baseline = {"fixture": "frozen register4"}
    cases = gate_cases()
    manifest = {
        "training_seed": benchmark.TRAINING_SEED,
        "validation_seed": benchmark.VALIDATION_SEED,
        "training_process_id": os.getpid() + 1,
        "gate_policy": deepcopy(benchmark.GATE_POLICY),
        "cases": benchmark.shared.benchmark_cases(),
        "variants": benchmark.variants(),
        "selection": benchmark.choose_policy(cases),
        "baseline_export_sha256": benchmark.json_hash(baseline),
        "compiler_implementation_sha256": "compiler",
        "benchmark_source_sha256": benchmark.benchmark_fingerprint(),
    }
    report = {
        "kind": "register_tiling_tune",
        "configuration": {"seed": benchmark.TRAINING_SEED},
        "process_id": manifest["training_process_id"],
        "gate_policy": deepcopy(benchmark.GATE_POLICY),
        "cases": cases,
        **{
            key: manifest[key]
            for key in (
                "compiler_implementation_sha256",
                "benchmark_source_sha256",
                "baseline_export_sha256",
            )
        },
    }
    path = tmp_path / "tuning.json"
    benchmark.write_json(path, report)
    manifest["tuning_report"] = str(path)
    manifest["tuning_report_sha256"] = benchmark.json_hash(report)
    return manifest, baseline


def test_manifest_accepts_unchanged_frozen_experiment_in_fresh_process(monkeypatch, tmp_path):
    manifest, baseline = manifest_fixture(monkeypatch, tmp_path)
    benchmark.validate_manifest(manifest, baseline, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "same_seed",
        "seed",
        "gate",
        "cases",
        "variants",
        "selected",
        "heldout",
        "baseline",
        "compiler",
        "driver",
        "process",
    ],
)
def test_manifest_rejects_post_selection_changes_or_training_reuse(change, monkeypatch, tmp_path):
    manifest, baseline = manifest_fixture(monkeypatch, tmp_path)
    if change == "same_seed":
        manifest["validation_seed"] = manifest["training_seed"]
    elif change == "seed":
        manifest["validation_seed"] += 1
    elif change == "gate":
        manifest["gate_policy"]["aligned_throughput_gpu_speedup_minimum"] = 1.01
    elif change == "cases":
        manifest["cases"].pop()
    elif change == "variants":
        manifest["variants"].pop()
    elif change == "selected":
        manifest["selection"]["selected"] = {"name": "unmeasured"}
    elif change == "heldout":
        manifest["selection"]["heldout_results_used"] = True
    elif change == "baseline":
        baseline["fixture"] = "changed"
    elif change == "compiler":
        manifest["compiler_implementation_sha256"] = "changed"
    elif change == "driver":
        manifest["benchmark_source_sha256"] = "changed"
    else:
        manifest["training_process_id"] = os.getpid()
    with pytest.raises(ValueError):
        benchmark.validate_manifest(manifest, baseline, tmp_path)


def test_manifest_rejects_changed_tuning_results(monkeypatch, tmp_path):
    manifest, baseline = manifest_fixture(monkeypatch, tmp_path)
    path = Path(manifest["tuning_report"])
    report = json.loads(path.read_text())
    report["cases"].pop()
    benchmark.write_json(path, report)
    with pytest.raises(ValueError, match="evidence changed"):
        benchmark.validate_manifest(manifest, baseline, tmp_path)


def test_manifest_rejects_valid_variant_not_selected_by_training_chooser(monkeypatch, tmp_path):
    manifest, baseline = manifest_fixture(monkeypatch, tmp_path)
    manifest["selection"]["selected"] = benchmark.variants()[1]
    with pytest.raises(ValueError, match="declared chooser"):
        benchmark.validate_manifest(manifest, baseline, tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "kind",
        "configuration",
        "process_id",
        "gate_policy",
        "compiler_implementation_sha256",
        "benchmark_source_sha256",
        "baseline_export_sha256",
    ],
)
def test_manifest_rejects_rehashed_foreign_tuning_experiment(field, monkeypatch, tmp_path):
    manifest, baseline = manifest_fixture(monkeypatch, tmp_path)
    path = Path(manifest["tuning_report"])
    report = json.loads(path.read_text())
    report[field] = {"seed": 0} if field == "configuration" else "changed"
    benchmark.write_json(path, report)
    manifest["tuning_report_sha256"] = benchmark.json_hash(report)
    with pytest.raises(ValueError, match="original training"):
        benchmark.validate_manifest(manifest, baseline, tmp_path)


def test_benchmark_fingerprint_includes_shared_timing_and_compilation_helpers():
    sources = benchmark.benchmark_sources()
    assert set(sources) == {"rmsnorm_tiling.py", "register_rmsnorm.py", "checkout.py"}
    assert (
        sources["register_rmsnorm.py"]
        == hashlib.sha256(Path(benchmark.shared.__file__).read_bytes()).hexdigest()
    )
    assert (
        sources["checkout.py"]
        == hashlib.sha256(Path(benchmark.checkout.__file__).read_bytes()).hexdigest()
    )
    assert benchmark.benchmark_fingerprint() == benchmark.json_hash(sources)


def test_frozen_import_binds_only_static_width_abi_without_runtime_n(monkeypatch):
    captured = {}

    class FakeBuffer:
        def __init__(self, data):
            self.data = data

    class FakeMetile:
        Buffer = FakeBuffer

    def dispatch(shader, resources, grid, device):
        captured.update(shader=shader, resources=resources, grid=grid, device=device)
        return "dispatch", "compilation"

    monkeypatch.setattr(benchmark.shared, "_dispatch_for_shader", dispatch)
    exported = frozen_shader()
    resources = (object(), object(), object())
    result = benchmark._import_frozen(exported, resources, "device", FakeMetile)
    assert result == ("dispatch", "compilation")
    assert captured["resources"][:3] == resources
    assert len(captured["resources"]) == 4
    np.testing.assert_array_equal(captured["resources"][3].data, np.array([1e-5], dtype=np.float32))
    assert captured["grid"] == (exported["batches"],)


def test_precision_label_explicitly_distinguishes_secondary_mlx_rounding():
    policy = benchmark.PRECISION_COMPARISON
    assert policy["class"] == "same_storage_precision"
    assert policy["same_weight_representation"] is True
    assert policy["relaxed_precision"] is False
    assert "secondary only" in policy["mlx_fast_caveat"]


@pytest.mark.parametrize("changed", ["driver", "compiler"])
def test_experiment_checks_sources_before_and_after_measurement(changed, monkeypatch, tmp_path):
    baseline = {
        "kind": "frozen_static_width_register4_baseline",
        "epsilon": benchmark.shared._EPSILON,
        "device": "test-device",
        "metal_compiler": "test-compiler",
        "cases": [{**frozen_shader(), **case} for case in benchmark.shared.benchmark_cases()],
    }
    baseline_path = tmp_path / "baseline.json"
    benchmark.write_json(baseline_path, baseline)
    device = SimpleNamespace(name="test-device", metal_compiler_version="test-compiler")
    monkeypatch.setattr(benchmark, "_runtime", lambda root: (None, device))

    def measure(case, variant, *arguments):
        return next(result for result in gate_cases(variant) if result["name"] == case["name"])

    monkeypatch.setattr(benchmark, "_measure", measure)
    hashes = iter(("before", "after"))
    if changed == "driver":
        monkeypatch.setattr(benchmark, "benchmark_fingerprint", lambda: next(hashes))
        monkeypatch.setattr(benchmark.shared, "_implementation_hash", lambda root: "compiler")
    else:
        monkeypatch.setattr(benchmark, "benchmark_fingerprint", lambda: "driver")
        monkeypatch.setattr(benchmark.shared, "_implementation_hash", lambda root: next(hashes))
    arguments = SimpleNamespace(
        root=tmp_path,
        baseline_json=baseline_path,
        mode="tune",
        output=tmp_path / "measurements.json",
        manifest=tmp_path / "selection.json",
    )
    with pytest.raises(RuntimeError, match="changed while measuring"):
        benchmark._experiment(arguments)
    assert not arguments.output.exists()
    assert not arguments.manifest.exists()
