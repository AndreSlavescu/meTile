"""Frozen register4 baseline, finite RMSNorm tuning, and fresh-process heldout validation."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.common import checkout
from benchmarks.compiler import register_rmsnorm as shared

TRAINING_SEED = 1741
VALIDATION_SEED = 9473
GATE_POLICY = {
    "aligned_throughput_gpu_speedup_minimum": 1.10,
    "aligned_throughput_batches": [32, 256],
    "every_case_wall_latency_ratio_maximum": 1.03,
    "every_ragged_case_gpu_latency_ratio_maximum": 1.03,
    "reference": "frozen prechange static-width register4 striped kernel",
    "promotion": "opt-in unless every heldout guard passes; no automatic default changes",
}
PRECISION_COMPARISON = {
    "class": "same_storage_precision",
    "same_weight_representation": True,
    "bitwise_exact": False,
    "primary_policy": "identical storage dtype, FP32 sum/normalization/weight multiply, final output cast",
    "primary_mlx_comparator": "mx.compile explicit full-FP32 graph then final output cast",
    "mlx_fast_caveat": "secondary only: FP16 normalization rounds to storage before weight multiply; checked against its own rounding reference",
    "mlx_fast_source": shared._MLX_SOURCE,
    "relaxed_precision": False,
}


def variants():
    return [
        {"name": f"register{count}_{layout}", "elements_per_thread": count, "layout": layout}
        for count in (4, 8, 16, 32)
        for layout in ("striped", "blocked4")
    ] + [{"name": "register2_striped", "elements_per_thread": 2, "layout": "striped"}]


def layout_bits(variant, block=1024):
    if variant not in variants():
        raise ValueError("variant must belong to the predeclared finite candidate set")
    bits = block.bit_length() - 1
    thread_bits = bits - (variant["elements_per_thread"].bit_length() - 1)
    if block != 1024:
        raise ValueError("the predeclared experiment uses a 1024-element row tile")
    if variant["layout"] == "striped":
        return tuple(range(bits))
    return (thread_bits, thread_bits + 1, *range(thread_bits), *range(thread_bits + 2, bits))


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def benchmark_sources():
    return checkout.benchmark_sources(__file__, shared.__file__)


def benchmark_fingerprint():
    return json_hash(benchmark_sources())


def input_hash(source, weight):
    return hashlib.sha256(source.tobytes() + weight.tobytes()).hexdigest()


def validate_shader(exported, case):
    if any(exported.get(name) != value for name, value in case.items()):
        raise ValueError("frozen shader identity differs from the requested case")
    shader = exported["shader"]
    if hashlib.sha256(shader["msl_source"].encode()).hexdigest() != shader["source_sha256"]:
        raise ValueError("frozen shader source fingerprint does not match")
    if shader["argument_indices"] != [0, 1, 2, 3] or shader["output_indices"] != [2]:
        raise ValueError("frozen shader must use static-N X,W,Out,eps ABI")
    if shader["is_gemm"] or shader["threadgroup_size"] != [256, 1, 1]:
        raise ValueError("frozen shader must use the prior register4 256-thread launch")
    if exported.get("variant") != variants()[0]:
        raise ValueError("frozen shader must be the prior striped register4 variant")
    if not exported.get("correctness", {}).get("passed"):
        raise ValueError("frozen shader must pass export correctness")


def validate_compilations(compilations):
    if set(compilations) != {"candidate", "frozen_register4"}:
        raise ValueError("both candidate and frozen register4 compilation records are required")
    identities = set()
    for record in compilations.values():
        if record["path"] != "offline" or record["precompiled"] is not True:
            raise ValueError("this controlled experiment requires offline Metal compilation")
        if record["flags"] != ["-O2", "-ffast-math"]:
            raise ValueError("the frozen experiment requires -O2 -ffast-math")
        identities.add((record["metal_compiler"], record["metal_standard"]))
    if len(identities) != 1:
        raise ValueError("all shader comparisons must use identical compiler and language policy")


def _prepare(case, variant, resources, device, metile):
    rmsnorm_register = checkout.load_kernel(
        Path(metile.__file__).resolve().parent.parent, "rmsnorm"
    ).rmsnorm_register

    layout = metile.ThreadLayout(
        layout_bits(variant), elements_per_thread=variant["elements_per_thread"]
    )
    launcher = rmsnorm_register[(case["batches"],)]
    prepared = launcher.prepare(
        *resources,
        shared._EPSILON,
        N=case["width"],
        BLOCK=1024,
        LAYOUT=layout,
        RELAXED_PRECISION=False,
    )
    shader = shared._kernel_export(launcher)
    dispatch, compilation = shared._dispatch_for_shader(
        shader,
        launcher._last_resources,
        launcher.grid,
        device,
        prepared.execution_report,
        launcher._last_metal_buffers,
    )
    return dispatch, shader, compilation


def _import_frozen(exported, resources, device, metile):
    import numpy as np

    return shared._dispatch_for_shader(
        exported["shader"],
        (*resources, metile.Buffer(data=np.array([shared._EPSILON], dtype=np.float32))),
        (exported["batches"],),
        device,
    )


def _runtime(root):
    metile = checkout.load_compiler(root)
    from metile.runtime.metal_device import MetalDevice

    checkout.load_kernel(root)
    device = MetalDevice.get()
    if not device.has_metal_compiler:
        raise RuntimeError("this experiment requires the offline Metal compiler")
    return metile, device


def export_baseline(arguments):
    import numpy as np

    root = arguments.root.resolve()
    driver_fingerprint = benchmark_fingerprint()
    metile, device = _runtime(root)
    fingerprint = shared._implementation_hash(root)
    results = []
    for case in shared.benchmark_cases():
        source, weight = shared.case_inputs(case, TRAINING_SEED)
        expected, _ = shared.references(source, weight)
        output = metile.Buffer(data=np.full_like(source, np.nan))
        resources = (metile.Buffer(data=source), metile.Buffer(data=weight), output)
        dispatch, shader, compilation = _prepare(case, variants()[0], resources, device, metile)
        validate_compilations({"candidate": compilation, "frozen_register4": compilation})
        dispatch()
        device.sync()
        checked = shared.assert_correct(output.numpy(), expected, "prior_register4")
        pair = shared.paired_measure(
            dispatch,
            dispatch,
            device.sync,
            arguments.warmup_ms,
            arguments.rep_ms,
            device.gpu_elapsed,
        )
        results.append(
            {
                **case,
                "variant": variants()[0],
                "input_sha256": input_hash(source, weight),
                "correctness": checked,
                "shader": shader,
                "compilation": compilation,
                "initial_same_kernel_pair": pair,
            }
        )
        print(
            f"prior register4 {case['name']}: GPU {pair['gpu_us']['candidate']:.3f} us, wall {pair['wall_us']['candidate']:.3f} us",
            flush=True,
        )
    if shared._implementation_hash(root) != fingerprint:
        raise RuntimeError("baseline compiler changed during export")
    if benchmark_fingerprint() != driver_fingerprint:
        raise RuntimeError("benchmark sources changed during export")
    write_json(
        arguments.output,
        {
            "kind": "frozen_static_width_register4_baseline",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "root": str(root),
            "compiler_implementation_sha256": fingerprint,
            "benchmark_source_sha256": driver_fingerprint,
            "benchmark_sources": benchmark_sources(),
            "device": device.name,
            "metal_compiler": device.metal_compiler_version,
            "seed": TRAINING_SEED,
            "epsilon": shared._EPSILON,
            "gate_policy": GATE_POLICY,
            "precision_comparison": PRECISION_COMPARISON,
            "cases": results,
        },
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def promotion_gate(results):
    expected = {case["name"]: case for case in shared.benchmark_cases()}
    names = [case["name"] for case in results]
    if len(names) != len(set(names)) or set(names) != set(expected):
        return {"passed": False, "reason": "the complete unique 12-case heldout matrix is required"}
    failed_gpu, failed_wall, failed_ragged = [], [], []
    for case in results:
        if any(case.get(key) != value for key, value in expected[case["name"]].items()):
            raise ValueError("case identity differs from the predeclared manifest")
        checks = case.get("correctness", {})
        if (
            not case.get("validated_before_timing")
            or not {"candidate", "frozen_register4"}.issubset(checks)
            or not all(check.get("passed") for check in checks.values())
        ):
            return {
                "passed": False,
                "reason": "all comparators must pass correctness before timing",
            }
        pair = case["summary"]["frozen_register4"]
        gpu, wall = pair["gpu_us"]["speedup"], pair["wall_us"]["speedup"]
        if any(not math.isfinite(value) or value <= 0 for value in (gpu, wall)):
            raise ValueError("promotion gates require positive finite ratios")
        if case["width"] == 1024 and case["batches"] in (32, 256) and gpu < 1.10:
            failed_gpu.append(case["name"])
        if 1.0 / wall > 1.03:
            failed_wall.append(case["name"])
        if case["width"] == 1009 and 1.0 / gpu > 1.03:
            failed_ragged.append(case["name"])
    return {
        "passed": not (failed_gpu or failed_wall or failed_ragged),
        "policy": GATE_POLICY,
        "failed_aligned_throughput_gpu_cases": failed_gpu,
        "failed_wall_cases": failed_wall,
        "failed_ragged_gpu_cases": failed_ragged,
    }


def choose_policy(measurements):
    expected = {case["name"] for case in shared.benchmark_cases()}
    scores = []
    for variant in variants():
        cases = [entry for entry in measurements if entry["variant"] == variant]
        if (
            len(cases) != len(expected)
            or {case["name"] for case in cases} != expected
            or any(case.get("status") != "measured" for case in cases)
        ):
            scores.append(
                {"variant": variant, "eligible": False, "reason": "incomplete/failed tuning matrix"}
            )
            continue
        gate = promotion_gate(cases)
        if "failed_wall_cases" not in gate:
            scores.append({"variant": variant, "eligible": False, "reason": gate["reason"]})
            continue
        aligned = [
            case["summary"]["frozen_register4"]["gpu_us"]["speedup"]
            for case in cases
            if case["width"] == 1024 and case["batches"] in (32, 256)
        ]
        scores.append(
            {
                "variant": variant,
                "eligible": True,
                "regression_guard_failures": len(gate["failed_wall_cases"])
                + len(gate["failed_ragged_gpu_cases"]),
                "minimum_aligned_throughput_gpu_speedup": min(aligned),
                "geomean_aligned_throughput_gpu_speedup": statistics.geometric_mean(aligned),
                "training_gate": gate,
            }
        )
    eligible = [score for score in scores if score["eligible"]]
    if not eligible:
        raise ValueError("no candidate has a complete correct tuning matrix")
    selected = max(
        eligible,
        key=lambda score: (
            -score["regression_guard_failures"],
            score["minimum_aligned_throughput_gpu_speedup"],
            score["geomean_aligned_throughput_gpu_speedup"],
        ),
    )
    return {
        "scope": "one layout/count policy for every dtype, width and batch size",
        "rule": "minimize training wall/ragged guard failures, then maximize worst aligned batch32/256 GPU speedup, then its geometric mean; ties follow declared variant order",
        "selected": selected["variant"],
        "scores": scores,
        "heldout_results_used": False,
    }


def validate_manifest(manifest, baseline, root):
    if manifest["training_seed"] == manifest["validation_seed"]:
        raise ValueError("heldout validation seed must differ from the training seed")
    if manifest["training_seed"] != TRAINING_SEED or manifest["validation_seed"] != VALIDATION_SEED:
        raise ValueError("manifest seeds must match the predeclared experiment")
    if manifest["gate_policy"] != GATE_POLICY or manifest["cases"] != shared.benchmark_cases():
        raise ValueError("manifest must preserve the predeclared gate and workload matrix")
    if manifest["variants"] != variants() or manifest["selection"]["selected"] not in variants():
        raise ValueError("manifest candidate set differs from the predeclared experiment")
    if manifest["selection"]["heldout_results_used"] is not False:
        raise ValueError("policy must be selected without heldout results")
    if manifest["baseline_export_sha256"] != json_hash(baseline):
        raise ValueError("frozen baseline export differs from the training baseline")
    if manifest["compiler_implementation_sha256"] != shared._implementation_hash(root):
        raise ValueError("compiler changed after policy selection; tune a new manifest")
    if manifest["benchmark_source_sha256"] != benchmark_fingerprint():
        raise ValueError("benchmark driver changed after policy selection")
    if manifest["training_process_id"] == os.getpid():
        raise ValueError("heldout validation must run in a fresh process")
    report = json.loads(Path(manifest["tuning_report"]).read_text())
    if json_hash(report) != manifest["tuning_report_sha256"]:
        raise ValueError("tuning evidence changed after policy selection")
    if (
        report["kind"] != "register_tiling_tune"
        or report["configuration"]["seed"] != manifest["training_seed"]
        or report["process_id"] != manifest["training_process_id"]
        or report["gate_policy"] != manifest["gate_policy"]
        or any(
            report[key] != manifest[key]
            for key in (
                "compiler_implementation_sha256",
                "benchmark_source_sha256",
                "baseline_export_sha256",
            )
        )
    ):
        raise ValueError("manifest must bind the original training experiment")
    if manifest["selection"] != choose_policy(report["cases"]):
        raise ValueError(
            "manifest selection differs from the declared chooser on training evidence"
        )


def _measure(case, variant, exported, arguments, device, metile, seed, mx=None):
    import numpy as np

    validate_shader(exported, case)
    source, weight = shared.case_inputs(case, seed)
    expected, fast_expected = shared.references(source, weight)
    source_buffer, weight_buffer = metile.Buffer(data=source), metile.Buffer(data=weight)
    outputs = {
        label: metile.Buffer(data=np.full_like(source, np.nan))
        for label in ("candidate", "frozen_register4")
    }
    candidate, shader, compilation = _prepare(
        case, variant, (source_buffer, weight_buffer, outputs["candidate"]), device, metile
    )
    baseline, baseline_compilation = _import_frozen(
        exported, (source_buffer, weight_buffer, outputs["frozen_register4"]), device, metile
    )
    compilations = {"candidate": compilation, "frozen_register4": baseline_compilation}
    validate_compilations(compilations)
    candidate()
    baseline()
    device.sync()
    correctness = {
        label: shared.assert_correct(output.numpy(), expected, label)
        for label, output in outputs.items()
    }
    comparators = [("frozen_register4", baseline, device.sync, device.gpu_elapsed)]
    if mx is not None:
        source_mlx, weight_mlx = mx.array(source), mx.array(weight)
        mx.eval(source_mlx, weight_mlx)

        @mx.compile
        def matched_norm(values, weights):
            widened = values.astype(mx.float32)
            inverse = 1.0 / mx.sqrt(
                mx.mean(widened * widened, axis=-1, keepdims=True) + shared._EPSILON
            )
            return (widened * inverse * weights.astype(mx.float32)).astype(values.dtype)

        def matched_work():
            mx.eval(matched_norm(source_mlx, weight_mlx))

        def fast_work():
            mx.eval(mx.fast.rms_norm(source_mlx, weight_mlx, shared._EPSILON))

        def synchronize_all():
            device.sync()
            mx.synchronize()

        matched_result = matched_norm(source_mlx, weight_mlx)
        fast_result = mx.fast.rms_norm(source_mlx, weight_mlx, shared._EPSILON)
        mx.eval(matched_result, fast_result)
        synchronize_all()
        correctness["mlx_matched"] = shared.assert_correct(
            np.array(matched_result), expected, "mlx_matched"
        )
        correctness["mlx_fast"] = shared.assert_correct(
            np.array(fast_result), fast_expected, "mlx_fast_own_rounding_reference"
        )
        comparators += [
            ("mlx_matched", matched_work, synchronize_all, None),
            ("mlx_fast", fast_work, synchronize_all, None),
        ]
    rounds = []
    for round_index in range(arguments.rounds):
        shifted = (
            comparators[round_index % len(comparators) :]
            + comparators[: round_index % len(comparators)]
        )
        rounds.append(
            {
                label: shared.paired_measure(
                    candidate, control, sync, arguments.warmup_ms, arguments.rep_ms, elapsed
                )
                for label, control, sync, elapsed in shifted
            }
        )
    return {
        **case,
        "status": "measured",
        "variant": variant,
        "input_sha256": input_hash(source, weight),
        "validated_before_timing": True,
        "correctness": correctness,
        "compilations": compilations,
        "candidate_shader": shader,
        "frozen_shader_sha256": exported["shader"]["source_sha256"],
        "rounds": rounds,
        "summary": shared.summarize_pairs(rounds),
    }


def _experiment(arguments):
    root = arguments.root.resolve()
    driver_fingerprint = benchmark_fingerprint()
    baseline = json.loads(arguments.baseline_json.read_text())
    if (
        baseline["kind"] != "frozen_static_width_register4_baseline"
        or baseline["epsilon"] != shared._EPSILON
    ):
        raise ValueError("baseline must be the prechange static-width register4 export")
    exported = {case["name"]: case for case in baseline["cases"]}
    if len(exported) != 12 or len(baseline["cases"]) != 12:
        raise ValueError("frozen baseline must contain the complete unique 12-case matrix")
    for case in shared.benchmark_cases():
        validate_shader(exported[case["name"]], case)
    if arguments.mode == "validate":
        manifest = json.loads(arguments.manifest.read_text())
        validate_manifest(manifest, baseline, root)
        candidates = [manifest["selection"]["selected"]]
        seed = VALIDATION_SEED
    else:
        manifest = None
        candidates = variants()
        seed = TRAINING_SEED
    fingerprint = shared._implementation_hash(root)
    metile, device = _runtime(root)
    if (
        baseline["device"] != device.name
        or baseline["metal_compiler"] != device.metal_compiler_version
    ):
        raise ValueError("frozen and current shaders require identical device and compiler version")
    mx = None
    if arguments.mode == "validate":
        import mlx.core as mx
    results = []
    for case_index, case in enumerate(shared.benchmark_cases()):
        shifted = (
            candidates[case_index % len(candidates) :] + candidates[: case_index % len(candidates)]
        )
        for variant in shifted:
            try:
                result = _measure(
                    case, variant, exported[case["name"]], arguments, device, metile, seed, mx
                )
            except Exception as error:
                if arguments.mode == "validate":
                    raise
                result = {
                    **case,
                    "variant": variant,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"{variant['name']} {case['name']}: FAILED {result['error']}", flush=True)
            else:
                pair = result["summary"]["frozen_register4"]
                print(
                    f"{variant['name']} {case['name']}: GPU {pair['gpu_us']['speedup']:.3f}x, wall {pair['wall_us']['speedup']:.3f}x",
                    flush=True,
                )
            results.append(result)
    if shared._implementation_hash(root) != fingerprint:
        raise RuntimeError("compiler changed while measuring; discard this run")
    if benchmark_fingerprint() != driver_fingerprint:
        raise RuntimeError("benchmark sources changed while measuring; discard this run")
    report = {
        "kind": f"register_tiling_{arguments.mode}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "platform": platform.platform(),
        "device": device.name,
        "metal_compiler": device.metal_compiler_version,
        "mlx_version": mx.__version__ if mx is not None else None,
        "compiler_implementation_sha256": fingerprint,
        "benchmark_source_sha256": driver_fingerprint,
        "benchmark_sources": benchmark_sources(),
        "baseline_export_sha256": json_hash(baseline),
        "frozen_baseline": baseline,
        "gate_policy": GATE_POLICY,
        "precision_comparison": PRECISION_COMPARISON,
        "configuration": {
            "seed": seed,
            "rounds": arguments.rounds,
            "warmup_ms": arguments.warmup_ms,
            "rep_ms": arguments.rep_ms,
        },
        "precision": {
            "primary": "identical storage dtype, FP32 sum/normalization/weight multiply, final output cast; not bitwise exact",
            "mlx_matched": "mx.compile explicit full-FP32 graph then final output cast; primary matched-policy wall comparison",
            "mlx_fast": "secondary only: FP16 normalization rounds to storage before weight multiply, checked against its own rounding reference",
            "mlx_fast_source": shared._MLX_SOURCE,
            "relaxed_precision": False,
        },
        "measurement": {
            "gpu": "paired Metal command-buffer timestamps for candidate and exact frozen register4 shader",
            "wall": "paired per-invocation synchronized wall latency, compile/setup excluded; MLX includes allocation/evaluation",
            "compilation": "all meTile shaders rebuilt offline with identical -O2 -ffast-math compiler and default Metal language version",
            "specialization": "both frozen and current kernels use static N and BLOCK=1024; only compiler/ownership changes vary",
            "ordering": "AB/BA dispatch pairs; variant order rotates across cases; comparator order rotates across rounds",
        },
        "cases": results,
    }
    if arguments.mode == "tune":
        try:
            selection = choose_policy(results)
        except ValueError:
            report["selection"] = {"selected": None, "reason": "no complete correct candidate"}
            write_json(arguments.output, report)
            raise
        report["selection"] = selection
        write_json(arguments.output, report)
        manifest = {
            "kind": "frozen_register_tiling_selection",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "training_process_id": os.getpid(),
            "training_seed": TRAINING_SEED,
            "validation_seed": VALIDATION_SEED,
            "compiler_implementation_sha256": fingerprint,
            "benchmark_source_sha256": report["benchmark_source_sha256"],
            "baseline_export_sha256": json_hash(baseline),
            "tuning_report_sha256": json_hash(report),
            "tuning_report": str(arguments.output.resolve()),
            "gate_policy": GATE_POLICY,
            "precision_comparison": PRECISION_COMPARISON,
            "cases": shared.benchmark_cases(),
            "variants": variants(),
            "selection": selection,
        }
        write_json(arguments.manifest, manifest)
        print(f"Frozen single policy: {selection['selected']['name']}", flush=True)
    else:
        report["frozen_manifest"] = manifest
        report["manifest_sha256"] = json_hash(manifest)
        report["promotion_gate"] = promotion_gate(results)
        write_json(arguments.output, report)
        print(
            f"Heldout promotion gate: {'PASS' if report['promotion_gate']['passed'] else 'NOT MET'}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "tune", "validate"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-ms", type=float, default=100)
    parser.add_argument("--rep-ms", type=float, default=250)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        not math.isfinite(arguments.warmup_ms)
        or arguments.warmup_ms < 0
        or not math.isfinite(arguments.rep_ms)
        or arguments.rep_ms <= 0
        or arguments.rounds < 1
    ):
        parser.error(
            "timing budgets must be finite, warmup nonnegative, repetitions/rounds positive"
        )
    if arguments.mode == "export":
        export_baseline(arguments)
        return
    if arguments.baseline_json is None:
        parser.error("--baseline-json is required")
    if arguments.manifest is None:
        parser.error("--manifest is required to freeze or validate the selected policy")
    if arguments.mode == "validate" and not arguments.worker:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], check=True
        )
        return
    _experiment(arguments)


if __name__ == "__main__":
    main()
