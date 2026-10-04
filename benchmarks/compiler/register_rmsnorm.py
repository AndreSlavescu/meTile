"""Paired RMSNorm comparison with a frozen compiler baseline and explicit promotion gates."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import checkout

_WIDTHS = (1009, 1024)
_BATCHES = (1, 32, 256)
_DTYPES = ("float16", "float32")
_EPSILON = 1e-5
_GPU_SPEEDUP_GATE = 1.10
_WALL_REGRESSION_GATE = 1.03
_MLX_SOURCE = (
    "https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/rms_norm.metal"
)


def benchmark_cases():
    return [
        {"name": f"{dtype}_{batches}x{width}", "dtype": dtype, "batches": batches, "width": width}
        for dtype in _DTYPES
        for width in _WIDTHS
        for batches in _BATCHES
    ]


def _implementation_hash(root):
    return checkout.implementation_hash(root)


def case_inputs(case, seed):
    import numpy as np

    case_seed = seed + case["width"] * 1009 + case["batches"] * 17 + (case["dtype"] == "float32")
    generator = np.random.default_rng(case_seed)
    source = generator.standard_normal((case["batches"], case["width"])).astype(case["dtype"])
    weight = generator.uniform(0.5, 1.5, case["width"]).astype(case["dtype"])
    return source, weight


def references(source, weight):
    import numpy as np

    values = source.astype(np.float32)
    inverse = np.float32(1.0) / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32) + np.float32(_EPSILON)
    )
    normalized = values * inverse
    matched = (normalized * weight.astype(np.float32)).astype(source.dtype)
    mlx_fast = (
        normalized.astype(source.dtype).astype(np.float32) * weight.astype(np.float32)
    ).astype(source.dtype)
    return matched, mlx_fast


def assert_correct(actual, expected, label):
    import numpy as np

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: output shape or storage precision differs")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{label}: non-finite or unwritten output")
    tolerance = (
        {"rtol": 1e-3, "atol": 2e-3}
        if expected.dtype == np.float16
        else {"rtol": 3e-5, "atol": 2e-5}
    )
    np.testing.assert_allclose(actual, expected, err_msg=label, **tolerance)
    return {
        "passed": True,
        "max_absolute_error": float(
            np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
        ),
        **tolerance,
    }


def _timed_call(function, synchronize, gpu_elapsed):
    start = time.perf_counter_ns()
    function()
    synchronize()
    result = {"wall_us": (time.perf_counter_ns() - start) * 1e-3}
    if gpu_elapsed is not None:
        result["gpu_us"] = gpu_elapsed() * 1e6
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise RuntimeError("timing samples must be finite and positive")
    return result


def paired_measure(first, second, synchronize, warmup_ms, rep_ms, gpu_elapsed=None):
    synchronize()
    deadline = time.perf_counter() + warmup_ms / 1000
    iteration = 0
    while time.perf_counter() < deadline:
        functions = (first, second) if iteration % 2 == 0 else (second, first)
        for function in functions:
            function()
            synchronize()
        iteration += 1
    samples = [[], []]
    was_enabled = gc.isenabled()
    gc.disable()
    deadline = time.perf_counter() + rep_ms / 1000
    try:
        iteration = 0
        while iteration < 20 or time.perf_counter() < deadline:
            order = (0, 1) if iteration % 2 == 0 else (1, 0)
            for index in order:
                samples[index].append(_timed_call((first, second)[index], synchronize, gpu_elapsed))
            iteration += 1
    finally:
        if was_enabled:
            gc.enable()
    result = {"pairs": iteration, "order": "alternating AB/BA with per-dispatch synchronization"}
    for metric in samples[0][0]:
        medians = [statistics.median(sample[metric] for sample in group) for group in samples]
        result[metric] = {
            "candidate": medians[0],
            "control": medians[1],
            "speedup": medians[1] / medians[0],
        }
    return result


def _kernel_export(launcher):
    compiled = launcher._last_compiled
    return {
        "msl_source": compiled.msl_source,
        "source_sha256": hashlib.sha256(compiled.msl_source.encode()).hexdigest(),
        "function_name": compiled.func_name,
        "threadgroup_size": list(compiled.threadgroup_size),
        "is_gemm": compiled.is_gemm,
        "prefer_ordered": compiled.prefer_ordered,
        "argument_indices": list(compiled.argument_indices),
        "output_indices": list(compiled.output_indices),
        "execution_report": compiled.execution_report.to_dict(),
    }


def validate_baseline_export(exported, case, input_hash):
    if any(exported.get(name) != case[name] for name in ("name", "dtype", "batches", "width")):
        raise ValueError("baseline shader case identity does not match the requested workload")
    if exported.get("input_sha256") != input_hash:
        raise ValueError("baseline and current must use identical input bytes")
    shader = exported["shader"]
    if hashlib.sha256(shader["msl_source"].encode()).hexdigest() != shader["source_sha256"]:
        raise ValueError("baseline shader source does not match its fingerprint")
    if shader["argument_indices"] != [0, 1, 2, 3, 4] or shader["output_indices"] != [2]:
        raise ValueError("baseline shader ABI is not the supported X,W,Out,N,eps contract")
    if shader["is_gemm"] or shader["threadgroup_size"] != [256, 1, 1]:
        raise ValueError("baseline shader must preserve the BLOCK=256 row-parallel launch")
    if not exported.get("correctness", {}).get("passed"):
        raise ValueError("baseline shader correctness must pass before comparison")


def _export_baseline(arguments, selected):
    root = arguments.export_root.resolve()
    metile = checkout.load_compiler(root)
    import numpy as np

    from metile.runtime.metal_device import MetalDevice

    rmsnorm = checkout.load_kernel(root, "rmsnorm").rmsnorm
    device = MetalDevice.get()
    fingerprint = _implementation_hash(root)
    results = []
    for case in selected:
        source, weight = case_inputs(case, arguments.seed)
        expected, _ = references(source, weight)
        output = metile.Buffer(data=np.full_like(source, np.nan))
        launcher = rmsnorm[(case["batches"],)]
        dispatch = launcher.prepare(
            metile.Buffer(data=source),
            metile.Buffer(data=weight),
            output,
            case["width"],
            _EPSILON,
            BLOCK=256,
            RELAXED_PRECISION=False,
        )
        dispatch()
        device.sync()
        results.append(
            {
                **case,
                "input_sha256": hashlib.sha256(source.tobytes() + weight.tobytes()).hexdigest(),
                "correctness": assert_correct(output.numpy(), expected, "baseline"),
                "shader": _kernel_export(launcher),
                "initial_same_kernel_pair": paired_measure(
                    dispatch,
                    dispatch,
                    device.sync,
                    arguments.warmup_ms,
                    arguments.rep_ms,
                    device.gpu_elapsed,
                ),
            }
        )
        print(f"baseline checked/exported {case['name']}", flush=True)
    if _implementation_hash(root) != fingerprint:
        raise RuntimeError("baseline compiler changed while exporting its shaders")
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "compiler_implementation_sha256": fingerprint,
        "device": device.name,
        "metal_compiler": device.metal_compiler_version,
        "seed": arguments.seed,
        "epsilon": _EPSILON,
        "cases": results,
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


def _compile_shader(device, shader):
    if device.has_metal_compiler:
        pipeline, precompiled = device.compile_msl_precompiled(
            shader["msl_source"], shader["function_name"]
        )
    else:
        pipeline = device.compile_msl(shader["msl_source"], shader["function_name"])
        precompiled = False
    return pipeline, {
        "path": "offline" if precompiled else "runtime_jit",
        "precompiled": precompiled,
        "offline_compiler_available": bool(device.has_metal_compiler),
        "metal_compiler": device.metal_compiler_version,
        "flags": ["-O2", "-ffast-math"] if precompiled else [],
        "metal_standard": "compiler_default",
    }


def _validate_compile_paths(compilations):
    if set(compilations) != {"candidate", "current_control", "frozen_baseline"}:
        raise RuntimeError("compilation records are required for all meTile comparators")
    identities = {
        (
            record["path"],
            record["precompiled"],
            record["metal_compiler"],
            tuple(record["flags"]),
            record["metal_standard"],
        )
        for record in compilations.values()
    }
    if len(identities) != 1:
        raise RuntimeError(
            "all meTile comparators must use the same Metal compilation path, flags and toolchain"
        )


def _dispatch_for_shader(
    shader, resources, grid, device, execution_report=None, metal_buffers=None
):
    from metile.frontend.kernel import CompiledKernel, FastDispatcher

    pipeline, compilation = _compile_shader(device, shader)
    compiled = CompiledKernel(
        pipeline,
        shader["msl_source"],
        shader["function_name"],
        tuple(shader["threadgroup_size"]),
        is_gemm=False,
        prefer_ordered=shader["prefer_ordered"],
        output_indices=tuple(shader["output_indices"]),
        argument_indices=tuple(shader["argument_indices"]),
        execution_report=execution_report,
    )
    bindings = (
        [resource.metal_buffer for resource in resources]
        if metal_buffers is None
        else metal_buffers
    )
    return FastDispatcher(compiled, bindings, grid, device, resources), compilation


def _import_baseline_shader(exported, source, weight, output, device, metile):
    import numpy as np

    resources = (
        source,
        weight,
        output,
        metile.Buffer(data=np.array([exported["width"]], dtype=np.int32)),
        metile.Buffer(data=np.array([_EPSILON], dtype=np.float32)),
    )
    return _dispatch_for_shader(
        exported["shader"],
        resources,
        (exported["batches"],),
        device,
    )


def summarize_pairs(rounds):
    if not rounds:
        raise ValueError("at least one measured round is required")
    keys = set(rounds[0])
    if any(set(round_result) != keys for round_result in rounds):
        raise ValueError("all rounds must include the same comparator set")
    result = {}
    for name in sorted(keys):
        metrics = {key for key in rounds[0][name] if key in {"gpu_us", "wall_us"}}
        result[name] = {}
        for metric in metrics:
            if any(
                not math.isfinite(round_result[name][metric][label])
                or round_result[name][metric][label] <= 0
                for round_result in rounds
                for label in ("candidate", "control")
            ):
                raise ValueError("timing durations must be finite and positive")
            ratios = [
                round_result[name][metric]["control"] / round_result[name][metric]["candidate"]
                for round_result in rounds
            ]
            if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
                raise ValueError("timing ratios must be finite and positive")
            result[name][metric] = {
                "speedup": statistics.geometric_mean(ratios),
                "round_speedups": ratios,
                "candidate": statistics.geometric_mean(
                    round_result[name][metric]["candidate"] for round_result in rounds
                ),
                "control": statistics.geometric_mean(
                    round_result[name][metric]["control"] for round_result in rounds
                ),
            }
    return result


def promotion_gate(results):
    expected = {case["name"]: case for case in benchmark_cases()}
    names = [case["name"] for case in results]
    if len(set(names)) != len(names) or set(names) != set(expected):
        return {
            "passed": False,
            "reason": "the complete predeclared 12-case workload matrix is required",
        }
    for case in results:
        if any(case.get(key) != value for key, value in expected[case["name"]].items()):
            raise ValueError("promotion workloads must preserve the predeclared case identities")
        if (
            not case.get("validated_before_timing")
            or not case.get("correctness")
            or not all(result.get("passed") for result in case["correctness"].values())
        ):
            return {
                "passed": False,
                "reason": "every comparator must pass correctness before timing",
            }
    comparisons = {}
    for comparator in ("frozen_baseline", "current_control"):
        failed_gpu, failed_wall = [], []
        for case in results:
            pair = case["summary"][comparator]
            gpu_speedup, wall_speedup = pair["gpu_us"]["speedup"], pair["wall_us"]["speedup"]
            if not all(math.isfinite(value) and value > 0 for value in (gpu_speedup, wall_speedup)):
                raise ValueError("promotion gates require finite positive timings")
            if case["batches"] in (32, 256) and gpu_speedup < _GPU_SPEEDUP_GATE:
                failed_gpu.append(case["name"])
            if 1.0 / wall_speedup > _WALL_REGRESSION_GATE:
                failed_wall.append(case["name"])
        comparisons[comparator] = {
            "passed": not failed_gpu and not failed_wall,
            "failed_gpu_cases": failed_gpu,
            "failed_wall_cases": failed_wall,
        }
    return {
        "passed": all(comparison["passed"] for comparison in comparisons.values()),
        "policy": "all batch32/256 cases must improve GPU throughput >=10%; every case must avoid >3% wall regression, against both frozen baseline and current control",
        "gpu_speedup_minimum": _GPU_SPEEDUP_GATE,
        "wall_latency_ratio_maximum": _WALL_REGRESSION_GATE,
        "comparisons": comparisons,
    }


def _measure_case(case, exported, arguments, device, metile, mx):
    import numpy as np

    kernel_module = checkout.load_kernel(Path(metile.__file__).resolve().parent.parent, "rmsnorm")
    rmsnorm, rmsnorm_register = kernel_module.rmsnorm, kernel_module.rmsnorm_register

    source, weight = case_inputs(case, arguments.seed)
    input_hash = hashlib.sha256(source.tobytes() + weight.tobytes()).hexdigest()
    validate_baseline_export(exported, case, input_hash)
    expected, fast_expected = references(source, weight)
    source_buffer, weight_buffer = metile.Buffer(data=source), metile.Buffer(data=weight)
    outputs = {
        label: metile.Buffer(data=np.full_like(source, np.nan))
        for label in ("candidate", "frozen_baseline", "current_control")
    }
    block = 1 << (case["width"] - 1).bit_length()
    if arguments.layout == "identity":
        layout = metile.ThreadLayout.identity(block, elements_per_thread=4)
    else:
        bits = block.bit_length() - 1
        layout = metile.ThreadLayout((bits - 2, bits - 1, *range(bits - 2)), elements_per_thread=4)
    launcher = rmsnorm_register[(case["batches"],)]
    candidate = launcher.prepare(
        source_buffer,
        weight_buffer,
        outputs["candidate"],
        _EPSILON,
        N=case["width"],
        BLOCK=block,
        LAYOUT=layout,
        RELAXED_PRECISION=False,
    )
    control_launcher = rmsnorm[(case["batches"],)]
    control = control_launcher.prepare(
        source_buffer,
        weight_buffer,
        outputs["current_control"],
        case["width"],
        _EPSILON,
        BLOCK=256,
        RELAXED_PRECISION=False,
    )
    candidate_shader = _kernel_export(launcher)
    control_shader = _kernel_export(control_launcher)
    candidate, candidate_compilation = _dispatch_for_shader(
        candidate_shader,
        launcher._last_resources,
        launcher.grid,
        device,
        candidate.execution_report,
        launcher._last_metal_buffers,
    )
    control, control_compilation = _dispatch_for_shader(
        control_shader,
        control_launcher._last_resources,
        control_launcher.grid,
        device,
        control.execution_report,
        control_launcher._last_metal_buffers,
    )
    baseline, baseline_compilation = _import_baseline_shader(
        exported, source_buffer, weight_buffer, outputs["frozen_baseline"], device, metile
    )
    compilations = {
        "candidate": candidate_compilation,
        "current_control": control_compilation,
        "frozen_baseline": baseline_compilation,
    }
    _validate_compile_paths(compilations)
    source_mlx, weight_mlx = mx.array(source), mx.array(weight)
    mx.eval(source_mlx, weight_mlx)

    @mx.compile
    def matched_norm(values, weights):
        widened = values.astype(mx.float32)
        inverse = 1.0 / mx.sqrt(mx.mean(widened * widened, axis=-1, keepdims=True) + _EPSILON)
        return (widened * inverse * weights.astype(mx.float32)).astype(values.dtype)

    def matched_work():
        mx.eval(matched_norm(source_mlx, weight_mlx))

    def fast_work():
        mx.eval(mx.fast.rms_norm(source_mlx, weight_mlx, _EPSILON))

    def synchronize_all():
        device.sync()
        mx.synchronize()

    for dispatch in (candidate, baseline, control):
        dispatch()
    matched_result = matched_norm(source_mlx, weight_mlx)
    fast_result = mx.fast.rms_norm(source_mlx, weight_mlx, _EPSILON)
    mx.eval(matched_result, fast_result)
    synchronize_all()
    correctness = {
        label: assert_correct(output.numpy(), expected, label) for label, output in outputs.items()
    }
    correctness["mlx_matched"] = assert_correct(np.array(matched_result), expected, "mlx_matched")
    correctness["mlx_fast"] = assert_correct(
        np.array(fast_result), fast_expected, "mlx_fast_own_rounding_reference"
    )
    correctness["mlx_fast"]["max_difference_from_matched_policy"] = float(
        np.max(np.abs(np.array(fast_result).astype(np.float32) - expected.astype(np.float32)))
    )
    comparators = (
        ("frozen_baseline", baseline, device.sync, device.gpu_elapsed),
        ("current_control", control, device.sync, device.gpu_elapsed),
        ("mlx_matched", matched_work, synchronize_all, None),
        ("mlx_fast", fast_work, synchronize_all, None),
    )
    rounds = []
    for round_index in range(arguments.rounds):
        shifted = (
            comparators[round_index % len(comparators) :]
            + comparators[: round_index % len(comparators)]
        )
        pairs = {}
        for name, function, synchronize, gpu_elapsed in shifted:
            synchronize_all()
            pairs[name] = paired_measure(
                candidate, function, synchronize, arguments.warmup_ms, arguments.rep_ms, gpu_elapsed
            )
        rounds.append(pairs)
    return {
        **case,
        "input_sha256": input_hash,
        "epsilon": _EPSILON,
        "candidate_layout": {"bit_order": list(layout.bit_order), "elements_per_thread": 4},
        "correctness": correctness,
        "validated_before_timing": True,
        "shaders": {
            "candidate": candidate_shader,
            "current_control": control_shader,
            "frozen_baseline": exported["shader"],
        },
        "compilations": compilations,
        "rounds": rounds,
        "summary": summarize_pairs(rounds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--export-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--layout", choices=("identity", "blocked"), default="identity")
    parser.add_argument("--seed", type=int, default=619)
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
        or arguments.seed < 0
    ):
        parser.error(
            "timing budgets must be finite/nonnegative, repetitions and rounds positive, seed nonnegative"
        )
    selected = [
        case
        for case in benchmark_cases()
        if arguments.cases is None or case["name"] in arguments.cases
    ]
    if not selected or (
        arguments.cases is not None and set(arguments.cases) != {case["name"] for case in selected}
    ):
        parser.error("unknown or empty case selection")
    if arguments.export_root is not None:
        _export_baseline(arguments, selected)
        return
    if arguments.baseline_json is None:
        if arguments.baseline_root is None:
            parser.error("--baseline-root or --baseline-json is required")
        with tempfile.TemporaryDirectory(prefix="metile-register-baseline-") as directory:
            output = Path(directory) / "shaders.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--export-root",
                    str(arguments.baseline_root.resolve()),
                    "--seed",
                    str(arguments.seed),
                    "--cases",
                    *(case["name"] for case in selected),
                    "--warmup-ms",
                    str(arguments.warmup_ms),
                    "--rep-ms",
                    str(arguments.rep_ms),
                    "--output",
                    str(output),
                ],
                cwd=arguments.baseline_root.resolve(),
                env=dict(os.environ, METILE_CACHE_DIR=str(Path(directory) / "cache")),
                check=True,
            )
            baseline = json.loads(output.read_text())
    else:
        baseline = json.loads(arguments.baseline_json.read_text())
    if baseline["seed"] != arguments.seed or baseline["epsilon"] != _EPSILON:
        parser.error("baseline fixture seed and epsilon must match the current run")
    exported = {case["name"]: case for case in baseline["cases"]}
    if any(case["name"] not in exported for case in selected):
        parser.error("baseline export is missing requested workload cases")
    root = arguments.current_root.resolve()
    metile = checkout.load_compiler(root)
    import mlx.core as mx

    from metile.runtime.metal_device import MetalDevice

    checkout.load_kernel(root)
    device = MetalDevice.get()
    if (
        baseline["metal_compiler"] != device.metal_compiler_version
        or baseline["device"] != device.name
    ):
        raise RuntimeError(
            "frozen source export and current measurement must use the same Metal toolchain and device"
        )
    fingerprint = _implementation_hash(root)
    results = []
    for case in selected:
        measured = _measure_case(case, exported[case["name"]], arguments, device, metile, mx)
        results.append(measured)
        primary = measured["summary"]["frozen_baseline"]
        print(
            f"{case['name']}: GPU {primary['gpu_us']['speedup']:.3f}x, wall {primary['wall_us']['speedup']:.3f}x baseline/candidate",
            flush=True,
        )
    if _implementation_hash(root) != fingerprint:
        raise RuntimeError("current compiler changed while measuring")
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "device": device.name,
        "metal_compiler": device.metal_compiler_version,
        "mlx_version": mx.__version__,
        "compiler_implementation_sha256": fingerprint,
        "baseline_compiler_implementation_sha256": baseline["compiler_implementation_sha256"],
        "baseline_root": baseline["root"],
        "current_root": str(root),
        "benchmark_source_sha256": checkout.benchmark_fingerprint(__file__),
        "benchmark_sources": checkout.benchmark_sources(__file__),
        "baseline_method": "baseline subprocess exports verified MSL and ABI; frozen baseline, candidate and current control shaders are explicitly rebuilt with the same offline Metal path (-O2 -ffast-math), or uniformly checked runtime fallback, then alternated in the current Metal runtime",
        "specialization_caveat": "candidate specializes N at compile time and retains a 1024-element row as four registers per thread; both controls use runtime N and BLOCK=256, so gains cannot be attributed solely to register retention",
        "metrics": {
            "gpu": "Metal command-buffer timestamps; candidate and meTile controls only",
            "wall": "per-invocation synchronized wall latency; compile/setup excluded; includes MLX allocation and graph evaluation",
        },
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "bitwise_exact": False,
            "primary_policy": "identical FP16/FP32 storage, FP32 sum-of-squares and normalization, FP32 weight multiply, one final output cast",
            "primary_mlx_comparator": "mx.compile explicit FP32 normalization/weight multiplication before final output cast",
            "mlx_fast_caveat": "mx.fast.rms_norm v0.32.0 rounds normalized input to storage dtype before multiplying weights; FP16 comparison is separately labeled and not identical rounding policy",
            "mlx_fast_source": _MLX_SOURCE,
            "reference": "seeded NumPy FP32 reduction; meTile/matched-MLX and fast-MLX each checked against their own rounding-boundary reference",
        },
        "configuration": {
            "seed": arguments.seed,
            "layout": arguments.layout,
            "rounds": arguments.rounds,
            "warmup_ms": arguments.warmup_ms,
            "rep_ms": arguments.rep_ms,
        },
        "cases": results,
        "promotion_gate": promotion_gate(results),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Promotion gate:", "PASS" if report["promotion_gate"]["passed"] else "NOT MET", flush=True
    )


if __name__ == "__main__":
    main()
