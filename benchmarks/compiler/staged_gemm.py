"""Paired baseline/current measurement of explicitly software-staged strict GEMM."""

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


def cases():
    return [
        {
            "name": f"{dtype}_{rows}x{columns}x{reduction}",
            "dtype": dtype,
            "shape": [rows, columns, reduction],
        }
        for dtype in ("float16", "float32")
        for rows, columns, reduction in (
            (64, 64, 64),
            (256, 256, 256),
            (1024, 1024, 1024),
            (257, 255, 259),
        )
    ]


def implementation_hash(root):
    return checkout.implementation_hash(root)


def _measure(dispatch, device, warmup_ms, rep_ms):
    device.sync()
    deadline = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < deadline:
        dispatch()
        device.sync()
    was_enabled = gc.isenabled()
    gc.disable()
    gpu, wall = [], []
    deadline = time.perf_counter() + rep_ms / 1000
    try:
        while len(gpu) < 10 or time.perf_counter() < deadline:
            start = time.perf_counter_ns()
            dispatch()
            device.sync()
            elapsed_wall = (time.perf_counter_ns() - start) * 1e-3
            elapsed_gpu = device.gpu_elapsed() * 1e6
            if not math.isfinite(elapsed_gpu) or elapsed_gpu <= 0:
                raise RuntimeError("a positive hardware GPU timestamp is required")
            gpu.append(elapsed_gpu)
            wall.append(elapsed_wall)
    finally:
        if was_enabled:
            gc.enable()
    return {
        "samples": len(gpu),
        "gpu_us": statistics.median(gpu),
        "wall_us": statistics.median(wall),
        "gpu_min_max_us": [min(gpu), max(gpu)],
        "wall_min_max_us": [min(wall), max(wall)],
    }


def _worker(arguments, selected):
    root = arguments.worker_root.resolve()
    metile = checkout.load_compiler(root)
    import numpy as np

    from metile.runtime.metal_device import MetalDevice

    matmul = checkout.load_kernel(root, "gemm").matmul
    device = MetalDevice.get()
    fingerprint = implementation_hash(root)
    results = []
    for index, case in enumerate(selected):
        result = dict(case)
        result["seed"] = arguments.seed + index
        result["schedule"] = {
            "backend": "simdgroup",
            "double_buffer": True,
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 16,
            "RELAXED_PRECISION": False,
        }
        try:
            rows, columns, reduction = case["shape"]
            dtype = getattr(np, case["dtype"])
            generator = np.random.default_rng(result["seed"])
            left = (generator.standard_normal((rows, reduction)) / np.sqrt(reduction)).astype(dtype)
            right = generator.standard_normal((reduction, columns)).astype(dtype)
            expected = (left.astype(np.float32) @ right.astype(np.float32)).astype(dtype)
            result["input_sha256"] = hashlib.sha256(left.tobytes() + right.tobytes()).hexdigest()
            destination = metile.Buffer(data=np.full((rows, columns), np.nan, dtype=dtype))
            launcher = matmul.kernel_fn[(metile.cdiv(rows, 64), metile.cdiv(columns, 64))]
            dispatch = launcher.prepare(
                metile.Buffer(data=left),
                metile.Buffer(data=right),
                destination,
                rows,
                columns,
                reduction,
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_K=16,
                RELAXED_PRECISION=False,
                SCHEDULE=metile.Schedule(backend="simdgroup", double_buffer=True),
            )
            dispatch()
            device.sync()
            actual = destination.numpy()
            tolerance = (
                {"rtol": 2e-3, "atol": 2e-3}
                if dtype == np.float16
                else {"rtol": 3e-4, "atol": 3e-5}
            )
            np.testing.assert_allclose(actual, expected, equal_nan=False, **tolerance)
            if not dispatch.execution_report.double_buffered:
                raise AssertionError("requested staged pipeline did not materialize")
            result["correctness"] = {
                "reference": "NumPy FP32 matmul rounded to the output storage dtype",
                "max_absolute_error": float(
                    np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
                ),
                **tolerance,
            }
            result["source_sha256"] = hashlib.sha256(
                launcher._last_compiled.msl_source.encode()
            ).hexdigest()
            result["execution_report"] = dispatch.execution_report.to_dict()
            result["timing"] = _measure(dispatch, device, arguments.warmup_ms, arguments.rep_ms)
            result["status"] = "ok"
        except AssertionError as error:
            result.update(status="incorrect", error=f"{type(error).__name__}: {error}")
        except Exception as error:
            result.update(status="unavailable", error=f"{type(error).__name__}: {error}")
        results.append(result)
        print(f"{root.name}: {case['name']}: {result['status']}", flush=True)
    if implementation_hash(root) != fingerprint:
        raise RuntimeError("compiler implementation changed while benchmarking")
    report = {
        "root": str(root),
        "compiler_implementation_sha256": fingerprint,
        "device": device.name,
        "metal_compiler": device.metal_compiler_version,
        "platform": platform.platform(),
        "cases": results,
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


def summarize(samples):
    if set(samples) != {"baseline", "current"} or any(not runs for runs in samples.values()):
        raise ValueError("both baseline and current samples are required")
    runs = [run for group in samples.values() for run in group]
    expected_names = [case["name"] for case in runs[0]["cases"]]
    if any([case["name"] for case in run["cases"]] != expected_names for run in runs):
        raise ValueError("every sample must include the same ordered cases, including failures")
    for label, group in samples.items():
        if len({run["compiler_implementation_sha256"] for run in group}) != 1:
            raise ValueError(f"{label} compiler changed between samples")
    comparisons = []
    for index, name in enumerate(expected_names):
        grouped = {
            label: [run["cases"][index] for run in group] for label, group in samples.items()
        }
        row = {
            "name": name,
            "shape": grouped["current"][0]["shape"],
            "dtype": grouped["current"][0]["dtype"],
        }
        if any(case["status"] != "ok" for group in grouped.values() for case in group):
            row.update(
                status="incomparable",
                failures={
                    label: [
                        {"status": case["status"], "error": case.get("error")}
                        for case in group
                        if case["status"] != "ok"
                    ]
                    for label, group in grouped.items()
                },
            )
            comparisons.append(row)
            continue
        identities = {
            (
                tuple(case["shape"]),
                case["dtype"],
                case["input_sha256"],
                json.dumps(case["schedule"], sort_keys=True),
            )
            for group in grouped.values()
            for case in group
        }
        if len(identities) != 1:
            raise ValueError(
                "paired samples must use identical shapes, storage, inputs and schedules"
            )
        row["status"] = "ok"
        for metric in ("gpu_us", "wall_us"):
            values = {
                label: [case["timing"][metric] for case in group]
                for label, group in grouped.items()
            }
            if any(
                not math.isfinite(value) or value <= 0
                for group in values.values()
                for value in group
            ):
                raise ValueError("benchmark durations must be finite and positive")
            centers = {label: statistics.geometric_mean(group) for label, group in values.items()}
            row[metric] = {
                **centers,
                "speedup": centers["baseline"] / centers["current"],
                "samples": values,
            }
        row["source_sha256"] = {}
        for label, group in grouped.items():
            hashes = {case["source_sha256"] for case in group}
            if len(hashes) != 1:
                raise ValueError("generated source changed between repeated samples")
            row["source_sha256"][label] = next(iter(hashes))
        comparisons.append(row)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--warmup-ms", type=float, default=200)
    parser.add_argument("--rep-ms", type=float, default=500)
    parser.add_argument("--cooldown", type=float, default=2)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.warmup_ms < 0 or arguments.rep_ms <= 0 or arguments.cooldown < 0:
        parser.error("warmup and cooldown must be nonnegative; repetition budget must be positive")
    selected = [
        case for case in cases() if arguments.cases is None or case["name"] in arguments.cases
    ]
    if not selected or (
        arguments.cases is not None and set(arguments.cases) != {case["name"] for case in selected}
    ):
        parser.error("unknown or empty case selection")
    if arguments.worker_root is not None:
        _worker(arguments, selected)
        return
    if arguments.baseline_root is None:
        parser.error("--baseline-root is required")
    roots = {
        "baseline": arguments.baseline_root.resolve(),
        "current": arguments.current_root.resolve(),
    }
    if any(not (root / "metile" / "__init__.py").is_file() for root in roots.values()):
        parser.error("both roots must contain a meTile source tree")
    samples = {"baseline": [], "current": []}
    order = ("baseline", "current", "current", "baseline")
    with tempfile.TemporaryDirectory(prefix="metile-staging-pair-") as directory:
        temporary = Path(directory)
        for index, label in enumerate(order):
            if index:
                time.sleep(arguments.cooldown)
            output = temporary / f"{index}-{label}.json"
            environment = dict(os.environ, METILE_CACHE_DIR=str(temporary / f"cache-{label}"))
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-root",
                    str(roots[label]),
                    "--cases",
                    *(case["name"] for case in selected),
                    "--warmup-ms",
                    str(arguments.warmup_ms),
                    "--rep-ms",
                    str(arguments.rep_ms),
                    "--seed",
                    str(arguments.seed),
                    "--output",
                    str(output),
                ],
                cwd=roots[label],
                env=environment,
                check=True,
            )
            samples[label].append(json.loads(output.read_text()))
    comparisons = summarize(samples)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_source_sha256": checkout.benchmark_fingerprint(__file__),
        "benchmark_sources": checkout.benchmark_sources(__file__),
        "metric": "warmed per-dispatch Metal GPU timestamps and synchronized wall latency; compilation and allocation excluded",
        "pair_order": order,
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "storage": "identical FP16 or FP32 inputs and outputs per case",
            "arithmetic": "SIMD-group FP32 accumulation; RELAXED_PRECISION=False for both variants",
            "reference": "seeded NumPy FP32 matmul checked before timing every case",
            "bitwise_exact": False,
        },
        "configuration": {
            "warmup_ms": arguments.warmup_ms,
            "rep_ms": arguments.rep_ms,
            "seed": arguments.seed,
            "cooldown_seconds": arguments.cooldown,
        },
        "samples": samples,
        "comparisons": comparisons,
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in comparisons:
        if row["status"] == "ok":
            print(
                f"{row['name']}: GPU {row['gpu_us']['speedup']:.3f}x, wall {row['wall_us']['speedup']:.3f}x baseline/current"
            )
        else:
            print(f"{row['name']}: incomparable; failure preserved in artifact")
    if any(case["status"] != "ok" for run in samples["current"] for case in run["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
