"""Validate and benchmark descriptor kernels against MLX at matching storage precision."""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import checkout

_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument("--metile-root", type=Path, default=Path(__file__).resolve().parents[2])
_source_arguments, _ = _source_parser.parse_known_args()
_root = _source_arguments.metile_root.resolve()
if not (_root / "metile" / "__init__.py").is_file():
    _source_parser.error("--metile-root must contain the meTile Python package")
checkout.activate_checkout(_root)

import mlx.core as mx
import numpy as np

from benchmarks.common.benchutils import bench_interleaved
from benchmarks.mlx.mlx_lm_backend import _git_revision, _hardware_metadata, _package_version
from metile.runtime.metal_device import MetalDevice

metile = checkout.load_compiler(_root)
matmul = checkout.load_kernel(_root, "gemm").matmul
layernorm = checkout.load_kernel(_root, "layernorm").layernorm
rmsnorm = checkout.load_kernel(_root, "rmsnorm").rmsnorm

_NORM_SHAPES = ((64, 1024), (64, 1027))
_GEMM_SHAPES = ((256, 256, 256), (1024, 1024, 1024))
_GEMM_CONFIG = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "RELAXED_PRECISION": False}


def _prepare(family, shape, seed):
    random = np.random.default_rng(seed)
    if family == "gemm":
        rows, columns, inner = shape
        arrays = [
            random.standard_normal((rows, inner)).astype(np.float16),
            random.standard_normal((inner, columns)).astype(np.float16),
        ]
        reference = (arrays[0].astype(np.float32) @ arrays[1].astype(np.float32)).astype(np.float16)
        output_shape = (rows, columns)
        config = dict(_GEMM_CONFIG)
        tolerance = {"rtol": 5e-3, "atol": 2e-2}
    else:
        rows, columns = shape
        arrays = [
            random.standard_normal(shape).astype(np.float32),
            random.standard_normal(columns).astype(np.float32),
        ]
        values = arrays[0].astype(np.float64)
        if family == "rmsnorm":
            reference = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-5)
        else:
            arrays.append(random.standard_normal(columns).astype(np.float32))
            centered = values - np.mean(values, axis=-1, keepdims=True)
            reference = centered / np.sqrt(
                np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5
            )
        reference = reference * arrays[1]
        if family == "layernorm":
            reference = reference + arrays[2]
        reference = reference.astype(np.float32)
        output_shape = shape
        config = {"BLOCK": 256}
        tolerance = {"rtol": 3e-5, "atol": 3e-5}

    buffers = [metile.Buffer(data=array.ravel()) for array in arrays]
    output = metile.Buffer.zeros(output_shape, dtype=arrays[0].dtype)
    mlx_inputs = tuple(mx.array(array) for array in arrays)
    mx.eval(*mlx_inputs)
    prepared_at = time.perf_counter()
    if family == "gemm":
        grid = (metile.cdiv(rows, config["BLOCK_M"]), metile.cdiv(columns, config["BLOCK_N"]))
        dispatch = matmul[grid].prepare(*buffers, output, rows, columns, inner, **config)

        def evaluate_mlx():
            return mlx_inputs[0] @ mlx_inputs[1]
    elif family == "rmsnorm":
        dispatch = rmsnorm[(rows,)].prepare(*buffers, output, columns, 1e-5, **config)

        def evaluate_mlx():
            return mx.fast.rms_norm(*mlx_inputs, eps=1e-5)
    else:
        dispatch = layernorm[(rows,)].prepare(*buffers, output, columns, **config)

        def evaluate_mlx():
            return mx.fast.layer_norm(*mlx_inputs, eps=1e-5)

    prepare_seconds = time.perf_counter() - prepared_at
    baseline_started = time.perf_counter()
    baseline = evaluate_mlx()
    mx.eval(baseline)
    first_mlx_seconds = time.perf_counter() - baseline_started
    actual = output.numpy().reshape(output_shape)
    expected = np.array(baseline)
    for result in (actual, expected):
        if not np.isfinite(result).all():
            raise AssertionError(f"{family} {shape}: non-finite output")
        np.testing.assert_allclose(result, reference, **tolerance)
    np.testing.assert_allclose(actual, expected, **tolerance)

    def run_mlx():
        mx.eval(evaluate_mlx())

    record = {
        "family": family,
        "shape": shape,
        "seed": seed,
        "dtype": str(arrays[0].dtype),
        "config": config,
        "correctness": {
            "passed": True,
            **tolerance,
            "max_absolute_difference_vs_mlx": float(
                np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
            ),
            "reference": "numpy_float32_matmul" if family == "gemm" else "numpy_float64_norm",
        },
        "setup_seconds": {"metile_prepare": prepare_seconds, "mlx_first_eval": first_mlx_seconds},
    }
    return dispatch, run_mlx, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metile-root", type=Path, default=_root)
    parser.add_argument("--family", choices=("all", "rmsnorm", "layernorm", "gemm"), default="all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-ms", type=float, default=25)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup_ms < 0 or arguments.rep_ms <= 0 or arguments.seed < 0:
        parser.error("seed and warmup-ms must be nonnegative; rep-ms must be positive")
    device = MetalDevice.get()

    def synchronize():
        device.sync()
        mx.synchronize()

    families = (
        ("rmsnorm", "layernorm", "gemm") if arguments.family == "all" else (arguments.family,)
    )
    results = []
    print("Tensor descriptors: synchronized wall latency, resident inputs, fixed configurations")
    for family in families:
        shapes = _GEMM_SHAPES if family == "gemm" else _NORM_SHAPES
        for shape in shapes:
            dispatch, baseline, record = _prepare(family, shape, arguments.seed)
            metile_seconds, mlx_seconds = bench_interleaved(
                dispatch,
                baseline,
                sync=synchronize,
                warmup_ms=arguments.warmup_ms,
                rep_ms=arguments.rep_ms,
            )
            record["timings"] = {
                "metile_seconds": float(metile_seconds),
                "mlx_seconds": float(mlx_seconds),
                "speedup": float(mlx_seconds / metile_seconds),
            }
            results.append(record)
            print(
                f"{family:10s} {'x'.join(map(str, shape)):16s} "
                f"meTile {metile_seconds * 1e6:9.2f} us  "
                f"MLX {mlx_seconds * 1e6:9.2f} us  {mlx_seconds / metile_seconds:.3f}x",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "baseline_weights": "float32_norms_float16_gemm",
            "optimized_weights": "float32_norms_float16_gemm",
            "bitwise_exact": False,
            "validation": "per_case_numpy_and_mlx_tolerance_checks",
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "revision": (
            _git_revision()
            if _root == Path(__file__).resolve().parents[2]
            else "external-tree-see-source-hashes"
        ),
        "metile_root": str(_root),
        "source_sha256": checkout.source_hashes(_root),
        "benchmark_source_sha256": checkout.benchmark_fingerprint(__file__),
        "benchmark_sources": checkout.benchmark_sources(__file__),
        "hardware": _hardware_metadata(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mlx": _package_version("mlx"),
            "metile": _package_version("metile"),
            "metal_compiler": device.metal_compiler_version,
        },
        "measurement": {
            "metric": "synchronized_wall_seconds",
            "clock": "perf_counter_ns",
            "statistic": "middle_80_percent_median",
            "order": "alternating_AB_BA",
            "warmup_ms": arguments.warmup_ms,
            "rep_ms": arguments.rep_ms,
            "synchronization": "meTile_queue_and_mlx_default_stream_after_each_call",
            "timed_scope": "prepared_meTile_dispatch_vs_MLX_operation_construction_and_eval",
            "output_allocation": "meTile_preallocated_MLX_framework_managed",
            "input_allocation_and_compilation_excluded": True,
            "numerics": "matching_storage_dtypes_tolerance_checked_not_bitwise_exact",
        },
        "results": results,
    }
    if arguments.output_json:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {arguments.output_json}")


if __name__ == "__main__":
    main()
