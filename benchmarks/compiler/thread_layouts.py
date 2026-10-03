"""Measure exact batched transposes using checked ownership, scalar scatter and MLX."""

import argparse
import json
import math
import os
import platform
import sys
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

import numpy as np

from benchmarks.common.benchutils import bench_interleaved
from benchmarks.mlx.mlx_lm_backend import _hardware_metadata, _package_version
from metile.runtime.metal_device import MetalDevice

metile = checkout.load_compiler(_root)
checkout.load_kernel(_root)

_SHAPES = {"4x8": (4, 8), "8x8": (8, 8), "8x16": (8, 16), "16x16": (16, 16)}


@metile.kernel
def ownership_transpose(
    source,
    destination,
    batches,
    ROWS: metile.constexpr,
    COLUMNS: metile.constexpr,
    DESTINATION_LAYOUT: metile.constexpr,
):
    inputs = metile.tensor(source, shape=(batches, ROWS, COLUMNS), access="read")
    outputs = metile.tensor(destination, shape=(batches, COLUMNS, ROWS), access="write")
    batch = metile.program_id(0)
    positions = metile.arange(0, ROWS * COLUMNS)
    values = inputs.load((batch, positions // COLUMNS, positions % COLUMNS))
    converted = metile.convert_layout(values, DESTINATION_LAYOUT)
    owned_positions = metile.arange(0, ROWS * COLUMNS, layout=DESTINATION_LAYOUT)
    outputs.store((batch, owned_positions % COLUMNS, owned_positions // COLUMNS), converted)


@metile.kernel
def scatter_transpose(
    source, destination, batches, ROWS: metile.constexpr, COLUMNS: metile.constexpr
):
    inputs = metile.tensor(source, shape=(batches, ROWS, COLUMNS), access="read")
    outputs = metile.tensor(destination, shape=(batches, COLUMNS, ROWS), access="write")
    batch = metile.program_id(0)
    positions = metile.arange(0, ROWS * COLUMNS)
    values = inputs.load((batch, positions // COLUMNS, positions % COLUMNS))
    outputs.store((batch, positions % COLUMNS, positions // COLUMNS), values)


def _transpose_layout(rows, columns):
    row_bits = rows.bit_length() - 1
    total_bits = (rows * columns).bit_length() - 1
    bit_order = tuple(range(row_bits, total_bits)) + tuple(range(row_bits))
    return metile.ThreadLayout(bit_order)


def _execution_report(dispatch):
    report = getattr(dispatch, "execution_report", None)
    return report.to_dict() if report is not None else None


def _assert_exact(actual, expected, name):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape or storage dtype differs from the reference")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{name}: non-finite output")
    unsigned = np.uint16 if expected.dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(unsigned), expected.view(unsigned), err_msg=name
    )


def _measure_case(rows, columns, batches, dtype, arguments, device):
    import mlx.core as mx

    case_seed = arguments.seed + rows * columns + batches
    generator = np.random.default_rng(case_seed)
    source = generator.standard_normal((batches, rows, columns)).astype(dtype)
    source.flat[0] = -0.0
    source.flat[1] = 0.0
    expected = np.ascontiguousarray(source.transpose(0, 2, 1))
    source_buffer = metile.Buffer(data=source)
    scatter_output = metile.Buffer(data=np.full(expected.shape, np.nan, dtype=dtype))
    options = {"ROWS": rows, "COLUMNS": columns, "RELAXED_PRECISION": False}
    scatter = scatter_transpose[(batches,)].prepare(
        source_buffer, scatter_output, batches, **options
    )
    ownership = None
    owned_output = None
    layout = None
    if not arguments.scatter_only:
        layout = _transpose_layout(rows, columns)
        owned_output = metile.Buffer(data=np.full(expected.shape, np.nan, dtype=dtype))
        ownership = ownership_transpose[(batches,)].prepare(
            source_buffer, owned_output, batches, DESTINATION_LAYOUT=layout, **options
        )

    @mx.compile
    def mlx_transpose(values):
        return mx.contiguous(mx.transpose(values, (0, 2, 1)))

    mlx_source = mx.array(source)
    mx.eval(mlx_source)

    def mlx_work():
        mx.eval(mlx_transpose(mlx_source))

    def synchronize():
        device.sync()
        mx.synchronize()

    scatter()
    if ownership is not None:
        ownership()
    mlx_result = mlx_transpose(mlx_source)
    mx.eval(mlx_result)
    synchronize()
    _assert_exact(scatter_output.numpy(), expected, "metile_scatter")
    _assert_exact(np.array(mlx_result), expected, "mlx_compiled_contiguous_transpose")
    if owned_output is not None:
        _assert_exact(owned_output.numpy(), expected, "metile_ownership")
    timing_options = {"warmup_ms": arguments.warmup_ms, "rep_ms": arguments.rep_ms}
    pairs = {}
    if ownership is not None:
        owned_seconds, scatter_seconds = bench_interleaved(
            ownership, scatter, synchronize, **timing_options
        )
        pairs["ownership_vs_scatter"] = {
            "ownership_seconds": float(owned_seconds),
            "scatter_seconds": float(scatter_seconds),
            "speedup": float(scatter_seconds / owned_seconds),
        }
        owned_seconds, mlx_seconds = bench_interleaved(
            ownership, mlx_work, synchronize, **timing_options
        )
        pairs["ownership_vs_mlx"] = {
            "ownership_seconds": float(owned_seconds),
            "mlx_seconds": float(mlx_seconds),
            "speedup": float(mlx_seconds / owned_seconds),
        }
    scatter_seconds, mlx_seconds = bench_interleaved(
        scatter, mlx_work, synchronize, **timing_options
    )
    pairs["scatter_vs_mlx"] = {
        "scatter_seconds": float(scatter_seconds),
        "mlx_seconds": float(mlx_seconds),
        "speedup": float(mlx_seconds / scatter_seconds),
    }
    return {
        "input_shape": list(source.shape),
        "output_shape": list(expected.shape),
        "tile_shape": [rows, columns],
        "dtype": str(source.dtype),
        "seed": case_seed,
        "bytes_read_and_written": 2 * source.nbytes,
        "destination_layout": (
            {"bit_order": list(layout.bit_order), "xor_mask": layout.xor_mask}
            if layout is not None
            else None
        ),
        "correctness": {
            "passed": True,
            "bitwise_exact": True,
            "reference": "numpy_contiguous_batched_transpose",
            "validated_before_timing": True,
            "signed_zero_bits_checked": True,
        },
        "wall_pairs": pairs,
        "execution_reports": {
            "ownership": _execution_report(ownership),
            "scatter": _execution_report(scatter),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metile-root", type=Path, default=_root)
    parser.add_argument("--shapes", choices=tuple(_SHAPES), nargs="+", default=["4x8", "8x8"])
    parser.add_argument("--batches", type=int, nargs="+", default=[1024, 16384])
    parser.add_argument("--dtypes", choices=["float16", "float32"], nargs="+", default=["float32"])
    parser.add_argument("--scatter-only", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-ms", type=float, default=50)
    parser.add_argument("--rep-ms", type=float, default=200)
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    if not arguments.scatter_only and not hasattr(metile, "ThreadLayout"):
        parser.error("selected meTile tree has no ThreadLayout support; use --scatter-only")
    if arguments.seed < 0 or any(batches < 1 for batches in arguments.batches):
        parser.error("seed must be nonnegative and batches positive")
    if (
        not math.isfinite(arguments.warmup_ms)
        or not math.isfinite(arguments.rep_ms)
        or arguments.warmup_ms < 0
        or arguments.rep_ms <= 0
    ):
        parser.error("warmup-ms must be finite/nonnegative and rep-ms finite/positive")
    for shape in arguments.shapes:
        rows, columns = _SHAPES[shape]
        if any(rows * columns * batches * 4 > 64 * 1024**2 for batches in arguments.batches):
            parser.error("each input must be at most 64 MiB at FP32 storage")
    device = MetalDevice.get()
    cases = []
    for shape in arguments.shapes:
        rows, columns = _SHAPES[shape]
        for batches in arguments.batches:
            for dtype in arguments.dtypes:
                case = _measure_case(rows, columns, batches, dtype, arguments, device)
                cases.append(case)
                label = f"{batches}x{shape} {dtype}"
                for comparison, pair in case["wall_pairs"].items():
                    print(f"{label:24s} {comparison:24s} {pair['speedup']:.3f}x", flush=True)
    payload = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "metile_root": str(_root),
        "benchmark_source_sha256": checkout.benchmark_fingerprint(__file__),
        "benchmark_sources": checkout.benchmark_sources(__file__),
        "source_sha256": checkout.source_hashes(_root),
        "hardware": _hardware_metadata(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mlx": _package_version("mlx"),
            "metile": _package_version("metile"),
            "metal_compiler": device.metal_compiler_version,
        },
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("METILE_", "MLX_"))
        },
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "baseline_weights": "matching_per_case_float16_or_float32_storage",
            "optimized_weights": "matching_per_case_float16_or_float32_storage",
            "bitwise_exact": True,
            "operation": "pure_reordering_without_arithmetic_or_quantization",
            "validation": "numpy_bitwise_reference_and_mlx_contiguous_materialization",
        },
        "measurement": {
            "metric": "synchronized_wall_seconds",
            "clock": "perf_counter_ns",
            "statistic": "middle_80_percent_median",
            "order": "alternating_AB_BA_within_each_pair",
            "warmup_ms": arguments.warmup_ms,
            "rep_ms": arguments.rep_ms,
            "synchronization": "meTile_queue_and_mlx_default_stream_after_each_call",
            "timed_scope": "prepared_meTile_dispatch_vs_compiled_MLX_call_and_eval",
            "output_allocation": "meTile_preallocated_MLX_framework_managed",
            "input_allocation_and_compilation_excluded": True,
            "mlx_output": "mx.contiguous_materialized_transpose_not_a_strided_view",
            "communication": "compiler_selected_identity_simd_shuffle_or_synchronous_threadgroup",
        },
        "configuration": vars(arguments)
        | {
            "metile_root": str(arguments.metile_root),
            "output_json": str(arguments.output_json) if arguments.output_json else None,
        },
        "cases": cases,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output_json:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(serialized, encoding="utf-8")
        print(f"Wrote {arguments.output_json}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
