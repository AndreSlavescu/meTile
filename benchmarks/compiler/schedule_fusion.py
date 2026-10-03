"""Compare a strict FP32 epilogue DAG with two launches and compiled MLX."""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.checkout import activate_checkout

activate_checkout(Path(__file__).resolve().parents[2])

import mlx.core as mx
import numpy as np

import metile
from benchmarks.common.benchutils import bench_interleaved
from metile.frontend.autotune import _compiler_identity
from metile.runtime.metal_device import MetalDevice
from metile_kernels.gemm import matmul


@metile.kernel
def fused_epilogue(
    left,
    right,
    destination,
    M,
    N,
    K,
    alpha,
    beta,
    BLOCK_M: metile.constexpr,
    BLOCK_N: metile.constexpr,
    BLOCK_K: metile.constexpr,
):
    left_tensor = metile.tensor(left, shape=(M, K), block_shape=(BLOCK_M, BLOCK_K), access="read")
    right_tensor = metile.tensor(right, shape=(K, N), block_shape=(BLOCK_K, BLOCK_N), access="read")
    output = metile.tensor(
        destination, shape=(M, N), block_shape=(BLOCK_M, BLOCK_N), access="write"
    )
    row = metile.program_id(0) * BLOCK_M
    column = metile.program_id(1) * BLOCK_N
    accumulator = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
    for start in metile.tile_range(0, K, BLOCK_K):
        accumulator = metile.dot(
            left_tensor.load((row, start)), right_tensor.load((start, column)), accumulator
        )
    scaled = accumulator * alpha + beta
    result = metile.where(scaled > 0, scaled, scaled * 0.1)
    output.store((row, column), result)


@metile.kernel
def separate_epilogue(source, destination, size, alpha, beta, BLOCK: metile.constexpr):
    inputs = metile.tensor(source, shape=(size,), access="read")
    outputs = metile.tensor(destination, shape=(size,), access="write")
    positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    scaled = inputs.load((positions,)) * alpha + beta
    outputs.store((positions,), metile.where(scaled > 0, scaled, scaled * 0.1))


def _execution_report(dispatch):
    report = getattr(dispatch, "execution_report", None)
    return report.to_dict() if report is not None else None


def _measure_case(size, arguments, device):
    generator = np.random.default_rng(arguments.seed + size)
    left = (generator.standard_normal((size, size)) / np.sqrt(size)).astype(np.float32)
    right = generator.standard_normal((size, size)).astype(np.float32)
    alpha, beta = 0.375, -0.25
    schedule = metile.Schedule(backend=arguments.backend)
    options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 16,
        "RELAXED_PRECISION": False,
        "SCHEDULE": schedule,
    }
    left_buffer, right_buffer = metile.Buffer(data=left), metile.Buffer(data=right)
    fused_output = metile.Buffer.empty((size, size), dtype=np.float32)
    intermediate = metile.Buffer.empty((size, size), dtype=np.float32)
    separate_output = metile.Buffer.empty((size, size), dtype=np.float32)
    grid = (metile.cdiv(size, 64), metile.cdiv(size, 64))
    fused = fused_epilogue[grid].prepare(
        left_buffer, right_buffer, fused_output, size, size, size, alpha, beta, **options
    )
    product = matmul[grid].prepare(
        left_buffer, right_buffer, intermediate, size, size, size, **options
    )
    epilogue = separate_epilogue[(metile.cdiv(size * size, 256),)].prepare(
        intermediate,
        separate_output,
        size * size,
        alpha,
        beta,
        BLOCK=256,
        SCHEDULE=metile.Schedule(),
    )

    def two_launches():
        product()
        epilogue()

    @mx.compile
    def mlx_epilogue(left_operand, right_operand):
        scaled = (left_operand @ right_operand) * alpha + beta
        return mx.where(scaled > 0, scaled, scaled * 0.1)

    left_mlx, right_mlx = mx.array(left), mx.array(right)
    mx.eval(left_mlx, right_mlx)

    def mlx_work():
        mx.eval(mlx_epilogue(left_mlx, right_mlx))

    def synchronize():
        device.sync()
        mx.synchronize()

    scaled = (left @ right) * np.float32(alpha) + np.float32(beta)
    expected = np.where(scaled > 0, scaled, scaled * np.float32(0.1))
    fused()
    two_launches()
    actual_mlx = mlx_epilogue(left_mlx, right_mlx)
    mx.eval(actual_mlx)
    synchronize()
    actual = {
        "metile_fused": fused_output.numpy(),
        "metile_two_launches": separate_output.numpy(),
        "mlx_compile": np.array(actual_mlx),
    }
    for name, values in actual.items():
        np.testing.assert_allclose(values, expected, rtol=3e-4, atol=3e-5, err_msg=name)
    np.testing.assert_allclose(actual["metile_fused"], actual["mlx_compile"], rtol=3e-4, atol=3e-5)
    timing = {"warmup_ms": arguments.warmup_ms, "rep_ms": arguments.rep_ms}
    fused_time, separate_time = bench_interleaved(fused, two_launches, synchronize, **timing)
    fused_mlx_time, mlx_time = bench_interleaved(fused, mlx_work, synchronize, **timing)
    return {
        "shape": [size, size, size],
        "seed": arguments.seed + size,
        "max_absolute_error": {
            name: float(np.max(np.abs(values - expected))) for name, values in actual.items()
        },
        "fusion_pair_us": {
            "fused": float(fused_time * 1e6),
            "two_launches": float(separate_time * 1e6),
            "speedup": float(separate_time / fused_time),
        },
        "mlx_pair_us": {
            "fused": float(fused_mlx_time * 1e6),
            "mlx_compile": float(mlx_time * 1e6),
            "speedup": float(mlx_time / fused_mlx_time),
        },
        "execution_reports": {
            "fused": _execution_report(fused),
            "product": _execution_report(product),
            "separate_epilogue": _execution_report(epilogue),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "simdgroup", "tensor_ops"], default="auto")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("strict FP32 comparisons require MLX_ENABLE_TF32=0 before starting Python")
    if any(size <= 0 for size in arguments.sizes):
        parser.error("sizes must be positive")
    if arguments.warmup_ms < 0 or arguments.rep_ms <= 0:
        parser.error("warmup must be nonnegative and repetitions positive")
    device = MetalDevice.get()
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compiler_implementation": _compiler_identity(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": device.name,
        "metal_compiler": device.metal_compiler_version,
        "mlx_version": mx.__version__,
        "environment": {"MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"]},
        "metric": "alternating, per-invocation synchronized wall latency; setup excluded",
        "precision_comparison": {
            "class": "same_storage_precision",
            "same_weight_representation": True,
            "baseline_weights": "float32",
            "optimized_weights": "float32",
            "bitwise_exact": False,
            "storage": "FP32 inputs, intermediate and outputs; no FP16 rounding boundary",
            "metile": "FP32 accumulation, RELAXED_PRECISION=False",
            "mlx": "mx.compile with FP32 inputs and MLX_ENABLE_TF32=0",
            "reference": "NumPy FP32 matmul and epilogue; tolerance, not bitwise equivalence",
            "rtol": 3e-4,
            "atol": 3e-5,
        },
        "configuration": vars(arguments)
        | {"output": str(arguments.output) if arguments.output else None},
        "cases": [_measure_case(size, arguments, device) for size in arguments.sizes],
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
