import argparse
import json
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from benchutils import bench

import metile

# Cooldown between kernel groups to avoid thermal carry-over
_COOLDOWN = 2.0

# Longer measurement window for CI stability
_WARMUP_MS = 100
_REP_MS = 500


def _bench(dispatch):
    """Bench with CI-tuned parameters for stability."""
    return bench(dispatch, warmup_ms=_WARMUP_MS, rep_ms=_REP_MS)


def bench_gemm():
    """Benchmark GEMM at representative sizes."""
    from kernels.gemm import matmul

    results = {}
    for M, N, K in [(256, 256, 256), (1024, 1024, 1024)]:
        A = metile.Buffer(data=np.random.randn(M, K).astype(np.float32).ravel())
        B = metile.Buffer(data=np.random.randn(K, N).astype(np.float32).ravel())
        C = metile.Buffer.zeros((M * N,))
        grid = (metile.cdiv(M, 64), metile.cdiv(N, 64))
        dispatch = matmul[grid].prepare(A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
        t = _bench(dispatch)
        results[f"gemm_{M}x{N}x{K}"] = t
    return results


def bench_softmax():
    """Benchmark softmax."""
    from kernels.softmax import softmax

    results = {}
    for nrows, hidden in [(256, 1024), (1024, 4096)]:
        X = metile.Buffer(data=np.random.randn(nrows, hidden).astype(np.float32).ravel())
        Out = metile.Buffer.zeros((nrows * hidden,))
        dispatch = softmax[(nrows,)].prepare(X, Out, hidden, BLOCK=256)
        t = _bench(dispatch)
        results[f"softmax_{nrows}x{hidden}"] = t
    return results


def bench_layernorm():
    """Benchmark layernorm."""
    from kernels.layernorm import layernorm

    results = {}
    for nrows, hidden in [(256, 1024), (1024, 4096)]:
        X = metile.Buffer(data=np.random.randn(nrows, hidden).astype(np.float32).ravel())
        W = metile.Buffer(data=np.random.randn(hidden).astype(np.float32))
        B = metile.Buffer(data=np.random.randn(hidden).astype(np.float32))
        Out = metile.Buffer.zeros((nrows * hidden,))
        dispatch = layernorm[(nrows,)].prepare(X, W, B, Out, hidden, BLOCK=256)
        t = _bench(dispatch)
        results[f"layernorm_{nrows}x{hidden}"] = t
    return results


def bench_fft():
    """Benchmark FFT."""
    from kernels.fft import fft_dispatch

    results = {}
    for batch, N in [(1, 256), (32, 256), (1, 1024), (128, 1024)]:
        xr = metile.Buffer(data=np.random.randn(batch * N).astype(np.float32))
        xi = metile.Buffer(data=np.zeros(batch * N, dtype=np.float32))
        yr = metile.Buffer.zeros((batch * N,))
        yi = metile.Buffer.zeros((batch * N,))
        dispatch = fft_dispatch(batch, N, xr, xi, yr, yi)
        t = _bench(dispatch)
        results[f"fft_{batch}x{N}"] = t
    return results


def run_all():
    results = {}

    groups = [
        ("GEMM", bench_gemm),
        ("Softmax", bench_softmax),
        ("LayerNorm", bench_layernorm),
        ("FFT", bench_fft),
    ]

    for _name, fn in groups:
        time.sleep(_COOLDOWN)
        results.update(fn())

    return results


def compare(current, baseline_path, threshold=0.15):
    """Compare current results against baseline. Returns True if no regressions."""
    with open(baseline_path) as f:
        raw = json.load(f)
    # Handle both flat dict and github-action-benchmark format
    if isinstance(raw, list):
        baseline = {entry["name"]: entry["value"] * 1e-6 for entry in raw}
    else:
        baseline = raw

    print(f"{'kernel':<30} {'baseline':>12} {'current':>12} {'change':>8}  status")
    print("-" * 75)

    regressions = []
    for key in sorted(current.keys()):
        cur = current[key]
        if key not in baseline:
            print(f"{key:<30} {'n/a':>12} {cur * 1e6:>10.1f}us {'new':>8}")
            continue
        base = baseline[key]
        change = (cur - base) / base
        status = "OK"
        if change > threshold:
            status = "REGRESSION"
            regressions.append((key, change))
        elif change < -threshold:
            status = "FASTER"
        print(f"{key:<30} {base * 1e6:>10.1f}us {cur * 1e6:>10.1f}us {change:>+7.1%}  {status}")

    if regressions:
        print(f"\n{len(regressions)} regression(s) detected (>{threshold:.0%} slower):")
        for key, change in regressions:
            print(f"  {key}: {change:+.1%}")
        return False
    print("\nNo regressions detected.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Performance regression benchmarks")
    parser.add_argument("--compare", type=str, help="Path to baseline JSON to compare against")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--threshold", type=float, default=0.15, help="Regression threshold (default: 15%%)"
    )
    args = parser.parse_args()

    results = run_all()

    if args.output:
        # github-action-benchmark expects [{name, unit, value}, ...]
        bench_json = [
            {"name": k, "unit": "us", "value": round(v * 1e6, 2)} for k, v in results.items()
        ]
        with open(args.output, "w") as f:
            json.dump(bench_json, f, indent=2)
        print(f"Results saved to {args.output}")

    if args.compare:
        ok = compare(results, args.compare, args.threshold)
        sys.exit(0 if ok else 1)
    elif not args.output:
        for k, v in sorted(results.items()):
            print(f"  {k:<30} {v * 1e6:>10.1f} us")


if __name__ == "__main__":
    main()
