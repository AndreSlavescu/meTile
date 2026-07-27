"""Compare mx.compile against the meTile compiler: warm-up cost and steady-state speed.

The two do different amounts of work to reach a fast kernel, so warm-up is reported
separately rather than folded into throughput:

  eager MLX          no compilation at all, the reference
  mx.compile         traces the Python function and fuses it, per input shape
  meTile             generates MSL, compiles it with xcrun, and creates a pipeline
  meTile + autotune  the above, then times candidate schedules and keeps the winner

Warm-up is first-call latency minus steady-state latency, which isolates the cost of
reaching a fast kernel from the cost of running it. It is reported twice:

  cold   nothing cached anywhere, what the first user on a machine pays
  warm   the same variant run again against the cache the cold run left behind,
         which is what everyone afterwards pays

Every measurement runs in its own process. meTile persists both the compiled metallib
and the chosen schedule, so a variant sharing a process with an earlier one would
inherit a warm cache and report a warm-up near zero.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

VARIANTS = ("eager", "compile", "metile", "metile-autotune")
WORKLOADS = ("mlp", "add_rmsnorm", "attention")
_LABELS = {
    "eager": "eager MLX",
    "compile": "mx.compile",
    "metile": "meTile",
    "metile-autotune": "meTile + autotune",
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=8960)
    parser.add_argument("--rows", type=int, nargs="*", default=[1, 8])
    parser.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--row", type=int)
    return parser.parse_args()


def _build_mlp(arguments, variant, mx, nn):
    from metile.backends import mlx_dense_residual, mlx_dense_swiglu
    from metile.backends.mlx_dense import MLXDenseWeight

    hidden, intermediate, rows = arguments.hidden, arguments.intermediate, arguments.row
    gate = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    up = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    down = mx.random.normal((hidden, intermediate)).astype(mx.bfloat16)
    values = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    residual = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    mx.eval(gate, up, down, values, residual)

    def eager():
        hidden_state = nn.silu(values @ gate.T) * (values @ up.T)
        return hidden_state @ down.T + residual

    if not variant.startswith("metile"):
        return mx.compile(eager) if variant == "compile" else eager

    paired = mx.stack((gate, up), axis=-1)
    gate_dense = MLXDenseWeight.from_mlx(gate)
    up_dense = MLXDenseWeight.from_mlx(up)
    mx.eval(paired, gate_dense.native_weight, up_dense.native_weight)
    autotune = variant == "metile-autotune"

    def build():
        hidden_state = mlx_dense_swiglu.mlx_dense_swiglu(
            values, gate_dense, up_dense, paired_weight=paired, autotune=autotune
        )
        return mlx_dense_residual.mlx_dense_residual_qmv(
            hidden_state, down, residual, autotune=autotune
        )

    return build


def _build_add_rmsnorm(arguments, variant, mx, nn):
    from metile.backends.mlx import mlx_add_rms_norm

    hidden, rows = arguments.hidden, arguments.row
    values = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    residual = mx.random.normal((rows, hidden)).astype(mx.bfloat16)
    weight = mx.random.normal((hidden,)).astype(mx.bfloat16)
    mx.eval(values, residual, weight)

    def eager():
        summed = values + residual
        return mx.fast.rms_norm(summed, weight, 1e-6)

    if not variant.startswith("metile"):
        return mx.compile(eager) if variant == "compile" else eager

    autotune = variant == "metile-autotune"

    def build():
        return mlx_add_rms_norm(values, residual, weight, 1e-6, autotune=autotune)[1]

    return build


def _build_attention(arguments, variant, mx, nn):
    from metile.backends.mlx import mlx_attention_decode

    heads, dim, keys = 8, 64, 1024
    rows = arguments.row
    query = mx.random.normal((1, heads, rows, dim)).astype(mx.float32)
    key = mx.random.normal((1, heads, keys, dim)).astype(mx.float32)
    value = mx.random.normal((1, heads, keys, dim)).astype(mx.float32)
    mx.eval(query, key, value)
    scale = dim**-0.5

    def eager():
        return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)

    if not variant.startswith("metile"):
        return mx.compile(eager) if variant == "compile" else eager
    if rows != 1:
        return None  # the generated decode kernel is single-query only

    autotune = variant == "metile-autotune"

    def build():
        return mlx_attention_decode(query, key, value, scale=scale, autotune=autotune)

    return build


_BUILDERS = {
    "mlp": _build_mlp,
    "add_rmsnorm": _build_add_rmsnorm,
    "attention": _build_attention,
}


def _measure(arguments):
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(0)
    build = _BUILDERS[arguments.workload](arguments, arguments.variant, mx, nn)
    if build is None:
        print(json.dumps({"skipped": True}))
        return

    mx.synchronize()
    start = time.perf_counter_ns()
    mx.eval(build())
    first = (time.perf_counter_ns() - start) / 1e9

    for _ in range(5):
        mx.eval([build() for _ in range(arguments.inner)])
    mx.synchronize()
    samples = []
    for _ in range(arguments.rounds):
        began = time.perf_counter_ns()
        mx.eval([build() for _ in range(arguments.inner)])
        samples.append((time.perf_counter_ns() - began) / arguments.inner / 1e9)

    print(json.dumps({"first": first, "steady": statistics.median(samples)}))


def _run(arguments, workload, variant, rows, cache_dir):
    command = [
        sys.executable, __file__,
        "--variant", variant, "--workload", workload, "--row", str(rows),
        "--hidden", str(arguments.hidden), "--intermediate", str(arguments.intermediate),
        "--rounds", str(arguments.rounds), "--inner", str(arguments.inner),
    ]
    environment = dict(os.environ)
    environment["METILE_CACHE_DIR"] = cache_dir
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=_root, env=environment, check=True
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _interleaved_steady(arguments, workload, rows):
    """Time every variant in one process, alternating, so the ratios are comparable.

    Steady-state numbers must not be compared across processes: a small kernel is well
    inside the run-to-run spread, which is how an earlier version of this benchmark
    reported a 1.4x for two paths that dispatch to identical work.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(0)
    builds = {}
    for variant in VARIANTS:
        build = _BUILDERS[workload](
            argparse.Namespace(**{**vars(arguments), "row": rows}), variant, mx, nn
        )
        if build is not None:
            builds[variant] = build

    for build in builds.values():
        for _ in range(5):
            mx.eval([build() for _ in range(arguments.inner)])
    mx.synchronize()

    samples = {variant: [] for variant in builds}
    order = list(builds)
    for index in range(arguments.rounds):
        rotated = order[index % len(order):] + order[: index % len(order)]
        if index & 1:
            rotated.reverse()
        for variant in rotated:
            began = time.perf_counter_ns()
            mx.eval([builds[variant]() for _ in range(arguments.inner)])
            samples[variant].append((time.perf_counter_ns() - began) / arguments.inner / 1e9)
    return {variant: statistics.median(values) for variant, values in samples.items()}


def main():
    arguments = _arguments()
    if arguments.variant is not None:
        _measure(arguments)
        return

    print("Compilation cost, measured in a fresh process per variant.")
    print("Steady-state speed, measured in one process with the variants interleaved.")
    print(f"MLP {arguments.hidden} -> {arguments.intermediate} -> {arguments.hidden} BF16 · "
          "add+RMSNorm BF16 · decode attention 8 heads over 1024 keys FP32\n")

    for workload in arguments.workloads:
        for rows in arguments.rows:
            compile_cost = {}
            for variant in VARIANTS:
                cache_dir = tempfile.mkdtemp(prefix="metile-cold-")
                cold = _run(arguments, workload, variant, rows, cache_dir)
                if cold.get("skipped"):
                    continue
                warm = _run(arguments, workload, variant, rows, cache_dir)
                compile_cost[variant] = (
                    max(cold["first"] - cold["steady"], 0.0) * 1e3,
                    max(warm["first"] - warm["steady"], 0.0) * 1e3,
                )
            if not compile_cost:
                continue

            steady = _interleaved_steady(arguments, workload, rows)
            print(f"{workload}, rows={rows}")
            print(f"{'':>2}{'variant':<20}{'cold ms':>10}{'warm ms':>10}"
                  f"{'steady us':>11}{'vs eager':>10}{'vs compile':>12}")
            for variant, (cold_ms, warm_ms) in compile_cost.items():
                seconds = steady[variant]
                print(
                    f"{'':>2}{_LABELS[variant]:<20}{cold_ms:10.1f}{warm_ms:10.1f}"
                    f"{seconds * 1e6:11.1f}{steady['eager'] / seconds:9.3f}x"
                    f"{steady['compile'] / seconds:11.3f}x"
                )
            print()


if __name__ == "__main__":
    main()
