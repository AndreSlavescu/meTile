"""Check that a scheduling bound really keeps kernels clear of the register budget.

The reading itself lives in metile.target.agx, because a register budget is a property of
the target and the compiler should be able to consult it. This is the command line over it:
a self-check that pins the reader against known counts, and an audit that walks every
configuration the dense SwiGLU tuner is allowed to choose and reports the worst one.

The audit is the point. `_MAX_QMV_ACCUMULATOR_PAIRS` was chosen before there was any way to
see register pressure, and the only way to know whether a guess like that is still doing its
job is to measure what it admits.

usage:
    python benchmarks/agx_registers.py                 # audit the dense SwiGLU bound
    python benchmarks/agx_registers.py --self-check    # verify the reader itself
"""

import argparse
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from metile.target.agx import REGISTER_BUDGET, Unavailable, inspect


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path(".metile-agx"))
    return parser.parse_args()


def _pressure_kernel(accumulators):
    """A kernel holding `accumulators` float4 values live across a loop."""
    declare = "\n    ".join(f"float4 acc{i} = float4(x[gid + {i}]);" for i in range(accumulators))
    update = "\n        ".join(
        f"acc{i} = fma(acc{i}, v, acc{(i + 1) % accumulators});" for i in range(accumulators)
    )
    total = " + ".join(f"acc{i}" for i in range(accumulators))
    return f"""#include <metal_stdlib>
using namespace metal;

kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {{
    {declare}
    for (uint i = 0; i < n; ++i) {{
        float4 v = float4(x[i]);
        {update}
    }}
    float4 total = {total};
    out[gid] = total.x + total.y + total.z + total.w;
}}
"""


def self_check(workdir):
    """Register counts for known kernels, so a bad read fails loudly rather than plausibly."""
    print(f"{'live floats':>12}{'expected':>10}{'read':>7}")
    ok = True
    for accumulators, expected in ((2, 12), (8, 36), (24, 100), (30, 124)):
        result = inspect(_pressure_kernel(accumulators), "probe", workdir)
        good = result["registers"] == expected
        ok = ok and good
        print(
            f"{accumulators * 4:>12}{expected:>10}{result['registers']:>7}"
            f"{'  ok' if good else '  MISMATCH'}"
        )
    return 0 if ok else 1


def audit_dense_swiglu(workdir, reduction=1536, output_features=8960):
    """Report the worst register count among configurations the tuner may select."""
    from metile.backends import mlx_dense_swiglu as backend
    from metile.codegen.msl_emitter import emit
    from metile.compiler.dense import lower_dense_swiglu_qmv

    print(f"dense SwiGLU QMV {reduction} -> {output_features}, budget {REGISTER_BUDGET}")
    print(f"{'rows':>5}{'configs':>9}{'max registers':>15}{'% of budget':>13}")
    worst = 0
    for rows in (1, 2, 4, 8, 16):
        configs = [
            config
            for config in backend._candidate_configs(
                rows, reduction, output_features, paired_available=True
            )
            if config.algorithm == "metile" and config.implementation.startswith("simdgroup")
        ]
        counts = []
        for config in configs:
            name = (
                f"audit_{rows}_{config.outputs_per_simdgroup}"
                f"_{config.simdgroups_per_threadgroup}_{config.k_unroll}"
            )
            metal_ir = lower_dense_swiglu_qmv(
                name,
                output_features,
                reduction,
                outputs_per_simdgroup=config.outputs_per_simdgroup,
                simdgroups_per_threadgroup=config.simdgroups_per_threadgroup,
                interleaved=True,
                k_unroll=config.k_unroll,
                rows=rows,
            )
            source = backend._specialize_mlx_source(emit(metal_ir), "bfloat16")
            try:
                counts.append(inspect(source, name, workdir)["registers"])
            except RuntimeError:
                continue
        peak = max(counts) if counts else 0
        worst = max(worst, peak)
        print(f"{rows:>5}{len(configs):>9}{peak:>15}{peak / REGISTER_BUDGET * 100:>12.0f}%")
    print(f"\nworst admitted kernel uses {worst} of {REGISTER_BUDGET} registers")
    print("spilling" if worst >= REGISTER_BUDGET else "no admitted kernel spills")
    return 0


def main():
    arguments = _arguments()
    try:
        if arguments.self_check:
            return self_check(arguments.workdir)
        return audit_dense_swiglu(arguments.workdir)
    except Unavailable as error:
        print(f"unavailable: {error}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
