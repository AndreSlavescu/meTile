"""How much of what meTile writes in MSL survives to the machine code.

A timing harness answers "is this faster" badly and slowly. Comparing the compiled `__text`
answers a sharper question outright: two source forms that produce identical bytes cannot differ
in speed, no measurement required. This asks that question of every code-generation choice a
compiler above MSL is in a position to make.

The answers, on M5 with the current toolchain, are the reason meTile's scheduling pass is off:

  statement order        byte-identical. Two independent fma chains written serially and written
                         interleaved compile to the same 190 bytes; a load at its use and the
                         same load hoisted compile to the same 218.

  reassociation          byte-identical. A serial addition chain and a balanced tree over the
                         same eight terms compile to the same 282 bytes.

  live-range shape       within one register, identical code size, up to 64 live values.

So nothing expressible in MSL moves the generated code in a way that could matter. That is not a
verdict on the passes; it is where the boundary of our control sits. The leverage above MSL is in
which algorithm, which tiling and which functional unit, and those are worth 2.4x to 3.7x where
scheduling is worth 1.09x.

Re-run this on a new toolchain. If a row starts reporting DIFFERS, code generation stopped being
normalised and the scheduling pass is worth turning back on.
"""

import argparse
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from metile.target import agx

HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
"""


def _kernel(body):
    return HEAD + body + "\n}\n"


def _chains(interleaved):
    """Two independent fma chains, written serially or interleaved."""
    steps = (
        ["a = fma(a, c, d); b = fma(b, c, d);"] * 4
        if interleaved
        else ["a = fma(a, c, d);"] * 4 + ["b = fma(b, c, d);"] * 4
    )
    return _kernel(
        "    float a = x[gid], b = x[gid + 1];\n"
        "    float c = float(n), d = float(n) + 1.0f;\n    "
        + "\n    ".join(steps)
        + "\n    out[gid] = a + b;"
    )


def _loads(hoisted):
    """Two loads per iteration, each placed at its use or both hoisted."""
    body = (
        "        float p = x[i];\n        float q = x[i + 64];\n"
        "        t = fma(t, p, p);\n        t = fma(t, q, q);"
        if hoisted
        else "        float p = x[i];\n        t = fma(t, p, p);\n"
        "        float q = x[i + 64];\n        t = fma(t, q, q);"
    )
    return _kernel(
        "    float t = 0.0f;\n    for (uint i = 0; i < n; ++i) {\n" + body + "\n    }\n"
        "    out[gid] = t;"
    )


def _association(tree):
    terms = "\n    ".join(f"float v{i} = x[gid + {i}];" for i in range(8))
    total = (
        "((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))"
        if tree
        else "((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7"
    )
    return _kernel(f"    {terms}\n    out[gid] = {total};")


def _pressure(count, streamed):
    if streamed:
        steps = "\n".join(
            f"    {{ float a = x[gid + {i}]; float b = x[gid + {i + 1}]; t += a * b; }}"
            for i in range(0, count, 2)
        )
        return _kernel(f"    float t = 0.0f;\n{steps}\n    out[gid] = t;")
    loads = "\n    ".join(f"float v{i} = x[gid + {i}];" for i in range(count))
    products = " + ".join(f"v{i} * v{i + 1}" for i in range(0, count, 2))
    return _kernel(f"    {loads}\n    out[gid] = {products};")


EXPERIMENTS = (
    ("statement order: fma chains serial vs interleaved", _chains(False), _chains(True)),
    ("statement order: loads at use vs hoisted", _loads(False), _loads(True)),
    ("reassociation: addition chain vs balanced tree", _association(False), _association(True)),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".metile-agx"))
    return parser.parse_args()


def main():
    arguments = _arguments()
    try:
        agx.machine_code(_chains(False), "probe", arguments.workdir)
    except agx.Unavailable as error:
        print(f"cannot read compiled kernels: {error}")
        return 1

    print("What survives from MSL to machine code, on this toolchain\n")
    normalised = 0
    for label, left, right in EXPERIMENTS:
        first = agx.machine_code(left, "probe", arguments.workdir)
        second = agx.machine_code(right, "probe", arguments.workdir)
        identical = first == second
        normalised += identical
        verdict = "IDENTICAL" if identical else "DIFFERS"
        detail = ""
        if not identical and len(first) == len(second):
            positions = sum(1 for a, b in zip(first, second) if a != b)
            detail = f", {positions} byte positions"
        print(f"  {label}")
        print(f"      {len(first)}B vs {len(second)}B -> {verdict}{detail}")

    print("\n  live-range shape: values held live vs consumed as they arrive")
    print(f"      {'live':>6}{'held':>16}{'streamed':>18}")
    for count in (8, 16, 32, 64):
        held = agx.inspect(_pressure(count, False), "probe", arguments.workdir)
        streamed = agx.inspect(_pressure(count, True), "probe", arguments.workdir)
        left = "{} regs {}B".format(held["registers"], held["text_bytes"])
        right = "{} regs {}B".format(streamed["registers"], streamed["text_bytes"])
        print(f"      {count:>6}{left:>16}{right:>18}")

    print(
        f"\n{normalised} of {len(EXPERIMENTS)} code-generation choices are normalised away by the"
        " backend."
    )
    if normalised == len(EXPERIMENTS):
        print("Nothing meTile can express in MSL changes these instructions. Effort above MSL")
        print("belongs in algorithm, tiling and functional-unit choice, not in code generation.")
    else:
        print("A choice stopped being normalised. metile/compiler/scheduling.py is off by")
        print("default on the assumption that they all are, and that assumption needs revisiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
