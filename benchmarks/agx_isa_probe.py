"""Re-derive what meTile knows about G17 machine code, from nothing, on this machine.

Everything in `metile.target.agx_isa` came out of this procedure, and running it is how to port
that knowledge to different hardware or a different toolchain. Nothing is read from a table; each
stage measures, and the last stage checks the result by predicting GPU output in advance and
comparing.

    1  edit               patch a byte range to nops and confirm the edit runs, which is the
                          premise for everything after it
    2  boundaries         find where instructions actually start, behaviourally
    3  immediates         read the constant field out of kernels compiled with known constants
    4  encode             synthesise constants the compiler never emitted and predict the answer
    5  flags              set one arithmetic bit at a time and predict the answer again

Stage 4 is the one that matters. Stages 1 to 3 could all be satisfied by a field map that is
merely consistent with what the compiler happens to emit; only predicting the result of bytes
nobody has compiled shows the encoding is understood.

usage:
    python benchmarks/agx_isa_probe.py
    python benchmarks/agx_isa_probe.py --verbose   # show every alignment the scan rejected
"""

import argparse
import itertools
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

from metile.target import agx, agx_isa

# Four dependent fmas of a*2+1. Dependent so the backend cannot reorder them, and with distinct
# results per step so a removed instruction is visible in the output rather than absorbed.
CHAIN = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float* x [[buffer(0)]],
                  device float* out     [[buffer(1)]],
                  constant uint& n      [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    float a = x[gid];
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    a = fma(a, 2.0f, 1.0f);
    out[gid] = a;
}
"""
INPUTS = [1.0, 2.0, 3.0, 4.0]
INTACT = 31.0  # ((((1*2+1)*2+1)*2+1)*2+1)
ONE_REMOVED = 15.0  # three fmas instead of four


# Rebuilt from a template rather than by substituting into CHAIN: replacing "2.0f" and then
# "1.0f" corrupts the source whenever one constant is textually the other.
def _constants(multiplier, addend):
    step = f"    a = fma(a, {multiplier}f, {addend}f);"
    return CHAIN.replace("    a = fma(a, 2.0f, 1.0f);", step)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path(".metile-agx"))
    return parser.parse_args()


def main():
    arguments = _arguments()
    work = arguments.workdir

    try:
        text = agx.machine_code(CHAIN, "probe", work)
    except agx.Unavailable as error:
        print(f"cannot read compiled kernels: {error}")
        return 1

    print(f"Kernel is {len(text)} bytes of machine code.")
    nops = text.count(agx_isa.NOP)
    print(
        f"Contains {nops} occurrences of the two-byte nop and "
        f"{text.count(agx_isa.BLOCK_END)} block terminators; "
        f"code resumes at each {agx_isa.BLOCK_ALIGNMENT}-byte boundary.\n"
    )

    print("1. Does an edited archive actually run?")
    baseline = agx_isa.execute(CHAIN, "probe", INPUTS, workdir=work)
    print(f"   unedited: x=1 -> {baseline[0]:g}, expected {INTACT:g}")
    if baseline[0] != INTACT:
        print("   the kernel does not compute what this probe assumes; stopping")
        return 1

    print("\n2. Where do instructions start? Nop out eight bytes at every even offset.")
    print(f"   {INTACT:g} means nothing was removed, {ONE_REMOVED:g} means exactly one fma was.")
    region = (0x50, len(text) - 12, agx_isa.FMA_LENGTH)
    offsets, _ = agx_isa.boundaries(
        CHAIN, "probe", region, INPUTS, INTACT, ONE_REMOVED, workdir=work
    )
    print(f"   instruction starts: {[hex(offset) for offset in offsets]}")
    strides = {b - a for a, b in itertools.pairwise(offsets)}
    print(f"   stride between them: {strides or 'n/a'}")

    # A confirmed boundary is not the same as a form whose fields are known. All four fmas sit on
    # an eight-byte stride, but the compiler encodes the first differently from the rest, and
    # reading the constant field out of that one yields nonsense. Being a boundary is measured;
    # being the mapped form is what the opcode nibble decides.
    compact = [offset for offset in offsets if text[offset] & 0x0F == agx_isa.FMA_OPCODE_NIBBLE]
    for offset in offsets:
        form = "compact, fields mapped" if offset in compact else "other form, not mapped"
        print(f"     0x{offset:04x}  {bytes(text[offset : offset + 8]).hex(' ')}   {form}")
    if arguments.verbose:
        candidates = agx_isa.find_fma(text)
        print(f"   pattern match alone would have proposed: {[hex(c) for c in candidates]}")
    if not compact:
        print("   no instruction in the mapped form; the remaining stages depend on one")
        return 1

    print("\n3. What does the constant field look like? Compile known constants and read it.")
    print(f"   {'multiplier':>11}{'addend':>8}{'byte 3':>9}{'byte 5':>9}   decoded")
    observed = []
    for multiplier, addend in (
        (2.0, 1.0),
        (3.0, 1.0),
        (4.0, 1.0),
        (8.0, 1.0),
        (2.0, 3.0),
        (2.0, 5.0),
    ):
        variant = agx.machine_code(_constants(multiplier, addend), "probe", work)
        instruction = variant[compact[0] : compact[0] + agx_isa.FMA_LENGTH]
        mul_byte = instruction[agx_isa.FMA_MULTIPLIER_BYTE]
        add_byte = instruction[agx_isa.FMA_ADDEND_BYTE]
        pair = (agx_isa.decode_immediate(mul_byte), agx_isa.decode_immediate(add_byte))
        agree = pair == (multiplier, addend)
        observed.append(agree)
        print(
            f"   {multiplier:>11}{addend:>8}{f'0x{mul_byte:02x}':>9}{f'0x{add_byte:02x}':>9}"
            f"   {pair[0]:g}, {pair[1]:g} {'ok' if agree else 'MISREAD'}"
        )
    print(f"   the derived format reads {sum(observed)} of {len(observed)} correctly")

    print("\n4. Can constants the compiler never emitted be synthesised? Predict, then run.")
    print(f"   {'rewrite':<26}{'predicted':>12}{'measured':>12}   verdict")
    checks = []
    for multiplier, addend, count in (
        (6.0, 7.0, 1),
        (6.0, 7.0, len(compact)),
        (1.25, 1.5, len(compact)),
    ):
        chosen = compact[:count]

        def rewrite(original, m=multiplier, a=addend, where=tuple(chosen)):
            patched = original
            for offset in where:
                patched = agx_isa.rewrite_fma_immediates(patched, offset, m, a)
            return patched

        # Predict by replaying the arithmetic, applying the new constants at exactly the steps
        # that were patched. Which steps those are has to come from the offsets, not from a
        # count: the compact instructions are the second, third and fourth fma here, so
        # assuming the patched ones are the trailing ones predicted 97 where the GPU said 103.
        rewritten = {offsets.index(offset) for offset in chosen}
        value = 1.0
        for step in range(len(offsets)):
            if step in rewritten:
                value = value * multiplier + addend
            else:
                value = value * 2.0 + 1.0
        got = agx_isa.execute(CHAIN, "probe", INPUTS, rewrite=rewrite, workdir=work)[0]
        agree = got == value
        checks.append(agree)
        label = f"{count} fma -> a*{multiplier:g}+{addend:g}"
        print(f"   {label:<26}{value:>12g}{got:>12g}   {'MATCHES' if agree else 'differs'}")

    print("\n5. Do the arithmetic flags mean what they claim? One bit at a time, four inputs.")
    print(f"   {'flag':<30}{'predicted':>28}   verdict")
    inputs = [1.0, 2.0, 3.0, 5.0]
    for flag, clear, step, label in (
        (agx_isa.PRODUCT_NEGATE, False, lambda v: -v * 2.0 + 1.0, "product negated: -a*m+d"),
        (agx_isa.ADDEND_NEGATE, False, lambda v: v * 2.0 - 1.0, "addend negated: a*m-d"),
        (agx_isa.ADDEND_ENABLE, True, lambda v: v * 2.0, "addend dropped: a*m"),
    ):

        def rewrite(original, f=flag, c=clear, where=tuple(compact)):
            patched = original
            for offset in where:
                patched = agx_isa.write_flag(patched, offset, f, not c)
            return patched

        predicted = []
        for value in inputs:
            running = value * 2.0 + 1.0
            for _ in compact:
                running = step(running)
            predicted.append(running)
        got = agx_isa.execute(CHAIN, "probe", inputs, rewrite=rewrite, workdir=work)
        agree = got == predicted
        checks.append(agree)
        shown = ", ".join(f"{value:g}" for value in predicted)
        print(f"   {label:<30}{shown:>28}   {'MATCHES' if agree else f'got {got}'}")

    print()
    if all(checks) and all(observed):
        print("Both the constant field and the arithmetic flags are encodable, not merely")
        print("readable: every prediction made before running matched the GPU exactly, for")
        print("bytes no Metal compiler produced.")
    else:
        print("A prediction missed. The field map in metile/target/agx_isa.py is wrong here,")
        print("or this toolchain encodes constants differently; re-derive before relying on it.")
    return 0 if all(checks) and all(observed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
