"""Optimising G17 machine code directly, where reordering is not undone.

meTile's other scheduler works on Metal IR and is measured to be inert: Apple's backend rebuilds
the schedule from the dataflow, so two source orders compile to byte-identical instructions and
statement order is a suggestion it declines. That is not a failure of the pass, it is where the
boundary of control sits when the output is MSL.

This pass is on the other side of that boundary. It reads the instructions the backend produced,
rewrites them, and puts them back, so nothing downstream gets to re-derive anything. It is the
same two jobs as the IR pass — dependence-preserving reordering, and the arithmetic rewriting
that instruction-level parallelism needs — done where they survive.

Register dependences are what make it possible at all. `metile.target.agx_isa` establishes that a
compact fma names its register in byte 0's high nibble and again as `(r << 1) | 1` in byte 1, so
which instructions actually depend on each other is readable rather than guessed, and two fmas on
different registers are known to be independent.

Two transformations, both bit-exact by construction, because neither changes any arithmetic:

    simplify    retire instructions that compute nothing. `a * 1` with no addend is the identity,
                and the flag that retires an instruction was verified to be indistinguishable
                from nopping it.
    reorder     move independent instructions, preserving every register dependence.

Instruction-level parallelism is the one job this cannot finish here, and the blocker has moved
once already, so it is worth stating precisely.

It is no longer the combine. `rd * 1 + rs` is a register-plus-register add, verified across
eighteen register pairs, so summing two partial chains is expressible. What is missing is a
register to put the second chain in. Splitting a chain needs one that nothing else uses, and
proving that requires reading every instruction in the kernel — a register touched only by an
instruction this file cannot decode is indistinguishable from a free one. Instruction lengths
cannot be recovered in general, so the stream cannot be walked, so no register can be shown free.
Guessing would corrupt whatever already lived there, and silently.

So the ILP half lives in `metile.compiler.scheduling` at the IR level, where reassociation needs
neither a new instruction nor a new register. Both halves are reached through `optimize` below.

Scope is deliberately narrow and self-enforcing. Only the compact f32 fma is decoded, and every
byte that is not part of one decoded instruction is a barrier nothing crosses. That is not
caution for its own sake: the instruction stream cannot be walked reliably, since the length
field theory that would allow it was measured wrong on eight of eight kernels, so an undecoded
region might be one instruction or twenty and moving anything across it is unsound.
"""

from metile.target import agx_isa

# The bit the compiler sets on every instruction of a run but the last. Reordering has to maintain
# it: it is positional, not a property of the operation, so moving an instruction to the end of a
# run without clearing it, or away from the end without setting it, changes the code's meaning
# rather than its order.
_CONTINUES = agx_isa.FmaFlag(2, 0x20, "more instructions follow in this run", "not the last")


class Fma:
    """One decoded compact fma: `register = register * multiplier + addend`.

    The addend is one of three things, because the slot is: `addend` holds an immediate,
    `addend_register` names a register, and a register index at or above the reachable range reads
    zero, which is the only way a zero addend is expressible at all.
    """

    def __init__(self, offset, register, multiplier, addend, addend_register, negate_product, last):
        self.offset = offset
        self.register = register
        self.multiplier = multiplier
        self.addend = addend
        self.addend_register = addend_register
        self.negate_product = negate_product
        self.last = last

    def addend_is_zero(self):
        """Whether the addend contributes nothing, which needs the register range to decide."""
        return (
            self.addend is None
            and self.addend_register is not None
            and self.addend_register >= agx_isa.ARCHITECTURAL_REGISTERS
        )

    def is_identity(self):
        """Whether this instruction leaves its register unchanged.

        `a * 1` plus something that reads zero. Deciding it needs the addend's register index, not
        merely its absence: an fma whose addend slot names a live register adds that register, and
        retiring it would drop a real term. An earlier version checked for `a * 1 + 0` as an
        immediate, which cannot exist — the field holds `(1 + m/8) * 2**(e - 11)`, whose smallest
        value is 2**-11, so it has no encoding for zero.

        A negated product is not the identity even at these constants, because it flips the sign.
        """
        return not self.negate_product and self.multiplier == 1.0 and self.addend_is_zero()

    def encode(self, last=None):
        register = self.addend_register
        if self.addend is not None or self.addend_is_zero():
            register = None
        return agx_isa.encode_fma(
            self.register,
            self.multiplier,
            self.addend,
            last=self.last if last is None else last,
            negate_product=self.negate_product,
            addend_register=register,
        )

    def __repr__(self):
        if self.addend is not None:
            addend = f" + {self.addend:g}"
        elif self.addend_is_zero():
            addend = ""
        else:
            addend = f" + r{self.addend_register}"
        sign = "-" if self.negate_product else ""
        return f"Fma(0x{self.offset:04x}: r{self.register} = {sign}r{self.register} * {self.multiplier:g}{addend})"


def decode(text, offsets):
    """Decode the compact fmas at `offsets`, which must have been confirmed behaviourally.

    Offsets are required rather than discovered. Pattern matching proposes candidates and is
    sometimes right, but a wrong offset here does not fail loudly: it decodes some other
    instruction's bytes as an fma and re-encoding them corrupts the kernel silently. Callers get
    their offsets from `agx_isa.boundaries`, which confirms each one by checking that nopping it
    removes exactly one operation's contribution.
    """
    found = []
    for offset in offsets:
        window = text[offset : offset + agx_isa.FMA_LENGTH]
        if len(window) < agx_isa.FMA_LENGTH:
            raise ValueError(f"offset 0x{offset:04x} runs past the end of the code")
        if window[0] & 0x0F != agx_isa.FMA_OPCODE_NIBBLE:
            raise ValueError(f"offset 0x{offset:04x} is not a compact fma")
        register = window[0] >> 4
        if window[1] != (register << 1) | 1:
            raise ValueError(
                f"offset 0x{offset:04x} disagrees with itself about its register: "
                f"byte 0 says {register}, byte 1 says {window[1] >> 1}"
            )
        addend = None
        addend_register = None
        if agx_isa.read_flag(window, 0, agx_isa.ADDEND_IMMEDIATE):
            addend = agx_isa.decode_immediate(window[agx_isa.FMA_ADDEND_BYTE])
            if agx_isa.read_flag(window, 0, agx_isa.ADDEND_NEGATE):
                addend = -addend
        else:
            addend_register = window[agx_isa.FMA_ADDEND_BYTE] >> 1
        found.append(
            Fma(
                offset=offset,
                register=register,
                multiplier=agx_isa.decode_immediate(window[agx_isa.FMA_MULTIPLIER_BYTE]),
                addend=addend,
                addend_register=addend_register,
                negate_product=agx_isa.read_flag(window, 0, agx_isa.PRODUCT_NEGATE),
                last=not agx_isa.read_flag(window, 0, _CONTINUES),
            )
        )
    return found


def _runs(instructions):
    """Group instructions into contiguous runs, split wherever undecoded bytes intervene.

    Two decoded fmas are in the same run only when they are adjacent in the code. Anything
    between them is unidentified, and since instruction lengths cannot be recovered in general
    there is no way to know what it does, so it bounds the region a reordering may touch.
    """
    groups = []
    current = []
    for instruction in instructions:
        if current and instruction.offset != current[-1].offset + agx_isa.FMA_LENGTH:
            groups.append(current)
            current = []
        current.append(instruction)
    if current:
        groups.append(current)
    return groups


def simplify(text, offsets):
    """Retire instructions that compute nothing. Returns (code, count).

    Uses the disable flag rather than nops so lengths and every neighbouring byte stay exactly as
    they were; the flag was verified to produce results indistinguishable from nopping.
    """
    patched = text
    retired = 0
    for instruction in decode(text, offsets):
        if not instruction.is_identity():
            continue
        patched = agx_isa.write_flag(patched, instruction.offset, agx_isa.INSTRUCTION_DISABLE, True)
        retired += 1
    return patched, retired


def reorder(text, offsets):
    """Reorder independent instructions within each run. Returns (code, moved count).

    Bit-exact by construction: instructions are moved, never rewritten, so every register still
    sees the same operations in an order that respects every dependence. Two fmas touching
    different registers are independent, and two touching the same one are not, which is the whole
    dependence relation for this instruction form — it both reads and writes exactly one register.

    Ties break towards the original position, so a run with no freedom comes back byte-identical
    rather than churned into an equivalent ordering.
    """
    instructions = decode(text, offsets)
    patched = bytearray(text)
    moved = 0

    for run in _runs(instructions):
        # Stable grouping by register: a register's own instructions keep their relative order,
        # which is what preserves the dependences, while whole registers may interleave. Emitting
        # one register's chain at a time is the schedule that shortens no dependence but also
        # breaks none, and it is the only reordering this form permits without renaming.
        by_register = {}
        for instruction in run:
            by_register.setdefault(instruction.register, []).append(instruction)
        if len(by_register) < 2:
            continue

        ordered = []
        while by_register:
            for register in sorted(by_register):
                ordered.append(by_register[register].pop(0))
            by_register = {register: rest for register, rest in by_register.items() if rest}

        for position, instruction in enumerate(ordered):
            offset = run[position].offset
            if instruction.offset != offset:
                moved += 1
            patched[offset : offset + agx_isa.FMA_LENGTH] = instruction.encode(
                last=(position == len(ordered) - 1 and run[-1].last)
            )
    return bytes(patched), moved


def optimize(text, offsets, simplify_identities=True, reorder_independent=True):
    """Run the machine-level passes in order. Returns (code, report).

    The single entry point, so a caller asks for optimisation rather than for a list of
    transformations. Simplification runs first: an instruction it retires is one reordering does
    not have to place, and retiring changes no offsets, so the second pass sees the same layout.

    The IR-level half of meTile's optimisation, including the instruction-level parallelism this
    level cannot express, is `metile.compiler.scheduling`. Both are native to the compiler; they
    differ in which side of the MSL boundary they act on, and only this side survives.
    """
    report = {"retired": 0, "moved": 0}
    if simplify_identities:
        text, report["retired"] = simplify(text, offsets)
    if reorder_independent:
        text, report["moved"] = reorder(text, offsets)
    return text, report


def summarise(text, offsets):
    """What the pass can see and what it would do, without doing it.

    For reporting and for deciding whether a kernel is worth rewriting at all. Reports the runs
    it found rather than a single count, because a run of one instruction offers nothing to
    reorder however many such runs there are.
    """
    instructions = decode(text, offsets)
    runs = _runs(instructions)
    registers = {instruction.register for instruction in instructions}
    return {
        "instructions": len(instructions),
        "runs": [len(run) for run in runs],
        "registers": sorted(registers),
        "identities": sum(1 for instruction in instructions if instruction.is_identity()),
        "reorderable": sum(len(run) for run in runs if len({i.register for i in run}) > 1),
    }
