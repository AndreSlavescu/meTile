"""Instruction scheduling and reassociation over Metal IR.

Metal's own backend schedules whatever MSL it is handed, and it is good at it. This pass is
not an attempt to beat it at that. It exists because the backend optimises a program it
receives after our own choices have already fixed the thing that matters most on this target,
and two measurements say which thing that is:

    reaching the register budget    1.3x to 6.7x slower  (metile.target.agx.REGISTER_BUDGET)
    perfect instruction-level ILP   1.09x fp32, 1.41x fp16  (agx.ILP_CEILING)

The spill cliff is worth between one and six times as much as the entire ILP prize, and it is
reached by an order that keeps too many values alive at once. So this scheduler's objective is
register pressure first and latency second, which is the opposite of what a scheduler for a
latency-bound machine would do. It is the standard integrated-prepass idea: schedule for
latency while pressure is comfortable, switch to relieving pressure once it is not.

Two passes live here, and they differ in a way worth stating plainly:

    reorder_for_latency        reorders whole operations, changes no arithmetic, so results
                               are bit-identical. Safe to run always.
    interleave_reduction_chains  reassociates floating-point addition, so results change in
                               the last bits. Off by default, and it should stay off unless a
                               caller has a reason: the most it can win is bounded by
                               ILP_CEILING above, and the test suite asserts bit-exact logits
                               against MLX.

That trade is the whole reason the second pass is not simply enabled. Reassociating buys at
most 9% on fp32, and it costs the property that a model's logits match MLX exactly.

Both are off by default in the pipeline, and the measurement that decided it for the reordering
pass is worth stating here rather than leaving to be rediscovered. Across six kernels from 14 to
126 registers, reordering changed the allocated register count in **none** of them, and no
timing difference survived a control that compares byte-identical MSL against itself. The cause
is structural: Metal's backend schedules and allocates from the MSL it is given, so statement
order is advice it is free to ignore, and on this evidence it does. The objective below is the
right objective and it is simply not reachable from here.

What that leaves is infrastructure rather than speed. The passes are correct, they are tested
against the hazards that would make them miscompile, and they operate at the level where a
scheduler has to operate if meTile ever emits machine code directly instead of MSL, which the
binary-archive injection work established is possible. Measured against MSL, they are inert;
that is a fact about where the boundary of our control is, not about the code.
"""

import dataclasses

from metile.ir import metal_ir as mir
from metile.ir.types import PtrType, ScalarType
from metile.target import agx

# Operations whose memory and register effects this file models. Anything else is treated as
# a scheduling fence: nothing moves across it.
#
# The default matters more than the list. Metal IR has around seventy operations, several of
# them macro-operations that expand to loops with their own barriers, and one that emits a
# barrier only when a private attribute is set on it. A whitelist that is wrong is a
# miscompile; a fence that is unnecessary only forgoes a reordering. So a new operation is
# immovable until someone classifies it here on purpose.
_REGISTER_ONLY = (
    mir.MConstant,
    mir.MBinOp,
    mir.MCast,
    mir.MUnary,
    mir.MSelect,
    mir.MCompare,
    mir.MPointerOffset,
    mir.ThreadPositionInGrid,
    mir.ThreadgroupPositionInGrid,
    mir.ThreadPositionInThreadgroup,
    mir.MSimdgroupId,
    mir.MThreadInSimdgroup,
    # Cross-lane, but they read and write registers only. Reordering two of them inside one
    # fence-free run cannot change which lanes are active, because control flow is a fence.
    mir.MSimdShuffleXor,
    mir.MSimdBroadcast,
)
_MOVABLE = (
    *_REGISTER_ONLY,
    mir.DeviceLoad,
    mir.DeviceStore,
    mir.MThreadgroupLoad,
    mir.MThreadgroupStore,
    mir.MVarDecl,
    mir.MVarAssign,
)

# Relative priority weights, deliberately not called cycles. Nothing here was measured, and
# nothing here can affect correctness: the weights choose which of several already-legal
# operations issues first. A device load outranks arithmetic because its result is the thing
# worth starting early; the exact ratio only has to order them.
_LATENCY = {
    mir.DeviceLoad: 64,
    mir.MThreadgroupLoad: 8,
    mir.MSimdShuffleXor: 4,
    mir.MSimdBroadcast: 4,
}
_DEFAULT_LATENCY = 1

# Fraction of the register budget at which the scheduler stops chasing latency and starts
# protecting live ranges. Below it, lengthening a live range is free. Above it, the next
# lengthened range is the one that spills, and a spill costs more than every reordering in
# this file can win.
PRESSURE_THRESHOLD = 0.8


def _fields(op):
    """Operand fields of an op, skipping the result."""
    for field in dataclasses.fields(op):
        if field.name == "result":
            continue
        yield field.name, getattr(op, field.name, None)


def _key(value):
    """The identity a dependence must be keyed on: the emitted name.

    Two things make this neither the object nor the raw name. The lowering hands out several
    distinct MValue objects carrying one name, so object identity splits one variable into
    many. CSE does the reverse, forwarding a redundant value to its equivalent without
    renaming it, so the raw name splits one variable into many the other way. Both mistakes
    lose dependence edges, and a lost edge is a use scheduled before its definition.

    Both were live here. Keyed on `id`, the attention kernel emitted uses of `v15` before
    declaring it. Keyed on the raw name it still did, because the operand object was called
    `v17` and CSE had forwarded it to `v15` -- the graph saw a value nothing in the block
    defined, so nothing ordered it. `mir.resolve` is the same rule the emitter applies when it
    prints a name, which is what makes agreeing with it sufficient.
    """
    return mir.resolve(value).name


def _operands(op):
    """Every MValue this operation reads."""
    found = []
    for _, value in _fields(op):
        if isinstance(value, mir.MValue):
            found.append(value)
        elif isinstance(value, list):
            found.extend(item for item in value if isinstance(item, mir.MValue))
    return found


def _register_cost(value):
    """Registers a value occupies, as a count the pressure model can add up.

    An estimate, and it says so. The point is not to predict the backend's allocation but to
    make two candidate orders comparable, and for that the only thing that must be right is
    the direction: keeping more values alive costs more.
    """
    if value is None:
        return 0
    if isinstance(value.type, PtrType):
        return 2
    if isinstance(value.type, ScalarType):
        return 1
    return 1


def _bodies(op):
    """Nested operation lists inside a control-flow operation."""
    found = []
    for _, value in _fields(op):
        if isinstance(value, list) and value and all(isinstance(item, mir.MOp) for item in value):
            found.append(value)
    return found


def _reads_everywhere(ops):
    """Every MValue read anywhere in these operations, nested bodies included."""
    seen = set()

    def walk(items):
        for op in items:
            for value in _operands(op):
                seen.add(_key(value))
            for body in _bodies(op):
                walk(body)

    walk(ops)
    return seen


def _runs(ops):
    """Split into maximal fence-free runs, as (start, stop) index pairs.

    Runs keep their relative order, so a value defined in one and read in a later one stays
    defined first without the dependence graph having to say so. That is what makes it safe
    to reason about one run at a time.
    """
    found = []
    start = 0
    for index, op in enumerate(ops):
        if not isinstance(op, _MOVABLE):
            if index > start:
                found.append((start, index))
            start = index + 1
    if len(ops) > start:
        found.append((start, len(ops)))
    return found


def _dependences(ops):
    """predecessors[i] = indices in `ops` that must issue before ops[i].

    Four kinds of edge, and the conservative choice is taken at every point where meTile
    cannot prove independence:

      values     a read follows the operation that defines it.
      variables  MVarDecl and MVarAssign write a name; a read of that name is an MValue with
                 no defining operation. Read-after-write, write-after-read and
                 write-after-write are all ordered.
      threadgroup  ordered per array name, which the IR always states literally.
      device     ordered as one bucket, not per pointer. MPointerOffset manufactures aliases
                 of a parameter, so two different pointer values can name the same memory and
                 keying on the pointer would be wrong.
    """
    predecessors = [set() for _ in ops]
    produced = {}
    var_write = {}
    var_reads = {}
    tg_write = {}
    tg_reads = {}
    device_write = None
    device_reads = []

    for index, op in enumerate(ops):
        for value in _operands(op):
            source = produced.get(_key(value))
            if source is not None:
                predecessors[index].add(source)

        names = {_key(value) for value in _operands(op) if value.defining_op is None}
        for name in names & var_write.keys():
            predecessors[index].add(var_write[name])
            var_reads.setdefault(name, []).append(index)

        if isinstance(op, mir.MVarDecl | mir.MVarAssign):
            if op.var_name in var_write:
                predecessors[index].add(var_write[op.var_name])
            predecessors[index].update(var_reads.pop(op.var_name, []))
            var_write[op.var_name] = index

        if isinstance(op, mir.MThreadgroupLoad):
            if op.array_name in tg_write:
                predecessors[index].add(tg_write[op.array_name])
            tg_reads.setdefault(op.array_name, []).append(index)
        elif isinstance(op, mir.MThreadgroupStore):
            if op.array_name in tg_write:
                predecessors[index].add(tg_write[op.array_name])
            predecessors[index].update(tg_reads.pop(op.array_name, []))
            tg_write[op.array_name] = index

        if isinstance(op, mir.DeviceLoad):
            if device_write is not None:
                predecessors[index].add(device_write)
            device_reads.append(index)
        elif isinstance(op, mir.DeviceStore):
            if device_write is not None:
                predecessors[index].add(device_write)
            predecessors[index].update(device_reads)
            device_reads = []
            device_write = index

        if op.result is not None:
            produced[_key(op.result)] = index

    return predecessors


def _heights(ops, predecessors):
    """Longest weighted path from each operation to the end of the run."""
    successors = [[] for _ in ops]
    for index, preds in enumerate(predecessors):
        for pred in preds:
            successors[pred].append(index)
    height = [0] * len(ops)
    for index in reversed(range(len(ops))):
        weight = _LATENCY.get(type(ops[index]), _DEFAULT_LATENCY)
        height[index] = weight + max((height[s] for s in successors[index]), default=0)
    return height


def _list_schedule(ops, predecessors, live_after, budget):
    """Order one run: latency first while pressure is comfortable, pressure first when not.

    Returns indices into `ops`. Ties break towards the original position, so a run with no
    scheduling freedom comes back byte-identical rather than churned into an equivalent but
    different order.
    """
    count = len(ops)
    successors = [[] for _ in ops]
    for index, preds in enumerate(predecessors):
        for pred in preds:
            successors[pred].append(index)

    height = _heights(ops, predecessors)
    outstanding = [len(preds) for preds in predecessors]

    # How many reads inside this run each produced value still has. A value also read after
    # the run never reaches zero, so it stays live for the whole run, which is the truth.
    pending = {}
    for op in ops:
        if op.result is not None:
            escapes = _key(op.result) in live_after
            pending[_key(op.result)] = float("inf") if escapes else 0
    for op in ops:
        for value in _operands(op):
            if _key(value) in pending and pending[_key(value)] != float("inf"):
                pending[_key(value)] += 1

    def delta(index):
        """Change in live registers from issuing ops[index] now."""
        change = 0
        op = ops[index]
        if op.result is not None and pending[_key(op.result)] > 0:
            change += _register_cost(op.result)
        counted = {}
        for value in _operands(op):
            if _key(value) in pending:
                counted[_key(value)] = counted.get(_key(value), 0) + 1
        for key, uses in counted.items():
            if pending[key] - uses <= 0:
                change -= _register_cost(_find_result(ops, key))
        return change

    ready = [index for index in range(count) if outstanding[index] == 0]
    order = []
    live = 0
    limit = PRESSURE_THRESHOLD * budget

    while ready:
        if live >= limit:
            # Pressure mode. Relieve first, and among equally relieving operations take the
            # earliest in source order rather than the longest path.
            #
            # Ranking ties by path length here does not work, and the reason is worth keeping.
            # The operation that would relieve pressure is often not ready yet: an accumulate
            # step cannot issue until its accumulator has been declared, and the declaration
            # itself looks pressure-positive. Preferring the longest path then picks another
            # load every time, and pressure never comes down. Source order reaches the
            # declaration, which unblocks the accumulate, which kills two values. It also
            # means pressure mode degrades towards the order the lowering emitted, which for
            # an accumulation loop is already the pressure-minimal one.
            choice = min(ready, key=lambda i: (delta(i), i))
        else:
            choice = min(ready, key=lambda i: (-height[i], delta(i), i))
        ready.remove(choice)
        order.append(choice)
        live += delta(choice)

        op = ops[choice]
        for value in _operands(op):
            if _key(value) in pending and pending[_key(value)] != float("inf"):
                pending[_key(value)] -= 1
        for successor in successors[choice]:
            outstanding[successor] -= 1
            if outstanding[successor] == 0:
                ready.append(successor)

    if len(order) != count:
        # A cycle means the dependence graph is wrong, and emitting a partial order would
        # produce a kernel that compiles and computes something else. Refuse instead.
        raise ScheduleError(
            f"dependence graph is cyclic over {count} operations; scheduled {len(order)}"
        )
    return order


def _find_result(ops, key):
    for op in ops:
        if op.result is not None and _key(op.result) == key:
            return op.result
    return None


class ScheduleError(RuntimeError):
    """The dependence graph came out inconsistent, so no reordering is safe."""


def reorder_for_latency(func, budget=None):
    """Reorder operations within each fence-free run. Arithmetic is untouched.

    Every reordering respects the dependences computed above, and no operation crosses a
    barrier, a loop, a conditional, or any operation this file does not model. Because only
    the order of whole operations changes and no expression is rewritten, results are
    bit-identical to the unscheduled kernel; what changes is how long values stay live and how
    early loads issue.
    """
    budget = agx.REGISTER_BUDGET if budget is None else budget

    def rewrite(ops):
        for op in ops:
            for body in _bodies(op):
                body[:] = rewrite(body)
        result = list(ops)
        for start, stop in _runs(result):
            window = result[start:stop]
            if len(window) < 2:
                continue
            live_after = _reads_everywhere(result[stop:])
            order = _list_schedule(window, _dependences(window), live_after, budget)
            result[start:stop] = [window[index] for index in order]
        return result

    func.ops = rewrite(func.ops)
    return func


def _absorbable(value, dtype, uses):
    """Whether this operand is an addition that a chain rooted above it can absorb.

    Single use is what makes absorbing safe: a shared intermediate is still needed in its
    original form by its other reader, so rebuilding around it would compute it twice or drop
    it. Same element type, because absorbing across a cast is a different transformation.
    """
    source = value.defining_op if value is not None else None
    return (
        isinstance(source, mir.MBinOp)
        and source.op == "+"
        and uses.get(_key(value), 0) == 1
        and source.result is not None
        and source.result.type == dtype
    )


def _use_counts(ops):
    """How many times each value is read, counting reads inside nested bodies.

    Nested bodies have to be counted. A partial sum read only inside a loop body looks unused
    at this level, and absorbing it would delete an operation the loop still needs.
    """
    counts = {}

    def walk(items):
        for op in items:
            for value in _operands(op):
                counts[_key(value)] = counts.get(_key(value), 0) + 1
            for body in _bodies(op):
                walk(body)

    walk(ops)
    return counts


def _chain_root(op, uses, consumers):
    """Whether this addition is the top of its chain rather than partway down one.

    Without this the search finds the shortest qualifying chain instead of the longest, and
    the result is worse than doing nothing: rebuilding the first four terms of an eight-term
    chain leaves a four-deep chain hanging off a tree, and the fixpoint guard then refuses to
    touch it, so the dependent path barely shortens.
    """
    if not (isinstance(op, mir.MBinOp) and op.op == "+" and op.result is not None):
        return False
    if not _absorbable(op.result, op.result.type, uses):
        return True
    above = consumers.get(_key(op.result), [])
    return not (len(above) == 1 and isinstance(above[0], mir.MBinOp) and above[0].op == "+")


def _chain(op, uses):
    """A serial addition chain rooted at `op` as (leaves, absorbed), or None.

    A serial chain is `((a + b) + c) + d`: every addition but the innermost has exactly one
    addition operand, so its dependent path is as long as it has terms. That is the shape
    worth rebuilding and the only shape this returns. `absorbed` is the inner additions, which
    exist only to feed the chain and have to be deleted when it is rebuilt — leaving them
    behind would add instructions rather than remove a dependence.

    Requiring that no leaf is itself absorbable is what makes the transformation a fixpoint.
    A balanced tree's operands *are* absorbable additions, so a rebuilt chain is not a chain
    again and the pass cannot rewrite its own output forever.
    """
    if not (isinstance(op, mir.MBinOp) and op.op == "+"):
        return None
    dtype = op.result.type
    leaves = []
    absorbed = []
    node = op
    while True:
        left, right = node.lhs, node.rhs
        if _absorbable(left, dtype, uses):
            leaves.append(right)
            node = left.defining_op
        elif _absorbable(right, dtype, uses):
            leaves.append(left)
            node = right.defining_op
        else:
            leaves.extend([left, right])
            break
        absorbed.append(node)
    if len(leaves) < 4 or any(_absorbable(leaf, dtype, uses) for leaf in leaves):
        return None
    return leaves, absorbed


def interleave_reduction_chains(func, ways=4, dtype_hint="f32"):
    """Rebuild serial addition chains as balanced trees, shortening the dependent path.

    A chain of n additions has depth n and a tree has depth log2(n), which is the whole of
    the instruction-level parallelism available inside one thread. On this target that is
    worth at most ILP_CEILING, 1.09x on fp32 and 1.41x on fp16, because the GPU already
    covers latency with thread-level parallelism rather than with ILP inside a thread. Five
    scheduling experiments on the int4 QMV measured flat against exactly that ceiling.

    It is off by default in the pipeline and callers should leave it off without a reason.
    Reassociating floating-point addition changes results in the last bits, and the model
    tests assert that meTile's logits equal MLX's exactly. Trading that for at most 9% is a
    bad trade; the pass exists so the choice is available and explicit rather than absent.

    Returns the function and the number of chains rebuilt, so a caller can tell whether the
    numeric change bought anything at all before keeping it.
    """
    if agx.ilp_headroom(dtype_hint) <= 1.0 or ways < 2:
        return func, 0

    rebuilt = 0

    def rewrite(ops):
        nonlocal rebuilt
        uses = _use_counts(ops)
        consumers = {}
        for op in ops:
            for value in _operands(op):
                consumers.setdefault(_key(value), []).append(op)
        for op in ops:
            for body in _bodies(op):
                rewrite(body)

        for index, op in enumerate(ops):
            if not _chain_root(op, uses, consumers):
                continue
            found = _chain(op, uses)
            if found is None:
                continue
            leaves, absorbed = found
            # The loop-carried value, if any, is the leaf no operation in this block defines.
            # Keeping it as the last addend means the accumulator is touched once, which is
            # what makes the remaining terms independent of it.
            carried = [leaf for leaf in leaves if leaf.defining_op is None]
            terms = [leaf for leaf in leaves if leaf.defining_op is not None]
            if not terms:
                continue
            built, level = [], list(terms)
            while len(level) > 1:
                nxt = []
                for position in range(0, len(level) - 1, 2):
                    pair = mir.MBinOp(op="+", lhs=level[position], rhs=level[position + 1])
                    pair.result = mir.MValue(f"_ilp_{id(pair) & 0xFFFFFF}", op.result.type, pair)
                    built.append(pair)
                    nxt.append(pair.result)
                if len(level) % 2:
                    nxt.append(level[-1])
                level = nxt
            total = level[0]
            for leaf in carried:
                joint = mir.MBinOp(op="+", lhs=leaf, rhs=total)
                joint.result = mir.MValue(f"_ilp_{id(joint) & 0xFFFFFF}", op.result.type, joint)
                built.append(joint)
                total = joint.result

            # The chain's own result is what everything downstream reads, so the last built
            # operation adopts it rather than the readers being rewritten.
            final = built[-1]
            final.result = op.result
            op.result.defining_op = final
            ops[index : index + 1] = built
            # The absorbed additions have no reader left, and each had exactly one, so
            # deleting them is safe. It is also the point: a rebuilt chain that leaves them
            # behind has emitted a tree *and* a chain, spending more instructions to shorten
            # a dependence rather than fewer.
            dead = {id(inner) for inner in absorbed}
            ops[:] = [candidate for candidate in ops if id(candidate) not in dead]
            rebuilt += 1
            return rewrite(ops)

    rewrite(func.ops)
    return func, rebuilt


def peak_pressure(func, budget=None):
    """Estimated peak live registers, and whether that estimate reaches the budget.

    For reporting and for tests that need to see a schedule change pressure rather than
    trusting that it did. The number is an estimate by construction; use
    `metile.target.agx.inspect` when the real count matters.
    """
    budget = agx.REGISTER_BUDGET if budget is None else budget
    peak = 0

    def walk(ops, live):
        nonlocal peak
        pending = {}
        for op in ops:
            if op.result is not None:
                pending[_key(op.result)] = 0
        for op in ops:
            for value in _operands(op):
                if _key(value) in pending:
                    pending[_key(value)] += 1
        for op in ops:
            for value in _operands(op):
                if _key(value) in pending:
                    pending[_key(value)] -= 1
                    if pending[_key(value)] == 0:
                        live -= _register_cost(value)
            if op.result is not None and pending[_key(op.result)] > 0:
                live += _register_cost(op.result)
            peak = max(peak, live)
            for body in _bodies(op):
                walk(body, live)
        return live

    walk(func.ops, 0)
    return {"peak": peak, "budget": budget, "at_budget": peak >= budget}
