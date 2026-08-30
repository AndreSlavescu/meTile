"""What the native scheduler is allowed to move, and what it must never move.

A scheduler that reorders one operation past a hazard produces a kernel that compiles
cleanly and computes something else, so the tests here are built around hazards rather than
around outcomes. Each one constructs an ordering the pass would have to break to be wrong,
and asserts it stays intact. The reordering tests are the mirror image: an order the pass
would have to leave alone to be useless.
"""

import os
import sys
from pathlib import Path

import pytest

from metile.compiler.scheduling import (
    ScheduleError,
    _dependences,
    _list_schedule,
    interleave_reduction_chains,
    peak_pressure,
    reorder_for_latency,
)
from metile.ir import metal_ir as mir
from metile.ir.types import U32, PtrType, ScalarType

F32 = ScalarType("f32")


def _emit(ops, op, name, type_):
    """Append an op and hand back its result, without MFunction's naming."""
    op.result = mir.MValue(name, type_, op)
    ops.append(op)
    return op.result


def _function(ops, params=()):
    return mir.MFunction(name="k", params=list(params), ops=list(ops))


def _index(ops, predicate):
    return next(i for i, op in enumerate(ops) if predicate(op))


def _pointer(name="A"):
    return mir.MValue(name, PtrType("f32"))


def test_a_serial_chain_comes_back_untouched():
    """No freedom means no churn.

    A scheduler that reports a different-but-equivalent order for a program with exactly one
    legal order is either ignoring dependences or shuffling for its own sake. Both are worth
    catching, and this is the cheapest place to catch them.
    """
    ops = []
    seed = _emit(ops, mir.MConstant(value=1.0, dtype="f32"), "c", F32)
    running = seed
    for step in range(6):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=seed), f"t{step}", F32)
    before = list(ops)

    reorder_for_latency(_function(ops))
    assert [id(op) for op in ops] == [id(op) for op in before]


def test_loads_issue_before_the_arithmetic_that_needs_them():
    """The reordering the pass exists to make, in the case where pressure allows it."""
    ops = []
    pointer = _pointer()
    zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
    first = _emit(ops, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "a", F32)
    scaled = _emit(ops, mir.MBinOp(op="*", lhs=first, rhs=first), "s", F32)
    second = _emit(ops, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "b", F32)
    _emit(ops, mir.MBinOp(op="+", lhs=scaled, rhs=second), "r", F32)

    function = _function(ops)
    reorder_for_latency(function)
    loads = [i for i, op in enumerate(function.ops) if isinstance(op, mir.DeviceLoad)]
    arithmetic = _index(function.ops, lambda op: isinstance(op, mir.MBinOp))
    assert max(loads) < arithmetic, "the second load should have been hoisted past the multiply"


def test_pressure_mode_holds_live_values_below_a_tight_budget():
    """The objective that makes this scheduler right for this target.

    Eight independent loads, each consumed once. Hoisting all eight keeps eight values alive;
    consuming each as it arrives keeps two. A latency-first scheduler picks the first, which
    on hardware whose spill cliff costs between 1.3x and 6.7x is the more expensive answer.
    """

    def build():
        ops = []
        pointer = _pointer()
        zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
        total = _emit(ops, mir.MConstant(value=0.0, dtype="f32"), "acc", F32)
        for step in range(8):
            loaded = _emit(
                ops, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), f"v{step}", F32
            )
            total = _emit(ops, mir.MBinOp(op="+", lhs=total, rhs=loaded), f"t{step}", F32)
        _emit(ops, mir.DeviceStore(ptr=pointer, index=zero, value=total), "st", None)
        return ops

    roomy = _function(build())
    reorder_for_latency(roomy, budget=1000)
    tight = _function(build())
    reorder_for_latency(tight, budget=4)

    assert peak_pressure(tight)["peak"] < peak_pressure(roomy)["peak"]


def test_nothing_crosses_a_barrier():
    ops = []
    zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
    _emit(ops, mir.MThreadgroupStore(array_name="tile", index=zero, value=zero), "st", None)
    ops.append(mir.MBarrier())
    after = _emit(ops, mir.MThreadgroupLoad(array_name="tile", index=zero, dtype="f32"), "ld", F32)
    _emit(ops, mir.MBinOp(op="*", lhs=after, rhs=after), "sq", F32)

    function = _function(ops)
    reorder_for_latency(function)
    fence = _index(function.ops, lambda op: isinstance(op, mir.MBarrier))
    assert isinstance(function.ops[fence - 1], mir.MThreadgroupStore)
    assert isinstance(function.ops[fence + 1], mir.MThreadgroupLoad)


def test_an_unmodelled_operation_is_a_fence():
    """The default that keeps the pass correct as the IR grows.

    Metal IR has around seventy operations and several are macro-operations that expand to
    loops with barriers inside them. If an operation nobody classified were treated as
    movable, adding one to the IR would silently become a miscompile in this pass. So an
    operation the scheduler does not model must pin everything around it, and this uses a
    simdgroup matrix multiply, which is exactly such an operation, to prove it.
    """
    ops = []
    pointer = _pointer()
    zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
    first = _emit(ops, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "a", F32)
    ops.append(mir.MSimdgroupMMA())  # a macro-operation: unmodelled, therefore a fence
    second = _emit(ops, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "b", F32)
    _emit(ops, mir.MBinOp(op="+", lhs=first, rhs=second), "r", F32)

    function = _function(ops)
    reorder_for_latency(function)
    opaque = _index(function.ops, lambda op: isinstance(op, mir.MSimdgroupMMA))
    loads = [i for i, op in enumerate(function.ops) if isinstance(op, mir.DeviceLoad)]
    assert loads == [opaque - 1, opaque + 1], (
        "a load moved across an operation with unknown effects"
    )


def test_a_variable_read_cannot_drift_past_the_next_write():
    """Write-after-read on a mutable name, which no SSA edge expresses.

    The accumulator lowering represents a variable read as an MValue with no defining
    operation, so nothing in the value graph connects the read to the assignment that
    overwrites it. Only the variable hazard does, and moving the read after the write would
    read the wrong iteration's value.
    """
    ops = []
    zero = _emit(ops, mir.MConstant(value=0.0, dtype="f32"), "z", U32)
    ops.append(mir.MVarDecl(var_name="acc", init_value=zero, dtype="f32"))
    read = mir.MValue("acc", F32)
    doubled = _emit(ops, mir.MBinOp(op="+", lhs=read, rhs=read), "d", F32)
    ops.append(mir.MVarAssign(var_name="acc", value=doubled))
    fresh = _emit(ops, mir.MConstant(value=1.0, dtype="f32"), "one", F32)
    ops.append(mir.MVarAssign(var_name="acc", value=fresh))

    function = _function(ops)
    reorder_for_latency(function)
    use = _index(function.ops, lambda op: isinstance(op, mir.MBinOp))
    writes = [i for i, op in enumerate(function.ops) if isinstance(op, mir.MVarAssign)]
    assert min(writes) > use, "the read of acc was scheduled after acc was reassigned"


def test_threadgroup_hazards_are_per_array():
    """Ordered where they alias, free where they cannot.

    Both halves matter. Ordering accesses to one array is correctness; leaving accesses to a
    different array free is the reason the pass can do anything at all in a GEMM inner loop,
    where several tiles are in flight at once.
    """
    ops = []
    zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
    _emit(ops, mir.MThreadgroupStore(array_name="a_tile", index=zero, value=zero), "s", None)
    same = _emit(
        ops, mir.MThreadgroupLoad(array_name="a_tile", index=zero, dtype="f32"), "same", F32
    )
    other = _emit(
        ops, mir.MThreadgroupLoad(array_name="b_tile", index=zero, dtype="f32"), "other", F32
    )
    _emit(ops, mir.MBinOp(op="+", lhs=same, rhs=other), "r", F32)

    # Legality, not the choice made from it. Whether the b_tile load actually moves depends on
    # its path length, so asserting a position here would pin a heuristic rather than the rule.
    store, aliased, unrelated = 1, 2, 3
    predecessors = _dependences(ops)
    assert store in predecessors[aliased], "a_tile's load must follow the store that fills it"
    assert store not in predecessors[unrelated], "b_tile cannot alias a_tile"

    function = _function(ops)
    reorder_for_latency(function)
    order = [
        _index(function.ops, lambda op: isinstance(op, mir.MThreadgroupStore)),
        _index(
            function.ops,
            lambda op: isinstance(op, mir.MThreadgroupLoad) and op.array_name == "a_tile",
        ),
    ]
    assert order[1] > order[0], "a load of a_tile moved above the store that fills it"


def test_a_device_store_orders_loads_through_a_different_pointer():
    """Aliasing conservatism, on the case where being clever would be wrong.

    MPointerOffset manufactures a second pointer value naming the same buffer, so two
    distinct MValues can address one location. Keying device hazards on the pointer value
    would let a load of the aliased pointer move above a store and read stale memory, which
    is why they are kept in one bucket.
    """
    ops = []
    base = _pointer()
    zero = _emit(ops, mir.MConstant(value=0, dtype="u32"), "z", U32)
    _emit(ops, mir.DeviceStore(ptr=base, index=zero, value=zero), "st", None)
    alias = _emit(ops, mir.MPointerOffset(ptr=base, offset="16"), "alias", PtrType("f32"))
    loaded = _emit(ops, mir.DeviceLoad(ptr=alias, index=zero, dtype="f32"), "ld", F32)
    _emit(ops, mir.MBinOp(op="*", lhs=loaded, rhs=loaded), "sq", F32)

    function = _function(ops)
    reorder_for_latency(function)
    store = _index(function.ops, lambda op: isinstance(op, mir.DeviceStore))
    load = _index(function.ops, lambda op: isinstance(op, mir.DeviceLoad))
    assert load > store, "a load moved above a store that may write the same memory"


def test_a_cyclic_dependence_graph_is_refused():
    """Emitting a partial order would be worse than failing.

    A cycle means the dependence analysis is inconsistent. Scheduling the operations it
    managed to order and dropping the rest yields a kernel that still compiles, so the
    failure would surface as a wrong answer somewhere else entirely.
    """
    ops = []
    left = _emit(ops, mir.MConstant(value=1.0, dtype="f32"), "a", F32)
    _emit(ops, mir.MBinOp(op="+", lhs=left, rhs=left), "b", F32)
    with pytest.raises(ScheduleError, match="cyclic"):
        _list_schedule(ops, [{1}, {0}], live_after=set(), budget=140)


def test_reassociation_shortens_the_dependent_path():
    ops = []
    terms = [_emit(ops, mir.MConstant(value=float(i), dtype="f32"), f"c{i}", F32) for i in range(8)]
    running = terms[0]
    for position, term in enumerate(terms[1:]):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=term), f"t{position}", F32)

    function, rebuilt = interleave_reduction_chains(_function(ops), ways=4)
    assert rebuilt == 1
    assert _dependent_depth(function.ops) < 7


def test_reassociation_keeps_the_value_its_readers_hold():
    """Downstream operations hold the chain's result object, so the rebuild must adopt it.

    Producing a new value and leaving the readers pointed at the old one would drop the
    rebuilt arithmetic on the floor, and the kernel would still compile.
    """
    ops = []
    terms = [_emit(ops, mir.MConstant(value=float(i), dtype="f32"), f"c{i}", F32) for i in range(6)]
    running = terms[0]
    for position, term in enumerate(terms[1:]):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=term), f"t{position}", F32)
    consumer = mir.MBinOp(op="*", lhs=running, rhs=running)
    _emit(ops, consumer, "out", F32)

    function, rebuilt = interleave_reduction_chains(_function(ops), ways=4)
    assert rebuilt == 1
    assert consumer.lhs is running
    assert running.defining_op in function.ops


def test_a_shared_partial_sum_survives_the_rebuild():
    """A second reader makes a partial sum a leaf, not something to absorb.

    Rebuilding deletes the additions it absorbs, because leaving them behind would emit a tree
    and a chain and cost more instructions than it saves. That makes the single-use rule load
    bearing: absorb an addition someone else reads and the rebuild deletes an operation that
    is still needed. Here the chain may be rebuilt around the shared partial sum, but the
    partial sum itself has to still be computed and its other reader still has to see it.
    """
    ops = []
    terms = [_emit(ops, mir.MConstant(value=float(i), dtype="f32"), f"c{i}", F32) for i in range(5)]
    partial = _emit(ops, mir.MBinOp(op="+", lhs=terms[0], rhs=terms[1]), "p", F32)
    elsewhere = mir.MBinOp(op="*", lhs=partial, rhs=partial)
    _emit(ops, elsewhere, "escapes", F32)
    running = partial
    for position, term in enumerate(terms[2:]):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=term), f"t{position}", F32)

    function, _ = interleave_reduction_chains(_function(ops), ways=4)
    assert partial.defining_op in function.ops, "the shared partial sum was deleted"
    assert elsewhere in function.ops and elsewhere.lhs is partial


def test_reassociation_stops_at_a_fixpoint():
    """Running twice must not rewrite the first run's output.

    A balanced tree's operands are themselves single-use additions, so a chain detector that
    only looked for additions feeding additions would absorb its own result forever.
    """
    ops = []
    terms = [_emit(ops, mir.MConstant(value=float(i), dtype="f32"), f"c{i}", F32) for i in range(9)]
    running = terms[0]
    for position, term in enumerate(terms[1:]):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=term), f"t{position}", F32)

    function, first = interleave_reduction_chains(_function(ops), ways=4)
    _, second = interleave_reduction_chains(function, ways=4)
    assert first == 1
    assert second == 0


def test_reassociation_declines_when_the_target_offers_no_headroom():
    """The machine model decides, not the pass.

    bf16 has no measured ILP ceiling, so `ilp_headroom` reports 1.0 for it rather than
    borrowing f16's. A transformation that costs bit-exactness must not run for a gain nobody
    measured.
    """
    ops = []
    terms = [_emit(ops, mir.MConstant(value=float(i), dtype="f32"), f"c{i}", F32) for i in range(8)]
    running = terms[0]
    for position, term in enumerate(terms[1:]):
        running = _emit(ops, mir.MBinOp(op="+", lhs=running, rhs=term), f"t{position}", F32)

    _, rebuilt = interleave_reduction_chains(_function(ops), ways=4, dtype_hint="bf16")
    assert rebuilt == 0


def _dependent_depth(ops):
    """Longest chain of value dependences through these operations."""
    depth = {}
    longest = 0
    for op in ops:
        inputs = [
            depth.get(id(getattr(op, field, None)), 0)
            for field in ("lhs", "rhs", "operand", "value")
            if isinstance(getattr(op, field, None), mir.MValue)
        ]
        here = (max(inputs) if inputs else 0) + 1
        if op.result is not None:
            depth[id(op.result)] = here
        longest = max(longest, here)
    return longest


def test_nested_bodies_are_scheduled_too():
    """The inner loop is where the arithmetic is, so a pass that skipped it would be inert."""
    inner = []
    pointer = _pointer()
    zero = _emit(inner, mir.MConstant(value=0, dtype="u32"), "z", U32)
    first = _emit(inner, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "a", F32)
    _emit(inner, mir.MBinOp(op="*", lhs=first, rhs=first), "s", F32)
    second = _emit(inner, mir.DeviceLoad(ptr=pointer, index=zero, dtype="f32"), "b", F32)
    _emit(inner, mir.MBinOp(op="+", lhs=second, rhs=second), "r", F32)
    loop = mir.MForLoop(iv_name="k", start=0, end=16, body=inner)

    function = _function([loop])
    reorder_for_latency(function)
    body = function.ops[0].body
    loads = [i for i, op in enumerate(body) if isinstance(op, mir.DeviceLoad)]
    assert max(loads) < _index(body, lambda op: isinstance(op, mir.MBinOp))


def test_scheduling_a_real_kernel_changes_no_output_bit():
    """The claim that matters, checked on the whole pipeline instead of on the IR.

    Every structural test above says a hazard was respected. None of them says the kernel that
    comes out the far end computes the same thing, and that is the property the pass sells:
    operations move, arithmetic does not, so results are identical to the last bit. Anything
    weaker than bit-identical would be a reassociation the pass is not supposed to perform.

    Run in subprocesses because the pass is chosen by environment variable at compile time, and
    flipping it inside one process means reloading the compiler underneath a live module graph.

    The MSL digests are compared too, and they have to *differ*. Without that the test passes
    for the wrong reason: if the environment variable stopped reaching the compiler, both runs
    would compile the same kernel and agreeing outputs would prove nothing at all.
    """
    import subprocess

    script = """
import hashlib, sys
import numpy as np
import metile
from metile.runtime.metal_device import MetalDevice
from metile.kernels.rmsnorm import rmsnorm
from metile.kernels.softmax import softmax

width, rows = 4096, 8
data = np.random.default_rng(7).standard_normal((rows, width), dtype=np.float32)
digest = []
for kernel, extra in ((rmsnorm, True), (softmax, False)):
    values = metile.Buffer(data=data.ravel())
    out = metile.Buffer.zeros((rows * width,))
    if extra:
        weight = metile.Buffer(data=np.linspace(0.5, 1.5, width, dtype=np.float32))
        kernel[(rows,)](values, weight, out, width, 1e-5, BLOCK=256)
    else:
        kernel[(rows,)](values, out, width, BLOCK=256)
    MetalDevice.get().sync()
    digest.append(hashlib.sha256(out.numpy().tobytes()).hexdigest())
from metile.frontend.kernel import _kernel_cache
sources = "".join(sorted(entry.msl_source for entry in _kernel_cache.values()))
sys.stdout.write(":".join(digest) + "|" + hashlib.sha256(sources.encode()).hexdigest())
"""
    results = {}
    for setting in ("0", "1"):
        finished = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**os.environ, "METILE_SCHEDULE": setting},
        )
        assert finished.returncode == 0, finished.stderr[-2000:]
        results[setting] = finished.stdout

    outputs = {setting: value.split("|")[0] for setting, value in results.items()}
    sources = {setting: value.split("|")[1] for setting, value in results.items()}
    assert outputs["0"], "the unscheduled run produced no output"
    assert sources["0"] != sources["1"], (
        "both runs compiled the same MSL, so METILE_SCHEDULE is not reaching the compiler and "
        "this test is not measuring anything"
    )
    assert outputs["0"] == outputs["1"], "scheduling changed a result bit"
