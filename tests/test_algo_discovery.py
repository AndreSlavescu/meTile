"""Tests for algorithmic discovery passes."""

from metile.compiler.algo_discovery import _detect_softmax_pattern
from metile.ir import tile_ir as tir
from metile.ir.types import I32, ScalarType, TileType


def _make_softmax_ir():
    """Build a minimal 3-loop softmax Tile IR."""
    S = ScalarType("f32")
    T = TileType((256,), "f32")
    N = tir.Value("N", I32)
    zero = tir.Value("zero", I32)

    # Loop 1: max
    load1 = tir.Load(ptr=tir.Value("p1", S), offsets=tir.Value("o1", I32))
    load1.result = tir.Value("l1", T, load1)
    init_m = tir.Constant(value=-1e38, dtype="f32")
    init_m.result = tir.Value("init_m", S, init_m)
    max_op = tir.BinOp(op="max", lhs=init_m.result, rhs=load1.result)
    max_op.result = tir.Value("tile_max", T, max_op)
    loop1 = tir.ForRange(
        start=zero, end=N, step=256, iv=tir.Value("i", I32), body=[load1, init_m, max_op]
    )

    # Reduce max
    r_max = tir.Reduce(op="max", operand=max_op.result)
    r_max.result = tir.Value("m", S, r_max)

    # Loop 2: sum exp
    load2 = tir.Load(ptr=tir.Value("p2", S), offsets=tir.Value("o2", I32))
    load2.result = tir.Value("l2", T, load2)
    sub_op = tir.BinOp(op="sub", lhs=load2.result, rhs=r_max.result)
    sub_op.result = tir.Value("sub", T, sub_op)
    exp_op = tir.Unary(op="exp", operand=sub_op.result)
    exp_op.result = tir.Value("exp", T, exp_op)
    init_s = tir.Constant(value=0.0, dtype="f32")
    init_s.result = tir.Value("init_s", S, init_s)
    add_op = tir.BinOp(op="add", lhs=init_s.result, rhs=exp_op.result)
    add_op.result = tir.Value("sum_tile", T, add_op)
    loop2 = tir.ForRange(
        start=zero,
        end=N,
        step=256,
        iv=tir.Value("j", I32),
        body=[load2, sub_op, exp_op, init_s, add_op],
    )

    # Reduce sum
    r_sum = tir.Reduce(op="sum", operand=add_op.result)
    r_sum.result = tir.Value("s", S, r_sum)

    # Loop 3: normalize
    loop3 = tir.ForRange(start=zero, end=N, step=256, iv=tir.Value("k", I32), body=[])

    return [loop1, r_max, loop2, r_sum, loop3]


def test_detect_softmax_pattern():
    """Pattern detection finds the 3-loop softmax structure."""
    ops = _make_softmax_ir()
    result = _detect_softmax_pattern(ops)
    assert result is not None

    l1, l2, l3, max_reduce, sum_reduce, indices = result
    assert max_reduce.op == "max"
    assert sum_reduce.op == "sum"
    assert l1.step == l2.step == l3.step == 256
    assert indices == [0, 2, 4]


def test_detect_rejects_non_softmax():
    """Pattern detection returns None for non-matching IR."""
    N = tir.Value("N", I32)
    zero = tir.Value("zero", I32)

    # Only 2 loops (not 3)
    loop1 = tir.ForRange(start=zero, end=N, step=256, iv=tir.Value("i", I32), body=[])
    loop2 = tir.ForRange(start=zero, end=N, step=256, iv=tir.Value("j", I32), body=[])
    assert _detect_softmax_pattern([loop1, loop2]) is None


def test_detect_rejects_mismatched_steps():
    """Loops with different steps are not softmax."""
    ops = _make_softmax_ir()
    # Change loop 2's step
    ops[2].step = 128
    result = _detect_softmax_pattern(ops)
    assert result is None


def test_detect_rejects_missing_max():
    """Loop 1 without element-wise max is not softmax."""
    ops = _make_softmax_ir()
    # Remove the max op from loop 1
    ops[0].body = [op for op in ops[0].body if not (isinstance(op, tir.BinOp) and op.op == "max")]
    result = _detect_softmax_pattern(ops)
    assert result is None


def test_detect_rejects_missing_exp():
    """Loop 2 without exp is not softmax."""
    ops = _make_softmax_ir()
    # Remove exp from loop 2
    ops[2].body = [op for op in ops[2].body if not isinstance(op, tir.Unary)]
    result = _detect_softmax_pattern(ops)
    assert result is None


def _certificate(verified=True):
    from metile.compiler.algo_discovery import (
        ProofObligation,
        ReductionCertificate,
        attention_monoid_certificate,
    )

    if verified:
        return attention_monoid_certificate()
    return ReductionCertificate(
        theorem="unproven",
        theory="none",
        obligations=(ProofObligation("associativity", (), (), False),),
    )


def _candidate(name, region, benefit, verified=True):
    from metile.compiler.algo_discovery import Candidate

    return Candidate(
        name=name,
        region=tuple(region),
        benefit=benefit,
        certificate=_certificate(verified),
    )


def test_select_candidates_drops_unproven_rewrites():
    """A rewrite whose obligations fail is never applied, however profitable."""
    from metile.compiler.algo_discovery import select_candidates

    node = object()
    unproven = _candidate("cheat", [node], 1000.0, verified=False)

    assert select_candidates([unproven]) == ()


def test_select_candidates_keeps_disjoint_rewrites():
    """Rewrites that touch different regions all survive."""
    from metile.compiler.algo_discovery import select_candidates

    left = _candidate("a", [object()], 1.0)
    right = _candidate("b", [object()], 1.0)

    assert set(select_candidates([left, right])) == {left, right}


def test_select_candidates_resolves_conflicts_by_benefit():
    """Overlapping rewrites are mutually exclusive; the min-cut keeps the better one."""
    from metile.compiler.algo_discovery import select_candidates

    shared = object()
    cheap = _candidate("cheap", [shared, object()], 1.0)
    valuable = _candidate("valuable", [shared, object()], 10.0)

    assert select_candidates([cheap, valuable]) == (valuable,)
    # Selection must not depend on the order candidates were proposed in.
    assert select_candidates([valuable, cheap]) == (valuable,)


def test_select_candidates_maximizes_total_benefit_not_local_choice():
    """Two disjoint small rewrites beat one overlapping large rewrite."""
    from metile.compiler.algo_discovery import select_candidates

    first, second = object(), object()
    left = _candidate("left", [first], 6.0)
    right = _candidate("right", [second], 6.0)
    straddling = _candidate("straddling", [first, second], 10.0)

    selected = select_candidates([straddling, left, right])

    assert set(selected) == {left, right}


def test_online_softmax_candidate_carries_a_verified_certificate():
    """The Tile IR rewrite is licensed by the same proven attention monoid."""
    from metile.compiler.algo_discovery import find_online_softmax

    func = tir.Function(name="softmax", params=[], ops=_make_softmax_ir())
    candidates = find_online_softmax(func)

    assert len(candidates) == 1
    assert candidates[0].name == "online_softmax"
    assert candidates[0].certificate.verified
    assert candidates[0].certificate.theorem == "stable_weighted_softmax_monoid"


def test_online_softmax_rewrite_removes_a_pass_and_both_reductions():
    """The fused loop replaces two passes, and neither post-loop reduce survives.

    Keeping them would re-reduce a value that is already threadgroup-wide, which
    silently inflates the normalizer by the thread count.
    """
    from metile.compiler.algo_discovery import discover_online_softmax

    func = tir.Function(name="softmax", params=[], ops=_make_softmax_ir())
    rewritten = discover_online_softmax(func)

    loops = [op for op in rewritten.ops if isinstance(op, tir.ForRange)]
    reduces = [op for op in rewritten.ops if isinstance(op, tir.Reduce)]

    assert getattr(rewritten, "_online_softmax", False)
    assert len(loops) == 2
    assert reduces == []


def test_online_softmax_predicates_the_ragged_tail_with_the_law_identity():
    """Masked lanes take -inf, the identity of both reductions in the monoid.

    A threadgroup reduction cannot sit inside a mask branch, so the fused loop asks
    lowering to predicate the load instead. -inf is the maximum's identity, and
    exp(-inf - m) = 0 is the normalizer's, so masked lanes cannot change either.
    """
    from metile.compiler.algo_discovery import discover_online_softmax

    func = tir.Function(name="softmax", params=[], ops=_make_softmax_ir())
    rewritten = discover_online_softmax(func)

    fused = next(op for op in rewritten.ops if isinstance(op, tir.ForRange))
    assert fused.masked_identity == -1e38
