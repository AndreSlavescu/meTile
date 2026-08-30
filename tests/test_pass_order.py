"""Pass-ordering invariants, and the wiring that enforces them.

The constraints in metile.compiler.passes are only worth anything if something checks them
against the passes that actually run. These tests cover both halves: the checker itself, and
the compile pipeline that feeds it.
"""

import ast
import inspect

import numpy as np
import pytest

import metile.frontend.kernel as kernel_module
from metile.compiler.passes import (
    _MUTUALLY_EXCLUSIVE,
    _PASS_ORDER_CONSTRAINTS,
    PassOrderError,
    validate_pass_order,
)
from metile.kernels.gemm import matmul

_CONSTRAINED = {name for pair in _PASS_ORDER_CONSTRAINTS + _MUTUALLY_EXCLUSIVE for name in pair}


def _record_pass_sequences(monkeypatch):
    """Capture every pass sequence the compiler validates during a compile.

    Swaps in an empty kernel cache so the compile actually happens. Without this the test
    passes alone and fails in a full run, because an earlier test has already compiled the
    same kernel and the cached one never re-enters the pass pipeline. monkeypatch restores
    the real cache afterwards.
    """
    monkeypatch.setattr(kernel_module, "_kernel_cache", {})

    recorded = []
    original = kernel_module.validate_pass_order

    def spy(applied):
        recorded.append(list(applied))
        return original(applied)

    monkeypatch.setattr(kernel_module, "validate_pass_order", spy)
    return recorded


def _run(BM, BN, BK, dtype, M=256, N=256, K=256):
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)
    matmul[(M // BM, N // BN)](A, B, C, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return A, B, C


class TestValidator:
    def test_accepts_the_double_buffer_path(self):
        validate_pass_order(
            ["pad_shared_memory", "double_buffer_k_loop", "vectorize_loads", "fold_constants"]
        )

    def test_accepts_the_split_k_fallback_path(self):
        validate_pass_order(
            [
                "pad_shared_memory",
                "double_buffer_k_loop",
                "split_k_loop",
                "vectorize_loads",
                "fold_constants",
            ]
        )

    def test_rejects_split_k_before_double_buffer(self):
        """split_k_loop is the fallback — running it first would split the loop the
        double-buffer attempt then looks for."""
        with pytest.raises(PassOrderError, match="double_buffer_k_loop"):
            validate_pass_order(["split_k_loop", "double_buffer_k_loop"])

    def test_rejects_vectorize_before_split_k(self):
        with pytest.raises(PassOrderError, match="split_k_loop"):
            validate_pass_order(["vectorize_loads", "split_k_loop"])

    def test_rejects_pad_and_swizzle_together(self):
        with pytest.raises(PassOrderError, match="mutually exclusive"):
            validate_pass_order(["pad_shared_memory", "swizzle_shared_memory"])

    def test_unconstrained_passes_are_left_alone(self):
        validate_pass_order(["serpentine_mma", "block_swizzle", "fold_constants"])


class TestPipelineIsValidated:
    """Guards the wiring. Without these, the validator can be silently orphaned again."""

    def test_compiling_validates_the_passes_that_ran(self, monkeypatch):
        recorded = _record_pass_sequences(monkeypatch)
        _run(64, 64, 32, np.float16)

        assert recorded, "compiling a kernel did not validate any pass sequence"
        assert all(seq for seq in recorded), "validated an empty pass sequence"
        # The recorded names must be real passes, not invented labels.
        for seq in recorded:
            for name in seq:
                assert hasattr(kernel_module, name) or name == "reorder_for_latency"

    def test_split_k_fallback_path_satisfies_the_constraints(self, monkeypatch):
        """The regression this file exists for.

        A 64x64x64 f16 tile needs 16 KB of threadgroup memory, so double_buffer_k_loop
        declines (doubling it exceeds max_tg_bytes) and split_k_loop runs as the fallback.
        That ordinary path once tripped the ordering check, because the recorded constraint
        had the two passes backwards.
        """
        recorded = _record_pass_sequences(monkeypatch)
        _, _, C = _run(64, 64, 64, np.float16)

        fallback = [s for s in recorded if "split_k_loop" in s and "double_buffer_k_loop" in s]
        if not fallback:
            pytest.skip("no kernel on this device took the split-k fallback path")

        for seq in fallback:
            assert seq.index("double_buffer_k_loop") < seq.index("split_k_loop")
            assert seq.index("split_k_loop") < seq.index("vectorize_loads")

        assert np.isfinite(np.asarray(C, dtype=np.float32)).all()

    def test_every_constrained_pass_goes_through_the_recorder(self):
        """Structural guard against silent drift.

        Calling a constrained pass directly instead of through _run_pass still compiles and
        still produces a valid kernel — it just quietly drops that pass from the sequence the
        constraints are checked against. Nothing at runtime notices, so check the source.
        """
        source = inspect.getsource(kernel_module)
        tree = ast.parse(source)

        unrecorded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _CONSTRAINED:
                unrecorded.append(f"{func.id}() at line {node.lineno}")

        assert not unrecorded, (
            "these constrained passes are invoked directly and so never reach "
            f"validate_pass_order: {unrecorded}. Call them via _run_pass instead."
        )

    def test_fallback_path_still_computes_the_right_answer(self):
        A, B, C = _run(64, 64, 64, np.float16)
        expected = A.astype(np.float32) @ B.astype(np.float32)
        np.testing.assert_allclose(C.astype(np.float32), expected, rtol=8e-2, atol=8e-2)
