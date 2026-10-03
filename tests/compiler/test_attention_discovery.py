from dataclasses import replace

import numpy as np
import pytest

from metile.compiler.algo_discovery import (
    discover_flash_attention,
    find_flash_attention,
)
from metile.compiler.reduction_algebra import (
    max_reduction,
    prove_reduction,
    sum_reduction,
    weighted_softmax_reduction,
)
from metile.ir.graph_ir import GraphBuilder, TensorSpec


def _attention_graph(*, queries=1, keys=64, causal=False, expose_scores=False):
    builder = GraphBuilder()
    query = builder.input("query", TensorSpec((1, 4, queries, 64), "f32"))
    key = builder.input("key", TensorSpec((1, 4, keys, 64), "f32"))
    value = builder.input("value", TensorSpec((1, 4, keys, 64), "f32"))
    scores = builder.matmul(query, key, transpose_right=True, name="scores")
    scaled = builder.scale(scores, 64**-0.5, name="scaled_scores")
    if causal:
        scaled = builder.causal_mask(scaled, name="causal_scores")
    probabilities = builder.softmax(scaled, name="probabilities")
    output = builder.matmul(probabilities, value, name="attention_output")
    outputs = (scores, output) if expose_scores else output
    return builder.build(outputs)


def test_builtin_reductions_discharge_monoid_obligations():
    for law in (sum_reduction(), max_reduction(), weighted_softmax_reduction()):
        certificate = prove_reduction(law)
        assert certificate.verified
        assert {obligation.name for obligation in certificate.obligations} == {
            "generated_associativity",
            "left_identity",
            "pair_homomorphism",
            "right_identity",
        }


def test_reduction_verifier_rejects_an_invalid_merge():
    law = weighted_softmax_reduction()

    def invalid_merge(left, right):
        merged = law.merge(left, right)
        return merged[0], left[1], merged[2]

    certificate = prove_reduction(replace(law, merge=invalid_merge))

    assert not certificate.verified
    assert any(not obligation.verified for obligation in certificate.obligations)


def test_flash_attention_discovery_replaces_exact_private_graph():
    graph = _attention_graph()

    matches = find_flash_attention(graph)
    rewritten = discover_flash_attention(graph)

    assert len(matches) == 1
    assert matches[0].certificate.verified
    assert matches[0].certificate.theorem == "stable_weighted_softmax_monoid"
    assert [node.op for node in rewritten.nodes] == ["flash_attention"]
    assert rewritten.nodes[0].attrs["scale"] == pytest.approx(0.125)
    assert not rewritten.nodes[0].attrs["causal"]


def test_flash_attention_discovery_preserves_causal_semantics():
    rewritten = discover_flash_attention(_attention_graph(queries=16, causal=True))

    assert [node.op for node in rewritten.nodes] == ["flash_attention"]
    assert rewritten.nodes[0].attrs["causal"]


def test_flash_attention_discovery_rejects_escaping_intermediate():
    graph = _attention_graph(expose_scores=True)

    assert find_flash_attention(graph) == ()
    assert discover_flash_attention(graph) is graph


def test_mlx_graph_executes_discovered_decode_attention():
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_graph import compile_mlx_graph

    random = np.random.default_rng(61)
    query = mx.array(random.normal(size=(1, 4, 1, 64)).astype(np.float32))
    key = mx.array(random.normal(size=(1, 4, 64, 64)).astype(np.float32))
    value = mx.array(random.normal(size=(1, 4, 64, 64)).astype(np.float32))
    executable = compile_mlx_graph(_attention_graph(), autotune=False)

    actual = executable(query, key, value)
    expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=64**-0.5)
    mx.eval(actual, expected)

    assert [node.op for node in executable.plan.graph.nodes] == ["flash_attention"]
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("causal", (False, True))
def test_generated_flash_attention_matches_mlx(causal):
    mx = pytest.importorskip("mlx.core")
    from metile.backends.mlx_attention import _native_attention, mlx_flash_attention

    random = np.random.default_rng(67)
    query = mx.array(random.normal(size=(1, 2, 16, 64)).astype(np.float32))
    key = mx.array(random.normal(size=(1, 2, 16, 64)).astype(np.float32))
    value = mx.array(random.normal(size=(1, 2, 16, 64)).astype(np.float32))

    actual = mlx_flash_attention(
        query,
        key,
        value,
        scale=64**-0.5,
        causal=causal,
        autotune=False,
    )
    expected = _native_attention(query, key, value, 64**-0.5, causal)
    mx.eval(actual, expected)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-3, atol=3e-3)
