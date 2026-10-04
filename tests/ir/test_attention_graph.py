from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from metile.backends.native_attention_graph import (
    NativeAttentionGraphTape,
    _signature,
    compile_native_attention_graph,
)
from metile.ir.attention_graph import (
    dual_chunk_attention,
    gated_delta_attention,
    kimi_delta_attention,
    stable_attention,
    validate_attention_node,
)
from metile.ir.graph_ir import GraphBuilder, TensorSpec
from metile.runtime.metal_device import MetalDevice


def _attention_inputs(builder):
    query = builder.input("query", TensorSpec((1, 4, 3, 32), "f32"))
    key = builder.input("key", TensorSpec((1, 2, 5, 32), "f32"))
    value = builder.input("value", key.spec)
    return query, key, value


def _delta_inputs(builder, channel):
    shapes = (
        (1, 3, 2, 4),
        (1, 3, 2, 4),
        (1, 3, 2, 5),
        (1, 3, 2, 4) if channel else (1, 3, 2),
        (1, 3, 2),
        (1, 2, 4, 5),
    )
    names = ("query", "key", "value", "log_decay", "beta", "initial_state")
    return tuple(
        builder.input(name, TensorSpec(shape, "f32")) for name, shape in zip(names, shapes)
    )


def test_stable_attention_records_explicit_causality_and_mask_metadata():
    builder = GraphBuilder()
    query, key, value = _attention_inputs(builder)
    mask = builder.input("mask", TensorSpec((1, 4, 3, 5), "u8"))
    output = stable_attention(builder, query, key, value, causal=True, mask=mask)
    graph = builder.build(output)
    assert output.spec == query.spec
    assert graph.nodes[0].attrs["causal_offset"] == 2
    assert graph.nodes[0].attrs["has_mask"] is True
    assert graph.nodes[0].op == "stable_attention"
    assert compile_native_attention_graph(graph).graph is graph


@pytest.mark.parametrize("channel", [False, True])
def test_recurrent_attention_has_distinct_decay_semantics_and_explicit_state_edges(channel):
    builder = GraphBuilder()
    inputs = _delta_inputs(builder, channel)
    operation = kimi_delta_attention if channel else gated_delta_attention
    output, final_state = operation(builder, *inputs)
    next_output, next_state = operation(builder, *inputs[:-1], final_state)
    graph = builder.build((output, final_state, next_output, next_state))
    expected_name = "kimi_delta_attention" if channel else "gated_delta_attention"
    assert [node.op for node in graph.nodes] == [expected_name, expected_name]
    assert final_state.spec == inputs[-1].spec
    assert graph.nodes[1].inputs[-1] is final_state
    assert next_output.spec == inputs[2].spec
    assert final_state.output_index == 1
    compile_native_attention_graph(graph)


def test_dca_requires_explicit_rotated_branches_and_global_positions():
    builder = GraphBuilder()
    query, key, value = _attention_inputs(builder)
    successive = builder.input("successive", query.spec)
    inter = builder.input("inter", query.spec)
    output = dual_chunk_attention(
        builder,
        query,
        successive,
        inter,
        key,
        value,
        chunk_size=8,
        local_window=2,
        query_start=11,
        key_start=3,
    )
    node = builder.build(output).nodes[0]
    assert len(node.inputs) == 5
    assert node.op == "dual_chunk_attention"
    assert node.attrs["query_start"] == 11
    assert node.attrs["key_start"] == 3
    assert node.attrs["chunk_size"] - node.attrs["local_window"] == 6
    validate_attention_node(node)


@pytest.mark.parametrize(
    "options",
    [
        {"scale": 0},
        {"scale": float("inf")},
        {"scale": True},
        {"causal": 1},
        {"causal_offset": 3},
        {"causal": True, "causal_offset": 2**31},
    ],
)
def test_stable_attention_rejects_invalid_semantic_options(options):
    builder = GraphBuilder()
    with pytest.raises((TypeError, ValueError)):
        stable_attention(builder, *_attention_inputs(builder), **options)


@pytest.mark.parametrize("channel", [False, True])
def test_recurrent_nodes_cannot_silently_swap_scalar_and_channel_decay(channel):
    builder = GraphBuilder()
    inputs = _delta_inputs(builder, channel)
    wrong_operation = gated_delta_attention if channel else kimi_delta_attention
    with pytest.raises(ValueError, match="log_decay"):
        wrong_operation(builder, *inputs)


@pytest.mark.parametrize(
    "options",
    [
        {"chunk_size": 0},
        {"local_window": 8},
        {"query_start": -1},
        {"key_start": 2**31},
        {"max_workspace_bytes": 0},
    ],
)
def test_dca_rejects_ambiguous_or_unbounded_configuration(options):
    builder = GraphBuilder()
    query, key, value = _attention_inputs(builder)
    with pytest.raises(ValueError):
        dual_chunk_attention(
            builder,
            query,
            query,
            query,
            key,
            value,
            **{"chunk_size": 8, "local_window": 2, **options},
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "output_dtype",
        "output_index",
        "missing_attribute",
        "unknown_attribute",
        "mask_count",
        "effectful",
    ],
)
def test_compile_rechecks_manually_corrupted_attention_metadata(corruption):
    builder = GraphBuilder()
    output = stable_attention(builder, *_attention_inputs(builder))
    graph = builder.build(output)
    node = graph.nodes[0]
    if corruption == "output_dtype":
        output.spec = replace(output.spec, dtype="f16")
    elif corruption == "output_index":
        output.output_index = 1
    elif corruption == "missing_attribute":
        del node.attrs["scale"]
    elif corruption == "unknown_attribute":
        node.attrs["pretend_rope"] = True
    elif corruption == "mask_count":
        node.attrs["has_mask"] = True
    else:
        node.side_effect = True
    with pytest.raises(ValueError):
        compile_native_attention_graph(graph)


def test_native_executor_rejects_generic_softmax_instead_of_rewriting_it():
    builder = GraphBuilder()
    source = builder.input("scores", TensorSpec((2, 4), "f32"))
    graph = builder.build(builder.softmax(source))
    with pytest.raises(ValueError, match=r"unsupported.*softmax"):
        compile_native_attention_graph(graph)
    assert graph.nodes[0].op == "softmax"


@pytest.mark.parametrize(
    "corruption", ["count", "dtype", "shape", "contiguity", "type", "changed_node"]
)
def test_runtime_contract_validation_precedes_metal_initialization(monkeypatch, corruption):
    def forbidden():
        raise AssertionError("invalid graph inputs must not initialize Metal")

    monkeypatch.setattr(MetalDevice, "get", forbidden)
    builder = GraphBuilder()
    output = stable_attention(builder, *_attention_inputs(builder))
    graph = builder.build(output)
    executable = compile_native_attention_graph(graph)
    inputs = [np.zeros(value.spec.shape, dtype=np.float32) for value in graph.inputs]
    if corruption == "count":
        inputs.pop()
    elif corruption == "dtype":
        inputs[0] = inputs[0].astype(np.float16)
    elif corruption == "shape":
        inputs[0] = inputs[0].reshape(-1)
    elif corruption == "contiguity":
        inputs[0] = inputs[0][..., ::-1]
    elif corruption == "type":
        inputs[0] = None
    else:
        graph.nodes[0].attrs["causal"] = "yes"
    with pytest.raises((TypeError, ValueError)):
        executable(*inputs)


@pytest.mark.parametrize(
    "corruption",
    [
        "tape_type",
        "owner",
        "mutation",
        "count",
        "dtype",
        "shape",
        "contiguity",
        "type",
        "mask_seed",
    ],
)
def test_backward_contract_validation_precedes_metal_initialization(monkeypatch, corruption):
    def forbidden():
        raise AssertionError("invalid cotangents must not initialize Metal")

    monkeypatch.setattr(MetalDevice, "get", forbidden)
    builder = GraphBuilder()
    inputs = _attention_inputs(builder)
    mask = builder.input("mask", TensorSpec((1, 4, 3, 5), "u8"))
    output = stable_attention(builder, *inputs, mask=mask)
    graph = builder.build((output, mask))
    executable = compile_native_attention_graph(graph)
    tape = NativeAttentionGraphTape(
        executable, _signature(graph), (), (SimpleNamespace(node=graph.nodes[0]),)
    )
    seeds = [np.zeros(output.spec.shape, dtype=np.float32), None]
    if corruption == "tape_type":
        tape = None
    elif corruption == "owner":
        tape = replace(tape, executable=compile_native_attention_graph(graph))
    elif corruption == "mutation":
        graph.nodes[0].attrs["scale"] *= 2
    elif corruption == "count":
        seeds.pop()
    elif corruption == "dtype":
        seeds[0] = seeds[0].astype(np.int32)
    elif corruption == "shape":
        seeds[0] = seeds[0].reshape(-1)
    elif corruption == "contiguity":
        seeds[0] = seeds[0][..., ::-1]
    elif corruption == "type":
        seeds[0] = 1.0
    else:
        seeds[1] = np.zeros(mask.spec.shape, dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        executable.backward(tape, seeds)
