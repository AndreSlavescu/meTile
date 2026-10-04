from dataclasses import FrozenInstanceError

import pytest

from metile.compiler.layout_optimizations import optimize_layout_conversions
from metile.ir import tile_ir as tir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import PtrType, TileType


def _function(size=64, layout=None):
    function = tir.Function("layouts")
    start = function.add_op(tir.Constant(value=0), "start")
    source = function.add_op(tir.Arange(start=start, size=size, layout=layout), "source")
    return function, source


def _conversion(function, value, layout, name):
    return function.add_op(tir.ConvertLayout(value=value, layout=layout), name)


def _conversions(function):
    return [operation for operation in function.ops if isinstance(operation, tir.ConvertLayout)]


def _cross_group(size=64):
    order = tuple(range(size.bit_length() - 1))
    return ThreadLayout(order[-1:] + order[:-1], xor_mask=7)


def _convert_physical(values, source, destination):
    return [
        values[source.owner(destination.logical_index(thread))] for thread in range(source.size)
    ]


def test_legacy_function_without_conversions_is_returned_without_copy():
    function, _ = _function()

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert report.conversions_before == report.conversions_after == 0
    assert report.conversions_removed == 0
    assert report.to_dict()["rewrites"] == ()


def test_identity_elimination_copies_ir_and_preserves_def_use_links():
    function, source = _function()
    identity = _conversion(function, source, ThreadLayout.identity(64), "identity")
    function.add_op(tir.BinOp(op="add", lhs=identity, rhs=identity), "doubled")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is not function
    assert len(_conversions(function)) == 1
    assert function.ops[-1].lhs is identity
    assert not _conversions(optimized)
    original = optimized.ops[1].result
    assert original is not source
    assert optimized.ops[-1].lhs is original
    assert optimized.ops[-1].rhs is original
    assert all(operation.result.defining_op is operation for operation in optimized.ops)
    assert report.identities_removed == report.conversions_removed == 1
    assert report.rewrites[0].replacement == "source"


def test_cross_group_inverse_pair_restores_original_without_communication():
    function, source = _function()
    exchanged = _conversion(function, source, _cross_group(), "exchanged")
    restored = _conversion(function, exchanged, ThreadLayout.identity(64), "restored")
    function.add_op(tir.Unary(op="neg", operand=restored), "output")

    optimized, report = optimize_layout_conversions(function)

    assert not _conversions(optimized)
    assert optimized.ops[-1].operand is optimized.ops[1].result
    assert report.inverse_pairs_cancelled == 1
    assert report.conversions_before == 2
    assert report.conversions_after == 0
    assert len(_conversions(function)) == 2


def test_inverse_preserves_intermediate_conversion_with_another_consumer():
    function, source = _function()
    exchanged = _conversion(function, source, _cross_group(), "exchanged")
    restored = _conversion(function, exchanged, ThreadLayout.identity(64), "restored")
    function.add_op(tir.Unary(op="neg", operand=exchanged), "side_output")
    function.add_op(tir.Unary(op="neg", operand=restored), "output")

    optimized, report = optimize_layout_conversions(function)

    assert len(_conversions(optimized)) == 1
    assert optimized.ops[-2].operand is _conversions(optimized)[0].result
    assert optimized.ops[-1].operand is optimized.ops[1].result
    assert report.inverse_pairs_cancelled == report.conversions_removed == 1


@pytest.mark.parametrize("size", [32, 64, 256])
@pytest.mark.parametrize("xor_mask", [0, 1, 7, 23])
def test_adjacent_composition_preserves_every_physical_value(size, xor_mask):
    initial = ThreadLayout.identity(size)
    middle = _cross_group(size)
    destination = ThreadLayout(initial.bit_order, xor_mask=xor_mask)
    values = [index * 1009 - 31 for index in range(size)]
    sequential = _convert_physical(_convert_physical(values, initial, middle), middle, destination)
    direct = _convert_physical(values, initial, destination)
    assert sequential == direct
    function, source = _function(size, initial)
    temporary = _conversion(function, source, middle, "temporary")
    _conversion(function, temporary, destination, "final")

    optimized, report = optimize_layout_conversions(function)

    if xor_mask:
        assert report.chains_composed == 1
        assert len(_conversions(optimized)) == 1
        conversion = _conversions(optimized)[0]
        assert conversion.value is optimized.ops[1].result
        assert conversion.layout == destination
        assert conversion.result.defining_op is conversion
    else:
        assert report.inverse_pairs_cancelled == 1
        assert not _conversions(optimized)


def test_composition_retains_first_conversion_used_by_other_operations():
    function, source = _function()
    first = _conversion(function, source, _cross_group(), "first")
    _conversion(function, first, ThreadLayout(tuple(range(6)), xor_mask=1), "second")
    function.add_op(tir.Unary(op="neg", operand=first), "side_output")

    optimized, report = optimize_layout_conversions(function)

    first_operation, second_operation = _conversions(optimized)
    assert first_operation.value is second_operation.value
    assert optimized.ops[-1].operand is first_operation.result
    assert report.chains_composed == 1
    assert report.conversions_removed == 0


def test_conversion_chain_collapses_without_dangling_intermediate_values():
    function, source = _function()
    first = _conversion(function, source, _cross_group(), "first")
    second = _conversion(function, first, ThreadLayout(tuple(range(6)), xor_mask=3), "second")
    third = _conversion(function, second, ThreadLayout.identity(64), "third")
    function.add_op(tir.Unary(op="neg", operand=third), "output")

    optimized, report = optimize_layout_conversions(function)

    assert not _conversions(optimized)
    assert optimized.ops[-1].operand is optimized.ops[1].result
    assert report.chains_composed == report.inverse_pairs_cancelled == 1
    assert report.conversions_removed == 3


def test_load_masks_fill_values_and_shared_tensor_metadata_are_rewritten():
    function, source = _function()
    identity = _conversion(function, source, ThreadLayout.identity(64), "identity")
    pointer = tir.Value("source_pointer", PtrType("i32"))
    dimensions = (identity,)
    memory = tir.TensorMemory(pointer, shape=dimensions, strides=dimensions)
    function.tensors.append(memory)
    function.add_op(
        tir.Load(ptr=pointer, offsets=identity, mask=identity, other=identity, tensor=memory),
        "loaded",
    )

    optimized, _ = optimize_layout_conversions(function)

    replacement = optimized.ops[1].result
    load = optimized.ops[-1]
    assert load.offsets is load.mask is load.other is replacement
    assert load.tensor is optimized.tensors[0]
    assert optimized.tensors[0].shape[0] is replacement
    assert optimized.tensors[0].strides[0] is replacement
    assert function.tensors[0].shape[0] is identity


def test_metadata_only_side_use_preserves_intermediate_conversion():
    function, source = _function()
    first = _conversion(function, source, _cross_group(), "first")
    _conversion(function, first, ThreadLayout.identity(64), "restored")
    function.tensors.append(tir.TensorMemory(tir.Value("pointer", PtrType("i32")), (first,), ()))

    optimized, report = optimize_layout_conversions(function)

    assert len(_conversions(optimized)) == 1
    assert optimized.tensors[0].shape[0] is _conversions(optimized)[0].result
    assert report.conversions_removed == 1


def test_conversion_simplification_does_not_cross_an_intervening_barrier():
    function, source = _function()
    first = _conversion(function, source, _cross_group(), "first")
    function.add_op(tir.Barrier())
    _conversion(function, first, ThreadLayout.identity(64), "second")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert report.conversions_removed == 0
    assert len(_conversions(optimized)) == 2


def test_inconsistent_result_dtype_is_not_optimized():
    function, source = _function()
    identity = _conversion(function, source, ThreadLayout.identity(64), "identity")
    identity.type = TileType((64,), "f32", ThreadLayout.identity(64))

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert report.conversions_removed == 0
    assert "inconsistent conversion" in report.skipped[0]


def test_nested_region_rewrites_use_the_same_cloned_outer_value():
    function, source = _function()
    identity = _conversion(function, source, ThreadLayout.identity(64), "identity")
    use = tir.Unary(op="neg", operand=identity)
    use.result = tir.Value("nested", identity.type, use)
    function.add_op(tir.ForRange(body=[use]))

    optimized, report = optimize_layout_conversions(function)

    assert optimized.ops[-1].body[0].operand is optimized.ops[1].result
    assert function.ops[-1].body[0].operand is identity
    assert report.identities_removed == 1


def test_report_is_immutable_and_second_pass_is_a_noop():
    function, source = _function()
    _conversion(function, source, ThreadLayout.identity(64), "identity")

    optimized, report = optimize_layout_conversions(function)
    repeated, repeated_report = optimize_layout_conversions(optimized)

    assert repeated is optimized
    assert repeated_report.conversions_removed == 0
    with pytest.raises(FrozenInstanceError):
        report.identities_removed = 5


@pytest.mark.parametrize("size", [128, 512, 4096])
def test_four_register_identity_is_removed_without_changing_ownership(size):
    layout = ThreadLayout.identity(size, elements_per_thread=4)
    function, source = _function(size, layout)
    identity = _conversion(function, source, layout, "identity")
    function.add_op(tir.Unary(op="neg", operand=identity), "output")

    optimized, report = optimize_layout_conversions(function)

    assert not _conversions(optimized)
    assert optimized.ops[-1].operand is optimized.ops[1].result
    assert optimized.ops[-1].operand.type.layout.elements_per_thread == 4
    assert report.identities_removed == report.conversions_removed == 1


@pytest.mark.parametrize("source_elements", [1, 4])
def test_same_logical_size_with_different_register_geometry_is_untouched(source_elements):
    source_layout = ThreadLayout.identity(128, elements_per_thread=source_elements)
    destination = ThreadLayout.identity(128, elements_per_thread=4 // source_elements)
    function, source = _function(128, source_layout)
    first = _conversion(function, source, destination, "first")
    _conversion(function, first, source_layout, "restored")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert len(_conversions(optimized)) == 2
    assert len(report.skipped) == 2
    assert report.conversions_removed == 0


def test_implicit_layout_is_not_reinterpreted_as_four_register_identity():
    function, source = _function(128)
    destination = ThreadLayout.identity(128, elements_per_thread=4)
    _conversion(function, source, destination, "different_geometry")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert report.identities_removed == report.conversions_removed == 0
    assert len(report.skipped) == 1


def test_nonidentity_four_register_round_trip_remains_outside_supported_proof():
    source_layout = ThreadLayout.identity(128, elements_per_thread=4)
    destination = ThreadLayout(tuple(range(7)), xor_mask=1, elements_per_thread=4)
    function, source = _function(128, source_layout)
    first = _conversion(function, source, destination, "first")
    _conversion(function, first, source_layout, "restored")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert len(_conversions(optimized)) == 2
    assert len(report.skipped) == 2
    assert report.conversions_removed == 0


def test_unsupported_nonidentity_dtype_is_not_hidden_by_cancellation():
    function, source = _function()
    source.type = TileType((64,), "f64")
    first = _conversion(function, source, _cross_group(), "first")
    _conversion(function, first, ThreadLayout.identity(64), "restored")

    optimized, report = optimize_layout_conversions(function)

    assert optimized is function
    assert report.conversions_removed == 0
    assert len(report.skipped) == 2
