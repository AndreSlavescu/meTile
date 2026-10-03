from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import PtrType, ScalarType, TileType, merge_tile_layouts


@pytest.mark.parametrize("size", [32, 64, 128, 256, 512, 1024])
def test_identity_layout_owns_one_element_per_thread(size):
    layout = ThreadLayout.identity(size)
    assert layout.size == size
    assert layout.elements_per_thread == 1
    assert [layout.logical_index(thread) for thread in range(size)] == list(range(size))
    assert [layout.owner(index) for index in range(size)] == list(range(size))


@pytest.mark.parametrize("xor_mask", [0, 1, 13, 31])
def test_all_five_bit_permutations_are_bijections_with_exact_inverses(xor_mask):
    for bit_order in permutations(range(5)):
        layout = ThreadLayout(bit_order, xor_mask=xor_mask)
        logical = [layout.logical_index(thread) for thread in range(layout.size)]
        assert sorted(logical) == list(range(layout.size))
        assert [layout.owner(index) for index in logical] == list(range(layout.size))


@pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
def test_cross_simdgroup_bit_permutations_and_masks_have_exact_owners(size):
    bits = tuple(range(size.bit_length() - 1))
    layout = ThreadLayout(bits[-1:] + bits[:-1], xor_mask=size - 7)
    for index in range(size):
        assert layout.logical_index(layout.owner(index)) == index
        assert layout.owner(layout.logical_index(index)) == index


def test_bit_order_maps_physical_source_bits_to_logical_destinations():
    layout = ThreadLayout((3, 4, 0, 1, 2))
    assert [layout.logical_index(thread) for thread in range(8)] == list(range(0, 32, 4))
    assert [layout.logical_index(thread) for thread in range(8, 16)] == list(range(1, 32, 4))


@pytest.mark.parametrize(
    "bit_order",
    [
        (),
        (0, 1, 2, 3),
        tuple(range(11)),
        (0, 1, 2, 3, 3),
        (0, 1, 2, 3, 5),
        (0, 1, 2, 3, -1),
        (False, 1, 2, 3, 4),
        (0, 1, 2, 3, 4.0),
        [0, 1, 2, 3, 4],
        None,
    ],
)
def test_invalid_bit_orders_are_rejected(bit_order):
    with pytest.raises(ValueError, match="bit_order"):
        ThreadLayout(bit_order)


@pytest.mark.parametrize("xor_mask", [-1, 32, 1.5, True, None])
def test_invalid_xor_masks_are_rejected(xor_mask):
    with pytest.raises(ValueError, match="xor_mask"):
        ThreadLayout(tuple(range(5)), xor_mask=xor_mask)


@pytest.mark.parametrize("size", [0, 1, 16, 33, 2048, True, 32.0, None])
def test_invalid_identity_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="layout size"):
        ThreadLayout.identity(size)


@pytest.mark.parametrize("index", [-1, 32, True, 2.0, None])
def test_mapping_rejects_out_of_domain_indices(index):
    layout = ThreadLayout.identity(32)
    with pytest.raises(ValueError, match="thread index"):
        layout.logical_index(index)
    with pytest.raises(ValueError, match="logical index"):
        layout.owner(index)


def test_layouts_are_immutable_hashable_values():
    layout = ThreadLayout.identity(32)
    assert layout == ThreadLayout((0, 1, 2, 3, 4))
    assert {layout: "value"}[ThreadLayout.identity(32)] == "value"
    with pytest.raises(FrozenInstanceError):
        layout.xor_mask = 1


def test_explicit_layout_requires_matching_one_dimensional_tile():
    layout = ThreadLayout.identity(32)
    assert TileType((32,), "f32", layout).layout == layout
    for shape in [(64,), (4, 8), (16,), ()]:
        with pytest.raises(ValueError, match="one-dimensional tile"):
            TileType(shape, "f32", layout)
    with pytest.raises(TypeError, match="ThreadLayout"):
        TileType((32,), "f32", "identity")


def test_legacy_tile_types_and_representations_are_unchanged():
    tile = TileType((4, 8), "f32")
    assert tile.layout is None
    assert repr(tile) == "tile<4x8, f32>"
    assert merge_tile_layouts(tile, TileType((32,), "i32"), ScalarType("f32")) is None


def test_implicit_identity_combines_only_with_explicit_identity():
    implicit = TileType((32,), "f32")
    identity = ThreadLayout.identity(32)
    assert merge_tile_layouts(implicit, TileType((32,), "f32", identity)) == identity
    with pytest.raises(ValueError, match="convert_layout"):
        merge_tile_layouts(implicit, TileType((32,), "f32", ThreadLayout((1, 0, 2, 3, 4))))


def test_layout_merge_checks_shapes_and_allows_scalar_broadcast():
    layout = ThreadLayout((4, 0, 1, 2, 3))
    tile = TileType((32,), "f32", layout)
    assert merge_tile_layouts(ScalarType("f32"), tile, ScalarType("i32")) == layout
    with pytest.raises(ValueError, match="matching one-dimensional tile shapes"):
        merge_tile_layouts(tile, TileType((64,), "f32"))


def _pointer(name="source", dtype="f32"):
    return TracingProxy(tir.Value(name, PtrType(dtype)))


def test_public_layout_api_traces_arange_conversion_and_reconversion():
    destination = metile.ThreadLayout((3, 4, 0, 1, 2))
    with TracingContext("redistribute") as context:
        original = metile.arange(0, 32)
        converted = metile.convert_layout(original, destination)
        restored = metile.convert_layout(converted, metile.ThreadLayout.identity(32))
    assert original._value.type.layout is None
    assert converted._value.type == TileType((32,), "i32", destination)
    assert restored._value.type == TileType((32,), "i32", ThreadLayout.identity(32))
    assert isinstance(context.func.ops[1], tir.ConvertLayout)
    assert context.func.ops[1].value is original._value
    assert context.func.ops[1].layout == destination


@pytest.mark.parametrize("size", [16, 64])
def test_arange_layout_size_mismatch_fails_at_trace_time(size):
    with TracingContext("bad_size"), pytest.raises(ValueError, match="matching size"):
        metile.arange(0, size, layout=ThreadLayout.identity(32))


def test_conversion_rejects_non_tile_and_invalid_destination():
    with TracingContext("bad_conversion"):
        tile = metile.arange(0, 32)
        with pytest.raises(TypeError, match="tile value"):
            metile.convert_layout(7, ThreadLayout.identity(32))
        with pytest.raises(TypeError, match="ThreadLayout destination"):
            metile.convert_layout(tile, None)
        with pytest.raises(ValueError, match="matching size"):
            metile.convert_layout(tile, ThreadLayout.identity(64))


def test_pointwise_operations_propagate_layouts():
    layout = ThreadLayout((1, 0, 2, 3, 4), xor_mask=3)
    with TracingContext("propagation"):
        positions = metile.arange(-2, 30, layout=layout)
        floating = metile.cast(positions, "f32")
        values = [
            positions + 1,
            1 + positions,
            floating * 2.0,
            metile.abs(floating),
            metile.reverse_bits(positions),
            positions < 20,
            metile.where(positions < 20, floating, -1.0),
            metile.where(positions < 20, 1.0, -1.0),
            metile.where(metile.scalar(1), floating, floating + 1),
        ]
    assert all(value._value.type.layout == layout for value in values)
    assert values[4]._value.type.dtype == "u32"


@pytest.mark.parametrize("operation", ["add", "compare", "select_condition", "select_value"])
def test_pointwise_operations_reject_mismatched_ownership(operation):
    with TracingContext("mismatch"):
        first = metile.arange(0, 32, layout=ThreadLayout.identity(32))
        second = metile.arange(0, 32, layout=ThreadLayout((1, 0, 2, 3, 4)))
        with pytest.raises(ValueError, match="convert_layout"):
            if operation == "add":
                first + second
            elif operation == "compare":
                first.__lt__(second)
            elif operation == "select_condition":
                metile.where(second < 20, first, 0)
            else:
                metile.where(first < 20, first, second)


def test_descriptor_loads_stores_and_masks_keep_ownership():
    layout = ThreadLayout((1, 0, 2, 3, 4))
    with TracingContext("descriptor_copy") as context:
        inputs = metile.tensor(_pointer(), shape=(27,), access="read")
        outputs = metile.tensor(_pointer("output"), shape=(29,), access="write")
        positions = metile.arange(0, 32, layout=layout)
        loaded = inputs.load((positions,), other=-4)
        outputs.store((positions,), loaded + 3)
    load = next(operation for operation in context.func.ops if isinstance(operation, tir.Load))
    store = next(operation for operation in context.func.ops if isinstance(operation, tir.Store))
    assert loaded._value.type.layout == layout
    for value in (load.offsets, load.mask, store.offsets, store.mask, store.value):
        assert value.type.layout == layout


def test_descriptor_store_rejects_values_with_different_ownership():
    with TracingContext("bad_store"):
        outputs = metile.tensor(_pointer("output"), shape=(32,), access="write")
        positions = metile.arange(0, 32)
        converted = metile.convert_layout(positions, ThreadLayout((1, 0, 2, 3, 4)))
        with pytest.raises(ValueError, match="convert_layout"):
            outputs.store((positions,), converted)


def test_descriptor_store_allows_uniform_scalar_broadcast():
    layout = ThreadLayout((1, 0, 2, 3, 4))
    with TracingContext("scalar_store") as context:
        outputs = metile.tensor(_pointer("output"), shape=(32,), access="write")
        positions = metile.arange(0, 32, layout=layout)
        outputs.store((positions,), 4.0)
    assert isinstance(context.func.ops[-1], tir.Store)


def test_multiple_tensor_coordinates_must_have_matching_ownership():
    with TracingContext("bad_coordinates"):
        inputs = metile.tensor(_pointer(), shape=(32, 32), access="read")
        identity = metile.arange(0, 32)
        permuted = metile.arange(0, 32, layout=ThreadLayout((1, 0, 2, 3, 4)))
        with pytest.raises(ValueError, match="convert_layout"):
            inputs.load((identity, permuted))


def test_load_and_store_ir_reject_mask_ownership_mismatches():
    pointer = tir.Value("pointer", PtrType("f32"))
    positions = tir.Value("positions", TileType((32,), "i32", ThreadLayout.identity(32)))
    mask = tir.Value("mask", TileType((32,), "bool", ThreadLayout((1, 0, 2, 3, 4))))
    value = tir.Value("value", TileType((32,), "f32", ThreadLayout.identity(32)))
    with pytest.raises(ValueError, match="convert_layout"):
        tir.Load(ptr=pointer, offsets=positions, mask=mask).result_type()
    with pytest.raises(ValueError, match="convert_layout"):
        tir.Store(ptr=pointer, offsets=positions, value=value, mask=mask).result_type()
