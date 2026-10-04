from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

import metile
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.ownership import ThreadLayout
from metile.ir.types import PtrType, ScalarType, TileType, merge_tile_layouts


@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512, 1024])
def test_four_register_identity_packs_threads_before_registers(threads):
    layout = ThreadLayout.identity(threads * 4, elements_per_thread=4)
    assert layout.size == threads * 4
    assert layout.thread_count == threads
    assert layout.elements_per_thread == 4
    for thread in range(threads):
        assert layout.logical_index(thread) == thread
        for register in range(4):
            logical = thread + register * threads
            assert layout.logical_index(thread, register) == logical
            assert layout.owner(logical) == thread
            assert layout.register(logical) == register


@pytest.mark.parametrize("threads", [32, 128, 1024])
def test_register_bit_permutation_can_make_four_adjacent_elements_per_thread(threads):
    thread_bits = threads.bit_length() - 1
    bit_order = (thread_bits, thread_bits + 1, *range(thread_bits))
    layout = ThreadLayout(bit_order, elements_per_thread=4)
    for thread in range(threads):
        for register in range(4):
            logical = thread * 4 + register
            assert layout.logical_index(thread, register) == logical
            assert layout.owner(logical) == thread
            assert layout.register(logical) == register


@pytest.mark.parametrize("xor_mask", [0, 35, 127])
def test_every_minimal_register_bit_permutation_has_exact_inverse_coordinates(xor_mask):
    for bit_order in permutations(range(7)):
        layout = ThreadLayout(bit_order, xor_mask=xor_mask, elements_per_thread=4)
        for logical in (0, 1, 31, 32, 63, 64, 127):
            thread = layout.owner(logical)
            register = layout.register(logical)
            assert 0 <= thread < 32
            assert 0 <= register < 4
            assert layout.logical_index(thread, register) == logical


@pytest.mark.parametrize("size", [128, 256, 512, 1024, 2048, 4096])
def test_cross_thread_register_and_xor_mappings_cover_every_logical_element(size):
    bits = tuple(range(size.bit_length() - 1))
    layout = ThreadLayout(tuple(reversed(bits)), xor_mask=size - 17, elements_per_thread=4)
    owned = [
        layout.logical_index(thread, register)
        for thread in range(layout.thread_count)
        for register in range(layout.elements_per_thread)
    ]
    assert sorted(owned) == list(range(size))
    for logical in range(size):
        assert layout.logical_index(layout.owner(logical), layout.register(logical)) == logical


def test_single_register_layout_constructor_and_owner_keep_existing_meaning():
    layout = ThreadLayout((1, 0, 2, 3, 4), 3)
    assert layout.elements_per_thread == 1
    assert layout.thread_count == layout.size == 32
    for thread in range(layout.thread_count):
        logical = layout.logical_index(thread)
        assert layout.logical_index(thread, 0) == logical
        assert layout.owner(logical) == thread
        assert layout.register(logical) == 0
    with pytest.raises(ValueError, match="register index"):
        layout.logical_index(0, 1)


@pytest.mark.parametrize("elements", [0, 3, 24, 64, -4, True, 4.0, None])
def test_unsupported_register_counts_are_rejected(elements):
    with pytest.raises(ValueError, match="elements_per_thread"):
        ThreadLayout(tuple(range(7)), elements_per_thread=elements)
    with pytest.raises(ValueError, match="elements_per_thread"):
        ThreadLayout.identity(128, elements_per_thread=elements)


@pytest.mark.parametrize("size", [0, 32, 64, 127, 129, 8192, True, 128.0, None])
def test_register_identity_size_requires_a_whole_legal_threadgroup(size):
    with pytest.raises(ValueError, match="layout size"):
        ThreadLayout.identity(size, elements_per_thread=4)


@pytest.mark.parametrize("bit_count", [5, 6, 13])
def test_register_constructor_rejects_too_few_or_too_many_threads(bit_count):
    with pytest.raises(ValueError, match="bit_order"):
        ThreadLayout(tuple(range(bit_count)), elements_per_thread=4)


@pytest.mark.parametrize("thread", [-1, 32, 128, True, 2.0, None])
def test_thread_lookup_is_bounded_by_physical_threads_not_logical_elements(thread):
    layout = ThreadLayout.identity(128, elements_per_thread=4)
    with pytest.raises(ValueError, match="thread index"):
        layout.logical_index(thread)


@pytest.mark.parametrize("register", [-1, 4, True, 2.0, None])
def test_register_lookup_is_bounded_by_physical_slots(register):
    layout = ThreadLayout.identity(128, elements_per_thread=4)
    with pytest.raises(ValueError, match="register index"):
        layout.logical_index(0, register)


@pytest.mark.parametrize("logical", [-1, 128, True, 2.0, None])
def test_both_inverse_coordinates_validate_logical_indices(logical):
    layout = ThreadLayout.identity(128, elements_per_thread=4)
    with pytest.raises(ValueError, match="logical index"):
        layout.owner(logical)
    with pytest.raises(ValueError, match="logical index"):
        layout.register(logical)


def test_register_count_is_part_of_immutable_layout_identity():
    scalar = ThreadLayout.identity(128)
    packed = ThreadLayout.identity(128, elements_per_thread=4)
    assert scalar != packed
    assert len({scalar, packed}) == 2
    assert scalar.owner(32) == 32
    assert packed.owner(32) == 0
    with pytest.raises(FrozenInstanceError):
        packed.elements_per_thread = 1


@pytest.mark.parametrize("size", [128, 1024, 4096])
def test_register_tile_types_keep_logical_size_and_scalar_broadcast(size):
    layout = ThreadLayout.identity(size, elements_per_thread=4)
    value_type = TileType((size,), "f32", layout)
    assert value_type.numel == size
    assert merge_tile_layouts(value_type, ScalarType("f32"), value_type) == layout
    with pytest.raises(ValueError, match="matching size"):
        TileType((layout.thread_count,), "f32", layout)


@pytest.mark.parametrize("size", [128, 1024, 4096])
def test_register_tiles_cannot_silently_reinterpret_implicit_single_value_ownership(size):
    layout = ThreadLayout.identity(size, elements_per_thread=4)
    with pytest.raises(ValueError, match="convert_layout"):
        merge_tile_layouts(TileType((size,), "f32", layout), TileType((size,), "f32"))


def test_identical_logical_bit_orders_with_different_thread_counts_cannot_combine():
    scalar = TileType((128,), "f32", ThreadLayout.identity(128))
    packed = TileType((128,), "f32", ThreadLayout.identity(128, elements_per_thread=4))
    with pytest.raises(ValueError, match="convert_layout"):
        merge_tile_layouts(scalar, packed)


def test_explicit_register_layout_propagates_through_tensor_math_and_reduction():
    layout = ThreadLayout.identity(1024, elements_per_thread=4)
    with TracingContext("register_norm") as context:
        inputs = metile.tensor(
            TracingProxy(tir.Value("source", PtrType("f16"))), shape=(1009,), access="read"
        )
        outputs = metile.tensor(
            TracingProxy(tir.Value("output", PtrType("f16"))), shape=(1009,), access="write"
        )
        positions = metile.arange(0, 1024, layout=layout)
        values = metile.cast(inputs.load((positions,)), "f32")
        squares = values * values
        total = metile.sum(squares)
        scaled = values / metile.sqrt(total / 1009 + 1e-5)
        converted = metile.convert_layout(scaled, layout)
        outputs.store((positions,), metile.cast(converted, "f16"))
    assert total._value.type == ScalarType("f32")
    for value in (positions, values, squares, scaled, converted):
        assert value._value.type.layout == layout
    store = context.func.ops[-1]
    assert isinstance(store, tir.Store)
    assert store.value.type.layout == store.offsets.type.layout == store.mask.type.layout == layout


def test_register_and_implicit_aranges_cannot_mix_in_public_pointwise_api():
    layout = ThreadLayout.identity(128, elements_per_thread=4)
    with TracingContext("mixed_ownership"):
        packed = metile.arange(0, 128, layout=layout)
        implicit = metile.arange(0, 128)
        with pytest.raises(ValueError, match="convert_layout"):
            packed + implicit
