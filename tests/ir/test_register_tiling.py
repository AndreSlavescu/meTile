from itertools import pairwise

import pytest

from metile.ir.ownership import ThreadLayout
from metile.ir.types import ScalarType, TileType, merge_tile_layouts


@pytest.mark.parametrize("elements", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("threads", [32, 64, 256, 1024])
def test_supported_register_tiles_have_exact_identity_geometry(elements, threads):
    layout = ThreadLayout.identity(elements * threads, elements_per_thread=elements)
    assert layout.size == elements * threads
    assert layout.thread_count == threads
    assert layout.elements_per_thread == elements
    for register in range(elements):
        for thread in (0, threads // 2, threads - 1):
            logical = thread + register * threads
            assert layout.logical_index(thread, register) == logical
            assert layout.owner(logical) == thread
            assert layout.register(logical) == register


@pytest.mark.parametrize("elements", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("threads", [32, 1024])
def test_register_major_tiles_cover_every_logical_element_exactly_once(elements, threads):
    thread_bits = threads.bit_length() - 1
    register_bits = elements.bit_length() - 1
    layout = ThreadLayout(
        (*range(thread_bits, thread_bits + register_bits), *range(thread_bits)),
        xor_mask=elements * threads - 1,
        elements_per_thread=elements,
    )
    assert {
        layout.logical_index(thread, register)
        for thread in range(threads)
        for register in range(elements)
    } == set(range(layout.size))
    for logical in range(layout.size):
        assert layout.logical_index(layout.owner(logical), layout.register(logical)) == logical


@pytest.mark.parametrize("elements", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("threads", [16, 2048])
def test_register_count_does_not_relax_physical_thread_bounds(elements, threads):
    with pytest.raises(ValueError, match="layout size"):
        ThreadLayout.identity(elements * threads, elements_per_thread=elements)
    with pytest.raises(ValueError, match="bit_order"):
        ThreadLayout(
            tuple(range((elements * threads).bit_length() - 1)), elements_per_thread=elements
        )


@pytest.mark.parametrize("elements", [2, 8, 16, 32])
def test_new_register_counts_keep_index_bounds_and_implicit_layout_safety(elements):
    layout = ThreadLayout.identity(32 * elements, elements_per_thread=elements)
    with pytest.raises(ValueError, match="register index"):
        layout.logical_index(0, elements)
    with pytest.raises(ValueError, match="thread index"):
        layout.logical_index(32, 0)
    with pytest.raises(ValueError, match="logical index"):
        layout.register(layout.size)
    explicit = TileType((layout.size,), "f32", layout)
    assert merge_tile_layouts(explicit, ScalarType("f32"), explicit) == layout
    with pytest.raises(ValueError, match="convert_layout"):
        merge_tile_layouts(explicit, TileType((layout.size,), "f32"))


def test_fixed_logical_tile_supports_distinct_checked_register_thread_partitions():
    layouts = [
        ThreadLayout.identity(1024, elements_per_thread=count) for count in (1, 2, 4, 8, 16, 32)
    ]
    assert [layout.thread_count for layout in layouts] == [1024, 512, 256, 128, 64, 32]
    assert len(set(layouts)) == 6
    for first, second in pairwise(layouts):
        with pytest.raises(ValueError, match="convert_layout"):
            merge_tile_layouts(TileType((1024,), "f32", first), TileType((1024,), "f32", second))
