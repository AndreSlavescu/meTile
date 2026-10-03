import pytest

from metile.ir import metal_ir as mir
from metile.ir.printer import print_metal_ir
from metile.ir.types import BOOL, I32, PtrType, ScalarType


def _vector_function(dtype):
    function = mir.MFunction("vector_memory")
    indices = tuple(mir.MValue(f"index{lane}", I32) for lane in range(4))
    masks = (mir.MValue("mask0", BOOL), None, mir.MValue("mask2", BOOL), None)
    fills = (None, mir.MValue("fill1", ScalarType(dtype)), mir.MValue("fill2", I32), None)
    loaded = function.add_op(
        mir.MVectorLoad(
            ptr=mir.MValue("source", PtrType(dtype)),
            indices=indices,
            masks=masks,
            others=fills,
            dtype=dtype,
        ),
        "loaded",
    )
    values = tuple(
        function.add_op(mir.MVectorExtract(value=loaded, lane=lane), f"scalar{lane}")
        for lane in range(4)
    )
    function.add_op(
        mir.MVectorStore(
            ptr=mir.MValue("output", PtrType(dtype)),
            indices=indices,
            values=values,
            masks=masks,
            dtype=dtype,
        )
    )
    return function


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_vector_load_printer_exposes_memory_type_indices_and_independent_masks_and_fills(dtype):
    printed = print_metal_ir(_vector_function(dtype))
    assert (
        f"  %loaded = vector_load(%source, dtype={dtype}, width=4, "
        "indices=(%index0, %index1, %index2, %index3), "
        "masks=(%mask0, none, %mask2, none), "
        f"fills=(none, %fill1, %fill2, none)) : vector<4, {dtype}>"
    ) in printed
    assert "MVectorLoad(...)" not in printed


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_vector_store_printer_exposes_memory_type_indices_values_and_independent_masks(dtype):
    printed = print_metal_ir(_vector_function(dtype))
    assert (
        f"  vector_store(%output, dtype={dtype}, width=4, "
        "indices=(%index0, %index1, %index2, %index3), "
        "values=(%scalar0, %scalar1, %scalar2, %scalar3), "
        "masks=(%mask0, none, %mask2, none))"
    ) in printed
    assert "MVectorStore(...)" not in printed


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("lane", range(4))
def test_vector_extract_printer_exposes_selected_lane_and_scalar_result_type(dtype, lane):
    printed = print_metal_ir(_vector_function(dtype))
    assert f"  %scalar{lane} = vector_extract(%loaded, lane={lane}) : {dtype}" in printed
    assert "MVectorExtract(...)" not in printed


def test_vector_printer_can_inspect_unbound_operations_before_result_assignment():
    function = _vector_function("f32")
    for operation in function.ops:
        operation.result = None
    printed = print_metal_ir(function)
    assert "  vector_load(%source, dtype=f32, width=4," in printed
    assert "  vector_extract(%loaded, lane=0)" in printed
