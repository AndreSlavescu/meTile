from metile.compiler.dense import lower_dense_swiglu
from metile.ir import metal_ir as mir


def _walk(operations):
    for operation in operations:
        yield operation
        if isinstance(operation, mir.MForLoop):
            yield from _walk(operation.body)


def test_dense_swiglu_reuses_activation_fragments_and_masks_ragged_rows():
    function = lower_dense_swiglu(
        "dense_swiglu_ir",
        33,
        64,
        64,
        block_m=64,
        block_n=64,
        schedule="hilbert",
        k_unroll=2,
    )
    operations = tuple(_walk(function.ops))

    schedule = next(
        operation for operation in operations if isinstance(operation, mir.MTileSchedule)
    )
    loads = [operation for operation in operations if isinstance(operation, mir.MNaxLoadFragment)]
    fmas = [operation for operation in operations if isinstance(operation, mir.MNaxFmaFragment)]
    combines = [
        operation for operation in operations if isinstance(operation, mir.MNaxBinaryFragment)
    ]
    stores = [operation for operation in operations if isinstance(operation, mir.MNaxStoreFragment)]

    activation_names = {
        operation.name for operation in loads if operation.ptr.name == "activations"
    }
    assert schedule.pattern == "hilbert"
    assert schedule.grid_m == 1
    assert all(sum(fma.left == name for fma in fmas) == 2 for name in activation_names)
    assert len(combines) == 4
    assert all(combine.operation == "swiglu" for combine in combines)
    assert all(combine.round_inputs == "half" for combine in combines)
    assert all(combine.round_intermediates == "half" for combine in combines)
    assert all(not combine.fast_math for combine in combines)
    assert all(store.row_bound == 33 for store in stores)
