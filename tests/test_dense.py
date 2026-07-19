from metile.compiler.dense import (
    lower_dense_residual_qmv,
    lower_dense_swiglu,
    lower_dense_swiglu_qmv,
)
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


def test_dense_swiglu_qmv_composes_exact_output_major_dot_pairs():
    function = lower_dense_swiglu_qmv(
        "dense_swiglu_qmv_ir",
        256,
        128,
        outputs_per_simdgroup=4,
        simdgroups_per_threadgroup=2,
    )
    operations = tuple(_walk(function.ops))

    layout = next(
        operation for operation in operations if isinstance(operation, mir.MSimdgroupQMVLayout)
    )
    accumulate = next(
        operation for operation in operations if isinstance(operation, mir.MPairedDotAccumulate)
    )
    store = next(
        operation for operation in operations if isinstance(operation, mir.MPairedDotSwiGLUStore)
    )

    assert function.threadgroup_size == (64, 1, 1)
    assert layout.outputs_per_simdgroup == 4
    assert layout.simdgroups_per_threadgroup == 2
    assert accumulate.input_features == 128
    assert accumulate.elements_per_lane == 4
    assert not store.fast_math
    assert store.round_intermediates == "half"


def test_dense_swiglu_qmv_unrolls_k_without_changing_lane_partition():
    function = lower_dense_swiglu_qmv(
        "dense_swiglu_qmv_unrolled_ir",
        256,
        384,
        outputs_per_simdgroup=4,
        simdgroups_per_threadgroup=2,
        k_unroll=2,
    )
    loops = [operation for operation in function.ops if isinstance(operation, mir.MForLoop)]
    unrolled = loops[0]
    tail = loops[1]

    assert (unrolled.start, unrolled.end, unrolled.step) == (0, 256, 256)
    assert [operation.k_offset for operation in unrolled.body] == [0, 128]
    assert all(operation.elements_per_lane == 4 for operation in unrolled.body)
    assert (tail.start, tail.end, tail.step) == (256, 384, 128)
    assert [operation.k_offset for operation in tail.body] == [0]


def test_dense_residual_qmv_composes_dot_and_residual_epilogue():
    function = lower_dense_residual_qmv(
        "dense_residual_qmv_ir",
        256,
        128,
        outputs_per_simdgroup=1,
        simdgroups_per_threadgroup=2,
    )
    operations = tuple(_walk(function.ops))

    accumulate = next(
        operation for operation in operations if isinstance(operation, mir.MDotAccumulate)
    )
    store = next(
        operation for operation in operations if isinstance(operation, mir.MDotResidualStore)
    )

    assert function.threadgroup_size == (64, 1, 1)
    assert accumulate.input_features == 128
    assert accumulate.outputs_per_simdgroup == 1
    assert accumulate.elements_per_lane == 4
    assert store.ptr_residual.name == "residual"
    assert store.round_intermediates == "half"
