from contextlib import nullcontext

import numpy as np
import pytest

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering.common import LoweringError
from metile.compiler.passes import decompose_nax_fragments
from metile.ir import metal_ir as mir


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("element_type", ["float", "half"])
def test_nax_precision_survives_fragment_decomposition_and_emission(relaxed, element_type):
    function = mir.MFunction(
        "nax_precision",
        kernel_type="tensor_ops_gemm",
        ops=[
            mir.MNaxGemmSetup(
                left_type=element_type,
                right_type=element_type,
                relaxed=relaxed,
            )
        ],
    )

    decompose_nax_fragments(function)

    declaration = next(
        operation for operation in function.ops if isinstance(operation, mir.MNaxMatmul2dDecl)
    )
    assert declaration.relaxed is relaxed
    assert declaration.left_type == element_type
    assert declaration.right_type == element_type
    assert declaration.accumulator_type == "float"
    source = emit(function)
    precision_literal = "true" if relaxed else "false"
    assert f"16, 32, 16, false, false, {precision_literal}," in source


def test_existing_nax_setup_defaults_keep_relaxed_precision():
    function = mir.MFunction(
        "nax_default_precision", kernel_type="tensor_ops_gemm", ops=[mir.MNaxGemmSetup()]
    )
    decompose_nax_fragments(function)

    declaration = next(
        operation for operation in function.ops if isinstance(operation, mir.MNaxMatmul2dDecl)
    )
    assert declaration.relaxed is True
    assert mir.MNaxMatmul2dDecl().relaxed is True
    assert "16, 32, 16, false, false, true," in emit(function)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_strict_nax_gpu_precision_contract(dtype):
    """Strict MMA precision does not promise bit-exact results under global fast math."""
    from metile.runtime.metal_device import MetalDevice
    from metile_kernels.gemm import matmul

    if not MetalDevice.get().supports_tensor_ops:
        pytest.skip("NAX regression requires Metal tensor operations")
    rows = 63
    columns = 64
    inner = 64
    generator = np.random.default_rng(807)
    left = generator.normal(size=(rows, inner)).astype(dtype)
    right = generator.normal(size=(inner, columns)).astype(dtype)
    output = np.zeros((rows, columns), dtype=dtype)
    launcher = matmul.kernel_fn[(1, 1)]
    expectation = (
        pytest.raises(LoweringError, match="Strict f32 NAX packed fragment layout")
        if dtype == np.float32
        else nullcontext()
    )
    with expectation:
        launcher(
            left,
            right,
            output,
            rows,
            columns,
            inner,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=16,
            WM=2,
            WN=2,
            SWIZZLE="linear",
            NAX_FRAGMENTS=True,
            RELAXED_PRECISION=False,
        )
    if dtype == np.float32:
        return

    source = launcher._last_compiled.msl_source
    assert "16, 32, 16, false, false, false," in source
    expected = (left.astype(np.float64) @ right.astype(np.float64)).astype(dtype)
    np.testing.assert_allclose(output, expected, rtol=1e-3, atol=1e-3)
