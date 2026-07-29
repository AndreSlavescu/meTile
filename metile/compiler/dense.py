from __future__ import annotations

from metile.ir import metal_ir as mir
from metile.ir.types import PtrType


def lower_dense_swiglu(
    function_name: str,
    rows: int,
    output_features: int,
    input_features: int,
    *,
    block_m: int = 64,
    block_n: int = 128,
    schedule: str = "linear",
    k_unroll: int = 2,
    fast_math: bool = False,
) -> mir.MFunction:
    """Build a ragged dense gate/up projection with a register-resident SwiGLU epilogue."""
    if rows < 1:
        raise ValueError("dense SwiGLU rows must be positive")
    if block_m < 32 or block_n < 32 or block_m % 32 or block_n % 32:
        raise ValueError("dense SwiGLU M/N tiles must align to 32")
    if output_features % block_n:
        raise ValueError("dense SwiGLU output features must align to the N tile")
    if block_m * block_n > 32 * 1024:
        raise ValueError("dense SwiGLU tile requires more than 1024 threads")
    if k_unroll not in {1, 2} or input_features % (16 * k_unroll):
        raise ValueError("dense SwiGLU K must align to its 16-element unroll")

    wm = block_m // 32
    wn = block_n // 32
    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(wm * wn * 32, 1, 1),
    )
    parameter_specs = (
        ("activations", False),
        ("gate_weight", False),
        ("up_weight", False),
        ("output", True),
    )
    function.params = [
        mir.MParam(name, PtrType("f16"), is_output=is_output) for name, is_output in parameter_specs
    ]
    values = {name: mir.MValue(name, PtrType("f16")) for name, _ in parameter_specs}

    function.add_op(mir.ThreadgroupPositionInGrid())
    function.add_op(mir.MSimdgroupId())
    function.add_op(mir.MThreadInSimdgroup())
    function.add_op(
        mir.MTileSchedule(
            pattern=schedule,
            block_m=block_m,
            block_n=block_n,
            block_size=4,
            grid_m=(rows + block_m - 1) // block_m,
            grid_n=output_features // block_n,
        )
    )
    function.add_op(
        mir.MNaxTileLayout(
            block_m=block_m,
            block_n=block_n,
            wn=wn,
            m=rows,
            n=output_features,
            k=input_features,
        )
    )
    gate_accumulators = ("gate00", "gate01", "gate10", "gate11")
    up_accumulators = ("up00", "up01", "up10", "up11")
    function.add_op(mir.MNaxAccumulatorInit(names=gate_accumulators + up_accumulators))
    function.add_op(mir.MNaxMatmul2dDecl(left_type="half", right_type="half"))

    body = []
    for k_offset in range(0, 16 * k_unroll, 16):
        suffix = str(k_offset // 16)
        activation_low = f"activation_low_{suffix}"
        activation_high = f"activation_high_{suffix}"
        body.extend(
            (
                mir.MNaxLoadFragment(
                    ptr=values["gate_weight"],
                    name=f"gate_low_{suffix}",
                    operand="right",
                    k_offset=k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=values["gate_weight"],
                    name=f"gate_high_{suffix}",
                    operand="right",
                    col_offset=16,
                    k_offset=k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=values["up_weight"],
                    name=f"up_low_{suffix}",
                    operand="right",
                    k_offset=k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=values["up_weight"],
                    name=f"up_high_{suffix}",
                    operand="right",
                    col_offset=16,
                    k_offset=k_offset,
                ),
                mir.MNaxLoadFragment(
                    ptr=values["activations"],
                    name=activation_low,
                    operand="left",
                    k_offset=k_offset,
                    row_bound=rows,
                ),
                mir.MNaxLoadFragment(
                    ptr=values["activations"],
                    name=activation_high,
                    operand="left",
                    row_offset=16,
                    k_offset=k_offset,
                    row_bound=rows,
                ),
                mir.MNaxPackRight(
                    low=f"gate_low_{suffix}",
                    high=f"gate_high_{suffix}",
                ),
                mir.MNaxFmaFragment(
                    left=activation_low,
                    destination_low=gate_accumulators[0],
                    destination_high=gate_accumulators[1],
                ),
                mir.MNaxFmaFragment(
                    left=activation_high,
                    destination_low=gate_accumulators[2],
                    destination_high=gate_accumulators[3],
                ),
                mir.MNaxPackRight(
                    low=f"up_low_{suffix}",
                    high=f"up_high_{suffix}",
                ),
                mir.MNaxFmaFragment(
                    left=activation_low,
                    destination_low=up_accumulators[0],
                    destination_high=up_accumulators[1],
                ),
                mir.MNaxFmaFragment(
                    left=activation_high,
                    destination_low=up_accumulators[2],
                    destination_high=up_accumulators[3],
                ),
            )
        )
    function.add_op(
        mir.MForLoop(
            iv_name="k",
            start=0,
            end=input_features,
            step=16 * k_unroll,
            body=body,
        )
    )
    for gate, up in zip(gate_accumulators, up_accumulators, strict=True):
        function.add_op(
            mir.MNaxBinaryFragment(
                left=gate,
                right=up,
                operation="swiglu",
                fast_math=fast_math,
                round_inputs="half",
                round_intermediates="half",
            )
        )
    for source, row_offset, col_offset in (
        (gate_accumulators[0], 0, 0),
        (gate_accumulators[1], 0, 16),
        (gate_accumulators[2], 16, 0),
        (gate_accumulators[3], 16, 16),
    ):
        function.add_op(
            mir.MNaxStoreFragment(
                ptr_c=values["output"],
                source=source,
                row_offset=row_offset,
                col_offset=col_offset,
                row_bound=rows,
            )
        )
    return function


def lower_dense_swiglu_qmv(
    function_name: str,
    output_features: int,
    input_features: int,
    *,
    outputs_per_simdgroup: int = 4,
    simdgroups_per_threadgroup: int = 4,
    interleaved: bool = False,
    k_unroll: int = 1,
    rows: int = 1,
) -> mir.MFunction:
    """Build an exact SwiGLU from output-major SIMDgroup dot pairs.

    ``rows`` > 1 shares one pass over the gate/up weights across a small batch of
    activation rows, which is the regime speculative decoding runs in.
    """
    if outputs_per_simdgroup not in {1, 2, 4}:
        raise ValueError("dense SwiGLU QMV supports 1, 2, or 4 outputs per SIMDgroup")
    if simdgroups_per_threadgroup not in {1, 2, 4, 8}:
        raise ValueError("dense SwiGLU QMV supports 1, 2, 4, or 8 SIMDgroups")
    if output_features % (outputs_per_simdgroup * simdgroups_per_threadgroup):
        raise ValueError("dense SwiGLU QMV outputs must align to the threadgroup tile")
    if input_features % 128:
        raise ValueError("dense SwiGLU QMV input features must align to 128")
    if k_unroll not in {1, 2}:
        raise ValueError("dense SwiGLU QMV K unroll must be 1 or 2")
    if rows < 1:
        raise ValueError("dense SwiGLU QMV requires at least one row")

    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(simdgroups_per_threadgroup * 32, 1, 1),
    )
    parameter_specs = (
        (("activations", False), ("paired_weight", False), ("output", True))
        if interleaved
        else (
            ("activations", False),
            ("gate_weight", False),
            ("up_weight", False),
            ("output", True),
        )
    )
    function.params = [
        mir.MParam(name, PtrType("f16"), is_output=is_output) for name, is_output in parameter_specs
    ]
    values = {name: mir.MValue(name, PtrType("f16")) for name, _ in parameter_specs}

    function.add_op(mir.ThreadgroupPositionInGrid())
    function.add_op(mir.MSimdgroupId())
    function.add_op(mir.MThreadInSimdgroup())
    function.add_op(
        mir.MSimdgroupQMVLayout(
            outputs_per_simdgroup=outputs_per_simdgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        )
    )
    function.add_op(
        mir.MPairedDotAccumulatorInit(outputs_per_simdgroup=outputs_per_simdgroup, rows=rows)
    )
    blocks = input_features // 128
    unrolled_blocks = blocks - blocks % k_unroll

    def accumulate(k_offset=0):
        return mir.MPairedDotAccumulate(
            ptr_input=values["activations"],
            ptr_left=None if interleaved else values["gate_weight"],
            ptr_right=None if interleaved else values["up_weight"],
            ptr_interleaved=values["paired_weight"] if interleaved else None,
            input_features=input_features,
            outputs_per_simdgroup=outputs_per_simdgroup,
            k_offset=k_offset,
            rows=rows,
        )

    if unrolled_blocks:
        function.add_op(
            mir.MForLoop(
                iv_name="k",
                start=0,
                end=unrolled_blocks * 128,
                step=128 * k_unroll,
                body=[accumulate(offset * 128) for offset in range(k_unroll)],
            )
        )
    if unrolled_blocks < blocks:
        function.add_op(
            mir.MForLoop(
                iv_name="k",
                start=unrolled_blocks * 128,
                end=input_features,
                step=128,
                body=[accumulate()],
            )
        )
    function.add_op(
        mir.MPairedDotSwiGLUStore(
            ptr_output=values["output"],
            outputs_per_simdgroup=outputs_per_simdgroup,
            fast_math=False,
            round_intermediates="half",
            rows=rows,
            output_features=output_features,
        )
    )
    return function


def lower_dense_residual_qmv(
    function_name: str,
    output_features: int,
    input_features: int,
    *,
    outputs_per_simdgroup: int = 1,
    simdgroups_per_threadgroup: int = 1,
    rows: int = 1,
) -> mir.MFunction:
    """Build an exact output-major QMV with a fused residual epilogue.

    ``rows`` > 1 keeps the same weight-streaming schedule but carries one accumulator
    per activation row, so a small batch of tokens shares a single pass over the
    weights instead of re-reading them per token.
    """
    if outputs_per_simdgroup not in {1, 2, 4}:
        raise ValueError("dense residual QMV supports 1, 2, or 4 outputs per SIMDgroup")
    if simdgroups_per_threadgroup not in {1, 2, 4, 8}:
        raise ValueError("dense residual QMV supports 1, 2, 4, or 8 SIMDgroups")
    if output_features % (outputs_per_simdgroup * simdgroups_per_threadgroup):
        raise ValueError("dense residual QMV outputs must align to the threadgroup tile")
    if input_features % 128:
        raise ValueError("dense residual QMV input features must align to 128")
    if rows < 1:
        raise ValueError("dense residual QMV requires at least one row")

    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(simdgroups_per_threadgroup * 32, 1, 1),
    )
    parameter_specs = (
        ("activations", False),
        ("weight", False),
        ("residual", False),
        ("output", True),
    )
    function.params = [
        mir.MParam(name, PtrType("f16"), is_output=is_output) for name, is_output in parameter_specs
    ]
    values = {name: mir.MValue(name, PtrType("f16")) for name, _ in parameter_specs}

    function.add_op(mir.ThreadgroupPositionInGrid())
    function.add_op(mir.MSimdgroupId())
    function.add_op(mir.MThreadInSimdgroup())
    function.add_op(
        mir.MSimdgroupQMVLayout(
            outputs_per_simdgroup=outputs_per_simdgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        )
    )
    function.add_op(mir.MDotAccumulatorInit(outputs_per_simdgroup=outputs_per_simdgroup, rows=rows))
    function.add_op(
        mir.MForLoop(
            iv_name="k",
            start=0,
            end=input_features,
            step=128,
            body=[
                mir.MDotAccumulate(
                    ptr_input=values["activations"],
                    ptr_weight=values["weight"],
                    input_features=input_features,
                    outputs_per_simdgroup=outputs_per_simdgroup,
                    rows=rows,
                )
            ],
        )
    )
    function.add_op(
        mir.MDotResidualStore(
            ptr_residual=values["residual"],
            ptr_output=values["output"],
            outputs_per_simdgroup=outputs_per_simdgroup,
            round_intermediates="half",
            rows=rows,
            output_features=output_features,
        )
    )
    return function


__all__ = [
    "lower_dense_residual_qmv",
    "lower_dense_swiglu",
    "lower_dense_swiglu_qmv",
]
