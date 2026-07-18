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


__all__ = ["lower_dense_swiglu"]
