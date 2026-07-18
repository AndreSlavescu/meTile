from __future__ import annotations

from metile.ir import metal_ir as mir
from metile.ir.types import PtrType


def lower_affine_matmul(
    function_name: str,
    rows: int,
    output_features: int,
    input_features: int,
    *,
    block_n: int = 64,
    group_size: int = 64,
    schedule: str = "linear",
) -> mir.MFunction:
    """Build a ragged affine uint4 matmul from native Metal 4 tensor operations."""
    if rows < 1:
        raise ValueError("affine matmul rows must be positive")
    if output_features % block_n or block_n % 32:
        raise ValueError("affine QMV output and block sizes must align to 32")
    if input_features % group_size or group_size % 16:
        raise ValueError("affine QMV input and group sizes must align to 16")
    if group_size != 64:
        raise ValueError("native affine QMV currently supports group size 64")

    block_m = 32
    simdgroups = block_n // 32
    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(simdgroups * 32, 1, 1),
    )
    function.params = [
        mir.MParam("activations", PtrType("f16")),
        mir.MParam("packed", PtrType("u8")),
        mir.MParam("scales", PtrType("f16")),
        mir.MParam("biases", PtrType("f16")),
        mir.MParam("output", PtrType("f16"), is_output=True),
    ]
    activations = mir.MValue("activations", PtrType("f16"))
    packed = mir.MValue("packed", PtrType("u8"))
    scales = mir.MValue("scales", PtrType("f16"))
    biases = mir.MValue("biases", PtrType("f16"))
    output = mir.MValue("output", PtrType("f16"))

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
        mir.MNaxGemmSetup(
            block_m=block_m,
            block_n=block_n,
            wm=1,
            wn=simdgroups,
            m=rows,
            n=output_features,
            k=input_features,
            left_type="half",
            right_type="half",
        )
    )
    function.add_op(
        mir.MForLoop(
            iv_name="k",
            start=0,
            end=input_features,
            step=group_size,
            body=[
                mir.MNaxAffineRun(
                    ptr_a=activations,
                    ptr_values=packed,
                    ptr_scales=scales,
                    ptr_biases=biases,
                    group_size=group_size,
                    fragment_type="half",
                    row_bound=rows,
                    k_offset=offset,
                )
                for offset in range(0, group_size, 16)
            ],
        )
    )
    function.add_op(mir.MNaxGemmStore(ptr_c=output, row_bound=rows))
    return function


def lower_affine_qmv(
    function_name: str,
    output_features: int,
    input_features: int,
    *,
    block_n: int = 64,
    group_size: int = 64,
    schedule: str = "linear",
) -> mir.MFunction:
    """Build a one-row affine uint4 QMV with native Metal 4 tensor operations."""
    return lower_affine_matmul(
        function_name,
        1,
        output_features,
        input_features,
        block_n=block_n,
        group_size=group_size,
        schedule=schedule,
    )


def lower_affine_swiglu_qmv(
    function_name: str,
    output_features: int,
    input_features: int,
    *,
    block_n: int = 64,
    group_size: int = 64,
    schedule: str = "linear",
    lifetime_schedule: str = "parallel",
) -> mir.MFunction:
    """Build fused gate/up affine QMV from composable NAX primitives."""
    if output_features % block_n or block_n % 32:
        raise ValueError("affine QMV output and block sizes must align to 32")
    if input_features % group_size or group_size != 64:
        raise ValueError("native affine QMV currently requires group size 64")
    if lifetime_schedule not in {"parallel", "scratch"}:
        raise ValueError("affine SwiGLU lifetime schedule must be parallel or scratch")

    block_m = 32
    simdgroups = block_n // 32
    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(simdgroups * 32, 1, 1),
    )
    parameter_specs = (
        ("activations", "f16", False),
        ("gate_packed", "u8", False),
        ("gate_scales", "f16", False),
        ("gate_biases", "f16", False),
        ("up_packed", "u8", False),
        ("up_scales", "f16", False),
        ("up_biases", "f16", False),
        ("output", "f16", True),
    )
    function.params = [
        mir.MParam(name, PtrType(dtype), is_output=is_output)
        for name, dtype, is_output in parameter_specs
    ]
    values = {name: mir.MValue(name, PtrType(dtype)) for name, dtype, _ in parameter_specs}

    function.add_op(mir.ThreadgroupPositionInGrid())
    function.add_op(mir.MSimdgroupId())
    function.add_op(mir.MThreadInSimdgroup())
    function.add_op(
        mir.MTileSchedule(
            pattern=schedule,
            block_m=block_m,
            block_n=block_n,
            block_size=4,
            grid_m=1,
            grid_n=output_features // block_n,
        )
    )
    function.add_op(
        mir.MNaxTileLayout(
            block_m=block_m,
            block_n=block_n,
            wn=simdgroups,
            m=1,
            n=output_features,
            k=input_features,
        )
    )
    gate_accumulators = ("gate00", "gate01")
    up_accumulators = ("up00", "up01")
    if lifetime_schedule == "parallel":
        function.add_op(mir.MNaxAccumulatorInit(names=gate_accumulators + up_accumulators))
    else:
        scratch_slots = len(gate_accumulators)
        function.add_op(
            mir.MThreadgroupAlloc(
                alloc_name="nax_gate_scratch",
                elem_type="float",
                size=simdgroups * scratch_slots * 32 * 8,
            )
        )
        function.add_op(mir.MNaxAccumulatorInit(names=gate_accumulators))
    function.add_op(mir.MNaxMatmul2dDecl(left_type="half", right_type="half"))

    def append_projection_parameters(body, prefix):
        for half_name, col_offset in (("low", 0), ("high", 16)):
            body.append(
                mir.MNaxLoadAffineParameters(
                    ptr_scales=values[f"{prefix}_scales"],
                    ptr_biases=values[f"{prefix}_biases"],
                    scale_name=f"{prefix}_{half_name}_scale",
                    bias_name=f"{prefix}_{half_name}_bias",
                    group_size=group_size,
                    col_offset=col_offset,
                )
            )

    def append_projection_step(
        body,
        prefix,
        accumulators,
        *,
        k_offset,
        activation,
        load_activation,
    ):
        suffix = str(k_offset // 16)
        right_low = f"{prefix}_right_low_{suffix}"
        right_high = f"{prefix}_right_high_{suffix}"
        body.extend(
            (
                mir.MNaxLoadAffineFragment(
                    ptr_values=values[f"{prefix}_packed"],
                    name=right_low,
                    scale=f"{prefix}_low_scale",
                    bias=f"{prefix}_low_bias",
                    k_offset=k_offset,
                ),
                mir.MNaxLoadAffineFragment(
                    ptr_values=values[f"{prefix}_packed"],
                    name=right_high,
                    scale=f"{prefix}_high_scale",
                    bias=f"{prefix}_high_bias",
                    col_offset=16,
                    k_offset=k_offset,
                ),
                mir.MNaxPackRight(low=right_low, high=right_high),
            )
        )
        if load_activation:
            body.append(
                mir.MNaxLoadFragment(
                    ptr=values["activations"],
                    name=activation,
                    operand="left",
                    k_offset=k_offset,
                    row_bound=1,
                )
            )
        body.append(
            mir.MNaxFmaFragment(
                left=activation,
                destination_low=accumulators[0],
                destination_high=accumulators[1],
            )
        )

    def projection_body(prefix, accumulators):
        body = []
        append_projection_parameters(body, prefix)
        for k_offset in range(0, group_size, 16):
            suffix = str(k_offset // 16)
            activation = f"activation_{suffix}"
            append_projection_step(
                body,
                prefix,
                accumulators,
                k_offset=k_offset,
                activation=activation,
                load_activation=True,
            )
        return body

    if lifetime_schedule == "parallel":
        body = []
        append_projection_parameters(body, "gate")
        append_projection_parameters(body, "up")
        for k_offset in range(0, group_size, 16):
            activation = f"activation_{k_offset // 16}"
            append_projection_step(
                body,
                "gate",
                gate_accumulators,
                k_offset=k_offset,
                activation=activation,
                load_activation=True,
            )
            append_projection_step(
                body,
                "up",
                up_accumulators,
                k_offset=k_offset,
                activation=activation,
                load_activation=False,
            )
        function.add_op(
            mir.MForLoop(iv_name="k", start=0, end=input_features, step=group_size, body=body)
        )
        for gate, up in zip(gate_accumulators, up_accumulators, strict=True):
            function.add_op(mir.MNaxBinaryFragment(left=gate, right=up, operation="swiglu"))
        output_sources = gate_accumulators
    else:
        function.add_op(
            mir.MForLoop(
                iv_name="k",
                start=0,
                end=input_features,
                step=group_size,
                body=projection_body("gate", gate_accumulators),
            )
        )
        for slot, source in enumerate(gate_accumulators):
            function.add_op(
                mir.MNaxSpillFragment(
                    source=source,
                    scratch_name="nax_gate_scratch",
                    slot=slot,
                    slots_per_simdgroup=scratch_slots,
                )
            )
        function.add_op(mir.MBarrier(kind="simdgroup", flags="mem_threadgroup"))
        function.add_op(mir.MNaxAccumulatorReset(names=gate_accumulators))
        function.add_op(
            mir.MForLoop(
                iv_name="k",
                start=0,
                end=input_features,
                step=group_size,
                body=projection_body("up", gate_accumulators),
            )
        )
        for slot, source in enumerate(gate_accumulators):
            function.add_op(
                mir.MNaxReloadFragment(
                    destination="nax_c",
                    scratch_name="nax_gate_scratch",
                    slot=slot,
                    slots_per_simdgroup=scratch_slots,
                )
            )
            function.add_op(
                mir.MNaxBinaryFragment(
                    left="nax_c",
                    right=source,
                    destination=source,
                    operation="swiglu",
                )
            )
        output_sources = gate_accumulators

    for source, col_offset in zip(output_sources, (0, 16), strict=True):
        function.add_op(
            mir.MNaxStoreFragment(
                ptr_c=values["output"],
                source=source,
                col_offset=col_offset,
                row_bound=1,
            )
        )
    return function


__all__ = ["lower_affine_matmul", "lower_affine_qmv", "lower_affine_swiglu_qmv"]
