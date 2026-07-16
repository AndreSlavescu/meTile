from __future__ import annotations

from metile.ir import metal_ir as mir
from metile.ir.types import PtrType


def lower_block_scaled_matmul(
    function_name: str,
    m: int,
    n: int,
    k: int,
    bits: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    register_fragments: bool = False,
    schedule: str = "auto",
    outer_k: int = 0,
    fragment_type: str = "float",
) -> mir.MFunction:
    """Build composable Metal IR for an aligned MXFP weight-only GEMM."""
    if bits not in {4, 8}:
        raise ValueError("block-scaled matmul supports 4-bit or 8-bit data")

    if block_m % 32 or block_n % 32 or block_k not in {32, 64}:
        raise ValueError("block-scaled M/N tiles must be multiples of 32 and K must be 32 or 64")
    if fragment_type not in {"float", "bfloat"}:
        raise ValueError("fragment_type must be float or bfloat")
    if outer_k and (not register_fragments or outer_k % 16 or k % outer_k):
        raise ValueError("outer_k requires register fragments and must evenly divide K")
    wm, wn = block_m // 32, block_n // 32
    num_simdgroups = wm * wn
    num_threads = num_simdgroups * 32

    function = mir.MFunction(
        name=function_name,
        kernel_type="tensor_ops_gemm",
        threadgroup_size=(num_threads, 1, 1),
    )
    function.params = [
        mir.MParam("activations", PtrType("f32")),
        mir.MParam("packed", PtrType("u8")),
        mir.MParam("scales", PtrType("u8")),
        mir.MParam("output", PtrType("f32"), is_output=True),
    ]
    activations = mir.MValue("activations", PtrType("f32"))
    packed = mir.MValue("packed", PtrType("u8"))
    scales = mir.MValue("scales", PtrType("u8"))
    output = mir.MValue("output", PtrType("f32"))

    function.add_op(mir.ThreadgroupPositionInGrid())
    if register_fragments:
        function.add_op(mir.MSimdgroupId())
        function.add_op(mir.MThreadInSimdgroup())
        function.add_op(
            mir.MTileSchedule(
                pattern=schedule,
                block_m=block_m,
                block_n=block_n,
                block_size=4,
                grid_m=m // block_m,
                grid_n=n // block_n,
            )
        )
        function.add_op(
            mir.MNaxGemmSetup(
                block_m=block_m,
                block_n=block_n,
                wm=wm,
                wn=wn,
                m=m,
                n=n,
                k=k,
                right_type=fragment_type,
            )
        )
        step = mir.MNaxBlockScaledRun(
            ptr_a=activations,
            ptr_values=packed,
            ptr_scales=scales,
            bits=bits,
            fragment_type=fragment_type,
        )
        if outer_k:
            inner_loop = mir.MForLoop(
                iv_name="k1",
                start=0,
                end=outer_k,
                step=16,
                body=[step],
                index_alias="k",
                index_expression="k0 + k1",
            )
            function.add_op(
                mir.MForLoop(
                    iv_name="k0",
                    start=0,
                    end=k,
                    step=outer_k,
                    body=[mir.MBarrier(kind="threadgroup", flags="mem_none"), inner_loop],
                )
            )
        else:
            function.add_op(mir.MForLoop(iv_name="k", start=0, end=k, step=16, body=[step]))
        function.add_op(mir.MNaxGemmStore(ptr_c=output))
        return function

    function.add_op(mir.ThreadPositionInThreadgroup())
    function.add_op(mir.MSimdgroupId())
    function.add_op(
        mir.MBlockScaledTensorViewDecl(
            ptr_a=activations,
            ptr_c=output,
            m=m,
            n=n,
            k=k,
            block_k=block_k,
            block_n=block_n,
            stage_type="float",
        )
    )
    function.add_op(
        mir.MTileSchedule(
            pattern=schedule,
            block_m=block_m,
            block_n=block_n,
            block_size=4,
            grid_m=m // block_m,
            grid_n=n // block_n,
        )
    )
    function.add_op(
        mir.MMatmul2dSetup(
            sm=32,
            sn=32,
            bk=32,
            block_m=block_m,
            block_n=block_n,
            wm=wm,
            wn=wn,
            relaxed=True,
            cooperative=False,
            num_sg=num_simdgroups,
            in_type="float",
            acc_type="float",
            out_type="float",
            use_separated=False,
        )
    )
    function.add_op(
        mir.MCoopTensorInit(
            ct_name="cT",
            acc_type="float",
            in_type="float",
            left_type="float",
            right_type="float",
            use_separated=False,
            left_address_space="device",
            right_address_space="threadgroup",
        )
    )
    k_body = [
        mir.MBlockScaledTileLoad(
            ptr_values=packed,
            ptr_scales=scales,
            bits=bits,
            matrix_n=n,
            block_k=block_k,
            block_n=block_n,
            num_threads=num_threads,
            stage_type="float",
        ),
        mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"),
    ]
    for k_offset in range(0, block_k, 32):
        k_body.append(
            mir.MMatmul2dRun(
                ct_a="ct_a",
                ct_b="ct_b",
                ct_out="cT",
                use_tensor_view=True,
                a_tensor="tA",
                b_tensor="tB",
                a_slice_d0=32,
                a_slice_d1=32,
                b_slice_d0=32,
                b_slice_d1=32,
                a_offset_0=f"k + {k_offset}u" if k_offset else "k",
                a_offset_1="tile_row",
                b_offset_0="sg_col * 32u",
                b_offset_1=f"{k_offset}u",
            )
        )
    k_body.append(mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"))
    function.add_op(
        mir.MForLoop(
            iv_name="k",
            start=0,
            end=k,
            step=block_k,
            body=k_body,
        )
    )
    function.add_op(mir.MCoopTensorStore(ct_name="cT", output_slice="mC"))
    return function
