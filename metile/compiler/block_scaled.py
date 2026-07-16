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
) -> mir.MFunction:
    """Build composable Metal IR for an aligned MXFP weight-only GEMM."""
    if bits not in {4, 8}:
        raise ValueError("block-scaled matmul supports 4-bit or 8-bit data")

    if block_m % 32 or block_n % 32:
        raise ValueError("block-scaled M/N tiles must be multiples of 32")
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
    function.add_op(mir.ThreadPositionInThreadgroup())
    function.add_op(mir.MSimdgroupId())
    function.add_op(
        mir.MBlockScaledTensorViewDecl(
            ptr_a=activations,
            ptr_c=output,
            m=m,
            n=n,
            k=k,
            block_k=32,
            block_n=block_n,
        )
    )
    function.add_op(
        mir.MTileSchedule(
            pattern="auto",
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
            use_separated=False,
            left_address_space="device",
            right_address_space="threadgroup",
        )
    )
    function.add_op(
        mir.MForLoop(
            iv_name="k",
            start=0,
            end=k,
            step=32,
            body=[
                mir.MBlockScaledTileLoad(
                    ptr_values=packed,
                    ptr_scales=scales,
                    bits=bits,
                    matrix_n=n,
                    block_k=32,
                    block_n=block_n,
                    num_threads=num_threads,
                ),
                mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"),
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
                    a_offset_0="k",
                    a_offset_1="tile_row",
                    b_offset_0="sg_col * 32u",
                    b_offset_1="0u",
                ),
                mir.MBarrier(kind="threadgroup", flags="mem_threadgroup"),
            ],
        )
    )
    function.add_op(mir.MCoopTensorStore(ct_name="cT", output_slice="mC"))
    return function
