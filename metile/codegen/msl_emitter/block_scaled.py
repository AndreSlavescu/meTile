"""MSL emission for block-scaled tile loads."""

from __future__ import annotations

from metile.codegen.msl_emitter.common import (
    _val_name_gemm,
)


def _block_scaled_helpers():
    return [
        "inline float mtile_decode_e2m1(uchar bits) {",
        "    const ushort raw = (ushort(bits & 7u) << 9u) | (ushort(bits & 8u) << 12u);",
        "    return float(as_type<half>(raw)) * 16384.0f;",
        "}",
        "",
        "inline float4 mtile_decode_e2m1(uchar4 bits) {",
        "    const ushort4 raw = (ushort4(bits & 7u) << 9u) | (ushort4(bits & 8u) << 12u);",
        "    return float4(as_type<half4>(raw)) * 16384.0f;",
        "}",
        "",
        "inline float mtile_decode_e4m3(uchar bits) {",
        "    const ushort raw = (ushort(bits & 127u) << 7u) | (ushort(bits & 128u) << 8u);",
        "    return float(as_type<half>(raw)) * 256.0f;",
        "}",
        "",
        "inline float4 mtile_decode_e4m3(uchar4 bits) {",
        "    const ushort4 raw = (ushort4(bits & 127u) << 7u) | (ushort4(bits & 128u) << 8u);",
        "    return float4(as_type<half4>(raw)) * 256.0f;",
        "}",
        "",
        "inline float mtile_decode_e8m0(uchar bits) {",
        "    const uint raw = bits == 0u ? 0x00400000u : uint(bits) << 23u;",
        "    return as_type<float>(raw);",
        "}",
        "",
        "inline float4 mtile_decode_e8m0(uchar4 bits) {",
        "    const uint4 raw = select(uint4(bits) << 23u, uint4(0x00400000u), bits == 0u);",
        "    return as_type<float4>(raw);",
        "}",
        "",
    ]


def _emit_block_scaled_tensor_views(op, lines, indent, func):
    pad = "    " * indent
    a_name = _val_name_gemm(op.ptr_a, func)
    c_name = _val_name_gemm(op.ptr_c, func)
    lines.append(f"{pad}constexpr uint M = {op.m}u;")
    lines.append(f"{pad}constexpr uint N = {op.n}u;")
    lines.append(f"{pad}constexpr uint K = {op.k}u;")
    lines.append(f"{pad}auto tA = tensor<device float, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {a_name}, dextents<int32_t, 2>(K, M));")
    lines.append(f"{pad}auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(")
    lines.append(f"{pad}    {c_name}, dextents<int32_t, 2>(N, M));")
    lines.append(f"{pad}threadgroup {op.stage_type} b_tile[{op.block_k * op.block_n}];")
    lines.append(
        f"{pad}auto tB = tensor<threadgroup {op.stage_type}, dextents<int32_t, 2>, tensor_inline>("
    )
    lines.append(f"{pad}    b_tile, dextents<int32_t, 2>({op.block_n}, {op.block_k}));")
    lines.append("")


def _emit_block_scaled_tile_load(op, lines, indent, func):
    pad = "    " * indent
    values = _val_name_gemm(op.ptr_values, func)
    scales = _val_name_gemm(op.ptr_scales, func)
    total = op.block_k * op.block_n
    scale_groups = op.block_k // 32
    lines.append(f"{pad}const uint scale_n = lid % {op.block_n}u;")
    lines.append(f"{pad}float block_scales[{scale_groups}];")
    lines.append(f"{pad}#pragma clang loop unroll(full)")
    lines.append(f"{pad}for (uint group = 0u; group < {scale_groups}u; ++group) {{")
    lines.append(
        f"{pad}    block_scales[group] = mtile_decode_e8m0({scales}[(uint(k) / 32u + group) * {op.matrix_n}u + pid_n * {op.block_n}u + scale_n]);"
    )
    lines.append(f"{pad}}}")
    lines.append(f"{pad}for (uint index = lid; index < {total}u; index += {op.num_threads}u) {{")
    lines.append(f"{pad}    const uint local_k = index / {op.block_n}u;")
    lines.append(f"{pad}    const uint local_n = index % {op.block_n}u;")
    lines.append(f"{pad}    const uint global_k = uint(k) + local_k;")
    lines.append(f"{pad}    const uint global_n = pid_n * {op.block_n}u + local_n;")
    lines.append(f"{pad}    const uint element = global_k * {op.matrix_n}u + global_n;")
    if op.bits == 4:
        lines.append(f"{pad}    const uchar byte = {values}[element >> 1u];")
        lines.append(
            f"{pad}    const uchar quantized = (element & 1u) ? (byte >> 4u) : (byte & 15u);"
        )
        decode = "mtile_decode_e2m1(quantized)"
    else:
        decode = f"mtile_decode_e4m3({values}[element])"
    lines.append(
        f"{pad}    b_tile[index] = {op.stage_type}(block_scales[local_k >> 5u] * {decode});"
    )
    lines.append(f"{pad}}}")
