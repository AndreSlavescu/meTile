import math

import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice


@metile.kernel
def fft_kernel(
    X_re, X_im, Y_re, Y_im, BIT_REV, TW_re, TW_im, N,
    NUM_STAGES: metile.constexpr, BLOCK: metile.constexpr,
):
    row = metile.program_id(0)
    tid = metile.thread_id()
    row_off = row * N

    s_re = metile.shared(BLOCK)
    s_im = metile.shared(BLOCK)
    tw_src_re = metile.shared(BLOCK)
    tw_src_im = metile.shared(BLOCK)

    metile.store(tw_src_re + tid, metile.load(TW_re + tid))
    metile.store(tw_src_im + tid, metile.load(TW_im + tid))

    rev = metile.load(BIT_REV + tid)
    metile.store(s_re + rev, metile.load(X_re + row_off + tid))
    metile.store(s_im + rev, metile.load(X_im + row_off + tid))
    metile.barrier()

    my_re = metile.load(s_re + tid)
    my_im = metile.load(s_im + tid)

    SIMD_STAGES = min(5, NUM_STAGES)  # log2(32) = 5
    for stage in range(SIMD_STAGES):
        half = 1 << stage
        is_even = (tid & half) == 0

        p_re = metile.simd_shuffle_xor(my_re, half)
        p_im = metile.simd_shuffle_xor(my_im, half)

        if stage == 0:
            my_re = metile.where(is_even, my_re + p_re, p_re - my_re)
            my_im = metile.where(is_even, my_im + p_im, p_im - my_im)
        else:
            half_mask = half - 1
            tw_pos = tid & half_mask
            tw_r = metile.load(tw_src_re + half_mask + tw_pos)
            tw_i = metile.load(tw_src_im + half_mask + tw_pos)

            t_re = p_re * tw_r - p_im * tw_i
            t_im = p_re * tw_i + p_im * tw_r

            pt_re = metile.simd_shuffle_xor(t_re, half)
            pt_im = metile.simd_shuffle_xor(t_im, half)

            my_re = metile.where(is_even, my_re + t_re, p_re - pt_re)
            my_im = metile.where(is_even, my_im + t_im, p_im - pt_im)

    REMAINING = NUM_STAGES - SIMD_STAGES
    if REMAINING > 0:
        metile.barrier()
        metile.store(s_re + tid, my_re)
        metile.store(s_im + tid, my_im)
        metile.barrier()

    if REMAINING > 0:
        for stage_offset in metile.tile_range(0, REMAINING):
            stage = stage_offset + SIMD_STAGES
            half = 1 << stage
            half_bit = tid & half
            half_mask = half - 1

            even_idx = tid - half_bit
            odd_idx = even_idx + half

            tw_pos = tid & half_mask
            tw_r = metile.load(tw_src_re + half_mask + tw_pos)
            tw_i = metile.load(tw_src_im + half_mask + tw_pos)

            e_re = metile.load(s_re + even_idx)
            e_im = metile.load(s_im + even_idx)
            o_re = metile.load(s_re + odd_idx)
            o_im = metile.load(s_im + odd_idx)

            t_re = o_re * tw_r - o_im * tw_i
            t_im = o_re * tw_i + o_im * tw_r

            out_re = metile.where(half_bit == 0, e_re + t_re, e_re - t_re)
            out_im = metile.where(half_bit == 0, e_im + t_im, e_im - t_im)

            metile.barrier()
            metile.store(s_re + tid, out_re)
            metile.store(s_im + tid, out_im)
            metile.barrier()

    if NUM_STAGES <= SIMD_STAGES:
        metile.store(s_re + tid, my_re)
        metile.store(s_im + tid, my_im)

    metile.store(Y_re + row_off + tid, metile.load(s_re + tid))
    metile.store(Y_im + row_off + tid, metile.load(s_im + tid))


def _make_fft_kernel_large(elems_per_thread):
    import math as _math

    LOG_ELEMS = int(_math.log2(elems_per_thread))

    @metile.kernel
    def fft_kernel_large(
        X_re, X_im, Y_re, Y_im, BIT_REV, TW_re, TW_im, N,
        NUM_STAGES: metile.constexpr, BLOCK: metile.constexpr,
    ):
        row = metile.program_id(0)
        tid = metile.thread_id()
        row_off = row * N

        n_total = BLOCK * elems_per_thread
        s_re = metile.shared(n_total)
        s_im = metile.shared(n_total)
        USE_TW_SHARED = elems_per_thread <= 2
        if USE_TW_SHARED:
            tw_src_re = metile.shared(n_total)
            tw_src_im = metile.shared(n_total)
            for e in range(elems_per_thread):
                idx = tid + e * BLOCK
                metile.store(tw_src_re + idx, metile.load(TW_re + idx))
                metile.store(tw_src_im + idx, metile.load(TW_im + idx))
        else:
            tw_src_re = TW_re
            tw_src_im = TW_im

        for e in range(elems_per_thread):
            idx = tid + e * BLOCK
            rev = metile.load(BIT_REV + idx)
            metile.store(s_re + rev, metile.load(X_re + row_off + idx))
            metile.store(s_im + rev, metile.load(X_im + row_off + idx))
        metile.barrier()

        elem_re = []
        elem_im = []
        for e in range(elems_per_thread):
            idx = tid + e * BLOCK
            elem_re.append(metile.load(s_re + idx))
            elem_im.append(metile.load(s_im + idx))

        SIMD_STAGES = min(5, NUM_STAGES)
        for stage in range(SIMD_STAGES):
            half = 1 << stage
            for e in range(elems_per_thread):
                idx = tid + e * BLOCK
                is_even = (tid & half) == 0

                p_re = metile.simd_shuffle_xor(elem_re[e], half)
                p_im = metile.simd_shuffle_xor(elem_im[e], half)

                if stage == 0:
                    elem_re[e] = metile.where(is_even, elem_re[e] + p_re, p_re - elem_re[e])
                    elem_im[e] = metile.where(is_even, elem_im[e] + p_im, p_im - elem_im[e])
                else:
                    half_mask = half - 1
                    tw_pos = tid & half_mask
                    tw_r = metile.load(tw_src_re + half_mask + tw_pos)
                    tw_i = metile.load(tw_src_im + half_mask + tw_pos)

                    t_re = p_re * tw_r - p_im * tw_i
                    t_im = p_re * tw_i + p_im * tw_r

                    pt_re = metile.simd_shuffle_xor(t_re, half)
                    pt_im = metile.simd_shuffle_xor(t_im, half)

                    elem_re[e] = metile.where(is_even, elem_re[e] + t_re, p_re - pt_re)
                    elem_im[e] = metile.where(is_even, elem_im[e] + t_im, p_im - pt_im)

        TG_STAGES = NUM_STAGES - SIMD_STAGES - LOG_ELEMS
        if TG_STAGES > 0:
            metile.barrier()
            for e in range(elems_per_thread):
                idx = tid + e * BLOCK
                metile.store(s_re + idx, elem_re[e])
                metile.store(s_im + idx, elem_im[e])
            metile.barrier()

        if TG_STAGES > 0:
            for stage_offset in metile.tile_range(0, TG_STAGES):
                stage = stage_offset + SIMD_STAGES
                half = 1 << stage
                half_mask = half - 1

                results_re = []
                results_im = []
                for e in range(elems_per_thread):
                    idx = tid + e * BLOCK
                    half_bit = idx & half
                    even_idx = idx - half_bit
                    odd_idx = even_idx + half

                    tw_pos = idx & half_mask
                    tw_r = metile.load(tw_src_re + half_mask + tw_pos)
                    tw_i = metile.load(tw_src_im + half_mask + tw_pos)

                    e_re = metile.load(s_re + even_idx)
                    e_im = metile.load(s_im + even_idx)
                    o_re = metile.load(s_re + odd_idx)
                    o_im = metile.load(s_im + odd_idx)

                    t_re = o_re * tw_r - o_im * tw_i
                    t_im = o_re * tw_i + o_im * tw_r

                    out_re = metile.where(half_bit == 0, e_re + t_re, e_re - t_re)
                    out_im = metile.where(half_bit == 0, e_im + t_im, e_im - t_im)

                    results_re.append(out_re)
                    results_im.append(out_im)

                metile.barrier()
                for e in range(elems_per_thread):
                    idx = tid + e * BLOCK
                    metile.store(s_re + idx, results_re[e])
                    metile.store(s_im + idx, results_im[e])
                metile.barrier()

            for e in range(elems_per_thread):
                idx = tid + e * BLOCK
                elem_re[e] = metile.load(s_re + idx)
                elem_im[e] = metile.load(s_im + idx)

        for local_stage in range(LOG_ELEMS):
            local_half_elems = 1 << local_stage
            local_stride = 2 * local_half_elems
            stage = NUM_STAGES - LOG_ELEMS + local_stage
            half = 1 << stage
            half_mask = half - 1

            for g in range(elems_per_thread // local_stride):
                e_even = g * local_stride
                e_odd = e_even + local_half_elems

                idx_even = tid + e_even * BLOCK
                tw_pos = idx_even & half_mask
                tw_r = metile.load(tw_src_re + half_mask + tw_pos)
                tw_i = metile.load(tw_src_im + half_mask + tw_pos)

                a_re, a_im = elem_re[e_even], elem_im[e_even]
                b_re, b_im = elem_re[e_odd], elem_im[e_odd]

                t_re = b_re * tw_r - b_im * tw_i
                t_im = b_re * tw_i + b_im * tw_r

                elem_re[e_even] = a_re + t_re
                elem_im[e_even] = a_im + t_im
                elem_re[e_odd] = a_re - t_re
                elem_im[e_odd] = a_im - t_im

        for e in range(elems_per_thread):
            idx = tid + e * BLOCK
            metile.store(Y_re + row_off + idx, elem_re[e])
            metile.store(Y_im + row_off + idx, elem_im[e])

    return fft_kernel_large


_FFT_KERNELS = {
    1: fft_kernel,
    2: _make_fft_kernel_large(2),
}


def _bit_reverse_permutation(N):
    bits = int(math.log2(N))
    perm = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev = 0
        for b in range(bits):
            rev = (rev << 1) | ((i >> b) & 1)
        perm[i] = rev
    return perm


def _twiddle_factors(num_stages):
    tw_re, tw_im = [], []
    for s in range(num_stages):
        half = 1 << s
        angles = -2.0 * np.pi * np.arange(half, dtype=np.float64) / (2 * half)
        tw_re.append(np.cos(angles).astype(np.float32))
        tw_im.append(np.sin(angles).astype(np.float32))
    return np.concatenate(tw_re), np.concatenate(tw_im)


def fft_dispatch(batch, N, x_re_buf, x_im_buf, y_re_buf, y_im_buf):
    assert N & (N - 1) == 0, "N must be a power of 2"
    assert N <= 2048, "N must be <= 2048 (shared memory limit)"
    num_stages = int(math.log2(N))
    elems = max(1, N // 1024)
    block = min(N, 1024)

    perm = _bit_reverse_permutation(N)
    tw_re, tw_im = _twiddle_factors(num_stages)

    kern = _FFT_KERNELS[elems]
    grid = (batch,)
    return kern[grid].prepare(
        x_re_buf, x_im_buf, y_re_buf, y_im_buf,
        metile.Buffer(data=perm),
        metile.Buffer(data=tw_re), metile.Buffer(data=tw_im),
        N, NUM_STAGES=num_stages, BLOCK=block,
    )


def fft(x_re_np, x_im_np, batch, N):
    x_re_buf = metile.Buffer(data=x_re_np)
    x_im_buf = metile.Buffer(data=x_im_np)
    y_re_buf = metile.Buffer.zeros((batch * N,))
    y_im_buf = metile.Buffer.zeros((batch * N,))

    dispatch = fft_dispatch(batch, N, x_re_buf, x_im_buf, y_re_buf, y_im_buf)
    dispatch()
    MetalDevice.get().sync()
    return y_re_buf.numpy(), y_im_buf.numpy()
