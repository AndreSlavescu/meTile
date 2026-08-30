import math

import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice


@metile.kernel
def fft_kernel(
    X_re,
    X_im,
    Y_re,
    Y_im,
    TW_re,
    TW_im,
    N,
    BATCH: metile.constexpr,
    NUM_STAGES: metile.constexpr,
    BLOCK: metile.constexpr,
    ELEMS_PER_THREAD: metile.constexpr,
    BIT_REVERSE_GATHER: metile.constexpr,
    TWIDDLE_SHARED: metile.constexpr,
):
    if BATCH <= 0:
        raise ValueError("BATCH must be positive")

    row = metile.program_id(0)
    tid = metile.thread_id()
    row_off = row * N

    log_elements = int(math.log2(ELEMS_PER_THREAD))
    n_total = BLOCK * ELEMS_PER_THREAD
    simd_stages = min(5, NUM_STAGES)
    threadgroup_stages = NUM_STAGES - simd_stages - log_elements
    if not BIT_REVERSE_GATHER or threadgroup_stages > 0:
        s_re = metile.shared(n_total)
        s_im = metile.shared(n_total)
    if TWIDDLE_SHARED:
        tw_source_re = metile.shared(n_total)
        tw_source_im = metile.shared(n_total)
        for element in range(ELEMS_PER_THREAD):
            index = tid + element * BLOCK
            metile.store(tw_source_re + index, metile.load(TW_re + index))
            metile.store(tw_source_im + index, metile.load(TW_im + index))
        if BIT_REVERSE_GATHER:
            metile.barrier()
    else:
        tw_source_re = TW_re
        tw_source_im = TW_im

    element_re = []
    element_im = []
    if BIT_REVERSE_GATHER:
        for element in range(ELEMS_PER_THREAD):
            index = tid + element * BLOCK
            reverse_index = metile.reverse_bits(index) >> (32 - NUM_STAGES)
            element_re.append(metile.load(X_re + row_off + reverse_index))
            element_im.append(metile.load(X_im + row_off + reverse_index))
    else:
        for element in range(ELEMS_PER_THREAD):
            index = tid + element * BLOCK
            reverse_index = metile.reverse_bits(index) >> (32 - NUM_STAGES)
            metile.store(s_re + reverse_index, metile.load(X_re + row_off + index))
            metile.store(s_im + reverse_index, metile.load(X_im + row_off + index))
        metile.barrier()

        for element in range(ELEMS_PER_THREAD):
            index = tid + element * BLOCK
            element_re.append(metile.load(s_re + index))
            element_im.append(metile.load(s_im + index))

    for stage in range(simd_stages):
        half = 1 << stage
        is_even = (tid & half) == 0
        if stage > 0:
            half_mask = half - 1
            twiddle_position = tid & half_mask
            twiddle_re = metile.load(tw_source_re + half_mask + twiddle_position)
            twiddle_im = metile.load(tw_source_im + half_mask + twiddle_position)
        for element in range(ELEMS_PER_THREAD):
            partner_re = metile.simd_shuffle_xor(element_re[element], half)
            partner_im = metile.simd_shuffle_xor(element_im[element], half)

            if stage == 0:
                element_re[element] = metile.where(
                    is_even,
                    element_re[element] + partner_re,
                    partner_re - element_re[element],
                )
                element_im[element] = metile.where(
                    is_even,
                    element_im[element] + partner_im,
                    partner_im - element_im[element],
                )
            else:
                butterfly_re = metile.where(is_even, partner_re, element_re[element])
                butterfly_im = metile.where(is_even, partner_im, element_im[element])
                transformed_re = butterfly_re * twiddle_re - butterfly_im * twiddle_im
                transformed_im = butterfly_re * twiddle_im + butterfly_im * twiddle_re

                element_re[element] = metile.where(
                    is_even,
                    element_re[element] + transformed_re,
                    partner_re - transformed_re,
                )
                element_im[element] = metile.where(
                    is_even,
                    element_im[element] + transformed_im,
                    partner_im - transformed_im,
                )

    if threadgroup_stages > 0:
        for element in range(ELEMS_PER_THREAD):
            index = tid + element * BLOCK
            metile.store(s_re + index, element_re[element])
            metile.store(s_im + index, element_im[element])
        metile.barrier()

        for stage_offset in range(threadgroup_stages):
            stage = stage_offset + simd_stages
            half = 1 << stage
            half_mask = half - 1
            twiddle_position = tid & half_mask
            twiddle_re = metile.load(tw_source_re + half_mask + twiddle_position)
            twiddle_im = metile.load(tw_source_im + half_mask + twiddle_position)

            results_re = []
            results_im = []
            for element in range(ELEMS_PER_THREAD):
                index = tid + element * BLOCK
                half_bit = index & half
                even_index = index - half_bit
                odd_index = even_index + half

                even_re = metile.load(s_re + even_index)
                even_im = metile.load(s_im + even_index)
                odd_re = metile.load(s_re + odd_index)
                odd_im = metile.load(s_im + odd_index)

                transformed_re = odd_re * twiddle_re - odd_im * twiddle_im
                transformed_im = odd_re * twiddle_im + odd_im * twiddle_re

                results_re.append(
                    metile.where(half_bit == 0, even_re + transformed_re, even_re - transformed_re)
                )
                results_im.append(
                    metile.where(half_bit == 0, even_im + transformed_im, even_im - transformed_im)
                )

            if stage_offset + 1 < threadgroup_stages:
                metile.barrier()
                for element in range(ELEMS_PER_THREAD):
                    index = tid + element * BLOCK
                    metile.store(s_re + index, results_re[element])
                    metile.store(s_im + index, results_im[element])
                metile.barrier()
            else:
                element_re = results_re
                element_im = results_im

    for local_stage in range(log_elements):
        local_half_elements = 1 << local_stage
        local_stride = 2 * local_half_elements
        stage = NUM_STAGES - log_elements + local_stage
        half = 1 << stage
        half_mask = half - 1

        for offset in range(local_half_elements):
            representative_index = tid + offset * BLOCK
            twiddle_position = representative_index & half_mask
            twiddle_re = metile.load(tw_source_re + half_mask + twiddle_position)
            twiddle_im = metile.load(tw_source_im + half_mask + twiddle_position)

            for group in range(ELEMS_PER_THREAD // local_stride):
                even_element = group * local_stride + offset
                odd_element = even_element + local_half_elements

                even_re, even_im = element_re[even_element], element_im[even_element]
                odd_re, odd_im = element_re[odd_element], element_im[odd_element]

                transformed_re = odd_re * twiddle_re - odd_im * twiddle_im
                transformed_im = odd_re * twiddle_im + odd_im * twiddle_re

                element_re[even_element] = even_re + transformed_re
                element_im[even_element] = even_im + transformed_im
                element_re[odd_element] = even_re - transformed_re
                element_im[odd_element] = even_im - transformed_im

    for element in range(ELEMS_PER_THREAD):
        index = tid + element * BLOCK
        metile.store(Y_re + row_off + index, element_re[element])
        metile.store(Y_im + row_off + index, element_im[element])


_FFT_TUNERS = {}


def _fft_configs(n):
    configs = []
    for elements_per_thread in (1, 2, 4, 8, 16, 32):
        if n % elements_per_thread:
            continue
        block = n // elements_per_thread
        if block > 1024 or (n >= 32 and block < 32):
            continue
        if n < 32 and elements_per_thread != 1:
            continue
        twiddle_placements = (False,) if n >= 2048 else (False, True)
        for bit_reverse_gather in (False, True):
            for twiddle_shared in twiddle_placements:
                configs.append(
                    metile.Config(
                        BLOCK=block,
                        ELEMS_PER_THREAD=elements_per_thread,
                        BIT_REVERSE_GATHER=bit_reverse_gather,
                        TWIDDLE_SHARED=twiddle_shared,
                    )
                )
    return configs


def _fft_tuner(n):
    tuner = _FFT_TUNERS.get(n)
    if tuner is None:
        tuner = metile.autotune(
            configs=_fft_configs(n),
            key=["BATCH", "N"],
            verbose=False,
        )(fft_kernel)
        _FFT_TUNERS[n] = tuner
    return tuner


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

    tw_re, tw_im = _twiddle_factors(num_stages)

    tuner = _fft_tuner(N)
    grid = (batch,)
    return tuner[grid].prepare(
        x_re_buf,
        x_im_buf,
        y_re_buf,
        y_im_buf,
        metile.Buffer(data=tw_re),
        metile.Buffer(data=tw_im),
        N,
        BATCH=batch,
        NUM_STAGES=num_stages,
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
