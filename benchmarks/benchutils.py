import gc
import time

import numpy as np

from metile.runtime.metal_device import MetalDevice

_clock = time.perf_counter_ns  # mach_absolute_time on macOS, integer ns

# Each timed sample averages over enough synced dispatches so that the
# total sample wall time is >= this many ms, reducing noise from single
# dispatch sync overhead jitter.
_SAMPLE_TARGET_MS = 25


def _stabilize(fn, sync, batch, max_ms=100):
    """Warmup until timings stabilize (coefficient of variation < 5%)."""
    window = []
    deadline = _clock() + max_ms * 1_000_000
    while _clock() < deadline:
        t0 = _clock()
        for _ in range(batch):
            fn()
            sync()
        window.append((_clock() - t0) / batch)
        if len(window) >= 10:
            recent = window[-8:]
            cv = np.std(recent) / np.mean(recent)
            if cv < 0.05:
                return
            window = window[-15:]


def _trimmed_median(times):
    """Median of middle 80% (trim top/bottom 10% outliers)."""
    times = sorted(times)
    n = len(times)
    lo, hi = n // 10, n - n // 10
    trimmed = times[lo:hi] if hi > lo else times
    return np.median(trimmed)


def bench(fn, warmup_ms=25, rep_ms=100, quantiles=None, gpu=True):
    """Benchmark a function with adaptive iteration count.

    Args:
        fn: callable to benchmark
        warmup_ms: time budget for warmup in ms
        rep_ms: time budget for timed repetitions in ms
        quantiles: tuple of quantiles to return (default: returns median only)
        gpu: if True, use Metal hardware GPU timestamps (raw kernel time).
             if False, use wall clock (for MLX etc).

    Returns:
        If quantiles is None: median time in seconds.
        If quantiles is set: tuple of times at those quantiles.
    """
    dev = MetalDevice.get()
    sync = dev.sync

    # Estimate per-call time from 5 runs (ns)
    sync()
    t0 = _clock()
    for _ in range(5):
        fn()
        sync()
    estimate_ns = (_clock() - t0) / 5

    # Batch size: group enough synced dispatches per sample so each sample
    # measures >= target ms. GPU mode always batch=1 (hardware timestamps).
    if gpu:
        batch = 1
    else:
        batch = max(1, int(_SAMPLE_TARGET_MS * 1_000_000 / max(estimate_ns, 1)))

    # Adaptive sample count
    sample_ns = estimate_ns * batch
    n_warmup = max(1, int(warmup_ms * 1_000_000 / sample_ns))
    n_rep = max(1, int(rep_ms * 1_000_000 / sample_ns))

    # Warmup: fixed budget then stabilize
    for _ in range(n_warmup):
        for _ in range(batch):
            fn()
            sync()
    _stabilize(fn, sync, batch)

    # Disable GC during measurement
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        times = []
        for _ in range(n_rep):
            if gpu:
                fn()
                sync()
                times.append(dev.gpu_elapsed())
            else:
                # Each sample averages `batch` synced dispatches.
                # Sync per dispatch keeps it fair (no pipeline advantage).
                t0 = _clock()
                for _ in range(batch):
                    fn()
                    sync()
                times.append((_clock() - t0) * 1e-9 / batch)
    finally:
        if gc_was_enabled:
            gc.enable()

    if quantiles is not None:
        return tuple(np.quantile(sorted(times), q) for q in quantiles)
    return _trimmed_median(times)


def bench_interleaved(fn_a, fn_b, sync=None, warmup_ms=50, rep_ms=200):
    """Benchmark two functions under identical thermal conditions.

    Interleaves dispatches: A, B, A, B, ... so both experience the same
    GPU temperature profile. Eliminates order bias from DVFS.

    Returns:
        (median_time_a, median_time_b) in seconds.
    """
    if sync is None:
        sync = MetalDevice.get().sync

    # Estimate per-pair time
    sync()
    t0 = _clock()
    for _ in range(3):
        fn_a()
        sync()
        fn_b()
        sync()
    pair_ns = (_clock() - t0) / 3

    n_warmup = max(5, int(warmup_ms * 1_000_000 / pair_ns))
    n_rep = max(10, int(rep_ms * 1_000_000 / pair_ns))

    # Warmup
    for _ in range(n_warmup):
        fn_a()
        sync()
        fn_b()
        sync()

    gc_was = gc.isenabled()
    gc.disable()

    try:
        times_a, times_b = [], []
        for _ in range(n_rep):
            t0 = _clock()
            fn_a()
            sync()
            ta = (_clock() - t0) * 1e-9

            t0 = _clock()
            fn_b()
            sync()
            tb = (_clock() - t0) * 1e-9

            times_a.append(ta)
            times_b.append(tb)
    finally:
        if gc_was:
            gc.enable()

    return _trimmed_median(times_a), _trimmed_median(times_b)
