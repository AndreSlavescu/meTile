import threading

import numpy as np
import pytest

import metile
from metile.runtime.metal_device import MetalDevice, completion_spin_budget_ns


@metile.kernel
def _add_one(source, destination, size, BLOCK: metile.constexpr):
    offsets = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    mask = offsets < size
    values = metile.load(source + offsets, mask=mask)
    metile.store(destination + offsets, values + 1.0, mask=mask)


@metile.kernel
def _fill_sequence(destination, BLOCK: metile.constexpr):
    offsets = metile.arange(0, BLOCK)
    metile.store(destination + offsets, offsets + 1)


def test_prepared_dependency_chain_is_ordered_on_concurrent_encoder():
    size = 4096
    source = metile.Buffer(data=np.arange(size, dtype=np.float32))
    intermediate = metile.Buffer.empty((size,))
    destination = metile.Buffer.empty((size,))
    grid = (metile.cdiv(size, 256),)
    first = _add_one[grid].prepare(source, intermediate, size, BLOCK=256)
    second = _add_one[grid].prepare(intermediate, destination, size, BLOCK=256)
    assert first._buffer_range.length == len(first._buffers)

    first()
    second()

    np.testing.assert_array_equal(destination.numpy(), np.arange(size, dtype=np.float32) + 2.0)


def test_unretained_command_buffer_keeps_dispatch_resources_alive_until_sync():
    size = 256
    source = metile.Buffer(data=np.arange(size, dtype=np.float32))
    destination = metile.Buffer.empty((size,))
    dispatch = _add_one[(1,)].prepare(source, destination, size, BLOCK=256)
    device = MetalDevice.get()

    dispatch()
    device.flush()
    assert dispatch in device._inflight_lifetimes

    device.sync()
    assert not device._inflight_lifetimes
    np.testing.assert_array_equal(destination.numpy(), np.arange(size, dtype=np.float32) + 1.0)


def test_prepared_dispatch_retains_bound_buffers():
    size = 256
    source = metile.Buffer(data=np.arange(size, dtype=np.float32))
    destination = metile.Buffer.empty((size,))
    dispatch = _add_one[(1,)].prepare(source, destination, size, BLOCK=256)

    assert source in dispatch._resources
    assert destination in dispatch._resources


def test_single_buffer_prepared_dispatch_uses_direct_binding():
    size = 256
    destination = metile.Buffer.empty((size,))
    dispatch = _fill_sequence[(1,)].prepare(destination, BLOCK=size)

    assert dispatch._buffer_array is None
    dispatch()
    np.testing.assert_array_equal(destination.numpy(), np.arange(size, dtype=np.float32) + 1.0)


def test_repeated_prepared_dispatch_reuses_encoder_bindings():
    size = 256
    source = metile.Buffer(data=np.arange(size, dtype=np.float32))
    destination = metile.Buffer.empty((size,))
    dispatch = _add_one[(1,)].prepare(source, destination, size, BLOCK=size)
    device = MetalDevice.get()
    calls = {"buffers": 0, "pipeline": 0}
    set_buffers = dispatch._set_bufs_fn
    set_pipeline = dispatch._set_pipe_fn

    def counted_set_buffers(*args):
        calls["buffers"] += 1
        set_buffers(*args)

    def counted_set_pipeline(*args):
        calls["pipeline"] += 1
        set_pipeline(*args)

    dispatch._set_bufs_fn = counted_set_buffers
    dispatch._set_pipe_fn = counted_set_pipeline
    dispatch()
    dispatch()
    assert calls == {"buffers": 1, "pipeline": 1}

    device.sync()
    dispatch.repeat(2)
    device.sync()
    assert calls == {"buffers": 2, "pipeline": 2}


def test_low_latency_dispatch_marks_pending_command_buffer():
    size = 256
    destination = metile.Buffer.empty((size,))
    dispatch = _fill_sequence[(1,)].prepare(destination, BLOCK=size)
    device = MetalDevice.get()
    dispatch._completion_spin_ns = 900_000

    dispatch()
    assert device._pending_completion_spin_ns == 900_000
    device.sync()


def test_unclassified_dispatch_disables_batch_completion_spin():
    first_output = metile.Buffer.empty((256,))
    second_output = metile.Buffer.empty((256,))
    first = _fill_sequence[(1,)].prepare(first_output, BLOCK=256)
    second = _fill_sequence[(1,)].prepare(second_output, BLOCK=256)
    device = MetalDevice.get()
    first._completion_spin_ns = 900_000

    first()
    second()
    assert device._pending_completion_spin_ns == 0
    device.sync()


def test_large_prepared_batch_disables_completion_spin():
    destination = metile.Buffer.empty((256,))
    dispatch = _fill_sequence[(1,)].prepare(destination, BLOCK=256)
    device = MetalDevice.get()
    dispatch._completion_spin_ns = 900_000

    dispatch.repeat(9)
    device.flush()
    assert device._last_completion_spin_ns == 0
    device.sync()


@pytest.mark.parametrize(
    ("gpu_seconds", "expected_ns"),
    [
        (0.0, 0),
        (0.0001, 900_000),
        (0.0003, 1_200_000),
        (0.0005, 1_500_000),
        (0.0008, 1_500_000),
        (0.0011, 0),
    ],
)
def test_completion_spin_budget_tracks_measured_gpu_latency(gpu_seconds, expected_ns):
    assert completion_spin_budget_ns(gpu_seconds) == expected_ns


def test_autotune_scores_short_kernels_by_end_to_end_latency():
    from metile.frontend.autotune import _selection_score

    assert _selection_score([0.0001, 0.00011], [0.0005, 0.0006]) == pytest.approx(0.00055)


def test_autotune_scores_long_kernels_by_gpu_latency():
    from metile.frontend.autotune import _selection_score

    assert _selection_score([0.002, 0.0022], [0.003, 0.0032]) == pytest.approx(0.0021)


def test_autotune_persists_measured_completion_budget(tmp_path, monkeypatch):
    from metile.frontend import autotune as autotune_module

    cache = dict(autotune_module._autotune_cache)
    latency_cache = dict(autotune_module._autotune_latency_cache)
    monkeypatch.setattr(
        autotune_module,
        "_persistent_cache_path",
        tmp_path / "autotune.json",
    )
    autotune_module._autotune_cache.clear()
    autotune_module._autotune_latency_cache.clear()
    try:
        tuned = metile.autotune(
            configs=[metile.Config(BLOCK=256)],
            key=["size"],
            verbose=False,
        )(_add_one)
        source = metile.Buffer(data=np.arange(256, dtype=np.float32))
        destination = metile.Buffer.empty((256,))
        first = tuned[(1,)].prepare(source, destination, 256)
        assert first._completion_spin_ns >= 900_000

        autotune_module._autotune_cache.clear()
        autotune_module._autotune_latency_cache.clear()
        second = tuned[(1,)].prepare(source, destination, 256)
        assert second._completion_spin_ns == first._completion_spin_ns
    finally:
        autotune_module._autotune_cache.clear()
        autotune_module._autotune_cache.update(cache)
        autotune_module._autotune_latency_cache.clear()
        autotune_module._autotune_latency_cache.update(latency_cache)


@pytest.mark.parametrize(
    ("spin_ns", "status", "expected_status_calls", "expected_wait_calls"),
    [(100_000, 4, 2, 0), (0, 2, 0, 1)],
)
def test_sync_selects_bounded_poll_or_blocking_wait(
    monkeypatch, spin_ns, status, expected_status_calls, expected_wait_calls
):
    device = MetalDevice.__new__(MetalDevice)
    device._dispatch_lock = threading.RLock()
    device._last_cmd_buffer = 1
    device._last_completion_spin_ns = spin_ns
    device._inflight_lifetimes = [object()]
    device._commit_pending_unlocked = lambda: None
    device._ensure_cached_selectors = lambda: None
    device.__dict__["low_latency_spin_ns"] = spin_ns
    calls = {"status": 0, "wait": 0}

    def command_buffer_status(*_):
        calls["status"] += 1
        return status

    def wait_until_completed(*_):
        calls["wait"] += 1

    monkeypatch.setattr(MetalDevice, "_msg_send_uint64", command_buffer_status)
    monkeypatch.setattr(MetalDevice, "_msg_send_void", wait_until_completed)
    monkeypatch.setattr(MetalDevice, "_sel_status", object())
    monkeypatch.setattr(MetalDevice, "_sel_waitUntilCompleted", object())

    device.sync()

    assert calls == {"status": expected_status_calls, "wait": expected_wait_calls}
    assert device._last_cmd_buffer is None
    assert not device._inflight_lifetimes
