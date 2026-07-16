import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice


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
