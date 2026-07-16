import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice


@metile.kernel
def _add_one(source, destination, size, BLOCK: metile.constexpr):
    offsets = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
    mask = offsets < size
    values = metile.load(source + offsets, mask=mask)
    metile.store(destination + offsets, values + 1.0, mask=mask)


def test_prepared_dependency_chain_is_ordered_on_concurrent_encoder():
    size = 4096
    source = metile.Buffer(data=np.arange(size, dtype=np.float32))
    intermediate = metile.Buffer.empty((size,))
    destination = metile.Buffer.empty((size,))
    grid = (metile.cdiv(size, 256),)
    first = _add_one[grid].prepare(source, intermediate, size, BLOCK=256)
    second = _add_one[grid].prepare(intermediate, destination, size, BLOCK=256)

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
