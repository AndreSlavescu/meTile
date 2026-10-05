import ctypes
import gc
import importlib
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import metile.runtime.buffer as buffer_module
import metile.runtime.metal_device as metal_device_module
from metile.frontend.kernel import FastDispatcher
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType
from metile.runtime.address_space import GlobalAddressSpace
from metile.runtime.buffer import MtileBuffer
from metile.runtime.metal_device import MetalDevice

kernel_module = importlib.import_module("metile.frontend.kernel")


class FakeDevice:
    def __init__(self):
        self.allocations = {}
        self.releases = []
        self.sync_count = 0
        self.next_handle = 1

    def new_empty_buffer(self, length):
        handle = self.next_handle
        self.next_handle += 1
        self.allocations[handle] = ctypes.create_string_buffer(length)
        return handle

    def new_buffer(self, data, length):
        handle = self.new_empty_buffer(length)
        ctypes.memmove(self.buffer_contents(handle), data, length)
        return handle

    def buffer_contents(self, handle):
        return ctypes.addressof(self.allocations[handle])

    def release_buffer(self, handle):
        self.releases.append(handle)
        del self.allocations[handle]

    def sync(self):
        self.sync_count += 1

    def _ensure_cached_selectors(self):
        pass


@pytest.fixture
def fake_device(monkeypatch):
    device = FakeDevice()
    monkeypatch.setattr(MetalDevice, "get", lambda: device)
    monkeypatch.setattr(buffer_module, "_buffer_cache", {})
    return device


def test_buffer_allocation_released_exactly_once(fake_device):
    buffer = MtileBuffer.empty((4,), np.float32)
    handle = buffer.metal_buffer
    assert handle in fake_device.allocations

    del buffer
    gc.collect()
    gc.collect()

    assert fake_device.releases == [handle]
    assert not fake_device.allocations


@pytest.mark.parametrize("slice_view", [False, True])
def test_escaped_numpy_view_retains_allocation(fake_device, slice_view):
    buffer = MtileBuffer(data=np.arange(8, dtype=np.float32))
    handle = buffer.metal_buffer
    view = buffer.numpy()
    if slice_view:
        view = view[1::2]
    assert fake_device.sync_count == 1

    del buffer
    gc.collect()

    assert not fake_device.releases
    np.testing.assert_array_equal(
        view, np.arange(8, dtype=np.float32)[1::2] if slice_view else np.arange(8, dtype=np.float32)
    )
    view[:] = 3
    np.testing.assert_array_equal(view, 3)

    del view
    gc.collect()
    assert fake_device.releases == [handle]


@pytest.mark.parametrize("keep_tensor", [False, True])
def test_arena_views_retain_allocation(fake_device, keep_tensor):
    arena = GlobalAddressSpace(1024)
    handle = arena.metal_buffer
    tensor = arena.tensor((8,), dtype=np.float32)
    tensor.fill(7)
    view = tensor.numpy()[::2]

    del arena
    if not keep_tensor:
        del tensor
    gc.collect()
    assert not fake_device.releases
    np.testing.assert_array_equal(view, 7)

    del view
    gc.collect()
    if keep_tensor:
        assert not fake_device.releases
        np.testing.assert_array_equal(tensor.numpy(), 7)
        del tensor
        gc.collect()
    assert fake_device.releases == [handle]


def test_empty_arena_releases_allocation(fake_device):
    arena = GlobalAddressSpace(1024)
    handle = arena.metal_buffer
    del arena
    gc.collect()
    assert fake_device.releases == [handle]


def test_implicit_cache_does_not_retain_source(fake_device):
    source = np.arange(8, dtype=np.float32)
    source_ref = weakref.ref(source)
    cache_key = id(source)
    buffer = MtileBuffer._from_numpy_implicit(source)
    handle = buffer.metal_buffer
    assert buffer_module._buffer_cache[cache_key] is buffer
    del buffer
    gc.collect()
    assert not fake_device.releases

    del source
    gc.collect()

    assert source_ref() is None
    assert cache_key not in buffer_module._buffer_cache
    assert fake_device.releases == [handle]


def test_implicit_buffer_remains_valid_after_source_dies(fake_device):
    source = np.arange(8, dtype=np.float32)
    buffer = MtileBuffer._from_numpy_implicit(source)
    handle = buffer.metal_buffer
    del source
    gc.collect()

    assert buffer._source_array is None
    assert not buffer_module._buffer_cache
    assert not fake_device.releases
    buffer.sync_to_source()
    buffer.sync_from_source()
    np.testing.assert_array_equal(buffer.numpy(), np.arange(8, dtype=np.float32))

    del buffer
    gc.collect()
    assert fake_device.releases == [handle]


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_implicit_cache_refresh_and_copy_back(fake_device, noncontiguous):
    original = np.arange(16, dtype=np.float32).reshape(4, 4)
    source = original[:, ::2] if noncontiguous else original
    buffer = MtileBuffer._from_numpy_implicit(source)
    source[:] = 9
    assert MtileBuffer._from_numpy_implicit(source) is buffer
    np.testing.assert_array_equal(buffer.numpy(), 9)

    buffer.numpy()[:] = 12
    buffer.sync_to_source()
    np.testing.assert_array_equal(source, 12)
    if noncontiguous:
        np.testing.assert_array_equal(original[:, 1::2], np.arange(16).reshape(4, 4)[:, 1::2])


def test_implicit_cache_replaces_reshaped_source(fake_device):
    source = np.arange(8, dtype=np.float32)
    first = MtileBuffer._from_numpy_implicit(source)
    source.resize((2, 4), refcheck=False)
    second = MtileBuffer._from_numpy_implicit(source)
    assert second is not first
    assert second.shape == (2, 4)
    np.testing.assert_array_equal(second.numpy(), source)

    handles = [first.metal_buffer, second.metal_buffer]
    del source, first, second
    gc.collect()
    assert sorted(fake_device.releases) == handles
    assert not buffer_module._buffer_cache


def test_prepared_resources_retain_allocation_through_completion(fake_device, monkeypatch):
    buffer = MtileBuffer.empty((8,), np.float32)
    handle = buffer.metal_buffer
    compiled = SimpleNamespace(
        pipeline=100,
        is_gemm=False,
        prefer_ordered=True,
        description_bits=0,
        execution_report={},
        output_indices=(0,),
        threadgroup_size=(32, 1, 1),
    )
    dispatch = FastDispatcher(compiled, [handle], (1,), fake_device, resources=[buffer])
    del buffer
    gc.collect()
    assert not fake_device.releases

    device = MetalDevice.__new__(MetalDevice)
    device._dispatch_lock = threading.RLock()
    device._pending_encoder = 3
    device._pending_cmd_buffer = 4
    device._pending_dispatches = 1
    device._pending_completion_spin_ns = 0
    device._pending_inputs = set()
    device._pending_outputs = set()
    device._pending_lifetimes = {id(dispatch): dispatch}
    device._inflight_lifetimes = []
    device._ensure_cached_selectors = lambda: None
    device.__dict__["low_latency_spin_ns"] = 0
    monkeypatch.setattr(MetalDevice, "_msg_send_void", lambda *_: None)
    monkeypatch.setattr(MetalDevice, "_msg_send_uint64", lambda *_: 4)

    del dispatch
    gc.collect()
    assert not fake_device.releases
    device.flush()
    gc.collect()
    assert not device._pending_lifetimes
    assert len(device._inflight_lifetimes) == 1
    assert not fake_device.releases

    device.sync()
    gc.collect()
    assert not device._inflight_lifetimes
    assert fake_device.releases == [handle]


def test_failed_contents_lookup_releases_new_allocation(fake_device, monkeypatch):
    def unavailable_contents(_):
        raise RuntimeError("contents unavailable")

    monkeypatch.setattr(fake_device, "buffer_contents", unavailable_contents)
    with pytest.raises(RuntimeError, match="contents unavailable"):
        MtileBuffer.empty((8,), np.float32)
    assert fake_device.releases == [1]
    assert not fake_device.allocations


def test_release_buffer_balances_native_new_ownership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        metal_device_module, "_send", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    device = MetalDevice.__new__(MetalDevice)
    device.release_buffer(123)
    assert calls == [((123, "release"), {"restype": None})]


@pytest.fixture
def mocked_launcher(fake_device, monkeypatch):
    @kernel_module.kernel
    def copy_values(source, ignored, destination):
        pass

    launcher = copy_values[(1,)]
    compiled = kernel_module.CompiledKernel(
        pipeline=100,
        msl_source="",
        func_name="copy_values",
        threadgroup_size=(32, 1, 1),
        output_indices=(0,),
        argument_indices=(2, 0),
    )
    dispatches = []

    def dispatch(_, arguments):
        dispatches.append(tuple(arguments))
        arguments[2]._np_view[:] = arguments[0]._np_view + 1
        return [arguments[2].metal_buffer, arguments[0].metal_buffer]

    monkeypatch.setattr(kernel_module, "_kernel_cache", {})
    monkeypatch.setattr(launcher, "_compile", lambda *_: compiled)
    monkeypatch.setattr(launcher, "_dispatch", dispatch)
    return launcher, dispatches


def test_launcher_accepts_readonly_input_and_copies_only_output(
    fake_device, mocked_launcher, monkeypatch
):
    launcher, dispatches = mocked_launcher
    source = np.arange(8, dtype=np.float32)
    source.flags.writeable = False
    destination = np.zeros_like(source)
    copied_sources = []
    original_copy = MtileBuffer.sync_to_source

    def record_copy(buffer):
        copied_sources.append(buffer._source_array)
        original_copy(buffer)

    monkeypatch.setattr(MtileBuffer, "sync_to_source", record_copy)
    launcher(source, 123, destination)

    assert len(dispatches) == 1
    assert fake_device.sync_count == 1
    assert len(copied_sources) == 1
    assert copied_sources[0] is destination
    np.testing.assert_array_equal(source, np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(destination, source + 1)


@pytest.mark.parametrize("prepare", [False, True])
def test_launcher_rejects_readonly_output_before_dispatch(fake_device, mocked_launcher, prepare):
    launcher, dispatches = mocked_launcher
    source = np.arange(8, dtype=np.float32)
    destination = np.zeros_like(source)
    destination.flags.writeable = False

    with pytest.raises(ValueError, match="output NumPy arrays must be writable"):
        if prepare:
            launcher.prepare(source, 123, destination)
        else:
            launcher(source, 123, destination)

    assert not dispatches
    assert fake_device.sync_count == 0
    np.testing.assert_array_equal(destination, 0)


def test_launcher_copy_back_updates_noncontiguous_output(fake_device, mocked_launcher):
    launcher, dispatches = mocked_launcher
    source = np.arange(8, dtype=np.float32)
    backing = np.zeros(16, dtype=np.float32)
    destination = backing[::2]

    launcher(source, 123, destination)

    assert len(dispatches) == 1
    np.testing.assert_array_equal(backing[::2], source + 1)
    np.testing.assert_array_equal(backing[1::2], 0)


def test_launcher_inplace_alias_copies_output_once(fake_device, mocked_launcher):
    launcher, dispatches = mocked_launcher
    source = np.arange(8, dtype=np.float32)

    launcher(source, 123, source)

    assert dispatches[0][0] is dispatches[0][2]
    np.testing.assert_array_equal(source, np.arange(8, dtype=np.float32) + 1)


@pytest.mark.parametrize("operation", [tir.Store, tir.TileStore])
def test_output_detection_tracks_offset_store_aliases(operation):
    function = tir.Function("aliased_output")
    function.params = [tir.Param("destination", PtrType("f32"))]
    base = tir.Value("destination", PtrType("f32"))
    offset = function.add_op(tir.Constant(value=8), "offset")
    alias = function.add_op(tir.PtrOffset(ptr=base, offsets=offset), "alias")
    function.ops.append(operation(ptr=alias))

    kernel_module._mark_outputs(function)

    assert function.params[0].is_output


def test_output_detection_includes_mutated_persistent_counter():
    function = tir.Function("persistent_output")
    function.params = [tir.Param("counter", PtrType("u32"))]
    counter = tir.Value("counter", PtrType("u32"))
    function.ops.append(tir.PersistentRange(counter=counter, total=1))

    kernel_module._mark_outputs(function)

    assert function.params[0].is_output
