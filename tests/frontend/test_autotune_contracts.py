from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from metile.frontend import autotune as autotune_module
from metile.frontend.autotune import AutotunedKernel, Config
from metile.frontend.tracing import constexpr


def _signature(source, destination, size, BLOCK: constexpr, LANES: constexpr):
    pass


def _other_signature(source, destination, size, BLOCK: constexpr, LANES: constexpr):
    return size


@dataclass(frozen=True)
class _Schedule:
    vector_width: int = 4
    stages: tuple[int, ...] = (1, 2)


class _Kernel:
    name = "contract_kernel"
    fn = staticmethod(_signature)

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        kernel = self

        class Launcher:
            def __call__(self, *args, **kwargs):
                kernel.calls.append(("launch", grid, args, kwargs))

            def prepare(self, *args, **kwargs):
                kernel.calls.append(("prepare", grid, args, kwargs))
                return _Dispatcher()

        return Launcher()


class _Dispatcher:
    description_bits = 10
    _completion_spin_ns = 0

    def __call__(self):
        pass


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(autotune_module, "_autotune_cache", {})
    monkeypatch.setattr(autotune_module, "_autotune_latency_cache", {})
    monkeypatch.setattr(autotune_module, "_persistent_cache_path", tmp_path / "autotune.json")
    monkeypatch.setattr(autotune_module, "_compiler_identity", lambda: "compiler-v1")
    monkeypatch.delenv("METILE_DISABLE_DISK_CACHE", raising=False)
    device = SimpleNamespace(
        name="CPU mock",
        metal_compiler_version="test",
        sync=lambda: None,
        gpu_elapsed=lambda: 0.0001,
    )
    monkeypatch.setattr(autotune_module.MetalDevice, "get", lambda: device)


def _launcher(configs=None, grid=(1,), kernel=None):
    tuned = AutotunedKernel(
        kernel or _Kernel(),
        configs or [Config(BLOCK=64, LANES=32)],
        ["size"],
        warmup=0,
        rep=1,
        verbose=False,
    )
    return tuned[grid]


def _keys(launcher, args, kwargs):
    values = launcher._extract_key_values(args, kwargs)
    contract = launcher._call_contract(args, kwargs)
    return launcher._cache_key(values, contract), launcher._persistent_key(values, contract)


def test_config_leaves_simdgroup_count_automatic_by_default():
    config = Config(BLOCK=64)

    assert config.num_simdgroups is None
    assert "NUM_SG" not in config.kwargs


def test_config_forwards_explicit_simdgroup_count():
    config = Config(num_simdgroups=8, BLOCK=64)

    assert config.num_simdgroups == 8
    assert config.kwargs["NUM_SG"] == 8
    assert config != Config(BLOCK=64)
    assert config == Config(NUM_SG=8, BLOCK=64)
    assert hash(config) == hash(Config(NUM_SG=8, BLOCK=64))


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "4"])
def test_config_rejects_invalid_simdgroup_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        Config(num_simdgroups=count)


def test_config_rejects_conflicting_simdgroup_spellings():
    with pytest.raises(ValueError, match="must agree"):
        Config(num_simdgroups=2, NUM_SG=4)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (np.zeros(64, np.float16), np.zeros(64, np.float32)),
        (np.zeros(64, np.float32), np.zeros(128, np.float32)),
        (np.zeros(128, np.float32)[::2], np.zeros(64, np.float32)),
        (1, 1.0),
    ],
)
def test_cache_keys_distinguish_operand_contracts(first, second):
    launcher = _launcher()
    destination = np.zeros(64, np.float32)

    first_keys = _keys(launcher, (first, destination, 64), {})
    second_keys = _keys(launcher, (second, destination, 64), {})

    assert first_keys[0] != second_keys[0]
    assert first_keys[1] != second_keys[1]


def test_cache_keys_do_not_depend_on_operand_contents_or_identity():
    launcher = _launcher()

    first = _keys(launcher, (np.zeros(64), np.zeros(64), 64), {})
    second = _keys(launcher, (np.ones(64), np.ones(64), 64), {})

    assert first == second


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"RELAXED_PRECISION": False}, {"RELAXED_PRECISION": True}),
        ({"BLOCK": 32}, {"BLOCK": 64}),
        ({"SCHEDULE": _Schedule(1)}, {"SCHEDULE": _Schedule(4)}),
        ({}, {"RELAXED_PRECISION": False}),
    ],
)
def test_cache_keys_distinguish_caller_compiler_overrides(first, second):
    launcher = _launcher()
    args = (np.zeros(64), np.zeros(64), 64)

    first_keys = _keys(launcher, args, first)
    second_keys = _keys(launcher, args, second)

    assert first_keys[0] != second_keys[0]
    assert first_keys[1] != second_keys[1]


def test_cache_keys_normalize_keyword_argument_order():
    launcher = _launcher()
    args = (np.zeros(64), np.zeros(64), 64)

    first = _keys(launcher, args, {"RELAXED_PRECISION": False, "SCHEDULE": _Schedule()})
    second = _keys(launcher, args, {"SCHEDULE": _Schedule(), "RELAXED_PRECISION": False})

    assert first == second


def test_cache_keys_distinguish_kernel_sources_with_the_same_name():
    first = _launcher()
    kernel = _Kernel()
    kernel.fn = _other_signature
    second = _launcher(kernel=kernel)
    args = (np.zeros(64), np.zeros(64), 64)

    first_keys = _keys(first, args, {})
    second_keys = _keys(second, args, {})

    assert first_keys[0] != second_keys[0]
    assert first_keys[1] != second_keys[1]


def test_cache_keys_distinguish_compiler_implementation(monkeypatch):
    launcher = _launcher()
    args = (np.zeros(64), np.zeros(64), 64)
    first = _keys(launcher, args, {})

    monkeypatch.setattr(autotune_module, "_compiler_identity", lambda: "compiler-v2")
    second = _keys(launcher, args, {})

    assert first[0] != second[0]
    assert first[1] != second[1]


@pytest.mark.parametrize("option", ["METILE_ONLINE_SOFTMAX", "METILE_SCHEDULE"])
def test_cache_keys_distinguish_compiler_environment(monkeypatch, option):
    launcher = _launcher()
    args = (np.zeros(64), np.zeros(64), 64)
    monkeypatch.setenv(option, "0")
    first = _keys(launcher, args, {})
    monkeypatch.setenv(option, "1")
    second = _keys(launcher, args, {})

    assert first[0] != second[0]
    assert first[1] != second[1]


def test_persistent_selection_round_trips_frozen_dataclass_configs():
    config = Config(BLOCK=64, LANES=32, SCHEDULE=_Schedule())
    launcher = _launcher([config])

    launcher._store_persistent("selection", config, 0.0001)

    assert launcher._load_persistent("selection") == (config, 0.0001)
    assert (
        _launcher([Config(BLOCK=64, LANES=32, SCHEDULE=_Schedule(1))])._load_persistent("selection")
        is None
    )


def test_candidate_preparation_and_grid_honor_caller_overrides():
    config = Config(BLOCK=64, LANES=32, RELAXED_PRECISION=True, SCHEDULE=_Schedule(1))
    launcher = _launcher([config], lambda meta: (meta["BLOCK"],))
    overrides = {"BLOCK": 128, "RELAXED_PRECISION": False, "SCHEDULE": _Schedule(4)}

    results = launcher._benchmark_candidates(
        (np.zeros(64), np.zeros(64), 64), overrides, autotune_module.MetalDevice.get()
    )

    assert results[0][3] is None
    kind, grid, _, kwargs = launcher.autotuned.kernel_fn.calls[0]
    assert kind == "prepare"
    assert grid == (128,)
    assert kwargs == {"LANES": 32, **overrides}


def test_launch_honors_caller_overrides():
    config = Config(BLOCK=64, LANES=32, RELAXED_PRECISION=True)
    launcher = _launcher([config], lambda meta: (meta["BLOCK"],))

    launcher._launch(config, (), {"BLOCK": 128, "RELAXED_PRECISION": False})

    assert launcher.autotuned.kernel_fn.calls == [
        ("launch", (128,), (), {"BLOCK": 128, "LANES": 32, "RELAXED_PRECISION": False})
    ]


def test_prepare_preserves_overrides_and_cached_completion_budget(monkeypatch):
    config = Config(BLOCK=64, LANES=32, RELAXED_PRECISION=True)
    launcher = _launcher([config], lambda meta: (meta["BLOCK"],))
    benchmark_calls = []

    def benchmark(*args):
        benchmark_calls.append(args)
        return [(config, 0.0005, 1, None, 0.0001)]

    monkeypatch.setattr(launcher, "_benchmark_candidates", benchmark)
    args = (np.zeros(64), np.zeros(64), 64)
    overrides = {"BLOCK": 128, "RELAXED_PRECISION": False}

    first = launcher.prepare(*args, **overrides)
    autotune_module._autotune_cache.clear()
    autotune_module._autotune_latency_cache.clear()
    second = launcher.prepare(*args, **overrides)

    assert len(benchmark_calls) == 1
    assert first._completion_spin_ns >= 900_000
    assert second._completion_spin_ns == first._completion_spin_ns
    for _, grid, _, kwargs in launcher.autotuned.kernel_fn.calls:
        assert grid == (128,)
        assert kwargs == {"LANES": 32, **overrides}


def test_tuning_repeats_when_dtype_or_precision_changes(monkeypatch):
    config = Config(BLOCK=64, LANES=32)
    launcher = _launcher([config])
    benchmark_calls = []

    def benchmark(*args):
        benchmark_calls.append(args)
        return [(config, 0.0005, 1, None, 0.0001)]

    monkeypatch.setattr(launcher, "_benchmark_candidates", benchmark)

    for dtype, relaxed in [(np.float16, False), (np.float32, False), (np.float32, True)]:
        args = (np.zeros(64, dtype), np.zeros(64, dtype), 64)
        launcher(*args, RELAXED_PRECISION=relaxed)
        launcher(*args, RELAXED_PRECISION=relaxed)

    assert len(benchmark_calls) == 3
    assert len(autotune_module._autotune_cache) == 3
    assert len(autotune_module._autotune_latency_cache) == 3
