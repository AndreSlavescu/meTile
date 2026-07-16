from __future__ import annotations

import inspect
import os
import statistics
import threading
import time

from metile.compiler.schedule_search import choose_mdl_tie
from metile.frontend.tracing import constexpr
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest
from metile.runtime.metal_device import MetalDevice, completion_spin_budget_ns


class Config:
    """A set of constexpr parameter values for a kernel."""

    def __init__(self, num_simdgroups: int = 4, num_stages: int = 1, **kwargs):
        self.kwargs = kwargs
        self.num_simdgroups = num_simdgroups
        self.num_stages = num_stages
        if num_stages > 1:
            self.kwargs["num_stages"] = num_stages

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"Config({params})"

    def __hash__(self):
        return hash(tuple(sorted(self.kwargs.items())))

    def __eq__(self, other):
        if not isinstance(other, Config):
            return NotImplemented
        return self.kwargs == other.kwargs


# Global cache: (func_name, config_digest, key_values) -> best Config
_autotune_cache: dict = {}
_autotune_latency_cache: dict = {}
_persistent_cache_lock = threading.Lock()
_persistent_cache_path = cache_root() / "autotune-v4.json"

_REFINEMENT_MARGIN = 0.08
_REFINEMENT_MAX_CANDIDATES = 8
_REFINEMENT_REPS = 30


def _selection_score(gpu_samples, wall_samples):
    """Score short kernels by launch-to-completion latency, long kernels by GPU time."""
    if not wall_samples:
        return None
    wall_time = statistics.median(wall_samples)
    if not gpu_samples:
        return wall_time
    gpu_time = statistics.median(gpu_samples)
    return wall_time if completion_spin_budget_ns(gpu_time) else gpu_time


class AutotunedKernel:
    """Wraps a KernelFunction with autotuning over a set of configs."""

    def __init__(
        self,
        kernel_fn,
        configs: list[Config],
        key: list[str],
        warmup: int = 5,
        rep: int = 20,
        verbose: bool = True,
    ):
        self.kernel_fn = kernel_fn
        self.configs = configs
        self.key = key
        self.warmup = warmup
        self.rep = rep
        self.verbose = verbose
        self._sig = inspect.signature(kernel_fn.fn) if hasattr(kernel_fn, "fn") else None
        self._config_digest = stable_digest([config.kwargs for config in configs])

    @property
    def name(self):
        return self.kernel_fn.name

    def __getitem__(self, grid):
        if isinstance(grid, int):
            grid = (grid,)
        return AutotunedLauncher(self, grid)


class AutotunedLauncher:
    """Launcher that autotunes on first call, then caches the best config."""

    def __init__(self, autotuned: AutotunedKernel, grid):
        self.autotuned = autotuned
        self.grid = grid

    def __call__(self, *args, **kwargs):
        at = self.autotuned
        if self._has_explicit_kernel_config(kwargs):
            grid = self.grid(kwargs) if callable(self.grid) else self.grid
            return at.kernel_fn[grid](*args, **kwargs)

        key_values = self._extract_key_values(args, kwargs)
        cache_key = self._cache_key(key_values)

        if cache_key in _autotune_cache:
            best = _autotune_cache[cache_key]
            self._launch(best, args, kwargs)
            return best

        persistent_key = self._persistent_key(key_values)
        cached = self._load_persistent(persistent_key)
        if cached is not None:
            best, gpu_seconds = cached
            _autotune_cache[cache_key] = best
            _autotune_latency_cache[cache_key] = gpu_seconds
            self._launch(best, args, kwargs)
            return best

        dev = MetalDevice.get()
        results = self._benchmark_candidates(args, kwargs, dev)
        successful = [
            (dt, description_bits, cfg)
            for cfg, dt, description_bits, error, _ in results
            if error is None and dt is not None and description_bits is not None
        ]

        if not successful:
            raise RuntimeError(f"All {len(at.configs)} configs failed for '{at.kernel_fn.name}'")
        best = choose_mdl_tie(successful)
        selection_time, gpu_seconds = next(
            (dt, gpu_time) for cfg, dt, _, _, gpu_time in results if cfg == best
        )
        gpu_seconds = gpu_seconds if gpu_seconds is not None else selection_time

        _autotune_cache[cache_key] = best
        _autotune_latency_cache[cache_key] = gpu_seconds
        self._store_persistent(persistent_key, best, gpu_seconds)

        if at.verbose:
            key_str = ", ".join(f"{k}={v}" for k, v in zip(at.key, key_values))
            print(f"autotune {at.kernel_fn.name} [{key_str}]: {best}")
            for cfg, dt, _, err, _ in results:
                tag = " <--" if cfg == best else ""
                if dt is not None:
                    print(f"  {cfg}: {dt * 1000:.2f}ms{tag}")
                else:
                    reason = f" ({err})" if err else ""
                    print(f"  {cfg}: FAILED{reason}")

        self._launch(best, args, kwargs)
        return best

    def prepare(self, *args, **kwargs):
        """Autotune then return a FastDispatcher for the best config."""
        if self._has_explicit_kernel_config(kwargs):
            grid = self.grid(kwargs) if callable(self.grid) else self.grid
            return self.autotuned.kernel_fn[grid].prepare(*args, **kwargs)
        cfg = self(*args, **kwargs)
        merged = {**kwargs, **cfg.kwargs}
        grid = self._resolve_grid(cfg)
        dispatch = self.autotuned.kernel_fn[grid].prepare(*args, **merged)
        key_values = self._extract_key_values(args, kwargs)
        cache_key = self._cache_key(key_values)
        gpu_seconds = _autotune_latency_cache.get(cache_key)
        if gpu_seconds is not None:
            dispatch._completion_spin_ns = completion_spin_budget_ns(gpu_seconds)
        return dispatch

    def _extract_key_values(self, args, kwargs):
        sig = self.autotuned._sig or inspect.signature(self.autotuned.kernel_fn.fn)
        params = list(sig.parameters.keys())
        values = []
        for name in self.autotuned.key:
            if name in kwargs:
                val = kwargs[name]
            else:
                idx = params.index(name) if name in params else -1
                val = args[idx] if 0 <= idx < len(args) else None
            values.append(val.shape if hasattr(val, "shape") else val)
        return tuple(values)

    def _cache_key(self, key_values):
        return (
            self.autotuned.kernel_fn.name,
            self.autotuned._config_digest,
            tuple(key_values),
        )

    def _resolve_grid(self, config):
        return self.grid(config.kwargs) if callable(self.grid) else self.grid

    def _has_explicit_kernel_config(self, kwargs):
        signature = self.autotuned._sig
        if signature is None:
            return False
        constexpr_names = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.annotation is constexpr
        }
        return bool(constexpr_names) and constexpr_names.issubset(kwargs)

    def _persistent_key(self, key_values):
        at = self.autotuned
        try:
            source = inspect.getsource(at.kernel_fn.fn)
        except (OSError, TypeError):
            source = at.kernel_fn.name
        dev = MetalDevice.get()
        return stable_digest(
            {
                "configs": [cfg.kwargs for cfg in at.configs],
                "device": dev.name,
                "kernel": at.kernel_fn.name,
                "keys": key_values,
                "source": source,
                "toolchain": dev.metal_compiler_version,
            }
        )

    def _load_persistent(self, key):
        if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
            return None
        with _persistent_cache_lock:
            payload = read_json(_persistent_cache_path, {})
        selection = payload.get(key)
        if not isinstance(selection, dict):
            return None
        kwargs = selection.get("config")
        gpu_seconds = selection.get("gpu_seconds")
        if not isinstance(kwargs, dict) or not isinstance(gpu_seconds, (int, float)):
            return None
        for config in self.autotuned.configs:
            if config.kwargs == kwargs:
                return config, float(gpu_seconds)
        return None

    def _store_persistent(self, key, config, gpu_seconds):
        if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
            return
        with _persistent_cache_lock:
            payload = read_json(_persistent_cache_path, {})
            payload[key] = {
                "config": config.kwargs,
                "gpu_seconds": gpu_seconds,
            }
            atomic_write_json(_persistent_cache_path, payload)

    def _launch(self, config, args, kwargs):
        merged = {**kwargs, **config.kwargs}
        self.autotuned.kernel_fn[self._resolve_grid(config)](*args, **merged)

    def _benchmark_candidates(self, args, kwargs, dev):
        at = self.autotuned
        states = []
        for config in at.configs:
            try:
                merged = {**kwargs, **config.kwargs}
                grid = self._resolve_grid(config)
                dispatch = at.kernel_fn[grid].prepare(*args, **merged)
                dispatch._completion_spin_ns = 1_500_000
                states.append(
                    {
                        "config": config,
                        "description_bits": dispatch.description_bits,
                        "dispatch": dispatch,
                        "gpu_samples": [],
                        "wall_samples": [],
                        "error": None,
                    }
                )
            except Exception as error:
                states.append(
                    {
                        "config": config,
                        "description_bits": None,
                        "dispatch": None,
                        "gpu_samples": [],
                        "wall_samples": [],
                        "error": error,
                    }
                )

        def run_round(round_index, timed, candidates=None):
            active = candidates or [state for state in states if state["error"] is None]
            if not active:
                return
            shift = round_index % len(active)
            ordered = active[shift:] + active[:shift]
            if round_index & 1:
                ordered.reverse()
            for state in ordered:
                try:
                    start = time.perf_counter_ns()
                    state["dispatch"]()
                    dev.sync()
                    if timed:
                        gpu_time = dev.gpu_elapsed()
                        state["wall_samples"].append((time.perf_counter_ns() - start) * 1e-9)
                        if gpu_time > 0:
                            state["gpu_samples"].append(gpu_time)
                except Exception as error:
                    state["error"] = error

        def score(state):
            return _selection_score(state["gpu_samples"], state["wall_samples"])

        for round_index in range(at.warmup):
            run_round(round_index, timed=False)
        for round_index in range(at.rep):
            run_round(round_index, timed=True)

        ranked = sorted(
            (state for state in states if state["wall_samples"] and state["error"] is None),
            key=score,
        )
        if ranked:
            cutoff = score(ranked[0]) * (1.0 + _REFINEMENT_MARGIN)
            finalists = [state for state in ranked if score(state) <= cutoff][
                :_REFINEMENT_MAX_CANDIDATES
            ]
            if len(finalists) > 1:
                for round_index in range(_REFINEMENT_REPS):
                    run_round(at.rep + round_index, timed=True, candidates=finalists)

        return [
            (
                state["config"],
                score(state) if state["wall_samples"] else None,
                state["description_bits"],
                state["error"],
                statistics.median(state["gpu_samples"]) if state["gpu_samples"] else None,
            )
            for state in states
        ]


def autotune(
    configs: list[Config], key: list[str], warmup: int = 5, rep: int = 20, verbose: bool = True
):
    """Decorator that automates kernel parameter search.

    Args:
        configs: List of Config objects to try.
        key: Argument names for cache key (e.g. ['M', 'N', 'K']).
        warmup: Warmup iterations per config.
        rep: Timed iterations per config.
        verbose: Print selected config and timing results.
    """

    def decorator(kernel_fn):
        if isinstance(kernel_fn, AutotunedKernel):
            kernel_fn = kernel_fn.kernel_fn
        return AutotunedKernel(kernel_fn, configs, key, warmup, rep, verbose)

    return decorator
