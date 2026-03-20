from __future__ import annotations

import inspect
import time

from metile.runtime.metal_device import MetalDevice


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


# Global cache: (func_name, key_values) -> best Config
_autotune_cache: dict = {}


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
        key_values = self._extract_key_values(args, kwargs)
        cache_key = (at.kernel_fn.name, tuple(key_values))

        if cache_key in _autotune_cache:
            best = _autotune_cache[cache_key]
            self._launch(best, args, kwargs)
            return best

        # Benchmark each config
        dev = MetalDevice.get()
        best, best_time = None, float("inf")
        results = []

        for cfg in at.configs:
            try:
                dt = self._bench(cfg, args, kwargs, dev)
                results.append((cfg, dt, None))
                if dt < best_time:
                    best_time = dt
                    best = cfg
            except Exception as e:
                results.append((cfg, None, e))

        if best is None:
            raise RuntimeError(f"All {len(at.configs)} configs failed for '{at.kernel_fn.name}'")

        _autotune_cache[cache_key] = best

        if at.verbose:
            key_str = ", ".join(f"{k}={v}" for k, v in zip(at.key, key_values))
            print(f"autotune {at.kernel_fn.name} [{key_str}]: {best}")
            for cfg, dt, err in results:
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
        cfg = self(*args, **kwargs)
        merged = {**kwargs, **cfg.kwargs}
        grid = self._resolve_grid(cfg)
        return self.autotuned.kernel_fn[grid].prepare(*args, **merged)

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

    def _resolve_grid(self, config):
        return self.grid(config.kwargs) if callable(self.grid) else self.grid

    def _launch(self, config, args, kwargs):
        merged = {**kwargs, **config.kwargs}
        self.autotuned.kernel_fn[self._resolve_grid(config)](*args, **merged)

    def _bench(self, config, args, kwargs, dev):
        at = self.autotuned
        merged = {**kwargs, **config.kwargs}
        grid = self._resolve_grid(config)
        launcher = at.kernel_fn[grid]

        for _ in range(at.warmup):
            launcher(*args, **merged)
        dev.sync()

        t0 = time.perf_counter()
        for _ in range(at.rep):
            launcher(*args, **merged)
            dev.sync()
        return (time.perf_counter() - t0) / at.rep


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
        return AutotunedKernel(kernel_fn, configs, key, warmup, rep, verbose)

    return decorator
