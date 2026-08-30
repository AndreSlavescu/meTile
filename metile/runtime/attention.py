from __future__ import annotations

import inspect
import os
import statistics
import threading
from dataclasses import asdict, dataclass
from math import prod

import metile
from metile.compiler.schedule_search import choose_mdl_tie
from metile.kernels.attention import (
    ATTENTION_PARTIAL_CONFIGS,
    attention_decode_kernel,
    attention_decode_merge_kernel,
    attention_decode_partial_kernel,
    attention_decode_single_pass,
)
from metile.runtime.cache import atomic_write_json, cache_root, read_json, stable_digest
from metile.runtime.metal_device import MetalDevice, completion_spin_budget_ns

_attention_config_cache = {}
_attention_cache_lock = threading.RLock()
_attention_cache_path = cache_root() / "attention-autotune-v1.json"


@dataclass(frozen=True)
class AttentionDecodeConfig:
    algorithm: str
    tokens_per_block: int = 0
    partial_block: int = 0
    merge_block: int = 0


class _TwoPassDispatcher:
    __slots__ = ("first", "second")

    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.second._concurrent = False

    def __call__(self):
        device = self.first._dev
        with device._dispatch_lock:
            self.first._encode_unlocked(device)
            self.second._encode_unlocked(device)

    def repeat(self, count):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        device = self.first._dev
        with device._dispatch_lock:
            for _ in range(count):
                self.first._encode_unlocked(device)
                self.second._encode_unlocked(device)

    @property
    def description_bits(self):
        return self.first.description_bits + self.second.description_bits

    def set_completion_budget(self, budget):
        if budget:
            self.first._completion_spin_ns = 1
            self.second._completion_spin_ns = budget - 1
        else:
            self.first._completion_spin_ns = 0
            self.second._completion_spin_ns = 0


def _two_pass_candidates(tokens):
    candidates = set(ATTENTION_PARTIAL_CONFIGS)
    dynamic_tokens = metile.cdiv(metile.cdiv(tokens, 32), 256) * 256
    candidates.add((dynamic_tokens, 512, 128))
    return [
        AttentionDecodeConfig("two_pass", tokens_per_block, partial_block, merge_block)
        for tokens_per_block, partial_block, merge_block in sorted(candidates)
        if metile.cdiv(tokens, tokens_per_block) <= 32
    ]


def _prepare_two_pass(
    query,
    key,
    value,
    output,
    batch,
    query_heads,
    key_value_heads,
    tokens,
    scale,
    dimension,
    config,
    scratch=None,
):
    total_query_heads = batch * query_heads
    num_blocks = metile.cdiv(tokens, config.tokens_per_block)
    if scratch is None:
        partial_output = metile.Buffer.empty((total_query_heads * num_blocks * dimension,))
        partial_maximum = metile.Buffer.empty((total_query_heads * num_blocks * 32,))
        partial_sum = metile.Buffer.empty((total_query_heads * num_blocks * 32,))
    else:
        partial_output, partial_maximum, partial_sum = scratch

    first = attention_decode_partial_kernel[(total_query_heads * num_blocks,)].prepare(
        query,
        key,
        value,
        partial_output,
        partial_maximum,
        partial_sum,
        tokens,
        scale,
        D=dimension,
        Q_HEADS=query_heads,
        KV_HEADS=key_value_heads,
        NUM_BLOCKS=num_blocks,
        TOKENS_PER_BLOCK=config.tokens_per_block,
        BLOCK=config.partial_block,
    )
    second = attention_decode_merge_kernel[(total_query_heads,)].prepare(
        partial_output,
        partial_maximum,
        partial_sum,
        output,
        D=dimension,
        NUM_BLOCKS=num_blocks,
        BLOCK=config.merge_block,
    )
    return _TwoPassDispatcher(first, second)


def _attention_persistent_key(batch, query_heads, key_value_heads, tokens, dimension, candidates):
    device = MetalDevice.get()
    return stable_digest(
        {
            "candidates": [asdict(candidate) for candidate in candidates],
            "device": device.name,
            "dimension": dimension,
            "batch": batch,
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "merge_source": inspect.getsource(attention_decode_merge_kernel.fn),
            "partial_source": inspect.getsource(attention_decode_partial_kernel.fn),
            "single_source": inspect.getsource(attention_decode_kernel.fn),
            "tokens": tokens,
            "toolchain": device.metal_compiler_version,
        }
    )


def _read_attention_config(persistent_key, candidates):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return None
    payload = read_json(_attention_cache_path, {}).get(persistent_key)
    if not isinstance(payload, dict):
        return None
    for candidate in candidates:
        if asdict(candidate) == payload.get("config"):
            return candidate, payload.get("gpu_seconds", 0.0)
    return None


def _write_attention_config(persistent_key, config, gpu_seconds):
    if os.environ.get("METILE_DISABLE_DISK_CACHE") == "1":
        return
    payload = read_json(_attention_cache_path, {})
    payload[persistent_key] = {
        "config": asdict(config),
        "gpu_seconds": gpu_seconds,
    }
    atomic_write_json(_attention_cache_path, payload)


def _tune_attention_dispatch(dispatches):
    device = MetalDevice.get()
    samples = {config: [] for config, _ in dispatches}
    for _, dispatch in dispatches:
        for _ in range(3):
            dispatch()
            device.sync()

    for round_index in range(15):
        shift = round_index % len(dispatches)
        ordered = dispatches[shift:] + dispatches[:shift]
        if round_index & 1:
            ordered.reverse()
        for config, dispatch in ordered:
            dispatch()
            device.sync()
            elapsed = device.gpu_elapsed()
            if elapsed > 0:
                samples[config].append(elapsed)

    results = [
        (statistics.median(samples[config]), dispatch.description_bits, config)
        for config, dispatch in dispatches
        if samples[config]
    ]
    if not results:
        raise RuntimeError("GPU timing failed while tuning decode attention")
    selected = choose_mdl_tie(results)
    gpu_seconds = next(elapsed for elapsed, _, config in results if config == selected)
    dispatch = next(dispatch for config, dispatch in dispatches if config == selected)
    return selected, gpu_seconds, dispatch


def _prepare_attention_decode(
    query,
    key,
    value,
    output,
    batch,
    query_heads,
    key_value_heads,
    tokens,
    scale,
    dimension,
):
    total_query_heads = batch * query_heads
    candidates = [AttentionDecodeConfig("single_pass")]
    if tokens >= 4096:
        candidates.extend(_two_pass_candidates(tokens))
    tuning_key = (batch, query_heads, key_value_heads, tokens, dimension)
    persistent_key = _attention_persistent_key(
        batch, query_heads, key_value_heads, tokens, dimension, candidates
    )

    with _attention_cache_lock:
        cached = _attention_config_cache.get(tuning_key)
        if cached is None:
            cached = _read_attention_config(persistent_key, candidates)

        if cached is not None:
            selected, gpu_seconds = cached
            if selected.algorithm == "single_pass":
                return attention_decode_single_pass[(total_query_heads,)].prepare(
                    query,
                    key,
                    value,
                    output,
                    tokens,
                    scale,
                    D=dimension,
                    Q_HEADS=query_heads,
                    KV_HEADS=key_value_heads,
                )
            dispatch = _prepare_two_pass(
                query,
                key,
                value,
                output,
                batch,
                query_heads,
                key_value_heads,
                tokens,
                scale,
                dimension,
                selected,
            )
            dispatch.set_completion_budget(completion_spin_budget_ns(gpu_seconds))
            return dispatch

        dispatches = []
        single_dispatch = attention_decode_single_pass[(total_query_heads,)].prepare(
            query,
            key,
            value,
            output,
            tokens,
            scale,
            D=dimension,
            Q_HEADS=query_heads,
            KV_HEADS=key_value_heads,
        )
        dispatches.append((candidates[0], single_dispatch))

        two_pass = candidates[1:]
        if two_pass:
            max_blocks = max(metile.cdiv(tokens, config.tokens_per_block) for config in two_pass)
            scratch = (
                metile.Buffer.empty((total_query_heads * max_blocks * dimension,)),
                metile.Buffer.empty((total_query_heads * max_blocks * 32,)),
                metile.Buffer.empty((total_query_heads * max_blocks * 32,)),
            )
            for config in two_pass:
                dispatches.append(
                    (
                        config,
                        _prepare_two_pass(
                            query,
                            key,
                            value,
                            output,
                            batch,
                            query_heads,
                            key_value_heads,
                            tokens,
                            scale,
                            dimension,
                            config,
                            scratch,
                        ),
                    )
                )

        selected, gpu_seconds, dispatch = _tune_attention_dispatch(dispatches)
        _attention_config_cache[tuning_key] = (selected, gpu_seconds)
        _write_attention_config(persistent_key, selected, gpu_seconds)
        budget = completion_spin_budget_ns(gpu_seconds)
        if isinstance(dispatch, _TwoPassDispatcher):
            dispatch.set_completion_budget(budget)
        else:
            dispatch._completion_spin_ns = budget
        return dispatch


class _AttentionDecodeLauncher:
    def __init__(self, batch, query_heads):
        self.batch = batch
        self.query_heads = query_heads

    def prepare(self, query, key, value, output, tokens, scale, *, D, KV_HEADS=None):
        key_value_heads = self.query_heads if KV_HEADS is None else KV_HEADS
        _validate_attention_arguments(
            query,
            key,
            value,
            output,
            self.batch,
            self.query_heads,
            key_value_heads,
            tokens,
            D,
        )
        return _prepare_attention_decode(
            query,
            key,
            value,
            output,
            self.batch,
            self.query_heads,
            key_value_heads,
            tokens,
            scale,
            D,
        )

    def __call__(self, query, key, value, output, tokens, scale, *, D, KV_HEADS=None):
        dispatch = self.prepare(query, key, value, output, tokens, scale, D=D, KV_HEADS=KV_HEADS)
        dispatch()


class AttentionDecode:
    def __getitem__(self, grid):
        if isinstance(grid, int):
            grid = (grid,)
        valid_grid = isinstance(grid, tuple) and len(grid) in (1, 2)
        if valid_grid:
            valid_grid = all(
                isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in grid
            )
        if not valid_grid:
            raise ValueError("decode attention requires a static (heads,) or (batch, heads) grid")
        batch, query_heads = (1, grid[0]) if len(grid) == 1 else grid
        return _AttentionDecodeLauncher(batch, query_heads)


def _validate_attention_arguments(
    query,
    key,
    value,
    output,
    batch,
    query_heads,
    key_value_heads,
    tokens,
    dimension,
):
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("decode attention token count must be a positive integer")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        or dimension % 32
    ):
        raise ValueError("decode attention head dimension must be a positive multiple of 32")
    if (
        not isinstance(key_value_heads, int)
        or isinstance(key_value_heads, bool)
        or key_value_heads <= 0
        or query_heads % key_value_heads
    ):
        raise ValueError("decode attention KV_HEADS must be a positive divisor of query heads")

    buffers = {
        "query": (query, batch * query_heads * dimension),
        "key": (key, batch * key_value_heads * tokens * dimension),
        "value": (value, batch * key_value_heads * tokens * dimension),
        "output": (output, batch * query_heads * dimension),
    }
    for name, (buffer, required_elements) in buffers.items():
        dtype = getattr(buffer, "dtype", None)
        if dtype is None or str(dtype) != "float32":
            raise TypeError(f"decode attention {name} must use float32 storage")
        shape = getattr(buffer, "shape", None)
        if shape is None or prod(shape) < required_elements:
            raise ValueError(
                f"decode attention {name} requires at least {required_elements} elements"
            )


attention_decode = AttentionDecode()

__all__ = ["AttentionDecode", "AttentionDecodeConfig", "attention_decode"]
