"""Benchmark proof-discovered exact attention against native MLX."""

import argparse
import os
import statistics
import time

import mlx.core as mx
import numpy as np

from metile.backends.mlx_attention import (
    _compile_flash_attention,
    _flash_schedule_cache,
    _native_attention,
    mlx_flash_attention,
    mlx_flash_attention_dispatches,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--seed", type=int, default=79)
    return parser.parse_args()


def _interleaved_microseconds(candidates, trials):
    for _, dispatch in candidates:
        for _ in range(8):
            mx.eval(dispatch())
    samples = {name: [] for name, _ in candidates}
    for round_index in range(trials):
        ordered = (
            candidates[round_index % len(candidates) :]
            + candidates[: round_index % len(candidates)]
        )
        if round_index & 1:
            ordered.reverse()
        for name, dispatch in ordered:
            start = time.perf_counter_ns()
            mx.eval(dispatch())
            samples[name].append((time.perf_counter_ns() - start) / 1e3)
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    arguments = _arguments()
    os.environ["METILE_DISABLE_DISK_CACHE"] = "1"
    _flash_schedule_cache.clear()
    random = np.random.default_rng(arguments.seed)
    shapes = (
        (1, 16, 128),
        (1, 64, 128),
        (1, 128, 64),
        (2, 64, 128),
        (8, 128, 64),
    )
    print("heads tokens dimension | MLX us | best meTile us | relative | selected")
    for heads, tokens, dimension in shapes:
        shape = (1, heads, tokens, dimension)
        scale = dimension**-0.5
        query = mx.array(random.normal(size=shape).astype(np.float16))
        key = mx.array(random.normal(size=shape).astype(np.float16))
        value = mx.array(random.normal(size=shape).astype(np.float16))

        def native(query=query, key=key, value=value, scale=scale):
            return _native_attention(query, key, value, scale, True)

        candidates = [("mlx", native)]
        reference = native()
        mx.eval(reference)
        for block in (32, 64, 128, 256):
            kernel = _compile_flash_attention(
                heads,
                heads,
                dimension,
                query.dtype,
                scale,
                True,
                block,
            )

            def dispatch(kernel=kernel, query=query, key=key, value=value):
                return kernel(query, key, value)

            result = dispatch()
            mx.eval(result)
            if bool(mx.allclose(result, reference, rtol=3e-3, atol=3e-3).item()):
                candidates.append((f"b{block}", dispatch))
        latencies = _interleaved_microseconds(candidates, arguments.trials)
        native_latency = latencies["mlx"]
        best_name, generated_latency = min(
            ((name, latency) for name, latency in latencies.items() if name != "mlx"),
            key=lambda candidate: candidate[1],
        )
        result = mlx_flash_attention(
            query,
            key,
            value,
            scale=scale,
            causal=True,
        )
        mx.eval(result)
        selected = next(
            dispatch
            for dispatch in mlx_flash_attention_dispatches()
            if dispatch["query_shape"] == shape and dispatch["key_shape"] == shape
        )
        print(
            f"{heads:5d} {tokens:6d} {dimension:9d} | "
            f"{native_latency:6.1f} | {generated_latency:14.1f} ({best_name}) | "
            f"{native_latency / generated_latency:7.3f}x | "
            f"{selected['algorithm']} b{selected['block']}"
        )


if __name__ == "__main__":
    main()
