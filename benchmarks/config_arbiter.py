"""Settle "the tuner is leaving performance on the table" claims, which are usually the harness.

The tuner keeps native whenever its two measurement passes disagree, and that conservatism looks like
timidity from the outside: pick a shape, measure a generated config in isolation, find it faster than the
native kernel the tuner chose, and conclude the selection is broken. This tool exists because that
inference is wrong often enough to be the default suspicion.

At rows 64 on Qwen2.5-1.5B's gate shape, a generated config measured 1.15x better than the tuner's choice
by median, and every generated config beat native on median while being *tighter* than native. It still
was not a win:

    batch    ratio   win rate
        1   1.049x        80%
        2   1.041x        78%
        4   1.007x        54%
        8   0.968x        20%
       16   0.957x         2%
       32   1.001x        56%

The answer depends on how many dispatches share an eval. There is no stable ordering to find, and the
honest response is the one the tuner already takes.

Two things this does that a single median comparison does not:

  interleave   the two arms alternate within every round, and the order flips between rounds. Measuring
               all of one arm and then all of the other lets drift between the blocks masquerade as a
               difference, which is exactly how the 1.15x claim arose.
  sweep batch  the ratio is reported across batch sizes rather than at one. A claim that survives only at
               a particular batch is a property of the measurement.

The baseline is whatever the tuner chose, not the native kernel. Those differ whenever the tuner picked a
generated config, and comparing candidates against native then answers a question nobody asked: at rows 16
every candidate beats native by 1.34x to 2.10x, and reporting that as "the tuner is leaving something
behind" is a false positive, because the config it chose is one of them.

A verdict of "no stable win" is the useful output, not a failure to find one. The tool is checked against a
case where a win does exist, so that verdict means something.

usage:
    python benchmarks/config_arbiter.py --rows 64
    python benchmarks/config_arbiter.py --rows 16 --hidden 2048 --intermediate 8192
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

GROUP, BITS = 64, 4
BATCHES = (1, 2, 4, 8, 16, 32)
# A win has to be this consistent to count. At 50% the two are indistinguishable; below it the other arm
# is winning.
WIN_RATE_FLOOR = 0.75
MARGIN = 1.03


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--intermediate", type=int, default=8960)
    parser.add_argument("--rounds", type=int, default=41)
    parser.add_argument(
        "--against-native",
        action="store_true",
        help="baseline against MLX rather than the tuner's choice; the tool's own sensitivity check, "
        "since a detector that never fires is indistinguishable from a broken one",
    )
    return parser.parse_args()


def _paired(mx, arms, batch, rounds):
    """Median ratio of the second arm's time to the first's, and how often the first won."""
    names = list(arms)
    for _ in range(4):
        for name in names:
            mx.eval([arms[name]() for _ in range(batch)])
    mx.synchronize()

    ratios, wins = [], 0
    for index in range(rounds):
        ordered = names if index % 2 == 0 else names[::-1]
        elapsed = {}
        for name in ordered:
            started = time.perf_counter_ns()
            mx.eval([arms[name]() for _ in range(batch)])
            elapsed[name] = (time.perf_counter_ns() - started) / batch
        ratios.append(elapsed[names[1]] / elapsed[names[0]])
        wins += elapsed[names[0]] < elapsed[names[1]]
    return statistics.median(ratios), wins / rounds


def main():
    arguments = _arguments()
    try:
        import mlx.core as mx
    except ImportError:
        print("mlx is required")
        return 1

    from metile.backends import mlx_affine
    from metile.backends.mlx_affine import MLXAffineWeight, mlx_affine_matmul

    hidden, intermediate, rows = arguments.hidden, arguments.intermediate, arguments.rows
    mx.random.seed(0)
    packed, scales, biases = mx.quantize(
        mx.random.normal((intermediate, hidden)).astype(mx.float16),
        group_size=GROUP,
        bits=BITS,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    weight = MLXAffineWeight.from_mlx(packed, scales, biases, group_size=GROUP, bits=BITS)
    values = mx.random.normal((rows, hidden)).astype(mx.float16)
    mx.eval(weight.packed, values)

    key = (rows, hidden, intermediate, str(values.dtype), GROUP, BITS)
    mlx_affine._schedule_cache.pop(key, None)
    mx.eval(mlx_affine_matmul(values, weight))
    chosen = mlx_affine._schedule_cache[key]
    chosen_label = (
        chosen.algorithm if chosen.algorithm == "mlx" else f"bn={chosen.block_n} {chosen.schedule}"
    )

    print(f"shape K={hidden} N={intermediate}, rows={rows}, int4 group {GROUP}")
    print(f"the tuner chose: {chosen_label}")

    def native():
        return mx.quantized_matmul(
            values,
            packed,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            mode="affine",
        )

    def generated(config):
        mlx_affine._schedule_cache[key] = config
        return lambda: mlx_affine_matmul(values, weight, autotune=False)

    # The baseline is what the tuner actually dispatches. Recompiled per call through the same path as the
    # candidates so neither arm gets a dispatch-overhead advantage the other does not.
    against_native = arguments.against_native or chosen.algorithm == "mlx"
    baseline = native if against_native else generated(chosen)
    baseline_label = "mlx" if against_native else chosen_label

    print(f"baseline: {baseline_label}\n")

    candidates = [
        config
        for config in mlx_affine._candidate_configs(rows, intermediate)
        if config.algorithm == "metile" and (against_native or vars(config) != vars(chosen))
    ]

    verdicts = []
    for config in candidates:
        candidate = generated(config)
        try:
            mx.eval(candidate())
        except Exception as error:
            print(f"bn={config.block_n} {config.schedule}: {type(error).__name__}")
            continue
        # Both arms set the cache entry they need before dispatching, so they cannot be prepared once
        # and reused; the lambda re-points the entry on every call.
        arms = {
            "candidate": lambda candidate=candidate, config=config: (
                mlx_affine._schedule_cache.__setitem__(key, config),
                candidate(),
            )[1],
            "baseline": lambda: (
                (
                    mlx_affine._schedule_cache.__setitem__(key, chosen),
                    baseline(),
                )[1]
                if not against_native
                else native()
            ),
        }
        cells = []
        stable = True
        for batch in BATCHES:
            ratio, rate = _paired(mx, arms, batch, arguments.rounds)
            cells.append(f"{ratio:.3f}x/{rate:.0%}")
            if ratio < MARGIN or rate < WIN_RATE_FLOOR:
                stable = False
        verdicts.append((config, stable))
        label = f"bn={config.block_n} {config.schedule}"
        print(f"{label:<24}" + "  ".join(f"{cell:>12}" for cell in cells))
    mlx_affine._schedule_cache[key] = chosen

    print("\nratio/win-rate per batch size " + str(BATCHES))
    winners = [config for config, stable in verdicts if stable]
    print(
        f"\nconfigs beating {baseline_label} by {MARGIN:.2f}x at a {WIN_RATE_FLOOR:.0%} win rate "
        f"at every batch size: {len(winners)} of {len(verdicts)}"
    )
    if winners:
        for config in winners:
            print(f"  bn={config.block_n} {config.schedule}")
        if arguments.against_native:
            print(
                "Baselined against MLX rather than the tuner's choice, so this is the sensitivity\n"
                "check rather than a finding: it shows the tool fires when a win exists. Whether the\n"
                "tuner mis-picked is the default mode's question."
            )
        else:
            print("The tuner is leaving something measurable behind; this is worth investigating.")
    else:
        print(f"No stable win over {baseline_label}. A claim that the tuner mis-picked here is a")
        print("claim about the harness, not about the selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
