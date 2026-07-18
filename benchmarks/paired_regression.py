import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_RUNNER = """
import importlib.util
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
rounds = int(sys.argv[3])
spec = importlib.util.spec_from_file_location(
    "_metile_regression", root / "benchmarks" / "regression.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load regression benchmark from {root}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def bench_end_to_end(dispatch):
    return module.bench(
        dispatch,
        warmup_ms=module._WARMUP_MS,
        rep_ms=module._REP_MS,
        gpu=False,
    )

module._bench = bench_end_to_end
groups = ("bench_gemm", "bench_softmax", "bench_layernorm", "bench_fft")
all_times = {}
for round_index in range(rounds):
    if round_index:
        time.sleep(module._COOLDOWN)
    for function_name in groups:
        time.sleep(module._COOLDOWN)
        try:
            group_results = getattr(module, function_name)()
        except Exception as error:
            print(
                f"unavailable {function_name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            continue
        for name, value in group_results.items():
            all_times.setdefault(name, []).append(value)
results = {name: module._geomean(values) for name, values in all_times.items()}
with output.open("w") as handle:
    json.dump(results, handle)
"""


def aggregate_results(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one benchmark sample is required")
    names = set.intersection(*(set(sample) for sample in samples))
    if not names:
        raise ValueError("benchmark samples have no kernels in common")
    return {
        name: math.exp(sum(math.log(sample[name]) for sample in samples) / len(samples))
        for name in names
    }


def regression_rows(
    current: dict[str, float], baseline: dict[str, float], threshold: float
) -> tuple[list[tuple[str, float, float, float]], list[str]]:
    missing = set(current) ^ set(baseline)
    if missing:
        raise ValueError(f"benchmark sets differ: {', '.join(sorted(missing))}")
    rows = []
    regressions = []
    for name in sorted(current):
        change = current[name] / baseline[name] - 1.0
        rows.append((name, baseline[name], current[name], change))
        if change > threshold:
            regressions.append(name)
    return rows, regressions


def _run_sample(root: Path, output: Path, rounds: int, cache_dir: Path) -> dict[str, float]:
    environment = os.environ.copy()
    environment["METILE_CACHE_DIR"] = str(cache_dir)
    subprocess.run(
        [sys.executable, "-c", _RUNNER, str(root), str(output), str(rounds)],
        check=True,
        env=environment,
    )
    with output.open() as handle:
        return {name: float(value) for name, value in json.load(handle).items()}


def _format_report(
    rows: list[tuple[str, float, float, float]],
    regressions: list[str],
    threshold: float,
    baseline_only: tuple[str, ...] = (),
    current_only: tuple[str, ...] = (),
) -> str:
    lines = [
        f"{'kernel':<30} {'baseline':>12} {'current':>12} {'change':>8}  status",
        "-" * 75,
    ]
    for name, baseline, current, change in rows:
        status = "REGRESSION" if name in regressions else "OK"
        lines.append(
            f"{name:<30} {baseline * 1e6:>10.1f}us "
            f"{current * 1e6:>10.1f}us {change:>+7.1%}  {status}"
        )
    if regressions:
        lines.append(f"\n{len(regressions)} regression(s) detected (>{threshold:.0%} slower).")
    else:
        lines.append("\nNo regressions detected by the paired comparison.")
    if current_only:
        lines.append("Base unavailable: " + ", ".join(current_only))
    if baseline_only:
        lines.append("PR unavailable: " + ", ".join(baseline_only))
    return "\n".join(lines)


def _write_step_summary(
    rows: list[tuple[str, float, float, float]],
    regressions: list[str],
    baseline_only: tuple[str, ...],
    current_only: tuple[str, ...],
):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    regression_set = set(regressions)
    lines = [
        "## Paired launch-to-completion comparison",
        "",
        "| Kernel | Base | PR | Change | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name, baseline, current, change in rows:
        status = "regression" if name in regression_set else "ok"
        lines.append(
            f"| `{name}` | {baseline * 1e6:.1f} us | {current * 1e6:.1f} us | "
            f"{change:+.1%} | {status} |"
        )
    if current_only:
        lines.extend(("", "Base unavailable: " + ", ".join(f"`{name}`" for name in current_only)))
    if baseline_only:
        lines.extend(("", "PR unavailable: " + ", ".join(f"`{name}`" for name in baseline_only)))
    with open(summary_path, "a") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paired base/PR performance regression check")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--rounds-per-sample", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    roots = {
        "baseline": args.baseline_root.resolve(),
        "current": args.current_root.resolve(),
    }
    for label, root in roots.items():
        if not (root / "benchmarks" / "regression.py").is_file():
            parser.error(f"{label} root has no benchmarks/regression.py: {root}")

    samples: dict[str, list[dict[str, float]]] = {"baseline": [], "current": []}
    order = ("baseline", "current", "current", "baseline")
    with tempfile.TemporaryDirectory(prefix="metile-paired-benchmark-") as temporary:
        temporary_path = Path(temporary)
        for index, label in enumerate(order):
            if index:
                time.sleep(args.cooldown)
            print(f"\n== {label} sample {len(samples[label]) + 1}/2 ==", flush=True)
            result = _run_sample(
                roots[label],
                temporary_path / f"{index}-{label}.json",
                args.rounds_per_sample,
                temporary_path / f"cache-{label}",
            )
            samples[label].append(result)

    baseline = aggregate_results(samples["baseline"])
    current = aggregate_results(samples["current"])
    common = set(baseline) & set(current)
    if not common:
        raise RuntimeError("base and PR have no comparable benchmark results")
    baseline_only = tuple(sorted(set(baseline) - common))
    current_only = tuple(sorted(set(current) - common))
    rows, regressions = regression_rows(
        {name: current[name] for name in common},
        {name: baseline[name] for name in common},
        args.threshold,
    )
    print("\n" + _format_report(rows, regressions, args.threshold, baseline_only, current_only))
    _write_step_summary(rows, regressions, baseline_only, current_only)
    raise SystemExit(1 if regressions else 0)


if __name__ == "__main__":
    main()
