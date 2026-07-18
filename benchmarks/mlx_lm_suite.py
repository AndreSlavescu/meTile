"""Run the MLX-LM backend benchmark across a reproducible model suite."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODELS = (
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model repository or local path; repeat for multiple models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m5-mlx-lm-models.json"),
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--disable-attention", action="store_true")
    parser.add_argument("--disable-rmsnorm", action="store_true")
    parser.add_argument("--disable-graph-fusion", action="store_true")
    parser.add_argument("--disable-quantized-mlp", action="store_true")
    parser.add_argument("--disable-affine-prefill", action="store_true")
    parser.add_argument("--disable-model-autotune", action="store_true")
    parser.add_argument("--plan-decode-steps", type=int, default=8)
    parser.add_argument("--plan-trials", type=int, default=7)
    parser.add_argument("--confirmation-trials", type=int, default=5)
    return parser.parse_args()


def _backend_command(arguments, model, output):
    command = [
        sys.executable,
        str(Path(__file__).with_name("mlx_lm_backend.py")),
        "--model",
        model,
        "--prompt-tokens",
        str(arguments.prompt_tokens),
        "--generation-tokens",
        str(arguments.generation_tokens),
        "--trials",
        str(arguments.trials),
        "--prefill-step-size",
        str(arguments.prefill_step_size),
        "--delay",
        str(arguments.delay),
        "--seed",
        str(arguments.seed),
        "--plan-decode-steps",
        str(arguments.plan_decode_steps),
        "--plan-trials",
        str(arguments.plan_trials),
        "--confirmation-trials",
        str(arguments.confirmation_trials),
        "--output-json",
        str(output),
    ]
    for name in (
        "skip_verify",
        "disable_attention",
        "disable_rmsnorm",
        "disable_graph_fusion",
        "disable_quantized_mlp",
        "disable_affine_prefill",
        "disable_model_autotune",
    ):
        if getattr(arguments, name):
            command.append("--" + name.replace("_", "-"))
    return command


def main():
    arguments = _arguments()
    models = tuple(arguments.models or DEFAULT_MODELS)
    environment = os.environ.copy()
    if arguments.offline:
        environment["HF_HUB_OFFLINE"] = "1"

    results = []
    with tempfile.TemporaryDirectory(prefix="metile-mlx-lm-suite-") as directory:
        directory = Path(directory)
        for index, model in enumerate(models):
            output = directory / f"model-{index}.json"
            print(f"\n=== {model} ===", flush=True)
            subprocess.run(
                _backend_command(arguments, model, output),
                check=True,
                env=environment,
            )
            results.append(json.loads(output.read_text()))

    suite = {
        "schema_version": 2,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {len(results)} model results to {arguments.output}")


if __name__ == "__main__":
    main()
