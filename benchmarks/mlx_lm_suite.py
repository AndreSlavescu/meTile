"""Run the MLX-LM backend benchmark across a reproducible model suite."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_4BIT_MODELS = (
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
)
DEFAULT_BF16_MODELS = (
    "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "mlx-community/Qwen2.5-3B-Instruct-bf16",
    "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "mlx-community/Qwen2.5-7B-Instruct-bf16",
    "mlx-community/Qwen3-8B-bf16",
)
MODEL_SUITES = {
    "4bit": DEFAULT_4BIT_MODELS,
    "bf16": DEFAULT_BF16_MODELS,
}
DEFAULT_OUTPUTS = {
    "4bit": Path("benchmarks/results/m5-mlx-lm-models.json"),
    "bf16": Path("benchmarks/results/m5-mlx-lm-bf16-models.json"),
}


def _compressed_down_group_size(value):
    if value == "auto":
        return value
    try:
        group_size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("group size must be auto, 32, 64, or 128") from error
    if group_size not in {32, 64, 128}:
        raise argparse.ArgumentTypeError("group size must be auto, 32, 64, or 128")
    return group_size


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(MODEL_SUITES), default="4bit")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model repository or local path; repeat for multiple models",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--model-delay",
        type=float,
        default=10.0,
        help="Cooldown between model subprocesses to reduce thermal order effects",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--disable-attention", action="store_true")
    parser.add_argument("--disable-rmsnorm", action="store_true")
    parser.add_argument("--disable-graph-fusion", action="store_true")
    parser.add_argument("--disable-quantized-mlp", action="store_true")
    parser.add_argument("--disable-affine-prefill", action="store_true")
    parser.add_argument("--disable-dense-mlp", action="store_true")
    parser.add_argument(
        "--compressed-down-format",
        choices=("none", "affine8", "mxfp8"),
        default="none",
    )
    parser.add_argument(
        "--compressed-down-group-size",
        type=_compressed_down_group_size,
        default="auto",
    )
    parser.add_argument("--allow-approximate-compressed-down", action="store_true")
    parser.add_argument("--compressed-gate-up", action="store_true")
    parser.add_argument(
        "--compressed-gate-up-group-size",
        type=_compressed_down_group_size,
        default="auto",
    )
    parser.add_argument("--compressed-vocab", action="store_true")
    parser.add_argument(
        "--compressed-vocab-group-size",
        type=_compressed_down_group_size,
        default="auto",
    )
    parser.add_argument("--compressed-attention", action="store_true")
    parser.add_argument(
        "--compressed-attention-group-size",
        type=_compressed_down_group_size,
        default="auto",
    )
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
    disabled = {
        name
        for name in (
            "skip_verify",
            "disable_attention",
            "disable_rmsnorm",
            "disable_graph_fusion",
            "disable_quantized_mlp",
            "disable_affine_prefill",
            "disable_dense_mlp",
            "disable_model_autotune",
        )
        if getattr(arguments, name)
    }
    if arguments.suite == "bf16":
        disabled.update(("disable_quantized_mlp", "disable_affine_prefill"))
    else:
        disabled.add("disable_dense_mlp")
    for name in (
        "skip_verify",
        "disable_attention",
        "disable_rmsnorm",
        "disable_graph_fusion",
        "disable_quantized_mlp",
        "disable_affine_prefill",
        "disable_dense_mlp",
        "disable_model_autotune",
    ):
        if name in disabled:
            command.append("--" + name.replace("_", "-"))
    compressed_down_format = getattr(arguments, "compressed_down_format", "none")
    if compressed_down_format != "none":
        command.extend(("--compressed-down-format", compressed_down_format))
        command.extend(
            (
                "--compressed-down-group-size",
                str(getattr(arguments, "compressed_down_group_size", "auto")),
            )
        )
    if getattr(arguments, "allow_approximate_compressed_down", False):
        command.append("--allow-approximate-compressed-down")
    if getattr(arguments, "compressed_gate_up", False):
        command.append("--compressed-gate-up")
        command.extend(
            (
                "--compressed-gate-up-group-size",
                str(getattr(arguments, "compressed_gate_up_group_size", "auto")),
            )
        )
    if getattr(arguments, "compressed_vocab", False):
        command.append("--compressed-vocab")
        command.extend(
            (
                "--compressed-vocab-group-size",
                str(getattr(arguments, "compressed_vocab_group_size", "auto")),
            )
        )
    if getattr(arguments, "compressed_attention", False):
        command.append("--compressed-attention")
        command.extend(
            (
                "--compressed-attention-group-size",
                str(getattr(arguments, "compressed_attention_group_size", "auto")),
            )
        )
    return command


def main():
    arguments = _arguments()
    if arguments.model_delay < 0:
        raise ValueError("model delay must be non-negative")
    models = tuple(arguments.models or MODEL_SUITES[arguments.suite])
    suite_output = arguments.output or DEFAULT_OUTPUTS[arguments.suite]
    environment = os.environ.copy()
    if arguments.offline:
        environment["HF_HUB_OFFLINE"] = "1"

    results = []
    with tempfile.TemporaryDirectory(prefix="metile-mlx-lm-suite-") as directory:
        directory = Path(directory)
        for index, model in enumerate(models):
            if index and arguments.model_delay:
                print(
                    f"\nCooling down for {arguments.model_delay:.1f}s before the next model...",
                    flush=True,
                )
                time.sleep(arguments.model_delay)
            model_output = directory / f"model-{index}.json"
            print(f"\n=== {model} ===", flush=True)
            subprocess.run(
                _backend_command(arguments, model, model_output),
                check=True,
                env=environment,
            )
            results.append(json.loads(model_output.read_text()))

    suite = {
        "schema_version": 5,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "suite": arguments.suite,
        "model_delay_seconds": arguments.model_delay,
        "models": results,
    }
    suite_output.parent.mkdir(parents=True, exist_ok=True)
    suite_output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {len(results)} model results to {suite_output}")


if __name__ == "__main__":
    main()
