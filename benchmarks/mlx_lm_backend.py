"""Benchmark meTile as an opt-in MLX-LM decode backend on a real model."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from metile.backends.mlx import (
    mlx_add_rms_norm_dispatches,
    mlx_attention_dispatches,
    mlx_rms_norm_dispatches,
)
from metile.backends.mlx_quantized import mlx_affine_swiglu_dispatches
from metile.integrations.mlx_lm import (
    MLXLMPlan,
    apply_metile_to_mlx_lm,
    autotune_metile_for_mlx_lm,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="MLX-LM model path or Hugging Face repository",
    )
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--disable-attention", action="store_true")
    parser.add_argument("--disable-rmsnorm", action="store_true")
    parser.add_argument("--disable-graph-fusion", action="store_true")
    parser.add_argument("--disable-quantized-mlp", action="store_true")
    parser.add_argument("--disable-model-autotune", action="store_true")
    parser.add_argument("--plan-decode-steps", type=int, default=8)
    parser.add_argument("--plan-trials", type=int, default=5)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _generate(model, tokenizer, prompt, arguments, patched, plan):
    patch = (
        apply_metile_to_mlx_lm(
            model=model,
            attention=not arguments.disable_attention,
            rms_norm=not arguments.disable_rmsnorm,
            graph_fusion=not arguments.disable_graph_fusion,
            quantized_mlp=not arguments.disable_quantized_mlp,
            plan=plan,
        )
        if patched
        else None
    )
    start = time.perf_counter()
    first_token_elapsed = None
    response = None
    try:
        for next_response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=arguments.generation_tokens,
            prefill_step_size=arguments.prefill_step_size,
        ):
            response = next_response
            if first_token_elapsed is None:
                first_token_elapsed = time.perf_counter() - start
    finally:
        if patch is not None:
            patch.restore()
    if response is None:
        raise RuntimeError("MLX-LM generation returned no timing response")
    return response, time.perf_counter() - start, first_token_elapsed


def _verify_model(model, prompt, arguments, plan):
    tokens = mx.array(prompt[: min(len(prompt), 128)])[None]
    baseline_cache = make_prompt_cache(model)
    patched_cache = make_prompt_cache(model)
    baseline_prefix = model(tokens[:, :-1], cache=baseline_cache)
    mx.eval(baseline_prefix)
    baseline = model(tokens[:, -1:], cache=baseline_cache)
    mx.eval(baseline)

    with apply_metile_to_mlx_lm(
        model=model,
        attention=not arguments.disable_attention,
        rms_norm=not arguments.disable_rmsnorm,
        graph_fusion=not arguments.disable_graph_fusion,
        quantized_mlp=not arguments.disable_quantized_mlp,
        plan=plan,
    ):
        patched_prefix = model(tokens[:, :-1], cache=patched_cache)
        mx.eval(patched_prefix)
        patched = model(tokens[:, -1:], cache=patched_cache)
        mx.eval(patched)

    baseline_token = int(mx.argmax(baseline, axis=-1).item())
    patched_token = int(mx.argmax(patched, axis=-1).item())
    max_error = float(mx.max(mx.abs(baseline - patched)).item())
    if baseline_token != patched_token:
        raise RuntimeError(
            f"patched next token {patched_token} differs from baseline {baseline_token}"
        )
    print(f"Verified next token {baseline_token}; max logit difference={max_error:.6f}")
    return {"next_token": baseline_token, "max_logit_error": max_error}


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _hardware_metadata():
    metadata = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if sys.platform != "darwin":
        return metadata
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
        )
        hardware = json.loads(completed.stdout)["SPHardwareDataType"][0]
    except (KeyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return metadata
    metadata.update(
        {
            "chip": hardware.get("chip_type", "unknown"),
            "memory": hardware.get("physical_memory", "unknown"),
            "model_name": hardware.get("machine_model", "unknown"),
        }
    )
    return metadata


def _git_revision():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _selected_dispatches():
    return {
        "attention": [dict(dispatch) for dispatch in mlx_attention_dispatches()],
        "rms_norm": [dict(dispatch) for dispatch in mlx_rms_norm_dispatches()],
        "add_rms_norm": [dict(dispatch) for dispatch in mlx_add_rms_norm_dispatches()],
        "affine_swiglu": [dict(dispatch) for dispatch in mlx_affine_swiglu_dispatches()],
    }


def _model_metadata(config):
    text_config = config.get("text_config", config)
    keys = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    return {key: text_config[key] for key in keys if key in text_config}


def _write_json_result(
    path,
    arguments,
    config,
    verification,
    results,
    medians,
    dispatches,
    plan,
):
    payload = {
        "schema_version": 3,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "revision": _git_revision(),
        "model": arguments.model,
        "model_config": _model_metadata(config),
        "hardware": _hardware_metadata(),
        "software": {
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
        },
        "workload": {
            "prompt_tokens": arguments.prompt_tokens,
            "generation_tokens": arguments.generation_tokens,
            "trials": arguments.trials,
            "prefill_step_size": arguments.prefill_step_size,
            "delay_seconds": arguments.delay,
            "plan_decode_steps": arguments.plan_decode_steps,
            "plan_trials": arguments.plan_trials,
            "seed": arguments.seed,
        },
        "features": {
            "attention": not arguments.disable_attention,
            "rms_norm": not arguments.disable_rmsnorm,
            "graph_fusion": not arguments.disable_graph_fusion,
            "quantized_mlp": not arguments.disable_quantized_mlp,
            "model_autotune": not arguments.disable_model_autotune,
        },
        "selected_plan": plan.as_dict(),
        "comparison_mode": "alternating" if plan.feature_count else "shared_native_fallback",
        "verification": verification,
        "samples": {
            name: [
                {
                    "decode_tokens_per_second": decode,
                    "prefill_tokens_per_second": prefill,
                    "elapsed_seconds": elapsed,
                    "time_to_first_token_seconds": ttft,
                }
                for decode, prefill, elapsed, ttft in samples
            ]
            for name, samples in results.items()
        },
        "medians": medians,
        "selected_dispatches": dispatches,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote benchmark result to {path}")


def main():
    arguments = _arguments()
    model, tokenizer, config = load(arguments.model, return_config=True)
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    mx.random.seed(arguments.seed)
    prompt = mx.random.randint(0, vocab_size, (arguments.prompt_tokens,)).tolist()

    requested_plan = MLXLMPlan(
        attention=not arguments.disable_attention,
        rms_norm=not arguments.disable_rmsnorm,
        graph_fusion=not arguments.disable_graph_fusion,
        quantized_mlp=not arguments.disable_quantized_mlp,
    )
    if arguments.disable_model_autotune:
        plan = requested_plan
    else:
        print("Autotuning the MLX-LM feature plan...")
        plan = autotune_metile_for_mlx_lm(
            model,
            mx.array(prompt)[None],
            **requested_plan.as_dict(),
            decode_steps=arguments.plan_decode_steps,
            trials=arguments.plan_trials,
        )
    enabled = ", ".join(name for name, active in plan.as_dict().items() if active) or "native MLX"
    print(f"Selected model plan: {enabled}")

    verification = None
    if not arguments.skip_verify:
        verification = _verify_model(model, prompt, arguments, plan)

    print("Warming MLX baseline...")
    _generate(model, tokenizer, prompt, arguments, patched=False, plan=plan)
    if plan.feature_count:
        print("Compiling and autotuning meTile MLX kernels...")
        _generate(model, tokenizer, prompt, arguments, patched=True, plan=plan)
    else:
        print("Native fallback selected; sharing each measurement across both labels.")

    results = {"MLX": [], "MLX + meTile": []}
    for trial in range(arguments.trials):
        if not plan.feature_count:
            if arguments.delay:
                time.sleep(arguments.delay)
            response, elapsed, ttft = _generate(
                model,
                tokenizer,
                prompt,
                arguments,
                patched=False,
                plan=plan,
            )
            sample = (
                float(response.generation_tps),
                float(response.prompt_tps),
                float(elapsed),
                float(ttft),
            )
            results["MLX"].append(sample)
            results["MLX + meTile"].append(sample)
            print(
                f"Trial {trial + 1} shared native: "
                f"decode={response.generation_tps:.2f} tok/s, "
                f"prefill={response.prompt_tps:.2f} tok/s, total={elapsed:.3f}s"
                f", TTFT={ttft * 1e3:.1f}ms"
            )
            continue
        order = (False, True) if trial % 2 == 0 else (True, False)
        for patched in order:
            if arguments.delay:
                time.sleep(arguments.delay)
            response, elapsed, ttft = _generate(
                model,
                tokenizer,
                prompt,
                arguments,
                patched,
                plan,
            )
            name = "MLX + meTile" if patched else "MLX"
            results[name].append(
                (
                    float(response.generation_tps),
                    float(response.prompt_tps),
                    float(elapsed),
                    float(ttft),
                )
            )
            print(
                f"Trial {trial + 1} {name:12s}: "
                f"decode={response.generation_tps:.2f} tok/s, "
                f"prefill={response.prompt_tps:.2f} tok/s, total={elapsed:.3f}s"
                f", TTFT={ttft * 1e3:.1f}ms"
            )

    baseline = statistics.median(sample[0] for sample in results["MLX"])
    metile_decode = statistics.median(sample[0] for sample in results["MLX + meTile"])
    baseline_total = statistics.median(sample[2] for sample in results["MLX"])
    metile_total = statistics.median(sample[2] for sample in results["MLX + meTile"])
    baseline_ttft = statistics.median(sample[3] for sample in results["MLX"])
    metile_ttft = statistics.median(sample[3] for sample in results["MLX + meTile"])
    medians = {
        "mlx_decode_tokens_per_second": baseline,
        "metile_decode_tokens_per_second": metile_decode,
        "decode_speedup": metile_decode / baseline,
        "mlx_elapsed_seconds": baseline_total,
        "metile_elapsed_seconds": metile_total,
        "end_to_end_speedup": baseline_total / metile_total,
        "mlx_time_to_first_token_seconds": baseline_ttft,
        "metile_time_to_first_token_seconds": metile_ttft,
        "ttft_speedup": baseline_ttft / metile_ttft,
    }
    dispatches = _selected_dispatches()
    print("\nMedian results")
    print(f"MLX decode:          {baseline:.2f} tok/s")
    print(f"MLX + meTile decode: {metile_decode:.2f} tok/s ({metile_decode / baseline:.3f}x)")
    print(f"End-to-end speedup:  {baseline_total / metile_total:.3f}x")
    print(
        f"TTFT:                {baseline_ttft * 1e3:.1f}ms -> "
        f"{metile_ttft * 1e3:.1f}ms ({baseline_ttft / metile_ttft:.3f}x)"
    )
    print("\nSelected attention schedules")
    for dispatch in dispatches["attention"]:
        print(
            f"tokens<={dispatch['token_bucket']}: {dispatch['algorithm']} block={dispatch['block']}"
        )
    print("Selected RMSNorm schedules")
    for dispatch in dispatches["rms_norm"]:
        print(
            f"rows<={dispatch['row_bucket']} hidden={dispatch['hidden']}: "
            f"{dispatch['algorithm']} block={dispatch['block']}"
        )
    print("Selected graph-fused residual/RMSNorm schedules")
    for dispatch in dispatches["add_rms_norm"]:
        print(
            f"rows<={dispatch['row_bucket']} hidden={dispatch['hidden']}: "
            f"{dispatch['algorithm']} block={dispatch['block']}"
        )
    print("Selected quantized SwiGLU schedules")
    for dispatch in dispatches["affine_swiglu"]:
        print(
            f"{dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} {dispatch['implementation']} block={dispatch['block']} "
            f"outputs/simdgroup={dispatch['outputs_per_simdgroup']} "
            f"decode={dispatch['decode_dtype']}"
        )
    if arguments.output_json is not None:
        _write_json_result(
            arguments.output_json,
            arguments,
            config,
            verification,
            results,
            medians,
            dispatches,
            plan,
        )


if __name__ == "__main__":
    main()
