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

_DECODE_CONFIRMATION_FLOOR = 0.995
_PREFILL_ONLY_DECODE_CONFIRMATION_FLOOR = 0.99


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
    parser.add_argument("--disable-affine-prefill", action="store_true")
    parser.add_argument("--disable-model-autotune", action="store_true")
    parser.add_argument("--plan-decode-steps", type=int, default=8)
    parser.add_argument("--plan-trials", type=int, default=7)
    parser.add_argument("--confirmation-trials", type=int, default=3)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _generate(model, tokenizer, prompt, arguments, patched, plan, affine_prefill):
    from mlx_lm import stream_generate

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    patch = (
        apply_metile_to_mlx_lm(
            model=model,
            attention=not arguments.disable_attention,
            rms_norm=not arguments.disable_rmsnorm,
            graph_fusion=not arguments.disable_graph_fusion,
            quantized_mlp=not arguments.disable_quantized_mlp,
            affine_prefill=affine_prefill,
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


def _verify_model(model, prompt, arguments, plan, affine_prefill):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from metile.integrations.mlx_lm import (
        _fidelity_compatible,
        _logit_fidelity,
        apply_metile_to_mlx_lm,
    )

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
        affine_prefill=affine_prefill,
        plan=plan,
    ):
        patched_prefix = model(tokens[:, :-1], cache=patched_cache)
        mx.eval(patched_prefix)
        patched = model(tokens[:, -1:], cache=patched_cache)
        mx.eval(patched)

    fidelity = _logit_fidelity(baseline, patched)
    if not _fidelity_compatible(fidelity):
        raise RuntimeError(
            "patched logits failed fidelity limits: "
            f"token {fidelity['actual_next_token']} vs {fidelity['next_token']}, "
            f"KL={fidelity['kl_divergence']:.6g}, "
            f"mean={fidelity['mean_logit_error']:.6f}, "
            f"max={fidelity['max_logit_error']:.6f}"
        )
    print(
        f"Verified next token {fidelity['next_token']}; "
        f"KL={fidelity['kl_divergence']:.6g}, "
        f"max logit difference={fidelity['max_logit_error']:.6f}"
    )
    return fidelity


def _confirm_plan(model, tokenizer, prompt, arguments, plan, affine_prefill):
    if not plan.feature_count:
        return plan, None
    if arguments.confirmation_trials < 1:
        raise ValueError("confirmation trials must be positive")

    pairs = []
    print("Confirming the plan on the complete generation workload...")
    for trial in range(arguments.confirmation_trials):
        samples = {}
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
                affine_prefill,
            )
            samples[patched] = (float(response.generation_tps), elapsed, ttft)
        native = samples[False]
        generated = samples[True]
        pairs.append(
            {
                "decode_speedup": generated[0] / native[0],
                "end_to_end_speedup": native[1] / generated[1],
                "ttft_speedup": native[2] / generated[2],
            }
        )

    medians = {
        name: statistics.median(pair[name] for pair in pairs)
        for name in ("decode_speedup", "end_to_end_speedup", "ttft_speedup")
    }
    required_wins = max(1, (len(pairs) * 2 + 2) // 3)
    decode_sensitive = plan.attention or plan.rms_norm or plan.graph_fusion
    decode_floor = (
        _DECODE_CONFIRMATION_FLOOR if decode_sensitive else _PREFILL_ONLY_DECODE_CONFIRMATION_FLOOR
    )
    no_regression = (
        medians["decode_speedup"] >= decode_floor
        and medians["end_to_end_speedup"] >= 0.995
        and sum(pair["decode_speedup"] >= 0.98 for pair in pairs) >= required_wins
        and sum(pair["end_to_end_speedup"] >= 0.98 for pair in pairs) >= required_wins
    )
    meaningful_win = medians["ttft_speedup"] >= 1.03 or medians["end_to_end_speedup"] >= 1.01
    accepted = no_regression and meaningful_win
    confirmation = {
        "accepted": accepted,
        "decode_speedup_floor": decode_floor,
        "medians": medians,
        "pairs": pairs,
    }
    print(
        "Confirmation: "
        f"decode={medians['decode_speedup']:.3f}x, "
        f"TTFT={medians['ttft_speedup']:.3f}x, "
        f"total={medians['end_to_end_speedup']:.3f}x -> "
        f"{'accepted' if accepted else 'native fallback'}"
    )
    return plan if accepted else type(plan)(False, False, False, False, False), confirmation


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
    from metile.backends.mlx import (
        mlx_add_rms_norm_dispatches,
        mlx_attention_dispatches,
        mlx_rms_norm_dispatches,
    )
    from metile.backends.mlx_affine import mlx_affine_matmul_dispatches
    from metile.backends.mlx_block_scaled import mlx_block_scaled_dispatches
    from metile.backends.mlx_quantized import (
        mlx_affine_residual_qmv_dispatches,
        mlx_affine_swiglu_dispatches,
    )

    return {
        "attention": [dict(dispatch) for dispatch in mlx_attention_dispatches()],
        "rms_norm": [dict(dispatch) for dispatch in mlx_rms_norm_dispatches()],
        "add_rms_norm": [dict(dispatch) for dispatch in mlx_add_rms_norm_dispatches()],
        "affine_residual_qmv": [
            dict(dispatch) for dispatch in mlx_affine_residual_qmv_dispatches()
        ],
        "affine_swiglu": [dict(dispatch) for dispatch in mlx_affine_swiglu_dispatches()],
        "affine_matmul": [dict(dispatch) for dispatch in mlx_affine_matmul_dispatches()],
        "block_scaled": [dict(dispatch) for dispatch in mlx_block_scaled_dispatches()],
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


def _mlx_memory_metadata():
    import mlx.core as mx

    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _write_json_result(
    path,
    arguments,
    config,
    verification,
    results,
    medians,
    dispatches,
    plan,
    candidate_plan,
    confirmation,
):
    payload = {
        "schema_version": 6,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "revision": _git_revision(),
        "model": arguments.model,
        "model_config": _model_metadata(config),
        "hardware": _hardware_metadata(),
        "memory": _mlx_memory_metadata(),
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
            "confirmation_trials": arguments.confirmation_trials,
            "seed": arguments.seed,
        },
        "features": {
            "attention": not arguments.disable_attention,
            "rms_norm": not arguments.disable_rmsnorm,
            "graph_fusion": not arguments.disable_graph_fusion,
            "quantized_mlp": not arguments.disable_quantized_mlp,
            "affine_prefill": not arguments.disable_affine_prefill,
            "model_autotune": not arguments.disable_model_autotune,
        },
        "selected_plan": plan.as_dict(),
        "candidate_plan": candidate_plan.as_dict(),
        "plan_confirmation": confirmation,
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
    import mlx.core as mx
    from mlx_lm import load

    from metile.integrations.mlx_lm import (
        MLXLMPlan,
        autotune_metile_for_mlx_lm,
        prepare_mlx_lm_affine_prefill,
    )

    arguments = _arguments()
    mx.reset_peak_memory()
    model, tokenizer, config = load(arguments.model, return_config=True)
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    mx.random.seed(arguments.seed)
    prompt = mx.random.randint(0, vocab_size, (arguments.prompt_tokens,)).tolist()

    affine_prefill = None
    if not arguments.disable_affine_prefill:
        try:
            print("AOT-repacking exact affine prefill projections...")
            affine_prefill = prepare_mlx_lm_affine_prefill(model)
            print(f"Prepared {affine_prefill.projection_count} affine projections")
        except ValueError as error:
            print(f"Affine prefill unavailable: {error}")

    requested_plan = MLXLMPlan(
        attention=not arguments.disable_attention,
        rms_norm=not arguments.disable_rmsnorm,
        graph_fusion=not arguments.disable_graph_fusion,
        quantized_mlp=not arguments.disable_quantized_mlp,
        affine_prefill=affine_prefill is not None,
    )
    if arguments.disable_model_autotune:
        candidate_plan = requested_plan
    else:
        print("Autotuning the MLX-LM feature plan...")
        candidate_plan = autotune_metile_for_mlx_lm(
            model,
            mx.array(prompt)[None],
            attention=requested_plan.attention,
            rms_norm=requested_plan.rms_norm,
            graph_fusion=requested_plan.graph_fusion,
            quantized_mlp=requested_plan.quantized_mlp,
            affine_prefill=affine_prefill,
            decode_steps=arguments.plan_decode_steps,
            trials=arguments.plan_trials,
        )
    candidate = (
        ", ".join(name for name, active in candidate_plan.as_dict().items() if active)
        or "native MLX"
    )
    print(f"Candidate model plan: {candidate}")

    verification = None
    if not arguments.skip_verify:
        verification = _verify_model(model, prompt, arguments, candidate_plan, affine_prefill)

    print("Warming MLX baseline...")
    _generate(
        model,
        tokenizer,
        prompt,
        arguments,
        patched=False,
        plan=candidate_plan,
        affine_prefill=affine_prefill,
    )
    if candidate_plan.feature_count:
        print("Compiling and autotuning meTile MLX kernels...")
        _generate(
            model,
            tokenizer,
            prompt,
            arguments,
            patched=True,
            plan=candidate_plan,
            affine_prefill=affine_prefill,
        )
    else:
        print("Native fallback selected; sharing each measurement across both labels.")

    plan, confirmation = _confirm_plan(
        model,
        tokenizer,
        prompt,
        arguments,
        candidate_plan,
        affine_prefill,
    )
    enabled = ", ".join(name for name, active in plan.as_dict().items() if active) or "native MLX"
    print(f"Selected model plan: {enabled}")

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
                affine_prefill=affine_prefill,
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
                affine_prefill,
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
    baseline_prefill = statistics.median(sample[1] for sample in results["MLX"])
    metile_prefill = statistics.median(sample[1] for sample in results["MLX + meTile"])
    baseline_total = statistics.median(sample[2] for sample in results["MLX"])
    metile_total = statistics.median(sample[2] for sample in results["MLX + meTile"])
    baseline_ttft = statistics.median(sample[3] for sample in results["MLX"])
    metile_ttft = statistics.median(sample[3] for sample in results["MLX + meTile"])
    paired = tuple(zip(results["MLX"], results["MLX + meTile"], strict=True))
    decode_speedup = statistics.median(generated[0] / native[0] for native, generated in paired)
    prefill_speedup = statistics.median(generated[1] / native[1] for native, generated in paired)
    end_to_end_speedup = statistics.median(native[2] / generated[2] for native, generated in paired)
    ttft_speedup = statistics.median(native[3] / generated[3] for native, generated in paired)
    medians = {
        "mlx_decode_tokens_per_second": baseline,
        "metile_decode_tokens_per_second": metile_decode,
        "decode_speedup": decode_speedup,
        "mlx_prefill_tokens_per_second": baseline_prefill,
        "metile_prefill_tokens_per_second": metile_prefill,
        "prefill_speedup": prefill_speedup,
        "mlx_elapsed_seconds": baseline_total,
        "metile_elapsed_seconds": metile_total,
        "end_to_end_speedup": end_to_end_speedup,
        "mlx_time_to_first_token_seconds": baseline_ttft,
        "metile_time_to_first_token_seconds": metile_ttft,
        "ttft_speedup": ttft_speedup,
    }
    dispatches = _selected_dispatches()
    print("\nMedian results")
    print(f"MLX decode:          {baseline:.2f} tok/s")
    print(f"MLX + meTile decode: {metile_decode:.2f} tok/s ({decode_speedup:.3f}x paired)")
    print(
        f"Prefill throughput:  {baseline_prefill:.2f} -> {metile_prefill:.2f} tok/s "
        f"({prefill_speedup:.3f}x paired)"
    )
    print(f"End-to-end speedup:  {end_to_end_speedup:.3f}x paired")
    print(
        f"TTFT:                {baseline_ttft * 1e3:.1f}ms -> "
        f"{metile_ttft * 1e3:.1f}ms ({ttft_speedup:.3f}x paired)"
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
    print("Selected quantized down/residual schedules")
    for dispatch in dispatches["affine_residual_qmv"]:
        print(
            f"{dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} block={dispatch['block']} "
            f"outputs/simdgroup={dispatch['outputs_per_simdgroup']} "
            f"decode={dispatch['decode_dtype']}"
        )
    print("Selected affine prefill schedules")
    for dispatch in dispatches["affine_matmul"]:
        print(
            f"rows={dispatch['rows']} {dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} block={dispatch['block_m']}x{dispatch['block_n']} "
            f"schedule={dispatch['schedule']}"
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
            candidate_plan,
            confirmation,
        )


if __name__ == "__main__":
    main()
