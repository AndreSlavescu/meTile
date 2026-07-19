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
_DECODE_CONFIRMATION_WIN = 1.01
_STRONG_DECODE_CONFIRMATION_WIN = 1.05
_PREFILL_ONLY_DECODE_CONFIRMATION_FLOOR = 0.99
_TTFT_CONFIRMATION_FLOOR = 0.995
_TTFT_CONFIRMATION_WIN = 1.02


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
    parser.add_argument("--confirmation-trials", type=int, default=3)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _generate(
    model,
    tokenizer,
    prompt,
    arguments,
    patched,
    plan,
    affine_prefill,
    dense_mlp,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
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
            dense_mlp=dense_mlp,
            compressed_down=compressed_down,
            compressed_gate_up=compressed_gate_up,
            compressed_vocab=compressed_vocab,
            compressed_attention=compressed_attention,
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


def _verify_model(
    model,
    prompt,
    arguments,
    plan,
    affine_prefill,
    dense_mlp,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
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
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
        plan=plan,
    ):
        patched_prefix = model(tokens[:, :-1], cache=patched_cache)
        mx.eval(patched_prefix)
        patched = model(tokens[:, -1:], cache=patched_cache)
        mx.eval(patched)

    fidelity = _logit_fidelity(baseline, patched)
    policies = []
    if plan.compressed_down and compressed_down is not None:
        policies.append(compressed_down.fidelity_compatible)
    if plan.compressed_gate_up and compressed_gate_up is not None:
        policies.append(compressed_gate_up.fidelity_compatible)
    if plan.compressed_vocab and compressed_vocab is not None:
        policies.append(compressed_vocab.fidelity_compatible)
    if plan.compressed_attention and compressed_attention is not None:
        policies.append(compressed_attention.fidelity_compatible)
    compatible = (
        all(policy(fidelity) for policy in policies) if policies else _fidelity_compatible(fidelity)
    )
    if not compatible:
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


def _confirm_plan(
    model,
    tokenizer,
    prompt,
    arguments,
    plan,
    affine_prefill,
    dense_mlp,
    compressed_down=None,
    compressed_gate_up=None,
    compressed_vocab=None,
    compressed_attention=None,
):
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
                dense_mlp,
                compressed_down,
                compressed_gate_up,
                compressed_vocab,
                compressed_attention,
            )
            samples[patched] = (
                float(response.generation_tps),
                float(response.prompt_tps),
                elapsed,
                ttft,
            )
        native = samples[False]
        generated = samples[True]
        pairs.append(
            {
                "decode_speedup": generated[0] / native[0],
                "prefill_speedup": generated[1] / native[1],
                "end_to_end_speedup": native[2] / generated[2],
                "ttft_speedup": native[3] / generated[3],
            }
        )

    medians = {
        name: statistics.median(pair[name] for pair in pairs)
        for name in (
            "decode_speedup",
            "prefill_speedup",
            "end_to_end_speedup",
            "ttft_speedup",
        )
    }
    required_wins = max(1, (len(pairs) * 2 + 2) // 3)
    decode_only_compression = plan.is_decode_only_compression
    decode_sensitive = (
        plan.attention
        or plan.rms_norm
        or plan.graph_fusion
        or plan.quantized_mlp
        or plan.dense_mlp
        or plan.dense_residual
        or plan.compressed_down
        or plan.compressed_gate_up
        or plan.compressed_vocab
        or plan.compressed_attention
    )
    decode_floor = (
        _DECODE_CONFIRMATION_FLOOR if decode_sensitive else _PREFILL_ONLY_DECODE_CONFIRMATION_FLOOR
    )
    stable_ttft = decode_only_compression or (
        sum(pair["ttft_speedup"] >= 0.98 for pair in pairs) >= required_wins
        or (
            medians["decode_speedup"] >= _STRONG_DECODE_CONFIRMATION_WIN
            and sum(pair["end_to_end_speedup"] >= 1.0 for pair in pairs) >= required_wins
        )
    )
    no_regression = (
        medians["decode_speedup"] >= decode_floor
        and (decode_only_compression or medians["ttft_speedup"] >= _TTFT_CONFIRMATION_FLOOR)
        and medians["end_to_end_speedup"] >= 0.995
        and sum(pair["decode_speedup"] >= 0.98 for pair in pairs) >= required_wins
        and stable_ttft
        and sum(pair["end_to_end_speedup"] >= 0.98 for pair in pairs) >= required_wins
    )
    meaningful_decode_win = (
        decode_sensitive
        and medians["decode_speedup"] >= _DECODE_CONFIRMATION_WIN
        and sum(pair["decode_speedup"] >= 1.0 for pair in pairs) >= required_wins
    )
    meaningful_win = (
        medians["ttft_speedup"] >= _TTFT_CONFIRMATION_WIN
        or medians["end_to_end_speedup"] >= 1.01
        or meaningful_decode_win
    )
    accepted = no_regression and meaningful_win
    confirmation = {
        "accepted": accepted,
        "decode_only_compression": decode_only_compression,
        "decode_speedup_win": _DECODE_CONFIRMATION_WIN,
        "decode_speedup_floor": decode_floor,
        "medians": medians,
        "pairs": pairs,
        "required_wins": required_wins,
        "strong_decode_speedup_win": _STRONG_DECODE_CONFIRMATION_WIN,
        "ttft_speedup_floor": _TTFT_CONFIRMATION_FLOOR,
    }
    print(
        "Confirmation: "
        f"decode={medians['decode_speedup']:.3f}x, "
        f"prefill={medians['prefill_speedup']:.3f}x, "
        f"TTFT={medians['ttft_speedup']:.3f}x, "
        f"total={medians['end_to_end_speedup']:.3f}x -> "
        f"{'accepted' if accepted else 'native fallback'}"
    )
    native_plan = type(plan)(**{name: False for name in plan.as_dict()})
    return plan if accepted else native_plan, confirmation


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
    from metile.backends.mlx_dense import mlx_dense_matmul_dispatches
    from metile.backends.mlx_dense_residual import mlx_dense_residual_dispatches
    from metile.backends.mlx_dense_swiglu import mlx_dense_swiglu_dispatches
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
        "dense_matmul": [dict(dispatch) for dispatch in mlx_dense_matmul_dispatches()],
        "dense_residual": [dict(dispatch) for dispatch in mlx_dense_residual_dispatches()],
        "dense_swiglu": [dict(dispatch) for dispatch in mlx_dense_swiglu_dispatches()],
    }


def _model_metadata(config):
    text_config = config.get("text_config", config)
    keys = (
        "dtype",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "torch_dtype",
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


def _source_weight_representation(config):
    if not isinstance(config, dict):
        return "model_native"
    text_config = config.get("text_config", config)
    for key in ("dtype", "torch_dtype"):
        value = text_config.get(key)
        if isinstance(value, str) and value:
            return value
    return "model_native"


def _precision_comparison(plan, compressed_down, config=None):
    selected = plan.as_dict()
    source_weights = _source_weight_representation(config)
    formats = []
    if selected["compressed_down"]:
        if compressed_down is None:
            raise ValueError("selected compressed down plan has no prepared weight format")
        formats.append(compressed_down.format)
    for feature in ("compressed_gate_up", "compressed_vocab", "compressed_attention"):
        if selected[feature]:
            formats.append("affine8")
    if not formats:
        return {
            "class": "same_precision",
            "same_weight_representation": True,
            "baseline_weights": source_weights,
            "optimized_decode_weights": [source_weights],
            "prefill_weights": source_weights,
            "native_weights_preserved": True,
        }
    unique_formats = sorted(set(formats))
    if unique_formats == ["affine8"]:
        comparison_class = "mixed_precision_affine_int8_decode"
    elif unique_formats == ["mxfp8"]:
        comparison_class = "mixed_precision_mxfp8_decode"
    else:
        comparison_class = "mixed_precision_hybrid_decode"
    return {
        "class": comparison_class,
        "same_weight_representation": False,
        "baseline_weights": source_weights,
        "optimized_decode_weights": unique_formats,
        "prefill_weights": source_weights,
        "native_weights_preserved": True,
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
    plan_autotune_seconds,
    dense_mlp,
    compressed_down,
    compressed_gate_up,
    compressed_vocab,
    compressed_attention,
):
    precision_comparison = _precision_comparison(plan, compressed_down, config)
    payload = {
        "schema_version": 19,
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
            "plan_decode_trajectory": "native_autoregressive",
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
            "dense_mlp": not arguments.disable_dense_mlp,
            "dense_residual": not arguments.disable_dense_mlp,
            "compressed_down": arguments.compressed_down_format != "none",
            "compressed_gate_up": arguments.compressed_gate_up,
            "compressed_vocab": arguments.compressed_vocab,
            "compressed_attention": arguments.compressed_attention,
            "model_autotune": not arguments.disable_model_autotune,
        },
        "selected_plan": plan.as_dict(),
        "candidate_plan": candidate_plan.as_dict(),
        "precision_comparison": precision_comparison,
        "plan_autotune_seconds": plan_autotune_seconds,
        "dense_mlp_implementation": (dense_mlp.implementation if dense_mlp is not None else None),
        "compressed_down": (
            {
                "allow_approximate": compressed_down.allow_approximate,
                "calibration_fidelity": compressed_down.calibration_fidelity,
                "format": compressed_down.format,
                "group_size": compressed_down.group_size,
                "group_tuning": compressed_down.group_tuning,
                "layer_indices": compressed_down.layer_indices,
                "projections": compressed_down.projection_count,
                "repack_bytes": compressed_down.repack_bytes,
                "selection": compressed_down.selection,
            }
            if compressed_down is not None
            else None
        ),
        "compressed_gate_up": (
            {
                "calibration_fidelity": compressed_gate_up.calibration_fidelity,
                "group_size": compressed_gate_up.group_size,
                "group_tuning": compressed_gate_up.group_tuning,
                "implementation": compressed_gate_up.implementation,
                "implementation_tuning": compressed_gate_up.implementation_tuning,
                "layer_indices": compressed_gate_up.layer_indices,
                "layers": compressed_gate_up.layer_count,
                "projections": compressed_gate_up.projection_count,
                "repack_bytes": compressed_gate_up.repack_bytes,
                "selection": compressed_gate_up.selection,
            }
            if compressed_gate_up is not None
            else None
        ),
        "compressed_vocab": (
            {
                "calibration_fidelity": compressed_vocab.calibration_fidelity,
                "group_size": compressed_vocab.group_size,
                "group_tuning": compressed_vocab.group_tuning,
                "projections": compressed_vocab.projection_count,
                "repack_bytes": compressed_vocab.repack_bytes,
                "tied": compressed_vocab.tied,
            }
            if compressed_vocab is not None
            else None
        ),
        "compressed_attention": (
            {
                "calibration_fidelity": compressed_attention.calibration_fidelity,
                "group_size": compressed_attention.group_size,
                "group_tuning": compressed_attention.group_tuning,
                "layer_indices": compressed_attention.layer_indices,
                "layers": compressed_attention.layer_count,
                "projections": compressed_attention.projection_count,
                "repack_bytes": compressed_attention.repack_bytes,
                "selection": compressed_attention.selection,
            }
            if compressed_attention is not None
            else None
        ),
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
        prepare_mlx_lm_compressed_attention,
        prepare_mlx_lm_compressed_down,
        prepare_mlx_lm_compressed_gate_up,
        prepare_mlx_lm_compressed_vocab,
        prepare_mlx_lm_dense_mlp,
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

    dense_mlp = None
    if not arguments.disable_dense_mlp:
        try:
            print("AOT-repacking dense BF16/FP16 gate/up projections...")
            dense_mlp = prepare_mlx_lm_dense_mlp(model)
            print(
                f"Prepared {dense_mlp.mlp_count} dense SwiGLU blocks "
                f"({dense_mlp.repack_bytes / 2**30:.2f} GiB repacked)"
            )
        except ValueError as error:
            print(f"Dense MLP unavailable: {error}")

    compressed_gate_up = None
    if arguments.compressed_gate_up:
        try:
            print("AOT-compressing dense gate/up projection pairs as affine8...")
            compressed_gate_up = prepare_mlx_lm_compressed_gate_up(
                model,
                group_size=arguments.compressed_gate_up_group_size,
            )
            print(
                f"Prepared {compressed_gate_up.layer_count} compressed gate/up layers "
                f"at group {compressed_gate_up.group_size} "
                f"({compressed_gate_up.repack_bytes / 2**30:.2f} GiB repacked)"
            )
            if compressed_gate_up.group_tuning is not None:
                timings = compressed_gate_up.group_tuning["median_nanoseconds"]
                print(
                    "Gate/up affine8 group autotune: "
                    + ", ".join(
                        f"g{group}={timings[str(group)] / 1e6:.3f}ms"
                        for group in sorted(map(int, timings))
                    )
                )
        except ValueError as error:
            print(f"Compressed gate/up unavailable: {error}")

    compressed_vocab = None
    if arguments.compressed_vocab:
        try:
            print("AOT-compressing the vocabulary projection as affine8...")
            compressed_vocab = prepare_mlx_lm_compressed_vocab(
                model,
                group_size=arguments.compressed_vocab_group_size,
            )
            print(
                "Prepared one compressed vocabulary projection "
                f"at group {compressed_vocab.group_size} "
                f"({compressed_vocab.repack_bytes / 2**30:.2f} GiB repacked)"
            )
            if compressed_vocab.group_tuning is not None:
                timings = compressed_vocab.group_tuning["median_nanoseconds"]
                print(
                    "Vocabulary affine8 group autotune: "
                    + ", ".join(
                        f"g{group}={timings[str(group)] / 1e6:.3f}ms"
                        for group in sorted(map(int, timings))
                    )
                )
        except ValueError as error:
            print(f"Compressed vocabulary unavailable: {error}")

    compressed_attention = None
    if arguments.compressed_attention:
        try:
            print("AOT-compressing dense attention projections as affine8...")
            compressed_attention = prepare_mlx_lm_compressed_attention(
                model,
                group_size=arguments.compressed_attention_group_size,
            )
            print(
                f"Prepared {compressed_attention.layer_count} compressed attention layers "
                f"at group {compressed_attention.group_size} "
                f"({compressed_attention.repack_bytes / 2**30:.2f} GiB repacked)"
            )
            if compressed_attention.group_tuning is not None:
                timings = compressed_attention.group_tuning["median_nanoseconds"]
                print(
                    "Attention affine8 group autotune: "
                    + ", ".join(
                        f"g{group}={timings[str(group)] / 1e6:.3f}ms"
                        for group in sorted(map(int, timings))
                    )
                )
        except ValueError as error:
            print(f"Compressed attention unavailable: {error}")

    compressed_down = None
    if arguments.compressed_down_format != "none":
        try:
            print(
                f"AOT-compressing dense down projections as {arguments.compressed_down_format}..."
            )
            compressed_down = prepare_mlx_lm_compressed_down(
                model,
                format=arguments.compressed_down_format,
                group_size=arguments.compressed_down_group_size,
                allow_approximate=arguments.allow_approximate_compressed_down,
            )
            print(
                f"Prepared {compressed_down.projection_count} compressed down projections "
                f"at group {compressed_down.group_size} "
                f"({compressed_down.repack_bytes / 2**30:.2f} GiB repacked)"
            )
            if compressed_down.group_tuning is not None:
                timings = compressed_down.group_tuning["median_nanoseconds"]
                print(
                    "Affine8 group autotune: "
                    + ", ".join(
                        f"g{group}={timings[str(group)] / 1e6:.3f}ms"
                        for group in sorted(map(int, timings))
                    )
                )
        except ValueError as error:
            print(f"Compressed down unavailable: {error}")

    requested_plan = MLXLMPlan(
        attention=not arguments.disable_attention,
        rms_norm=not arguments.disable_rmsnorm,
        graph_fusion=not arguments.disable_graph_fusion,
        quantized_mlp=not arguments.disable_quantized_mlp,
        affine_prefill=affine_prefill is not None,
        dense_mlp=dense_mlp is not None,
        dense_residual=dense_mlp is not None,
        compressed_down=compressed_down is not None,
        compressed_gate_up=compressed_gate_up is not None,
        compressed_vocab=compressed_vocab is not None,
        compressed_attention=compressed_attention is not None,
    )
    if arguments.disable_model_autotune:
        candidate_plan = requested_plan
        plan_autotune_seconds = None
    else:
        print("Autotuning the MLX-LM feature plan...")
        plan_autotune_started = time.perf_counter()
        candidate_plan = autotune_metile_for_mlx_lm(
            model,
            mx.array(prompt)[None],
            attention=requested_plan.attention,
            rms_norm=requested_plan.rms_norm,
            graph_fusion=requested_plan.graph_fusion,
            quantized_mlp=requested_plan.quantized_mlp,
            affine_prefill=affine_prefill,
            dense_mlp=dense_mlp,
            compressed_down=compressed_down,
            compressed_gate_up=compressed_gate_up,
            compressed_vocab=compressed_vocab,
            compressed_attention=compressed_attention,
            decode_steps=arguments.plan_decode_steps,
            trials=arguments.plan_trials,
        )
        plan_autotune_seconds = time.perf_counter() - plan_autotune_started
        print(f"Model-plan autotune completed in {plan_autotune_seconds:.2f}s")
    candidate = (
        ", ".join(name for name, active in candidate_plan.as_dict().items() if active)
        or "native MLX"
    )
    if compressed_down is not None and compressed_down.calibrated:
        print(
            "Compressed down calibration: "
            f"{compressed_down.selection} "
            f"({compressed_down.projection_count} projections, "
            f"{compressed_down.repack_bytes / 2**30:.2f} GiB active)"
        )
    if compressed_gate_up is not None and compressed_gate_up.calibrated:
        print(
            "Compressed gate/up calibration: "
            f"{compressed_gate_up.selection} at group {compressed_gate_up.group_size} "
            f"using {compressed_gate_up.implementation} execution "
            f"({compressed_gate_up.layer_count} layers, "
            f"{compressed_gate_up.repack_bytes / 2**30:.2f} GiB active)"
        )
    if compressed_vocab is not None and compressed_vocab.calibrated:
        fidelity = compressed_vocab.calibration_fidelity
        print(
            "Compressed vocabulary calibration: "
            f"{'accepted' if compressed_vocab.projection_count else 'rejected'} "
            f"(KL={fidelity['kl_divergence']:.6g}, "
            f"mean={fidelity['mean_logit_error']:.6f}, "
            f"max={fidelity['max_logit_error']:.6f})"
        )
    if compressed_attention is not None and compressed_attention.calibrated:
        print(
            "Compressed attention calibration: "
            f"{compressed_attention.selection} at group {compressed_attention.group_size} "
            f"({compressed_attention.layer_count} layers, "
            f"{compressed_attention.repack_bytes / 2**30:.2f} GiB active)"
        )
    print(f"Candidate model plan: {candidate}")
    if dense_mlp is not None:
        print(f"Dense MLP implementation: {dense_mlp.implementation}")

    verification = None
    if not arguments.skip_verify:
        verification = _verify_model(
            model,
            prompt,
            arguments,
            candidate_plan,
            affine_prefill,
            dense_mlp,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )

    print("Warming MLX baseline...")
    _generate(
        model,
        tokenizer,
        prompt,
        arguments,
        patched=False,
        plan=candidate_plan,
        affine_prefill=affine_prefill,
        dense_mlp=dense_mlp,
        compressed_down=compressed_down,
        compressed_gate_up=compressed_gate_up,
        compressed_vocab=compressed_vocab,
        compressed_attention=compressed_attention,
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
            dense_mlp=dense_mlp,
            compressed_down=compressed_down,
            compressed_gate_up=compressed_gate_up,
            compressed_vocab=compressed_vocab,
            compressed_attention=compressed_attention,
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
        dense_mlp,
        compressed_down,
        compressed_gate_up,
        compressed_vocab,
        compressed_attention,
    )
    enabled = ", ".join(name for name, active in plan.as_dict().items() if active) or "native MLX"
    print(f"Selected model plan: {enabled}")
    precision_comparison = _precision_comparison(plan, compressed_down, config)
    if not precision_comparison["same_weight_representation"]:
        formats = ", ".join(precision_comparison["optimized_decode_weights"])
        print(
            "Precision comparison: mixed precision "
            f"({precision_comparison['baseline_weights']} vs {formats} decode); not same-format"
        )

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
                dense_mlp=dense_mlp,
                compressed_down=compressed_down,
                compressed_gate_up=compressed_gate_up,
                compressed_vocab=compressed_vocab,
                compressed_attention=compressed_attention,
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
                dense_mlp,
                compressed_down,
                compressed_gate_up,
                compressed_vocab,
                compressed_attention,
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
    print("Selected dense SwiGLU schedules")
    for dispatch in dispatches["dense_swiglu"]:
        if dispatch["implementation"].startswith("simdgroup"):
            schedule = (
                f"{dispatch['implementation']} "
                f"outputs/simdgroup={dispatch['outputs_per_simdgroup']} "
                f"simdgroups/threadgroup={dispatch['simdgroups_per_threadgroup']} "
                f"k-unroll={dispatch['k_unroll']}"
            )
        else:
            schedule = (
                f"block={dispatch['block_m']}x{dispatch['block_n']} "
                f"schedule={dispatch['schedule']} k-unroll={dispatch['k_unroll']}"
            )
        print(
            f"rows={dispatch['rows']} "
            f"{dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} {schedule}"
        )
    print("Selected dense projection schedules")
    for dispatch in dispatches["dense_matmul"]:
        print(
            f"rows={dispatch['rows']} "
            f"{dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} block={dispatch['block_m']}x{dispatch['block_n']} "
            f"schedule={dispatch['schedule']} k-unroll={dispatch['k_unroll']}"
        )
    print("Selected dense down/residual schedules")
    for dispatch in dispatches["dense_residual"]:
        print(
            f"rows={dispatch['rows']} "
            f"{dispatch['input_features']}->{dispatch['output_features']}: "
            f"{dispatch['algorithm']} outputs/simdgroup={dispatch['outputs_per_simdgroup']} "
            f"simdgroups/threadgroup={dispatch['simdgroups_per_threadgroup']}"
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
            plan_autotune_seconds,
            dense_mlp,
            compressed_down,
            compressed_gate_up,
            compressed_vocab,
            compressed_attention,
        )


if __name__ == "__main__":
    main()
