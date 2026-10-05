"""Plot independently validated chunk tuning and held-out Qwen3 generation.

Tuning observations select a chunk size, not a performance claim on the
held-out document. Every plotted dot is a recorded observation from its own
phase. Five trials do not justify reconstructed densities or confidence bands.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.megakernels.qwen3 import _fidelity_tolerances
from benchmarks.megakernels.qwen3_chunked_prefill import _require_disjoint_prompts, _summarize
from benchmarks.plots import chartstyle as style

DEFAULT_INPUT = Path("benchmarks/results/m5-qwen3-chunked-prefill-end-to-end.json")
DEFAULT_OUTPUT = Path("docs/_static/qwen3-chunked-prefill.png")
ARMS = (("MLX", "MLX model loop", style.PREFILL), ("chunked", "meTile chunked", style.DECODE))
METRICS = (
    ("time_to_first_token_seconds", "Time to first token", "seconds · lower is faster"),
    (
        "decode_tokens_per_second",
        "Decode after the first token",
        "tokens / second · higher is faster",
    ),
    ("total_wall_seconds", "Complete generation", "seconds · lower is faster"),
)


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _tokens(values, count, vocabulary):
    return (
        isinstance(values, list)
        and len(values) == count
        and all(_integer(value, 0) and value < vocabulary for value in values)
    )


def _finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _weight_packing(report, held, geometry):
    weights = report["weights"]
    precision = report["precision_comparison"]
    dtype = weights["both_backends"]
    declared = report.get("candidate_weight_packing")
    configured = geometry.get("decode_weight_packing")
    if declared is None and configured is None:
        if (
            precision.get("same_weight_representation") is not True
            or precision.get("class", "same_storage_precision") != "same_storage_precision"
            or weights.get("candidate_decode_projection_storage", dtype) != dtype
        ):
            raise ValueError("unpacked weights require matching storage representation metadata")
        return False
    expected = {
        "enabled": True,
        "format": "fp32_high16x2_u32_lossless",
        "decoded_dtype": "float32",
        "storage_dtype": "uint32",
        "values_per_word": 2,
        "scopes": ["layer_weights", "embedding"],
        "consumers": ["decode_projections", "first_token_vocabulary_projection"],
        "original_buffers_retained": True,
    }
    hidden = geometry.get("hidden_size")
    parameters = held.get("parameter_count")
    if not _integer(hidden) or not _integer(parameters) or parameters <= hidden:
        raise ValueError("packed weights require model dimensions and parameter counts")
    for metadata in (declared, configured):
        metadata = _mapping(metadata, "candidate weight packing")
        if any(
            type(metadata.get(name)) is not type(value) or metadata[name] != value
            for name, value in expected.items()
        ):
            raise ValueError("candidate weight packing requires the verified lossless FP32 format")
        verification = _mapping(metadata.get("verification"), "weight packing verification")
        if any(
            verification.get(name) is not True
            for name in ("finite", "even_elements", "zero_low16_bits", "roundtrip_bitwise")
        ):
            raise ValueError("all lossless weight packing verification checks are required")
        count = metadata.get("element_count")
        if (
            not _integer(count)
            or count % 2
            or count != parameters - hidden
            or not _integer(metadata.get("unpacked_bytes"))
            or metadata["unpacked_bytes"] != count * 4
            or not _integer(metadata.get("packed_bytes"))
            or metadata["packed_bytes"] != count * 2
        ):
            raise ValueError("packed weight element and byte counts must match model storage")
    if declared != configured:
        raise ValueError("candidate weight packing must match held-out geometry")
    if (
        dtype != "float32"
        or weights.get("dtype_scope") != "decoded_weight_values"
        or weights.get("native_projection_storage") != dtype
        or weights.get("candidate_prefill_projection_storage") != dtype
        or weights.get("candidate_decode_projection_storage") != "packed_uint32_bfloat16_pairs"
        or precision.get("class") != "lossless_weight_storage"
        or precision.get("same_weight_representation") is not False
        or precision.get("same_weight_values") is not True
        or precision.get("bitwise_exact") is not False
        or precision.get("storage_dtype_scope") != "activations_and_kv_cache"
    ):
        raise ValueError("lossless packed weights require explicit unchanged FP32 arithmetic scope")
    return True


def _execution_metadata(execution, geometry):
    if "fused_row_arithmetic" in geometry and type(geometry["fused_row_arithmetic"]) is not bool:
        raise ValueError("candidate fused row arithmetic metadata must be boolean")
    fields = {
        "candidate_projection_backend": "prefill_projection_backend",
        "candidate_projection_tile": "prefill_projection_tile",
        "candidate_strict_math": "prefill_strict_math",
        "candidate_projection_relaxed_precision": "prefill_projection_relaxed_precision",
    }
    modern = any(name in execution for name in (*fields, "candidate_attention_backend")) or any(
        name in geometry for name in ("prefill_strict_math", "prefill_projection_relaxed_precision")
    )
    if modern and (
        "candidate_attention_backend" not in execution
        or any(name not in execution or target not in geometry for name, target in fields.items())
    ):
        raise ValueError("complete candidate backend and precision metadata are required")
    for name, target in fields.items():
        if name in execution and execution[name] != geometry[target]:
            raise ValueError(
                "candidate backend and precision metadata must match held-out geometry"
            )
    if "prefill_projection_backend" in geometry and geometry["prefill_projection_backend"] not in (
        "simdgroup",
        "tensor_ops",
    ):
        raise ValueError("candidate projection backend must be simdgroup or tensor_ops")
    for metadata, name in (
        (geometry, "prefill_projection_tile"),
        (execution, "candidate_projection_tile"),
    ):
        if name not in metadata:
            continue
        tile = metadata[name]
        if (
            not isinstance(tile, list)
            or len(tile) != 3
            or any(not _integer(size, 8) or size % 8 for size in tile)
        ):
            raise ValueError(
                "candidate projection tile must contain three positive multiples of eight"
            )
    for name, target, required in (
        ("candidate_strict_math", "prefill_strict_math", True),
        ("candidate_projection_relaxed_precision", "prefill_projection_relaxed_precision", False),
    ):
        if modern and (execution[name] is not required or geometry[target] is not required):
            raise ValueError(
                "candidate requires strict math and disabled relaxed projection precision"
            )
    if "prefill_attention" not in geometry and not modern:
        return
    attention = _mapping(geometry.get("prefill_attention"), "prefill attention geometry")
    kinds = {"tiled": "query_tiled_shared_kv", "matrix": "matrix_tiled_online_softmax"}
    backend = execution.get("candidate_attention_backend")
    if attention.get("kind") not in kinds.values() or (
        modern and (not isinstance(backend, str) or kinds.get(backend) != attention.get("kind"))
    ):
        raise ValueError("candidate attention backend must match its held-out geometry kind")
    if attention["kind"] != kinds["matrix"]:
        if any(
            name in attention
            for name in (
                "softmax_exponential",
                "denominator_fma",
                "normalization",
                "direct_device_memory",
                "masked_device_tile_scratch_bytes",
            )
        ):
            raise ValueError("matrix attention metadata requires the matrix attention backend")
        return
    if (
        attention.get("compiler_backend") != "simdgroup_inline"
        or any(
            not _integer(attention.get(name), 8) or attention[name] % 8
            for name in ("query_rows_per_threadgroup", "key_tile")
        )
        or not _integer(attention.get("threads"), 32)
        or attention["threads"] > 1024
        or attention["threads"] % 32
        or not _integer(attention.get("shared_padding"), 0)
        or type(attention.get("unroll_mma")) is not bool
    ):
        raise ValueError("matrix attention needs valid SIMD-group tile and shared-layout metadata")
    for name in ("transpose_keys", "register_stats", "unroll_softmax", "softmax_base2"):
        if name in attention and type(attention[name]) is not bool:
            raise ValueError("matrix attention layout options must be boolean")
    reuse_fields = ("cached_query_fragments", "query_kv_shared_storage")
    device_fields = ("direct_device_memory", "masked_device_tile_scratch_bytes")
    direct = attention.get("direct_device_memory", False)
    if any(name in attention for name in device_fields):
        scratch = attention.get("masked_device_tile_scratch_bytes")
        dtype = geometry.get("storage_dtype")
        if (
            any(name not in attention for name in (*device_fields, *reuse_fields))
            or type(direct) is not bool
            or not _integer(scratch, 0)
            or dtype not in ("float16", "float32")
            or scratch
            != (attention["threads"] * 2 * (2 if dtype == "float16" else 4) if direct else 0)
        ):
            raise ValueError(
                "matrix attention device memory metadata needs matching explicit scratch"
            )
    cached_layout = (
        attention["query_rows_per_threadgroup"] == (attention["threads"] // 4 if direct else 32)
        and attention["shared_padding"] == 0
        and attention["unroll_mma"]
    )
    if direct and not cached_layout:
        raise ValueError("direct matrix attention query ownership must match its SIMD-group layout")
    if any(name in attention for name in reuse_fields):
        cached = attention.get("cached_query_fragments")
        reused = attention.get("query_kv_shared_storage")
        dimension = geometry.get("head_dim")
        if (
            type(reused) is not bool
            or not _integer(cached, 0)
            or not _integer(dimension, 8)
            or dimension % 8
            or cached != (dimension // 8 if cached_layout else 0)
            or reused != (cached_layout and not direct)
        ):
            raise ValueError("matrix attention query reuse metadata must match its tile layout")
    if "load_vector" in attention and not _integer(attention["load_vector"]):
        raise ValueError("matrix attention copy-group width must be a positive integer")
    if "softmax_lanes" in attention:
        lanes = attention["softmax_lanes"]
        if not _integer(lanes) or lanes > 32 or lanes & (lanes - 1):
            raise ValueError("matrix attention softmax lanes must be a power of two up to 32")
    math_fields = ("softmax_exponential", "denominator_fma", "normalization")
    if not any(name in attention for name in math_fields):
        return
    if "softmax_base2" not in attention:
        raise ValueError("matrix attention math metadata requires an explicit softmax base")
    expected = (
        ("fast_exp2", True, "divide")
        if attention["softmax_base2"]
        else ("exp", False, "reciprocal_multiply")
    )
    for name, value in zip(math_fields, expected, strict=True):
        if name in attention and (
            type(attention[name]) is not type(value) or attention[name] != value
        ):
            raise ValueError("matrix attention math metadata must match its declared softmax base")


def _prompt(prompt, vocabulary):
    prompt = _mapping(prompt, "document prompt")
    count = prompt.get("actual_prompt_tokens")
    hashes = prompt.get("sources_sha256")
    if (
        not _integer(count)
        or not _tokens(prompt.get("token_ids"), count, vocabulary)
        or not _integer(prompt.get("source_token_count"), count)
        or prompt.get("token_offset") != 0
        or prompt.get("chat_template") is not False
        or prompt.get("special_tokens_added") is not False
        or not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in hashes.items()
        )
    ):
        raise ValueError("document prompt needs exact token IDs, counts and source SHA-256 hashes")
    return prompt["token_ids"]


def _fidelity(check, tolerance, name):
    check = _mapping(check, name)
    if (
        check.get("passed") is not True
        or not _finite(check.get("max_absolute_error"))
        or any(check.get(key) != value for key, value in tolerance.items())
    ):
        raise ValueError(f"{name} must pass the benchmark's recorded precision tolerance")
    return check


def _cache(check, count, layers, full, tolerance, position=None):
    cache = _fidelity(check, tolerance, "cache validation")
    if (
        type(cache.get("layers_checked")) is not int
        or cache["layers_checked"] != layers
        or type(cache.get("valid_tokens_checked")) is not int
        or cache["valid_tokens_checked"] != count
        or cache.get("keys_and_values_checked") is not True
        or cache.get("full_prefix_checked") is not full
        or (not full and cache.get("positions_checked") != [position])
    ):
        raise ValueError(
            "complete layer/key/value coverage and the required cache check scope are mandatory"
        )


def _correctness(
    evidence,
    prompt,
    output_count,
    chunk,
    geometry,
    interval,
    dtype,
    sequential=False,
    require_replay=False,
):
    evidence = _mapping(evidence, "correctness evidence")
    generated = evidence.get("generated_token_ids")
    checks = evidence.get("checks")
    prefix = len(prompt) - int(sequential)
    boundaries = [min(start + chunk, prefix) for start in range(0, prefix, chunk)]
    if (
        evidence.get("passed") is not True
        or type(evidence.get("actual_output_tokens")) is not int
        or evidence["actual_output_tokens"] != output_count
        or not _tokens(generated, output_count, geometry["vocab_size"])
        or not isinstance(checks, list)
        or len(checks) != len(boundaries) + output_count
    ):
        raise ValueError("complete chunk-boundary and generation correctness evidence is required")
    logit_tolerance, cache_tolerance = _fidelity_tolerances(dtype)
    for boundary, check in zip(boundaries, checks[: len(boundaries)], strict=True):
        check = _mapping(check, "prefill check")
        if (
            check.get("stage") != "prefill"
            or check.get("passed") is not True
            or check.get("valid_tokens") != boundary
        ):
            raise ValueError("every prefill chunk boundary must pass in order")
        _cache(check.get("cache"), boundary, geometry["num_hidden_layers"], True, cache_tolerance)
    for offset, check in enumerate(checks[len(boundaries) :]):
        check = _mapping(check, "generation check")
        position = len(prompt) - 1 + offset
        token = prompt[-1] if offset == 0 else generated[offset - 1]
        selected = generated[offset]
        if (
            check.get("stage") != "generation"
            or check.get("passed") is not True
            or type(check.get("position")) is not int
            or check["position"] != position
            or type(check.get("input_token")) is not int
            or check["input_token"] != token
            or type(check.get("gpu_greedy_token")) is not int
            or check["gpu_greedy_token"] != selected
        ):
            raise ValueError("every generated step must validate its position, input and GPU token")
        logits = _fidelity(check.get("logits"), logit_tolerance, "logit validation")
        if (
            logits.get("greedy_token_equal") is not True
            or not _tokens(
                [logits.get("mlx_next_token"), logits.get("metile_next_token")],
                2,
                geometry["vocab_size"],
            )
            or logits["mlx_next_token"] != selected
            or logits["metile_next_token"] != selected
        ):
            raise ValueError("full-logit validation and all greedy token comparisons must pass")
        full = offset == 0 or (offset + 1) % interval == 0 or offset == output_count - 1
        _cache(
            check.get("cache"),
            position + 1 if full else 1,
            geometry["num_hidden_layers"],
            full,
            cache_tolerance,
            position,
        )
    modern = require_replay or any(
        name in evidence for name in ("whole_prompt_replay", "validation_counts")
    )
    if modern:
        counts = _mapping(evidence.get("validation_counts"), "validation counts")
        expected_counts = {
            "chunk_boundary_cache_checks": len(boundaries),
            "whole_prompt_prefill_calls": int(not sequential),
            "generation_full_logit_checks": output_count,
            "generation_cache_checks": output_count,
            "generation_gpu_greedy_checks": output_count,
            "post_first_token_decode_forwards": output_count - 1,
        }
        if any(
            type(counts.get(name)) is not int or counts[name] != count
            for name, count in expected_counts.items()
        ):
            raise ValueError("validation counts must match the actual independent checks")
        if sequential:
            if "whole_prompt_replay" in evidence:
                raise ValueError("a sequential baseline cannot claim chunked whole-prompt replay")
        else:
            replay = _mapping(evidence.get("whole_prompt_replay"), "whole-prompt replay")
            if (
                replay.get("passed") is not True
                or replay.get("shares_first_generation_check") is not True
                or any(
                    type(replay.get(name)) is not int or replay[name] != expected
                    for name, expected in (
                        ("prefill_calls", 1),
                        ("actual_prompt_tokens", len(prompt)),
                        ("generation_check_index", len(boundaries)),
                    )
                )
                or checks[len(boundaries)].get("prefill_mode") != "single_whole_prompt_call"
            ):
                raise ValueError("whole-prompt replay must reference its first generation check")
    return generated


def _observations(result, trials, expected, sequential):
    samples = result.get("samples")
    if not isinstance(samples, list) or len(samples) != trials:
        raise ValueError("raw observation counts must match the phase's trial count")
    labels = ["MLX", "chunked"] + (["sequential"] if sequential else [])
    for index, sample in enumerate(samples):
        sample = _mapping(sample, "trial")
        if type(sample.get("trial")) is not int or sample["trial"] != index:
            raise ValueError("trials need consecutive, distinct identifiers")
        if set(sample) != {"trial", "order", *labels}:
            raise ValueError("trial arms must match the declared phase and optional baseline")
        for label in labels:
            observation = _mapping(sample.get(label), "trial arm")
            if (
                type(observation.get("actual_output_tokens")) is not int
                or observation["actual_output_tokens"] != len(expected)
                or observation.get("generated_token_ids") != expected
            ):
                raise ValueError(
                    "every measured output count and token trajectory must match validation"
                )
            for metric in (
                "time_to_first_token_seconds",
                "decode_wall_seconds",
                "total_wall_seconds",
            ):
                value = observation.get(metric)
                if (
                    not _finite(value)
                    or value <= 0
                    or not math.isfinite((len(expected) - 1) / value)
                ):
                    raise ValueError("timings and derived rates must be finite and positive")
        for metric in ("time_to_first_token_seconds", "decode_wall_seconds", "total_wall_seconds"):
            ratio = sample["MLX"][metric] / sample["chunked"][metric]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("paired speed ratios must be finite and positive")
    summary = _summarize(samples)
    metrics = {}
    for key, _, _ in METRICS:
        source = "decode_wall_seconds" if key == "decode_tokens_per_second" else key
        metrics[key] = {
            "paired_speedup": summary["paired_speedups"]["chunked_over_mlx"][source],
            "arms": [
                {
                    "label": label,
                    "color": color,
                    "values": [
                        (len(expected) - 1) / sample[backend][source]
                        if key == "decode_tokens_per_second"
                        else sample[backend][source]
                        for sample in samples
                    ],
                }
                for backend, label, color in ARMS
            ],
        }
    return metrics


def chart_data(report):
    """Separate selection from held-out evidence and reject incomplete validation."""
    report = _mapping(report, "report")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("status") != "ok"
    ):
        raise ValueError("only successful schema-1 reports can be plotted")
    if not isinstance(report.get("model"), str) or not report["model"]:
        raise ValueError("model checkpoint identity is required")
    weights = _mapping(report.get("weights"), "weights")
    dtype = weights.get("both_backends")
    if (
        dtype not in ("float16", "float32")
        or weights.get("same_values") is not True
        or weights.get("quantized") is not False
    ):
        raise ValueError("identical dense weight values and storage precision are required")
    software = _mapping(report.get("software"), "software")
    if dtype == "float32" and software.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("FP32 comparisons require MLX TF32 disabled")
    precision = _mapping(report.get("precision_comparison"), "precision comparison")
    if precision.get("storage_dtype") != dtype or precision.get("accumulation_dtype") != "float32":
        raise ValueError("storage and accumulation precision metadata must match")
    selection = _mapping(report.get("selection"), "selection")
    workload = _mapping(report.get("workload"), "workload")
    policy = _mapping(report.get("correctness_policy"), "correctness policy")
    execution = _mapping(report.get("execution"), "execution")
    chunks = selection.get("candidate_chunk_sizes")
    if (
        not isinstance(chunks, list)
        or not chunks
        or any(not _integer(chunk) for chunk in chunks)
        or len(set(chunks)) != len(chunks)
    ):
        raise ValueError("candidate chunk sizes must be distinct positive integers")
    if selection.get("held_out_used_for_selection") is not False:
        raise ValueError("held-out measurements must not choose the chunk size")
    if (
        type(workload.get("batch")) is not int
        or workload["batch"] != 1
        or execution.get("not_a_single_dispatch_megakernel") is not True
    ):
        raise ValueError("this chart requires batch-one multi-dispatch generation")
    for count in (selection.get("tuning_trials"), workload.get("trials")):
        if not _integer(count, 5):
            raise ValueError("each phase requires at least five paired trials")
    for count in (selection.get("tuning_output_tokens"), workload.get("requested_output_tokens")):
        if not _integer(count, 2):
            raise ValueError("each phase requires at least two output tokens")
    interval = policy.get("full_prefix_check_interval_generated_tokens")
    if not _integer(interval):
        raise ValueError("a complete-cache verification interval is required")
    if type(workload.get("sequential_baseline_included")) is not bool:
        raise ValueError("optional sequential-baseline inclusion must be explicit")
    held_out = report.get("held_out")
    if not isinstance(held_out, list) or len(held_out) != 1:
        raise ValueError("exactly one independent held-out result is required")
    held = _mapping(held_out[0], "held-out result")
    geometry = _mapping(held.get("geometry"), "held-out geometry")
    if (
        geometry.get("storage_dtype") != dtype
        or not _integer(geometry.get("num_hidden_layers"))
        or not _integer(geometry.get("vocab_size"))
    ):
        raise ValueError("held-out layer count, vocabulary and storage dtype are required")
    packed_weights = _weight_packing(report, held, geometry)
    _execution_metadata(execution, geometry)
    if (
        "prefill_prunes_unused_final_layer" in geometry
        and type(geometry["prefill_prunes_unused_final_layer"]) is not bool
    ):
        raise ValueError("final-layer pruning metadata must be boolean")
    require_replay = "whole_prompt_replay" in policy or geometry.get(
        "prefill_prunes_unused_final_layer", False
    )
    tuning_prompt = _prompt(selection.get("tuning_prompt"), geometry["vocab_size"])
    held_prompt = _prompt(workload.get("prompt"), geometry["vocab_size"])
    _require_disjoint_prompts(selection["tuning_prompt"], workload["prompt"])
    tuning_output, held_output = (
        selection["tuning_output_tokens"],
        workload["requested_output_tokens"],
    )
    if workload.get("candidate_cache_capacity") != max(
        len(tuning_prompt) + tuning_output - 1, len(held_prompt) + held_output - 1
    ):
        raise ValueError("cache capacity must cover both requested workloads")
    tuning = report.get("tuning")
    if not isinstance(tuning, list) or len(tuning) != len(chunks):
        raise ValueError("all configured chunk sizes need successful tuning evidence")
    tuning_rows, tuning_tokens = [], None
    for chunk, result in zip(chunks, tuning, strict=True):
        result = _mapping(result, "tuning result")
        if result.get("chunk_size") != chunk or result.get("status") != "ok":
            raise ValueError("each chunk must have exactly one successful ordered tuning result")
        generated = _correctness(
            result.get("correctness"),
            tuning_prompt,
            tuning_output,
            chunk,
            geometry,
            interval,
            dtype,
            require_replay=require_replay,
        )
        if tuning_tokens is not None and tuning_tokens != generated:
            raise ValueError("all tuning candidates must produce the same validated tokens")
        tuning_tokens = generated
        metrics = _observations(result, selection["tuning_trials"], generated, False)
        tuning_rows.append({"chunk_size": chunk, "metrics": metrics})
    selected = min(
        tuning_rows,
        key=lambda row: (
            statistics.median(row["metrics"]["time_to_first_token_seconds"]["arms"][1]["values"]),
            row["chunk_size"],
        ),
    )["chunk_size"]
    if (
        type(report.get("selected_chunk_size")) is not int
        or report["selected_chunk_size"] != selected
        or held.get("chunk_size") != selected
        or held.get("status") != "ok"
        or geometry.get("prefill_chunk_size") != selected
    ):
        raise ValueError(
            "selected chunk must match recomputed tuning winner and validated held-out configuration"
        )
    generated = _correctness(
        held.get("correctness"),
        held_prompt,
        held_output,
        selected,
        geometry,
        interval,
        dtype,
        require_replay=require_replay,
    )
    sequential = workload["sequential_baseline_included"]
    if (
        sequential
        and _correctness(
            held.get("sequential_correctness"),
            held_prompt,
            held_output,
            selected,
            geometry,
            interval,
            dtype,
            True,
            require_replay=require_replay,
        )
        != generated
    ):
        raise ValueError("optional sequential baseline must match the held-out token trajectory")
    metrics = _observations(held, workload["trials"], generated, sequential)
    return {
        "tuning": tuning_rows,
        "held_out": metrics,
        "selected_chunk_size": selected,
        "packed_weights": packed_weights,
    }


def _points(axis, values, position, color):
    jitter = [0.13 * (index / (len(values) - 1) - 0.5) for index in range(len(values))]
    dots = axis.scatter(
        values,
        [position + offset for offset in jitter],
        s=25,
        color=color,
        edgecolors=style.SURFACE,
        linewidths=0.5,
        zorder=3,
    )
    dots.set_gid("recorded-trials")
    median = statistics.median(values)
    bar = axis.vlines(median, position - 0.14, position + 0.14, color=color, linewidth=2, zorder=4)
    bar.set_gid("sample-median")
    return median


def render(report, output, source_name=DEFAULT_INPUT.name):
    data = chart_data(report)
    pyplot = style.matplotlib_pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    tune_height = max(2.7, 0.69 * len(data["tuning"]))
    height = tune_height + 11.2 + (0.55 if data["packed_weights"] else 0)
    figure = pyplot.figure(figsize=(style.WIDTH, height), dpi=style.DPI)
    tuning_bottom = height - 1.95 - tune_height
    axis = figure.add_axes((0.26, tuning_bottom / height, 0.45, tune_height / height))
    transform = blended_transform_factory(figure.transFigure, axis.transData)
    values = [
        value
        for row in data["tuning"]
        for arm in row["metrics"]["time_to_first_token_seconds"]["arms"]
        for value in arm["values"]
    ]
    axis.set_xlim(0, max(values) * 1.1)
    axis.set_ylim(len(data["tuning"]) - 0.5, -0.65)
    axis.set_yticks([])
    axis.xaxis.set_major_locator(MaxNLocator(5))
    selection, workload = report["selection"], report["workload"]
    axis.set_title(
        f"Tuning · {selection['tuning_prompt']['actual_prompt_tokens']:,} prompt / {selection['tuning_output_tokens']:,} output",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=17,
    )
    axis.set_xlabel("Time to first token · seconds · lower is faster", fontsize=10)
    for position, row in enumerate(data["tuning"]):
        selected = row["chunk_size"] == data["selected_chunk_size"]
        if selected:
            axis.axhspan(position - 0.43, position + 0.43, color=style.ROW_SURFACE, zorder=0)
        label = f"Chunk {row['chunk_size']:,}" + ("\nSelected on tuning" if selected else "")
        axis.text(
            0.045,
            position,
            label,
            transform=transform,
            fontsize=10,
            va="center",
            color=style.INK,
            fontweight="bold" if selected else "normal",
        )
        metric = row["metrics"]["time_to_first_token_seconds"]
        axis.text(
            0.97,
            position,
            f"{metric['paired_speedup']:.2f}x",
            transform=transform,
            ha="right",
            va="center",
            fontsize=10,
            color=style.INK_SOFT,
        )
        for arm_index, arm in enumerate(metric["arms"]):
            center = position + (arm_index - 0.5) * 0.46
            median = _points(axis, arm["values"], center, arm["color"])
            axis.text(
                0.84,
                center,
                f"{median:,.2f}",
                transform=transform,
                ha="right",
                va="center",
                fontsize=10,
                color=arm["color"],
            )
    _headers(axis, figure)
    style.frame(axis)
    figure.text(
        0.045,
        (tuning_bottom - 0.85) / height,
        f"HELD-OUT · {workload['prompt']['actual_prompt_tokens']:,} prompt / "
        f"{workload['requested_output_tokens']:,} output\n"
        f"Excluded from chunk selection · selected chunk {data['selected_chunk_size']:,}",
        fontsize=12,
        color=style.INK,
        fontweight="bold",
        va="top",
    )
    for index, (key, title, units) in enumerate(METRICS):
        bottom = tuning_bottom - 2.65 - index * 2.0
        axis = figure.add_axes((0.26, bottom / height, 0.45, 0.85 / height))
        transform = blended_transform_factory(figure.transFigure, axis.transData)
        metric = data["held_out"][key]
        values = [value for arm in metric["arms"] for value in arm["values"]]
        axis.set_xlim(0, max(values) * 1.1)
        axis.set_ylim(1.5, -0.5)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=15)
        axis.set_xlabel(units, fontsize=10)
        for position, arm in enumerate(metric["arms"]):
            median = _points(axis, arm["values"], position, arm["color"])
            axis.text(
                0.045,
                position,
                arm["label"],
                transform=transform,
                va="center",
                fontsize=10,
                color=arm["color"],
            )
            axis.text(
                0.84,
                position,
                f"{median:,.2f}",
                transform=transform,
                ha="right",
                va="center",
                fontsize=10,
                color=arm["color"],
            )
        axis.text(
            0.97,
            0.5,
            f"{metric['paired_speedup']:.2f}x",
            transform=transform,
            ha="right",
            va="center",
            fontsize=10,
            color=style.INK_SOFT,
        )
        _headers(axis, figure)
        style.frame(axis)
    hardware, software = report.get("hardware", {}), report["software"]
    chip = hardware.get("device_name", hardware.get("chip", "Apple silicon"))
    dtype = report["weights"]["both_backends"].replace("float", "FP")
    precision = f"identical {dtype} weights" + (" · MLX TF32 disabled" if dtype == "FP32" else "")
    subtitle = (
        f"{report['model']} · FP32 · meTile lossless-packed decode/head weights\n"
        "Native MLX unchanged · TF32 disabled · batch one · GPU greedy selection"
        if data["packed_weights"]
        else f"{report['model']} · {precision} · batch one\n"
        "Technical-document prompts · GPU greedy selection · exact output lengths"
    )
    packing_note = (
        "\nmeTile decode/head weights use exact 16-bit packing; native and prefill weights remain FP32."
        "\nActivations, KV cache and accumulation stay FP32. Original FP32 buffers are retained."
        if data["packed_weights"]
        else ""
    )
    optional = (
        " Optional sequential baseline is validated but not shown."
        if workload["sequential_baseline_included"]
        else ""
    )
    style.headings(
        figure,
        (
            "Matrix-tiled prefill, then real generation"
            if report["execution"].get("candidate_attention_backend") == "matrix"
            else "Chunked prefill, then real generation"
        ),
        subtitle,
        f"{chip} · MLX {software.get('mlx', 'unknown')} · source: {source_name}\n"
        f"Dots: all {selection['tuning_trials']} tuning / {workload['trials']} held-out trials per arm. "
        "Bars: medians. No confidence intervals.\n"
        "Phases are separate: minimum tuning median TTFT selects the chunk; held-out results do not affect selection.\n"
        "VS MLX: median paired time ratio; 1.00x is parity, above 1.00x favors meTile.\n"
        "Decode rate = (output count - 1) / decode time. Native batched prefill versus compiler-tiled meTile chunks.\n"
        "Includes cache reset/construction, submission and token retrieval; excludes compilation and tokenization.\n"
        "meTile reuses prepared buffers; native MLX manages allocations.\n"
        "MLX baseline: synchronous model loop, not pipelined mlx_lm.stream_generate.\n"
        "Held out from chunk selection only; this document also informed implementation diagnostics."
        f"{optional}{packing_note}",
    )
    handles = [
        Line2D([], [], marker="o", color=color, linestyle="none", label=label, markersize=6)
        for _, label, color in ARMS
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 1 - 1.08 / height),
        ncol=2,
        borderaxespad=0,
    )
    style.save(figure, output)
    pyplot.close(figure)


def _headers(axis, figure):
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(figure.transFigure, axis.transAxes)
    for column, text in ((0.84, "MEDIAN"), (0.97, "VS MLX")):
        axis.text(
            column,
            1.11,
            text,
            transform=transform,
            fontsize=9,
            ha="right",
            va="bottom",
            color=style.INK_MUTED,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    render(json.loads(arguments.input.read_text()), arguments.output, arguments.input.name)


if __name__ == "__main__":
    main()
