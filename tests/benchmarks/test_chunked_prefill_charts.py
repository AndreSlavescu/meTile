"""Synthetic fixtures exercise chart provenance; no generated fixture is benchmark evidence."""

import statistics
from copy import deepcopy

import pytest

from benchmarks.megakernels.qwen3 import _fidelity_tolerances
from benchmarks.megakernels.qwen3_chunked_prefill import _trial_order
from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_chunked_prefill import ARMS, METRICS, chart_data, render


def _cache(count, full, position=None):
    return {
        "passed": True,
        "max_absolute_error": 0.0,
        "rtol": 0.001,
        "atol": 0.001,
        "layers_checked": 2,
        "valid_tokens_checked": count,
        "keys_and_values_checked": True,
        "full_prefix_checked": full,
        **({"positions_checked": [position]} if not full else {}),
    }


def _correctness(prompt, output_count, chunk, interval=64, sequential=False):
    generated = [7200 + index % 50 for index in range(output_count)]
    prefix = len(prompt) - int(sequential)
    checks = []
    for start in range(0, prefix, chunk):
        count = min(start + chunk, prefix)
        checks.append(
            {
                "stage": "prefill",
                "valid_tokens": count,
                "cache": _cache(count, True),
                "passed": True,
            }
        )
    for offset, selected in enumerate(generated):
        position = len(prompt) - 1 + offset
        full = offset == 0 or (offset + 1) % interval == 0 or offset == output_count - 1
        checks.append(
            {
                "stage": "generation",
                "position": position,
                "input_token": prompt[-1] if offset == 0 else generated[offset - 1],
                "gpu_greedy_token": selected,
                "passed": True,
                "logits": {
                    "passed": True,
                    "max_absolute_error": 0.0,
                    "rtol": 0.001,
                    "atol": 0.001,
                    "greedy_token_equal": True,
                    "mlx_next_token": selected,
                    "metile_next_token": selected,
                },
                "cache": _cache(position + 1 if full else 1, full, position),
            }
        )
    return {
        "passed": True,
        "checks": checks,
        "generated_token_ids": generated,
        "actual_output_tokens": output_count,
    }


def _result(prompt, outputs, chunk, candidate_ttft):
    correctness = _correctness(prompt, outputs, chunk)
    samples = []
    for trial in range(5):
        sample = {"trial": trial, "order": _trial_order(trial)}
        for backend, _, _ in ARMS:
            ttft = (candidate_ttft if backend == "chunked" else 0.7) + trial * 0.01
            decode = (outputs - 1) / ((39 if backend == "chunked" else 40) + trial * 0.2)
            sample[backend] = {
                "time_to_first_token_seconds": ttft,
                "decode_wall_seconds": decode,
                "total_wall_seconds": ttft + decode,
                "actual_output_tokens": outputs,
                "generated_token_ids": list(correctness["generated_token_ids"]),
            }
        samples.append(sample)
    return {
        "chunk_size": chunk,
        "status": "ok",
        "correctness": correctness,
        "samples": samples,
        "medians": {"chunked": {"time_to_first_token_seconds": -123}},
        "paired_speedups": {},
    }


@pytest.fixture
def report():
    tuning_prompt = list(range(2048))
    held_prompt = list(range(3000, 7096))
    held = _result(held_prompt, 1024, 256, 1.8)
    held["geometry"] = {
        "storage_dtype": "float32",
        "num_hidden_layers": 2,
        "vocab_size": 8192,
        "prefill_chunk_size": 256,
    }

    def prompt(tokens, source, digest):
        return {
            "actual_prompt_tokens": len(tokens),
            "token_ids": tokens,
            "source_token_count": len(tokens) + 10,
            "token_offset": 0,
            "chat_template": False,
            "special_tokens_added": False,
            "sources_sha256": {source: digest * 64},
        }

    return {
        "schema_version": 1,
        "status": "ok",
        "model": "Qwen/Qwen3-0.6B",
        "weights": {"both_backends": "float32", "same_values": True, "quantized": False},
        "software": {"MLX_ENABLE_TF32": "0", "mlx": "test"},
        "hardware": {"device_name": "Synthetic test fixture"},
        "precision_comparison": {
            "same_weight_representation": True,
            "storage_dtype": "float32",
            "accumulation_dtype": "float32",
        },
        "execution": {"not_a_single_dispatch_megakernel": True},
        "selection": {
            "candidate_chunk_sizes": [128, 256, 512, 1024],
            "held_out_used_for_selection": False,
            "tuning_trials": 5,
            "tuning_output_tokens": 8,
            "tuning_prompt": prompt(tuning_prompt, "unit-test-tuning.rst", "a"),
        },
        "workload": {
            "batch": 1,
            "trials": 5,
            "requested_output_tokens": 1024,
            "candidate_cache_capacity": 5119,
            "sequential_baseline_included": False,
            "prompt": prompt(held_prompt, "unit-test-held-out.rst", "b"),
        },
        "correctness_policy": {"full_prefix_check_interval_generated_tokens": 64},
        "tuning": [
            _result(tuning_prompt, 8, chunk, ttft)
            for chunk, ttft in zip((128, 256, 512, 1024), (1.2, 0.8, 0.9, 1.0), strict=True)
        ],
        "selected_chunk_size": 256,
        "held_out": [held],
    }


def test_phases_keep_separate_raw_observations_and_recompute_selection(report):
    original = deepcopy(report)
    data = chart_data(report)
    assert data["selected_chunk_size"] == 256
    assert [row["chunk_size"] for row in data["tuning"]] == [128, 256, 512, 1024]
    for row, result in zip(data["tuning"], report["tuning"], strict=True):
        metric = row["metrics"]["time_to_first_token_seconds"]
        for arm, (backend, _, _) in zip(metric["arms"], ARMS, strict=True):
            assert arm["values"] == [
                sample[backend]["time_to_first_token_seconds"] for sample in result["samples"]
            ]
    for key, _, _ in METRICS:
        metric = data["held_out"][key]
        source = "decode_wall_seconds" if key == "decode_tokens_per_second" else key
        samples = report["held_out"][0]["samples"]
        assert metric["paired_speedup"] == statistics.median(
            sample["MLX"][source] / sample["chunked"][source] for sample in samples
        )
        for arm, (backend, _, _) in zip(metric["arms"], ARMS, strict=True):
            expected = [
                1023 / sample[backend][source]
                if key == "decode_tokens_per_second"
                else sample[backend][source]
                for sample in samples
            ]
            assert arm["values"] == expected
    assert report == original


@pytest.mark.parametrize("status", [None, "validation_failed", "validated"])
def test_failed_or_unmeasured_reports_never_produce_charts(report, status, tmp_path):
    report["status"] = status
    output = tmp_path / "not-created.png"
    with pytest.raises(ValueError, match="successful"):
        render(report, output)
    assert not output.exists() and not output.with_suffix(".svg").exists()


@pytest.mark.parametrize("phase", ["tuning", "held_out"])
@pytest.mark.parametrize("status", ["validated", "validation_failed"])
def test_each_phase_requires_successful_measurement(report, phase, status):
    report[phase][0]["status"] = status
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "selected", "configuration"])
def test_every_chunk_and_selected_configuration_are_verified(report, mutation):
    if mutation == "missing":
        report["tuning"].pop()
    elif mutation == "duplicate":
        report["tuning"][1] = deepcopy(report["tuning"][0])
    elif mutation == "selected":
        report["selected_chunk_size"] = 512
    else:
        report["held_out"][0]["geometry"]["prefill_chunk_size"] = 128
    with pytest.raises(ValueError):
        chart_data(report)


def test_held_out_cannot_be_used_for_chunk_selection(report):
    report["selection"]["held_out_used_for_selection"] = True
    with pytest.raises(ValueError, match="must not choose"):
        chart_data(report)


def test_selection_recomputed_from_raw_trials_not_saved_medians(report):
    report["tuning"][0]["medians"]["chunked"]["time_to_first_token_seconds"] = 0
    assert chart_data(report)["selected_chunk_size"] == 256
    for sample in report["tuning"][0]["samples"]:
        previous = sample["chunked"]["time_to_first_token_seconds"]
        sample["chunked"]["time_to_first_token_seconds"] = 0.1
        sample["chunked"]["total_wall_seconds"] += 0.1 - previous
    with pytest.raises(ValueError, match="recomputed tuning winner"):
        chart_data(report)


@pytest.mark.parametrize("collision", ["path", "content", "tokens"])
def test_tuning_and_held_out_documents_are_disjoint(report, collision):
    tuning = report["selection"]["tuning_prompt"]
    held = report["workload"]["prompt"]
    if collision == "path":
        held["sources_sha256"] = {"unit-test-tuning.rst": "b" * 64}
    elif collision == "content":
        held["sources_sha256"] = {"unit-test-held-out.rst": "a" * 64}
    else:
        report["selection"]["tuning_prompt"] = deepcopy(held)
        report["selection"]["tuning_prompt"]["sources_sha256"] = tuning["sources_sha256"]
    with pytest.raises(ValueError, match=r"disjoint|duplicate|sequences must differ"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value", [("same_values", False), ("quantized", True), ("both_backends", "bfloat16")]
)
def test_mismatched_weight_representation_is_rejected(report, field, value):
    report["weights"][field] = value
    with pytest.raises(ValueError, match="identical dense"):
        chart_data(report)


def test_tf32_must_be_explicitly_disabled(report):
    report["software"].pop("MLX_ENABLE_TF32")
    with pytest.raises(ValueError, match="TF32 disabled"):
        chart_data(report)


def _add_weight_packing(report):
    held = report["held_out"][0]
    held["geometry"]["hidden_size"] = 1024
    held["parameter_count"] = 1049600
    packing = {
        "enabled": True,
        "format": "fp32_high16x2_u32_lossless",
        "decoded_dtype": "float32",
        "storage_dtype": "uint32",
        "values_per_word": 2,
        "scopes": ["layer_weights", "embedding"],
        "consumers": ["decode_projections", "first_token_vocabulary_projection"],
        "element_count": 1048576,
        "unpacked_bytes": 4194304,
        "packed_bytes": 2097152,
        "verification": {
            "finite": True,
            "even_elements": True,
            "zero_low16_bits": True,
            "roundtrip_bitwise": True,
        },
        "original_buffers_retained": True,
    }
    report["candidate_weight_packing"] = packing
    held["geometry"]["decode_weight_packing"] = deepcopy(packing)
    report["weights"].update(
        dtype_scope="decoded_weight_values",
        native_projection_storage="float32",
        candidate_prefill_projection_storage="float32",
        candidate_decode_projection_storage="packed_uint32_bfloat16_pairs",
    )
    report["precision_comparison"].update(
        **{
            "class": "lossless_weight_storage",
            "same_weight_representation": False,
            "same_weight_values": True,
            "bitwise_exact": False,
            "storage_dtype_scope": "activations_and_kv_cache",
        }
    )


def test_verified_lossless_weight_packing_preserves_report_and_raw_metrics(report):
    before = chart_data(report)
    _add_weight_packing(report)
    original = deepcopy(report)
    after = chart_data(report)
    assert after == {**before, "packed_weights": True}
    assert report == original


@pytest.mark.parametrize("location", ["report", "geometry"])
@pytest.mark.parametrize("mutation", ["missing", "none", "list"])
def test_weight_packing_requires_both_report_and_geometry_evidence(report, location, mutation):
    _add_weight_packing(report)
    metadata = report if location == "report" else report["held_out"][0]["geometry"]
    key = "candidate_weight_packing" if location == "report" else "decode_weight_packing"
    if mutation == "missing":
        metadata.pop(key)
    else:
        metadata[key] = None if mutation == "none" else []
    with pytest.raises(ValueError, match="weight packing"):
        chart_data(report)


@pytest.mark.parametrize("location", ["report", "geometry", "both"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", False),
        ("enabled", 1),
        ("format", "bfloat16"),
        ("decoded_dtype", "float16"),
        ("storage_dtype", "float32"),
        ("values_per_word", 2.0),
        ("values_per_word", 1),
        ("scopes", ["layer_weights"]),
        ("consumers", ["decode_projections"]),
        ("original_buffers_retained", False),
        ("original_buffers_retained", 1),
    ],
)
def test_weight_packing_format_and_scope_fail_closed(report, location, field, value):
    _add_weight_packing(report)
    if location in ("report", "both"):
        report["candidate_weight_packing"][field] = value
    if location in ("geometry", "both"):
        report["held_out"][0]["geometry"]["decode_weight_packing"][field] = value
    with pytest.raises(ValueError, match="weight packing"):
        chart_data(report)


@pytest.mark.parametrize(
    "check", ["finite", "even_elements", "zero_low16_bits", "roundtrip_bitwise"]
)
@pytest.mark.parametrize("value", [False, 1, None])
def test_each_weight_packing_verification_must_be_explicitly_true(report, check, value):
    _add_weight_packing(report)
    for metadata in (
        report["candidate_weight_packing"],
        report["held_out"][0]["geometry"]["decode_weight_packing"],
    ):
        metadata["verification"][check] = value
    with pytest.raises(ValueError, match="verification"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("element_count", 1048575),
        ("element_count", 1048578),
        ("element_count", 1048576.0),
        ("element_count", True),
        ("unpacked_bytes", 4194304.0),
        ("unpacked_bytes", 2097152),
        ("packed_bytes", 2097152.0),
        ("packed_bytes", 4194304),
    ],
)
def test_weight_packing_counts_must_match_model_and_exact_byte_ratio(report, field, value):
    _add_weight_packing(report)
    report["candidate_weight_packing"][field] = value
    report["held_out"][0]["geometry"]["decode_weight_packing"][field] = value
    with pytest.raises(ValueError, match="element and byte counts"):
        chart_data(report)


@pytest.mark.parametrize("field", ["parameter_count", "hidden_size"])
@pytest.mark.parametrize("value", [None, True, 0, 1024.0])
def test_packed_model_dimensions_and_parameter_count_are_required(report, field, value):
    _add_weight_packing(report)
    held = report["held_out"][0]
    metadata = held if field == "parameter_count" else held["geometry"]
    metadata[field] = value
    with pytest.raises(ValueError, match="model dimensions and parameter counts"):
        chart_data(report)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("weights", "dtype_scope", "physical_storage"),
        ("weights", "native_projection_storage", "bfloat16"),
        ("weights", "candidate_prefill_projection_storage", "uint32"),
        ("weights", "candidate_decode_projection_storage", "float32"),
        ("precision_comparison", "class", "same_storage_precision"),
        ("precision_comparison", "same_weight_representation", True),
        ("precision_comparison", "same_weight_representation", 0),
        ("precision_comparison", "same_weight_values", False),
        ("precision_comparison", "same_weight_values", 1),
        ("precision_comparison", "bitwise_exact", True),
        ("precision_comparison", "bitwise_exact", 0),
        ("precision_comparison", "storage_dtype_scope", "weights"),
    ],
)
def test_packing_requires_honest_native_storage_and_arithmetic_precision(
    report, section, field, value
):
    _add_weight_packing(report)
    report[section][field] = value
    with pytest.raises(ValueError, match="unchanged FP32 arithmetic scope"):
        chart_data(report)


def test_packed_report_cannot_claim_half_precision_values(report):
    _add_device_memory_metadata(report, dtype="float16")
    _add_weight_packing(report)
    with pytest.raises(ValueError, match="unchanged FP32 arithmetic scope"):
        chart_data(report)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_legacy_and_unpacked_reports_keep_identical_storage_labels(report, explicit_none):
    if explicit_none:
        report["candidate_weight_packing"] = None
    assert chart_data(report)["packed_weights"] is False


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("precision_comparison", "same_weight_representation", False),
        ("precision_comparison", "same_weight_representation", 1),
        ("precision_comparison", "class", "lossless_weight_storage"),
        ("weights", "candidate_decode_projection_storage", "packed_uint32_bfloat16_pairs"),
    ],
)
def test_absent_packing_evidence_cannot_hide_different_weight_storage(
    report, section, field, value
):
    report[section][field] = value
    with pytest.raises(ValueError, match="unpacked weights"):
        chart_data(report)


def _add_execution_metadata(report, attention_backend="matrix", projection_backend="tensor_ops"):
    report["execution"].update(
        candidate_attention_backend=attention_backend,
        candidate_projection_backend=projection_backend,
        candidate_projection_tile=[64, 64, 64],
        candidate_strict_math=True,
        candidate_projection_relaxed_precision=False,
    )
    geometry = report["held_out"][0]["geometry"]
    geometry.update(
        prefill_projection_backend=projection_backend,
        prefill_projection_tile=[64, 64, 64],
        prefill_strict_math=True,
        prefill_projection_relaxed_precision=False,
        prefill_attention=(
            {
                "kind": "matrix_tiled_online_softmax",
                "compiler_backend": "simdgroup_inline",
                "query_rows_per_threadgroup": 32,
                "key_tile": 16,
                "threads": 128,
                "shared_padding": 0,
                "unroll_mma": True,
                "transpose_keys": True,
                "register_stats": True,
                "unroll_softmax": True,
                "load_vector": 16,
                "softmax_lanes": 8,
            }
            if attention_backend == "matrix"
            else {"kind": "query_tiled_shared_kv"}
        ),
    )


@pytest.mark.parametrize("attention_backend", ["matrix", "tiled"])
@pytest.mark.parametrize("projection_backend", ["tensor_ops", "simdgroup"])
def test_matching_execution_backend_and_precision_metadata_are_accepted(
    report, attention_backend, projection_backend
):
    _add_execution_metadata(report, attention_backend, projection_backend)
    original = deepcopy(report)
    assert chart_data(report)["selected_chunk_size"] == 256
    assert report == original


def test_matrix_query_reuse_metadata_is_validated(report):
    _add_execution_metadata(report)
    geometry = report["held_out"][0]["geometry"]
    geometry["head_dim"] = 128
    geometry["prefill_attention"].update(cached_query_fragments=16, query_kv_shared_storage=True)
    assert chart_data(report)["selected_chunk_size"] == 256


@pytest.mark.parametrize(
    "field,value",
    [
        ("cached_query_fragments", None),
        ("cached_query_fragments", True),
        ("cached_query_fragments", 0),
        ("cached_query_fragments", 8),
        ("query_kv_shared_storage", None),
        ("query_kv_shared_storage", 1),
        ("query_kv_shared_storage", False),
        ("query_rows_per_threadgroup", 8),
        ("shared_padding", 4),
        ("unroll_mma", False),
    ],
)
def test_inconsistent_matrix_query_reuse_metadata_is_rejected(report, field, value):
    _add_execution_metadata(report)
    geometry = report["held_out"][0]["geometry"]
    geometry["head_dim"] = 128
    geometry["prefill_attention"].update(cached_query_fragments=16, query_kv_shared_storage=True)
    geometry["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="query reuse"):
        chart_data(report)


def _add_device_memory_metadata(report, dtype="float32", threads=128, direct=True):
    _add_execution_metadata(report)
    report["weights"]["both_backends"] = dtype
    report["precision_comparison"]["storage_dtype"] = dtype
    geometry = report["held_out"][0]["geometry"]
    geometry.update(storage_dtype=dtype, head_dim=128)
    geometry["prefill_attention"].update(
        threads=threads,
        query_rows_per_threadgroup=threads // 4 if direct else 32,
        direct_device_memory=direct,
        masked_device_tile_scratch_bytes=threads * 2 * (2 if dtype == "float16" else 4)
        if direct
        else 0,
        cached_query_fragments=16,
        query_kv_shared_storage=not direct,
    )
    logit_tolerance, cache_tolerance = _fidelity_tolerances(dtype)
    for phase in ("tuning", "held_out"):
        for result in report[phase]:
            for check in result["correctness"]["checks"]:
                check["cache"].update(cache_tolerance)
                if "logits" in check:
                    check["logits"].update(logit_tolerance)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("threads", [32, 128])
def test_direct_matrix_memory_metadata_matches_dtype_and_simd_group_count(report, dtype, threads):
    _add_device_memory_metadata(report, dtype, threads)
    original = deepcopy(report)
    assert chart_data(report)["selected_chunk_size"] == 256
    assert report == original


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_explicit_shared_matrix_path_records_zero_device_scratch(report, dtype):
    _add_device_memory_metadata(report, dtype, direct=False)
    assert chart_data(report)["selected_chunk_size"] == 256


@pytest.mark.parametrize(
    "field",
    [
        "direct_device_memory",
        "masked_device_tile_scratch_bytes",
        "cached_query_fragments",
        "query_kv_shared_storage",
    ],
)
def test_direct_matrix_memory_metadata_requires_all_related_fields(report, field):
    _add_device_memory_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"].pop(field)
    with pytest.raises(ValueError, match="device memory metadata"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("direct_device_memory", None),
        ("direct_device_memory", 1),
        ("direct_device_memory", "true"),
        ("direct_device_memory", False),
        ("masked_device_tile_scratch_bytes", None),
        ("masked_device_tile_scratch_bytes", True),
        ("masked_device_tile_scratch_bytes", 1024.0),
        ("masked_device_tile_scratch_bytes", -1),
        ("masked_device_tile_scratch_bytes", 0),
        ("masked_device_tile_scratch_bytes", 512),
    ],
)
def test_direct_matrix_memory_metadata_rejects_inconsistent_or_untyped_scratch(
    report, field, value
):
    _add_device_memory_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="device memory metadata"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_kv_shared_storage", True),
        ("query_kv_shared_storage", 0),
        ("cached_query_fragments", 0),
        ("cached_query_fragments", 8),
        ("cached_query_fragments", 16.0),
        ("cached_query_fragments", True),
    ],
)
def test_direct_matrix_memory_still_requires_cached_queries_without_shared_kv_alias(
    report, field, value
):
    _add_device_memory_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="query reuse"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [("query_rows_per_threadgroup", 8), ("shared_padding", 4), ("unroll_mma", False)],
)
def test_direct_matrix_memory_layout_requires_matching_query_ownership(report, field, value):
    _add_device_memory_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="query ownership"):
        chart_data(report)


@pytest.mark.parametrize("dtype,incorrect_bytes", [("float16", 1024), ("float32", 512)])
def test_direct_matrix_memory_scratch_cannot_describe_another_storage_dtype(
    report, dtype, incorrect_bytes
):
    _add_device_memory_metadata(report, dtype)
    report["held_out"][0]["geometry"]["prefill_attention"]["masked_device_tile_scratch_bytes"] = (
        incorrect_bytes
    )
    with pytest.raises(ValueError, match="device memory metadata"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value", [("direct_device_memory", True), ("masked_device_tile_scratch_bytes", 1024)]
)
def test_direct_matrix_memory_fields_cannot_be_attached_to_a_nonmatrix_backend(
    report, field, value
):
    _add_execution_metadata(report, attention_backend="tiled")
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="requires the matrix attention backend"):
        chart_data(report)


def test_historical_geometry_without_new_execution_metadata_remains_accepted(report):
    report["held_out"][0]["geometry"].update(
        prefill_projection_backend="simdgroup",
        prefill_projection_tile=[64, 64, 32],
        prefill_attention={"kind": "query_tiled_shared_kv"},
    )
    assert chart_data(report)["selected_chunk_size"] == 256


@pytest.mark.parametrize(
    "field",
    [
        "candidate_attention_backend",
        "candidate_projection_backend",
        "candidate_projection_tile",
        "candidate_strict_math",
        "candidate_projection_relaxed_precision",
    ],
)
def test_partial_new_execution_metadata_is_rejected(report, field):
    _add_execution_metadata(report)
    report["execution"].pop(field)
    with pytest.raises(ValueError, match="complete candidate"):
        chart_data(report)


@pytest.mark.parametrize(
    "field",
    [
        "prefill_projection_backend",
        "prefill_projection_tile",
        "prefill_strict_math",
        "prefill_projection_relaxed_precision",
    ],
)
def test_new_execution_metadata_requires_corresponding_held_out_geometry(report, field):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"].pop(field)
    with pytest.raises(ValueError, match="complete candidate"):
        chart_data(report)


def test_execution_projection_tile_cannot_use_float_values_equal_to_geometry_integers(report):
    _add_execution_metadata(report)
    report["execution"]["candidate_projection_tile"] = [64.0, 64, 64]
    with pytest.raises(ValueError, match="projection tile"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("projection_backend", "simdgroup"),
        ("projection_tile", [32, 64, 32]),
        ("strict_math", False),
        ("projection_relaxed_precision", True),
    ],
)
def test_execution_metadata_must_match_geometry(report, field, value):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"][f"prefill_{field}"] = value
    with pytest.raises(ValueError, match="must match"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("projection_backend", "nax"),
        ("projection_tile", [64, 64]),
        ("projection_tile", [64, 64, True]),
        ("projection_tile", [64, 64, 7]),
        ("strict_math", False),
        ("strict_math", 1),
        ("projection_relaxed_precision", True),
        ("projection_relaxed_precision", 0),
    ],
)
def test_matching_but_invalid_execution_contract_is_rejected(report, field, value):
    _add_execution_metadata(report)
    report["execution"][f"candidate_{field}"] = value
    report["held_out"][0]["geometry"][f"prefill_{field}"] = value
    with pytest.raises(ValueError, match="candidate"):
        chart_data(report)


@pytest.mark.parametrize("backend", ["tiled", "unknown", None, {}])
def test_attention_backend_must_match_matrix_geometry(report, backend):
    _add_execution_metadata(report)
    report["execution"]["candidate_attention_backend"] = backend
    with pytest.raises(ValueError, match="attention backend"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "unknown"),
        ("compiler_backend", "tensor_ops"),
        ("query_rows_per_threadgroup", True),
        ("key_tile", 7),
        ("threads", 33),
        ("threads", 2048),
        ("shared_padding", -1),
        ("unroll_mma", 1),
        ("transpose_keys", "true"),
        ("register_stats", 0),
        ("unroll_softmax", None),
        ("softmax_base2", 1),
        ("load_vector", True),
        ("softmax_lanes", 7),
    ],
)
def test_invalid_matrix_layout_metadata_is_rejected(report, field, value):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="attention"):
        chart_data(report)


def test_matrix_layout_validation_is_not_pinned_to_one_measured_configuration(report):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"].update(
        query_rows_per_threadgroup=8,
        key_tile=32,
        shared_padding=4,
        unroll_mma=False,
        transpose_keys=False,
        register_stats=False,
        unroll_softmax=False,
        load_vector=1,
        softmax_lanes=32,
    )
    assert chart_data(report)["selected_chunk_size"] == 256


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("base2", [False, True])
def test_optional_matrix_math_metadata_accepts_both_explicit_paths(report, fused, base2):
    _add_execution_metadata(report)
    geometry = report["held_out"][0]["geometry"]
    geometry["fused_row_arithmetic"] = fused
    geometry["prefill_attention"].update(
        softmax_base2=base2,
        softmax_exponential="fast_exp2" if base2 else "exp",
        denominator_fma=base2,
        normalization="divide" if base2 else "reciprocal_multiply",
    )
    original = deepcopy(report)
    assert chart_data(report)["selected_chunk_size"] == 256
    assert report == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("softmax_exponential", "fast_exp2"),
        ("denominator_fma", True),
        ("normalization", "divide"),
    ],
)
def test_each_optional_matrix_math_field_can_be_recorded_independently(report, field, value):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"].update(
        softmax_base2=True, **{field: value}
    )
    assert chart_data(report)["selected_chunk_size"] == 256


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_fused_row_arithmetic_requires_boolean_even_without_other_metadata(report, modern, value):
    if modern:
        _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["fused_row_arithmetic"] = value
    with pytest.raises(ValueError, match="fused row arithmetic"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("softmax_exponential", None),
        ("softmax_exponential", True),
        ("softmax_exponential", {}),
        ("softmax_exponential", "unknown"),
        ("softmax_exponential", "exp"),
        ("softmax_exponential", "exp2"),
        ("denominator_fma", 1),
        ("denominator_fma", "true"),
        ("denominator_fma", None),
        ("denominator_fma", False),
        ("normalization", None),
        ("normalization", True),
        ("normalization", []),
        ("normalization", "unknown"),
        ("normalization", "reciprocal_multiply"),
    ],
)
def test_matrix_math_metadata_rejects_wrong_types_values_and_base2_inconsistency(
    report, field, value
):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"].update(
        softmax_base2=True, **{field: value}
    )
    with pytest.raises(ValueError, match="matrix attention math"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("softmax_exponential", "fast_exp2"),
        ("denominator_fma", True),
        ("normalization", "divide"),
    ],
)
def test_matrix_math_metadata_rejects_base_two_semantics_on_natural_path(report, field, value):
    _add_execution_metadata(report)
    report["held_out"][0]["geometry"]["prefill_attention"].update(
        softmax_base2=False, **{field: value}
    )
    with pytest.raises(ValueError, match="matrix attention math"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("softmax_exponential", "fast_exp2"),
        ("denominator_fma", True),
        ("normalization", "divide"),
    ],
)
def test_matrix_math_metadata_needs_an_explicit_base_and_matching_backend(report, field, value):
    _add_execution_metadata(report)
    attention = report["held_out"][0]["geometry"]["prefill_attention"]
    attention[field] = value
    with pytest.raises(ValueError, match="explicit softmax base"):
        chart_data(report)
    _add_execution_metadata(report, attention_backend="tiled")
    report["held_out"][0]["geometry"]["prefill_attention"][field] = value
    with pytest.raises(ValueError, match="requires the matrix attention backend"):
        chart_data(report)


@pytest.mark.parametrize("phase", ["tuning", "held_out"])
def test_every_chunk_boundary_and_generation_step_must_be_present(report, phase):
    report[phase][0]["correctness"]["checks"].pop()
    with pytest.raises(ValueError, match="complete chunk-boundary"):
        chart_data(report)


def test_prefill_boundaries_must_cover_full_prefix_in_order(report):
    report["tuning"][0]["correctness"]["checks"][0]["valid_tokens"] = 127
    with pytest.raises(ValueError, match="every prefill chunk boundary"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("layers_checked", 1),
        ("keys_and_values_checked", False),
        ("full_prefix_checked", False),
        ("valid_tokens_checked", 1),
    ],
)
def test_prefill_cache_checks_need_complete_all_layer_coverage(report, field, value):
    report["tuning"][0]["correctness"]["checks"][0]["cache"][field] = value
    with pytest.raises(ValueError, match="coverage"):
        chart_data(report)


@pytest.mark.parametrize("offset", [0, 63, 1023])
def test_required_full_generation_cache_checks_cannot_be_downgraded(report, offset):
    checks = [
        check
        for check in report["held_out"][0]["correctness"]["checks"]
        if check["stage"] == "generation"
    ]
    checks[offset]["cache"] = _cache(1, False, checks[offset]["position"])
    with pytest.raises(ValueError, match="coverage"):
        chart_data(report)


def test_noncheckpoint_generation_checks_require_the_actual_new_cache_slot(report):
    checks = [
        check
        for check in report["held_out"][0]["correctness"]["checks"]
        if check["stage"] == "generation"
    ]
    checks[1]["cache"]["positions_checked"] = [0]
    with pytest.raises(ValueError, match="coverage"):
        chart_data(report)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("logits", "passed", False),
        ("logits", "greedy_token_equal", False),
        ("logits", "mlx_next_token", 0),
        ("logits", "rtol", 0.1),
        ("cache", "passed", False),
        ("cache", "max_absolute_error", float("nan")),
    ],
)
def test_logit_token_cache_and_precision_validation_all_must_pass(report, section, field, value):
    checks = [
        check
        for check in report["held_out"][0]["correctness"]["checks"]
        if check["stage"] == "generation"
    ]
    checks[2][section][field] = value
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize("phase", ["tuning", "held_out"])
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "order", "tokens", "count", "extra_arm"]
)
def test_raw_trials_counts_order_arms_and_exact_tokens_are_checked(report, phase, mutation):
    samples = report[phase][0]["samples"]
    if mutation == "missing":
        samples.pop()
    elif mutation == "duplicate":
        samples[1]["trial"] = 0
    elif mutation == "order":
        samples[1]["order"] = ["MLX", "chunked"]
    elif mutation == "tokens":
        samples[1]["chunked"]["generated_token_ids"][0] = 0
    elif mutation == "count":
        samples[1]["chunked"]["actual_output_tokens"] -= 1
    else:
        samples[0]["unknown"] = {}
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize("value", [None, True, "0.1", 0, -1, float("inf"), float("nan"), 1e-320])
def test_invalid_raw_timing_or_derived_rate_is_rejected(report, value):
    report["tuning"][0]["samples"][0]["chunked"]["time_to_first_token_seconds"] = value
    with pytest.raises(ValueError, match="finite and positive"):
        chart_data(report)


def test_inconsistent_total_duration_is_rejected(report):
    report["held_out"][0]["samples"][0]["MLX"]["total_wall_seconds"] += 1
    with pytest.raises(ValueError, match="TTFT plus decode"):
        chart_data(report)


def _add_sequential(report):
    held = report["held_out"][0]
    report["workload"]["sequential_baseline_included"] = True
    held["sequential_correctness"] = _correctness(
        report["workload"]["prompt"]["token_ids"], 1024, 256, sequential=True
    )
    for sample in held["samples"]:
        sample["order"] = _trial_order(sample["trial"], True)
        sample["sequential"] = deepcopy(sample["chunked"])


def test_optional_sequential_arm_requires_independent_validation(report):
    _add_sequential(report)
    assert chart_data(report)["selected_chunk_size"] == 256
    report["held_out"][0].pop("sequential_correctness")
    with pytest.raises(ValueError, match="correctness evidence"):
        chart_data(report)


@pytest.mark.parametrize("sequential", [False, True])
def test_plot_preserves_all_coordinates_and_separates_tuning_from_held_out(
    report, monkeypatch, tmp_path, sequential
):
    pytest.importorskip("matplotlib")
    if sequential:
        _add_sequential(report)
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    render(report, tmp_path / "synthetic-fixture.png", "unit-test-fixture.json")
    assert len(figures[0].axes) == 4
    data = chart_data(report)
    expected_arms = [
        [
            arm
            for row in data["tuning"]
            for arm in row["metrics"]["time_to_first_token_seconds"]["arms"]
        ]
    ]
    expected_arms.extend(data["held_out"][key]["arms"] for key, _, _ in METRICS)
    observations = 0
    for axis, arms in zip(figures[0].axes, expected_arms, strict=True):
        assert axis.get_xlim()[0] == 0 and axis.get_xscale() == "linear"
        dots = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "recorded-trials"
        ]
        medians = [
            collection for collection in axis.collections if collection.get_gid() == "sample-median"
        ]
        for points, median, arm in zip(dots, medians, arms, strict=True):
            assert points.get_offsets()[:, 0].tolist() == arm["values"]
            assert median.get_segments()[0][:, 0].tolist() == [statistics.median(arm["values"])] * 2
            observations += len(arm["values"])
    assert observations == 70
    prose = "\n".join(text.get_text() for text in figures[0].texts)
    assert "HELD-OUT · 4,096 prompt / 1,024 output" in prose
    assert "Phases are separate" in prose and "held-out results do not affect selection" in prose
    assert "Held out from chunk selection only" in prose
    assert "implementation diagnostics" in prose
    assert "not pipelined mlx_lm.stream_generate" in prose
    assert "identical FP32 weights" in prose and "TF32 disabled" in prose
    assert "No confidence intervals" in prose and "output count - 1" in prose
    assert ("sequential baseline is validated but not shown" in prose) == sequential


def test_exports_contain_source_identity_and_no_density(report, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "synthetic-fixture.png"
    render(report, output, "unit-test-fixture.json")
    assert output.read_bytes().startswith(b"\x89PNG")
    vector = output.with_suffix(".svg").read_text()
    assert "<text" in vector and "unit-test-fixture.json" in vector
    assert "2,048 prompt / 8 output" in vector and "4,096 prompt / 1,024 output" in vector
    assert "observed-range-density" not in vector and "No confidence intervals" in vector


def test_packed_weight_chart_discloses_asymmetric_storage_without_claiming_half_math(
    report, monkeypatch, tmp_path
):
    pytest.importorskip("matplotlib")
    _add_weight_packing(report)
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    render(report, tmp_path / "synthetic-packed-fixture.png", "unit-test-fixture.json")
    prose = "\n".join(text.get_text() for text in figures[0].texts)
    assert "lossless-packed decode/head weights" in prose
    assert "Native MLX unchanged" in prose and "TF32 disabled" in prose
    assert "native and prefill weights remain FP32" in prose
    assert "Activations, KV cache and accumulation stay FP32" in prose
    assert "Original FP32 buffers are retained" in prose
    assert "identical FP32 weights" not in prose


def _add_replay_evidence(report):
    report["correctness_policy"]["whole_prompt_replay"] = "Validate the timed prefill call."
    report["held_out"][0]["geometry"]["prefill_prunes_unused_final_layer"] = True
    for result in [*report["tuning"], *report["held_out"]]:
        for name in ("correctness", "sequential_correctness"):
            if name not in result:
                continue
            evidence = result[name]
            sequential = name == "sequential_correctness"
            boundaries = sum(check["stage"] == "prefill" for check in evidence["checks"])
            outputs = evidence["actual_output_tokens"]
            evidence["validation_counts"] = {
                "chunk_boundary_cache_checks": boundaries,
                "whole_prompt_prefill_calls": int(not sequential),
                "generation_full_logit_checks": outputs,
                "generation_cache_checks": outputs,
                "generation_gpu_greedy_checks": outputs,
                "post_first_token_decode_forwards": outputs - 1,
            }
            if not sequential:
                first = evidence["checks"][boundaries]
                first["prefill_mode"] = "single_whole_prompt_call"
                evidence["whole_prompt_replay"] = {
                    "passed": True,
                    "prefill_calls": 1,
                    "actual_prompt_tokens": first["position"] + 1,
                    "generation_check_index": boundaries,
                    "shares_first_generation_check": True,
                }


@pytest.mark.parametrize("sequential", [False, True])
def test_whole_prompt_replay_reuses_validated_generation_evidence(report, sequential):
    if sequential:
        _add_sequential(report)
    expected = chart_data(report)
    _add_replay_evidence(report)
    assert chart_data(report) == expected


@pytest.mark.parametrize("phase", ["tuning", "held_out"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("prefill_calls", True),
        ("prefill_calls", 2),
        ("actual_prompt_tokens", 0),
        ("generation_check_index", 0),
        ("generation_check_index", 16.0),
        ("shares_first_generation_check", 1),
    ],
)
def test_incorrect_whole_prompt_replay_metadata_is_rejected(report, phase, field, value):
    _add_replay_evidence(report)
    report[phase][0]["correctness"]["whole_prompt_replay"][field] = value
    with pytest.raises(ValueError, match="whole-prompt replay"):
        chart_data(report)


@pytest.mark.parametrize("phase", ["tuning", "held_out"])
@pytest.mark.parametrize("missing", ["whole_prompt_replay", "validation_counts", "prefill_mode"])
def test_pruned_execution_requires_whole_prompt_evidence(report, phase, missing):
    _add_replay_evidence(report)
    report["correctness_policy"].pop("whole_prompt_replay")
    evidence = report[phase][0]["correctness"]
    if missing == "prefill_mode":
        evidence["checks"][evidence["whole_prompt_replay"]["generation_check_index"]].pop(missing)
    else:
        evidence.pop(missing)
    with pytest.raises(ValueError, match=r"whole-prompt replay|validation counts"):
        chart_data(report)


@pytest.mark.parametrize(
    "field",
    [
        "chunk_boundary_cache_checks",
        "whole_prompt_prefill_calls",
        "generation_full_logit_checks",
        "generation_cache_checks",
        "generation_gpu_greedy_checks",
        "post_first_token_decode_forwards",
    ],
)
@pytest.mark.parametrize("value", [True, -1, 1.0])
def test_replay_does_not_inflate_or_mislabel_validation_counts(report, field, value):
    _add_replay_evidence(report)
    report["held_out"][0]["correctness"]["validation_counts"][field] = value
    with pytest.raises(ValueError, match="validation counts"):
        chart_data(report)
