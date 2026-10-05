"""Synthetic scheduler traces test chart integrity, not hardware performance."""

import json
import statistics
from copy import deepcopy

import pytest

from benchmarks.megakernels import qwen3_pd_interleaving as driver
from benchmarks.megakernels.qwen3 import _fidelity_tolerances
from benchmarks.plots import chartstyle as style
from benchmarks.plots.render_pd_interleaving import METRICS, chart_data, render
from tests.benchmarks.test_qwen3_end_to_end import FakeModel
from tests.benchmarks.test_qwen3_end_to_end import fake_mlx as fake_mlx
from tests.benchmarks.test_qwen3_pd_interleaving import Clock, _args, _setup


@pytest.fixture
def report(fake_mlx):
    candidates, prompts, _ = _setup()
    arguments = _args(trials=5)
    validations = {
        policy: driver._validate_policy(
            FakeModel(), candidates, prompts, arguments, policy, lambda: None
        )
        for policy in arguments.policies
    }
    samples = []
    for trial in range(arguments.trials):
        order = driver._order(arguments.policies, trial)
        policies = {}
        for policy in order:
            initial = driver._prime(candidates["A"], prompts["A"], lambda: None)
            timer = Clock()

            def clock(timer=timer, trial=trial):
                return timer() * (trial + 1)

            policies[policy] = driver._execute(
                policy,
                candidates,
                prompts,
                arguments.output_tokens,
                initial,
                lambda: None,
                0.1,
                clock=clock,
            )
        samples.append({"trial": trial, "order": order, "policies": policies})
    medians = {
        policy: {
            metric: statistics.median(
                sample["policies"][policy]["metrics"][metric] for sample in samples
            )
            for metric in samples[0]["policies"][policy]["metrics"]
        }
        for policy in arguments.policies
    }
    ratios = {
        policy: {
            metric: statistics.median(
                sample["policies"]["fifo"]["metrics"][metric]
                / sample["policies"][policy]["metrics"][metric]
                for sample in samples
            )
            for metric in ("makespan_seconds", "b_ttft_seconds", "a_gap_max_seconds")
        }
        for policy in arguments.policies[1:]
    }
    documents = {
        name: {
            "token_ids": prompt,
            "actual_prompt_tokens": len(prompt),
            "source_token_count": len(prompt) + 10,
            "token_offset": 0,
            "chat_template": False,
            "special_tokens_added": False,
            "sources_sha256": {f"synthetic-{name}.rst": name.lower() * 64},
        }
        for name, prompt in prompts.items()
    }
    result = {
        "schema_version": 1,
        "benchmark": "qwen3_pd_interleaving",
        "status": "ok",
        "model": "Synthetic/Qwen3",
        "source_sha256": {"driver.py": "c" * 64},
        "checkpoint_sha256": {"model.safetensors": "d" * 64},
        "tokenizer_sha256": {"tokenizer.json": "e" * 64},
        "compiler_and_kernels_sha256": "f" * 64,
        "hardware": {"device_name": "Synthetic fixture"},
        "software": {"MLX_ENABLE_TF32": "0"},
        "geometry": {
            "storage_dtype": "float32",
            "num_hidden_layers": 2,
            "vocab_size": 13,
            "hidden_size": 4,
            "head_dim": 128,
            "capacity": 8,
            "prefill_chunk_size": 2,
            "prefill_projection_backend": "tensor_ops",
            "prefill_projection_tile": [64, 64, 64],
            "prefill_strict_math": True,
            "prefill_projection_relaxed_precision": False,
            "prefill_attention": {
                "kind": "matrix_tiled_online_softmax",
                "compiler_backend": "simdgroup_inline",
                "query_rows_per_threadgroup": 32,
                "key_tile": 16,
                "threads": 128,
                "shared_padding": 0,
                "unroll_mma": True,
                "cached_query_fragments": 16,
                "query_kv_shared_storage": False,
                "direct_device_memory": True,
                "masked_device_tile_scratch_bytes": 1024,
            },
        },
        "parameter_count": 100,
        "precision": {
            "storage_dtype": "float32",
            "accumulation_dtype": "float32",
            "candidate_weight_packing": None,
            "same_backend_and_weight_storage_across_policies": True,
        },
        "precision_comparison": {
            "class": "same_backend_scheduling",
            "same_weight_representation": True,
            "same_weight_values": True,
            "storage_dtype": "float32",
            "storage_dtype_scope": "activations_and_kv_cache",
            "accumulation_dtype": "float32",
            "bitwise_exact": False,
        },
        "execution": {
            "cache_reset_policy": "zero_fill",
            "cooperative_serial_interleaving": True,
            "hardware_overlap_claimed": False,
            "self_request_prefill_decode_overlap": False,
            "shared_immutable_weights": True,
            "independent_request_cache_and_workspace": True,
        },
        "workload": {
            "requests": documents,
            "output_tokens_per_request": 4,
            "timed_output_tokens": 7,
            "chunk_size": 2,
            "policies": arguments.policies,
            "trials": 5,
            "warmups_per_policy": 1,
        },
        "correctness": {"cache_check_interval": 2, "tolerances": _fidelity_tolerances("float32")},
        "validation": validations,
        "measurement": {"stall_threshold_seconds": 0.1, "paired_baseline": "fifo"},
        "samples": samples,
        "medians": medians,
        "paired_ratios": ratios,
    }
    return json.loads(json.dumps(result))


def _packing(report):
    packing = {
        "enabled": True,
        "format": "fp32_high16x2_u32_lossless",
        "decoded_dtype": "float32",
        "storage_dtype": "uint32",
        "values_per_word": 2,
        "scopes": ["layer_weights", "embedding"],
        "consumers": ["decode_projections", "first_token_vocabulary_projection"],
        "original_buffers_retained": True,
        "element_count": 96,
        "packed_bytes": 192,
        "unpacked_bytes": 384,
        "verification": {
            name: True
            for name in ("finite", "even_elements", "zero_low16_bits", "roundtrip_bitwise")
        },
    }
    report["precision"]["candidate_weight_packing"] = packing
    report["geometry"]["decode_weight_packing"] = deepcopy(packing)


def test_validated_chart_data_preserves_every_raw_policy_trial_without_mutation(report):
    original = deepcopy(report)
    data = chart_data(report)
    assert data["policies"] == ["fifo", "eager", "chunked"]
    for policy in data["policies"]:
        assert data["observations"][policy] == [
            sample["policies"][policy]["metrics"] for sample in report["samples"]
        ]
    assert report == original


@pytest.mark.parametrize("status", [None, "validated", "validation_failed"])
def test_unsuccessful_report_cannot_create_a_chart(report, status, tmp_path):
    report["status"] = status
    output = tmp_path / "absent.png"
    with pytest.raises(ValueError, match="successful"):
        render(report, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["source_sha256", "checkpoint_sha256", "tokenizer_sha256"])
def test_source_checkpoint_and_tokenizer_fingerprints_are_required(report, field):
    report[field] = {}
    with pytest.raises(ValueError, match="SHA-256"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trials", 4),
        ("warmups_per_policy", 0),
        ("timed_output_tokens", 8),
        ("output_tokens_per_request", True),
        ("chunk_size", 3),
    ],
)
def test_declared_workload_counts_cannot_hide_missing_trials_or_the_untimed_token(
    report, field, value
):
    report["workload"][field] = value
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize(
    "field",
    [
        "cooperative_serial_interleaving",
        "shared_immutable_weights",
        "independent_request_cache_and_workspace",
    ],
)
def test_request_execution_contract_cannot_change(report, field):
    report["execution"][field] = False
    with pytest.raises(ValueError, match="independent requests"):
        chart_data(report)


@pytest.mark.parametrize("policy", [None, True, "unknown"])
def test_cache_reset_semantics_must_be_explicit(report, policy):
    report["execution"]["cache_reset_policy"] = policy
    with pytest.raises(ValueError, match="cache reset"):
        chart_data(report)


def test_explicit_logical_cache_invalidation_is_accepted(report):
    report["execution"]["cache_reset_policy"] = "logical"
    assert chart_data(report)["policies"] == report["workload"]["policies"]


@pytest.mark.parametrize(
    "field", ["hardware_overlap_claimed", "self_request_prefill_decode_overlap"]
)
def test_chart_rejects_claimed_hardware_or_self_request_overlap(report, field):
    report["execution"][field] = True
    with pytest.raises(ValueError, match="cooperative serial"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("same_weight_representation", False),
        ("same_weight_values", 1),
        ("bitwise_exact", True),
        ("storage_dtype", "float16"),
        ("class", "lossless_weight_storage"),
    ],
)
def test_policy_precision_contract_is_strict(report, field, value):
    report["precision_comparison"][field] = value
    with pytest.raises(ValueError, match="same physical weights"):
        chart_data(report)


@pytest.mark.parametrize(
    "field,value", [("prefill_strict_math", False), ("prefill_projection_relaxed_precision", True)]
)
def test_relaxed_candidate_arithmetic_is_rejected(report, field, value):
    report["geometry"][field] = value
    with pytest.raises(ValueError, match="strict math"):
        chart_data(report)


def test_native_tf32_must_remain_disabled_for_correctness_reference(report):
    report["software"]["MLX_ENABLE_TF32"] = "1"
    with pytest.raises(ValueError, match="strict native"):
        chart_data(report)


def test_verified_lossless_packing_is_same_storage_across_scheduling_policies(report):
    _packing(report)
    assert chart_data(report)["packed_weights"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", 1),
        ("packed_bytes", 384),
        ("element_count", 98),
        ("decoded_dtype", "float16"),
        ("original_buffers_retained", False),
        ("consumers", ["decode_projections"]),
    ],
)
def test_invalid_or_incomplete_packing_is_rejected_even_if_both_declarations_match(
    report, field, value
):
    _packing(report)
    report["precision"]["candidate_weight_packing"][field] = value
    report["geometry"]["decode_weight_packing"][field] = value
    with pytest.raises(ValueError, match="lossless weight packing"):
        chart_data(report)


@pytest.mark.parametrize(
    "field", ["finite", "even_elements", "zero_low16_bits", "roundtrip_bitwise"]
)
def test_every_packing_verification_is_required(report, field):
    _packing(report)
    report["precision"]["candidate_weight_packing"]["verification"][field] = False
    report["geometry"]["decode_weight_packing"]["verification"][field] = False
    with pytest.raises(ValueError, match="all verification"):
        chart_data(report)


@pytest.mark.parametrize("request_id", ["A", "B"])
@pytest.mark.parametrize(
    "mutation", ["missing_check", "missing_request", "failed", "counts", "tokens"]
)
def test_every_policy_and_request_needs_complete_validation(report, request_id, mutation):
    requests = report["validation"]["chunked"]["requests"]
    evidence = requests[request_id]
    if mutation == "missing_check":
        evidence["checks"].pop()
    elif mutation == "missing_request":
        requests.pop(request_id)
    elif mutation == "failed":
        evidence["passed"] = False
    elif mutation == "counts":
        evidence["generation_logit_cache_greedy_checks"] -= 1
    else:
        evidence["generated_token_ids"][-1] ^= 1
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize(
    "mutation",
    ["position", "input", "offset", "greedy", "layers", "kv", "prefix", "slot", "tolerance"],
)
def test_fidelity_requires_correct_generation_states_cache_coverage_and_fixed_tolerances(
    report, mutation
):
    checks = report["validation"]["eager"]["requests"]["B"]["checks"]
    check = checks[5]
    if mutation in ("position", "offset"):
        check[mutation] += 1
    elif mutation == "input":
        check["input_token"] ^= 1
    elif mutation == "greedy":
        check["logits"]["mlx_next_token"] ^= 1
    elif mutation == "layers":
        check["cache"]["layers_checked"] = 1
    elif mutation == "kv":
        check["cache"]["keys_and_values_checked"] = False
    elif mutation == "prefix":
        checks[3]["cache"]["full_prefix_checked"] = False
    elif mutation == "slot":
        check["cache"]["positions_checked"] = [0]
    else:
        check["logits"]["atol"] = 0.1
    with pytest.raises(ValueError):
        chart_data(report)


@pytest.mark.parametrize(
    "mutation",
    [
        "trial",
        "order",
        "policy",
        "tokens",
        "timed_count",
        "initial_timed",
        "timestamp",
        "missing_event",
        "event_position",
        "event_overlap",
        "emission_end",
        "metrics",
        "gap",
        "active",
        "median",
        "ratio",
    ],
)
def test_raw_timed_policy_evidence_and_all_derived_summaries_are_verified(report, mutation):
    trial = report["samples"][1]
    sample = trial["policies"]["chunked"]
    if mutation == "trial":
        trial["trial"] = 0
    elif mutation == "order":
        trial["order"].reverse()
    elif mutation == "policy":
        sample["policy"] = "eager"
    elif mutation == "tokens":
        sample["requests"]["A"]["generated_token_ids"][-1] ^= 1
    elif mutation == "timed_count":
        sample["requests"]["A"]["timed_output_tokens"] += 1
    elif mutation == "initial_timed":
        sample["requests"]["A"]["initial_token_timed"] = True
    elif mutation == "timestamp":
        sample["requests"]["A"]["emission_times_seconds"][0] = 0.1
    elif mutation == "missing_event":
        sample["events"].pop()
    elif mutation == "event_position":
        sample["events"][1]["position"] = 0
    elif mutation == "event_overlap":
        sample["events"][1]["started_seconds"] = 0
    elif mutation == "emission_end":
        sample["events"][1]["completed_seconds"] += 1e-4
    elif mutation == "metrics":
        sample["metrics"]["a_gap_max_seconds"] += 1
    elif mutation == "gap":
        sample["requests"]["A"]["gap_distribution"]["p95_seconds"] += 1
    elif mutation == "active":
        sample["active_interference"]["b_prefill"]["window_seconds"][0] = 0
    elif mutation == "median":
        report["medians"]["chunked"]["makespan_seconds"] = 0
    else:
        report["paired_ratios"]["chunked"]["makespan_seconds"] *= 2
    with pytest.raises(ValueError):
        chart_data(report)


def test_plots_every_observation_and_discloses_the_actual_scope(report, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    _packing(report)
    figures = []

    def capture(figure, output):
        style.validate_text_layout(figure)
        figures.append(figure)

    monkeypatch.setattr(style, "save", capture)
    render(report, tmp_path / "fixture.png", "synthetic.json")
    figure = figures[0]
    assert len(figure.axes) == 3
    observed = 0
    for axis, (metric, _, _, scale) in zip(figure.axes, METRICS, strict=True):
        assert axis.get_xlim()[0] == 0 and axis.get_xscale() == "linear"
        dots = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "recorded-trials"
        ]
        bars = [
            collection for collection in axis.collections if collection.get_gid() == "sample-median"
        ]
        for policy, points, bar in zip(report["workload"]["policies"], dots, bars, strict=True):
            expected = [
                trial["policies"][policy]["metrics"][metric] * scale for trial in report["samples"]
            ]
            assert points.get_offsets()[:, 0].tolist() == expected
            assert bar.get_segments()[0][:, 0].tolist() == [statistics.median(expected)] * 2
            observed += len(expected)
    assert observed == 45
    prose = "\n".join(text.get_text() for text in figure.texts)
    assert "same meTile backend" in prose and "lossless-packed" in prose
    assert "3 remaining A tokens + 4 B tokens = 7 output IDs" in prose
    assert "Cooperative serial interleaving" in prose
    assert "no hardware overlap or SM reservation" in prose
    assert "not an MLX speed comparison" in prose
    assert "No confidence intervals or invented densities" in prose
    assert "Configured warmups per policy: 1" in prose


def test_exports_retain_source_identity_and_do_not_invent_density(report, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "synthetic-pd.png"
    render(report, output, "synthetic-evidence.json")
    assert output.read_bytes().startswith(b"\x89PNG")
    vector = output.with_suffix(".svg").read_text()
    assert "<text" in vector and "synthetic-evidence.json" in vector
    assert "observed-range-density" not in vector
