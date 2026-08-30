"""Patch real models' decode path with meTile and require identical tokens to MLX.

One registry of models, one registry of patch surfaces, and a parametrised test over the
product. Adding a model is a line in MODEL_CASES; adding a subsystem is a line in
FEATURE_SETS.

Why per-subsystem rather than one all-features run: an all-on failure tells you the model
broke, not what broke it. Running each subsystem alone turns a red cell into a named
suspect, and the all-on case still catches interactions between them.

Why token equality rather than a tolerance: the kernel tests already check numerics per
kernel, and the model plan gate checks logits for one next-token step. Neither can see what
this does. The quantized compatibility gates pass at rtol 3e-2, which is invisible per layer
and compounds across 32 to 64 of them; and greedy decoding takes an argmax, which is
discontinuous, so logits 1e-3 apart agree almost always and disagree exactly when the top two
candidates are close. After one differing token the sequences never reconverge. At
temperature 0 the contract is equality, so that is what gets asserted.

Models come from the local Hugging Face cache and are skipped when absent, so a fresh
checkout and CI stay green instead of failing for the wrong reason. The large tier is opt-in
via METILE_TEST_LARGE_MODELS=1 because it loads up to 15 GB of weights.
"""

import gc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from metile.compile import _decode_logits


@pytest.fixture(autouse=True)
def _release_gpu_memory():
    """Drop each test's model weights before the next test loads its own.

    Every test here loads a checkpoint and none released one, so across the matrix several models'
    weights stayed resident at once, up to about fifteen gigabytes for the largest. Letting the
    Python reference go is not enough: MLX keeps freed device buffers in a cache, so the memory is
    not returned until that cache is dropped.

    This is a robustness fix rather than a proven root cause. The symptom was one test in five full
    matrix runs failing on bit-exactness, always passing in isolation and always passing with its
    own model's cases run alone, so it needed the whole matrix to appear. Memory pressure has
    produced exactly that signature in this project before, in a benchmark where two models' weights
    overlapping corrupted single measurements and the damage read as a result rather than as noise.
    """
    yield
    gc.collect()
    mx = __import__("sys").modules.get("mlx.core")
    if mx is not None:
        mx.clear_cache()


CACHE = Path.home() / ".cache/huggingface/hub"
PROMPT = "Explain tiled matrix multiplication in two sentences."


@dataclass(frozen=True)
class ModelCase:
    """One checkpoint to verify, and how much of it to verify."""

    name: str
    steps: int = 48
    # "small" runs under the slow marker; "large" additionally needs METILE_TEST_LARGE_MODELS.
    tier: str = "small"
    # Vision language checkpoints nest the language model, and only that tower is patched.
    note: str = ""

    @property
    def repo(self):
        return f"mlx-community/{self.name}"


MODEL_CASES = (
    ModelCase("Qwen2.5-0.5B-Instruct-4bit"),
    ModelCase("Qwen2.5-1.5B-Instruct-4bit"),
    ModelCase("Llama-3.2-1B-Instruct-4bit"),
    ModelCase("Llama-3.2-3B-Instruct-4bit", tier="large"),
    ModelCase("Qwen3.5-4B-4bit", tier="large"),
    ModelCase("Qwen3.5-9B-4bit", tier="large"),
    ModelCase("Qwen3.6-27B-4bit", steps=24, tier="large"),
    ModelCase("Qwen3-VL-4B-Instruct-4bit", tier="large", note="language tower only"),
)

# Each entry disables everything except the subsystem under test, so a failure names one
# suspect. "all" is the default configuration a user actually gets, and is the only one that
# can catch interactions between subsystems.
_OFF = {"attention": False, "rms_norm": False, "graph_fusion": False, "quantized_mlp": False}
FEATURE_SETS = {
    "all": {},
    "attention": {**_OFF, "attention": True},
    "rms_norm": {**_OFF, "rms_norm": True},
    "graph_fusion": {**_OFF, "graph_fusion": True},
    "quantized_mlp": {**_OFF, "quantized_mlp": True},
}

# Divergences that are real, reproduced, and not yet fixed. Recorded as strict xfail so they
# cannot masquerade as passes, and so fixing one shows up as an unexpected pass rather than
# staying quietly green.
#
# Empty right now. The Qwen3-VL-4B attention entry lived here and was retired once the cause
# was found: the kernel multiplied two storage-dtype loads together, so for bfloat16 every
# dot-product term rounded to an 8-bit significand before reaching the f32 accumulator. That
# cost 4x MLX's accuracy and moved a logit by 0.43. Casting the loads to f32 made all 36
# decode-step calls bit-exact, and this xfail then reported XPASS(strict), which is how the
# mechanism is supposed to announce a fix.
KNOWN_DIVERGENCES = {}

# Pairs whose logits differ only by floating-point reduction order, where meTile is measured
# to be as accurate as MLX or better. Bit-exactness is the default and any pair not listed
# here must achieve it; these are bounded instead, as a fraction of the logit magnitude.
#
# Both were measured against a float32 reference at kernel level before being allowed here:
#
#   quantized SwiGLU, hidden 2048, inter 8192, float16
#     MLX errs 18.05 from truth, meTile 4.10. MLX's f16 accumulation degrades at this
#     reduction width and meTile's does not.
#   RMSNorm, hidden 3072, float16
#     MLX errs 0.00293, meTile 0.00185.
#
# Matching MLX bit-for-bit in these two cases means adopting its summation order, which is
# measurably less accurate. That is the wrong trade, so the difference is documented rather
# than eliminated. A pair that drifts the other way, where meTile is worse, must fail: that
# is what caught the attention kernel multiplying storage-dtype loads together.
ORDERING_TOLERANCE = 0.005
ORDERING_DIFFERENCES = {
    ("Llama-3.2-1B-Instruct-4bit", "quantized_mlp"): "f16 SwiGLU reduction; meTile 4.4x closer",
    ("Llama-3.2-1B-Instruct-4bit", "all"): "same cause as its quantized_mlp case",
    ("Llama-3.2-3B-Instruct-4bit", "rms_norm"): "f16 RMSNorm reduction; meTile 1.6x closer",
    ("Llama-3.2-3B-Instruct-4bit", "all"): "same cause as its rms_norm case",
}

# Layer attributes whose bound implementation meTile may replace. Used to prove the patch
# actually took effect, and to report which subsystem a run really exercised.
_WATCHED = (
    "mlp",
    "self_attn",
    # Qwen3.5 and Qwen3.6 are hybrid: most layers carry a GatedDeltaNet under linear_attn
    # instead of a standard attention module, and their MLP is a Qwen3NextMLP / SparseMoeBlock.
    "linear_attn",
    "input_layernorm",
    "post_attention_layernorm",
)


@dataclass
class _Swap:
    swapped: list = field(default_factory=list)
    restored: bool = True


def _config_path(case):
    found = sorted(CACHE.glob(f"models--mlx-community--{case.name}/snapshots/*/config.json"))
    return found[0] if found else None


def _require(case):
    """Skip unless this checkpoint is cached and its tier is enabled."""
    if _config_path(case) is None:
        pytest.skip(f"{case.name} is not in the local Hugging Face cache")
    if case.tier == "large" and os.environ.get("METILE_TEST_LARGE_MODELS") != "1":
        pytest.skip(f"{case.name} is large tier; set METILE_TEST_LARGE_MODELS=1 to include it")


def _implementation(cls):
    """The __call__ a class actually defines, walking the MRO, or None.

    Not `getattr(cls, "__call__")`. For a class that does not define __call__, that resolves
    through the metaclass and hands back a fresh method-wrapper on every access, so an
    identity comparison reports a change that never happened. The guard below exists to catch
    a patch that silently stopped working; a false positive there is worse than no guard,
    because it claims coverage that is not present.
    """
    for klass in cls.__mro__:
        if "__call__" in vars(klass):
            return vars(klass)["__call__"]
    return None


def _transformer_blocks(model):
    """Find the decoder block list without assuming where it lives.

    `model.model.layers` holds for Qwen2.5, Llama and Qwen3-VL but raises AttributeError on
    Qwen3.5, which nests differently. Searching for the list structurally keeps the registry
    architecture-agnostic, and accepting linear_attn as well as self_attn is what lets a
    hybrid model be recognised at all.
    """
    seen, stack = set(), [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for name in dir(node):
            if name.startswith("_"):
                continue
            try:
                value = getattr(node, name)
            except Exception:  # probing an unknown module tree; any attribute may raise
                continue
            if (
                isinstance(value, list)
                and value
                and any(hasattr(value[0], attr) for attr in ("self_attn", "linear_attn", "attn"))
            ):
                return value
            if hasattr(value, "children"):
                stack.append(value)
    return None


def _probe_points(model):
    """Every implementation meTile is known to replace, as (label, getter) pairs.

    Three different surfaces, which is why an earlier version of this file reported
    "patches nothing" for two subsystems that were in fact active:

      block      graph fusion replaces the transformer block's own __call__
      <attr>     the quantized MLP and RMSNorm paths replace a submodule's __call__
      sdpa       attention replaces a module-level function in mlx_lm.models.base,
                 not anything reachable from the model object at all
    """
    blocks = _transformer_blocks(model)
    if blocks is None:
        return []
    layer = blocks[0]
    points = [("block", lambda layer=layer: _implementation(type(layer)))]
    for name in _WATCHED:
        if hasattr(layer, name):
            points.append((name, lambda name=name: _implementation(type(getattr(layer, name)))))
    try:
        from mlx_lm.models import base

        points.append(("sdpa", lambda: getattr(base, "scaled_dot_product_attention", None)))
    except ImportError:
        pass
    return points


def _observe_patch(model, patch, features):
    """Record which implementations a patch swaps, and whether it restores them.

    A token comparison passes trivially when meTile is not actually installed, so the
    dangerous failure is not a wrong answer but a green test that stopped measuring. Callers
    assert on the result.
    """
    points = _probe_points(model)
    before = {label: getter() for label, getter in points}
    with patch(model=model, **features):
        swapped = [label for label, getter in points if getter() is not before[label]]
    restored = all(getter() is before[label] for label, getter in points)
    return _Swap(swapped=swapped, restored=restored)


def _greedy(model, tokens, steps):
    """Generate `steps` tokens by argmax.

    Spelled out rather than delegated to a generate helper so both sides provably share one
    decoding rule: any difference in the sequence comes from the logits, not from sampling,
    cache handling, or a stop condition.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    produced = []
    for _ in range(steps):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(nxt)
        produced.append(int(nxt.item()))
        logits = model(nxt[None, :], cache=cache)
    return produced


def _divergence_report(case, feature_name, swap, reference, actual, tokenizer):
    first = next(
        index for index, (left, right) in enumerate(zip(reference, actual)) if left != right
    )
    return json.dumps(
        {
            "model": case.repo,
            "features": feature_name,
            "patched": swap.swapped,
            "diverged_at_token": first,
            "of_steps": case.steps,
            "mlx_token": reference[first],
            "metile_token": actual[first],
            "mlx_tail": tokenizer.decode(reference[: first + 1])[-60:],
            "metile_tail": tokenizer.decode(actual[: first + 1])[-60:],
        },
        indent=2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("feature_name", sorted(FEATURE_SETS))
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.name)
def test_decode_matches_mlx_token_for_token(case, feature_name, request):
    """Greedy decode under meTile must reproduce MLX exactly, per subsystem and all together."""
    known = KNOWN_DIVERGENCES.get((case.name, feature_name))
    if known:
        request.node.add_marker(pytest.mark.xfail(reason=known, strict=True))
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    _require(case)
    from mlx_lm import load

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    features = FEATURE_SETS[feature_name]
    model, tokenizer = load(case.repo)
    swap = _observe_patch(model, apply_metile_to_mlx_lm, features)
    assert swap.restored, "apply_metile_to_mlx_lm did not restore the original implementations"
    if not swap.swapped:
        # Not a silent pass. Some architectures expose surfaces meTile does not recognise:
        # Qwen3.5 and Qwen3.6 put a GatedDeltaNet under linear_attn and a Qwen3NextMLP under
        # mlp, and neither the MLP nor the block patch matches, so the subsystem genuinely
        # does not run there. Saying so beats asserting equality that is trivially true.
        pytest.skip(
            f"{feature_name} patches nothing on {case.name}; "
            f"surfaces present: {[label for label, _ in _probe_points(model)]}"
        )

    tokens = mx.array([tokenizer.encode(PROMPT)])
    # MLX first, on an untouched model, so the reference cannot be affected by patching.
    reference = _greedy(model, tokens, case.steps)
    with apply_metile_to_mlx_lm(model=model, **features):
        actual = _greedy(model, tokens, case.steps)

    if actual != reference:
        pytest.fail(
            "greedy decode diverged from MLX:\n"
            + _divergence_report(case, feature_name, swap, reference, actual, tokenizer)
        )


@pytest.mark.slow
@pytest.mark.parametrize("feature_name", sorted(FEATURE_SETS))
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.name)
def test_decode_logits_are_bit_exact(case, feature_name):
    """The real contract: identical logits, not merely identical tokens.

    Token equality is a weaker and noisier property. Two logit vectors can differ and still
    argmax the same way for many steps, so a token test can pass over a genuine numeric
    regression and then fail later on an unrelated change. Comparing logits catches the
    regression where it happens, and needs no decoding to do it.
    """
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    _require(case)
    from mlx_lm import load

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    features = FEATURE_SETS[feature_name]
    model, tokenizer = load(case.repo)
    swap = _observe_patch(model, apply_metile_to_mlx_lm, features)
    assert swap.restored, "apply_metile_to_mlx_lm did not restore the original implementations"
    if not swap.swapped:
        pytest.skip(f"{feature_name} patches nothing on {case.name}")

    tokens = mx.array([tokenizer.encode(PROMPT)])
    reference = _decode_logits(model, tokens, steps=6)
    with apply_metile_to_mlx_lm(model=model, **features):
        actual = _decode_logits(model, tokens, steps=6)
    mx.eval(reference, actual)

    difference = float(mx.max(mx.abs(actual - reference)).item())
    magnitude = float(mx.max(mx.abs(reference)).item())
    ordering = ORDERING_DIFFERENCES.get((case.name, feature_name))
    if ordering:
        allowed = ORDERING_TOLERANCE * magnitude
        assert difference <= allowed, (
            f"{case.name} / {feature_name}: logits differ by {difference:.6f}, beyond the "
            f"{allowed:.6f} allowed for a reduction-order difference ({ordering}). "
            f"A larger gap is a regression, not reordering; patched {swap.swapped}"
        )
        return
    assert difference == 0.0, (
        f"{case.name} / {feature_name}: decode logits differ by {difference:.6f} "
        f"against a maximum magnitude of {magnitude:.2f}; patched {swap.swapped}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", MODEL_CASES[:1], ids=lambda c: c.name)
def test_decode_is_reproducible_under_metile(case):
    """meTile must be deterministic before matching MLX means anything.

    Kernel selection is decided by measurement, so a second run can select differently. If
    that changed the tokens, comparing against MLX would be measuring the tuner rather than
    the kernels, and this test says which of the two failed.
    """
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    _require(case)
    from mlx_lm import load

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    model, tokenizer = load(case.repo)
    tokens = mx.array([tokenizer.encode(PROMPT)])
    with apply_metile_to_mlx_lm(model=model):
        first = _greedy(model, tokens, case.steps)
    with apply_metile_to_mlx_lm(model=model):
        second = _greedy(model, tokens, case.steps)

    assert first == second, "meTile generated different tokens on two identical runs"


def test_patch_observation_rejects_a_no_op_context():
    """The guard the tests above rely on has to actually fire.

    If _observe_patch reported a swap for a context manager that changes nothing, every
    equivalence test would pass while exercising no meTile kernel. This checks the detector
    rather than trusting it.
    """
    import contextlib

    class _Layer:
        def __call__(self):  # pragma: no cover - never invoked
            return None

    class _Inner:
        """Stands in for an mlx Module: needs children() to be walked, and an attention
        attribute on its blocks to be recognised as decoder layers."""

        def __init__(self):
            self.layers = [type("L", (), {"mlp": _Layer(), "self_attn": _Layer()})()]

        def children(self):
            return {}

    model = type("M", (), {"model": _Inner()})()

    @contextlib.contextmanager
    def noop(model=None, **features):
        yield

    observed = _observe_patch(model, noop, {})
    assert observed.swapped == [], "a context that changes nothing must report no swaps"
    assert observed.restored, "nothing was changed, so everything is trivially restored"
    # The detector must be looking at more than one surface, or it would miss the subsystems
    # that patch the block and the module-level attention function.
    assert len(_probe_points(model)) >= 2
