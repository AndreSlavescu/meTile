"""Does a whole model generate the same tokens under meTile as under MLX?

The kernel tests check numerics one kernel at a time, and the plan gate checks logits for a
single next-token step. Neither answers the question a user actually has, because both can
pass while generation still diverges:

  - The quantized compatibility gates are tolerance based (rtol 3e-2). Error that small is
    invisible per layer and compounds across 32 to 64 of them.
  - Greedy decoding takes an argmax, which is discontinuous. Two logit vectors differing by
    1e-3 pick the same token almost always and a different one when the top two are close,
    and after one different token the sequences never reconverge.

So this generates greedily, temperature 0, and compares token ids position by position. On
divergence it reports where and what, because "token 37 of 48" and "token 2 of 48" are
different bugs.

Needs a model in the local Hugging Face cache; skipped when there is none, so it is quiet on
a fresh checkout and in CI rather than failing for the wrong reason.
"""

import json
from pathlib import Path

import pytest

CACHE = Path.home() / ".cache/huggingface/hub"
# Smallest first: a correctness test should be the cheap one to run.
CANDIDATES = (
    "Qwen2.5-0.5B-Instruct-4bit",
    "Qwen2.5-1.5B-Instruct-4bit",
    "Llama-3.2-1B-Instruct-4bit",
)
PROMPT = "Explain tiled matrix multiplication in two sentences."
STEPS = 48


def _cached(name):
    """Locate one cached model, or (None, None). Reads config only; no weights loaded here."""
    found = sorted(CACHE.glob(f"models--mlx-community--{name}/snapshots/*/config.json"))
    return (f"mlx-community/{name}", found[0]) if found else (None, None)


def _cached_model():
    """The smallest cached model, or None."""
    for name in CANDIDATES:
        repo, path = _cached(name)
        if repo:
            return repo, path
    return None, None


def _assert_patched(model, patch):
    """Fail loudly if the patch context stopped swapping anything.

    A token comparison against MLX passes trivially when meTile is not actually installed,
    so the dangerous failure here is not a wrong answer, it is a green test that stopped
    measuring. This pins that down by checking a layer's bound implementations change inside
    the context and are restored on exit.
    """
    layer = model.model.layers[0]
    watched = ("mlp", "self_attn", "input_layernorm")
    before = {name: type(getattr(layer, name)).__call__ for name in watched}
    with patch(model=model):
        swapped = [
            name for name in watched if type(getattr(layer, name)).__call__ is not before[name]
        ]
    restored = all(type(getattr(layer, name)).__call__ is before[name] for name in watched)
    assert swapped, (
        "apply_metile_to_mlx_lm swapped nothing, so a token comparison would pass without "
        "exercising any meTile kernel"
    )
    assert restored, "apply_metile_to_mlx_lm did not restore the original implementations"
    return swapped


def _greedy(model, tokens, steps):
    """Generate `steps` tokens by argmax, returning the ids.

    Written out rather than calling a generate helper so both sides provably share one
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


@pytest.mark.slow
@pytest.mark.parametrize("candidate", CANDIDATES)
def test_greedy_generation_matches_mlx_token_for_token(candidate):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm import load

    repo, config_path = _cached(candidate)
    if repo is None:
        pytest.skip(f"{candidate} is not in the local Hugging Face cache")

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    model, tokenizer = load(repo)
    swapped = _assert_patched(model, apply_metile_to_mlx_lm)
    tokens = mx.array([tokenizer.encode(PROMPT)])

    # MLX first, on an untouched model, so the reference cannot be affected by patching.
    reference = _greedy(model, tokens, STEPS)

    with apply_metile_to_mlx_lm(model=model):
        actual = _greedy(model, tokens, STEPS)

    if actual != reference:
        first = next(
            index for index, (left, right) in enumerate(zip(reference, actual)) if left != right
        )
        detail = json.dumps(
            {
                "model": repo,
                "hidden": json.loads(config_path.read_text())
                .get("text_config", json.loads(config_path.read_text()))
                .get("hidden_size"),
                "patched": swapped,
                "diverged_at": first,
                "of_steps": STEPS,
                "mlx_token": reference[first],
                "metile_token": actual[first],
                "mlx_text": tokenizer.decode(reference[: first + 1])[-60:],
                "metile_text": tokenizer.decode(actual[: first + 1])[-60:],
            },
            indent=2,
        )
        pytest.fail(f"greedy generation diverged from MLX:\n{detail}")


@pytest.mark.slow
def test_greedy_generation_is_reproducible_under_metile():
    """meTile must be deterministic before matching MLX means anything.

    Autotuning picks kernels by measurement, so a second run can select differently. If that
    changed the output, a token comparison against MLX would be measuring the tuner's mood
    rather than the kernels' correctness, and this test says which of the two failed.
    """
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm import load

    repo, _ = _cached_model()
    if repo is None:
        pytest.skip("no mlx-community model in the local Hugging Face cache")

    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    model, tokenizer = load(repo)
    tokens = mx.array([tokenizer.encode(PROMPT)])

    with apply_metile_to_mlx_lm(model=model):
        first = _greedy(model, tokens, STEPS)
    with apply_metile_to_mlx_lm(model=model):
        second = _greedy(model, tokens, STEPS)

    assert first == second, "meTile generated different tokens on two identical runs"
