"""The one-call entry point, and the failure it is built to make impossible.

`metile.compile(model)` is the API people actually use, so the thing it must never do is quietly change
nothing and look like it worked. That failure has a track record here: sixteen equivalence tests reported
"skipped" for models meTile was not touching, which reads like success in a summary, for as long as it took
someone to read the skip list.

So the report is falsy when nothing was replaced, it names what it declined and by how much, and it
verifies against the unpatched model before keeping anything.
"""

import pytest

import metile
from metile.compile import CompileReport


def test_a_report_that_replaced_nothing_is_falsy():
    """`if not metile.compile(model)` has to be a usable check, or silence looks like success."""
    assert not CompileReport(model="whatever")
    assert CompileReport(model="whatever", features=("attention",))


def test_the_report_says_plainly_when_it_did_nothing():
    """A summary someone skims must not read as success."""
    text = str(CompileReport(model="exotic"))
    assert "nothing" in text
    assert "runs entirely on MLX" in text


def test_the_report_distinguishes_exact_from_within_tolerance():
    """Reporting a tolerated difference as "exact" would hide the trade the caller opted into."""
    exact = str(CompileReport(model="m", features=("attention",), verified=True, difference=0.0))
    assert "match MLX exactly" in exact

    tolerated = str(
        CompileReport(
            model="m",
            features=("attention",),
            verified=True,
            difference=0.035,
            relative=2.3e-3,
        )
    )
    assert "within tolerance" in tolerated
    assert "exactly" not in tolerated


def test_compile_rejects_something_that_is_not_a_model():
    for value in (None, 42, "a string"):
        with pytest.raises(TypeError, match="MLX-LM model"):
            metile.compile(value)


def test_structural_detection_admits_unlisted_architectures():
    """The generalisation, checked on a class that is deliberately not in the name list.

    A name list only covers what someone remembered, and this one excluded Qwen3.5, Qwen3.6 and Qwen3-VL
    without saying so. Structure is what makes an unseen model a candidate; verification is what makes that
    safe, and the two are separate steps on purpose.
    """
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import gemma2, llama

    from metile.integrations.mlx_lm import (
        _FUSED_BLOCK_CLASSES,
        _GATED_MLP_CLASSES,
        _recognised,
    )

    assert _recognised(llama.MLP, _GATED_MLP_CLASSES)
    assert _recognised(llama.MLP, _GATED_MLP_CLASSES, structural=False)

    # Not in the list, and admitted anyway because it has the parts.
    assert _recognised(gemma2.MLP, _GATED_MLP_CLASSES)
    assert not _recognised(gemma2.MLP, _GATED_MLP_CLASSES, structural=False)

    # Structure has to exclude as well as admit, or it is not doing any work.
    assert not _recognised(llama.Attention, _GATED_MLP_CLASSES)
    assert not _recognised(llama.MLP, _FUSED_BLOCK_CLASSES)


def test_structural_detection_needs_a_call_to_replace():
    """A class that defines no __call__ has nothing to swap, and getattr would not say so.

    `getattr(cls, "__call__")` on such a class resolves through the metaclass and returns a fresh
    method-wrapper every time, which reads as "present". This project has already been caught by that once.
    """
    from metile.integrations.mlx_lm import _GATED_MLP_CLASSES, _structurally_matches

    class HasThePartsButNoCall:
        gate_proj = up_proj = down_proj = staticmethod(lambda x: x)

    assert not _structurally_matches(HasThePartsButNoCall, _GATED_MLP_CLASSES)


@pytest.mark.slow
def test_compile_on_a_real_model_verifies_and_reports():
    """End to end, on the smallest checkpoint the cache is likely to hold."""
    from pathlib import Path

    pytest.importorskip("mlx_lm")
    repo = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    if not cache.exists():
        pytest.skip(f"{repo} is not in the local cache")

    from mlx_lm import load

    model, _ = load(repo)
    report = metile.compile(model)
    try:
        assert report, f"compile replaced nothing:\n{report}"
        assert report.verified is True
        assert report.surfaces, "features were kept but no layer reports a replacement"
        assert report.difference == 0.0
    finally:
        report.restore()


@pytest.mark.slow
def test_restore_puts_back_what_it_replaced():
    """A patch that cannot be undone is a patch nobody can measure against."""
    from pathlib import Path

    pytest.importorskip("mlx_lm")
    repo = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    if not cache.exists():
        pytest.skip(f"{repo} is not in the local cache")

    from mlx_lm import load

    from metile.compile import _surfaces

    model, _ = load(repo)
    assert not _surfaces(model)
    report = metile.compile(model, verify=False)
    assert _surfaces(model)
    report.restore()
    assert not _surfaces(model), "restore left meTile implementations bound"
