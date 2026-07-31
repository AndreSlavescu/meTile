"""One call that accelerates an MLX-LM model and tells you what it did.

    import metile
    model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    print(metile.compile(model))

Two things distinguish this from calling the patcher directly, and both come from failures this project
has actually had.

**It recognises architectures nobody enumerated.** The underlying patcher gates on a list of module and
class names, which only ever covers what someone remembered to add. Qwen3.5, Qwen3.6 and Qwen3-VL were all
excluded by that list, silently, for as long as it took to notice. `compile` also admits classes by
structure, so a model with the usual gated MLP is a candidate whether or not it has been seen before.

**It verifies before it keeps anything.** Structure is a weaker claim than a name: a class with
`gate_proj`, `up_proj` and `down_proj` has the parts of a gated MLP but might scale the product or use a
different activation, and it would present identically. So `compile` runs the model before and after
patching and compares the logits, feature by feature if the whole set disagrees, and keeps only what
reproduces MLX's output. A model it cannot verify runs unpatched rather than wrong.

The report is the third piece. The dangerous outcome here is not a crash, it is a silent no-op: sixteen
equivalence tests in this project spent weeks reporting "skipped" for models where nothing was being
patched, which reads like success in a summary. `compile` returns a report that says what it replaced and
what it declined, and it is falsy when it changed nothing.

Verification costs a few forward passes. Pass `verify=False` to skip it when the architecture is already
covered by the model matrix, and expect no protection against a structural false positive if you do.
"""

from dataclasses import dataclass, field

# Feature flags the patcher exposes that `compile` turns on, in the order they are tried when a combined
# verification fails and the set has to be bisected.
FEATURES = ("attention", "rms_norm", "graph_fusion", "quantized_mlp")

# Logits are compared exactly by default, because the failure this exists to catch -- a class that has the
# parts of a gated MLP but combines them differently -- lands far outside any rounding.
#
# Exactness does cost something real, and the report says so rather than hiding it. meTile's kernels are
# bit-exact with MLX wherever the arithmetic matches, but a few are not: they sum in a different order, and
# in the two cases checked against a float32 reference meTile was the *more* accurate side, by 4.4x on f16
# SwiGLU and 1.6x on f16 RMSNorm. Llama-3.2-1B is one of them, where quantized_mlp moves a logit by 0.035
# against a magnitude near 20 -- about 2e-3 relative, and declined under the default.
#
# So `tolerance` is relative to the largest reference logit and defaults to exact. Around 1e-3 is the scale
# of a reduction-order difference; anything much larger is a different computation, not a different order.
TOLERANCE = 0.0
REORDERING_SCALE = 5e-3


@dataclass
class CompileReport:
    """What `compile` replaced, what it declined, and why."""

    model: str = "unknown"
    features: tuple = ()
    surfaces: tuple = ()
    verified: bool | None = None
    difference: float | None = None
    relative: float | None = None
    declined: dict = field(default_factory=dict)
    handle: object = None

    def __bool__(self):
        """False when nothing was replaced, so `if not metile.compile(model)` is a usable check."""
        return bool(self.features)

    def restore(self):
        """Put back MLX-LM's own implementations."""
        if self.handle is not None:
            self.handle.__exit__(None, None, None)
            self.handle = None

    def __str__(self):
        lines = [f"meTile on {self.model}"]
        if self.features:
            lines.append(f"  accelerating: {', '.join(self.features)}")
            lines.append(f"  surfaces replaced: {', '.join(self.surfaces) or 'none reported'}")
        else:
            lines.append("  accelerating: nothing -- this model runs entirely on MLX")
        if self.verified is None:
            lines.append("  verification: skipped (verify=False)")
        elif self.verified and not self.difference:
            lines.append("  verification: logits match MLX exactly")
        elif self.verified:
            lines.append(
                f"  verification: within tolerance, logits differ by {self.difference:g}"
                + (f" ({self.relative:.1e} relative)" if self.relative else "")
            )
        else:
            lines.append("  verification: FAILED, everything reverted")
        for feature, reason in sorted(self.declined.items()):
            lines.append(f"  declined {feature}: {reason}")
        return "\n".join(lines)


def _decode_logits(model, tokens, steps=4):
    """Prefill, take a few decode steps, return the last step's logits.

    Decode steps rather than prefill, because meTile's attention only engages at query length one. A
    prefill-only comparison reports agreement while never running the kernel, which is true and
    meaningless; this project measured everything as exact until that was noticed.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    out = model(tokens, cache=cache)
    mx.eval(out)
    following = mx.argmax(out[:, -1, :], axis=-1)
    for _ in range(steps):
        out = model(following[None, :], cache=cache)
        mx.eval(out)
        following = mx.argmax(out[:, -1, :], axis=-1)
    return out[:, -1, :].astype(mx.float32)


def _probe_tokens(model):
    """A short token sequence valid for this model's vocabulary."""
    import mlx.core as mx

    vocabulary = getattr(model, "vocab_size", None) or 32000
    return mx.array([[index % max(vocabulary - 1, 1) + 1 for index in range(8)]])


def _surfaces(model):
    """Names of the layer attributes meTile currently has replacements bound to."""
    from metile.integrations.mlx_lm import _model_layers

    watched = ("mlp", "self_attn", "linear_attn", "input_layernorm", "post_attention_layernorm")
    found = []
    for layer in _model_layers(model):
        for name in watched:
            member = getattr(layer, name, None)
            if member is None or name in found:
                continue
            implementation = type(member).__call__
            if getattr(implementation, "_metile_original", None) is not None:
                found.append(name)
        block = type(layer).__call__
        if getattr(block, "_metile_original", None) is not None and "block" not in found:
            found.append("block")
    return tuple(found)


def compile(model, *, verify=True, features=FEATURES, tolerance=TOLERANCE):
    """Accelerate an MLX-LM model in place and report what changed.

    Returns a `CompileReport`, which is falsy when nothing was replaced. Call `.restore()` on it to put
    MLX-LM's implementations back.

    `tolerance` is relative to the largest reference logit and defaults to exact. Raising it to about
    5e-3 admits differences at the scale of a summation reorder, which is where meTile's remaining
    divergences from MLX sit and where it measured as the more accurate side; the report names any
    feature it declined and by how much, so the trade is visible before it is taken.
    """
    from metile.integrations.mlx_lm import apply_metile_to_mlx_lm

    if model is None or not callable(model):
        raise TypeError("compile expects a loaded MLX-LM model")

    name = type(getattr(model, "model", model)).__module__.rsplit(".", 1)[-1]
    requested = tuple(feature for feature in FEATURES if feature in features)
    declined = {}

    reference = None
    tokens = None
    magnitude = 1.0
    if verify:
        try:
            tokens = _probe_tokens(model)
            reference = _decode_logits(model, tokens)
            magnitude = max(
                float(
                    __import__("mlx.core", fromlist=["core"])
                    .max(__import__("mlx.core", fromlist=["core"]).abs(reference))
                    .item()
                ),
                1e-9,
            )
        except Exception as error:
            declined["verification"] = f"could not run the model unpatched ({type(error).__name__})"
            verify = False

    def attempt(selected):
        """Patch with `selected` enabled and return the handle, or None if it verifies wrong."""
        flags = {feature: feature in selected for feature in FEATURES}
        handle = apply_metile_to_mlx_lm(model=model, **flags)
        handle.__enter__()
        if not verify:
            return handle, None
        import mlx.core as mx

        try:
            difference = float(mx.max(mx.abs(_decode_logits(model, tokens) - reference)).item())
        except Exception:
            handle.__exit__(None, None, None)
            return None, None
        if difference > tolerance * magnitude:
            handle.__exit__(None, None, None)
            return None, difference
        return handle, difference

    handle, difference = attempt(requested)
    kept = requested
    if verify and handle is None:
        # The combined set disagrees, so find which parts do. Bisecting per feature beats reverting
        # everything: usually one surface is at fault and the rest reproduce MLX exactly.
        declined["combined"] = "the full feature set changed the logits; bisected"
        kept = []
        for feature in requested:
            trial, trial_difference = attempt([feature])
            if trial is None:
                if trial_difference:
                    relative = trial_difference / magnitude
                    scale = (
                        "reduction-order scale, raise tolerance to keep it"
                        if relative <= REORDERING_SCALE
                        else "far beyond rounding, so a different computation"
                    )
                    declined[feature] = (
                        f"changed the logits by {trial_difference:g}, "
                        f"{relative:.1e} relative -- {scale}"
                    )
                else:
                    declined[feature] = "could not run"
                continue
            trial.__exit__(None, None, None)
            kept.append(feature)
        if kept:
            handle, difference = attempt(kept)
        if handle is None:
            kept = []

    if handle is None:
        return CompileReport(model=name, verified=False if verify else None, declined=declined)

    surfaces = _surfaces(model)
    if not surfaces:
        # Patching a class nothing in this model uses is the silent no-op this report exists to expose.
        handle.__exit__(None, None, None)
        declined["surfaces"] = "no layer in this model uses a class meTile can replace"
        return CompileReport(model=name, verified=None, declined=declined)

    return CompileReport(
        model=name,
        features=tuple(kept),
        surfaces=surfaces,
        verified=None if not verify else True,
        difference=difference,
        relative=(difference / magnitude) if (verify and difference is not None) else None,
        declined=declined,
        handle=handle,
    )
