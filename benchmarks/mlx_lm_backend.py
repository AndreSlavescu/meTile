"""Benchmark meTile as an opt-in MLX-LM decode backend on a real model."""

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from metile.backends.mlx import mlx_attention_dispatches, mlx_rms_norm_dispatches
from metile.integrations.mlx_lm import apply_metile_to_mlx_lm


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
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def _generate(model, tokenizer, prompt, arguments, patched):
    patch = (
        apply_metile_to_mlx_lm(
            model=model,
            attention=not arguments.disable_attention,
            rms_norm=not arguments.disable_rmsnorm,
        )
        if patched
        else None
    )
    start = time.perf_counter()
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
    finally:
        if patch is not None:
            patch.restore()
    if response is None:
        raise RuntimeError("MLX-LM generation returned no timing response")
    return response, time.perf_counter() - start


def _verify_model(model, prompt, arguments):
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


def main():
    arguments = _arguments()
    model, tokenizer, config = load(arguments.model, return_config=True)
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (arguments.prompt_tokens,)).tolist()

    if not arguments.skip_verify:
        _verify_model(model, prompt, arguments)

    print("Warming MLX baseline...")
    _generate(model, tokenizer, prompt, arguments, patched=False)
    print("Compiling and autotuning meTile MLX kernels...")
    _generate(model, tokenizer, prompt, arguments, patched=True)

    results = {"MLX": [], "MLX + meTile": []}
    for trial in range(arguments.trials):
        order = (False, True) if trial % 2 == 0 else (True, False)
        for patched in order:
            if arguments.delay:
                time.sleep(arguments.delay)
            response, elapsed = _generate(model, tokenizer, prompt, arguments, patched)
            name = "MLX + meTile" if patched else "MLX"
            results[name].append((response.generation_tps, response.prompt_tps, elapsed))
            print(
                f"Trial {trial + 1} {name:12s}: "
                f"decode={response.generation_tps:.2f} tok/s, "
                f"prefill={response.prompt_tps:.2f} tok/s, total={elapsed:.3f}s"
            )

    baseline = statistics.median(sample[0] for sample in results["MLX"])
    metile_decode = statistics.median(sample[0] for sample in results["MLX + meTile"])
    baseline_total = statistics.median(sample[2] for sample in results["MLX"])
    metile_total = statistics.median(sample[2] for sample in results["MLX + meTile"])
    print("\nMedian results")
    print(f"MLX decode:          {baseline:.2f} tok/s")
    print(f"MLX + meTile decode: {metile_decode:.2f} tok/s ({metile_decode / baseline:.3f}x)")
    print(f"End-to-end speedup:  {baseline_total / metile_total:.3f}x")
    print("\nSelected attention schedules")
    for dispatch in mlx_attention_dispatches():
        print(
            f"tokens<={dispatch['token_bucket']}: {dispatch['algorithm']} block={dispatch['block']}"
        )
    print("Selected RMSNorm schedules")
    for dispatch in mlx_rms_norm_dispatches():
        print(
            f"rows<={dispatch['row_bucket']} hidden={dispatch['hidden']}: "
            f"{dispatch['algorithm']} block={dispatch['block']}"
        )


if __name__ == "__main__":
    main()
