"""meTile against MLX at the exact layer shapes each model uses.

An end-to-end model number answers whether a model gets faster. It does not answer why,
and for these models the answer is mostly "no" for a reason worth seeing. A transformer is
one shape repeated, so measuring that shape separates the two things an end-to-end figure
conflates:

  width   MLX's int4 kernel is weak below an output width of about 2560, and that band is
          where the prefill win lives. Only the down projection can land in it, because
          gate and up always output `intermediate`, which is wide in every model here.

  batch   MLX re-reads weights per row tile above one row, so batched decode is where
          meTile wins regardless of width. Single-token decode is bandwidth bound and has
          nothing to give.

Every row is matched representation: identical int4 group-64 weights on both sides, so a
ratio is a kernel difference and not a change of numeric format. Shapes come from whatever
is in the local Hugging Face cache, so this measures models you actually have.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

CACHE = Path.home() / ".cache/huggingface/hub"
GROUP, BITS = 64, 4
# Output width below which MLX switches to a weaker int4 kernel, measured on M5.
CLIFF = 2560
PROMPT_ROWS = 127
BATCHES = (1, 8, 16)

MODELS = (
    "Qwen2.5-0.5B-Instruct-4bit",
    "Qwen2.5-1.5B-Instruct-4bit",
    "Llama-3.2-1B-Instruct-4bit",
    "Llama-3.2-3B-Instruct-4bit",
    "Qwen3.5-4B-4bit",
    "Qwen3.5-9B-4bit",
    "Qwen3.6-27B-4bit",
    # Vision language models. Only the language tower is measured, which is where every
    # shape this compares lives; the vision encoder runs once per image, not per token.
    "Qwen3-VL-4B-Instruct-4bit",
    "Qwen2.5-VL-7B-Instruct-4bit",
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    # 15 rounds is not enough precision for a published cell. At that count Qwen2.5-1.5B's
    # rows-16 figure came out 1.12x in one run while three independent measurements of the
    # same shape gave 1.31x, 1.31x and 1.32x. A 15% error in a number people read off a
    # table is worth the extra minutes.
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def dimensions(name):
    """Read hidden and intermediate width from a cached model, or None if absent."""
    found = list(CACHE.glob(f"models--mlx-community--{name}/snapshots/*/config.json"))
    if not found:
        return None
    config = json.loads(found[0].read_text())
    # Multimodal checkpoints nest the language model under text_config.
    text = config.get("text_config", config)
    if not text.get("hidden_size"):
        return None
    return text["hidden_size"], text["intermediate_size"], text.get("num_hidden_layers")


def paired(builders, mx, rounds, inner=3):
    """Interleave the variants so thermal drift lands on all of them equally."""
    for _ in range(3):
        for build in builders.values():
            mx.eval([build() for _ in range(inner)])
    mx.synchronize()
    samples = {name: [] for name in builders}
    order = list(builders)
    for index in range(rounds):
        for name in order[index % len(order) :] + order[: index % len(order)]:
            started = time.perf_counter_ns()
            mx.eval([builders[name]() for _ in range(inner)])
            samples[name].append((time.perf_counter_ns() - started) / inner / 1e9)
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    arguments = _arguments()
    import mlx.core as mx
    import mlx.nn as nn

    from metile.backends.mlx_affine import MLXAffineWeight, mlx_affine_matmul

    def quantize(shape):
        dense = mx.random.normal(shape).astype(mx.float16)
        parts = mx.quantize(dense, group_size=GROUP, bits=BITS, mode="affine")
        mx.eval(parts)
        del dense
        return parts

    def native(tensor, parts):
        packed, scales, biases = parts
        return mx.quantized_matmul(
            tensor,
            packed,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            mode="affine",
        )

    print("meTile vs MLX at each model's own layer shapes, int4 group 64, matched weights")
    print(f"MLX's int4 kernel is weak below output width {CLIFF}\n")
    batch_header = "".join(f"{f'rows {rows}':>9}" for rows in BATCHES)
    header = f"{'model':<26}{'hidden':>7}{'inter':>7}{'pre up':>8}{'pre down':>10}{batch_header}"
    print(header)
    print("-" * len(header))

    records = []
    for name in arguments.models or MODELS:
        dims = dimensions(name)
        if dims is None:
            print(f"{name:<26}  not in the local cache")
            continue
        hidden, intermediate, layers = dims

        mx.random.seed(0)
        gate, up, down = (
            quantize((intermediate, hidden)),
            quantize((intermediate, hidden)),
            quantize((hidden, intermediate)),
        )
        gate_w, up_w, down_w = (
            MLXAffineWeight.from_mlx(*gate, group_size=GROUP, bits=BITS),
            MLXAffineWeight.from_mlx(*up, group_size=GROUP, bits=BITS),
            MLXAffineWeight.from_mlx(*down, group_size=GROUP, bits=BITS),
        )
        mx.eval(gate_w.packed, up_w.packed, down_w.packed)

        narrow = mx.random.normal((PROMPT_ROWS, hidden)).astype(mx.float16)
        wide = mx.random.normal((PROMPT_ROWS, intermediate)).astype(mx.float16)
        mx.eval(narrow, wide)

        medians = paired(
            {
                "mlx": lambda: native(narrow, gate),
                "metile": lambda: mlx_affine_matmul(narrow, gate_w),
            },
            mx,
            arguments.rounds,
        )
        prefill_up = medians["mlx"] / medians["metile"]
        medians = paired(
            {
                "mlx": lambda: native(wide, down),
                "metile": lambda: mlx_affine_matmul(wide, down_w),
            },
            mx,
            arguments.rounds,
        )
        prefill_down = medians["mlx"] / medians["metile"]

        block = {}
        for rows in BATCHES:
            values = mx.random.normal((rows, hidden)).astype(mx.float16)
            mx.eval(values)
            medians = paired(
                {
                    "mlx": lambda v=values: native(nn.silu(native(v, gate)) * native(v, up), down),
                    "metile": lambda v=values: mlx_affine_matmul(
                        nn.silu(mlx_affine_matmul(v, gate_w)) * mlx_affine_matmul(v, up_w),
                        down_w,
                    ),
                },
                mx,
                arguments.rounds,
            )
            block[rows] = medians["mlx"] / medians["metile"]

        records.append(
            {
                "model": name,
                "hidden": hidden,
                "intermediate": intermediate,
                "layers": layers,
                "below_cliff": hidden < CLIFF,
                "prefill_up_speedup": prefill_up,
                "prefill_down_speedup": prefill_down,
                "block_speedup": {str(rows): value for rows, value in block.items()},
            }
        )
        marker = "  <- below cliff" if hidden < CLIFF else ""
        batches = "".join(f"{block[rows]:>8.2f}x" for rows in BATCHES)
        print(
            f"{name:<26}{hidden:>7}{intermediate:>7}"
            f"{prefill_up:>7.2f}x{prefill_down:>9.2f}x{batches}{marker}"
        )

        # Release this model's arrays before quantizing the next one. Relying on rebinding
        # at the top of the next iteration is not enough: the old and new weights are both
        # live across the transition, which at 27B scale is about a gigabyte of overlap.
        # That was enough to corrupt single measurements, and the damage did not look like
        # noise. It looked like a result: one of the two prefill numbers came out near
        # 0.76x, and which projection it landed on changed between runs.
        #
        # Assigning None rather than `del` because the timing closures above still name
        # these, and deleting the binding makes them unresolvable to static analysis.
        gate = up = down = None
        gate_w = up_w = down_w = None
        narrow = wide = values = None
        gc.collect()
        mx.clear_cache()

    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(
            json.dumps(
                {
                    "scope": "model_shape_matrix",
                    "precision_comparison": {
                        "class": "same_representation",
                        "same_weight_representation": True,
                    },
                    "cliff": CLIFF,
                    "prompt_rows": PROMPT_ROWS,
                    "rounds": arguments.rounds,
                    "models": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nWrote {arguments.output_json}")


if __name__ == "__main__":
    main()
