"""Host-side packing for the experimental, single-threadgroup Qwen3 decoder.

This adapter is benchmark infrastructure, not a production model backend.
Model arithmetic lives in metile_kernels.megakernels.qwen3. Packing weights,
precomputing rotary constants, and importing a prefix cache are untimed setup.
"""

import math
from dataclasses import asdict

import numpy as np

import metile
from metile.runtime.metal_device import MetalDevice
from metile_kernels.megakernels.qwen3 import qwen3_decode_megakernel, qwen3_layer_offsets


def validate_config(config, capacity, threads):
    """Reject unsupported architectures before allocating or compiling anything."""
    if config.get("model_type") != "qwen3":
        raise ValueError("only dense Qwen3 is supported; Qwen3.5/3.6 are different architectures")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("the prototype requires tied token embeddings")
    if config.get("rope_scaling") is not None:
        raise ValueError("scaled RoPE is not supported by this prototype")
    dimensions = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    for name in dimensions:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if any(config[name] % 32 for name in ("hidden_size", "intermediate_size", "head_dim")):
        raise ValueError("hidden, intermediate, and head dimensions must be multiples of 32")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by key/value heads")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if capacity > config["max_position_embeddings"]:
        raise ValueError("capacity exceeds the model's supported positions")
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("threads must be 32, 64, 128, or 256")
    for name in ("rms_norm_eps", "rope_theta"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    dimension = config["head_dim"]
    query_width = config["num_attention_heads"] * dimension
    key_width = config["num_key_value_heads"] * dimension
    scratch_bytes = max(2 * hidden + query_width + 2 * key_width, 2 * hidden + intermediate) * 4
    if scratch_bytes > 32 * 1024:
        raise ValueError("model exceeds the prototype's 32 KiB threadgroup scratch budget")
    layout = qwen3_layer_offsets(
        hidden,
        intermediate,
        config["num_attention_heads"],
        config["num_key_value_heads"],
        dimension,
    )
    sizes = (
        config["vocab_size"] * hidden,
        config["num_hidden_layers"] * layout["layer_size"],
        2 * config["num_hidden_layers"] * key_width * capacity,
        capacity * dimension,
    )
    if max(sizes) >= 2**31:
        raise ValueError("tensor offsets exceed the DSL's signed 32-bit address arithmetic")
    return scratch_bytes


def _validate_model_structure(model):
    import mlx.nn as nn
    from mlx_lm.models import qwen3

    def expect(module, expected):
        if type(module) is not expected:
            raise ValueError(f"expected unmodified {expected.__name__}; adapters are unsupported")

    def norm(module):
        expect(module, nn.RMSNorm)
        if module.eps != model.args.rms_norm_eps or set(module) != {"weight"}:
            raise ValueError("RMSNorm differs from the model configuration")

    expect(model.model, qwen3.Qwen3Model)
    expect(model.model.embed_tokens, nn.Embedding)
    norm(model.model.norm)
    for layer in model.layers:
        expect(layer, qwen3.TransformerBlock)
        expect(layer.self_attn, qwen3.Attention)
        expect(layer.mlp, qwen3.MLP)
        attention = layer.self_attn
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            expect(projection, nn.Linear)
            if set(projection) != {"weight"}:
                raise ValueError("biased or augmented projections are unsupported")
        for module in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            attention.q_norm,
            attention.k_norm,
        ):
            norm(module)
        if (
            attention.n_heads != model.args.num_attention_heads
            or attention.n_kv_heads != model.args.num_key_value_heads
            or attention.scale != model.args.head_dim**-0.5
        ):
            raise ValueError("attention differs from the model configuration")
        expect(attention.rope, nn.RoPE)
        if (
            attention.rope.dims != model.args.head_dim
            or attention.rope.traditional
            or attention.rope.base != model.args.rope_theta
            or attention.rope.scale != 1.0
            or len(attention.rope)
        ):
            raise ValueError("RoPE differs from the model configuration")


class Qwen3Megakernel:
    """One batch-one token-to-logits dispatch, with a fixed-capacity KV cache.

    ``prepare`` executes once and returns an idempotent replay at that same token
    and position. It does not advance the token or position on subsequent calls.
    Do not replay it after changing the prefix that it depends on.
    """

    def __init__(self, model, capacity, threads=256):
        import mlx.core as mx
        from mlx_lm.models.qwen3 import Model

        if type(model) is not Model:
            raise ValueError("expected an unmodified mlx_lm.models.qwen3.Model")
        self.config = asdict(model.args)
        self.scratch_bytes = validate_config(self.config, capacity, threads)
        _validate_model_structure(model)
        self._mx_dtype = model.model.embed_tokens.weight.dtype
        if self._mx_dtype not in (mx.float16, mx.float32):
            raise ValueError("expected dense FP16 or FP32 weights")
        self.dtype = np.dtype(np.float16 if self._mx_dtype == mx.float16 else np.float32)
        self.capacity = capacity
        self.threads = threads
        self.cached_tokens = 0
        self.config.update(capacity=capacity, threads=threads, storage_dtype=self.dtype.name)
        hidden = self.config["hidden_size"]
        intermediate = self.config["intermediate_size"]
        query_heads = self.config["num_attention_heads"]
        key_heads = self.config["num_key_value_heads"]
        dimension = self.config["head_dim"]
        layers = self.config["num_hidden_layers"]
        vocabulary = self.config["vocab_size"]
        layout = qwen3_layer_offsets(hidden, intermediate, query_heads, key_heads, dimension)
        if len(model.layers) != layers:
            raise ValueError("model layers do not match its configuration")

        def weight(array, shape):
            if array.shape != shape or array.dtype != self._mx_dtype:
                raise ValueError(f"expected dense {self.dtype.name} weight of shape {shape}")
            mx.eval(array)
            return np.asarray(array)

        self.embedding = metile.Buffer(
            data=weight(model.model.embed_tokens.weight, (vocabulary, hidden))
        )
        packed = np.empty((layers, layout["layer_size"]), dtype=self.dtype)
        for index, layer in enumerate(model.layers):
            attention = layer.self_attn
            weights = (
                ("input_norm", layer.input_layernorm.weight, (hidden,)),
                ("q_proj", attention.q_proj.weight, (query_heads * dimension, hidden)),
                ("k_proj", attention.k_proj.weight, (key_heads * dimension, hidden)),
                ("v_proj", attention.v_proj.weight, (key_heads * dimension, hidden)),
                ("o_proj", attention.o_proj.weight, (hidden, query_heads * dimension)),
                ("post_norm", layer.post_attention_layernorm.weight, (hidden,)),
                ("gate_proj", layer.mlp.gate_proj.weight, (intermediate, hidden)),
                ("up_proj", layer.mlp.up_proj.weight, (intermediate, hidden)),
                ("down_proj", layer.mlp.down_proj.weight, (hidden, intermediate)),
                ("q_norm", attention.q_norm.weight, (dimension,)),
                ("k_norm", attention.k_norm.weight, (dimension,)),
            )
            for name, array, shape in weights:
                values = weight(array, shape).reshape(-1)
                start = layout[name]
                packed[index, start : start + values.size] = values
        self.layer_weights = metile.Buffer(data=packed)
        self.final_norm = metile.Buffer(data=weight(model.model.norm.weight, (hidden,)))
        frequencies = self.config["rope_theta"] ** (
            -np.arange(0, dimension, 2, dtype=np.float64) / dimension
        )
        angles = np.arange(capacity, dtype=np.float64)[:, None] * frequencies[None, :]
        rotary = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
        self.rotary = metile.Buffer(data=rotary)
        self.parameter_count = vocabulary * hidden + layers * layout["layer_size"] + hidden
        self._allocate_cache_state()

    def _allocate_cache_state(self):
        self.cache = metile.Buffer.zeros(
            (
                2,
                self.config["num_hidden_layers"],
                self.config["num_key_value_heads"],
                self.capacity,
                self.config["head_dim"],
            ),
            dtype=self.dtype,
        )
        self.logits = metile.Buffer.empty((self.config["vocab_size"],), dtype=self.dtype)
        self.cached_tokens = 0

    def reset(self, *, clear_cache=True):
        """Discard the prefix, optionally retaining its physical cache contents.

        The default waits for GPU work and clears the cache. ``clear_cache=False``
        only invalidates logical state; it neither waits nor erases previous data.
        This is not secure erasure. Subsequent calls retain their usual control
        synchronization before reusing request buffers.
        """
        if type(clear_cache) is not bool:
            raise ValueError("clear_cache must be bool")
        if clear_cache:
            self.cache.numpy().fill(0)
        self.cached_tokens = 0

    def load_cache(self, cache):
        """Copy a native batch-one prefix; never import a quantized/rotating cache."""
        if cache is None:
            self.reset()
            return
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        if len(cache) != self.config["num_hidden_layers"]:
            raise ValueError("prefix cache must contain every model layer")
        offsets = set()
        imported = []
        for layer in cache:
            if type(layer) is not KVCache:
                raise ValueError("only ordinary MLX KVCache prefixes are supported")
            offset = layer.offset
            if not 0 <= offset < self.capacity:
                raise ValueError("prefix must leave capacity for the next token")
            offsets.add(offset)
            if not offset:
                imported.append(None)
                continue
            shape = (1, self.config["num_key_value_heads"], offset, self.config["head_dim"])
            keys = layer.keys[:, :, :offset, :]
            values = layer.values[:, :, :offset, :]
            if keys.shape != shape or values.shape != shape:
                raise ValueError("prefix cache shape does not match the model")
            if keys.dtype != self._mx_dtype or values.dtype != self._mx_dtype:
                raise ValueError("prefix cache must use the model's storage dtype")
            mx.eval(keys, values)
            imported.append((np.asarray(keys)[0], np.asarray(values)[0]))
        if len(offsets) != 1:
            raise ValueError("all prefix cache layers must have the same offset")
        offset = offsets.pop()
        self.reset()
        destination = self.cache.numpy()
        for index, arrays in enumerate(imported):
            if arrays is not None:
                destination[0, index, :, :offset, :] = arrays[0]
                destination[1, index, :, :offset, :] = arrays[1]
        self.cached_tokens = offset

    def prepare(self, token, position):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token must be an integer ID")
        if not 0 <= token < self.config["vocab_size"]:
            raise ValueError("token is outside the vocabulary")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("position must be an integer")
        if not 0 <= position < self.capacity:
            raise ValueError("position is outside the allocated cache")
        if position > self.cached_tokens:
            raise ValueError("position would read an uninitialized cache prefix")
        dispatcher = qwen3_decode_megakernel[(1,)].prepare(
            self.embedding,
            self.layer_weights,
            self.final_norm,
            self.rotary,
            self.cache,
            self.logits,
            token,
            position,
            HIDDEN=self.config["hidden_size"],
            INTERMEDIATE=self.config["intermediate_size"],
            QUERY_HEADS=self.config["num_attention_heads"],
            KV_HEADS=self.config["num_key_value_heads"],
            HEAD_DIM=self.config["head_dim"],
            LAYERS=self.config["num_hidden_layers"],
            VOCAB=self.config["vocab_size"],
            MAX_CONTEXT=self.capacity,
            BLOCK=self.threads,
            EPS=self.config["rms_norm_eps"],
            STORAGE_DTYPE="f16" if self.dtype == np.float16 else "f32",
            STRICT_MATH=True,
        )
        MetalDevice.get().sync()
        self.cached_tokens = position + 1
        return dispatcher
