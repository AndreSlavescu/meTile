"""Benchmark adapter for GPU-wide, ordered Qwen3 DSL stages.

Weights and cache layouts are shared with the original megakernel adapter.
This is a multi-dispatch implementation, not a persistent GPU-wide kernel.
Preparation allocates scratch, executes each stage once, then clears state.
"""

import numpy as np

import metile
from benchmarks.megakernels.qwen3_runtime import Qwen3Megakernel
from metile.runtime.metal_device import MetalDevice
from metile_kernels.megakernels import qwen3_staged as stages
from metile_kernels.megakernels.qwen3 import qwen3_layer_offsets
from metile_kernels.megakernels.qwen3_selection import (
    qwen3_argmax_finalize,
    qwen3_argmax_partials,
)


def _bfloat16_packable(values):
    if values.dtype != np.float32 or values.size % 2 or not values.flags.c_contiguous:
        return False
    bits = values.reshape(-1).view(np.uint32)
    for start in range(0, bits.size, 1 << 20):
        chunk = bits[start : start + (1 << 20)]
        if np.any(chunk & np.uint32(0xFFFF)) or np.any(
            (chunk & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
        ):
            return False
    return True


def _write_bfloat16_pairs(values, destination):
    if not _bfloat16_packable(values):
        raise ValueError("lossless packing requires finite, even-length, BF16-exact FP32 values")
    bits = values.reshape(-1).view(np.uint32)
    if (
        destination.dtype != np.uint32
        or destination.shape != (bits.size // 2,)
        or np.shares_memory(values, destination)
    ):
        raise ValueError("packed destination must contain one uint32 per pair of values")
    for start in range(0, bits.size, 1 << 20):
        chunk = bits[start : start + (1 << 20)]
        packed = (chunk[::2] >> np.uint32(16)) | chunk[1::2]
        written = destination[start // 2 : (start + chunk.size) // 2]
        written[:] = packed
        if not np.array_equal(written << np.uint32(16), chunk[::2]) or not np.array_equal(
            written & np.uint32(0xFFFF0000), chunk[1::2]
        ):
            raise RuntimeError("lossless weight packing failed its bitwise roundtrip")


class Qwen3Staged(Qwen3Megakernel):
    """Batch-one forward stages with device control and GPU greedy selection.

    ``forward`` enqueues work; reading logits or calling ``greedy`` waits for
    completion. Host control updates synchronize the previous forward before
    reusing its storage. No host waits occur between a forward's GPU stages.
    Instances are stateful and must not be shared between concurrent callers.
    """

    def __init__(self, model, capacity, threads=256):
        super().__init__(model, capacity, threads)
        self.decode_layer_weights = self.layer_weights
        self.decode_embedding = self.embedding
        self._packed_decode_weights = False
        self._allocate_decode_state()

    def _allocate_decode_state(self):
        hidden = self.config["hidden_size"]
        intermediate = self.config["intermediate_size"]
        query_width = self.config["num_attention_heads"] * self.config["head_dim"]
        key_width = self.config["num_key_value_heads"] * self.config["head_dim"]
        self.control = metile.Buffer.zeros((2,), dtype=np.int32)
        self.hidden = metile.Buffer.empty((hidden,), dtype=self.dtype)
        self.residual = metile.Buffer.empty((hidden,), dtype=self.dtype)
        self.normalized = metile.Buffer.empty((hidden,), dtype=self.dtype)
        self.qkv = metile.Buffer.empty((query_width + 2 * key_width,), dtype=self.dtype)
        self.queries = metile.Buffer.empty((query_width,), dtype=self.dtype)
        self.attention = metile.Buffer.empty((query_width,), dtype=self.dtype)
        self.intermediate = metile.Buffer.empty((intermediate,), dtype=self.dtype)
        self._body = ()
        self._projection = ()
        self._selection = ()
        self._logits_valid = False
        self._prepared = False
        self.max_projection_threadgroups = metile.cdiv(
            self.config["vocab_size"], self.threads // 32
        )

    def reset(self, *, clear_cache=True):
        super().reset(clear_cache=clear_cache)
        self._logits_valid = False

    def pack_decode_weights(self):
        """Pack exact BF16-valued FP32 weights without changing arithmetic or values."""
        if self._prepared:
            raise RuntimeError("pack decode weights before preparing dispatches")
        if self._packed_decode_weights:
            return True
        if self.dtype != np.float32 or any(
            self.config[name] % 2 for name in ("hidden_size", "intermediate_size")
        ):
            return False
        sources = (self.layer_weights.numpy(), self.embedding.numpy())
        if any(not _bfloat16_packable(values) for values in sources):
            return False
        packed = []
        for values in sources:
            destination = metile.Buffer.empty((values.size // 2,), dtype=np.uint32)
            _write_bfloat16_pairs(values, destination.numpy())
            packed.append(destination)
        self.decode_layer_weights, self.decode_embedding = packed
        self._packed_decode_weights = True
        elements = sum(values.size for values in sources)
        self.config["decode_weight_packing"] = {
            "enabled": True,
            "format": "fp32_high16x2_u32_lossless",
            "decoded_dtype": "float32",
            "storage_dtype": "uint32",
            "values_per_word": 2,
            "scopes": ["layer_weights", "embedding"],
            "consumers": ["decode_projections", "first_token_vocabulary_projection"],
            "element_count": elements,
            "unpacked_bytes": elements * 4,
            "packed_bytes": elements * 2,
            "verification": {
                "finite": True,
                "even_elements": True,
                "zero_low16_bits": True,
                "roundtrip_bitwise": True,
            },
            "original_buffers_retained": True,
        }
        return True

    def prepare(self):
        """Compile and bind all stages outside timing, then discard warmup state."""
        if self._prepared:
            self.reset()
            return self
        self.reset()
        self.control.numpy()[:] = (0, 0)
        hidden = self.config["hidden_size"]
        intermediate = self.config["intermediate_size"]
        query_heads = self.config["num_attention_heads"]
        key_heads = self.config["num_key_value_heads"]
        dimension = self.config["head_dim"]
        query_width = query_heads * dimension
        layout = qwen3_layer_offsets(hidden, intermediate, query_heads, key_heads, dimension)
        common = {
            "BLOCK": self.threads,
            "STORAGE_DTYPE": "f16" if self.dtype == np.float16 else "f32",
            "STRICT_MATH": True,
        }
        geometry = {
            "HIDDEN": hidden,
            "INTERMEDIATE": intermediate,
            "QUERY_HEADS": query_heads,
            "KV_HEADS": key_heads,
            "HEAD_DIM": dimension,
        }

        def bind(kernel, groups, *buffers, **constants):
            dispatch = kernel[(groups,)].prepare(*buffers, **constants, **common)
            dispatch._concurrent = False
            return dispatch

        def row_groups(rows):
            return metile.cdiv(rows, self.threads // 32)

        body = [
            bind(
                stages.qwen3_staged_embedding,
                metile.cdiv(hidden, self.threads),
                self.embedding,
                self.control,
                self.hidden,
                HIDDEN=hidden,
                VOCAB=self.config["vocab_size"],
            )
        ]
        for layer in range(self.config["num_hidden_layers"]):
            body.append(
                bind(
                    stages.qwen3_staged_rmsnorm,
                    1,
                    self.hidden,
                    self.layer_weights,
                    self.normalized,
                    layer,
                    WIDTH=hidden,
                    WEIGHT_OFFSET=layout["input_norm"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    EPS=self.config["rms_norm_eps"],
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_qkv,
                    row_groups((query_heads + 2 * key_heads) * dimension),
                    self.normalized,
                    self.decode_layer_weights,
                    self.qkv,
                    layer,
                    PACKED_WEIGHTS=self._packed_decode_weights,
                    **geometry,
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_qk_rope,
                    row_groups(query_heads + key_heads),
                    self.qkv,
                    self.layer_weights,
                    self.rotary,
                    self.control,
                    self.queries,
                    self.cache,
                    layer,
                    **geometry,
                    LAYERS=self.config["num_hidden_layers"],
                    MAX_CONTEXT=self.capacity,
                    EPS=self.config["rms_norm_eps"],
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_attention,
                    row_groups(query_heads),
                    self.queries,
                    self.cache,
                    self.control,
                    self.attention,
                    layer,
                    QUERY_HEADS=query_heads,
                    KV_HEADS=key_heads,
                    HEAD_DIM=dimension,
                    LAYERS=self.config["num_hidden_layers"],
                    MAX_CONTEXT=self.capacity,
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_residual,
                    row_groups(hidden),
                    self.attention,
                    self.decode_layer_weights,
                    self.hidden,
                    self.residual,
                    layer,
                    ROWS=hidden,
                    COLUMNS=query_width,
                    WEIGHT_OFFSET=layout["o_proj"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    PACKED_WEIGHTS=self._packed_decode_weights,
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_rmsnorm,
                    1,
                    self.residual,
                    self.layer_weights,
                    self.normalized,
                    layer,
                    WIDTH=hidden,
                    WEIGHT_OFFSET=layout["post_norm"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    EPS=self.config["rms_norm_eps"],
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_swiglu,
                    row_groups(intermediate),
                    self.normalized,
                    self.decode_layer_weights,
                    self.intermediate,
                    layer,
                    PACKED_WEIGHTS=self._packed_decode_weights,
                    **geometry,
                )
            )
            body.append(
                bind(
                    stages.qwen3_staged_residual,
                    row_groups(hidden),
                    self.intermediate,
                    self.decode_layer_weights,
                    self.residual,
                    self.hidden,
                    layer,
                    ROWS=hidden,
                    COLUMNS=intermediate,
                    WEIGHT_OFFSET=layout["down_proj"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    PACKED_WEIGHTS=self._packed_decode_weights,
                )
            )
        self._body = tuple(body)
        self._projection = (
            bind(
                stages.qwen3_staged_rmsnorm,
                1,
                self.hidden,
                self.final_norm,
                self.normalized,
                0,
                WIDTH=hidden,
                EPS=self.config["rms_norm_eps"],
            ),
            bind(
                stages.qwen3_staged_gemv,
                row_groups(self.config["vocab_size"]),
                self.normalized,
                self.decode_embedding,
                self.logits,
                0,
                ROWS=self.config["vocab_size"],
                COLUMNS=hidden,
                PACKED_WEIGHTS=self._packed_decode_weights,
            ),
        )
        self._prepare_selection()
        MetalDevice.get().sync()
        self._prepared = True
        self.reset()
        return self

    def _prepare_selection(self):
        vocabulary = self.config["vocab_size"]
        chunks = metile.cdiv(vocabulary, 256)
        self.partial_values = metile.Buffer.empty((chunks,), dtype=np.float32)
        self.partial_indices = metile.Buffer.empty((chunks,), dtype=np.int32)
        self.selected = metile.Buffer.empty((1,), dtype=np.int32)
        constants = {"VOCAB": vocabulary, "CHUNKS": chunks, "BLOCK": 256, "STRICT_MATH": True}
        self._selection = (
            qwen3_argmax_partials[(chunks,)].prepare(
                self.logits,
                self.partial_values,
                self.partial_indices,
                STORAGE_DTYPE="f16" if self.dtype == np.float16 else "f32",
                **constants,
            ),
            qwen3_argmax_finalize[(1,)].prepare(
                self.partial_values, self.partial_indices, self.selected, **constants
            ),
        )
        for dispatch in self._selection:
            dispatch._concurrent = False

    def forward(self, token, position, project=True):
        if not self._prepared:
            raise RuntimeError("prepare the staged model before forwarding")
        if type(token) is not int or not 0 <= token < self.config["vocab_size"]:
            raise ValueError("token must be an integer ID inside the vocabulary")
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError("position must be an integer inside the allocated cache")
        if position > self.cached_tokens:
            raise ValueError("position would read an uninitialized cache prefix")
        if type(project) is not bool:
            raise ValueError("project must be boolean")
        self._logits_valid = False
        self.control.numpy()[:] = (token, position)
        for dispatch in self._body:
            dispatch()
        if project:
            for dispatch in self._projection:
                dispatch()
        self.cached_tokens = position + 1
        self._logits_valid = project

    def greedy(self):
        if not self._logits_valid:
            raise RuntimeError("greedy selection requires a projected forward")
        for dispatch in self._selection:
            dispatch()
        return int(self.selected.numpy()[0])

    def dispatches_per_forward(self, project=True, include_greedy=False):
        if include_greedy and not project:
            raise ValueError("greedy selection requires projection")
        return 1 + 8 * self.config["num_hidden_layers"] + 2 * project + 2 * include_greedy
