"""Chunked Qwen3 prefill with tiled DSL GEMMs and bounded intermediate storage.

Projection weights are transposed once during untimed setup. Every prompt
chunk reuses those packs and the same scratch buffers. Attention reads the
complete causal prefix, including earlier chunks; chunks are not independent
attention windows. Generation retains the staged decoder and GPU argmax.
"""

import math
from copy import deepcopy

import numpy as np

import metile
from benchmarks.megakernels.qwen3_staged_runtime import Qwen3Staged
from metile.runtime.metal_device import MetalDevice
from metile_kernels.megakernels import qwen3_prefill as stages
from metile_kernels.megakernels.qwen3 import qwen3_layer_offsets
from metile_kernels.megakernels.qwen3_prefill_projection import qwen3_prefill_projection
from metile_kernels.megakernels.qwen3_prefill_tiled_attention import qwen3_prefill_tiled_attention
from metile_kernels.megakernels.qwen3_rotary import qwen3_rotary_table


class Qwen3ChunkedPrefill(Qwen3Staged):
    """Append token batches with causal attention over a fixed-capacity KV cache.

    ``prefill`` appends tokens without computing vocabulary logits. Call
    ``project_last_prefill`` to predict from its final token, then ``greedy``
    and the inherited single-token ``forward`` for autoregressive generation.
    All three execution methods enqueue GPU work; host reads synchronize.
    """

    def __init__(
        self,
        model,
        capacity,
        threads=256,
        chunk_size=128,
        projection_backend="simdgroup",
        projection_tile=(64, 64, 32),
        attention_backend="tiled",
        lossless_decode_weights=True,
    ):
        if type(lossless_decode_weights) is not bool:
            raise ValueError("lossless_decode_weights must be bool")
        if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
            raise ValueError("chunk_size must be an integer in [1, 4096]")
        if projection_backend not in ("simdgroup", "tensor_ops"):
            raise ValueError("projection_backend must be simdgroup or tensor_ops")
        if attention_backend not in ("tiled", "matrix"):
            raise ValueError("attention_backend must be tiled or matrix")
        if (
            not isinstance(projection_tile, tuple)
            or len(projection_tile) != 3
            or any(type(size) is not int or size < 8 or size % 8 for size in projection_tile)
        ):
            raise ValueError("projection_tile must contain three positive multiples of eight")
        head_dimension = getattr(getattr(model, "args", None), "head_dim", None)
        maximum_head_dimension = 128 if attention_backend == "matrix" else 256
        if type(head_dimension) is int and head_dimension > maximum_head_dimension:
            raise ValueError(
                f"{attention_backend} prefill requires head_dim <= {maximum_head_dimension} "
                "for its shared KV tiles"
            )
        super().__init__(model, capacity, threads)
        self.chunk_size = chunk_size
        self.projection_backend = projection_backend
        self.projection_tile = projection_tile
        self.attention_backend = attention_backend
        self._rotary_prepared = False
        self._allocate_prefill_state()
        hidden = self.config["hidden_size"]
        intermediate = self.config["intermediate_size"]
        query_width = self.config["num_attention_heads"] * self.config["head_dim"]
        key_width = self.config["num_key_value_heads"] * self.config["head_dim"]
        layout = qwen3_layer_offsets(
            hidden,
            intermediate,
            self.config["num_attention_heads"],
            self.config["num_key_value_heads"],
            self.config["head_dim"],
        )
        original = self.layer_weights.numpy()
        packs = []
        for layer in range(self.config["num_hidden_layers"]):
            packed = {}
            for name, offset, rows, columns in (
                ("qkv", layout["q_proj"], query_width + 2 * key_width, hidden),
                ("output", layout["o_proj"], hidden, query_width),
                ("gate_up", layout["gate_proj"], 2 * intermediate, hidden),
                ("down", layout["down_proj"], hidden, intermediate),
            ):
                weights = original[layer, offset : offset + rows * columns].reshape(rows, columns)
                packed[name] = metile.Buffer(data=np.ascontiguousarray(weights.T))
            packs.append(packed)
        self.prefill_weights = tuple(packs)
        self.prefill_weight_bytes = sum(
            weight.nbytes for layer in self.prefill_weights for weight in layer.values()
        )
        if lossless_decode_weights:
            self.pack_decode_weights()
        self.config.update(
            prefill_chunk_size=chunk_size,
            prefill_projection_backend=projection_backend,
            prefill_projection_tile=list(projection_tile),
            prefill_strict_math=True,
            prefill_projection_relaxed_precision=False,
            prefill_prunes_unused_final_layer=True,
            fused_row_arithmetic=attention_backend == "matrix",
            prefill_attention=(
                {
                    "kind": "matrix_tiled_online_softmax",
                    "query_rows_per_threadgroup": 32,
                    "key_tile": 16,
                    "threads": 128,
                    "shared_padding": 0,
                    "unroll_mma": True,
                    "cached_query_fragments": self.config["head_dim"] // 8,
                    "query_kv_shared_storage": False,
                    "direct_device_memory": True,
                    "masked_device_tile_scratch_bytes": 256
                    * (2 if self.dtype == np.float16 else 4),
                    "softmax_lanes": 8,
                    "transpose_keys": True,
                    "register_stats": True,
                    "unroll_softmax": True,
                    "load_vector": 16,
                    "softmax_base2": True,
                    "softmax_exponential": "fast_exp2",
                    "denominator_fma": True,
                    "normalization": "divide",
                    "compiler_backend": "simdgroup_inline",
                }
                if attention_backend == "matrix"
                else {
                    "kind": "query_tiled_shared_kv",
                    "query_rows_per_threadgroup": threads // 32,
                    "key_tile": 16,
                    "softmax_partitions_per_query": 8,
                }
            ),
        )

    def _allocate_prefill_state(self):
        self._prefill_prepared = False
        self._prefill_body = ()
        self._prefill_cache_body = ()
        self._last_prefill_available = False
        hidden = self.config["hidden_size"]
        intermediate = self.config["intermediate_size"]
        query_width = self.config["num_attention_heads"] * self.config["head_dim"]
        key_width = self.config["num_key_value_heads"] * self.config["head_dim"]
        self.prefill_tokens = metile.Buffer.zeros((self.chunk_size,), dtype=np.int32)
        self.prefill_control = metile.Buffer.zeros((2,), dtype=np.int32)
        for name, width in (
            ("hidden", hidden),
            ("residual", hidden),
            ("normalized", hidden),
            ("qkv", query_width + 2 * key_width),
            ("queries", query_width),
            ("attention", query_width),
            ("projected", hidden),
            ("gate_up", 2 * intermediate),
            ("intermediate", intermediate),
        ):
            setattr(
                self,
                f"prefill_{name}",
                metile.Buffer.empty((self.chunk_size, width), dtype=self.dtype),
            )

    def fork_request(self):
        """Allocate an empty request sharing finalized, read-only model buffers.

        The source must be prepared. The returned request has the same capacity
        and geometry but needs its own ``prepare()`` to bind its private state.
        Neither request may mutate shared weights or rotary tables afterward.
        This does not make request methods thread-safe or execute them concurrently.
        """
        if not self._prepared or not self._prefill_prepared or not self._rotary_prepared:
            raise RuntimeError("prepare the source request before forking")
        MetalDevice.get().sync()
        request = object.__new__(type(self))
        request.config = deepcopy(self.config)
        for name in (
            "scratch_bytes",
            "_mx_dtype",
            "dtype",
            "capacity",
            "threads",
            "parameter_count",
            "chunk_size",
            "projection_backend",
            "projection_tile",
            "attention_backend",
            "prefill_weight_bytes",
            "_packed_decode_weights",
            "embedding",
            "layer_weights",
            "final_norm",
            "rotary",
            "decode_layer_weights",
            "decode_embedding",
        ):
            setattr(request, name, getattr(self, name))
        request.prefill_weights = tuple(dict(layer) for layer in self.prefill_weights)
        request._rotary_prepared = True
        request._allocate_cache_state()
        request._allocate_decode_state()
        request._allocate_prefill_state()
        return request

    def reset(self, *, clear_cache=True):
        super().reset(clear_cache=clear_cache)
        self._last_prefill_available = False

    def set_chunk_size(self, chunk_size):
        """Resize only batch scratch; weights, KV allocation and decoder stay shared."""
        if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
            raise ValueError("chunk_size must be an integer in [1, 4096]")
        self.reset()
        if chunk_size == self.chunk_size:
            return self
        for name in (
            "prefill_tokens",
            "prefill_hidden",
            "prefill_residual",
            "prefill_normalized",
            "prefill_qkv",
            "prefill_queries",
            "prefill_attention",
            "prefill_projected",
            "prefill_gate_up",
            "prefill_intermediate",
        ):
            previous = getattr(self, name)
            setattr(
                self,
                name,
                metile.Buffer.zeros((chunk_size, *previous.shape[1:]), dtype=previous.dtype),
            )
        self.chunk_size = chunk_size
        self.config["prefill_chunk_size"] = chunk_size
        self._prefill_body = ()
        self._prefill_cache_body = ()
        self._prefill_prepared = False
        return self

    def prepare(self):
        if self._prefill_prepared:
            self.reset()
            return self
        if not self._rotary_prepared:
            qwen3_rotary_table[(metile.cdiv(self.capacity * (self.config["head_dim"] // 2), 128),)](
                self.rotary,
                HEAD_DIM=self.config["head_dim"],
                MAX_CONTEXT=self.capacity,
                LOG2_BASE=float(np.float32(math.log2(self.config["rope_theta"]))),
                STRICT_MATH=True,
            )
            MetalDevice.get().sync()
            self._rotary_prepared = True
        super().prepare()
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
        attention_geometry = {
            "QUERY_HEADS": query_heads,
            "KV_HEADS": key_heads,
            "HEAD_DIM": dimension,
            "LAYERS": self.config["num_hidden_layers"],
            "MAX_CONTEXT": self.capacity,
        }
        fused_arithmetic = self.config["fused_row_arithmetic"]

        def bind(kernel, grid, *buffers, **constants):
            dispatch = kernel[grid].prepare(*buffers, **(common | constants))
            dispatch._concurrent = False
            return dispatch

        def project(source, weights, destination, rows, columns):
            tile_rows, tile_columns, tile_reduction = self.projection_tile
            dispatch = qwen3_prefill_projection[
                (metile.cdiv(self.chunk_size, tile_rows), metile.cdiv(rows, tile_columns))
            ].prepare(
                source,
                weights,
                destination,
                self.chunk_size,
                rows,
                columns,
                BLOCK_M=tile_rows,
                BLOCK_N=tile_columns,
                BLOCK_K=tile_reduction,
                STRICT_MATH=True,
                RELAXED_PRECISION=False,
                SCHEDULE=metile.Schedule(backend=self.projection_backend),
            )
            dispatch._concurrent = False
            return dispatch

        if self.attention_backend == "matrix":
            from metile_kernels.megakernels.qwen3_prefill_matrix_attention import (
                qwen3_prefill_matrix_attention,
            )

            attention_kernel = qwen3_prefill_matrix_attention
            attention_grid = (metile.cdiv(self.chunk_size, 32), query_heads)
            attention_constants = {
                "BLOCK": 128,
                "QUERY_TILE": 32,
                "KEY_TILE": 16,
                "SHARED_PADDING": 0,
                "UNROLL_MMA": True,
                "SOFTMAX_LANES": 8,
                "TRANSPOSE_KEYS": True,
                "REGISTER_STATS": True,
                "UNROLL_SOFTMAX": True,
                "LOAD_VECTOR": 16,
                "SOFTMAX_BASE2": True,
                "DIRECT_MEMORY": True,
                "SCHEDULE": metile.Schedule(backend="simdgroup_inline"),
            }
        else:
            attention_kernel = qwen3_prefill_tiled_attention
            attention_grid = (metile.cdiv(self.chunk_size, self.threads // 32), query_heads)
            attention_constants = {"KEY_TILE": 16, "PARTITIONS": 8}

        decode = list(self._body)
        for layer in range(self.config["num_hidden_layers"]):
            for stage_index, source, norm in (
                (0, self.hidden, "input_norm"),
                (5, self.residual, "post_norm"),
            ):
                decode[1 + 8 * layer + stage_index] = bind(
                    stages.qwen3_prefill_rmsnorm,
                    (1,),
                    source,
                    self.layer_weights,
                    self.normalized,
                    layer,
                    CHUNK=1,
                    WIDTH=hidden,
                    WEIGHT_OFFSET=layout[norm],
                    WEIGHT_STRIDE=layout["layer_size"],
                    EPS=self.config["rms_norm_eps"],
                    FUSED_ARITHMETIC=fused_arithmetic,
                )
            decode[1 + 8 * layer + 2] = bind(
                stages.qwen3_prefill_qk_rope,
                (metile.cdiv(query_heads + key_heads, self.threads // 32), 1),
                self.qkv,
                self.layer_weights,
                self.rotary,
                self.control,
                self.queries,
                self.cache,
                layer,
                CHUNK=1,
                DECODE=True,
                EPS=self.config["rms_norm_eps"],
                FUSED_ARITHMETIC=fused_arithmetic,
                LAYERS=self.config["num_hidden_layers"],
                MAX_CONTEXT=self.capacity,
                **geometry,
            )
            decode[1 + 8 * layer + 3] = bind(
                stages.qwen3_prefill_attention,
                (query_heads, 1),
                self.queries,
                self.cache,
                self.control,
                self.attention,
                layer,
                CHUNK=1,
                DECODE=True,
                **attention_geometry,
            )
        self._body = tuple(decode)
        self._projection = (
            bind(
                stages.qwen3_prefill_rmsnorm,
                (1,),
                self.hidden,
                self.final_norm,
                self.normalized,
                0,
                CHUNK=1,
                WIDTH=hidden,
                EPS=self.config["rms_norm_eps"],
                FUSED_ARITHMETIC=fused_arithmetic,
            ),
            self._projection[-1],
        )
        self.prefill_control.numpy()[:] = (0, 1)
        hidden_grid = (metile.cdiv(self.chunk_size * hidden, self.threads),)
        body = [
            bind(
                stages.qwen3_prefill_embedding,
                hidden_grid,
                self.prefill_tokens,
                self.prefill_control,
                self.embedding,
                self.prefill_hidden,
                CHUNK=self.chunk_size,
                HIDDEN=hidden,
                VOCAB=self.config["vocab_size"],
            )
        ]
        for layer, weights in enumerate(self.prefill_weights):
            body.append(
                bind(
                    stages.qwen3_prefill_rmsnorm,
                    (self.chunk_size,),
                    self.prefill_hidden,
                    self.layer_weights,
                    self.prefill_normalized,
                    layer,
                    CHUNK=self.chunk_size,
                    WIDTH=hidden,
                    WEIGHT_OFFSET=layout["input_norm"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    EPS=self.config["rms_norm_eps"],
                    FUSED_ARITHMETIC=fused_arithmetic,
                )
            )
            body.append(
                project(
                    self.prefill_normalized,
                    weights["qkv"],
                    self.prefill_qkv,
                    query_width + 2 * key_heads * dimension,
                    hidden,
                )
            )
            body.append(
                bind(
                    stages.qwen3_prefill_qk_rope,
                    (metile.cdiv(query_heads + key_heads, self.threads // 32), self.chunk_size),
                    self.prefill_qkv,
                    self.layer_weights,
                    self.rotary,
                    self.prefill_control,
                    self.prefill_queries,
                    self.cache,
                    layer,
                    CHUNK=self.chunk_size,
                    LAYERS=self.config["num_hidden_layers"],
                    MAX_CONTEXT=self.capacity,
                    EPS=self.config["rms_norm_eps"],
                    FUSED_ARITHMETIC=fused_arithmetic,
                    **geometry,
                )
            )
            body.append(
                bind(
                    attention_kernel,
                    attention_grid,
                    self.prefill_queries,
                    self.cache,
                    self.prefill_control,
                    self.prefill_attention,
                    layer,
                    CHUNK=self.chunk_size,
                    **attention_constants,
                    **attention_geometry,
                )
            )
            if layer == self.config["num_hidden_layers"] - 1:
                self._prefill_cache_body = tuple(body[:-1])
            body.append(
                project(
                    self.prefill_attention,
                    weights["output"],
                    self.prefill_projected,
                    hidden,
                    query_width,
                )
            )
            body.append(
                bind(
                    stages.qwen3_prefill_residual,
                    hidden_grid,
                    self.prefill_projected,
                    self.prefill_hidden,
                    self.prefill_residual,
                    CHUNK=self.chunk_size,
                    WIDTH=hidden,
                )
            )
            body.append(
                bind(
                    stages.qwen3_prefill_rmsnorm,
                    (self.chunk_size,),
                    self.prefill_residual,
                    self.layer_weights,
                    self.prefill_normalized,
                    layer,
                    CHUNK=self.chunk_size,
                    WIDTH=hidden,
                    WEIGHT_OFFSET=layout["post_norm"],
                    WEIGHT_STRIDE=layout["layer_size"],
                    EPS=self.config["rms_norm_eps"],
                    FUSED_ARITHMETIC=fused_arithmetic,
                )
            )
            body.append(
                project(
                    self.prefill_normalized,
                    weights["gate_up"],
                    self.prefill_gate_up,
                    2 * intermediate,
                    hidden,
                )
            )
            body.append(
                bind(
                    stages.qwen3_prefill_swiglu,
                    (metile.cdiv(self.chunk_size * intermediate, self.threads),),
                    self.prefill_gate_up,
                    self.prefill_intermediate,
                    CHUNK=self.chunk_size,
                    INTERMEDIATE=intermediate,
                    FUSED_ARITHMETIC=fused_arithmetic,
                )
            )
            body.append(
                project(
                    self.prefill_intermediate,
                    weights["down"],
                    self.prefill_projected,
                    hidden,
                    intermediate,
                )
            )
            body.append(
                bind(
                    stages.qwen3_prefill_residual,
                    hidden_grid,
                    self.prefill_projected,
                    self.prefill_residual,
                    self.prefill_hidden,
                    CHUNK=self.chunk_size,
                    WIDTH=hidden,
                )
            )
        self._prefill_body = tuple(body)
        self._last_hidden = bind(
            stages.qwen3_prefill_last_hidden,
            (metile.cdiv(hidden, self.threads),),
            self.prefill_hidden,
            self.prefill_control,
            self.hidden,
            CHUNK=self.chunk_size,
            HIDDEN=hidden,
        )
        MetalDevice.get().sync()
        self._prefill_prepared = True
        self.reset()
        return self

    def _validate_prefill_tokens(self, tokens):
        if not self._prefill_prepared:
            raise RuntimeError("prepare the chunked model before prefill")
        tokens = list(tokens)
        if any(
            type(token) is not int or not 0 <= token < self.config["vocab_size"] for token in tokens
        ):
            raise ValueError("prefill tokens must be integer IDs inside the vocabulary")
        if self.cached_tokens + len(tokens) > self.capacity:
            raise ValueError("prefill exceeds the allocated cache capacity")
        return tokens

    def prefill_chunk(self, tokens, *, final_chunk: bool):
        """Append one chunk, retaining final-token activations only when requested.

        A nonfinal chunk writes every layer's KV cache but skips the final
        layer's post-KV work. It cannot be projected. Calls preserve ordered
        dispatches and the existing host synchronization policy.
        """
        tokens = self._validate_prefill_tokens(tokens)
        if type(final_chunk) is not bool:
            raise ValueError("final_chunk must be bool")
        if not 1 <= len(tokens) <= self.chunk_size:
            raise ValueError("prefill chunk must contain between 1 and chunk_size tokens")
        self._logits_valid = False
        self._last_prefill_available = False
        destination = self.prefill_tokens.numpy()
        destination.fill(0)
        destination[: len(tokens)] = tokens
        self.prefill_control.numpy()[:] = (self.cached_tokens, len(tokens))
        dispatches = self._prefill_body if final_chunk else self._prefill_cache_body
        for dispatch in dispatches:
            dispatch()
        self.cached_tokens += len(tokens)
        self._last_prefill_available = final_chunk

    def prefill(self, tokens):
        tokens = self._validate_prefill_tokens(tokens)
        if not tokens:
            return
        for start in range(0, len(tokens), self.chunk_size):
            chunk = tokens[start : start + self.chunk_size]
            self.prefill_chunk(chunk, final_chunk=start + len(chunk) == len(tokens))

    def project_last_prefill(self):
        if not self._last_prefill_available:
            raise RuntimeError(
                "projection requires the most recent operation to be a nonempty prefill"
            )
        self._last_hidden()
        for dispatch in self._projection:
            dispatch()
        self._logits_valid = True

    def forward(self, token, position, project=True):
        super().forward(token, position, project)
        self._last_prefill_available = False

    def dispatches_per_prefill(self, tokens):
        if type(tokens) is not int or tokens < 0:
            raise ValueError("prefill token count must be a nonnegative integer")
        chunks = metile.cdiv(tokens, self.chunk_size)
        return chunks * (1 + 11 * self.config["num_hidden_layers"]) - 8 * max(chunks - 1, 0)
