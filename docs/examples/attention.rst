Decode Attention
================

``metile_kernels.attention_decode`` computes attention for one query token
per head. It supports multi-head attention (MHA), grouped-query attention
(GQA), and multi-query attention (MQA). The kernel streams keys and values
while maintaining an online softmax state, avoiding a full attention-score
matrix in device memory.

Run a small GQA example
-----------------------

This example uses four query heads sharing two key/value heads. All buffers
use contiguous float32 storage.

.. code-block:: python

   import numpy as np
   import metile
   from metile_kernels import attention_decode

   batch, query_heads, key_value_heads = 1, 4, 2
   context_length, head_dim = 37, 32
   rng = np.random.default_rng(0)
   query_data = rng.standard_normal((batch, query_heads, head_dim)).astype(np.float32)
   key_data = rng.standard_normal(
       (batch, key_value_heads, context_length, head_dim)
   ).astype(np.float32)
   value_data = rng.standard_normal(key_data.shape).astype(np.float32)
   query = metile.Buffer(data=query_data)
   key = metile.Buffer(data=key_data)
   value = metile.Buffer(data=value_data)
   output = metile.Buffer.zeros(query_data.shape, dtype=np.float32)

   dispatch = attention_decode[(batch, query_heads)].prepare(
       query, key, value, output, context_length, head_dim ** -0.5,
       D=head_dim, KV_HEADS=key_value_heads,
   )
   dispatch()

   heads_per_group = query_heads // key_value_heads
   expanded_keys = np.repeat(key_data, heads_per_group, axis=1)
   expanded_values = np.repeat(value_data, heads_per_group, axis=1)
   scores = np.einsum("bhd,bhtd->bht", query_data, expanded_keys) * head_dim ** -0.5
   probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
   probabilities /= probabilities.sum(axis=-1, keepdims=True)
   reference = np.einsum("bht,bhtd->bhd", probabilities, expanded_values)
   np.testing.assert_allclose(output.numpy(), reference, rtol=1e-4, atol=1e-5)

The layout contract is query/output ``[batch, query_heads, head_dim]`` and
key/value ``[batch, key_value_heads, tokens, head_dim]``. Batch size, head
counts, and context length must be positive. Query heads must be divisible by
key/value heads; the head dimension must be a positive multiple of 32.
The public decode wrapper validates float32 storage and buffer capacity.

Preparation and reuse
---------------------

``prepare`` runs candidate kernels when tuning is needed, so it can write
the output before returning. The returned dispatcher retains the supplied
buffers and fixed dimensions. It is useful for repeating that same problem;
prepare again when the context length, shape, or buffer bindings change.
Reading ``output.numpy()`` synchronizes pending work.

The search compares single-pass threadgroups with 32 through 1024 threads
and, for longer contexts, two-pass candidates. The latter split tokens across
threadgroups, then merge unnormalized softmax partials in another kernel.
Those two passes share an ordered Metal encoder, with no intermediate CPU
readback.

The tuning key includes batch size, query and key/value head counts, context
length, and head dimension, as well as implementation/device identity. The
selected configuration and measured latency can be reused across processes.

Inside the recurrence
---------------------

Each SIMDgroup tracks a running maximum, normalization sum, and weighted
output. When the maximum changes, the previous sum and output are rescaled
before the next contribution is added. Shared memory and a threadgroup
barrier make the final merge possible.

The implementation uses ordinary frontend operations: ``scalar`` for
loop-carried scalar state, ``tile_range`` for token iteration,
``simd_sum`` and ``simd_max`` for SIMDgroup reductions, and ``fast_exp``
for normalization. Its fast exponential and reduction order can produce
small differences from the NumPy reference.
