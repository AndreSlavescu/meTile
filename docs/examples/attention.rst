Decode Attention
================

``metile.kernels.attention_decode`` implements single-query MHA/GQA/MQA decode as composable
meTile eDSL kernels. It never materializes the score matrix. Each SIMDgroup streams a
subset of key/value tokens, maintains a numerically stable online-softmax recurrence in
registers, and merges its maximum, denominator, and output partials.

.. code-block:: python

   from metile.kernels import attention_decode

   dispatch = attention_decode[(batch, query_heads)].prepare(
       query,
       key,
       value,
       output,
       context_length,
       head_dim ** -0.5,
       D=head_dim,
       KV_HEADS=key_value_heads,
   )
   dispatch()

The contiguous float32 layouts are query/output ``[batch, query_heads, D]`` and
key/value ``[batch, key_value_heads, tokens, D]``. Query heads must be divisible by
key/value heads, and ``D`` must be divisible by 32.

The runtime measures both a single-pass threadgroup and, for long contexts, a two-pass
form that partitions tokens across threadgroups and merges unnormalized online-softmax
partials in a second kernel. Both passes share one ordered Metal encoder and require no
host-visible intermediate synchronization.

The implementation is composed from reusable frontend operations rather than emitted
as a whole-kernel source template:

* ``scalar`` identifies loop-carried scalar SSA state.
* ``tile_range`` expresses the runtime token recurrence.
* ``simd_sum`` and ``simd_max`` map to native SIMD-scoped Metal reductions.
* ``fast_exp`` maps to Metal's fast exponential for online normalization.
* ``shared`` and ``barrier`` assemble partial results across SIMDgroups.

The autotuned wrapper searches single-pass threadgroup widths from 32 through 1024
threads plus two-pass token partitions and partial/merge threadgroup shapes. Its cache
key includes the launch grid as well as context length and head dimension, so a schedule
measured for a highly parallel head batch is not reused for a low-head decode. The
selected algorithm and its measured GPU latency persist across processes.
