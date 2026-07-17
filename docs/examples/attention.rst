Decode Attention
================

``kernels.attention.attention_decode`` implements single-query MHA decode as an ordinary
meTile eDSL kernel. It never materializes the score matrix. Each SIMDgroup streams a
subset of key/value tokens, maintains a numerically stable online-softmax recurrence in
registers, and merges its maximum, denominator, and output partials through threadgroup
memory.

.. code-block:: python

   from kernels.attention import attention_decode

   dispatch = attention_decode[(num_heads,)].prepare(
       query,
       key,
       value,
       output,
       context_length,
       head_dim ** -0.5,
       D=head_dim,
   )
   dispatch()

The contiguous float32 layouts are query/output ``[heads, D]`` and key/value
``[heads, tokens, D]``. The current kernel requires ``D`` to be divisible by 32.

The implementation is composed from reusable frontend operations rather than emitted
as a whole-kernel source template:

* ``scalar`` identifies loop-carried scalar SSA state.
* ``tile_range`` expresses the runtime token recurrence.
* ``simd_sum`` and ``simd_max`` map to native SIMD-scoped Metal reductions.
* ``fast_exp`` maps to Metal's fast exponential for online normalization.
* ``shared`` and ``barrier`` assemble partial results across SIMDgroups.

The autotuned wrapper searches threadgroup widths from 64 through 1024 threads. Its
cache key includes the launch grid as well as context length and head dimension, so a
schedule measured for a highly parallel head batch is not reused for a low-head decode.
