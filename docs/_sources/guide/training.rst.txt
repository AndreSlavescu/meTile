Training kernels and gradients
==============================

The separate ``metile-kernels`` package contains the DSL kernels.
``metile.backends`` allocates buffers, manages saved tensors, and coordinates
multi-kernel launches. Install both projects when working from source::

   python -m pip install -e . -e ./kernels

These APIs return explicit first-order gradients. They do **not** register
MLX or PyTorch autograd rules, automatically patch models, or implement
higher-order differentiation. :doc:`kernel-coverage` lists the contracts and
remaining work for each operation. Liger parity is a tracked target, not a
claim that every upstream operation or option is already supported.

Most native entry points accept NumPy arrays or ``metile.Buffer`` objects.
NumPy inputs are copied; saved Buffer inputs must not be mutated before
backward. Calling ``.numpy()`` synchronizes outstanding GPU work and returns a
NumPy view of the buffer's shared memory. Floating gradients use FP32 storage
even when inputs use FP16. Integer
indices, boolean masks, geometry, and fixed scalar configuration have no
gradient. Finite inputs and finite intermediate dot products are required.

Normalization
-------------

``metile.backends.training_norms`` provides RMSNorm, LayerNorm, and
residual-add RMSNorm forward/backward pairs. Inputs are two-dimensional rows
with widths 1--8192 and FP16 or FP32 storage. Statistics and all gradients use
FP32. Weight and bias have one value per column; parameter gradients are
summed across rows in a fixed order without atomics.

.. code-block:: python

   import numpy as np
   from metile.backends.training_norms import layer_norm_forward, norm_backward

   values = np.arange(21, dtype=np.float32).reshape(3, 7) / 10
   output, saved = layer_norm_forward(
       values, np.ones(7, dtype=np.float32), np.zeros(7, dtype=np.float32)
   )
   gradients = norm_backward(saved, np.ones_like(values))
   assert gradients.source.shape == values.shape
   assert gradients.weight.shape == gradients.bias.shape == (7,)

RMSNorm uses the standard zero-offset weight convention. LayerNorm computes
population variance from centered values. Both default to ``epsilon=1e-5``.
``add_rms_norm_forward`` returns ``(output, residual_sum, saved)`` and
normalizes the unrounded FP32 sum; both outputs use input storage dtype.
Its backward accepts ``residual_output_gradient=...`` for the second output,
defaulting to zero, and returns both source and residual gradients. Cast
rounding is not differentiated.

The conservative parameter-partial scratch bound is 128 MiB. There are no
in-place, BF16, arbitrary-axis or Liger casting-mode variants. See
:doc:`kernel-coverage` for the exact dtype, memory and gradient contract.

Gated activations
-----------------

.. code-block:: python

   import numpy as np
   from metile.backends.pointwise import activation_forward, activation_backward

   generator = np.random.default_rng(7)
   gate = generator.normal(size=(2, 128)).astype(np.float32)
   up = generator.normal(size=gate.shape).astype(np.float32)
   output, saved = activation_forward(gate, kind="silu", up=up)
   dgate, dup = activation_backward(saved, np.ones_like(gate))
   assert dgate.numpy().shape == gate.shape

Specify ``up`` for a gated activation, or leave it out for a unary operation.
Supported names are ``silu``, ``sigmoid``, ``gelu_tanh``,
``quick_gelu``, ``relu``, and ``tanh``. ``silu`` plus ``up`` is SwiGLU;
``gelu_tanh`` plus ``up`` is the tanh-GELU GeGLU variant. The legacy
sigmoid-based GELU approximation is called ``quick_gelu`` here; it is not
interchangeable with tanh-GELU. ReLU uses derivative zero at zero.

Rotary embeddings
-----------------

``metile.backends.pointwise.rope_forward(values, cosine, sine, ...)`` accepts
``values[rows, dimension]`` and already gathered, expanded coefficient tables
``[rows, rotary_dim / 2]``. ``rotary_dim`` defaults to the full dimension.
``interleaved=False`` pairs the two rotary halves; ``True`` pairs adjacent
components. The unrotated suffix is copied.

``rope_backward(saved, gradient)`` returns gradients for the values, cosine,
and sine arrays. It uses the transpose of the supplied rotation rather than
assuming an inverse: coefficients need not have unit norm. If a model adapter
broadcast coefficient rows, the adapter must also sum their gradients.
Q and K with different head counts require separate calls. Position IDs,
YaRN frequency construction, MRoPE axis selection, and cache layouts remain
the adapter's responsibility. See the original `RoFormer paper
<https://arxiv.org/abs/2104.09864>`_.

Softmax and cross entropy
-------------------------

``metile.backends.training_losses`` normalizes the final axis of contiguous
FP32 arrays. ``softmax_forward`` and ``log_softmax_forward`` return device
buffers; their backward functions take the saved output and a cotangent of
the same shape. Log-softmax subtracts the row maximum before the log
denominator, avoiding the underflow that can occur when computing
``log(softmax(x))``.

.. code-block:: python

   import numpy as np
   from metile.backends.training_losses import (
       softmax_forward, softmax_backward,
       cross_entropy_forward, cross_entropy_backward,
   )

   logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
   probabilities = softmax_forward(logits)
   dprobabilities = np.ones_like(logits)
   dlogits = softmax_backward(probabilities, dprobabilities)

   targets = np.array([2, -100], dtype=np.int32)
   loss, saved = cross_entropy_forward(
       logits, targets, reduction="mean", ignore_index=-100,
       label_smoothing=0.1, z_loss=0.001,
   )
   dlogits = cross_entropy_backward(saved)
   assert np.all(dlogits.numpy()[1] == 0)

Cross entropy accepts integer class indices. Uniform label smoothing uses
``(1 - epsilon) * one_hot + epsilon / classes``. The optional ``z_loss``
coefficient adds ``z_loss * logsumexp(logits)**2`` to each non-ignored row
before reduction. ``reduction`` is ``"none"``, ``"sum"``, or ``"mean"``;
mean divides by the number of non-ignored rows. An all-ignored loss and its
gradient are zero, including mean reduction. Backward defaults to unit
cotangents; an explicit cotangent has the loss shape, which is scalar for
sum/mean and the input's leading shape for no reduction.

The implementation supports 1--262144 columns, at most 65536 rows, and a
256 MiB bound on new outputs/workspace, excluding NumPy input copies.
It saves the maximum and log denominator separately to preserve small loss
differences on large common logit shifts. Targets are range-checked on the
CPU, so Buffer targets synchronize. Reductions use fixed-order sums rather
than atomics. FP16/BF16 storage, class weights, soft targets, and fused-linear
cross entropy are not implemented by this API.

Dense products
--------------

``metile.backends.training_matmul.matmul_forward(left, right)`` implements
``A[M,K] @ B[K,N]`` with FP32 output. ``matmul_backward(saved, gradient)``
returns ``dA = gradient @ B.T`` and ``dB = A.T @ gradient``. Optional
``activation="relu"``, ``"silu"``, ``"quick_gelu"``, or another supported
activation saves the unrounded preactivation for its derivative.

This baseline packs operands into FP32 and explicitly selects the SIMD-group
matrix backend; it does not silently switch to a reduced-precision backend.
Transposes and both gradient products execute as DSL kernels. The
default per-call workspace bound is 256 MiB, excluding caller inputs and
NumPy input copies. This is not a fused-linear loss, batched product,
quantized-weight gradient, or optimized throughput claim.

Stable softmax attention
------------------------

.. code-block:: python

   import numpy as np
   from metile.backends.attention import attention_forward, attention_backward

   generator = np.random.default_rng(8)
   query = generator.normal(size=(1, 4, 3, 32)).astype(np.float32)
   key = generator.normal(size=(1, 2, 5, 32)).astype(np.float32)
   value = generator.normal(size=key.shape).astype(np.float32)
   output, saved = attention_forward(query, key, value, causal=True)
   dquery, dkey, dvalue = attention_backward(saved, np.ones_like(query))
   assert dkey.numpy().shape == key.shape

Layouts are ``[batch, heads, sequence, dimension]``. Query heads must be a
multiple of KV heads; head dimensions are multiples of 32 from 32 through
256. The default causal offset is ``key_length - query_length``, appropriate
for cached suffix queries. Set ``causal_offset=0`` for top-left causality.
Boolean/uint8 masks mean *nonzero is visible*, not additive score bias.
Fully masked rows produce zero output and zero gradients.

The implementation independently adopts the numerical lessons in
`KohakuFA's precision analysis
<https://github.com/KohakuBlueleaf/KohakuFA/blob/041ac512ac474709ad910e505c545a1ab94853c2/docs/precision.md>`_:

* Store the raw score maximum and log denominator separately.
* Subtract the raw maximum before applying the attention scale.
* Keep accumulation and saved, unrounded output in FP32; apply dQ/dK scale
  after accumulation.

The kernels assign ownership of each output and use deterministic reductions
rather than floating-point atomics. They do not save a quadratic attention
matrix. This bounded Metal implementation prioritizes correctness; it is not
a port of KohakuFA's CUDA schedule, and upstream GPU performance numbers do
not apply here.

Different attention mechanisms stay different
---------------------------------------------

Recurrent delta attention
~~~~~~~~~~~~~~~~~~~~~~~~~

``metile.backends.gated_delta`` implements decay-before-correction delta
recurrences with explicit initial and final states. Query/key use
``[B,T,H,K]``; values use ``[B,T,H,V]``; state uses ``[B,H,K,V]``.
Scalar ``log_decay[B,T,H]`` gives Gated DeltaNet; per-key-channel
``log_decay[B,T,H,K]`` gives the Kimi Delta Attention core. These layouts
deliberately differ from softmax attention. All heads must match.

For each token, the recurrence is::

   decayed = exp(log_decay)[:, None] * previous_state
   error = value - decayed.T @ key
   state = decayed + beta * outer(key, error)
   output = scale * state.T @ query

Call ``gated_delta_forward(..., save_states=True)`` when backward is needed.
``gated_delta_backward`` consumes those states and both output and final-state
cotangents. It returns all six tensor-input gradients, including the initial
state, so gradients can flow through a chain of states rather than stopping
silently at a detached cache. The initial implementation is FP32 only, with
documented shape and workspace bounds. Retaining every state and the per-value
gradient partials can consume substantial memory.

The formula follows `Kimi Linear, equation (1)
<https://arxiv.org/html/2510.26692v1>`_; the scalar-decay model context is
`Qwen3-Next <https://qwen.ai/blog?id=qwen3-next>`_. Normalization, short
convolution, gate transformations, grouped-head expansion, padding policy,
and model-specific cache conversion are not part of this recurrence kernel.
Supporting the recurrence does not establish complete Kimi or Qwen model
compatibility.

Dual Chunk Attention
~~~~~~~~~~~~~~~~~~~~

``metile.backends.dual_chunk_attention`` takes three **already rotated** query
branches: intra-chunk, successive-chunk, and inter-chunk. It partitions keys
by global chunk coordinates and causality, then merges the branches under one
global softmax denominator. Summing three independently normalized outputs
would give a different result. Backward returns three query gradients and
combined K/V gradients. The positional helper follows pinned
`ChunkLlama semantics
<https://github.com/HKUNLP/ChunkLlama/tree/2add4d7c99d24dcc1ab03414cc602abb2e28cf2c>`_;
see also the `Dual Chunk Attention paper <https://arxiv.org/abs/2402.17463>`_
and `Qwen2.5-1M report
<https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf>`_.
This reference path materializes partition masks with a workspace guard. It
is causal, not bidirectional Qwen attention, and is not a complete RoPE/YaRN
or model-loading implementation.

Explicit attention graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

``metile.ir.attention_graph`` exposes distinct typed builders for these
mechanisms. ``compile_native_attention_graph`` in
``metile.backends.native_attention_graph`` dispatches those explicit nodes;
it rejects unrelated operators. Ordinary softmax graph patterns must not be
rewritten into delta recurrences or Dual Chunk Attention: their semantics and
state requirements differ.

The native executable supplies a reverse pass for all four operators.
``forward_with_context`` records a tape; ``backward`` takes output cotangents
and returns input gradients in graph-input order:

.. code-block:: python

   import numpy as np
   from metile.ir.graph_ir import GraphBuilder, TensorSpec
   from metile.ir.attention_graph import stable_attention
   from metile.backends.native_attention_graph import compile_native_attention_graph

   generator = np.random.default_rng(9)
   arrays = [generator.normal(size=(1, 2, 3, 32)).astype(np.float32)
             for _ in range(3)]
   builder = GraphBuilder()
   query, key, value = [
       builder.input(name, TensorSpec(array.shape, "f32"))
       for name, array in zip(("query", "key", "value"), arrays)
   ]
   output_node = stable_attention(builder, query, key, value, causal=True)
   executable = compile_native_attention_graph(builder.build(output_node))
   output, tape = executable.forward_with_context(*arrays)
   dquery, dkey, dvalue = executable.backward(tape, np.ones_like(arrays[0]))

For multiple graph outputs, supply a tuple or list of cotangents in output
order. ``None`` supplies zero, including for an unused recurrent final state.
GDN and KDA nodes expose both output and final state, so state edges between
nodes participate in the reverse pass. Shared inputs and repeated graph
outputs accumulate every contribution in FP32 without atomics. Disconnected
floating inputs receive zero buffers; masks receive ``None``.

Tape recording copies NumPy inputs, but device inputs and saved buffers must
remain unchanged until backward finishes. Changing graph structure or metadata
invalidates the tape. Backward does not overwrite inputs or cotangents. A
normal call to the executable retains neither a tape nor recurrent state
history.
Per-operator workspace limits do not bound the entire graph tape and its
accumulated gradients. This API does not register framework autograd or
differentiate operators outside the four explicit attention families.

Expression VJPs inside the DSL
------------------------------

``metile.vjp(output, inputs, cotangent)`` builds a reverse expression during
tracing. For example, inside a kernel::

   values = inputs.load((positions,))
   seed = seeds.load((positions,))
   transformed = metile.tanh(values) * values
   gradients.store((positions,), metile.vjp(transformed, values, seed))

It supports FP16/FP32 scalar and one-dimensional expressions: arithmetic,
exp/log/sqrt/tanh/abs, floating casts, selections, and sum/max/min reductions.
Requested inputs are differentiation leaves. Loads not requested as inputs
are frozen coefficients; memory is not automatically given a gradient buffer.
Scalar broadcast cotangents reduce back to the scalar; max/min ties split
equally; abs uses derivative zero at zero. Floating casts use the usual
real-arithmetic training convention, not a derivative of discrete rounding.

Differentiating an unsupported operation raises ``NotImplementedError``. This
is not whole-kernel autodiff: matrix dot, loop reversal, memory scatter
adjoints, layout/collective transposes, and alias-aware accumulation still
need dedicated rules. For ``fast_exp``, the VJP uses the exponential
derivative convention; it does not prove the derivative of the hardware
approximation.

Explicit state and strict math
------------------------------

Declare loop-carried state explicitly rather than relying on Python rebinding::

   state = metile.loop_state(initial.load((positions,)))
   for step in metile.tile_range(0, steps, 1):
       previous = state.value
       state.update(previous * 0.9 + increments.load((step, positions)))
   output.store((positions,), state.value)

``.value`` creates a snapshot at that program point; ``.update`` assigns
same-dtype, same-shape state. A loop that never iterates leaves the initial state
unchanged.
State is per-thread scalar/one-dimensional tile storage, not shared memory.
Explicit ``ThreadLayout`` state is not supported yet. A state declaration
must dominate its uses. This API enables recurrences, not their automatic
reverse-mode differentiation.

Training backends pass ``STRICT_MATH=True`` to disable Metal fast-math in both
offline and runtime compilation. The setting is part of the cache identity.
It does not turn explicit ``fast_exp`` into an accurate intrinsic, prove every
compiler rewrite IEEE-exact, or change a selected matrix backend's precision.
Geometry and bounds contracts still apply.
