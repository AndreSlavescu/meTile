API Reference
=============

Use kernel operations inside ``@metile.kernel``, where tracing records GPU
work. Use buffers, launchers and tuning APIs from host Python code. This page
indexes both; see :doc:`/guide/language` for tracing rules and
:doc:`/guide/memory` for storage, dtypes and synchronization.

Install ``metile-kernels`` for ready-made operations under ``metile_kernels``,
including ``metile_kernels.gemm.matmul`` and
``metile_kernels.attention.attention_decode_kernel``. Their source lives in
``kernels/src/metile_kernels/``. The library depends on the compiler, not the
other way around. See :doc:`/getting-started/install`.

Host-side decode attention is provided by
``metile.backends.attention_runtime.attention_decode``. This optional backend
requires the kernel library, but not MLX. See :doc:`/examples/attention` for
its launcher API and migration from the former kernel-package import.

Training and Manual VJPs
------------------------

See :doc:`/guide/training` for examples, precision contracts, and saved-buffer
lifetime rules, and :doc:`/guide/kernel-coverage` for the Liger parity checklist.
These are native APIs, not automatic framework-autograd registrations.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Entry points
     - Contract
   * - ``metile.backends.pointwise.activation_forward / activation_backward``
     - Unary or gated activations; separate gradients for both gate and up
   * - ``metile.backends.pointwise.rope_forward / rope_backward``
     - Full/partial RoPE with pre-expanded coefficient tables and all three input gradients
   * - ``metile.backends.training_norms.rms_norm_forward / layer_norm_forward / add_rms_norm_forward / norm_backward``
     - FP32 statistics, input and affine-parameter gradients, residual cotangents
   * - ``metile.backends.training_matmul.matmul_forward / matmul_backward``
     - FP32 two-dimensional dense products and optional activation derivatives
   * - ``metile.backends.training_losses.softmax_forward / softmax_backward / log_softmax_forward / log_softmax_backward``
     - Final-axis FP32 normalization and its explicit cotangent map
   * - ``metile.backends.training_losses.cross_entropy_forward / cross_entropy_backward``
     - Integer-label cross entropy, ignored labels, smoothing and optional z-loss
   * - ``metile.backends.attention.attention_forward / attention_backward``
     - Stable dense/causal/masked attention, MHA/GQA/MQA, and dQ/dK/dV
   * - ``metile.backends.gated_delta.gated_delta_forward / gated_delta_backward``
     - Scalar/channel-decay recurrences with all six tensor-input gradients
   * - ``metile.backends.dual_chunk_attention.dual_chunk_attention_forward / dual_chunk_attention_backward``
     - Three pre-rotated branches, one global normalization, and branch/K/V gradients
   * - ``metile.vjp(output, inputs, cotangent)``
     - Trace-time expression VJP; unsupported differentiated operators fail explicitly
   * - ``metile.loop_state(initial)``
     - Explicit mutable per-lane state with ``.value`` snapshots and ``.update(next_value)``

Raw device implementations live in ``metile_kernels.training_activations``,
``rope``, ``training_norms``, ``training_losses``, ``stable_attention``,
``gated_delta``, and ``dual_chunk_attention``. Allocation and dispatch logic
stays outside those kernel modules.

Kernel Definition & Launch
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``@metile.kernel``
     - Decorate a Python function for GPU compilation
   * - ``kernel[grid](*args, **constexprs)``
     - Launch kernel with given grid shape and compile-time constants
   * - ``metile.constexpr``
     - Compile-time parameter annotation; pass values by keyword at launch
   * - ``kernel[grid].prepare(*args, **constexprs)``
     - Execute once, synchronize, and bind a reusable dispatcher to these arguments
   * - ``dispatch()``
     - Enqueue work using the prepared buffers, scalar values, and grid

Preparation can modify outputs. For a reusable dispatcher, pass explicit
``Buffer`` objects: later calls do not repeat implicit NumPy conversion or
copy results back to the original arrays. The ordinary launch cache depends
on input dtypes and relevant compilation settings as well as constexprs.

Pass ``STRICT_MATH=True`` to disable Metal fast-math in both compilation paths.
This boolean is part of compilation cache identity. It does not override an
explicit fast intrinsic or change the selected matrix arithmetic precision.


Buffers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.Buffer(data=np_array)``
     - Allocate shared Metal storage and copy the array's contents into it
   * - ``metile.Buffer.zeros(shape, dtype=np.float32)``
     - Allocate a zeroed buffer with the requested shape and storage dtype
   * - ``metile.Buffer.empty(shape, dtype=np.float32)``
     - Allocate a buffer whose initial contents are unspecified
   * - ``metile.Buffer.from_numpy(np_array)``
     - Copy a NumPy array into a new shared Metal buffer
   * - ``buf.numpy()``
     - Synchronize pending GPU work and return a writable view of the allocation

Keep the buffer alive until its NumPy view is no longer needed. Storage dtypes
must be supported by the launcher and the selected operation; float64 arrays are not
supported kernel inputs. Declaring a tensor view does not convert storage.


Program Identity
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.program_id(axis)``
     - Threadgroup index along ``axis`` (0, 1, or 2)
   * - ``metile.thread_id()``
     - Thread index within the threadgroup
   * - ``metile.simd_lane_id()``
     - Lane index within the simdgroup (0-31)


Index Generation
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.arange(start, end, *, layout=None)``
     - Half-open integer range for constant bounds, with optional thread ownership
   * - ``metile.cdiv(a, b)``
     - Ceiling division: ``ceil(a / b)``
   * - ``metile.next_power_of_2(n)``
     - Smallest power of 2 >= ``n``

For a traced scalar ``start``, the current ``arange`` overload treats its
second argument as a length. Prefer ``offset + metile.arange(0, BLOCK)`` to
make a dynamic origin explicit.


Thread Ownership
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.ThreadLayout(bit_order, xor_mask=0, elements_per_thread=1)``
     - Immutable register/thread-to-logical-index bit permutation, with 1/2/4/8/16/32 values per thread
   * - ``metile.ThreadLayout.identity(size, elements_per_thread=1)``
     - Identity ownership for a power-of-two tile on 32--1024 threads
   * - ``metile.convert_layout(value, layout)``
     - Preserve logical values while redistributing physical owners

See :doc:`/guide/thread-layouts` for supported operations and communication rules.


Tensor Views
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tensor(pointer, *, shape, strides=None, access="readwrite", block_shape=None, address_space=None)``
     - Declare logical shape, memory layout, and access permissions
   * - ``view.load(indices, other=0, *, scratch=None)``
     - Load coordinates with declared bounds and a scalar fill value
   * - ``view.store(indices, value, *, scratch=None)``
     - Store coordinates within declared bounds

See :doc:`/guide/tensor-memory` for tensor declarations, tiled matrix support,
integer-range constraints, and compiler responsibilities. Strides are in
elements. Bounds checks use the declared shape; the caller must supply enough
storage for its valid addresses. Matrix tile loads currently require zero fill.

For ``Schedule(backend="simdgroup_inline")``, device ``block_shape=(8, 8)``
accesses require an explicit ``scratch`` tensor. It must directly reference
a shared allocation with shape ``(BLOCK // 4, 8)``, strides ``(8, 1)``,
read/write access and matching FP16 or FP32 storage dtype. Keep the allocation
exclusive to these accesses; the compiler handles its masked-tail copies and
SIMD-group barriers. Scalar and shared-memory accesses reject ``scratch``.
Scalar shared-memory pointers in this backend must resolve to a declared
shared allocation through pointer offsets only; conditional pointer selection
is unsupported.

Device views have constant positive row-major or column-major strides and
SIMD-uniform bases, logical extents and origins. Complete tiles access device
memory directly; incomplete loads zero-fill and incomplete stores skip invalid
coordinates. Bounds refer to valid data, not spare allocation capacity.
A device allocation used by fragments cannot also be read and written through
any alias within the kernel. Separate input/output parameters must reference
disjoint storage at launch. See :doc:`/guide/execution-schedules` for a complete
DSL example and the shared-memory fragment contract.


Element-wise Memory
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.load(ptr, mask=None)``
     - Load elements; masked-off lanes read 0
   * - ``metile.store(ptr, value, mask=None)``
     - Store elements; masked-off lanes are skipped


Tile Memory
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tile_load(ptr, row, col, stride, shape)``
     - Legacy two-dimensional tile load with an element row stride
   * - ``metile.tile_store(ptr, row, col, stride, value, shape)``
     - Legacy two-dimensional tile store with an element row stride
   * - ``metile.zeros(shape, dtype="f32")``
     - Zero-initialized tile (accumulator init)
   * - ``metile.dot(a, b, acc)``
     - Return the tile matrix multiply-accumulate result ``acc + a @ b``

For new matrix kernels, prefer ``tensor(..., block_shape=...)`` loads and
stores, which declare full logical bounds. Backend-specific dtype, tile, and
layout restrictions are described in :doc:`/guide/tile-ops`.


Control Flow
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tile_range(start, end, step=1, num_stages=1)``
     - Traced loop with an exclusive end; staging support depends on lowering
   * - ``metile.scalar(value, dtype=None)``
     - Explicit scalar SSA value for loop-carried recurrences


Math Operations
---------------

All operate element-wise on scalars and tiles:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Description
   * - ``metile.exp(x)``
     - Exponential
   * - ``metile.exp2(x)``
     - Base-two exponential
   * - ``metile.fast_cos(x)``
     - Cosine in radians using Metal's ``fast::cos`` intrinsic
   * - ``metile.fast_exp(x)``
     - Exponential using Metal's fast-math intrinsic
   * - ``metile.fast_exp2(x)``
     - Base-two exponential using Metal's ``fast::exp2`` intrinsic
   * - ``metile.fast_sin(x)``
     - Sine in radians using Metal's ``fast::sin`` intrinsic
   * - ``metile.fma(a, b, c)``
     - Fused multiply-add: ``a * b + c`` with one final rounding
   * - ``metile.log(x)``
     - Natural logarithm
   * - ``metile.sqrt(x)``
     - Square root
   * - ``metile.rsqrt(x)``
     - Reciprocal square root using Metal's ``precise::rsqrt`` intrinsic
   * - ``metile.abs(x)``
     - Absolute value
   * - ``metile.tanh(x)``
     - Hyperbolic tangent
   * - ``metile.where(cond, x, y)``
     - Conditional select
   * - ``metile.maximum(a, b)``
     - Element-wise max
   * - ``metile.minimum(a, b)``
     - Element-wise min
   * - ``metile.cast(value, dtype)``
     - Convert a scalar or tile to a supported IR dtype such as ``"f32"``
   * - ``metile.bitcast(value, dtype)``
     - Reinterpret ``f32``/``i32``/``u32`` element bits without numeric conversion

Supported Python arithmetic and comparisons build elementwise operations.
``where`` selects between already-computed values; it does not make an
unmasked memory access safe. Mask the load itself or use a tensor view.

``bitcast`` preserves all 32 bits of each element, including signed zero and
NaN payloads, using Metal's ``as_type`` operation. The input must be an
explicitly typed scalar or tile proxy; use ``scalar(value, dtype=...)`` for
literals. Tile shape and thread ownership are unchanged. Identical source
and destination dtypes are an identity. Other widths, booleans, pointers,
and matrix-fragment reinterpretation are unsupported. Expression VJP rejects
bit reinterpretation rather than inventing a derivative.

``rsqrt`` emits the reciprocal-square-root operation directly rather than
dividing by ``sqrt``; their floating-point rounding can differ. For positive
finite inputs, its expression VJP uses ``-0.5 * rsqrt(x)**3`` multiplied by
the incoming cotangent. This is the derivative of the real-valued function,
not of floating-point rounding.

The ``exp2`` and ``fast_exp2`` VJPs multiply the incoming cotangent by
``ln(2)`` times the computed output.
The fast cosine and sine VJPs use ``-fast_sin(x)`` and ``fast_cos(x)``,
respectively, as real-function derivatives evaluated with the paired
intrinsic; they do not differentiate approximation or rounding errors.

``fast_exp2`` explicitly selects an approximate intrinsic without enabling
fast math for the rest of a ``STRICT_MATH=True`` kernel. Use it only when
the approximation is acceptable; it is not interchangeable with ``exp2``
for bitwise comparisons.

``fma`` explicitly requests Metal's fused operation even under strict math.
This follows Apple's guidance on `explicit FMA without fast math
<https://developer.apple.com/videos/play/wwdc2016/606/>`_.
It accepts matching ``f16`` or ``f32`` operands, equal-shaped tiles, and
scalar broadcasting; Python numeric literals adopt the proxy operands'
dtype. Cast mixed proxy dtypes explicitly. Its expression VJP contributes
``cotangent * b``, ``cotangent * a``, and ``cotangent`` to the three inputs,
with reductions for broadcast scalar inputs. These are real-function
derivatives, not derivatives of rounding. Ordinary multiplication followed
by addition is not rewritten into ``fma`` under strict math.

For example, an online softmax can request these operations locally::

   factor = metile.fast_exp2(previous_maximum - updated_maximum)
   denominator = metile.fma(previous_denominator, factor, local_sum)


Reductions
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Description
   * - ``metile.sum(x)``
     - Sum-reduce tile to scalar
   * - ``metile.max(x)``
     - Max-reduce tile to scalar
   * - ``metile.min(x)``
     - Min-reduce tile to scalar

These operations reduce a supported logical tile. By contrast, ``simd_sum``
and ``simd_max`` below reduce only within the current SIMDgroup. Choose padding
that matches the reduction's identity; see :doc:`/examples/softmax`.


Simdgroup Operations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.simdgroup_role(role, num_roles=2, num_sgs=0)``
     - Context manager: execute on a subset of simdgroups
   * - ``metile.simd_shuffle_xor(value, mask)``
     - XOR-based lane exchange within a simdgroup
   * - ``metile.simd_broadcast(value, lane)``
     - Broadcast from one lane to all lanes
   * - ``metile.simd_sum(value)``
     - Native sum across the current simdgroup
   * - ``metile.simd_max(value)``
     - Native maximum across the current simdgroup
   * - ``metile.barrier()``
     - Synchronize participating threads and order threadgroup memory accesses
   * - ``metile.shared(size, dtype="f32")``
     - Allocate uninitialized threadgroup memory

All participating threads must reach a threadgroup barrier. Role regions do
not synchronize with each other merely because they appear in source order.


Tile Scheduling
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.Schedule(...)``
     - Checked backend, SIMD-group, staging, vector-width and buffering requirements
   * - ``dispatch.explain()``
     - JSON report of the selected schedule and materialized compiler decisions
   * - ``metile.tile_swizzle(pid_m, pid_n, pattern="auto", block_size=4)``
     - Request automatic, linear, diagonal, grouped2/4/8, Morton, or Hilbert traversal

Pass the immutable requirements as ``SCHEDULE=metile.Schedule(...)`` to a kernel
launch or ``prepare`` call. See :doc:`/guide/execution-schedules` for supported
combinations, inspection and pointwise GEMM fusion.

The opt-in ``simdgroup_inline`` backend composes multiple ``dot`` operations
inside scalar/loop DSL programs. It uses 8-by-8 shared or device tiles, explicit
scratch for device tails, and FP32 register-fragment accumulators. See the composable matrix
fragment example and restrictions in :doc:`/guide/execution-schedules`.


Autotuning
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.autotune(configs, key, warmup=5, rep=20, verbose=True)``
     - Decorator for automatic parameter search
   * - ``metile.Config(num_simdgroups=None, num_stages=1, **constexprs)``
     - Candidate constants and optional SIMDgroup/staging requests
   * - ``autotuned[grid].prepare(*args, **kwargs)``
     - Select a configuration, execute preparation, and return a bound dispatcher

Use a callable grid when candidates change tile dimensions. Tuning runs
kernels and can modify their outputs; see :doc:`/guide/autotuning` for cache
keys and repeated-dispatch behavior.


Block Scaling
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.BlockScaledWeight.quantize(weight, format="mxfp4")``
     - Quantize a KxN weight to MXFP4 or MXFP8 with E8M0 group scales
   * - ``weight.dequantize()``
     - Reconstruct a dense NumPy array for inspection or reference checks
   * - ``metile.block_scaled_matmul(activations, weight, output=None)``
     - Run the automatically tuned fused block-scaled MPP kernel
   * - ``metile.prepare_block_scaled_matmul(activations, weight, output)``
     - Return a reusable dispatcher for the selected block-scaled tile family

The block-scaled fast path requires Metal 4 tensor operations, two-dimensional
activation/output buffers, M and N divisible by 64, and K divisible by 32.
Quantization changes the represented weights; compare numerical results with
an appropriate reference and tolerance.


Model Integration
-----------------

``metile.compile(model, *, verify=True, features=..., tolerance=...)`` modifies
a loaded MLX-LM model in place and returns a ``CompileReport`` listing accepted
and declined replacements. Call ``report.restore()`` to restore the previous
implementations. Verification checks a finite probe; it does not prove
correctness for every model input. See :doc:`/guide/mlx-backend` for setup,
verification settings and the limits of that check.


Related Host APIs
-----------------

``Layout``, ``make_layout``, ``row_major``, and ``col_major`` describe index
layouts on the host. They are distinct from ``ThreadLayout``, which specifies
physical thread/register ownership inside a kernel.

``GlobalAddressSpace``, ``TensorView``, ``TiledView``, and ``KernelPipeline``
provide host-side helpers for allocating memory, constructing views and
composing pipelines. These are separate from the traced ``metile.tensor``
declaration. For graph construction and fusion planning, see
:doc:`/guide/graph-fusion`.
