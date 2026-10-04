Language Reference
==================

meTile's kernel language is embedded in Python. A decorated function executes
with symbolic arguments during tracing; its operations become a GPU program.
Python still controls tracing, so a Python ``if`` or ``range`` must depend
on values known at compile time. Use ``where`` for a per-element selection
and ``tile_range`` for a traced loop.

The snippets on this page illustrate individual operations inside a kernel.
For complete programs with launch code and numerical checks, start with
:doc:`/getting-started/first-kernel` or :doc:`/examples/matmul`.

Define and launch a kernel
--------------------------

.. code-block:: python

   @metile.kernel
   def my_kernel(input_ptr, output_ptr, count, BLOCK: metile.constexpr):
       ...

   my_kernel[(metile.cdiv(count, 256),)](
       input_buffer, output_buffer, count, BLOCK=256
   )

Buffer arguments become device pointers with their storage dtype. For example,
float32 buffers use ``device float*`` and float16 buffers use
``device half*``. Python integers and floats become 32-bit integer and
float scalar arguments. Pass runtime arguments in signature order and
``constexpr`` arguments by keyword, as shown above.

The launcher recognizes the actual ``metile.constexpr`` annotation object.
Avoid postponed/string annotations on kernel parameters: a file containing
``from __future__ import annotations`` does not currently preserve this
identity check.

A launch grid is a tuple of program counts. Elementwise kernels support one,
two, or three axes; the standard GEMM path uses a two-dimensional output grid.
Each program is a Metal threadgroup. A grid is not the number of elements or
the number of threads inside each group.

``kernel[grid].prepare(...)`` performs a launch, synchronizes, and returns
a dispatcher bound to those resources and scalar values. Preparation is not
a compile-only operation. Prefer explicit ``Buffer`` arguments when reusing
a dispatcher; see :doc:`memory`.

Program identity and indices
----------------------------

.. function:: metile.program_id(axis)

   Return the program's threadgroup coordinate along axis 0, 1, or 2.

.. function:: metile.arange(start, end, *, layout=None)

   For integer ``start`` and ``end``, create the half-open sequence
   ``[start, end)``. Its length is a compile-time tile size. For clarity,
   write ``offset + metile.arange(0, BLOCK)`` when the origin is dynamic.

   The current overload also accepts a traced scalar ``start``; in that
   form, the second argument is the tile's length, not an endpoint.

   An optional ``ThreadLayout`` assigns logical elements to physical
   threads and registers. See :doc:`thread-layouts` for supported geometry.

.. function:: metile.cdiv(numerator, denominator)

   Ceiling division, commonly used on the host to compute a launch grid.

.. function:: metile.next_power_of_2(value)

   Return the smallest power of two greater than or equal to the value.
   Use it on host integers when selecting a tile size.

Tensor declarations
-------------------

.. function:: metile.tensor(pointer, *, shape, strides=None, access="readwrite", block_shape=None, address_space=None)

   Declare a logical view of existing storage. Shapes and strides are tuples;
   strides are measured in elements and default to contiguous row-major
   storage. Access is ``"read"``, ``"write"``, or ``"readwrite"``.
   An explicit address space must match the pointer's allocation.

   Declare views near the start of the kernel, then use ``view.load(indices,
   other=0)`` and ``view.store(indices, value)``. Loads and stores check
   logical coordinate bounds. The caller remains responsible for allocation
   capacity and signed 32-bit index/offset arithmetic.

   ``block_shape`` selects supported two-dimensional matrix-tile access.
   It does not allocate memory or set thread/register ownership.
   See :doc:`tensor-memory` for the full bounds, dtype, and layout contract.

.. code-block:: python

   inputs = metile.tensor(input_ptr, shape=(count,), access="read")
   outputs = metile.tensor(output_ptr, shape=(count,), access="write")
   positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
   outputs.store((positions,), inputs.load((positions,)) * 2.0)

Raw memory operations
---------------------

.. function:: metile.load(ptr, mask=None)

   Load elements from a pointer expression. Masked-off elements read zero.
   Without a mask, the caller must ensure every address is valid.

.. function:: metile.store(ptr, value, mask=None)

   Store through a pointer expression. Masked-off stores are skipped.

.. code-block:: python

   positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
   valid = positions < count
   values = metile.load(input_ptr + positions, mask=valid)
   metile.store(output_ptr + positions, values, mask=valid)

.. function:: metile.tile_load(ptr, row_offset, col_offset, stride, shape)

   Load a two-dimensional tile. ``stride`` is the row stride in elements;
   ``shape`` is the tile's ``(rows, columns)``. This legacy interface
   does not declare full tensor bounds. New matrix kernels should use
   ``tensor(..., block_shape=...)`` so lowering can validate dimensions
   and memory layout.

.. function:: metile.tile_store(ptr, row_offset, col_offset, stride, value, shape)

   Store a matrix tile using the same legacy layout convention.

Accumulators and matrix multiply
--------------------------------

.. function:: metile.zeros(shape, dtype="f32")

   Create a zero-valued tile, commonly a float32 matrix accumulator.

.. function:: metile.dot(left, right, accumulator)

   Return ``accumulator + left @ right``. The left and right tile shapes
   must have compatible reduction dimensions. Supported matrix kernels
   accumulate in float32.

   Lowering selects SIMDgroup matrix operations or Metal 4 tensor operations
   according to device/toolchain support and the execution schedule. The
   direct NAX path has additional restrictions. See :doc:`tile-ops` and
   :doc:`/examples/matmul`.

Control flow and scalar state
-----------------------------

.. function:: metile.tile_range(start, end, step=1, num_stages=1)

   Record a GPU loop with an exclusive end. Its body runs once during tracing,
   producing the operations repeated at execution time. A positive
   compile-time integer step is the usual tiling pattern.

   ``num_stages`` records a staging request; support depends on the chosen
   lowering path. It is not a guarantee of overlapped memory and compute.

.. code-block:: python

   partial_sum = 0.0
   for start in metile.tile_range(0, count, BLOCK):
       positions = start + metile.arange(0, BLOCK)
       partial_sum = partial_sum + inputs.load((positions,))
   total = metile.sum(partial_sum)

.. function:: metile.scalar(value, dtype=None)

   Make explicit scalar SSA state for a loop-carried recurrence. This is
   useful when a state value is scalar per physical thread, as in
   :doc:`/examples/attention`.

Math and conversions
--------------------

Arithmetic such as ``+``, ``-``, ``*``, ``/``, comparisons, and
supported integer bit operations builds elementwise IR. Use ``&`` and
``|`` to combine traced boolean masks; Python ``and`` and ``or`` do
not describe elementwise operations.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Behavior
   * - ``metile.exp(value)``
     - Exponential
   * - ``metile.fast_exp(value)``
     - Metal's fast exponential; accuracy differs from the regular intrinsic
   * - ``metile.log(value)``
     - Natural logarithm
   * - ``metile.sqrt(value)``
     - Square root
   * - ``metile.abs(value)``
     - Absolute value
   * - ``metile.tanh(value)``
     - Hyperbolic tangent
   * - ``metile.where(condition, left, right)``
     - Select a value per element; it is not short-circuit control flow
   * - ``metile.maximum(left, right)``, ``metile.minimum(left, right)``
     - Elementwise maximum or minimum
   * - ``metile.cast(value, dtype)``
     - Convert a scalar or tile, for example to ``"f32"``

A ``where`` around an unmasked load does not make that load safe: its inputs
are computed before selection. Put the bounds in the load's mask or use a
tensor view.

Reductions
----------

.. function:: metile.sum(value)
.. function:: metile.max(value)
.. function:: metile.min(value)

   Reduce a supported tile to a scalar sum, maximum, or minimum. Lowering may
   combine register-local work, SIMDgroup operations, and shared memory
   between SIMDgroups.

Choose padding values appropriate to the reduction, and mask any transformed
padding that must not contribute. See :doc:`/examples/softmax` for maximum
and exponential-sum reductions over a partial tile. Floating-point reduction
order may differ from a CPU reference.

Thread and SIMDgroup operations
-------------------------------

These operations expose physical execution details. A SIMDgroup has 32
threads on the supported Apple GPU paths.

.. function:: metile.thread_id()

   Thread index inside the threadgroup.

.. function:: metile.simd_lane_id()

   Lane index inside its SIMDgroup, from 0 through 31.

.. function:: metile.simd_shuffle_xor(value, mask)

   Exchange a value with the lane whose index is XORed with ``mask``.

.. function:: metile.simd_broadcast(value, lane)

   Broadcast a value from one lane inside the SIMDgroup.

.. function:: metile.simd_sum(value)
.. function:: metile.simd_max(value)

   Native reduction across the current SIMDgroup, not the whole threadgroup.

.. function:: metile.simdgroup_role(role, num_roles=2, num_sgs=0)

   Context manager recording work for a subset of SIMDgroups. ``num_sgs=0``
   requests an even division among roles. A role region does not establish
   ordering or data dependencies with another role. See
   :doc:`/examples/fused-activations`.

.. function:: metile.shared(size, dtype="f32")

   Allocate uninitialized threadgroup memory. The size and dtype are fixed
   for the specialization.

.. function:: metile.barrier()

   Synchronize the threadgroup and order threadgroup memory accesses. All
   participating threads must reach the barrier. It does not synchronize
   separate threadgroups or provide a device-memory producer/consumer protocol.

Ownership and schedules
-----------------------

``metile.ThreadLayout`` describes physical ownership of tile elements.
``metile.convert_layout(value, layout)`` preserves logical values while the
compiler inserts required redistribution. See :doc:`thread-layouts` for the
supported operations and communication rules.

.. function:: metile.tile_swizzle(pid_m, pid_n, pattern="auto", block_size=4)

   Record a traversal for output tiles. Patterns include ``"auto"``,
   ``"linear"``, ``"diagonal"``, ``"morton"``, ``"hilbert"``, and
   grouped traversals. Unsupported panel geometries fall back to a valid
   traversal. See :doc:`tile-ops` for the panel-size constraints.

Pass ``SCHEDULE=metile.Schedule(...)`` at launch to request a backend,
SIMDgroup geometry, staging, or supported vector width. The compiler checks
these requirements. A prepared dispatch's ``explain()`` report records the
selected schedule and materialized decisions; see :doc:`execution-schedules`.

Host APIs and specialized integrations are listed in :doc:`/api/reference`.
They are separate from the traced kernel operations described here.
