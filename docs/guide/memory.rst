Memory Model
============

On Apple Silicon, the CPU and GPU can access the same physical memory.
meTile's ``Buffer`` owns a shared Metal allocation, so reading it does not
require a separate GPU-to-CPU transfer. Creating a buffer from other storage
can still involve a copy.

.. image:: /_static/unified-memory.svg
   :target: ../_static/unified-memory.svg
   :alt: CPU and GPU access a shared Metal buffer; constructing it from NumPy copies the input
   :width: 100%

Allocation and ownership
------------------------

.. code-block:: python

   import numpy as np
   import metile

   source = np.arange(1024, dtype=np.float32)
   buffer = metile.Buffer(data=source)
   output = metile.Buffer.zeros(source.shape, dtype=np.float32)
   view = buffer.numpy()
   view[0] = 42.0
   assert source[0] == 0.0

``Buffer(data=source)`` and ``Buffer.from_numpy(source)`` copy the source
into a new Metal allocation. Modifying the original array afterward does not
affect an explicit buffer. ``Buffer.zeros`` initializes storage to zero;
``Buffer.empty`` leaves its initial contents unspecified.

``buffer.numpy()`` waits for pending GPU work, then returns a writable view
of the Metal allocation, not a copy. Keep the ``Buffer`` alive while using
that view. After another dispatch, synchronize before reading a retained
view: calling ``numpy()`` again waits, but reading an earlier view does not.

All examples use an explicit storage dtype. NumPy commonly creates float64
arrays by default, but the kernel launcher does not support arbitrary NumPy
dtypes. Its current dtype mapping covers float32, float16, int32, uint32, and
uint8; individual operations and backends support subsets of these. Convert
inputs to a dtype supported by the chosen kernel.

Passing NumPy arrays directly
------------------------------

A normal launch accepts contiguous NumPy arrays. It copies their current
contents into cached Metal buffers, dispatches the kernel, waits and copies
the results back. Each launch pays those copy and synchronization costs.

Use explicit buffers for repeated GPU work. In particular,
``kernel[grid].prepare(...)`` binds resources and returns a dispatcher that
reuses them. Repeated calls don't rerun NumPy conversion or copy-back.
Read results through an explicit output buffer's ``numpy()`` method.

Use contiguous arrays for implicit outputs. A noncontiguous array can be
converted to a temporary contiguous array, and that temporary does not provide
a reliable write-back path to the original view. To express strided access,
allocate an explicit buffer and declare element strides inside the kernel.

Logical bounds
--------------

Inside a kernel, ``metile.tensor`` describes a pointer's shape, strides,
and access mode. For example, this fragment loads one tile and doubles it:

.. code-block:: python

   inputs = metile.tensor(input_ptr, shape=(count,), access="read")
   outputs = metile.tensor(output_ptr, shape=(count,), access="write")
   positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
   outputs.store((positions,), inputs.load((positions,)) * 2.0)

A tensor load checks each coordinate against its declared dimension. Invalid
coordinates return the scalar ``other`` value, which defaults to zero;
invalid stores are skipped. These checks enforce the logical shape, not the
allocation size. The caller must provide storage for every valid shape/stride
address and keep index and offset arithmetic within signed 32-bit range.
See :doc:`tensor-memory` for the complete contract.

The lower-level ``load`` and ``store`` operations take pointer expressions
and optional masks. They do not know the allocation's bounds:

.. code-block:: python

   positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
   valid = positions < count
   values = metile.load(input_ptr + positions, mask=valid)
   metile.store(output_ptr + positions, values, mask=valid)

For ``count=10``, ``BLOCK=4``, and program 2, the positions are
``[8, 9, 10, 11]`` and the mask is ``[True, True, False, False]``.
The last two loads produce zero and their stores are skipped. A computation
that can produce negative positions also needs a lower-bound check.

For a masked load, choose a fill value that suits the reduction that follows:
zero for sums, negative infinity for maxima, and positive infinity for minima.
After operations such as subtraction or exponentiation, padding may need
another mask. The :doc:`/examples/softmax` and :doc:`/examples/layernorm`
examples show why.

Threadgroup memory
------------------

``metile.shared(size, dtype="f32")`` allocates threadgroup-local scratch
storage. It is uninitialized and visible only to threads in the same
threadgroup. Initialize every element that will be read.

Use ``metile.barrier()`` between a cooperative write and dependent reads
by other threads. It synchronizes the threadgroup and orders threadgroup
memory accesses. Every participating thread must reach the barrier; do not
hide one inside a region executed by only some SIMDgroups. A threadgroup
barrier does not synchronize separate threadgroups. For a global dependency,
use separate ordered kernel dispatches.
