Memory Model
============

Apple Silicon lets the CPU and GPU access shared physical memory. In meTile,
``Buffer`` owns a Metal allocation in shared storage. This removes the need
for a separate GPU-to-CPU transfer when reading that allocation, but it does
not mean every buffer conversion is zero-copy.

.. image:: /_static/unified-memory.svg
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
into a new Metal allocation. Changing the original array afterward does not
change an explicit buffer. ``Buffer.zeros`` initializes storage to zero;
``Buffer.empty`` leaves its initial contents unspecified.

``buffer.numpy()`` waits for pending GPU work and returns a writable NumPy
view of the Metal allocation. It is not a copy. Keep the ``Buffer`` alive
for as long as that view is in use, and synchronize before accessing a retained
view after another dispatch. Calling ``numpy()`` again performs that wait;
reading a view returned earlier does not.

All examples use an explicit storage dtype. NumPy commonly creates float64
arrays by default, but the kernel launcher does not support arbitrary NumPy
dtypes. Its current dtype mapping covers float32, float16, int32, uint32, and
uint8; individual operations and backends support subsets of these. Convert
inputs to a dtype supported by the chosen kernel.

Passing NumPy arrays directly
------------------------------

A normal kernel launch can accept contiguous NumPy arrays. The launcher
copies their current contents into cached Metal buffers, dispatches, waits,
and copies the results back. That convenience includes transfer and
synchronization costs on each launch.

Use explicit buffers for repeated GPU work. In particular,
``kernel[grid].prepare(...)`` binds resources and returns a dispatcher that
reuses them; repeated calls do not rerun the NumPy conversion/copy-back path.
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
coordinates use the scalar ``other`` value, which defaults to zero; invalid
stores are skipped. Bounds are logical contracts, not a check against the
allocation size: the caller must provide enough storage for every valid
shape/stride address. Index and offset arithmetic must fit signed 32-bit
indexing. See :doc:`tensor-memory` for the complete contract.

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

A masked load's fill value must suit the reduction that follows. Zero works
for sums, negative infinity for maxima, and positive infinity for minima.
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
