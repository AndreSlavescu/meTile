Your First Kernel
=================

This example adds two float32 arrays on the GPU and checks the result with
NumPy. Save both Python blocks in one file and run it after following
:doc:`install`.

Write the kernel
----------------

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def add(left_ptr, right_ptr, output_ptr, count, BLOCK: metile.constexpr):
       left = metile.tensor(left_ptr, shape=(count,), access="read")
       right = metile.tensor(right_ptr, shape=(count,), access="read")
       output = metile.tensor(output_ptr, shape=(count,), access="write")
       positions = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
       output.store((positions,), left.load((positions,)) + right.load((positions,)))

``@metile.kernel`` traces the function with symbolic values when a new
specialization is needed. The compiler turns that trace into a Metal shader.
The function body describes GPU work; it does not run on the input arrays as
ordinary Python.

The three ``metile.tensor`` declarations describe existing buffers. Their
shapes are logical bounds, and their access modes tell the compiler which
loads and stores are allowed. They do not allocate memory.

Each program instance is a threadgroup. ``program_id(0)`` identifies its
position in the launch grid, while ``arange(0, BLOCK)`` creates the indices
inside its tile. With ``BLOCK=256``, the first program handles elements 0--255,
the next handles 256--511, and so on.

``count`` is a runtime integer. ``BLOCK`` is a compile-time constant: changing
it creates a different specialization. Tensor views fill out-of-bounds loads
with zero and skip out-of-bounds stores, so the final tile may be partial.

Launch and check the result
-----------------------------

.. code-block:: python

   count = 1003
   rng = np.random.default_rng(0)
   left_data = rng.standard_normal(count).astype(np.float32)
   right_data = rng.standard_normal(count).astype(np.float32)
   left_buffer = metile.Buffer(data=left_data)
   right_buffer = metile.Buffer(data=right_data)
   output_buffer = metile.Buffer.zeros((count,), dtype=np.float32)

   block = 256
   add[(metile.cdiv(count, block),)](
       left_buffer, right_buffer, output_buffer, count, BLOCK=block
   )
   result = output_buffer.numpy()
   np.testing.assert_allclose(result, left_data + right_data, rtol=1e-6, atol=1e-6)
   print(result[:5])

The grid has four program instances because ``cdiv`` rounds the division up.
The uneven input length exercises the last tile's bounds checks.

``Buffer(data=...)`` allocates shared Metal storage and **copies** the NumPy
data into it. CPU and GPU then access that allocation. ``numpy()`` waits for
pending GPU work and returns a NumPy view of the buffer; it does not copy the
result back to the original array. Keep the buffer alive while using its view.
See :doc:`/guide/memory` for ownership and synchronization details.

The first launch includes tracing and compilation. Later launches reuse a
cached specialization when its input types, constants, and other compilation
settings match.

Inspect the compilation
-----------------------

The compiler builds Tile IR, selects an execution schedule, lowers to Metal IR,
runs compiler passes, and emits Metal Shading Language (MSL). The runtime then
compiles the shader and dispatches it. See :doc:`/guide/architecture` for the
full path.

.. image:: /_static/compilation-pipeline.svg
   :alt: Python kernel traced to Tile IR, lowered to Metal IR, and compiled to GPU code
   :width: 100%

Run the script in a fresh process with ``METILE_DEBUG`` to inspect a stage:

.. code-block:: bash

   METILE_DEBUG=msl python my_script.py
   METILE_DEBUG=tile_ir python my_script.py
   METILE_DEBUG=all python my_script.py

Debug output appears on standard error and is saved under ``debug_output/``.
``METILE_DEBUG_DIR`` changes that directory. Output is generated when a
specialization is compiled, so an in-process cache hit does not produce a new
dump.

Continue with :doc:`/examples/softmax` for reductions,
:doc:`/examples/matmul` for matrix tiles, or :doc:`/guide/language` for the
kernel language.
