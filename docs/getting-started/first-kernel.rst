Your First Kernel
=================

This tutorial walks through writing, launching, and understanding a simple GPU kernel with meTile.

The Kernel
----------

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def add(X, Y, Out, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N
       x = metile.load(X + offs, mask=mask)
       y = metile.load(Y + offs, mask=mask)
       metile.store(Out + offs, x + y, mask=mask)

Let's break this down line by line.

``@metile.kernel``
   Marks this function for GPU compilation. When you call it, meTile traces the Python
   code, compiles it to a Metal shader, and dispatches it on the GPU.

``X, Y, Out``
   Device pointers to GPU memory. These map to ``device float*`` in Metal.

``N``
   A runtime scalar, passed as a ``constant int&`` to the shader.

``BLOCK: metile.constexpr``
   A **compile-time constant**. The value is baked directly into the shader. Changing it
   triggers recompilation.

``metile.program_id(0)``
   Returns the index of this program instance along axis 0. If you launch 4 instances,
   they get ``pid = 0, 1, 2, 3``. This is analogous to ``blockIdx.x`` in CUDA or
   ``get_program_id(0)`` in Triton.

``metile.arange(0, BLOCK)``
   Creates a tile (vector) of consecutive indices ``[0, 1, 2, ..., BLOCK-1]``.

``offs = pid * BLOCK + metile.arange(0, BLOCK)``
   Each program instance handles a contiguous chunk of ``BLOCK`` elements.
   Instance 0 handles ``[0..BLOCK-1]``, instance 1 handles ``[BLOCK..2*BLOCK-1]``, etc.

``mask = offs < N``
   A boolean mask that prevents out-of-bounds accesses when ``N`` is not a multiple of ``BLOCK``.

``metile.load(X + offs, mask=mask)``
   Loads ``BLOCK`` elements from memory. Masked-off lanes read zero.

``metile.store(Out + offs, x + y, mask=mask)``
   Stores results. Masked-off lanes are skipped.


Launching
---------

.. code-block:: python

   N = 1024
   x = metile.Buffer(data=np.random.randn(N).astype(np.float32))
   y = metile.Buffer(data=np.random.randn(N).astype(np.float32))
   out = metile.Buffer.zeros((N,))

   grid = (metile.cdiv(N, 256),)   # ceil(1024 / 256) = 4 program instances
   add[grid](x, y, out, N, BLOCK=256)

   print(out.numpy()[:5])

``metile.Buffer``
   Wraps a Metal buffer in unified memory. CPU and GPU share the same physical memory on
   Apple Silicon, so there is no copy between host and device.

``metile.Buffer.zeros((N,))``
   Allocates a zeroed buffer of ``N`` float32 elements.

``metile.cdiv(N, 256)``
   Ceiling division: ``ceil(N / 256)``. Utility for computing grid sizes.

``add[grid](...)``
   The ``[grid]`` subscript sets the number of program instances (threadgroups). The kernel
   is compiled on first call and cached for subsequent calls with the same constexprs.


The Compilation Pipeline
------------------------

When you call ``add[grid](...)``, meTile:

1. **Traces** the Python function with symbolic values to build a Tile IR
2. **Lowers** the Tile IR to Metal IR (Apple GPU-specific primitives)
3. **Optimizes** via IR-to-IR passes (vectorization, loop splitting, constant folding)
4. **Emits** MSL (Metal Shading Language) source code
5. **Compiles** with ``xcrun metal -O2`` (or JIT if Xcode is unavailable)
6. **Dispatches** the compute pipeline on the GPU

.. image:: /_static/compilation-pipeline.svg
   :alt: meTile compilation pipeline: Python to Tile IR to Metal IR to MSL to GPU
   :width: 100%

You can inspect any stage with the ``METILE_DEBUG`` environment variable:

.. code-block:: bash

   METILE_DEBUG=msl python my_script.py       # see the generated Metal shader
   METILE_DEBUG=tile_ir python my_script.py   # see the Tile IR
   METILE_DEBUG=all python my_script.py       # see everything


What's Next
-----------

- :doc:`/guide/language` for the full language reference
- :doc:`/examples/softmax` for a more complex kernel with reductions and multiple passes
- :doc:`/examples/matmul` for tile-level matrix multiply with ``dot`` and ``tile_load``
