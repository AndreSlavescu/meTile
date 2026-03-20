Softmax
=======

A fused row-wise softmax in a single kernel. Demonstrates multi-pass tiling with reductions.

.. code-block:: python

   import metile

   @metile.kernel
   def softmax(X, Out, N, BLOCK: metile.constexpr):
       row = metile.program_id(0)

       # Pass 1: find row maximum (for numerical stability)
       m = -1e38
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           m = metile.maximum(m, x)
       m = metile.max(m)

       # Pass 2: sum of exponentials
       s = 0.0
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           s = s + metile.exp(x - m)
       s = metile.sum(s)

       # Pass 3: normalize and write output
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           metile.store(Out + row * N + cols, metile.exp(x - m) / s, mask=mask)


Launching
---------

Each program instance handles one row. The grid is 1D with one instance per row:

.. code-block:: python

   import numpy as np

   rows, cols = 256, 1024
   X = metile.Buffer(data=np.random.randn(rows, cols).astype(np.float32))
   Out = metile.Buffer.zeros((rows * cols,))

   softmax[(rows,)](X, Out, cols, BLOCK=256)


Concepts Introduced
-------------------

- ``metile.tile_range``: tiling loop for iterating over a dimension
- ``metile.maximum`` / ``metile.max``: element-wise max and reduction
- ``metile.sum``: sum reduction
- ``metile.exp``: element-wise exponential
- Multi-pass algorithms: reading the same data multiple times in different passes
- Scalar accumulators (``m``, ``s``) carried across loop iterations
