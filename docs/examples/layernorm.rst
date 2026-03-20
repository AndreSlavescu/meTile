Layer Normalization
===================

Row-wise layer normalization: ``Out = ((X - mean) / sqrt(var + eps)) * W + B``.


Kernel
------

.. code-block:: python

   import metile

   @metile.kernel
   def layernorm(X, W, B, Out, N, BLOCK: metile.constexpr):
       row = metile.program_id(0)

       # Pass 1: compute mean
       _sum = 0.0
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           _sum = _sum + x
       mean = metile.sum(_sum) / N

       # Pass 2: compute variance
       _var = 0.0
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           diff = x - mean
           _var = _var + diff * diff
       var = metile.sum(_var) / N

       # Pass 3: normalize, scale, shift
       inv_std = 1.0 / metile.sqrt(var + 1e-5)
       for i in metile.tile_range(0, N, BLOCK):
           cols = i + metile.arange(0, BLOCK)
           mask = cols < N
           x = metile.load(X + row * N + cols, mask=mask)
           w = metile.load(W + cols, mask=mask)
           b = metile.load(B + cols, mask=mask)
           out = (x - mean) * inv_std * w + b
           metile.store(Out + row * N + cols, out, mask=mask)


Launching
---------

.. code-block:: python

   import numpy as np

   rows, hidden = 128, 512
   X = metile.Buffer(data=np.random.randn(rows, hidden).astype(np.float32))
   W = metile.Buffer(data=np.ones(hidden, dtype=np.float32))
   B = metile.Buffer(data=np.zeros(hidden, dtype=np.float32))
   Out = metile.Buffer.zeros((rows * hidden,))

   layernorm[(rows,)](X, W, B, Out, hidden, BLOCK=256)


Concepts Introduced
-------------------

- Three-pass algorithm (mean, variance, normalize)
- Scalar accumulators across tiled loops
- ``metile.sum`` reduction
- ``metile.sqrt`` — element-wise square root
- Loading separate weight/bias arrays (shared across all rows)
