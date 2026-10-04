Softmax
=======

This kernel computes softmax independently for each row of a contiguous
float32 matrix. It makes three passes: find the maximum, sum the shifted
exponentials, then normalize. Subtracting the maximum keeps the exponential
arguments nonpositive for finite inputs.

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def softmax(input_ptr, output_ptr, rows, columns, BLOCK: metile.constexpr):
       inputs = metile.tensor(input_ptr, shape=(rows, columns), access="read")
       outputs = metile.tensor(output_ptr, shape=(rows, columns), access="write")
       row = metile.program_id(0)

       local_maximum = -float("inf")
       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           values = inputs.load((row, positions), other=-float("inf"))
           local_maximum = metile.maximum(local_maximum, values)
       row_maximum = metile.max(local_maximum)

       local_sum = 0.0
       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           values = inputs.load((row, positions))
           exponentials = metile.where(
               positions < columns, metile.exp(values - row_maximum), 0.0
           )
           local_sum = local_sum + exponentials
       denominator = metile.sum(local_sum)

       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           values = inputs.load((row, positions))
           outputs.store((row, positions), metile.exp(values - row_maximum) / denominator)

   rows, columns = 8, 300
   rng = np.random.default_rng(0)
   data = rng.standard_normal((rows, columns)).astype(np.float32) - 5.0
   input_buffer = metile.Buffer(data=data)
   output_buffer = metile.Buffer.zeros(data.shape, dtype=np.float32)
   softmax[(rows,)](input_buffer, output_buffer, rows, columns, BLOCK=256)

   shifted = data - data.max(axis=1, keepdims=True)
   reference = np.exp(shifted)
   reference /= reference.sum(axis=1, keepdims=True)
   result = output_buffer.numpy()
   np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-6)
   np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)

Mask the reduction, too
-------------------------

The width is 300, so the second 256-element tile contains padding. Bounds
checks alone do not make every reduction correct: a zero-filled load would
contribute ``exp(0 - row_maximum)`` to the denominator. The explicit ``where``
sets those contributions to zero. The maximum pass instead fills padding with
negative infinity, the identity for a maximum reduction.

The loop-carried values contain one partial result per lane. ``metile.max``
and ``metile.sum`` combine those partials across the tile. This example assumes
positive row and column counts and finite input values; it does not define
special behavior for rows containing NaNs or infinities.

meTile can recognize some softmax forms and rewrite them into an online
algorithm. That optimization is pattern-dependent. Inspect the generated IR
when evaluating a particular kernel, and compare its output against a
reference regardless of whether the rewrite occurs.
