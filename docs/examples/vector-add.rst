Vector Addition
===============

Each program adds one tile of two contiguous float32 arrays. Tensor views
provide the bounds checks, including the partial tile at the end.

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

   count = 100_003
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
   np.testing.assert_allclose(
       output_buffer.numpy(), left_data + right_data, rtol=1e-6, atol=1e-6
   )

``Buffer(data=...)`` copies each input into shared Metal storage. Reading
``output_buffer.numpy()`` waits for the GPU and returns a view of the output
allocation. An extra ``sync()`` is unnecessary here.

The grid counts program instances, not elements. ``cdiv(count, block)`` makes
room for every element; the declared tensor shape prevents the last program
from storing beyond ``count``. See :doc:`/getting-started/first-kernel` for a
step-by-step explanation and :doc:`/guide/memory` for raw pointer access.
