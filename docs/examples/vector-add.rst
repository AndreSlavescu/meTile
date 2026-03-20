Vector Addition
===============

The simplest meTile kernel: add two arrays element by element.

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


   N = 100_000
   x = metile.Buffer(data=np.random.randn(N).astype(np.float32))
   y = metile.Buffer(data=np.random.randn(N).astype(np.float32))
   out = metile.Buffer.zeros((N,))

   BLOCK = 256
   grid = (metile.cdiv(N, BLOCK),)
   add[grid](x, y, out, N, BLOCK=BLOCK)

   # Verify
   from metile.runtime.metal_device import MetalDevice
   MetalDevice.get().sync()
   np.testing.assert_allclose(out.numpy(), x.numpy() + y.numpy(), rtol=1e-5)
   print("passed!")


Concepts Introduced
-------------------

- ``@metile.kernel``: compile a Python function to Metal
- ``metile.program_id``: which program instance am I?
- ``metile.arange``: tile of consecutive indices
- ``metile.load`` / ``metile.store``: masked memory access
- ``metile.Buffer``: zero-copy GPU memory
- ``kernel[grid]()``: launch with a grid of instances
