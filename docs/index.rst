meTile
======

**Tile-based GPU programming in Python for Apple Silicon.**

Write GPU kernels in Python, compile them to Metal. No Objective-C, no Swift, no CUDA.

.. code-block:: python

   import metile

   @metile.kernel
   def add(X, Y, Out, N, BLOCK: metile.constexpr):
       pid = metile.program_id(0)
       offs = pid * BLOCK + metile.arange(0, BLOCK)
       mask = offs < N
       x = metile.load(X + offs, mask=mask)
       y = metile.load(Y + offs, mask=mask)
       metile.store(Out + offs, x + y, mask=mask)

meTile traces your Python function, lowers it through multiple IR layers, and emits optimized
`Metal Shading Language <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_
targeting simdgroup matrix ops and Metal 4 tensor_ops.

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   getting-started/install
   getting-started/first-kernel

Programming Guide
-----------------

.. toctree::
   :maxdepth: 2

   guide/language
   guide/memory
   guide/tile-ops
   guide/autotuning

Kernel Examples
---------------

.. toctree::
   :maxdepth: 2

   examples/vector-add
   examples/softmax
   examples/matmul
   examples/layernorm
   examples/fused-activations

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/reference
