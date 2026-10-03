meTile
======

**GPU kernels in Python. Compiled for Apple silicon.**

Write the computation in Python without writing Objective-C, Swift or a Metal
shader by hand. Declare tensor memory at the top of the kernel, then let meTile
lower the operations. Explicit layouts and schedules give you more control when
you need it.

.. code-block:: python

   import metile

   @metile.kernel
   def add(X, Y, Out, N, BLOCK: metile.constexpr):
       left = metile.tensor(X, shape=(N,), access="read")
       right = metile.tensor(Y, shape=(N,), access="read")
       output = metile.tensor(Out, shape=(N,), access="write")
       offsets = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
       output.store((offsets,), left.load((offsets,)) + right.load((offsets,)))

Tensor descriptors generate the bounds masks in this example. The
:doc:`first-kernel tutorial <getting-started/first-kernel>` includes allocation,
launch and a numerical check.

meTile traces Python into Tile IR, lowers it to Metal IR and emits Metal Shading
Language. Apple's compiler produces the device code. Available matrix backends
depend on the GPU and toolchain; native tensor operations are not available on
every Apple silicon Mac. See :doc:`guide/architecture` for the full path.

Start here
----------

- **Write a kernel:** install meTile and work through the first example.
- **Tune a kernel:** learn tensor declarations before layouts and schedules.
- **Use MLX:** read the integration guide and its correctness checks.
- **Evaluate performance:** start with the benchmark methodology, not a headline speedup.

.. toctree::
   :maxdepth: 2

   getting-started/install
   getting-started/first-kernel

Write and tune kernels
----------------------

.. toctree::
   :maxdepth: 2

   guide/language
   guide/tensor-memory
   guide/memory
   guide/tile-ops
   guide/execution-schedules
   guide/thread-layouts
   guide/autotuning

Kernel examples
---------------

.. toctree::
   :maxdepth: 2

   examples/vector-add
   examples/softmax
   examples/attention
   examples/matmul
   examples/layernorm
   examples/fused-activations

Benchmarks and integration
--------------------------

.. toctree::
   :maxdepth: 1

   guide/benchmarks
   guide/mlx-backend

Compiler internals
------------------

.. toctree::
   :maxdepth: 1

   guide/architecture
   guide/graph-fusion
   guide/compiler-bypasses
   guide/references

API reference
-------------

.. toctree::
   :maxdepth: 2

   api/reference
