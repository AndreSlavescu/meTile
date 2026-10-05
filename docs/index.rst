meTile
======

**GPU kernels in Python. Compiled for Apple silicon.**

Declare tensor memory at the top of a Python kernel, then write the computation.
meTile compiles it to Metal; you do not need to write an Objective-C host, Swift
wrapper or Metal shader. Use explicit layouts and schedules when you need to
control execution.

.. code-block:: python

   import metile

   @metile.kernel
   def add(X, Y, Out, N, BLOCK: metile.constexpr):
       left = metile.tensor(X, shape=(N,), access="read")
       right = metile.tensor(Y, shape=(N,), access="read")
       output = metile.tensor(Out, shape=(N,), access="write")
       offsets = metile.program_id(0) * BLOCK + metile.arange(0, BLOCK)
       output.store((offsets,), left.load((offsets,)) + right.load((offsets,)))

The tensor descriptors supply this example's bounds checks. Follow the
:doc:`first-kernel tutorial <getting-started/first-kernel>` to allocate buffers,
launch the kernel and check its output.

meTile traces Python into Tile IR, lowers it to Metal IR and emits Metal Shading
Language. Apple's compiler then produces device code. Matrix backend support
depends on the GPU and toolchain; not every Apple silicon Mac supports native
tensor operations. See :doc:`guide/architecture` for the compilation steps.

Start here
----------

- **Write a kernel:** install meTile and work through the first example.
- **Tune a kernel:** learn tensor declarations before layouts and schedules.
- **Use MLX:** read the integration guide and its correctness checks.
- **Evaluate performance:** check the benchmark methodology before comparing results.

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
   guide/training
   guide/kernel-coverage

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
   guide/benchmark-methodology
   guide/megakernels
   guide/prefill-decode-scheduling
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
