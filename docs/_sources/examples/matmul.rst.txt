Matrix Multiply (GEMM)
======================

A matrix multiply accumulates products of input tiles into an output tile.
This example uses contiguous float32 matrices with uneven dimensions to
exercise partial loads and stores.

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def matmul(
       left_ptr, right_ptr, output_ptr, rows, columns, reduction,
       BLOCK_M: metile.constexpr, BLOCK_N: metile.constexpr,
       BLOCK_K: metile.constexpr, RELU: metile.constexpr,
   ):
       left = metile.tensor(
           left_ptr, shape=(rows, reduction),
           block_shape=(BLOCK_M, BLOCK_K), access="read",
       )
       right = metile.tensor(
           right_ptr, shape=(reduction, columns),
           block_shape=(BLOCK_K, BLOCK_N), access="read",
       )
       output = metile.tensor(
           output_ptr, shape=(rows, columns),
           block_shape=(BLOCK_M, BLOCK_N), access="write",
       )
       row = metile.program_id(0) * BLOCK_M
       column = metile.program_id(1) * BLOCK_N
       accumulator = metile.zeros((BLOCK_M, BLOCK_N), dtype="f32")
       for start in metile.tile_range(0, reduction, BLOCK_K):
           accumulator = metile.dot(
               left.load((row, start)), right.load((start, column)), accumulator
           )
       if RELU:
           accumulator = metile.maximum(accumulator, 0.0)
       output.store((row, column), accumulator)

   rows, columns, reduction = 65, 77, 53
   rng = np.random.default_rng(0)
   left_data = rng.standard_normal((rows, reduction)).astype(np.float32)
   right_data = rng.standard_normal((reduction, columns)).astype(np.float32)
   left_buffer = metile.Buffer(data=left_data)
   right_buffer = metile.Buffer(data=right_data)
   output_buffer = metile.Buffer.zeros((rows, columns), dtype=np.float32)

   grid = (metile.cdiv(rows, 32), metile.cdiv(columns, 32))
   matmul[grid](
       left_buffer, right_buffer, output_buffer, rows, columns, reduction,
       BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, RELU=False,
   )
   reference = left_data @ right_data
   np.testing.assert_allclose(output_buffer.numpy(), reference, rtol=1e-4, atol=1e-4)

How the tiles fit together
--------------------------

.. image:: /_static/gemm-tiling.svg
   :target: ../_static/gemm-tiling.svg
   :alt: Each program owns an output matrix tile and accumulates products along K
   :width: 100%

``block_shape`` sets the matrix tile that a view loads or stores. Here,
each program owns a 32-by-32 output tile. Every loop iteration loads a
32-by-16 left tile and a 16-by-32 right tile, then accumulates their product
with ``dot``. The logical tensor shapes supply bounds for all three axes:
out-of-bounds inputs contribute zero, and out-of-bounds outputs are skipped.

The matrix path requires matching input/output storage dtypes and supported
contiguous row-major layouts, with a separate output buffer. Float32 and
supported float16 configurations accumulate in float32; the store converts
to the output storage dtype. Use positive dimensions. Not every tile size,
dtype, and schedule combination is legal. See :doc:`/guide/tensor-memory` and
:doc:`/guide/tile-ops` before changing them.

The compiler chooses a matrix backend supported by the device and toolchain.
A newer chip name alone does not establish Metal 4 tensor-ops support. The
direct NAX path also has tile and alignment requirements.

Fuse an activation
------------------

The ``RELU`` argument is a compile-time boolean. Set ``RELU=True`` to include
the pointwise activation in the GEMM epilogue:

.. code-block:: python

   matmul[grid](
       left_buffer, right_buffer, output_buffer, rows, columns, reduction,
       BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, RELU=True,
   )
   np.testing.assert_allclose(
       output_buffer.numpy(), np.maximum(reference, 0.0), rtol=1e-4, atol=1e-4
   )

Fusion saves an intermediate device-memory write/read and a separate launch.
The activation still takes instructions and may affect register use. See
:doc:`/guide/execution-schedules` for the supported epilogue expressions.

Tune the tile sizes
-------------------

This small search compares two legal configurations. The callable grid is
recomputed for each candidate, so every candidate covers the full output.

.. code-block:: python

   autotuned_matmul = metile.autotune(
       configs=[
           metile.Config(BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, RELU=False),
           metile.Config(BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, RELU=False),
       ],
       key=["rows", "columns", "reduction"],
       verbose=False,
   )(matmul)

   def tuned_grid(config):
       return (
           metile.cdiv(rows, config["BLOCK_M"]),
           metile.cdiv(columns, config["BLOCK_N"]),
       )

   autotuned_matmul[tuned_grid](
       left_buffer, right_buffer, output_buffer, rows, columns, reduction
   )
   np.testing.assert_allclose(output_buffer.numpy(), reference, rtol=1e-4, atol=1e-4)

Tuning runs candidates and writes the output; measure that cost separately
from repeated execution of the winner. You can also tune tile traversal:
``tile_swizzle`` supports automatic selection and explicit patterns such as
Morton and Hilbert. See :doc:`/guide/autotuning` and :doc:`/guide/tile-ops`
for the search and grid constraints.
