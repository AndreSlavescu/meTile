Layer Normalization
===================

Layer normalization computes a mean and variance for each row, then applies
per-column scale and bias:
``output = (input - mean) / sqrt(variance + epsilon) * weight + bias``.
The variance divides by the row width, as in ``numpy.var`` with its default
``ddof=0``.

.. code-block:: python

   import numpy as np
   import metile

   @metile.kernel
   def layernorm(
       input_ptr, weight_ptr, bias_ptr, output_ptr,
       rows, columns, epsilon, BLOCK: metile.constexpr,
   ):
       inputs = metile.tensor(input_ptr, shape=(rows, columns), access="read")
       weights = metile.tensor(weight_ptr, shape=(columns,), access="read")
       biases = metile.tensor(bias_ptr, shape=(columns,), access="read")
       outputs = metile.tensor(output_ptr, shape=(rows, columns), access="write")
       row = metile.program_id(0)

       local_sum = 0.0
       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           local_sum = local_sum + inputs.load((row, positions))
       mean = metile.sum(local_sum) / columns

       local_variance = 0.0
       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           values = inputs.load((row, positions))
           difference = metile.where(positions < columns, values - mean, 0.0)
           local_variance = local_variance + difference * difference
       variance = metile.sum(local_variance) / columns
       inverse_std = 1.0 / metile.sqrt(variance + epsilon)

       for start in metile.tile_range(0, columns, BLOCK):
           positions = start + metile.arange(0, BLOCK)
           values = inputs.load((row, positions))
           normalized = (values - mean) * inverse_std
           result = normalized * weights.load((positions,)) + biases.load((positions,))
           outputs.store((row, positions), result)

   rows, columns = 8, 300
   epsilon = 1e-5
   rng = np.random.default_rng(0)
   data = rng.standard_normal((rows, columns)).astype(np.float32) + 3.0
   weight_data = rng.standard_normal(columns).astype(np.float32)
   bias_data = rng.standard_normal(columns).astype(np.float32)
   input_buffer = metile.Buffer(data=data)
   weight_buffer = metile.Buffer(data=weight_data)
   bias_buffer = metile.Buffer(data=bias_data)
   output_buffer = metile.Buffer.zeros(data.shape, dtype=np.float32)

   layernorm[(rows,)](
       input_buffer, weight_buffer, bias_buffer, output_buffer,
       rows, columns, epsilon, BLOCK=256,
   )
   mean = data.mean(axis=1, keepdims=True)
   variance = data.var(axis=1, keepdims=True)
   reference = (data - mean) / np.sqrt(variance + epsilon) * weight_data + bias_data
   np.testing.assert_allclose(output_buffer.numpy(), reference, rtol=1e-4, atol=1e-5)

Each program handles one row. The three passes keep intermediate statistics
inside the kernel, although they reread the input from device memory. Computing
variance from centered values avoids subtracting two large, nearly equal
quantities as ``mean(x * x) - mean(x) ** 2`` would.

The explicit mask in the variance pass matters. A padded load returns zero;
subtracting the mean would turn that padding into a nonzero squared difference.
``where`` removes it before the reduction. The 300-column example checks this
partial-tile case with a nonzero input mean.

Use contiguous float32 inputs, positive dimensions, and a positive epsilon.
Weights and biases each contain ``columns`` elements and are shared by all
rows. Floating-point reductions may differ slightly from NumPy because their
addition order differs.
