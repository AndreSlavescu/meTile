API Reference
=============

Kernel Definition & Launch
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``@metile.kernel``
     - Decorate a Python function for GPU compilation
   * - ``kernel[grid](*args, **constexprs)``
     - Launch kernel with given grid shape and compile-time constants
   * - ``metile.constexpr``
     - Type annotation for compile-time constant parameters


Buffers
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.Buffer(data=np_array)``
     - Create a GPU buffer from a numpy array (unified memory, zero-copy)
   * - ``metile.Buffer.zeros((size,))``
     - Allocate a zeroed float32 buffer
   * - ``metile.Buffer.from_numpy(np_array)``
     - Create a GPU buffer from a numpy array
   * - ``buf.numpy()``
     - Return a numpy view of the buffer data


Program Identity
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.program_id(axis)``
     - Threadgroup index along ``axis`` (0, 1, or 2)
   * - ``metile.thread_id()``
     - Thread index within the threadgroup
   * - ``metile.simd_lane_id()``
     - Lane index within the simdgroup (0-31)


Index Generation
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.arange(start, size)``
     - Tile of ``size`` consecutive integers from ``start``
   * - ``metile.cdiv(a, b)``
     - Ceiling division: ``ceil(a / b)``
   * - ``metile.next_power_of_2(n)``
     - Smallest power of 2 >= ``n``


Element-wise Memory
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.load(ptr, mask=None)``
     - Load elements; masked-off lanes read 0
   * - ``metile.store(ptr, value, mask=None)``
     - Store elements; masked-off lanes are skipped


Tile Memory
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tile_load(ptr, row, col, stride, shape)``
     - Load a 2D tile from row-major memory
   * - ``metile.tile_store(ptr, row, col, stride, value, shape)``
     - Store a 2D tile to row-major memory
   * - ``metile.zeros(shape, dtype="f32")``
     - Zero-initialized tile (accumulator init)
   * - ``metile.dot(a, b, acc)``
     - Tile matrix multiply-accumulate: ``acc += a @ b``


Control Flow
------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tile_range(start, end, step)``
     - Tiling loop (K-dimension iteration, multi-pass algorithms)


Math Operations
---------------

All operate element-wise on scalars and tiles:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Description
   * - ``metile.exp(x)``
     - Exponential
   * - ``metile.log(x)``
     - Natural logarithm
   * - ``metile.sqrt(x)``
     - Square root
   * - ``metile.abs(x)``
     - Absolute value
   * - ``metile.tanh(x)``
     - Hyperbolic tangent
   * - ``metile.where(cond, x, y)``
     - Conditional select
   * - ``metile.maximum(a, b)``
     - Element-wise max
   * - ``metile.minimum(a, b)``
     - Element-wise min

Standard Python arithmetic (``+``, ``-``, ``*``, ``/``, ``<``, ``>``, etc.) works inside kernels.


Reductions
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Description
   * - ``metile.sum(x)``
     - Sum-reduce tile to scalar
   * - ``metile.max(x)``
     - Max-reduce tile to scalar
   * - ``metile.min(x)``
     - Min-reduce tile to scalar


Simdgroup Operations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.simdgroup_role(role, num_roles)``
     - Context manager: execute on a subset of simdgroups
   * - ``metile.simd_shuffle_xor(value, mask)``
     - XOR-based lane exchange within a simdgroup
   * - ``metile.simd_broadcast(value, lane)``
     - Broadcast from one lane to all lanes
   * - ``metile.barrier()``
     - Threadgroup memory barrier
   * - ``metile.shared(size, dtype)``
     - Allocate threadgroup (shared) memory


Tile Scheduling
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.tile_swizzle(pid_m, pid_n, pattern, block_size)``
     - Apply tile scheduling pattern (``"morton"``, ``"diagonal"``)


Autotuning
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - ``metile.autotune(configs, key, warmup=5, rep=20, verbose=True)``
     - Decorator for automatic parameter search
   * - ``metile.Config(**constexprs)``
     - A set of constexpr values to benchmark
   * - ``autotuned[grid].prepare(*args, **kwargs)``
     - Autotune once and return a fast dispatcher
