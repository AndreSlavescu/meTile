Compiler Bypasses and Native Code
========================================

Research and local checks: October 2, 2026, Apple M5, macOS 26.2,
MLX 0.32.0, Apple Metal compiler 32023.864.

meTile has two experiments below its normal MSL compilation path: modifying
AIR before Apple's backend runs, and rewriting selected AGX instructions after
it runs. The probes demonstrate that modified code can execute. Neither has
established a new speedup over MLX, and neither is enabled in ordinary kernel
compilation.

What each route controls
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Route
     - What it controls
     - What remains
   * - Tile IR / Metal IR to MSL
     - Fusion, reuse, layouts, vector operations and matrix operations
     - Apple selects instructions, allocates registers and schedules them
   * - Textual AIR to ``metal-as`` to ``metallib``
     - LLVM-level operations, intrinsics, metadata and arithmetic flags
     - Apple's backend still selects and schedules AGX instructions
   * - Rewrite a binary archive's ``__text``
     - Actual encoded instructions in a verified region
     - Must preserve the surrounding executable's ABI and resource metadata
   * - A complete AGX backend
     - Instruction selection, register allocation and scheduling
     - Requires far more ISA and executable-format coverage than exists here

Apple documents the source-to-AIR-to-device-code split and binary archives as
pipeline compilation caches. ``failOnBinaryArchiveMiss`` prevents a missing
archive entry from silently falling back to compilation. It does not make
editing archive contents a supported API. See Apple's
`binary archive presentation <https://developer.apple.com/videos/play/wwdc2020/10615/>`_.

The existing source-order experiment was rerun on this machine:

.. list-table::
   :header-rows: 1

   * - Source change
     - Compiled result
   * - Serial versus interleaved independent FMA chains
     - Identical, 190 bytes
   * - Loads at use versus hoisted loads
     - Identical, 218 bytes
   * - Addition chain versus balanced tree
     - Identical, 282 bytes

Apple's compiler produced identical code for each of these source-order pairs.
Other source transformations may still change the result. The register and
throughput numbers in ``metile/target/agx.py`` describe specific M5 probes and
need new measurements for different kernels or hardware.

Direct AIR experiments
-----------------------

The recorded toolchain accepted this sequence:

.. code-block:: bash

   xcrun metal -S -emit-llvm -c kernel.metal -o kernel.ll
   xcrun metal-as kernel.ll -o kernel.air
   xcrun metallib kernel.air -o kernel.metallib

``xcrun metal -x ir -c kernel.ll -o kernel.air`` also worked locally.
To inspect binary AIR, ``metal -x ir -S -emit-llvm`` worked; a separate
``metal-dis`` executable was not installed.

``benchmarks/hardware/agx_air_probe.py`` makes this reproducible: compile a small MSL
seed, change one FMA multiplier in textual AIR, assemble both versions, and
dispatch both through meTile's Metal runtime. It checks each against an
independent numerical expectation. A changed AIR format causes an explicit
failure instead of an unchecked string replacement. This demonstrates executing
modified IR, not a standalone AIR backend or a faster kernel.
On the setup above, both ``fma(input, 2, 1)`` and the rewritten
``fma(input, 3, 1)`` matched all 257 quarter-step test inputs exactly.

For the general compiler, a candidate implementation would lower a restricted
subset of Metal IR directly into AIR: loads/stores, integer address arithmetic,
floating-point operations, then SIMD reductions. Preserve the versioned kernel
metadata and address spaces from a toolchain-generated seed initially. Do not
assume arbitrary system LLVM bitcode has the dialect and metadata Metal needs.
`Metal.jl's compiler <https://github.com/JuliaGPU/Metal.jl/blob/main/src/compiler/compilation.jl>`_
is a useful primary implementation to study for this boundary.

The experiments worth trying there are precise contraction/reassociation
permissions, proven alignment/range information, and vector unpack/conversion
forms that MSL obscures. Compare each candidate's final code and execution
against equivalent MSL before growing the backend. AIR instruction order still
does not specify the final schedule. Handwriting ``air.*`` declarations is
toolchain-dependent: a local MSL declaration using ``__asm("air.fma.f32")``
was rejected, so that syntax is not a working inline-assembly escape hatch here.

Native instruction rewriting
----------------------------

``metile/target/agx_isa.py`` already assembles a restricted compact F32 FMA
form and executes modified binary archives. ``metile/compiler/agx_schedule.py``
operates on caller-supplied instruction offsets. The execution tests verify
that synthesised arithmetic really reaches the GPU.

The scheduler retains original encodings, observes register read/write
dependencies including FMA addends, and treats unknown controls and disabled
instructions as barriers. Archive rewriting extracts code from the archive
it will patch, avoiding a second compilation that might produce different code.

Recognizing eight plausible bytes does not establish an instruction boundary
or operand mode. A confirmed boundary
also does not decode surrounding loads, waits, stores, branches or matrix
instructions. The low-nibble length heuristic already failed the repository's
other kernels. Consequently these passes are not wired into ordinary kernel
compilation or MLX dispatch.

The next native experiment should target one demonstrated missed peephole in
a real generated kernel. Keep instruction length, register allocation, memory
operations and control flow unchanged. Preserve every field whose meaning is
unknown, and reject a replacement unless both its operand semantics and its
dependencies are understood. A speedup must be measured after validating the
output; retiring an instruction does not establish a cycle saving.

For broader decoding, study the
`Apple G13 architecture reference <https://dougallj.github.io/applegpu/docs.html>`_
and the `Mesa Asahi driver <https://docs.mesa3d.org/drivers/asahi.html>`_.
They provide a starting point for understanding encoding and compiler structure,
not evidence that G13 instructions or Linux driver interfaces work unchanged on
this G17/macOS target. A full backend also needs spills, scheduling controls,
executable metadata and generation-specific validation.

Research directions
--------------------

The following are research priorities, not new benchmark results:

1. **Reuse weights across small batches in the general lowering.** Sweep
   2, 4, 8, 16 and 32 rows with identical BF16 or INT4 representations. Carry
   row tiling and accumulator limits through the compiler rather than adding
   a model-specific source template. Existing matched-representation benchmarks
   identify this as a productive regime; register growth is the tradeoff.
2. **Fuse projection epilogues and adjacent operations.** Use graph semantics
   to eliminate an intermediate write/read or launch. Validate the whole fused
   boundary and retain the native candidate. This can save work even when the
   individual arithmetic instructions are already good.
3. **Compare direct device access against staging for tensor operations.**
   Apple's `MPP programming guide
   <https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>`_
   recommends direct device access for GEMM and discusses when thread
   parallelism makes manual software pipelining unnecessary. Treat this as a
   candidate policy, measured by tile and dtype. meTile's existing NAX lowering
   is the appropriate place to experiment.
4. **Try AIR vector unpack and conversion candidates on INT4 compute-heavy
   shapes.** First inspect whether the candidate changes final code; then
   measure the full projection, including tails, scales and biases. Preserve
   the representation and numerical contract. A faster isolated unpack loop
   may be irrelevant to a bandwidth-bound projection.
5. **Use native rewriting only when the previous comparison exposes a specific
   backend limitation.** Start with proven arithmetic regions; general load
   scheduling or register renaming needs much more decoding and liveness data.

MLX's `NAX implementation
<https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/nax.h>`_
already uses ``mpp::tensor_ops::matmul2d`` and SIMD-group execution. Emitting
the same primitive is therefore not itself a differentiator. The opportunity
is the surrounding workload, representation-preserving reuse and fusion.

The results in :doc:`benchmarks` motivate small-batch reuse and fusion
experiments. They do not establish a hardware-wide performance ceiling.

Reproduce and evaluate a candidate
----------------------------------

.. code-block:: bash

   python -m benchmarks.hardware.agx_source_order
   python -m benchmarks.hardware.agx_air_probe
   python -m pytest tests/target/test_agx_isa.py tests/compiler/test_agx_schedule.py \
       tests/compiler/test_agx_schedule_safety.py tests/target/test_target.py -q

Run GPU experiments serially; the archive harness uses shared filenames in its
working directory. Its subprocess duration includes compilation, loading and
dispatch overhead and must not be presented as GPU kernel latency.

Before a new backend becomes a runtime candidate, compile both arms once,
check outputs on random values and edge cases, and measure interleaved GPU
dispatches with a byte-identical control. Separate compilation time from steady
state. Reassociation requires an explicit numerical policy: collapsing FMA
chains is not bit-exact and can change overflow or underflow as well as rounding.
The native identity-removal pass also lacks a general signed-zero and NaN-payload
guarantee; its current ordinary-value probes do not establish those edge cases.

Use the existing tuning tournament and native fallback for promotion. Persist
any edited archive only with its GPU architecture, OS build, compiler identity,
source/options, launch layout and rewrite version. Record correctness failures
and inconclusive timings as rejected candidates. The current probes establish
that these compilation routes work; an MLX performance claim requires that
additional comparison.
