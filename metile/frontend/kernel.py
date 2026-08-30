from __future__ import annotations

import ctypes
import inspect
import os
import struct
import sys

import numpy as np

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import (
    block_swizzle,
    decompose_nax_fragments,
    double_buffer_k_loop,
    fold_constants,
    pad_shared_memory,
    preload_mma_tiles,
    serpentine_mma,
    split_elementwise_loops,
    split_k_loop,
    swizzle_shared_memory,
    validate_pass_order,
    vectorize_elementwise,
    vectorize_loads,
)
from metile.compiler.schedule_search import compressed_description_bits, optimize_tile_schedules
from metile.frontend.tracing import TracingContext, TracingProxy, constexpr
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile.runtime.buffer import MtileBuffer
from metile.runtime.metal_device import MetalDevice, MTLSize, NSRange

# Global kernel cache: (func_name, constexprs_tuple, dtypes_tuple) -> CompiledKernel
_kernel_cache: dict = {}
# Scalar buffer cache: (value, format_char) -> metal_buffer
_scalar_buffer_cache: dict = {}

_ELEM_SIZES = {"float": 4, "half": 2, "int": 4, "uint": 4, "uchar": 1}


class OutOfResources(RuntimeError):
    """A configuration asks for more threadgroup memory than the device has.

    Typed rather than a bare RuntimeError so tuners can prune the config and keep going, which is
    the distinction Triton draws with its own OutOfResources: exceeding a hardware limit is a fact
    about one candidate, while any other compile failure is a bug that should surface. Catching
    RuntimeError broadly in a tuning loop would swallow both.
    """


def _validate_threadgroup_memory(metal_ir: mir.MFunction):
    """Raise OutOfResources if threadgroup memory exceeds the hardware limit."""
    total_bytes = 0
    for op in metal_ir.ops:
        if isinstance(op, mir.MThreadgroupAlloc):
            total_bytes += op.size * _ELEM_SIZES.get(op.elem_type, 4)
    if total_bytes == 0:
        return
    limit = MetalDevice.get().max_threadgroup_memory
    if total_bytes > limit:
        raise OutOfResources(
            f"Kernel '{metal_ir.name}' requires {total_bytes} bytes threadgroup memory "
            f"but device limit is {limit} bytes. Reduce tile sizes."
        )


def _dump(path: str, content: str):
    """Write debug output to a file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


class CompiledKernel:
    def __init__(
        self,
        pipeline,
        msl_source: str,
        func_name: str,
        threadgroup_size: tuple[int, int, int],
        is_gemm: bool = False,
        prefer_ordered: bool = False,
        output_indices: tuple[int, ...] = (),
        argument_indices: tuple[int, ...] | None = None,
    ):
        self.pipeline = pipeline
        self.msl_source = msl_source
        self.func_name = func_name
        self.threadgroup_size = threadgroup_size
        self.is_gemm = is_gemm
        self.prefer_ordered = prefer_ordered
        self.output_indices = output_indices
        self.argument_indices = argument_indices
        self.description_bits = compressed_description_bits(msl_source)


def kernel(fn):
    """Decorator that transforms a Python function into a launchable GPU kernel."""
    return KernelFunction(fn)


class KernelFunction:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._sig = inspect.signature(fn)

    def __getitem__(self, grid):
        """kernel[(grid_x,)] or kernel[(grid_x, grid_y)]"""
        if isinstance(grid, int):
            grid = (grid,)
        return KernelLauncher(self, grid)

    def get_compiled(self, *args, **kwargs) -> CompiledKernel:
        """Compile the kernel for given args/constexprs and return the CompiledKernel.

        Triggers a dummy dispatch to compile + cache, then returns the cached result.
        Useful for batched dispatch where you need the pipeline directly.
        """
        launcher = KernelLauncher(self, (1,))
        launcher(*args, **kwargs)

        # Build cache key
        sig = self._sig
        sig_names = set(sig.parameters.keys())
        constexprs = {}
        dtypes = []
        for name, param in sig.parameters.items():
            if param.annotation is constexpr and name in kwargs:
                val = kwargs[name]
                constexprs[name] = val._value if isinstance(val, constexpr) else val
        for name, val in kwargs.items():
            if name not in sig_names and name not in constexprs:
                constexprs[name] = val._value if isinstance(val, constexpr) else val

        for a in args:
            if isinstance(a, (MtileBuffer, np.ndarray)):
                dtypes.append(_numpy_to_dtype(a.dtype))
            elif isinstance(a, int):
                dtypes.append("i32")
            elif isinstance(a, float):
                dtypes.append("f32")

        cache_key = (
            self.name,
            tuple(sorted(constexprs.items())),
            tuple(dtypes),
        )
        return _kernel_cache[cache_key]


class FastDispatcher:
    """Low-overhead prepared dispatch. Created by KernelLauncher.prepare().

    Pre-resolves all Metal buffers, MTLSize structs, and ctypes function
    pointers. Consecutive prepared calls automatically share a command buffer
    and compute encoder until sync(), reducing composed-kernel launch overhead.
    """

    __slots__ = (
        "_binding_key",
        "_buffer_array",
        "_buffer_offsets",
        "_buffer_range",
        "_buffers",
        "_completion_spin_ns",
        "_concurrent",
        "_description_bits",
        "_dev",
        "_dispatch_fn",
        "_dispatch_sel",
        "_grid",
        "_input_resources",
        "_output_resources",
        "_pipeline",
        "_resources",
        "_set_buf_fn",
        "_set_buf_sel",
        "_set_bufs_fn",
        "_set_bufs_sel",
        "_set_pipe_fn",
        "_set_pipe_sel",
        "_tg",
    )

    def __init__(
        self,
        compiled,
        metal_buffers,
        grid,
        dev,
        resources=(),
        completion_spin_ns=0,
    ):
        dev._ensure_cached_selectors()
        self._pipeline = compiled.pipeline
        self._buffers = tuple(metal_buffers)
        self._resources = tuple(resources)
        self._concurrent = not compiled.is_gemm and not compiled.prefer_ordered
        self._dev = dev
        self._description_bits = compiled.description_bits
        self._completion_spin_ns = max(0, int(completion_spin_ns))
        buffer_values = tuple(
            buffer.value if isinstance(buffer, ctypes.c_void_p) else int(buffer)
            for buffer in self._buffers
        )
        self._input_resources = frozenset(buffer_values)
        self._output_resources = frozenset(
            buffer_values[index] for index in compiled.output_indices
        )
        pipeline_value = (
            self._pipeline.value
            if isinstance(self._pipeline, ctypes.c_void_p)
            else int(self._pipeline)
        )
        self._binding_key = (pipeline_value, buffer_values)

        # Pre-cache all ctypes functions as instance attrs
        self._set_pipe_fn = MetalDevice._set_pipeline_fn
        self._set_pipe_sel = MetalDevice._set_pipeline_sel
        self._set_buf_fn = MetalDevice._set_buffer_fn
        self._set_buf_sel = MetalDevice._set_buffer_sel
        self._set_bufs_fn = MetalDevice._set_buffers_fn
        self._set_bufs_sel = MetalDevice._set_buffers_sel
        if len(buffer_values) > 1:
            self._buffer_array = (ctypes.c_void_p * len(buffer_values))(*buffer_values)
            self._buffer_offsets = (ctypes.c_uint64 * len(buffer_values))()
            self._buffer_range = NSRange(0, len(buffer_values))
        else:
            self._buffer_array = None
            self._buffer_offsets = None
            self._buffer_range = None
        tg = compiled.threadgroup_size
        self._tg = MTLSize(tg[0], tg[1], tg[2])
        if compiled.is_gemm:
            self._grid = MTLSize(grid[0], grid[1] if len(grid) > 1 else 1, 1)
            self._dispatch_fn = MetalDevice._dispatch_tg_fn
            self._dispatch_sel = MetalDevice._dispatch_tg_sel
        else:
            self._grid = MTLSize(
                grid[0] * tg[0],
                (grid[1] * tg[1]) if len(grid) > 1 else 1,
                (grid[2] * tg[2]) if len(grid) > 2 else 1,
            )
            self._dispatch_fn = MetalDevice._dispatch_threads_fn
            self._dispatch_sel = MetalDevice._dispatch_threads_sel

    def __call__(self):
        dev = self._dev
        with dev._dispatch_lock:
            self._encode_unlocked(dev)

    def repeat(self, count: int):
        """Encode the same prepared dispatch repeatedly under one runtime lock."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        dev = self._dev
        with dev._dispatch_lock:
            for _ in range(count):
                self._encode_unlocked(dev)

    def _encode_unlocked(self, dev):
        encoder = dev._pending_encoder_unlocked(self._concurrent)
        concurrent = bool(dev._pending_concurrent)
        if concurrent and (
            (dev._pending_outputs & self._input_resources)
            or (dev._pending_inputs & self._output_resources)
        ):
            dev._memory_barrier_fn(encoder, dev._memory_barrier_sel, 1)
            dev._pending_inputs.clear()
            dev._pending_outputs.clear()
        if dev._pending_pipeline != self._binding_key[0]:
            self._set_pipe_fn(encoder, self._set_pipe_sel, self._pipeline)
            dev._pending_pipeline = self._binding_key[0]

        if dev._pending_binding_key != self._binding_key:
            if self._buffer_array is not None:
                self._set_bufs_fn(
                    encoder,
                    self._set_bufs_sel,
                    self._buffer_array,
                    self._buffer_offsets,
                    self._buffer_range,
                )
            elif self._buffers:
                self._set_buf_fn(encoder, self._set_buf_sel, self._buffers[0], 0, 0)
            dev._pending_binding_key = self._binding_key

        self._dispatch_fn(encoder, self._dispatch_sel, self._grid, self._tg)
        if self._concurrent:
            dev._pending_inputs.update(self._input_resources)
            dev._pending_outputs.update(self._output_resources)
        dev._pending_lifetimes[id(self)] = self
        if dev._pending_dispatches == 0:
            dev._pending_completion_spin_ns = self._completion_spin_ns
        elif dev._pending_completion_spin_ns and self._completion_spin_ns:
            dev._pending_completion_spin_ns += self._completion_spin_ns
        else:
            dev._pending_completion_spin_ns = 0
        dev._pending_dispatches += 1
        if dev._pending_dispatches >= 64:
            dev._commit_pending_unlocked()

    @property
    def description_bits(self) -> int:
        return self._description_bits


class KernelLauncher:
    def __init__(self, kernel_fn: KernelFunction, grid: tuple):
        self.kernel_fn = kernel_fn
        self.grid = grid

    def __call__(self, *args, **kwargs):
        """Full pipeline: trace -> lower -> codegen -> compile -> dispatch."""
        # Separate constexpr kwargs from regular args
        sig = self.kernel_fn._sig
        param_names = list(sig.parameters.keys())

        constexprs = {}
        regular_args = list(args)
        sig_names = set(sig.parameters.keys())

        # Check annotations for constexpr
        for name, param in sig.parameters.items():
            if param.annotation is constexpr and name in kwargs:
                val = kwargs[name]
                # Unwrap constexpr instances to plain values
                constexprs[name] = val._value if isinstance(val, constexpr) else val
            elif name in kwargs:
                regular_args.append(kwargs[name])

        # Collect config kwargs not in function signature as compiler parameters.
        # This lets Config(WM=4, COOPERATIVE=True, ...) reach the lowering
        # without requiring these to be kernel function parameters.
        for name, val in kwargs.items():
            if name not in sig_names and name not in constexprs:
                constexprs[name] = val._value if isinstance(val, constexpr) else val

        bound_kwargs = {name: value for name, value in kwargs.items() if name in sig_names}
        bound = sig.bind_partial(*args, **bound_kwargs)
        for axis in ("M", "N", "K"):
            block = constexprs.get(f"BLOCK_{axis}")
            value = bound.arguments.get(axis)
            if block is not None and isinstance(value, (int, np.integer)):
                constexprs[f"_ALIGNED_{axis}"] = int(value) % int(block) == 0
        nax_outer_k = constexprs.get("NAX_OUTER_K")
        k_value = bound.arguments.get("K")
        if nax_outer_k is not None and isinstance(k_value, (int, np.integer)):
            constexprs["_ALIGNED_NAX_OUTER_K"] = int(k_value) % int(nax_outer_k) == 0
        if constexprs.get("NAX_FRAGMENTS", False):
            for axis in ("M", "N", "K"):
                value = bound.arguments.get(axis)
                if isinstance(value, (int, np.integer)):
                    constexprs[f"_STATIC_{axis}"] = int(value)

        if isinstance(self.grid, tuple) and len(self.grid) >= 2:
            constexprs["_GRID_M"] = int(self.grid[0])
            constexprs["_GRID_N"] = int(self.grid[1])

        # Auto-convert numpy arrays to MtileBuffer (implicit composition)
        converted_args = []
        for a in regular_args:
            if isinstance(a, np.ndarray):
                converted_args.append(MtileBuffer._from_numpy_implicit(a))
            else:
                converted_args.append(a)

        # Determine dtypes from arrays
        dtypes = []
        for a in converted_args:
            if isinstance(a, MtileBuffer):
                dtypes.append(_numpy_to_dtype(a.dtype))
            elif isinstance(a, (int, float)):
                dtypes.append("i32" if isinstance(a, int) else "f32")

        # Cache key
        cache_key = (
            self.kernel_fn.name,
            tuple(sorted(constexprs.items())),
            tuple(dtypes),
        )

        if cache_key not in _kernel_cache:
            compiled = self._compile(converted_args, constexprs, param_names)
            _kernel_cache[cache_key] = compiled
        else:
            compiled = _kernel_cache[cache_key]

        # Dispatch
        metal_buffers = self._dispatch(compiled, converted_args)

        # Stash for prepare()
        self._last_compiled = compiled
        self._last_metal_buffers = metal_buffers
        self._last_resources = tuple(converted_args)

        # Sync results back to source numpy arrays (requires GPU completion)
        needs_sync = any(
            isinstance(a, MtileBuffer) and a._source_array is not None for a in converted_args
        )
        if needs_sync:
            MetalDevice.get().sync()
            for a in converted_args:
                if isinstance(a, MtileBuffer) and a._source_array is not None:
                    a.sync_to_source()

    def prepare(self, *args, **kwargs):
        """Compile and return a zero-overhead callable for repeated dispatch.

        Performs one full dispatch to compile and cache, then returns a
        FastDispatcher that skips all Python arg processing.
        """
        self(*args, **kwargs)
        MetalDevice.get().sync()
        parameter_names = tuple(self.kernel_fn._sig.parameters)
        arguments = dict(zip(parameter_names, args))
        arguments.update((name, value) for name, value in kwargs.items() if name in parameter_names)
        dimensions = tuple(arguments.get(axis) for axis in ("M", "N", "K"))
        prefer_low_latency = self._last_compiled.is_gemm and all(
            isinstance(dimension, (int, np.integer)) for dimension in dimensions
        )
        if prefer_low_latency:
            prefer_low_latency = np.prod(dimensions, dtype=np.int64) <= 512**3
        completion_spin_ns = 900_000 if prefer_low_latency else 0
        return FastDispatcher(
            self._last_compiled,
            self._last_metal_buffers,
            self.grid,
            MetalDevice.get(),
            self._last_resources,
            completion_spin_ns=completion_spin_ns,
        )

    def _compile(self, args, constexprs: dict, param_names: list[str]) -> CompiledKernel:
        """Trace, lower, codegen, and compile to Metal pipeline."""
        sig = self.kernel_fn._sig
        fn = self.kernel_fn.fn

        # Step 1: Trace
        ctx = TracingContext(self.kernel_fn.name)

        with ctx:
            # Create proxy objects for each parameter
            proxies = []
            param_idx = 0
            for name, param in sig.parameters.items():
                if param.annotation is constexpr:
                    # Pass as plain int (not a proxy)
                    continue

                if param_idx < len(args):
                    arg = args[param_idx]
                    if isinstance(arg, (MtileBuffer, np.ndarray)):
                        dtype = _numpy_to_dtype(arg.dtype)
                        p_type = PtrType(dtype)
                        ir_param = tir.Param(name, p_type, is_output=False)
                        ctx.func.params.append(ir_param)
                        val = tir.Value(name, p_type)
                        proxies.append(TracingProxy(val))
                    elif isinstance(arg, int):
                        ir_param = tir.Param(name, I32)
                        ctx.func.params.append(ir_param)
                        val = tir.Value(name, I32)
                        proxies.append(TracingProxy(val))
                    elif isinstance(arg, float):
                        ir_param = tir.Param(name, ScalarType("f32"))
                        ctx.func.params.append(ir_param)
                        val = tir.Value(name, ScalarType("f32"))
                        proxies.append(TracingProxy(val))
                    param_idx += 1

            ctx.func.constexprs = constexprs

            # Call the function with proxies + constexprs in the signature.
            # Compiler params (WM, WN, COOPERATIVE, etc.) are in constexprs
            # but not in the function signature — don't pass them as kwargs.
            sig_names = set(sig.parameters.keys())
            call_args = list(proxies)
            call_kwargs = {k: v for k, v in constexprs.items() if k in sig_names}
            fn(*call_args, **call_kwargs)

        # Mark output pointers: any array arg that appears in a store is output
        _mark_outputs(ctx.func)

        tile_ir = ctx.func

        # Debug output: METILE_DEBUG env var
        # "tile_ir" — print tile IR after tracing
        # "metal_ir" — print metal IR after lowering
        # "metal_ir_opt" — print metal IR after optimization passes
        # "msl" — print generated MSL source
        # "all" — print everything
        _debug = os.environ.get("METILE_DEBUG", "")
        _debug_flags = set(_debug.split(",")) if _debug else set()
        _debug_all = "all" in _debug_flags
        _debug_dir = os.environ.get("METILE_DEBUG_DIR", "debug_output")

        if _debug_all or "tile_ir" in _debug_flags:
            from metile.ir.printer import print_tile_ir

            ir_text = print_tile_ir(tile_ir)
            print(f"\n=== Tile IR: {tile_ir.name} ===", file=sys.stderr)
            print(ir_text, file=sys.stderr)
            if _debug_dir:
                _dump(os.path.join(_debug_dir, "tile_ir", f"{tile_ir.name}.txt"), ir_text)

        # Step 1.5: Algorithmic discovery (Tile IR -> Tile IR)
        # Rewrites a 3-pass softmax into a 2-pass online softmax, which moves three
        # arrays instead of four and measures 1.28x at DRAM-bound sizes against a
        # 1.33x transfer-ratio ceiling. Only applied when the reduction law it is
        # proved against discharges its obligations. Set METILE_ONLINE_SOFTMAX=0 to
        # skip discovery.
        if os.environ.get("METILE_ONLINE_SOFTMAX") != "0":
            from metile.compiler.algo_discovery import discover

            tile_ir = discover(tile_ir)

        # Step 2: Lower to Metal IR (handles both element-wise and GEMM)
        metal_ir = lower(tile_ir)

        if _debug_all or "metal_ir" in _debug_flags:
            from metile.ir.printer import print_metal_ir

            ir_text = print_metal_ir(metal_ir)
            print(f"\n=== Metal IR (pre-opt): {metal_ir.name} ===", file=sys.stderr)
            print(ir_text, file=sys.stderr)
            if _debug_dir:
                _dump(os.path.join(_debug_dir, "metal_ir", f"{metal_ir.name}.pre_opt.txt"), ir_text)

        # Step 3: Apply optimization passes
        #
        # Every pass goes through _run_pass so the ordering invariants in
        # metile.compiler.passes are checked against the passes that actually ran, not against a
        # hand-maintained list that can drift from this code. The name is read off the function
        # object, so a call recorded here cannot disagree with the pass it invoked.
        applied: list[str] = []

        def _run_pass(fn, ir, *args, **kwargs):
            applied.append(fn.__name__)
            return fn(ir, *args, **kwargs)

        is_gemm = metal_ir.kernel_type in ("gemm", "persistent_gemm")
        is_tensor_ops = metal_ir.kernel_type == "tensor_ops_gemm"
        is_specialized = metal_ir.kernel_type == "specialized_gemm"
        use_swizzle = constexprs.get("SWIZZLE_SMEM", False)
        if is_tensor_ops:
            # Tensor_ops kernels use register-resident cooperative_tensors —
            # no threadgroup memory passes needed. K-loop unrolling and
            # barrier removal are handled at lowering time.
            metal_ir = _run_pass(optimize_tile_schedules, metal_ir)
            metal_ir = _run_pass(decompose_nax_fragments, metal_ir)
        elif is_specialized:
            # Specialized GEMM: double-buffered + padded in lowering
            # Only apply vectorize and serpentine
            metal_ir = _run_pass(vectorize_loads, metal_ir, vec_size=4)
            metal_ir = _run_pass(serpentine_mma, metal_ir)
        elif is_gemm:
            if use_swizzle:
                metal_ir = _run_pass(swizzle_shared_memory, metal_ir)
            else:
                metal_ir = _run_pass(pad_shared_memory, metal_ir)
            metal_ir, did_db = _run_pass(double_buffer_k_loop, metal_ir)
            if not did_db:
                metal_ir = _run_pass(split_k_loop, metal_ir)
            metal_ir = _run_pass(vectorize_loads, metal_ir, vec_size=4)
            metal_ir = _run_pass(serpentine_mma, metal_ir)
            metal_ir = _run_pass(preload_mma_tiles, metal_ir)
            metal_ir = _run_pass(block_swizzle, metal_ir)
        else:
            metal_ir = _run_pass(split_elementwise_loops, metal_ir)
            metal_ir = _run_pass(vectorize_elementwise, metal_ir, vec_size=4)

        # Constant folding (all kernel types)
        metal_ir = _run_pass(fold_constants, metal_ir)

        # Instruction scheduling, off by default and last in the pipeline when on, so that it
        # sees the operations that actually get emitted.
        #
        # Off by default because it was measured to do nothing. Reordering MSL statements does
        # not move the register count in any of six kernels spanning 14 to 126 registers, and
        # no timing difference survives the benchmark's own control rows, which compare
        # byte-identical MSL against itself and spread by 0.6% to 7.6% between runs. The reason
        # is structural rather than a shortcoming of the pass: Apple's backend does its own
        # scheduling and allocation from the MSL it receives, so statement order is a
        # suggestion. See benchmarks/agx_schedule_effect.py.
        #
        # It stays because it is correct, tested, and the thing that would become load bearing
        # if meTile emitted below MSL, which the binary-archive work established is possible.
        # Set METILE_SCHEDULE=1 to compile with it.
        if os.environ.get("METILE_SCHEDULE") == "1":
            from metile.compiler.scheduling import reorder_for_latency

            metal_ir = _run_pass(reorder_for_latency, metal_ir)

        # Checked once the whole pipeline has run, so `applied` is the complete sequence. A
        # violation means the passes above would emit wrong code, so fail the compile rather
        # than hand back a bad kernel.
        validate_pass_order(applied)

        if _debug_all or "metal_ir_opt" in _debug_flags:
            from metile.ir.printer import print_metal_ir

            ir_text = print_metal_ir(metal_ir)
            print(f"\n=== Metal IR (post-opt): {metal_ir.name} ===", file=sys.stderr)
            print(ir_text, file=sys.stderr)
            if _debug_dir:
                _dump(
                    os.path.join(_debug_dir, "metal_ir", f"{metal_ir.name}.post_opt.txt"), ir_text
                )

        # Validate threadgroup memory fits within hardware limit
        _validate_threadgroup_memory(metal_ir)

        # Step 4: Generate MSL
        msl_source = emit(metal_ir)

        if _debug_all or "msl" in _debug_flags:
            print(f"\n=== MSL: {metal_ir.name} ===", file=sys.stderr)
            print(msl_source, file=sys.stderr)
            if _debug_dir:
                _dump(os.path.join(_debug_dir, "msl", f"{metal_ir.name}.metal"), msl_source)

        # Step 5: Compile
        dev = MetalDevice.get()
        if is_tensor_ops:
            # tensor_ops requires Metal 4 offline compilation
            pipeline, _ = dev.compile_msl_precompiled(
                msl_source, metal_ir.name, metal_std="metal4.0"
            )
        elif dev.has_metal_compiler:
            pipeline, _ = dev.compile_msl_precompiled(msl_source, metal_ir.name)
        else:
            pipeline = dev.compile_msl(msl_source, metal_ir.name)

        source_param_names = [
            name
            for name, parameter in sig.parameters.items()
            if parameter.annotation is not constexpr
        ]
        argument_indices = tuple(source_param_names.index(param.name) for param in metal_ir.params)
        return CompiledKernel(
            pipeline=pipeline,
            msl_source=msl_source,
            func_name=metal_ir.name,
            threadgroup_size=metal_ir.threadgroup_size,
            is_gemm=is_gemm or is_tensor_ops or is_specialized,
            prefer_ordered=any(
                isinstance(op, (mir.MBarrier, mir.MThreadgroupAlloc)) for op in metal_ir.ops
            ),
            output_indices=tuple(
                index for index, param in enumerate(metal_ir.params) if param.is_output
            ),
            argument_indices=argument_indices,
        )

    def _dispatch(self, compiled: CompiledKernel, args):
        """Bind buffers and dispatch kernel. Returns metal buffer list for prepare()."""
        dev = MetalDevice.get()

        buffers = []
        argument_indices = compiled.argument_indices
        selected_args = (
            args if argument_indices is None else (args[index] for index in argument_indices)
        )
        for arg in selected_args:
            if isinstance(arg, MtileBuffer):
                buffers.append(arg.metal_buffer)
            elif isinstance(arg, int):
                key = ("i", arg)
                if key not in _scalar_buffer_cache:
                    _scalar_buffer_cache[key] = dev.new_buffer(struct.pack("<i", arg), 4)
                buffers.append(_scalar_buffer_cache[key])
            elif isinstance(arg, float):
                key = ("f", arg)
                if key not in _scalar_buffer_cache:
                    _scalar_buffer_cache[key] = dev.new_buffer(struct.pack("<f", arg), 4)
                buffers.append(_scalar_buffer_cache[key])

        tg = compiled.threadgroup_size

        if compiled.is_gemm:
            grid_tg = (
                self.grid[0],
                self.grid[1] if len(self.grid) > 1 else 1,
                1,
            )
            dev.dispatch_threadgroups(compiled.pipeline, buffers, grid_tg, tg)
        else:
            if len(self.grid) == 1:
                total_threads = self.grid[0] * tg[0]
                grid_size = (total_threads, 1, 1)
            elif len(self.grid) == 2:
                grid_size = (self.grid[0] * tg[0], self.grid[1] * tg[1], 1)
            else:
                grid_size = (self.grid[0] * tg[0], self.grid[1] * tg[1], self.grid[2] * tg[2])
            dev.dispatch_kernel(compiled.pipeline, buffers, grid_size, tg)

        return buffers


def _mark_outputs(func: tir.Function):
    """Mark pointer params that are stored to as outputs."""
    store_ptrs = set()
    _collect_store_ptrs(func.ops, store_ptrs)

    for param in func.params:
        if isinstance(param.type, PtrType) and param.name in store_ptrs:
            param.is_output = True


def _collect_store_ptrs(ops: list, store_ptrs: set):
    """Find all pointer names that appear as store destinations, recursing into loops."""
    for op in ops:
        if isinstance(op, tir.Store):
            _collect_ptr_names(op.ptr, store_ptrs)
        elif isinstance(op, tir.TileStore):
            store_ptrs.add(op.ptr.name)
        elif isinstance(op, (tir.ForRange, tir.PersistentRange, tir.SimdgroupRole)):
            _collect_store_ptrs(op.body, store_ptrs)


def _collect_ptr_names(val: tir.Value, names: set):
    """Walk back through PtrOffset chains to find the base pointer name."""
    if val.defining_op and isinstance(val.defining_op, tir.PtrOffset):
        _collect_ptr_names(val.defining_op.ptr, names)
    else:
        names.add(val.name)


def _numpy_to_dtype(np_dtype) -> str:
    mapping = {
        np.float32: "f32",
        np.float16: "f16",
        np.int32: "i32",
        np.uint32: "u32",
        np.uint8: "u8",
    }
    return mapping.get(np_dtype.type, "f32")
