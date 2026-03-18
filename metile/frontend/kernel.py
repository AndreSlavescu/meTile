from __future__ import annotations

import inspect
import os
import struct
import sys

import numpy as np

from metile.codegen.msl_emitter import emit
from metile.compiler.lowering import lower
from metile.compiler.passes import (
    block_swizzle,
    double_buffer_k_loop,
    fold_constants,
    pad_shared_memory,
    preload_mma_tiles,
    serpentine_mma,
    split_elementwise_loops,
    split_k_loop,
    swizzle_shared_memory,
    vectorize_elementwise,
    vectorize_loads,
)
from metile.frontend.tracing import TracingContext, TracingProxy, constexpr
from metile.ir import metal_ir as mir
from metile.ir import tile_ir as tir
from metile.ir.types import I32, PtrType, ScalarType
from metile.runtime.buffer import MtileBuffer
from metile.runtime.metal_device import MetalDevice, MTLSize

# Global kernel cache: (func_name, constexprs_tuple, dtypes_tuple) -> CompiledKernel
_kernel_cache: dict = {}
# Scalar buffer cache: (value, format_char) -> metal_buffer
_scalar_buffer_cache: dict = {}

_ELEM_SIZES = {"float": 4, "half": 2, "int": 4, "uint": 4}


def _validate_threadgroup_memory(metal_ir: mir.MFunction):
    """Raise RuntimeError if threadgroup memory exceeds hardware limit."""
    total_bytes = 0
    for op in metal_ir.ops:
        if isinstance(op, mir.MThreadgroupAlloc):
            total_bytes += op.size * _ELEM_SIZES.get(op.elem_type, 4)
    if total_bytes == 0:
        return
    limit = MetalDevice.get().max_threadgroup_memory
    if total_bytes > limit:
        raise RuntimeError(
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
    ):
        self.pipeline = pipeline
        self.msl_source = msl_source
        self.func_name = func_name
        self.threadgroup_size = threadgroup_size
        self.is_gemm = is_gemm


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
    """Zero-overhead repeated dispatch. Created by KernelLauncher.prepare().

    Pre-resolves all Metal buffers, MTLSize structs, and ctypes function
    pointers as instance attributes so __call__ goes straight to cached
    ctypes Metal API calls with no Python arg processing, cache lookup,
    isinstance checks, or class attribute lookups.
    """

    __slots__ = (
        "_buffers",
        "_cq",
        "_dev",
        "_dispatch_fn",
        "_dispatch_sel",
        "_grid",
        "_pipeline",
        "_sel_cb",
        "_sel_commit",
        "_sel_enc",
        "_sel_end",
        "_send_id",
        "_send_void",
        "_set_buf_fn",
        "_set_buf_sel",
        "_set_pipe_fn",
        "_set_pipe_sel",
        "_tg",
    )

    def __init__(self, compiled, metal_buffers, grid, dev):
        dev._ensure_cached_selectors()
        self._pipeline = compiled.pipeline
        self._buffers = tuple(metal_buffers)
        self._dev = dev
        self._cq = dev.command_queue

        # Pre-cache all ctypes functions as instance attrs
        self._send_id = MetalDevice._msg_send_id
        self._send_void = MetalDevice._msg_send_void
        self._sel_cb = MetalDevice._sel_commandBufferUnretained
        self._sel_enc = MetalDevice._sel_computeCommandEncoder
        self._set_pipe_fn = MetalDevice._set_pipeline_fn
        self._set_pipe_sel = MetalDevice._set_pipeline_sel
        self._set_buf_fn = MetalDevice._set_buffer_fn
        self._set_buf_sel = MetalDevice._set_buffer_sel
        self._sel_end = MetalDevice._sel_endEncoding
        self._sel_commit = MetalDevice._sel_commit

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
        send_id = self._send_id
        send_void = self._send_void

        cmd_buffer = send_id(self._cq, self._sel_cb)
        encoder = send_id(cmd_buffer, self._sel_enc)

        self._set_pipe_fn(encoder, self._set_pipe_sel, self._pipeline)

        fn = self._set_buf_fn
        sel = self._set_buf_sel
        bufs = self._buffers
        for idx in range(len(bufs)):
            fn(encoder, sel, bufs[idx], 0, idx)

        self._dispatch_fn(encoder, self._dispatch_sel, self._grid, self._tg)

        send_void(encoder, self._sel_end)
        send_void(cmd_buffer, self._sel_commit)
        self._dev._last_cmd_buffer = cmd_buffer


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
        return FastDispatcher(
            self._last_compiled, self._last_metal_buffers, self.grid, MetalDevice.get()
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
        is_gemm = metal_ir.kernel_type in ("gemm", "persistent_gemm")
        is_tensor_ops = metal_ir.kernel_type == "tensor_ops_gemm"
        is_specialized = metal_ir.kernel_type == "specialized_gemm"
        use_swizzle = constexprs.get("SWIZZLE_SMEM", False)
        if is_specialized:
            # Specialized GEMM: double-buffered + padded in lowering
            # Only apply vectorize and serpentine
            metal_ir = vectorize_loads(metal_ir, vec_size=4)
            metal_ir = serpentine_mma(metal_ir)
        elif is_gemm:
            if use_swizzle:
                metal_ir = swizzle_shared_memory(metal_ir)
            else:
                metal_ir = pad_shared_memory(metal_ir)
            metal_ir, did_db = double_buffer_k_loop(metal_ir)
            if not did_db:
                metal_ir = split_k_loop(metal_ir)
            metal_ir = vectorize_loads(metal_ir, vec_size=4)
            metal_ir = serpentine_mma(metal_ir)
            metal_ir = preload_mma_tiles(metal_ir)
            metal_ir = block_swizzle(metal_ir)
        else:
            metal_ir = split_elementwise_loops(metal_ir)
            metal_ir = vectorize_elementwise(metal_ir, vec_size=4)

        # Constant folding (all kernel types)
        metal_ir = fold_constants(metal_ir)

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

        return CompiledKernel(
            pipeline=pipeline,
            msl_source=msl_source,
            func_name=metal_ir.name,
            threadgroup_size=metal_ir.threadgroup_size,
            is_gemm=is_gemm or is_tensor_ops or is_specialized,
        )

    def _dispatch(self, compiled: CompiledKernel, args):
        """Bind buffers and dispatch kernel. Returns metal buffer list for prepare()."""
        dev = MetalDevice.get()

        buffers = []
        for arg in args:
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
    }
    return mapping.get(np_dtype.type, "f32")
