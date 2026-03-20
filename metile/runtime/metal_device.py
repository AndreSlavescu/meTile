import contextlib
import ctypes
import ctypes.util
import os
import subprocess
import tempfile
from functools import cached_property
from typing import ClassVar

# Load frameworks
_objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
_metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")

# Objective-C runtime types
_id = ctypes.c_void_p
_sel = ctypes.c_void_p
_cls = ctypes.c_void_p
_bool = ctypes.c_bool
_NSUInteger = ctypes.c_uint64


class MTLSize(ctypes.Structure):
    _fields_: ClassVar = [
        ("width", ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("depth", ctypes.c_uint64),
    ]


# Objective-C runtime functions
_objc.objc_getClass.restype = _cls
_objc.objc_getClass.argtypes = [ctypes.c_char_p]

_objc.sel_registerName.restype = _sel
_objc.sel_registerName.argtypes = [ctypes.c_char_p]

_objc.objc_msgSend.restype = _id
_objc.objc_msgSend.argtypes = [_id, _sel]

# We need to cast objc_msgSend for different return types/arg types
_msg = _objc.objc_msgSend


def _sel(name: str) -> ctypes.c_void_p:
    return _objc.sel_registerName(name.encode())


def _cls(name: str) -> ctypes.c_void_p:
    return _objc.objc_getClass(name.encode())


def _send(obj, sel_name: str, *args, restype=ctypes.c_void_p, argtypes=None):
    """Send an Objective-C message with proper ctypes typing."""
    sel = _objc.sel_registerName(sel_name.encode())
    if argtypes is None:
        argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
    func = ctypes.cast(_msg, ctypes.CFUNCTYPE(restype, *argtypes))
    return func(obj, sel, *args)


def _send_ptr(obj, sel_name: str, *args, argtypes=None):
    """Send message expecting a pointer result."""
    return _send(obj, sel_name, *args, restype=ctypes.c_void_p, argtypes=argtypes)


def _send_uint64(obj, sel_name: str, *args, argtypes=None):
    """Send message expecting a uint64 result."""
    return _send(obj, sel_name, *args, restype=ctypes.c_uint64, argtypes=argtypes)


def _nsstring(s: str) -> ctypes.c_void_p:
    """Create an NSString from a Python string."""
    NSString = _cls("NSString")
    return _send_ptr(
        NSString,
        "stringWithUTF8String:",
        s.encode(),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p],
    )


def _nsstring_to_str(nsstr) -> str:
    """Convert NSString to Python string."""
    if not nsstr:
        return ""
    utf8 = _send(
        nsstr,
        "UTF8String",
        restype=ctypes.c_char_p,
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    )
    return utf8.decode() if utf8 else ""


# Metal API function: MTLCreateSystemDefaultDevice
_metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
_metal.MTLCreateSystemDefaultDevice.argtypes = []


class MetalDevice:
    """Singleton wrapper around the Metal device and command queue."""

    _instance = None

    def __init__(self):
        self.device = _metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError(
                "Metal is not available. metile requires Apple Silicon with Metal support."
            )
        self.command_queue = _send_ptr(self.device, "newCommandQueue")
        if not self.command_queue:
            raise RuntimeError("Failed to create Metal command queue.")
        self._last_cmd_buffer = None

    @classmethod
    def get(cls) -> "MetalDevice":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @cached_property
    def name(self) -> str:
        name_ns = _send_ptr(self.device, "name")
        return _nsstring_to_str(name_ns)

    @cached_property
    def max_threads_per_threadgroup(self) -> int:
        return _send_uint64(self.device, "maxThreadsPerThreadgroup")

    def compile_msl(self, source: str, function_name: str):
        """Compile MSL source and return (library, function, pipeline_state)."""
        source_ns = _nsstring(source)

        # Create compile options
        MTLCompileOptions = _cls("MTLCompileOptions")
        options = _send_ptr(MTLCompileOptions, "new")

        # newLibraryWithSource:options:error:
        error = ctypes.c_void_p(0)
        sel = _objc.sel_registerName(b"newLibraryWithSource:options:error:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,  # self
            ctypes.c_void_p,  # _cmd
            ctypes.c_void_p,  # source
            ctypes.c_void_p,  # options
            ctypes.POINTER(ctypes.c_void_p),  # error
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        library = func(self.device, sel, source_ns, options, ctypes.byref(error))

        if error.value:
            desc = _send_ptr(error, "localizedDescription")
            err_str = _nsstring_to_str(desc)
            raise RuntimeError(f"Metal compilation failed:\n{err_str}\n\nSource:\n{source}")

        if not library:
            raise RuntimeError("Metal compilation returned null library")

        # Get function from library
        func_name_ns = _nsstring(function_name)
        metal_func = _send_ptr(
            library,
            "newFunctionWithName:",
            func_name_ns,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )
        if not metal_func:
            raise RuntimeError(f"Function '{function_name}' not found in compiled Metal library")

        # Create compute pipeline state
        error = ctypes.c_void_p(0)
        sel = _objc.sel_registerName(b"newComputePipelineStateWithFunction:error:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        pipeline = func(self.device, sel, metal_func, ctypes.byref(error))

        if error.value:
            desc = _send_ptr(error, "localizedDescription")
            err_str = _nsstring_to_str(desc)
            raise RuntimeError(f"Pipeline creation failed: {err_str}")

        if not pipeline:
            raise RuntimeError("Pipeline creation returned null")

        return pipeline

    def compile_msl_precompiled(
        self, source: str, function_name: str, metal_std: str | None = None
    ):
        """Compile MSL via offline Metal compiler for better GPU performance.

        Uses xcrun metal -O2 for aggressive optimization. Falls back to
        runtime compilation if the Metal compiler is not available.

        Args:
            metal_std: Metal language standard (e.g. "metal4.0" for tensor_ops).
                       If None, uses the compiler default.
        Returns (pipeline_state, was_precompiled).
        """
        try:
            metal_path = subprocess.run(
                ["xcrun", "--find", "metal"], capture_output=True, text=True, timeout=5
            )
            if metal_path.returncode != 0:
                if metal_std:
                    raise RuntimeError(f"-std={metal_std} requires offline Metal compiler (Xcode)")
                return self.compile_msl(source, function_name), False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if metal_std:
                raise RuntimeError(
                    f"-std={metal_std} requires offline Metal compiler (Xcode)"
                ) from e
            return self.compile_msl(source, function_name), False

        with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
            f.write(source)
            msl_path = f.name

        air_path = msl_path.replace(".metal", ".air")
        lib_path = msl_path.replace(".metal", ".metallib")

        try:
            metal_cmd = ["xcrun", "-sdk", "macosx", "metal", "-O2", "-ffast-math"]
            if metal_std:
                metal_cmd.append(f"-std={metal_std}")
            metal_cmd.extend(["-o", air_path, "-c", msl_path])

            subprocess.run(metal_cmd, check=True, capture_output=True, text=True, timeout=30)
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            pipeline = self._load_metallib(lib_path, function_name)
            return pipeline, True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if metal_std:
                err_msg = getattr(e, "stderr", "") or str(e)
                raise RuntimeError(
                    f"Metal 4 compilation failed: {err_msg}\n\nSource:\n{source}"
                ) from e
            return self.compile_msl(source, function_name), False
        finally:
            for p in [msl_path, air_path, lib_path]:
                with contextlib.suppress(OSError):
                    os.unlink(p)

    def _load_metallib(self, lib_path: str, function_name: str):
        """Load a precompiled .metallib and create a pipeline state."""
        NSURL = _cls("NSURL")
        path_ns = _nsstring(lib_path)
        url = _send_ptr(
            NSURL,
            "fileURLWithPath:",
            path_ns,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )

        error = ctypes.c_void_p(0)
        sel = _objc.sel_registerName(b"newLibraryWithURL:error:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        library = func(self.device, sel, url, ctypes.byref(error))

        if error.value:
            desc = _send_ptr(error, "localizedDescription")
            raise RuntimeError(f"Failed to load metallib: {_nsstring_to_str(desc)}")

        func_name_ns = _nsstring(function_name)
        metal_func = _send_ptr(
            library,
            "newFunctionWithName:",
            func_name_ns,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )
        if not metal_func:
            raise RuntimeError(f"Function '{function_name}' not found in metallib")

        error = ctypes.c_void_p(0)
        sel = _objc.sel_registerName(b"newComputePipelineStateWithFunction:error:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        pipeline = func(self.device, sel, metal_func, ctypes.byref(error))

        if error.value:
            desc = _send_ptr(error, "localizedDescription")
            raise RuntimeError(f"Pipeline creation failed: {_nsstring_to_str(desc)}")

        return pipeline

    @cached_property
    def has_metal_compiler(self) -> bool:
        """Check if the offline Metal compiler is available (requires Xcode)."""
        try:
            result = subprocess.run(
                ["xcrun", "--find", "metal"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @cached_property
    def max_threadgroup_memory(self) -> int:
        """Max threadgroup memory in bytes (MTLDevice.maxThreadgroupMemoryLength)."""
        return _send_uint64(self.device, "maxThreadgroupMemoryLength")

    @cached_property
    def supports_tensor_ops(self) -> bool:
        """Check if device supports Metal 4 tensor_ops (M5+ and Xcode required)."""
        if not self.has_metal_compiler:
            return False
        test_src = (
            "#include <metal_stdlib>\n"
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp::tensor_ops;\n"
            "kernel void _mtile_probe() {}\n"
        )
        try:
            with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
                f.write(test_src)
                path = f.name
            result = subprocess.run(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-std=metal4.0",
                    "-c",
                    path,
                    "-o",
                    "/dev/null",
                ],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def new_buffer(self, data: bytes, length: int) -> ctypes.c_void_p:
        """Create a Metal buffer from bytes data."""
        # MTLResourceStorageModeShared = 0
        sel = _objc.sel_registerName(b"newBufferWithBytes:length:options:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,  # self
            ctypes.c_void_p,  # _cmd
            ctypes.c_void_p,  # bytes
            ctypes.c_uint64,  # length
            ctypes.c_uint64,  # options (MTLResourceOptions)
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        buf = func(self.device, sel, data, length, 0)  # 0 = StorageModeShared
        if not buf:
            raise RuntimeError("Failed to create Metal buffer")
        return buf

    def new_empty_buffer(self, length: int) -> ctypes.c_void_p:
        """Create an empty Metal buffer of given size."""
        sel = _objc.sel_registerName(b"newBufferWithLength:options:")
        func_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        )
        func = func_type(ctypes.cast(_msg, ctypes.c_void_p).value)
        buf = func(self.device, sel, length, 0)
        if not buf:
            raise RuntimeError("Failed to create empty Metal buffer")
        return buf

    def buffer_contents(self, buffer: ctypes.c_void_p) -> ctypes.c_void_p:
        """Get pointer to buffer contents."""
        return _send_ptr(buffer, "contents")

    # Cached ctypes function wrappers for hot dispatch path
    _set_buffer_sel = None
    _set_buffer_fn = None
    _dispatch_tg_sel = None
    _dispatch_tg_fn = None
    _dispatch_threads_sel = None
    _dispatch_threads_fn = None
    _set_pipeline_sel = None
    _set_pipeline_fn = None
    # Command buffer lifecycle (cached to avoid _send_ptr overhead)
    _msg_send_id = None  # zero-arg -> pointer
    _msg_send_void = None  # zero-arg -> void
    _msg_send_double = None  # zero-arg -> double (for GPU timestamps)
    _sel_commandBuffer = None
    _sel_commandBufferUnretained = None
    _sel_computeCommandEncoder = None
    _sel_endEncoding = None
    _sel_commit = None
    _sel_waitUntilCompleted = None
    _sel_GPUStartTime = None
    _sel_GPUEndTime = None

    def _ensure_cached_selectors(self):
        """Cache ctypes selectors and function types on first use."""
        if MetalDevice._set_buffer_sel is not None:
            return
        msg_ptr = ctypes.cast(_msg, ctypes.c_void_p).value

        MetalDevice._set_buffer_sel = _objc.sel_registerName(b"setBuffer:offset:atIndex:")
        MetalDevice._set_buffer_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        )(msg_ptr)

        MetalDevice._set_pipeline_sel = _objc.sel_registerName(b"setComputePipelineState:")
        MetalDevice._set_pipeline_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(msg_ptr)

        _tg_fn_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            MTLSize,
            MTLSize,
        )
        MetalDevice._dispatch_tg_sel = _objc.sel_registerName(
            b"dispatchThreadgroups:threadsPerThreadgroup:"
        )
        MetalDevice._dispatch_tg_fn = _tg_fn_type(msg_ptr)

        MetalDevice._dispatch_threads_sel = _objc.sel_registerName(
            b"dispatchThreads:threadsPerThreadgroup:"
        )
        MetalDevice._dispatch_threads_fn = _tg_fn_type(msg_ptr)

        # Command buffer lifecycle — eliminates _send_ptr overhead
        MetalDevice._msg_send_id = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(msg_ptr)
        MetalDevice._msg_send_void = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(msg_ptr)
        MetalDevice._sel_commandBuffer = _objc.sel_registerName(b"commandBuffer")
        MetalDevice._sel_commandBufferUnretained = _objc.sel_registerName(
            b"commandBufferWithUnretainedReferences"
        )
        MetalDevice._sel_computeCommandEncoder = _objc.sel_registerName(b"computeCommandEncoder")
        MetalDevice._sel_endEncoding = _objc.sel_registerName(b"endEncoding")
        MetalDevice._sel_commit = _objc.sel_registerName(b"commit")
        MetalDevice._sel_waitUntilCompleted = _objc.sel_registerName(b"waitUntilCompleted")

        # GPU timestamps (CFTimeInterval = double)
        MetalDevice._msg_send_double = ctypes.CFUNCTYPE(
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(msg_ptr)
        MetalDevice._sel_GPUStartTime = _objc.sel_registerName(b"GPUStartTime")
        MetalDevice._sel_GPUEndTime = _objc.sel_registerName(b"GPUEndTime")

    def _setup_encoder(self, pipeline, buffers, offsets=None):
        """Create command buffer, encoder, set pipeline and buffers."""
        self._ensure_cached_selectors()

        send_id = MetalDevice._msg_send_id
        cmd_buffer = send_id(self.command_queue, MetalDevice._sel_commandBuffer)
        encoder = send_id(cmd_buffer, MetalDevice._sel_computeCommandEncoder)

        MetalDevice._set_pipeline_fn(encoder, MetalDevice._set_pipeline_sel, pipeline)

        buf_fn = MetalDevice._set_buffer_fn
        buf_sel = MetalDevice._set_buffer_sel
        if offsets is None:
            for idx in range(len(buffers)):
                buf_fn(encoder, buf_sel, buffers[idx], 0, idx)
        else:
            for idx in range(len(buffers)):
                buf_fn(encoder, buf_sel, buffers[idx], offsets[idx], idx)

        return cmd_buffer, encoder

    def _finish_encoder(self, cmd_buffer, encoder):
        """End encoding, commit. Defers wait for pipelined execution."""
        send_void = MetalDevice._msg_send_void
        send_void(encoder, MetalDevice._sel_endEncoding)
        send_void(cmd_buffer, MetalDevice._sel_commit)
        self._last_cmd_buffer = cmd_buffer

    def sync(self):
        """Wait for all submitted GPU work to complete.

        Metal command queues execute in submission order, so waiting on
        the last submitted command buffer ensures all prior work is done.
        """
        cb = self._last_cmd_buffer
        if cb is not None:
            self._ensure_cached_selectors()
            MetalDevice._msg_send_void(cb, MetalDevice._sel_waitUntilCompleted)
            self._completed_cmd_buffer = cb
            self._last_cmd_buffer = None

    def gpu_elapsed(self) -> float:
        """Return GPU execution time (seconds) of last completed command buffer.

        Uses Metal's hardware GPU timestamps (GPUStartTime/GPUEndTime)
        for nanosecond-precision measurement independent of CPU scheduling.
        Must be called after sync().
        """
        cb = getattr(self, "_completed_cmd_buffer", None)
        if cb is None:
            return 0.0
        self._ensure_cached_selectors()
        start = MetalDevice._msg_send_double(cb, MetalDevice._sel_GPUStartTime)
        end = MetalDevice._msg_send_double(cb, MetalDevice._sel_GPUEndTime)
        return end - start

    def dispatch_kernel(
        self,
        pipeline,
        buffers: list[ctypes.c_void_p],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        offsets: list[int] | None = None,
    ):
        """Dispatch a compute kernel by total thread count."""
        cmd_buffer, encoder = self._setup_encoder(pipeline, buffers, offsets)

        MetalDevice._dispatch_threads_fn(
            encoder,
            MetalDevice._dispatch_threads_sel,
            MTLSize(grid[0], grid[1], grid[2]),
            MTLSize(threadgroup[0], threadgroup[1], threadgroup[2]),
        )

        self._finish_encoder(cmd_buffer, encoder)

    def dispatch_threadgroups(
        self,
        pipeline,
        buffers: list[ctypes.c_void_p],
        threadgroups: tuple[int, int, int],
        threadgroup_size: tuple[int, int, int],
        offsets: list[int] | None = None,
    ):
        """Dispatch a compute kernel by threadgroup grid (for GEMM etc)."""
        cmd_buffer, encoder = self._setup_encoder(pipeline, buffers, offsets)

        MetalDevice._dispatch_tg_fn(
            encoder,
            MetalDevice._dispatch_tg_sel,
            MTLSize(threadgroups[0], threadgroups[1], threadgroups[2]),
            MTLSize(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]),
        )

        self._finish_encoder(cmd_buffer, encoder)

    def dispatch_batch_threadgroups(
        self,
        pipeline,
        batch_buffer_sets: list[list[tuple[ctypes.c_void_p, int]]],
        threadgroups: tuple[int, int, int],
        threadgroup_size: tuple[int, int, int],
    ):
        """Batch multiple dispatches into one command buffer with buffer offsets.

        Each entry in batch_buffer_sets is a list of (metal_buffer, byte_offset) pairs.
        All dispatches use the same pipeline and grid/threadgroup dimensions.
        Eliminates per-dispatch command buffer overhead.
        """
        cmd_buffer = _send_ptr(self.command_queue, "commandBuffer")
        encoder = _send_ptr(cmd_buffer, "computeCommandEncoder")

        _send_ptr(
            encoder,
            "setComputePipelineState:",
            pipeline,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )

        tg_grid = MTLSize(threadgroups[0], threadgroups[1], threadgroups[2])
        tg_size = MTLSize(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2])

        dispatch_sel = _objc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
        dispatch_func_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            MTLSize,
            MTLSize,
        )
        dispatch_func = dispatch_func_type(ctypes.cast(_msg, ctypes.c_void_p).value)

        set_buf_sel = _objc.sel_registerName(b"setBuffer:offset:atIndex:")
        set_buf_func_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        )
        set_buf_func = set_buf_func_type(ctypes.cast(_msg, ctypes.c_void_p).value)

        for buffer_set in batch_buffer_sets:
            for idx, (buf, offset) in enumerate(buffer_set):
                set_buf_func(encoder, set_buf_sel, buf, offset, idx)
            dispatch_func(encoder, dispatch_sel, tg_grid, tg_size)

        self._finish_encoder(cmd_buffer, encoder)
