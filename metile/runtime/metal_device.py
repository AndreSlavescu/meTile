import contextlib
import ctypes
import ctypes.util
import os
import platform
import subprocess
import tempfile
import threading
import time
from functools import cached_property
from typing import ClassVar

from metile.runtime.cache import atomic_write_bytes, cache_root, stable_digest

# Load frameworks
_objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
_metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")

# Objective-C runtime types
_id = ctypes.c_void_p
_sel = ctypes.c_void_p
_cls = ctypes.c_void_p
_bool = ctypes.c_bool
_NSUInteger = ctypes.c_uint64

_MIN_COMPLETION_SPIN_NS = 900_000
_MAX_COMPLETION_SPIN_NS = 1_500_000
_MAX_SPINNABLE_GPU_NS = 1_000_000


def completion_spin_budget_ns(gpu_seconds: float) -> int:
    """Choose a bounded completion-poll budget from measured GPU latency."""
    try:
        gpu_ns = int(gpu_seconds * 1_000_000_000)
    except (OverflowError, TypeError, ValueError):
        return 0
    if gpu_ns <= 0 or gpu_ns > _MAX_SPINNABLE_GPU_NS:
        return 0
    return min(
        _MAX_COMPLETION_SPIN_NS,
        max(_MIN_COMPLETION_SPIN_NS, 3 * gpu_ns + 300_000),
    )


class MTLSize(ctypes.Structure):
    _fields_: ClassVar = [
        ("width", ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("depth", ctypes.c_uint64),
    ]


class NSRange(ctypes.Structure):
    _fields_: ClassVar = [
        ("location", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
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


def _responds_to(obj, selector) -> bool:
    """Return whether an Objective-C object implements a selector."""
    return bool(
        _send(
            obj,
            "respondsToSelector:",
            selector,
            restype=ctypes.c_bool,
            argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )
    )


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
        self._pending_cmd_buffer = None
        self._pending_encoder = None
        self._pending_pipeline = None
        self._pending_binding_key = None
        self._pending_dispatches = 0
        self._pending_concurrent = None
        self._pending_inputs = set()
        self._pending_outputs = set()
        self._pending_lifetimes = {}
        self._pending_completion_spin_ns = 0
        self._last_completion_spin_ns = 0
        self._inflight_lifetimes = []
        self._dispatch_lock = threading.RLock()

    @classmethod
    def get(cls) -> "MetalDevice":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @cached_property
    def name(self) -> str:
        name_ns = _send_ptr(self.device, "name")
        return _nsstring_to_str(name_ns)

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

        cache_key = stable_digest(
            {
                "device": self.name,
                "function": function_name,
                "metal_std": metal_std,
                "platform": platform.mac_ver()[0],
                "source": source,
                "toolchain": self.metal_compiler_version,
            }
        )
        cached_library = cache_root() / "metallib" / f"{cache_key}.metallib"
        if cached_library.is_file():
            try:
                return self._load_metallib(str(cached_library), function_name), True
            except RuntimeError:
                with contextlib.suppress(OSError):
                    cached_library.unlink()

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
            with open(lib_path, "rb") as library_file:
                atomic_write_bytes(cached_library, library_file.read())
            pipeline = self._load_metallib(str(cached_library), function_name)
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
    def metal_compiler_version(self) -> str:
        """Return an identity for invalidating cached offline binaries."""
        if not self.has_metal_compiler:
            return "runtime"
        try:
            result = subprocess.run(
                ["xcrun", "metal", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return (result.stdout or result.stderr).strip() or "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    @cached_property
    def max_threadgroup_memory(self) -> int:
        """Max threadgroup memory in bytes (MTLDevice.maxThreadgroupMemoryLength)."""
        return _send_uint64(self.device, "maxThreadgroupMemoryLength")

    @cached_property
    def supports_tensor_ops(self) -> bool:
        """Check for both a Metal 4 GPU and a tensor-ops-capable toolchain."""
        if not self.supports_gpu_family(5002) or not self.has_metal_compiler:
            return False
        path = None
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
            if path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(path)

    def supports_gpu_family(self, family: int) -> bool:
        """Query ``MTLDevice.supportsFamily:`` without requiring PyObjC."""
        selector = _sel("supportsFamily:")
        if not _responds_to(self.device, selector):
            return False
        return bool(
            _send(
                self.device,
                "supportsFamily:",
                ctypes.c_int64(family),
                restype=ctypes.c_bool,
                argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64],
            )
        )

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
    _set_buffers_sel = None
    _set_buffers_fn = None
    _dispatch_tg_sel = None
    _dispatch_tg_fn = None
    _dispatch_threads_sel = None
    _dispatch_threads_fn = None
    _set_pipeline_sel = None
    _set_pipeline_fn = None
    # Command buffer lifecycle (cached to avoid _send_ptr overhead)
    _msg_send_id = None  # zero-arg -> pointer
    _msg_send_id_uint64 = None  # one uint64 arg -> pointer
    _msg_send_void = None  # zero-arg -> void
    _msg_send_double = None  # zero-arg -> double (for GPU timestamps)
    _msg_send_uint64 = None  # zero-arg -> uint64 (for command-buffer status)
    _sel_commandBuffer = None
    _sel_commandBufferUnretained = None
    _sel_computeCommandEncoder = None
    _sel_computeCommandEncoderConcurrent = None
    _sel_endEncoding = None
    _sel_commit = None
    _sel_waitUntilCompleted = None
    _sel_GPUStartTime = None
    _sel_GPUEndTime = None
    _sel_status = None
    _memory_barrier_sel = None
    _memory_barrier_fn = None

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
        MetalDevice._set_buffers_sel = _objc.sel_registerName(b"setBuffers:offsets:withRange:")
        MetalDevice._set_buffers_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            NSRange,
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
        MetalDevice._msg_send_id_uint64 = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
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
        MetalDevice._sel_computeCommandEncoderConcurrent = _objc.sel_registerName(
            b"computeCommandEncoderWithDispatchType:"
        )
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
        MetalDevice._msg_send_uint64 = ctypes.CFUNCTYPE(
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )(msg_ptr)
        MetalDevice._sel_status = _objc.sel_registerName(b"status")
        MetalDevice._memory_barrier_sel = _objc.sel_registerName(b"memoryBarrierWithScope:")
        MetalDevice._memory_barrier_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        )(msg_ptr)

    def _setup_encoder(self, pipeline, buffers, offsets=None):
        """Create command buffer, encoder, set pipeline and buffers."""
        self.flush()
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
        self._last_completion_spin_ns = 0

    def _pending_encoder_unlocked(self, concurrent: bool):
        """Return the shared hot-path encoder, creating it on first use."""
        concurrent = concurrent and self.supports_concurrent_dispatch
        if self._pending_encoder is not None and self._pending_concurrent != concurrent:
            self._commit_pending_unlocked()
        if self._pending_encoder is None:
            self._ensure_cached_selectors()
            send_id = MetalDevice._msg_send_id
            command_buffer_selector = (
                MetalDevice._sel_commandBufferUnretained
                if self.supports_unretained_command_buffers
                else MetalDevice._sel_commandBuffer
            )
            self._pending_cmd_buffer = send_id(self.command_queue, command_buffer_selector)
            if concurrent:
                self._pending_encoder = MetalDevice._msg_send_id_uint64(
                    self._pending_cmd_buffer,
                    MetalDevice._sel_computeCommandEncoderConcurrent,
                    1,
                )
            else:
                self._pending_encoder = send_id(
                    self._pending_cmd_buffer, MetalDevice._sel_computeCommandEncoder
                )
            self._pending_concurrent = concurrent
        return self._pending_encoder

    @cached_property
    def supports_unretained_command_buffers(self) -> bool:
        """Whether the queue supports the lower-overhead unretained command-buffer path."""
        self._ensure_cached_selectors()
        return _responds_to(self.command_queue, MetalDevice._sel_commandBufferUnretained)

    @cached_property
    def supports_concurrent_dispatch(self) -> bool:
        """Whether command buffers expose concurrent compute encoders."""
        self._ensure_cached_selectors()
        command_buffer = MetalDevice._msg_send_id(
            self.command_queue, MetalDevice._sel_commandBuffer
        )
        return bool(command_buffer) and _responds_to(
            command_buffer, MetalDevice._sel_computeCommandEncoderConcurrent
        )

    def _commit_pending_unlocked(self):
        """Close and submit the shared hot-path encoder if it has work."""
        if self._pending_encoder is None:
            return
        send_void = MetalDevice._msg_send_void
        send_void(self._pending_encoder, MetalDevice._sel_endEncoding)
        send_void(self._pending_cmd_buffer, MetalDevice._sel_commit)
        self._last_cmd_buffer = self._pending_cmd_buffer
        self._last_completion_spin_ns = (
            self._pending_completion_spin_ns if self._pending_dispatches <= 8 else 0
        )
        self._pending_cmd_buffer = None
        self._pending_encoder = None
        self._pending_pipeline = None
        self._pending_binding_key = None
        self._pending_dispatches = 0
        self._pending_concurrent = None
        self._pending_inputs.clear()
        self._pending_outputs.clear()
        self._pending_completion_spin_ns = 0
        self._inflight_lifetimes.extend(self._pending_lifetimes.values())
        self._pending_lifetimes.clear()

    def flush(self):
        """Submit automatically batched prepared dispatches without waiting."""
        with self._dispatch_lock:
            self._commit_pending_unlocked()

    def sync(self):
        """Wait for all submitted GPU work to complete.

        Metal command queues execute in submission order, so waiting on
        the last submitted command buffer ensures all prior work is done.
        """
        with self._dispatch_lock:
            self._commit_pending_unlocked()
            cb = self._last_cmd_buffer
            if cb is not None:
                self._ensure_cached_selectors()
                completed = False
                spin_ns = min(self._last_completion_spin_ns, self.low_latency_spin_ns)
                if spin_ns:
                    deadline = time.perf_counter_ns() + spin_ns
                    while MetalDevice._msg_send_uint64(cb, MetalDevice._sel_status) < 4:
                        if time.perf_counter_ns() >= deadline:
                            break
                    completed = MetalDevice._msg_send_uint64(cb, MetalDevice._sel_status) >= 4
                if not completed:
                    MetalDevice._msg_send_void(cb, MetalDevice._sel_waitUntilCompleted)
                self._completed_cmd_buffer = cb
                self._last_cmd_buffer = None
                self._last_completion_spin_ns = 0
                self._inflight_lifetimes.clear()

    @cached_property
    def low_latency_spin_ns(self) -> int:
        """Bounded active-wait window for latency-sensitive prepared dispatches."""
        value = os.environ.get("METILE_LOW_LATENCY_SPIN_US", "1500")
        try:
            microseconds = int(value)
        except ValueError:
            microseconds = 1500
        return max(0, microseconds) * 1_000

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
