import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from benchmarks.compiler.thread_layouts import (
    _assert_exact,
    _transpose_layout,
    ownership_transpose,
    scatter_transpose,
)
from metile.compiler.ownership import validate_thread_layouts
from metile.frontend.tracing import TracingContext, TracingProxy
from metile.ir import tile_ir as tir
from metile.ir.types import PtrType


def test_benchmark_helpers_import_without_optional_mlx():
    script = dedent(
        """
        import importlib.abc
        import sys

        class BlockMLX(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlx" or fullname.startswith("mlx."):
                    raise ModuleNotFoundError("MLX is unavailable", name=fullname)
                return None

        sys.meta_path.insert(0, BlockMLX())

        import numpy as np
        from benchmarks.compiler import thread_layouts

        expected = np.array([-0.0, 0.0, 1.0], dtype=np.float32)
        thread_layouts._assert_exact(expected.copy(), expected, "exact")
        assert thread_layouts._transpose_layout(4, 8).size == 32
        assert callable(thread_layouts.ownership_transpose.fn)
        assert callable(thread_layouts.scatter_transpose.fn)
        assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shape", [(4, 8), (8, 8), (8, 16), (16, 16)])
def test_transpose_ownership_matches_contiguous_destination_order(shape):
    rows, columns = shape
    layout = _transpose_layout(rows, columns)
    expected = np.arange(rows * columns).reshape(rows, columns).T.ravel()
    np.testing.assert_array_equal(
        [layout.logical_index(thread) for thread in range(layout.size)], expected
    )


@pytest.mark.parametrize("shape", [(4, 8), (8, 8), (8, 16), (16, 16)])
@pytest.mark.parametrize("function", [ownership_transpose, scatter_transpose])
def test_benchmark_kernels_trace_with_top_level_tensor_memory(shape, function):
    rows, columns = shape
    with TracingContext("benchmark") as context:
        source = TracingProxy(tir.Value("source", PtrType("f32")))
        destination = TracingProxy(tir.Value("destination", PtrType("f32")))
        options = {"ROWS": rows, "COLUMNS": columns}
        if function is ownership_transpose:
            options["DESTINATION_LAYOUT"] = _transpose_layout(rows, columns)
        function.fn(source, destination, 3, **options)
    assert len(context.func.tensors) == 2
    assert context.func.tensors[0].access == "read"
    assert context.func.tensors[1].access == "write"
    ownerships = validate_thread_layouts(context.func)
    assert bool(ownerships) == (function is ownership_transpose)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_benchmark_correctness_checks_bits_including_signed_zero(dtype):
    expected = np.array([-0.0, 0.0, -2.0, 3.0], dtype=dtype)
    _assert_exact(expected.copy(), expected, "exact")
    modified = expected.copy()
    modified[0] = 0.0
    with pytest.raises(AssertionError):
        _assert_exact(modified, expected, "signed_zero_mismatch")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_benchmark_correctness_rejects_nonfinite_outputs(dtype):
    expected = np.zeros(3, dtype=dtype)
    with pytest.raises(AssertionError, match="non-finite"):
        _assert_exact(np.full(3, np.nan, dtype=dtype), expected, "unwritten")


def test_benchmark_correctness_rejects_shape_or_storage_changes():
    expected = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(AssertionError, match="shape or storage dtype"):
        _assert_exact(np.zeros((3, 2), dtype=np.float32), expected, "wrong_shape")
    with pytest.raises(AssertionError, match="shape or storage dtype"):
        _assert_exact(expected.astype(np.float16), expected, "wrong_dtype")
