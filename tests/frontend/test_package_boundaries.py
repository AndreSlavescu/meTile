import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNELS_SOURCE = ROOT / "kernels" / "src"


def _run_isolated(source, *source_paths):
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(source), *map(str, source_paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compiler_traces_and_emits_without_reference_kernels():
    _run_isolated(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        sys.path[:0] = sys.argv[1:]

        class BlockOptionalPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"metile_kernels", "mlx", "mlx_lm"}:
                    raise AssertionError(f"Compiler imported optional package: {fullname}")

        sys.meta_path.insert(0, BlockOptionalPackages())

        import metile
        from metile.codegen.msl_emitter import emit
        from metile.compiler.lowering import lower
        from metile.frontend.tracing import TracingContext, TracingProxy
        from metile.ir import tile_ir as tir
        from metile.ir.types import PtrType

        assert Path(metile.__file__).resolve() == Path(sys.argv[1]) / "metile" / "__init__.py"

        @metile.kernel
        def increment(source, destination):
            offsets = metile.arange(0, 32)
            values = metile.load(source + offsets)
            metile.store(destination + offsets, values + 1.0)

        with TracingContext("increment") as context:
            context.func.params = [
                tir.Param("source", PtrType("f32")),
                tir.Param("destination", PtrType("f32"), is_output=True),
            ]
            source = TracingProxy(tir.Value("source", PtrType("f32")))
            destination = TracingProxy(tir.Value("destination", PtrType("f32")))
            increment.fn(source, destination)

        source = emit(lower(context.func))
        assert "[[kernel]]" in source
        assert "void mtile_increment(" in source
        assert not any(name.startswith("metile_kernels") for name in sys.modules)
        """,
        ROOT,
    )


def test_reference_kernels_import_from_separate_namespace_without_device():
    _run_isolated(
        """
        import importlib
        import pkgutil
        import sys
        from pathlib import Path

        sys.path[:0] = sys.argv[1:]

        import metile
        from metile.runtime.metal_device import MetalDevice

        def unexpected_device_access():
            raise AssertionError("Importing reference kernels must not initialize a GPU")

        MetalDevice.get = staticmethod(unexpected_device_access)

        import metile_kernels

        expected = Path(sys.argv[2]) / "metile_kernels" / "__init__.py"
        assert Path(metile_kernels.__file__).resolve() == expected
        for module in pkgutil.iter_modules(metile_kernels.__path__, "metile_kernels."):
            importlib.import_module(module.name)

        from metile_kernels.attention_runtime import attention_decode

        assert metile_kernels.attention_decode is attention_decode
        assert callable(attention_decode[(1,)].prepare)
        assert metile_kernels.matmul.kernel_fn.fn.__module__ == "metile_kernels.gemm"
        assert not hasattr(metile, "kernels")
        assert "metile.runtime.attention" not in sys.modules
        """,
        ROOT,
        KERNELS_SOURCE,
    )


def test_distribution_metadata_keeps_kernels_optional_for_compiler():
    tomllib = pytest.importorskip("tomllib")
    compiler = tomllib.loads((ROOT / "pyproject.toml").read_text())
    kernels = tomllib.loads((ROOT / "kernels" / "pyproject.toml").read_text())

    assert compiler["tool"]["setuptools"]["packages"]["find"]["include"] == ["metile", "metile.*"]
    assert kernels["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert kernels["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "metile_kernels",
        "metile_kernels.*",
    ]
    assert not any(
        dependency.startswith("metile-kernels")
        for dependency in compiler["project"]["dependencies"]
    )
    assert f"metile=={compiler['project']['version']}" in kernels["project"]["dependencies"]
    for extra in ("mlx", "mlx-lm"):
        assert (
            f"metile-kernels=={kernels['project']['version']}"
            in compiler["project"]["optional-dependencies"][extra]
        )


def test_dense_backend_signature_tracks_companion_kernel_source(monkeypatch):
    monkeypatch.syspath_prepend(str(KERNELS_SOURCE))
    from metile.backends import mlx_dense

    original_signature = mlx_dense.mlx_dense_backend_signature()
    original_getsource = mlx_dense.inspect.getsource

    def changed_kernel_source(function):
        source = original_getsource(function)
        if function is mlx_dense.matmul.kernel_fn.fn:
            return source + "\n"
        return source

    monkeypatch.setattr(mlx_dense.inspect, "getsource", changed_kernel_source)

    assert mlx_dense.mlx_dense_backend_signature() != original_signature
