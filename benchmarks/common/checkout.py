"""Resolve benchmark source trees without mixing compiler and kernel installations."""

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkout:
    root: Path
    kernel_package: str
    kernel_directory: Path

    @property
    def compiler_directory(self):
        return self.root / "metile"


def resolve_checkout(root):
    root = Path(root).resolve()
    if not (root / "metile" / "__init__.py").is_file():
        raise ValueError(f"checkout has no meTile compiler package: {root}")
    candidates = (
        ("metile.kernels", root / "metile" / "kernels"),
        ("metile_kernels", root / "kernels" / "src" / "metile_kernels"),
    )
    available = [(name, path) for name, path in candidates if (path / "__init__.py").is_file()]
    if len(available) != 1:
        raise ValueError(
            f"checkout must contain exactly one legacy or split kernel package: {root}"
        )
    package, directory = available[0]
    return Checkout(root, package, directory)


def _verify_loaded(checkout):
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        if name == "metile_kernels" or name.startswith("metile_kernels."):
            if checkout.kernel_package != "metile_kernels":
                raise RuntimeError("split kernels are already loaded for a legacy checkout")
            directory = checkout.kernel_directory
        elif name == "metile" or name.startswith("metile."):
            if (
                name == "metile.kernels" or name.startswith("metile.kernels.")
            ) and checkout.kernel_package != "metile.kernels":
                raise RuntimeError("legacy kernels are already loaded for a split checkout")
            directory = checkout.compiler_directory
        else:
            continue
        source = getattr(module, "__file__", None)
        if source is None or not Path(source).resolve().is_relative_to(directory):
            raise RuntimeError(f"{name} was imported outside the selected checkout: {source}")


def activate_checkout(root):
    """Select import paths, rejecting already-loaded modules from another checkout."""
    checkout = resolve_checkout(root)
    _verify_loaded(checkout)
    paths = [str(checkout.root)]
    if checkout.kernel_package == "metile_kernels":
        paths.append(str(checkout.kernel_directory.parent))
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    importlib.invalidate_caches()
    return checkout


def load_compiler(root):
    checkout = activate_checkout(root)
    compiler = importlib.import_module("metile")
    _verify_loaded(checkout)
    if Path(compiler.__file__).resolve().parent != checkout.compiler_directory:
        raise RuntimeError("benchmark imported the wrong compiler package")
    return compiler


def load_kernel(root, module=""):
    checkout = activate_checkout(root)
    load_compiler(checkout.root)
    package = importlib.import_module(checkout.kernel_package)
    if Path(package.__file__).resolve().parent != checkout.kernel_directory:
        raise RuntimeError("benchmark imported the wrong kernel package")
    kernel = importlib.import_module(f"{checkout.kernel_package}.{module}") if module else package
    _verify_loaded(checkout)
    return kernel


def source_paths(root):
    checkout = resolve_checkout(root)
    paths = list(checkout.compiler_directory.rglob("*.py"))
    if checkout.kernel_package == "metile_kernels":
        paths.extend(checkout.kernel_directory.rglob("*.py"))
    return sorted(paths)


def source_hashes(root):
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths(root)
    }


def implementation_hash(root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in source_paths(root):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def benchmark_sources(*drivers):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), *(Path(driver) for driver in drivers))
    }


def benchmark_fingerprint(*drivers):
    import json

    payload = json.dumps(benchmark_sources(*drivers), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def regression_script(root):
    root = Path(root).resolve()
    for relative in ("benchmarks/regression/regression.py", "benchmarks/regression.py"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise ValueError(f"checkout has no regression benchmark: {root}")
