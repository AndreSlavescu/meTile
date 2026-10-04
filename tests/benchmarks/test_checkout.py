import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.common import checkout
from benchmarks.regression.paired_regression import _run_sample

ROOT = Path(__file__).resolve().parents[2]


def make_checkout(root, layout):
    compiler = root / "metile"
    compiler.mkdir(parents=True)
    (compiler / "__init__.py").write_text("identity = 'selected compiler'\n")
    kernels = compiler / "kernels" if layout == "legacy" else root / "kernels/src/metile_kernels"
    kernels.mkdir(parents=True)
    (kernels / "__init__.py").write_text("identity = 'selected kernels'\n")
    (kernels / "probe.py").write_text("value = 17\n")
    return root


def run_probe(root, setup=""):
    script = (
        "import importlib, json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from benchmarks.common import checkout\n"
        + setup
        + "module = checkout.load_kernel(sys.argv[1], 'probe')\n"
        "compiler = checkout.load_compiler(sys.argv[1])\n"
        "print(json.dumps({'compiler': compiler.__file__, 'kernel': module.__file__, 'value': module.value}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(root)], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("layout", ("legacy", "split"))
def test_selected_checkout_supplies_compiler_and_kernels_even_with_foreign_installation(
    tmp_path, layout
):
    selected = make_checkout(tmp_path / "selected", layout)
    foreign = make_checkout(tmp_path / "installed", "split")
    setup = f"sys.path[:0] = {[str(foreign), str(foreign / 'kernels/src')]!r}\n"
    result = run_probe(selected, setup)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert Path(paths["compiler"]).is_relative_to(selected)
    assert Path(paths["kernel"]).is_relative_to(selected)
    assert paths["value"] == 17


@pytest.mark.parametrize("module", ("metile", "metile_kernels"))
def test_loaded_foreign_compiler_or_kernels_are_rejected(tmp_path, module):
    selected = make_checkout(tmp_path / "selected", "split")
    foreign = make_checkout(tmp_path / "installed", "split")
    setup = (
        f"sys.path[:0] = {[str(foreign), str(foreign / 'kernels/src')]!r}\n"
        f"importlib.import_module({module!r})\n"
    )
    result = run_probe(selected, setup)
    assert result.returncode != 0
    assert "outside the selected checkout" in result.stderr


def test_legacy_checkout_rejects_preloaded_split_kernels(tmp_path):
    selected = make_checkout(tmp_path / "selected", "legacy")
    foreign = make_checkout(tmp_path / "installed", "split")
    result = run_probe(
        selected,
        f"sys.path.insert(0, {str(foreign / 'kernels/src')!r})\n"
        "importlib.import_module('metile_kernels')\n",
    )
    assert result.returncode != 0
    assert "split kernels are already loaded for a legacy checkout" in result.stderr


def test_missing_selected_kernels_never_fall_back_to_an_installed_package(tmp_path):
    selected = tmp_path / "selected"
    (selected / "metile").mkdir(parents=True)
    (selected / "metile/__init__.py").write_text("")
    foreign = make_checkout(tmp_path / "installed", "split")
    result = run_probe(selected, f"sys.path.insert(0, {str(foreign / 'kernels/src')!r})\n")
    assert result.returncode != 0
    assert "exactly one legacy or split kernel package" in result.stderr


def test_ambiguous_checkout_rejected(tmp_path):
    root = make_checkout(tmp_path / "selected", "split")
    (root / "metile/kernels").mkdir()
    (root / "metile/kernels/__init__.py").write_text("")
    with pytest.raises(ValueError, match="exactly one"):
        checkout.resolve_checkout(root)


def test_foreign_kernel_submodule_is_rejected_even_with_matching_package(tmp_path):
    root = make_checkout(tmp_path / "selected", "split")
    foreign = make_checkout(tmp_path / "installed", "split")
    package = root / "kernels/src/metile_kernels/__init__.py"
    package.write_text(f"__path__.insert(0, {str(foreign / 'kernels/src/metile_kernels')!r})\n")
    result = run_probe(root)
    assert result.returncode != 0
    assert "metile_kernels.probe was imported outside the selected checkout" in result.stderr


def test_split_kernel_changes_are_bound_by_root_relative_source_hashes(tmp_path):
    root = make_checkout(tmp_path / "selected", "split")
    sources = checkout.source_hashes(root)
    assert set(sources) == {
        "metile/__init__.py",
        "kernels/src/metile_kernels/__init__.py",
        "kernels/src/metile_kernels/probe.py",
    }
    before = checkout.implementation_hash(root)
    (root / "kernels/src/metile_kernels/probe.py").write_text("value = 18\n")
    assert checkout.implementation_hash(root) != before


def test_legacy_fingerprint_preserves_the_original_hash_algorithm(tmp_path):
    root = make_checkout(tmp_path / "selected", "legacy")
    expected = hashlib.sha256()
    for path in sorted((root / "metile").rglob("*.py")):
        expected.update(path.relative_to(root).as_posix().encode() + b"\0")
        expected.update(path.read_bytes() + b"\0")
    assert checkout.implementation_hash(root) == expected.hexdigest()


@pytest.mark.parametrize("layout", ("legacy", "split"))
def test_paired_regression_worker_loads_each_checkout_layout_without_gpu(tmp_path, layout):
    root = make_checkout(tmp_path / "selected", layout)
    relative = (
        "benchmarks/regression.py" if layout == "legacy" else "benchmarks/regression/regression.py"
    )
    driver = root / relative
    driver.parent.mkdir(parents=True)
    driver.write_text(
        "_COOLDOWN = 0\n"
        "def _geomean(values): return sum(values) / len(values)\n"
        "def bench_gemm(): return {'gemm': 1.0}\n"
        "def bench_softmax(): return {'softmax': 2.0}\n"
        "def bench_layernorm(): return {'layernorm': 3.0}\n"
        "def bench_fft(): return {'fft': 4.0}\n"
    )
    assert checkout.regression_script(root) == driver
    result = _run_sample(root, tmp_path / "result.json", 1, tmp_path / "cache")
    assert result == {"gemm": 1.0, "softmax": 2.0, "layernorm": 3.0, "fft": 4.0}


def test_metadata_helper_import_preserves_selected_checkout_paths(tmp_path):
    root = make_checkout(tmp_path / "selected", "legacy")
    setup = (
        "checkout.activate_checkout(sys.argv[1])\n"
        "before = list(sys.path)\n"
        "from benchmarks.mlx import mlx_lm_backend\n"
        "assert sys.path == before\n"
    )
    result = run_probe(root, setup)
    assert result.returncode == 0, result.stderr


def test_driver_fingerprint_binds_the_checkout_helper(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("value = 1\n")
    sources = checkout.benchmark_sources(driver)
    assert (
        sources["checkout.py"] == hashlib.sha256(Path(checkout.__file__).read_bytes()).hexdigest()
    )
    before = checkout.benchmark_fingerprint(driver)
    driver.write_text("value = 2\n")
    assert checkout.benchmark_fingerprint(driver) != before
