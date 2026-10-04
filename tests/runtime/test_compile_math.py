import subprocess
from pathlib import Path

import pytest

from metile.runtime import metal_device as runtime
from metile.runtime.metal_device import MetalDevice


def test_offline_math_policy_changes_flags_and_cache_identity(monkeypatch, tmp_path):
    device = MetalDevice.__new__(MetalDevice)
    device.__dict__.update(name="test", metal_compiler_version="test")
    commands = []
    libraries = []

    def run(command, **kwargs):
        commands.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"compiled")
        return subprocess.CompletedProcess(command, 0, stdout="metal", stderr="")

    def load(path, name):
        libraries.append(path)
        return name

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(device, "_load_metallib", load)
    for fast_math in (True, False, True):
        assert device.compile_msl_precompiled("source", "kernel", fast_math=fast_math) == (
            "kernel",
            True,
        )

    compilations = [command for command in commands if "-c" in command]
    assert len(compilations) == 2
    assert "-ffast-math" in compilations[0]
    assert "-fno-fast-math" in compilations[1]
    assert libraries[0] != libraries[1]
    assert libraries[0] == libraries[2]


@pytest.mark.parametrize("failure", ["missing", "timeout", "compile"])
@pytest.mark.parametrize("fast_math", [False, True])
def test_offline_fallback_preserves_math_policy(monkeypatch, tmp_path, failure, fast_math):
    device = MetalDevice.__new__(MetalDevice)
    device.__dict__.update(name="test", metal_compiler_version="test")
    calls = []

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        if failure == "missing":
            return subprocess.CompletedProcess(command, 1)
        if "-c" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    def compile_runtime(source, name, **options):
        calls.append((source, name, options))
        return "pipeline"

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(device, "compile_msl", compile_runtime)

    assert device.compile_msl_precompiled("source", "kernel", fast_math=fast_math) == (
        "pipeline",
        False,
    )
    assert calls == [("source", "kernel", {"fast_math": fast_math})]
