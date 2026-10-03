from types import SimpleNamespace

import pytest

import metile.frontend.kernel as kernel_module
from metile.compiler.options import Schedule
from metile.frontend.kernel import KernelFunction, KernelLauncher
from metile.frontend.tracing import constexpr


def _first(value, BLOCK: constexpr):
    return value + 1


def _second(value, BLOCK: constexpr):
    return value + 2


@pytest.fixture
def compilations(monkeypatch):
    records = []

    def compile_kernel(launcher, arguments, constants, parameter_names):
        compiled = SimpleNamespace(function=launcher.kernel_fn.fn, constants=dict(constants))
        records.append(compiled)
        return compiled

    monkeypatch.setattr(kernel_module, "_kernel_cache", {})
    monkeypatch.setattr(KernelLauncher, "_compile", compile_kernel)
    monkeypatch.setattr(KernelLauncher, "_dispatch", lambda *_: [])
    return records


def test_same_name_functions_do_not_share_compiled_kernels(compilations):
    first, second = KernelFunction(_first), KernelFunction(_second)
    first.name = second.name = "same_name"
    first_launcher, second_launcher = first[(1,)], second[(1,)]

    first_launcher(64, BLOCK=64)
    second_launcher(64, BLOCK=64)

    assert len(compilations) == 2
    assert first_launcher._last_compiled is not second_launcher._last_compiled


def test_closures_with_identical_source_do_not_share_compiled_kernels(compilations):
    def make_kernel(increment):
        def closure(value, BLOCK: constexpr):
            return value + increment

        return KernelFunction(closure)

    first, second = make_kernel(1), make_kernel(2)
    assert first.fn.__code__ is second.fn.__code__

    first[(1,)](64, BLOCK=64)
    second[(1,)](64, BLOCK=64)

    assert len(compilations) == 2


def test_rewrapping_the_same_function_reuses_its_compiled_kernel(compilations):
    first, second = KernelFunction(_first), KernelFunction(_first)

    first[(1,)](64, BLOCK=64)
    second[(1,)](64, BLOCK=64)

    assert len(compilations) == 1


@pytest.mark.parametrize("option", ["METILE_ONLINE_SOFTMAX", "METILE_SCHEDULE"])
def test_compiler_environment_changes_invalidate_kernel_cache(compilations, monkeypatch, option):
    launcher = KernelFunction(_first)[(1,)]
    monkeypatch.setenv(option, "0")
    launcher(64, BLOCK=64)
    monkeypatch.setenv(option, "1")
    launcher(64, BLOCK=64)
    launcher(64, BLOCK=64)

    assert len(compilations) == 2


def test_debug_environment_does_not_change_compiled_code_identity(compilations, monkeypatch):
    launcher = KernelFunction(_first)[(1,)]
    monkeypatch.delenv("METILE_DEBUG", raising=False)
    launcher(64, BLOCK=64)
    monkeypatch.setenv("METILE_DEBUG", "schedule")
    launcher(64, BLOCK=64)

    assert len(compilations) == 1


def test_equivalent_default_compiler_environment_reuses_kernel_cache(compilations, monkeypatch):
    launcher = KernelFunction(_first)[(1,)]
    monkeypatch.delenv("METILE_ONLINE_SOFTMAX", raising=False)
    monkeypatch.delenv("METILE_SCHEDULE", raising=False)
    launcher(64, BLOCK=64)
    monkeypatch.setenv("METILE_ONLINE_SOFTMAX", "1")
    monkeypatch.setenv("METILE_SCHEDULE", "0")
    launcher(64, BLOCK=64)

    assert len(compilations) == 1


def test_schedule_requirements_distinguish_kernel_variants(compilations):
    launcher = KernelFunction(_first)[(1,)]

    launcher(64, BLOCK=64, SCHEDULE=Schedule(vector_width=1))
    launcher(64, BLOCK=64, SCHEDULE=Schedule(vector_width=4))
    launcher(64, BLOCK=64, SCHEDULE=Schedule(vector_width=4))

    assert len(compilations) == 2
