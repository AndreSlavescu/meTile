"""Patch a name across every module of a split package.

Several integrations are packages whose modules import names from one another, so one name
can be bound in several module namespaces and each caller resolves it from its own globals.
Replacing every binding is what a single setattr on the module gave when each package was a
single file, and it is the only correct thing to do for shared mutable state, whose writers
and readers can sit in different modules.
"""

import sys


def patch_package(monkeypatch, package, name, value):
    """Bind `name` to `value` in `package` and in every module beneath it that defines it."""
    modules = [
        module
        for modname, module in list(sys.modules.items())
        if (modname == package or modname.startswith(package + "."))
        and module is not None
        and name in vars(module)
    ]
    assert modules, f"{name!r} is not bound anywhere in {package}"
    for module in modules:
        monkeypatch.setattr(module, name, value)


def _patch_mlx_lm(monkeypatch, name, value):
    patch_package(monkeypatch, "metile.integrations.mlx_lm", name, value)


def _patch_mlx_quantized(monkeypatch, name, value):
    patch_package(monkeypatch, "metile.backends.mlx_quantized", name, value)
