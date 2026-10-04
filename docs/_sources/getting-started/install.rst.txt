Installation
============

meTile runs GPU kernels on Apple Silicon Macs through Metal. Use Python 3.10
or later. NumPy is the only required Python runtime dependency; the runtime
also loads the macOS Metal and Objective-C frameworks.

Install from the repository
---------------------------

.. code-block:: bash

   git clone https://github.com/AndreSlavescu/meTile.git
   cd meTile
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]" -e ./kernels

This installs two editable projects: ``metile`` provides the language,
compiler, and runtime; ``metile-kernels`` provides ready-made kernels under
the ``metile_kernels`` namespace. The ``dev`` extra adds pytest, Ruff, and
Vulture.

To write your own kernels without the ready-made library, install only the
compiler project with ``python -m pip install -e .``. It has no dependency on
``metile-kernels``. You can add the library later with
``python -m pip install -e ./kernels``. Its source lives in
``kernels/src/metile_kernels/`` and it depends on ``metile``.

Optional extras have separate dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m pip install -e ".[benchmarks]"
   python -m pip install -e ".[mlx-lm]" -e ./kernels

The ``docs`` extra installs Sphinx and the documentation theme;
``benchmarks`` adds plotting support; ``mlx-lm`` adds the MLX model
dependencies. The MLX integration also needs ``metile-kernels``, so install
both projects with the combined command above. Plotting saved results and
building documentation do not require the kernel library.

Metal compiler and feature support
----------------------------------

Basic kernels can use Metal's runtime source compiler without the offline
Metal compiler. When ``xcrun`` can locate ``metal``, meTile normally compiles
with ``-O2 -ffast-math`` and caches the resulting ``.metallib``.
The cache avoids repeating offline compilation for matching source, device,
OS, and toolchain identities; it does not eliminate kernel launch overhead.

Check the selected developer toolchain with:

.. code-block:: bash

   xcrun --find metal
   xcrun -sdk macosx metal --version

If those commands fail, install or select an Xcode toolchain with the Metal
compiler available. Installing the standalone Command Line Tools does not,
by itself, guarantee that ``metal`` is present.

Metal 4 tensor operations require both a capable GPU/runtime and an offline
compiler that accepts ``-std=metal4.0`` and the tensor-ops headers. meTile
probes this combination before choosing that backend. The direct NAX path
adds its own target and shape requirements. Do not infer these capabilities
from an M-series chip name or a Python package installation alone.

The project does not enforce a single minimum macOS version in its package
metadata. Available backends depend on the installed OS, device, and
toolchain; passing the checks below is more useful than assuming that every
feature works on an older macOS release.

Check the installation
----------------------

Start with the IR tests, then run a small GPU example:

.. code-block:: bash

   python -m pytest tests/ir/test_ir.py -q
   python -m pytest tests/docs/test_documentation_examples.py -q

The first command checks compiler data structures. The second executes the
complete tutorial examples and compares GPU output with NumPy. GPU execution
requires the Apple Silicon/Metal environment described above.

The tutorial's attention example uses ``metile.backends.attention_runtime``,
whose GPU kernels come from the companion library; it does not need MLX.
For the full suite, install both projects and the benchmark extra:

.. code-block:: bash

   python -m pip install -e ".[dev,benchmarks]" -e ./kernels
   python -m pytest tests/ -x -q

``make test`` is the corresponding Makefile target. Some integration tests
need optional MLX packages or locally cached models. Continue with
:doc:`first-kernel` once the basic examples pass.
