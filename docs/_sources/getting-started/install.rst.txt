Installation
============

meTile runs GPU kernels through Metal on Apple Silicon Macs. It requires Python
3.10 or later and NumPy. At runtime, it also loads the macOS Metal and
Objective-C frameworks.

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

If you only want to write your own kernels, install the compiler with
``python -m pip install -e .``. It does not depend on ``metile-kernels``.
Add the library later with
``python -m pip install -e ./kernels``. Its source lives in
``kernels/src/metile_kernels/`` and it depends on ``metile``.

Optional extras have separate dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m pip install -e ".[benchmarks]"
   python -m pip install -e ".[mlx-lm]" -e ./kernels

The ``docs`` extra installs Sphinx and the documentation theme.
``benchmarks`` adds plotting support, and ``mlx-lm`` adds MLX model
dependencies. MLX integration also needs ``metile-kernels``; the combined
command installs both projects. You do not need the kernel library to plot
saved results or build the docs.

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

The package metadata does not enforce a single minimum macOS version.
Backend support depends on the installed OS, device and toolchain. Run the
checks below rather than assuming every feature works on an older macOS release.

Check the installation
----------------------

Start with the IR tests, then run a small GPU example:

.. code-block:: bash

   python -m pytest tests/ir/test_ir.py -q
   python -m pytest tests/docs/test_documentation_examples.py -q

The first command checks compiler data structures without running kernels.
The second runs the complete tutorial examples and compares GPU output with
NumPy, so it requires the Apple Silicon/Metal setup described above.

The tutorial's attention example uses ``metile.backends.attention_runtime``,
whose GPU kernels come from the companion library; it does not need MLX.
For the full suite, install both projects and the benchmark extra:

.. code-block:: bash

   python -m pip install -e ".[dev,benchmarks]" -e ./kernels
   python -m pytest tests/ -x -q

``make test`` is the corresponding Makefile target. Some integration tests
need optional MLX packages or locally cached models. Continue with
:doc:`first-kernel` once the basic examples pass.
