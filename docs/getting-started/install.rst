Installation
============

Requirements
------------

- macOS 13 (Ventura) or later
- Apple Silicon (M1, M2, M3, M4 — any variant)
- Python 3.10+

Install
-------

.. code-block:: bash

   git clone https://github.com/AndreSlavescu/meTile.git
   cd meTile
   pip install -e ".[dev]"

This installs meTile in editable mode with development dependencies (pytest, ruff, vulture).

The only runtime dependency is **numpy**.

Xcode Command Line Tools (optional)
------------------------------------

meTile works out of the box without Xcode. It uses JIT compilation via Metal's
``newLibraryWithSource`` API.

For faster **ahead-of-time** compilation, install the Xcode Command Line Tools:

.. code-block:: bash

   xcode-select --install

When available, meTile compiles shaders with ``xcrun metal -O2`` and caches the resulting
``.metallib`` files. This is faster for repeated launches.

Verify Installation
-------------------

.. code-block:: bash

   python -m pytest tests/test_ir.py -v

You should see all tests pass. To run the full test suite:

.. code-block:: bash

   python -m pytest tests/ -x -q

Or use the Makefile shorthand:

.. code-block:: bash

   make test
