import ast
import platform
import runpy
from pathlib import Path

import pytest

DOCUMENTATION_ROOT = Path(__file__).resolve().parents[2] / "docs"
RUNNABLE_PAGES = (
    "getting-started/first-kernel.rst",
    "examples/vector-add.rst",
    "examples/softmax.rst",
    "examples/layernorm.rst",
    "examples/matmul.rst",
    "examples/fused-activations.rst",
    "examples/attention.rst",
)


def _python_examples(relative_path):
    lines = (DOCUMENTATION_ROOT / relative_path).read_text().splitlines()
    code = []
    inside_python = False
    for line in lines:
        if line == ".. code-block:: python":
            inside_python = True
            code.append("")
        elif inside_python and (not line or line.startswith("   ")):
            code.append(line[3:] if line else "")
        else:
            inside_python = False
    return "\n".join(code) + "\n"


@pytest.mark.parametrize("relative_path", RUNNABLE_PAGES)
def test_documentation_example_syntax(relative_path):
    source = _python_examples(relative_path)
    assert source.strip()
    ast.parse(source, filename=relative_path)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Examples require the Metal runtime")
@pytest.mark.parametrize("relative_path", RUNNABLE_PAGES)
def test_documentation_example_results(relative_path, tmp_path):
    script = tmp_path / f"{Path(relative_path).stem.replace('-', '_')}.py"
    script.write_text(_python_examples(relative_path))
    runpy.run_path(str(script), run_name="__main__")
