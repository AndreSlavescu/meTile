# Contributing to meTile

Start with a focused change and a way to check its behavior. Compiler and
runtime changes need an Apple Silicon Mac for GPU validation; formatting and
documentation builds can run elsewhere.

## Set up the checkout

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,benchmarks,docs]" -e ./kernels
```

The root project provides the compiler and runtime. The separate `kernels/`
project provides the `metile_kernels` library; development and the full test
suite use both. For the optional MLX integration, install
`python -m pip install -e ".[mlx-lm]" -e ./kernels` as well. A compiler-only
installation remains `python -m pip install -e .`.

See the installation guide for Metal compiler requirements; installing Python
dependencies does not install an offline Metal compiler.

## Check the change

```bash
make check
make code-qual
python -m pytest tests/kernels/test_gemm.py -q
python -m pytest tests/compiler -q
```

Choose the test file or subsystem that exercises your change, then run `make ci` before
submitting. It combines lint, formatting checks, dead-code detection, and the
test suite. `make format` rewrites files and applies available lint fixes;
it is separate from the read-only checks.

Tests are grouped by the behavior they check:

| Directory | Coverage |
| --- | --- |
| `tests/benchmarks/` | Benchmark harnesses and reports |
| `tests/codegen/` | Lowering and shader emission |
| `tests/compiler/` | Compiler passes, planning, and fusion |
| `tests/docs/` | Runnable documentation examples |
| `tests/frontend/` | Tracing, tensors, cache, and autotuning contracts |
| `tests/integrations/` | MLX integration and model compatibility |
| `tests/ir/` | IR types and layouts |
| `tests/kernels/` | Kernel numerical correctness |
| `tests/runtime/` | Memory and dispatch |
| `tests/target/` | Apple GPU capabilities and ISA |

For documentation, build with warnings treated as errors and run the examples
on a Metal-capable machine:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m pytest tests/docs/test_documentation_examples.py -q
```

The example tests execute code extracted from the tutorial pages and compare
its results with NumPy. Keep complete examples runnable, state their dtype
and layout assumptions, and exercise partial tiles when bounds matter.

## Inspect generated code

```bash
METILE_DEBUG=all python your_script.py
```

Useful flags include `tile_ir`, `metal_ir`, `metal_ir_opt`, `schedule`,
`msl`, and `all`; combine them with commas. Compilation dumps appear on
standard error and, where supported, under `debug_output/`. Set
`METILE_DEBUG_DIR` to choose another directory. Start a fresh process when
you need a dump of a specialization that is already cached.

For compiler-pass changes, check both matrix and elementwise kernels. For
code-generation changes, inspect the MSL and run numerical comparisons,
including relevant edge shapes and dtypes.

## Measure performance

The [performance dashboard](https://andreslavescu.github.io/meTile/dev/bench/)
tracks the regression benchmark suite. The workflow uses `macos-latest`;
do not assume a fixed chip model across runs.

Pushes to `main` publish benchmark results and configure alerts at 115% of
the baseline. The alert does not fail that job. Pull requests run a paired
base/PR benchmark with `continue-on-error: true`, so a reported regression
is currently advisory rather than a blocking gate.

Run the same benchmark locally before and after a performance-sensitive change:

```bash
python -m benchmarks.regression.regression --output baseline.json
python -m benchmarks.regression.regression --compare baseline.json
```

Save the baseline before changing the code. Use the same device and toolchain,
record the shape and dtype, and avoid simultaneous GPU workloads. `make bench`
runs only the numerical regression suite by default. Choose other modules
explicitly, for example:

```bash
make bench BENCH_MODULES="benchmarks.kernels.gemm benchmarks.kernels.softmax"
```

Benchmark modules are grouped under `compiler/`, `hardware/`, `kernels/`,
`mlx/`, `regression/`, and `plots/`; shared helpers live in `common/` and
recorded artifacts remain in `results/`. Run module commands from the repository
root. Model studies and hardware probes require their own explicit commands.

Source relocation changes benchmark fingerprints. Re-export baselines and
retune before validating a new run against a frozen manifest; keep historical
manifests and result artifacts unchanged.

## Open a pull request

Explain the behavior that changed, why it matters, and how you checked it.
Include numerical tolerances when results can differ, and benchmark conditions
when claiming a speedup. Mention checks you could not run. Keep unrelated
cleanup separate so reviewers can assess the change on its own.
