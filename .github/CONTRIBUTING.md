# Contributing to meTile

## Setup

```bash
pip install -e ".[dev]"
```

## Development Workflow

```bash
make format    # auto-format + fix lint
make check     # lint + format check (no changes)
make test      # run tests
make code-qual # dead code detection
make ci        # all of the above
make bench     # run all benchmarks
```

## Debug Output

Set `METILE_DEBUG` to inspect the compilation pipeline:

```bash
METILE_DEBUG=all python your_script.py
```

Flags: `tile_ir`, `metal_ir`, `metal_ir_opt`, `msl`, `all` (comma-separated). Output is written to `debug_output/` by default.

## Running Tests

```bash
python -m pytest tests/ -x -q           # all tests
python -m pytest tests/test_gemm.py -v  # specific file
```

## Performance Regression Dashboard

Kernel performance is tracked across commits:

**https://andreslavescu.github.io/meTile/dev/bench/**

Every push to `main` runs benchmarks on an M1 runner and publishes results. PRs automatically compare against the baseline and fail if any kernel regresses by more than 15%.

To run regression benchmarks locally:

```bash
python benchmarks/regression.py                          # print results
python benchmarks/regression.py --output baseline.json   # save baseline
python benchmarks/regression.py --compare baseline.json  # compare + fail on regression
```

## PR Guidelines

- Run `make ci` before submitting
- If modifying compiler passes, verify both GEMM and elementwise tests pass
- If modifying codegen, check generated MSL with `METILE_DEBUG=msl`
- Performance-sensitive changes should include benchmark results in the PR description
