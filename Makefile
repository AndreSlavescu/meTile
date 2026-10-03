BENCH_MODULES ?= benchmarks.regression.regression
PYTHON ?= python3

.PHONY: lint format check test bench code-qual ci docs

lint:
	ruff check metile/ kernels/src/ tests/ benchmarks/

format:
	ruff format metile/ kernels/src/ tests/ benchmarks/
	ruff check --fix metile/ kernels/src/ tests/ benchmarks/

check: lint
	ruff format --check metile/ kernels/src/ tests/ benchmarks/

code-qual:
	vulture metile/ kernels/src/ --min-confidence 90 \
		--exclude "metile/ir/printer.py" \
		--ignore-names "result_type,to_msl,to_msl_mut"

test:
	$(PYTHON) -m pytest tests/ -x -q

bench:
	@for benchmark_module in $(BENCH_MODULES); do \
		$(PYTHON) -m "$$benchmark_module" || exit $$?; \
	done

ci: check code-qual test

docs:
	$(MAKE) -C docs html
