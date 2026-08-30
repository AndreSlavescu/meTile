BENCH_FILES := $(filter-out benchmarks/benchutils.py benchmarks/__init__.py, $(wildcard benchmarks/*.py))
PYTHON ?= python3

.PHONY: lint format check test bench code-qual ci docs

lint:
	ruff check metile/ tests/ benchmarks/

format:
	ruff format metile/ tests/ benchmarks/
	ruff check --fix metile/ tests/ benchmarks/

check: lint
	ruff format --check metile/ tests/ benchmarks/

code-qual:
	vulture metile/ --min-confidence 90 \
		--exclude "metile/ir/printer.py" \
		--ignore-names "result_type,to_msl,to_msl_mut"

test:
	$(PYTHON) -m pytest tests/ -x -q

bench:
	@for f in $(BENCH_FILES); do echo "=== $$f ===" && $(PYTHON) $$f && echo; done

ci: check code-qual test

docs:
	$(MAKE) -C docs html
