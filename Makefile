BENCH_FILES := $(filter-out benchmarks/benchutils.py benchmarks/__init__.py, $(wildcard benchmarks/*.py))

.PHONY: lint format check test bench clean

lint:
	ruff check metile/ tests/ benchmarks/

format:
	ruff format metile/ tests/ benchmarks/
	ruff check --fix metile/ tests/ benchmarks/

check: lint
	ruff format --check metile/ tests/ benchmarks/

test:
	python -m pytest tests/ -x -q

bench:
	@for f in $(BENCH_FILES); do echo "=== $$f ===" && python $$f && echo; done
