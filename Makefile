BENCH_FILES := $(filter-out benchmarks/benchutils.py benchmarks/__init__.py, $(wildcard benchmarks/*.py))

.PHONY: lint format check test bench code-qual ci docs

lint:
	ruff check metile/ tests/ benchmarks/ kernels/

format:
	ruff format metile/ tests/ benchmarks/ kernels/
	ruff check --fix metile/ tests/ benchmarks/ kernels/

check: lint
	ruff format --check metile/ tests/ benchmarks/ kernels/

code-qual:
	vulture metile/ kernels/ --min-confidence 90 \
		--exclude "metile/ir/printer.py" \
		--ignore-names "result_type,to_msl,to_msl_mut"

test:
	python -m pytest tests/ -x -q

bench:
	@for f in $(BENCH_FILES); do echo "=== $$f ===" && python $$f && echo; done

ci: check code-qual test

docs:
	$(MAKE) -C docs html
