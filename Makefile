.PHONY: install install-ml test reproduce reproduce-smoke reproduce-real example figures audit clean

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[ml,data,dev]"

test:
	python -m pytest tests/ -v

reproduce:
	python experiments/reproduce_all.py

reproduce-smoke:
	python experiments/reproduce_all.py --smoke

reproduce-real:
	python experiments/reproduce_all.py --real

example:
	python examples/worked_example.py

figures:
	python figures/generate_all.py

audit:
	python audit/audit_analysis.py

clean:  # never touches tracked results/reference/
	find results -maxdepth 1 -type f ! -name .gitkeep -delete
	rm -rf results/worked_example
