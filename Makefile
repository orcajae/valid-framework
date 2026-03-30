.PHONY: install test reproduce figures audit clean

install:
	pip install -e ".[ml,dev]"

test:
	python -m pytest tests/ -v

reproduce:
	python experiments/reproduce_all.py

figures:
	python figures/generate_all.py

audit:
	python audit/audit_analysis.py

clean:
	rm -rf results/* paper/figures/*.png
