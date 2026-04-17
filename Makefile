.PHONY: install install-dev test lint format clean score warm-cache nbclear nb-execute \
        book book-clean book-serve

PYTHON ?= python

install:
	pip install -e ".[inference,rag,training,agents,serving,eval,gpu,dev]"

install-dev:
	pip install -e ".[dev]"

warm-cache:
	$(PYTHON) scripts/warm_cache.py

fetch-data:
	$(PYTHON) scripts/fetch_data.py

test:
	pytest scoring/ -v

lint:
	ruff check src/ scoring/ scripts/
	ruff format --check src/ scoring/ scripts/

format:
	ruff format src/ scoring/ scripts/
	ruff check --fix src/ scoring/ scripts/

nbclear:
	find notebooks -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' \
		-exec jupyter nbconvert --clear-output --inplace {} +

nb-execute:
	@for nb in $$(find notebooks -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*'); do \
		echo "Executing $$nb"; \
		jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

score:
	$(PYTHON) -m scoring.run_all

book:
	jupyter-book build .

book-clean:
	rm -rf _build/

book-serve: book
	@echo "Open file://$$(pwd)/_build/html/index.html in your browser."
	$(PYTHON) -m http.server --directory _build/html 8765

clean:
	rm -rf .pytest_cache .ruff_cache _build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf scores/ data/ .cache/
