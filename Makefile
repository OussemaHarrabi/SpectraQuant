# SpectraQuant developer tasks.
#
# `make` is not installed on the Windows workstation of record; these targets are the Linux/CI
# spelling of the commands documented in AGENTS.md section 8 and are what CI runs.

UV ?= uv
PYTEST_ARGS ?= -q

.PHONY: help sync check lint format type test smoke clean env

help:
	@echo "sync    - create/refresh the uv environment (all extras)"
	@echo "check   - lint + format check + type check + tests (what CI runs)"
	@echo "smoke   - run the tiny end-to-end experiment"
	@echo "clean   - remove caches and throwaway outputs (never touches artifacts/)"

sync:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format --check .

type:
	$(UV) run pyright

test:
	$(UV) run pytest $(PYTEST_ARGS)

check: lint format type test

smoke:
	$(UV) run spectraquant smoke --config configs/experiment/smoke.yaml

env:
	$(UV) run spectraquant env

clean:
	rm -rf outputs mlruns .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
