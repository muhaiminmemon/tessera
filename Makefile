.PHONY: test lint fmt check typecheck clean

# ── Dev shortcuts ────────────────────────────────────────────────────────────

## Run full test suite with coverage
test:
	pytest tests/ -v --tb=short --cov=tessera --cov-report=term-missing

## Run tests quickly (no coverage)
test-fast:
	pytest tests/ -q

## Lint with ruff and mypy
lint:
	ruff check tessera/ tests/
	mypy tessera/ --ignore-missing-imports

## Auto-format with ruff
fmt:
	ruff format tessera/ tests/ scripts/ examples/
	ruff check tessera/ tests/ --fix

## Run lint + tests (use before every PR)
check: lint test

## Type check only
typecheck:
	mypy tessera/ --ignore-missing-imports

## Remove build artifacts and caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

## Install in dev mode
install:
	pip install -e ".[dev]"
