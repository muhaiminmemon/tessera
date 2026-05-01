# Contributing to Tessera

Thank you for your interest in contributing. This document covers how to get set up, the coding standards we follow, and how to submit changes.

## Setup

```bash
git clone https://github.com/muhaiminmemon/tessera
cd tessera
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then add your API keys
```

## Running checks

```bash
make test       # run all tests
make lint       # ruff + mypy
make fmt        # auto-format with ruff
make check      # lint + test (run before every PR)
```

Or without make:

```bash
pytest tests/ -v
ruff check tessera/ tests/
mypy tessera/ --ignore-missing-imports
```

## Adding a new task type

1. Add a `TaskType.<NAME>` member to `tessera/core/models.py`
2. Add a `<Name>Spec` Pydantic model in `tessera/core/models.py`
3. Add prompt functions to `tessera/core/prompts.py` following the naming convention: `<name>_taxonomy_system/user`, `<name>_generation_system/user`, `<name>_critique_system/user`
4. Register the prompt functions in the dispatch dicts in `tessera/pipeline/taxonomy.py`, `tessera/pipeline/generation.py`, and `tessera/pipeline/critique.py`
5. Create `tessera/tasks/<name>.py` implementing `TaskTemplate`
6. Export from `tessera/__init__.py`
7. Add a CLI subcommand in `tessera/cli.py`
8. Add tests in `tests/`

## Code style

- **Formatter**: ruff (line length 100)
- **Types**: all public functions must have type annotations; Pydantic v2 syntax only
- **Logging**: use `logging.getLogger(__name__)` — no `print()` in library code
- **Exceptions**: raise from `tessera.core.exceptions`, not bare `ValueError`/`Exception`
- **No magic numbers**: temperatures, thresholds, and concurrency limits belong in `tessera/core/config.py`
- **No hardcoded strings in task files**: all prompts go in `prompts.py`

## Pull request checklist

- [ ] `make check` passes (lint + all tests green)
- [ ] New code has corresponding tests
- [ ] No new `print()` calls
- [ ] `CHANGELOG.md` updated under `## Unreleased`
