# Contributing to pdfmux

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/NameetP/pdfmux.git
cd pdfmux
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
```

## Running tests

```bash
# full suite (151 tests)
pytest

# with coverage
pytest --cov=pdfmux --cov-report=term-missing

# single file
pytest tests/test_pipeline.py -v
```

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Both checks run in CI and must pass before merge.

## Code style

- **Type annotations** on all public functions and method signatures.
- **Frozen dataclasses** for all data objects flowing through the pipeline.
- **Docstrings** on all public functions (Google style).
- **Imports**: use `from __future__ import annotations` in every module.
- Line length: 100 characters (configured in `pyproject.toml`).

## Adding a new extractor

1. Create `src/pdfmux/extractors/your_extractor.py`.
2. Implement the `Extractor` protocol (see `extractors/__init__.py`):
   - `name` property
   - `available()` method — return `True` only if deps are installed
   - `extract(file_path, pages=None)` — yield `PageResult` per page
3. Register with `@register(name="your_name", priority=N)`.
4. Add an optional dependency group in `pyproject.toml`.
5. Add tests in `tests/test_your_extractor.py`.
6. Update `pdfmux doctor` in `cli.py` to check for the new extractor.

## Pull request process

1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Run `pytest` and `ruff check` locally.
4. Open a PR against `main` with a clear description.
5. CI must pass (lint + test on Python 3.11/3.12/3.13 + build).

## Commit messages

Use imperative mood: "Add feature" not "Added feature". Keep the first line under 72 characters.

## Questions?

Open an [issue](https://github.com/NameetP/pdfmux/issues) or start a discussion.
