# Contributing to EEG Motor Imagery Classification

Thank you for contributing to this research codebase.

## Development Workflow
1. Fork and clone the repository.
2. Create a virtual environment (`uv venv` or `python -m venv .venv`).
3. Install dependencies: `make install`.
4. Install pre-commit hooks: `pre-commit install`.
5. Run code checks before opening a pull request: `make check && make lint && make test`.

## Code Style Guidelines
- Format code using Ruff: `make format`.
- Ensure strict type annotations for public APIs.
- Write unit tests in `tests/` for new functionality.
- Never commit dataset files, checkpoints, or secrets.
