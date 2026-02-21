# Bayesian Engine v0.1 (Consensus Nexus)

[![CI](https://github.com/mfold111/bayesian-consensus-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/mfold111/bayesian-consensus-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Open-source Python tool for Bayesian-weighted consensus from multiple signals with persistent reliability tracking.

## MVP Scope
- Python implementation
- CLI + importable library
- SQLite reliability DB
- JSON input via file or stdin
- Structured JSON output report
- Exponential reliability decay

## Quickstart
```bash
poetry install
poetry run bayesian-engine --help
```

## Development

### Local Preflight
Before pushing, run all quality checks locally:
```bash
# Install dependencies
poetry install

# Lint
poetry run ruff check .
poetry run ruff format --check .

# Type check
poetry run mypy src

# Run all tests
poetry run pytest -v

# Or run specific test markers
poetry run pytest -m unit -v
poetry run pytest -m integration -v
poetry run pytest -m simulation -v
```

### CI Checks
All pull requests must pass:
- **Lint**: `ruff check` and `ruff format --check`
- **Type check**: `mypy src`
- **Unit tests**: Core functionality tests

## License
MIT
