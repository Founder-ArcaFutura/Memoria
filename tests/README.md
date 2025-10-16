# Memoria Test Suite

This directory hosts the automated tests for the Memoria project. The suite
covers unit-level logic, API behaviour, data pipelines, and optional
integrations with external providers.

## 📁 Directory Overview

```
tests/
├── __init__.py
├── README.md               # This file
├── config.json             # Shared configuration for tests
├── conftest.py             # Global fixtures and helpers
├── comprehensive_database_comparison.py
├── litellm_support/
│   ├── litellm_test.py
│   └── litellm_test_suite.py
├── mysql_support/
│   ├── README.md
│   ├── compare_databases.py
│   ├── litellm_mysql_test.py
│   ├── litellm_mysql_test_suite.py
│   ├── mysql_test_suite.py
│   └── setup_mysql.py
├── ollama_support/
│   ├── README.md
│   └── ollama_test.py
├── openai/
│   ├── azure_support/
│   └── openai_support/
├── openai_support/
│   ├── __init__.py
│   ├── openai_test.py
│   └── openai_test_suite.py
├── postgresql_support/
│   ├── README.md
│   ├── postgresql_test_suite.py
│   └── setup_postgresql.py
├── unit/                   # Focused unit tests
│   ├── test_anchor_search.py
│   ├── test_context_injection.py
│   ├── test_memory_manager_search.py
│   └── …
├── utils/
│   ├── factories.py
│   ├── test_exceptions.py
│   └── test_utils.py
├── test_*.py               # Feature, API, and regression coverage
└── …                       # Additional helpers and resources
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
Fine-grained coverage of the memory engine, ranking heuristics, schedulers, and
query builders. These tests isolate small pieces of logic and are fast to run.

### Specialized & Integration Tests (`tests/test_*.py` and `tests/*_support/`)
Top-level modules validate broader behaviours such as API routing, ingestion
flows, clustering, and regression scenarios. Support directories (`*_support/`)
exercise optional external systems like MySQL, PostgreSQL, LiteLLM, OpenAI, and
Ollama. These tests may require credentials or local services when run outside
of CI. Opt into running the interactive demos by exporting `MEMORIA_RUN_INTEGRATION=1`
before invoking `pytest`; otherwise they are skipped by default to avoid
triggering network calls or external dependencies during normal test runs.

## 🚀 Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run the entire suite
pytest

# Run a specific module
pytest tests/test_memory_manager.py

# Run a single test
pytest tests/unit/test_anchor_search.py::test_anchor_distance_filtering
```

### Useful Pytest Options

```bash
# Collect coverage information
pytest --cov=memoria --cov-report=html

# Limit to tests marked as unit or integration
pytest -m unit
pytest -m integration

# Skip tests marked as slow
pytest -m "not slow"

# Increase verbosity or stop after first failure
pytest -vv
pytest -x

# Focus by keyword
pytest -k "cluster and not slow"
```

## 🏷️ Available Markers

Markers are defined in `pyproject.toml`:

- `unit`: Fast tests that isolate a component.
- `integration`: Cross-component or external-service scenarios.
- `slow`: Longer-running tests that can be excluded locally.

Combine markers to narrow execution, for example:

```bash
pytest -m "unit and not slow"
pytest -m "integration and not slow"
pytest -m "not integration"
```

## ⚙️ Configuration & Fixtures

- **Global settings**: `pyproject.toml` configures pytest discovery and default
  options.
- **Fixtures**: `tests/conftest.py` provides shared fixtures for database
  managers, clients, and synthetic memory records.
- **Factories**: `tests/utils/factories.py` centralises reusable object builders
  for complex test data.
- **Test configuration**: `tests/config.json` controls default model names and
  other runtime toggles. Override with `TEST_MODEL` or related environment
  variables when needed.

## 🐛 Debugging Tips

```bash
# Run with full traceback and verbosity
pytest -vvv --tb=long

# Drop into the debugger on failure
pytest --pdb

# Show available fixtures
pytest --fixtures

# Re-run the last failing tests
pytest --last-failed
```

## 🔍 API Test Setup

The API tests in `test_compound_search.py` and `test_api_structure.py` use a
shared `sample_client` fixture (defined in `tests/conftest.py`). This fixture
seeds an in-memory database with deterministic data so that combinations of
keyword, category, spatial, and temporal filters produce predictable results.
Reuse this fixture when adding new API tests to ensure consistent behaviour
across runs.

For questions or issues with the suite, open a GitHub issue with the `testing`
label so the maintainers can triage quickly.
