# Contributing to Memoria

We welcome contributions to Memoria! This document provides guidelines to help you get started, collaborate smoothly, and ship high-quality improvements.

## 🚀 Quick Start

1. **Fork** the repository.
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/memoria.git`.
3. **Create** a branch: `git checkout -b feature/your-feature-name`.
4. **Install** development dependencies: `pip install -e ".[dev]"`.
5. **Make** your changes.
6. **Test** your changes: `pytest`.
7. **Format** your code: `black memoria/ tests/` and `ruff check memoria/ tests/ --fix`.
8. **Run** syntax checks: `python -m py_compile $(git ls-files '*.py')`.
9. **Commit** and **push** your changes.
10. **Create** a pull request.

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear, descriptive title**.
2. **Steps to reproduce** the bug.
3. **Expected behavior**.
4. **Actual behavior**.
5. **Environment details**, such as:
   - Python version
   - Memoria version
   - Operating system
   - Database type (if applicable)
6. **Code snippet** or minimal example.
7. **Error messages** and stack traces.

Use the bug report template when creating issues.

## 💡 Feature Requests

When suggesting new features:

1. **Check existing issues** to avoid duplicates.
2. **Describe the problem** the feature would solve.
3. **Explain the proposed solution**.
4. **Consider implementation complexity**.
5. **Provide use cases** and examples.

Use the feature request template when creating issues.

## 👥 Community

- **Be respectful** and inclusive.
- **Help others** learn and contribute.
- **Ask questions** if you're unsure.
- **Share knowledge** through discussions.

### Getting Help

- **GitHub Discussions**: For questions and general discussion.
- **GitHub Issues**: For bug reports and feature requests.
- **Documentation**: Check docs for common questions.

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setup Development Environment

```bash
# Clone the repository
   git clone https://github.com/Founder-ArcaFutura/Memoria.git
   cd Memoria

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=memoria --cov-report=html

# Run specific test file
pytest tests/test_memory_manager.py

# Run integration-style tests
pytest -m integration
```

### Keep evaluation scenarios up to date

- Follow the [evaluation quickstart](docs/getting-started/quick-start.md) to run
  `examples/evaluation_quickstart.py` against any retrieval, policy, or ranking change.
- Attach the console or JSON report to pull requests that alter behaviour so
  reviewers can see benchmark deltas.
- When expectations shift, update the shared scenario JSON so downstream
  contributors inherit the new baseline.
- Need broader smoke coverage? The [Evaluation suites
  workflow](.github/workflows/evaluation.yml) runs automatically on pull requests
  that touch retrieval, policy, or evaluation modules and can also be dispatched
  manually from the Actions tab. Share the `summary.md` artifact produced by
  `scripts/ci/run_evaluation_suites.py` with reviewers. Secret requirements and
  override options are catalogued in
  [docs/configuration/evaluation-suites.md](docs/configuration/evaluation-suites.md#ci-automation).

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black memoria/ tests/ examples/ scripts/
isort memoria/ tests/ examples/ scripts/

# Lint code
ruff check memoria/ tests/ examples/ scripts/

# Syntax check
python -m py_compile $(git ls-files '*.py')

# Type checking (strict settings enabled for core storage modules)
python -m mypy

# Security checks
bandit -r memoria/
safety check
```

> **Note:** The mypy configuration now runs with `strict`, `allow_untyped_globals = false`,
> and other strictness flags enabled for `memoria/core/memory.py` and
> `memoria/storage/service.py`. Keep these files fully typed, add precise Protocols or
> typed aliases when calling helpers from less-typed modules, and prefer tightening or
> removing `ignore_errors` overrides instead of adding new ones as you expand type
> coverage.

## 📋 Contribution Guidelines

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Use [Black](https://black.readthedocs.io/) for code formatting.
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Write type hints for all functions and methods.
- Keep line length to 88 characters.

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes

Examples:

```
feat(memory): add context-aware memory retrieval
fix(database): resolve connection timeout issues
docs(readme): update installation instructions
```

### Pull Request Process

1. **Create descriptive PR title** following conventional commit format.
2. **Fill out PR template** with all required information, including the new evidence checklist.
3. **Link related issues** using keywords (fixes #123, closes #456).
4. **Ensure all checks pass**:
   - Tests pass
   - Code coverage maintained
   - Code style checks pass
   - Security scans pass
5. **Attach governance policy dry-run evidence whenever policies or automated safeguards change.**
6. **Attach evaluation suite summaries for retrieval, ranking, or quality-affecting changes.**
7. **Request review** from maintainers.

### Governance & Evaluation Workflow

- **Issue templates**: Use the `Change Request` issue template for governance, automation, or evaluation initiatives. Populate the impact grid, attach dry-run artifacts, and capture stakeholder sign-off so reviewers can route approval quickly.
- **Pull request template**: Complete the governance/evaluation evidence checklist in `.github/pull_request_template.md`. Link to evaluation suite summaries, governance dry-run traces, and automation diffs that demonstrate guardrails remain intact.
- **Release guides**: Run `python scripts/releases/generate_release_guides.py --phase <phase>` (or render the full guide) to align your rollout plan with the staged launch playbooks. Reference the generated checklist inside the "Additional Notes" section of the PR template when proposing beta, release candidate, or GA changes.
- **Telemetry & analytics**: Whenever automation or analytics surfaces change, update the roadmap trackers and include a note in your PR about how the metrics roll up into the phased release dashboards.
- **Escalations**: Email [maintainers@memoria.dev](mailto:maintainers@memoria.dev) for sensitive governance discussions or to coordinate incident response alongside template submissions.
8. **Address feedback** promptly.

### Reviewer Guidance for Automation Evidence

- **Expect policy dry-run attachments** whenever the PR modifies governance rules, automated enforcement, or compliance workflows. Request the dry-run output if the checkbox is unchecked or evidence is missing.
- **Expect evaluation suite artifacts** (logs, summary reports, or dashboards) for any change that impacts retrieval quality, ranking behaviour, or benchmark definitions. Trigger the evaluation workflow or ask authors to rerun it if attachments are missing.
- **Confirm automation coverage** when scripts, workflows, or bots are updated. Ensure authors document how the automation was validated and include any relevant logs.

### Documentation

- Update documentation for any new features or API changes.
- Add docstrings to all public functions and classes.
- Use Google style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

## 🏗️ Development Guidelines

### Architecture Principles

Memoria follows these architectural principles:

1. **Modular Design**: Keep components loosely coupled.
2. **Clean Interfaces**: Use clear, documented APIs.
3. **Database Agnostic**: Support multiple database backends.
4. **LLM Agnostic**: Work with any LLM provider.
5. **Type Safety**: Use static typing throughout.
6. **Error Handling**: Provide clear, actionable error messages.

### Adding New Features

When adding new features:

1. **Start with an issue** describing the feature.
2. **Design the API** before implementation.
3. **Write tests first** (TDD approach recommended).
4. **Implement incrementally** with small, focused commits.
5. **Document thoroughly** including examples.
6. **Consider backward compatibility**.

### Database Migrations

When modifying database schemas:

1. **Create migration files** in `memoria/database/migrations/`.
2. **Test migrations** on sample data.
3. **Document migration steps**.
4. **Consider rollback procedures**.
5. Legacy migration scripts live in `scripts/migrations/archive/` and are only needed for upgrading databases from older versions (for example, `remove_emotional_intensity.py`).

### Integration Testing

For new integrations:

1. **Create integration tests** in `tests/integration/`.
2. **Mock external services** when possible.
3. **Test error conditions** and edge cases.
4. **Document integration setup**.

## 🚢 Release Process

Releases are managed by maintainers:

1. Version bump in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Create release tag.
4. Automated CI/CD handles PyPI publishing.

- **Maintainer Contact**: Adrian Hau <adrian@arca-futura.com>

## 📄 License

By contributing to Memoria, you agree that your contributions will be licensed under the Apache License 2.0.

## 🏆 Recognition

Contributors will be recognized in:

- `CHANGELOG.md` for their contributions.
- GitHub contributors list.
- Release notes for significant contributions.

Thank you for contributing to Memoria!
