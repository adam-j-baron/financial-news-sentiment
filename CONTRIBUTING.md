# Contributing to Financial News Sentiment Analysis

First off, thank you for considering contributing to the Financial News Sentiment Analysis project! We welcome contributions from the community and are happy to work with you to make this project better.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching on GitHub under Issues.
- If you're unable to find an open issue addressing the problem, open a new one.
- Include a clear title and description, as much relevant information as possible, and a code sample demonstrating the issue.

### Suggesting Enhancements

- First, read the documentation and make sure the feature doesn't already exist.
- Open an issue on GitHub describing the enhancement you'd like to see.
- Include clear reasoning why this enhancement would be useful to most users.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure all tests pass and code quality checks succeed.
5. Submit your pull request!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone <your-fork-url>
   cd financial-news-sentiment
   ```

2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install dependencies:
   ```bash
   poetry install --with dev
   ```

4. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Development Process

### Code Quality

Before submitting a PR, ensure your code:

1. Is formatted with Black:
   ```bash
   poetry run black src tests
   ```

2. Has imports sorted with isort:
   ```bash
   poetry run isort src tests
   ```

3. Passes type checking with mypy:
   ```bash
   poetry run mypy src
   ```

4. Passes security checks with bandit:
   ```bash
   poetry run bandit -r src
   ```

5. Has adequate test coverage:
   ```bash
   poetry run pytest --cov=financial_news_sentiment
   ```

### Documentation

- All public modules, classes, and functions must have docstrings.
- Use Google-style docstring format.
- Update API documentation if you add/change functionality:
  ```bash
  poetry run make docs
  ```

### Testing

- Write unit tests for new functionality.
- Update existing tests when changing behavior.
- Ensure all tests pass before submitting a PR:
  ```bash
  poetry run pytest
  ```

### Commit Messages

Follow conventional commits format:
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code changes that neither fix bugs nor add features
- perf: Performance improvements
- test: Adding or updating tests
- chore: Updating build tasks, package manager configs, etc.

Example:
```
feat(models): add support for custom model configurations

- Add CustomModelConfig class
- Update documentation
- Add tests
```

### Branch Naming

Use descriptive branch names that reflect the changes:
- feature/add-new-model
- fix/incorrect-sentiment
- docs/update-api-docs
- refactor/improve-performance

## Release Process

1. Update version in pyproject.toml and version.py
2. Update CHANGELOG.md
3. Create a tagged release on GitHub
4. CI/CD will handle publishing to PyPI

## Questions?

Feel free to open an issue for:
- Help with development setup
- Questions about contributing
- Clarification on project direction
- Technical questions about implementation

Thank you for contributing!
