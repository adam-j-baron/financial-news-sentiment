.PHONY: install test lint format check docs clean

install:
	poetry install

test:
	poetry run pytest tests/ --cov=financial_news_sentiment --cov-report=term-missing

lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/
	poetry run bandit -r src/
	poetry run interrogate src/ -v

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

check: lint test

docs:
	poetry run pdoc --html --output-dir docs/_build/html src/financial_news_sentiment

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf docs/_build
	rm -rf dist
	rm -rf *.egg-info
