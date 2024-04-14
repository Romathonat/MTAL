.PHONY: test coverage

test:
	pdm run pytest

coverage:
	pdm run pytest --cov=src --cov-report=xml

coverage-html:
	pdm run pytest --cov=src --cov-report=html