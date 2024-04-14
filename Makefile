.PHONY: test coverage

test:
	pdm run pytest --cov=src --cov-report=term --cov-report=xml

coverage-html:
	pdm run pytest --cov=src --cov-report=html