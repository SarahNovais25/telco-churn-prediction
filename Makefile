install:
	python -m pip install -e .

lint:
	python -m ruff check .

format:
	python -m ruff format .

test:
	python -m pytest

train:
	python -m src.train

run:
	python -m uvicorn src.api:app --reload

mlflow:
	python -m mlflow ui