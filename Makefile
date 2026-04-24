install:
	pip install -e .

lint:
	ruff check .

test:
	python3 -m pytest

run:
	python3 -m uvicorn src.api:app --reload

train:
	python3 src/train.py

mlflow:
	mlflow ui