install:
	pip install -e .

lint:
	ruff check .

test:
	python3 -m pytest

run:
	uvicorn src.api:app --reload

train:
	python3 src/train.py

mlflow:
	mlflow ui