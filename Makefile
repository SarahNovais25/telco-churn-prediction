install:
	pip install -e .

lint:
	ruff check .

test:
	pytest

run:
	uvicorn src.api:app --reload

train:
	python src/train.py

mlflow:
	mlflow ui