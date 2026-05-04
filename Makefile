install:
	python3 -m pip install -e .

lint:
	python3 -m ruff check .

format:
	python3 -m ruff format .

test:
	python3 -m pytest

train:
	python3 src/train.py

run:
	python3 -m uvicorn src.api:app --reload

mlflow:
	python3 -m mlflow ui