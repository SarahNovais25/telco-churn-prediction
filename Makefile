install:
	pip install -e .

lint:
	python -m ruff format .
	python -m ruff check . --fix
	
test:
	python -m pytest

run:
	python -m uvicorn src.api:app --reload

train:
	python src/train.py

mlflow:
	mlflow ui --backend-store-uri models/mlruns