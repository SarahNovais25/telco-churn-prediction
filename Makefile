install:
	python -m pip install -e .

lint:
	python -m ruff check .

format:
	python -m ruff format .

test:
	python -m pytest

train:
<<<<<<< HEAD
	python -m src.train
=======
	python src/train.py
>>>>>>> 9b6d07bfed8ae1096e8194e6a7549e00e22c7ad8

run:
	python -m uvicorn src.api:app --reload

mlflow:
<<<<<<< HEAD
	python -m mlflow ui
=======
	python -m mlflow ui
>>>>>>> 9b6d07bfed8ae1096e8194e6a7549e00e22c7ad8
