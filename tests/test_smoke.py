from src.train import train_model


def test_train_runs():
    model, X_test, y_test = train_model()
    assert model is not None