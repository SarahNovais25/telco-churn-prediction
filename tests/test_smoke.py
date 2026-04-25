from src.train import get_models, load_data


def test_load_data():
    X, y = load_data()

    assert X is not None
    assert y is not None
    assert len(X) > 0
    assert len(y) > 0
    assert len(X) == len(y)


def test_get_models():
    models = get_models()

    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models
    assert "mlp_classifier" in models
