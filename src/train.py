import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import load_data, preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/random_forest_churn.joblib")


def train_model():
    df = load_data("data/Telco_customer_churn.xlsx")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "columns": X.columns.tolist(),
        },
        MODEL_PATH,
    )

    logger.info("Model saved at %s", MODEL_PATH)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()