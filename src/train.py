from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.mlflow_tracking import (
    log_artifacts, log_metrics, log_params, start_run,
    log_sklearn_model, set_tracking_uri, setup_mlflow,
)

DATA_PATH = "data/Telco_customer_churn.xlsx"
TARGET = "Churn Value"
EXPERIMENT_NAME = "telco-churn-prediction"


def load_data():
    """Load dataset, remove leakage and non-production columns, and split X/y."""
    df = pd.read_excel(DATA_PATH)

    leakage_cols = [
        "Churn Label",
        "Churn Score",
        "Churn Reason",
    ]

    non_production_cols = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "CLTV",
    ]

    cols_to_remove = leakage_cols + non_production_cols

    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

    pd.set_option("future.no_silent_downcasting", True)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def build_preprocessor(X):
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def get_models():
    """Return all models used in the comparison."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
        ),
    }


def run_cross_validation(X, y):
    """Run cross validation for all models and log results in MLflow."""
    setup_mlflow(EXPERIMENT_NAME)

    preprocessor = build_preprocessor(X)
    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    for model_name, model in models.items():
        print(f"Running cross validation for {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        with start_run(run_name=f"cv_{model_name}"):
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )

            metrics = {
                "accuracy_mean": scores["test_accuracy"].mean(),
                "precision_mean": scores["test_precision"].mean(),
                "recall_mean": scores["test_recall"].mean(),
                "f1_mean": scores["test_f1"].mean(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
                "pr_auc_mean": scores["test_pr_auc"].mean(),
                "pr_auc_std": scores["test_pr_auc"].std(),
            }

            log_params({"model_name": model_name})
            log_params(model.get_params())
            log_metrics(metrics)

            results.append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="roc_auc_mean",
        ascending=False,
    )

    return results_df


def train_best_model(X, y, best_model_name):
    """Train the best model on the full dataset and save it as a pipeline."""
    setup_mlflow(EXPERIMENT_NAME)

    models = get_models()
    preprocessor = build_preprocessor(X)

    best_model = models[best_model_name]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model),
        ]
    )

    pipeline.fit(X, y)

    Path("models").mkdir(parents=True, exist_ok=True)

    model_path = "models/best_model.pkl"
    joblib.dump(pipeline, model_path)

    with start_run(run_name=f"final_model_{best_model_name}"):
        log_params({"final_model": best_model_name})
        log_artifacts(model_path)
        log_sklearn_model(pipeline, "model")

    print(f"\nBest model trained on full dataset: {best_model_name}")
    print("Saved at: models/best_model.pkl")


def main():
    """Run model comparison, save results, and train final model."""
    X, y = load_data()

    results_df = run_cross_validation(X, y)

    print("\nCross Validation Results:")
    print(results_df)

    Path("models").mkdir(parents=True, exist_ok=True)

    comparison_path = "models/model_comparison_cv.csv"
    results_df.to_csv(comparison_path, index=False)

    mlruns_path = Path("mlruns").resolve()
    set_tracking_uri(f"file:///{mlruns_path.as_posix()}")

    setup_mlflow(EXPERIMENT_NAME)

    with start_run(run_name="model_comparison_summary"):
        log_artifacts(comparison_path)

        best_model_name = results_df.iloc[0]["model"]

        log_params({"selected_model": best_model_name})
        log_metrics({
            "best_roc_auc_mean": results_df.iloc[0]["roc_auc_mean"],
            "best_pr_auc_mean":results_df.iloc[0]["pr_auc_mean"],
         })

    best_model_name = results_df.iloc[0]["model"]

    train_best_model(X, y, best_model_name)

    print("\nComparison saved at: models/model_comparison_cv.csv")


if __name__ == "__main__":
    main()