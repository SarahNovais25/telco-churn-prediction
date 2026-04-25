import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlflow_tracking import setup_mlflow, log_cv_run, log_final_model

DATA_PATH = "data/Telco_customer_churn.xlsx"
TARGET = "Churn Value"
MLFLOW_EXPERIMENT = "telco-churn-prediction"
MODEL_PATH = Path("models/")


def load_data():
    df = pd.read_excel(DATA_PATH)

    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason"]
    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude", "CLTV",
    ]
    remove_cols = leakage_cols + drop_cols

    df = df.drop(columns=[c for c in remove_cols if c in df.columns])
    df = df.replace(r"^\s*$", np.nan, regex=True).infer_objects(copy=False)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        ),
    }


def run_cross_validation(X, y):
    preprocessor = build_preprocessor(X)
    models = get_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for model_name, model in models.items():
        print(f"Running cross validation for {model_name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        scores = cross_validate(
            pipeline, X, y,
            cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1,
        )

        row = {
            "model": model_name,
            "accuracy_mean":  scores["test_accuracy"].mean(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean":    scores["test_recall"].mean(),
            "f1_mean":        scores["test_f1"].mean(),
            "roc_auc_mean":   scores["test_roc_auc"].mean(),
            "roc_auc_std":    scores["test_roc_auc"].std(),
        }
        results.append(row)

        log_cv_run(
            model_name=model_name,
            model=model,
            scores={
                "cv_accuracy_mean":  row["accuracy_mean"],
                "cv_precision_mean": row["precision_mean"],
                "cv_recall_mean":    row["recall_mean"],
                "cv_f1_mean":        row["f1_mean"],
                "cv_roc_auc_mean":   row["roc_auc_mean"],
                "cv_roc_auc_std":    row["roc_auc_std"],
            },
            fold_scores=scores["test_roc_auc"].tolist(),
        )

    results_df = pd.DataFrame(results).sort_values(by="roc_auc_mean", ascending=False)
    return results_df


def train_best_model(X, y, best_model_name, results_df):
    models = get_models()
    preprocessor = build_preprocessor(X)
    best_model = models[best_model_name]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])

    pipeline.fit(X, y)

    best_row = results_df[results_df["model"] == best_model_name].iloc[0].to_dict()

    log_final_model(
        model_name=best_model_name,
        model=best_model,
        pipeline=pipeline,
        best_row=best_row,
    )


def main():
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    setup_mlflow(MLFLOW_EXPERIMENT)

    X, y = load_data()

    results_df = run_cross_validation(X, y)

    print("\nCross Validation Results:")
    print(results_df)

    results_df.to_csv(MODEL_PATH / "model_comparison_cv.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    train_best_model(X, y, best_model_name, results_df)

    print(f"\nComparison saved at: {MODEL_PATH / 'model_comparison_cv.csv'}")


if __name__ == "__main__":
    main()