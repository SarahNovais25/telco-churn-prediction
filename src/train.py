import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


DATA_PATH = "data/Telco_customer_churn.xlsx"
TARGET = "Churn Value"


def load_data():
    df = pd.read_excel(DATA_PATH)

    # remove data leakage
    leakage_cols = [
        "Churn Label",
        "Churn Score",
        "Churn Reason",
    ]

    # remove columns not useful for production API
    drop_cols = [
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

    remove_cols = leakage_cols + drop_cols

    df = df.drop(columns=[c for c in remove_cols if c in df.columns])

    # clean missing values
    df = df.replace(r"^\s*$", np.nan, regex=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def build_preprocessor(X):
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

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        results.append({
            "model": model_name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "precision_mean": scores["test_precision"].mean(),
            "recall_mean": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="roc_auc_mean",
        ascending=False
    )

    return results_df


def train_best_model(X, y, best_model_name):
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

    joblib.dump(pipeline, "models/best_model.pkl")

    print(f"\nBest model trained on full dataset: {best_model_name}")
    print("Saved at: models/best_model.pkl")


def main():
    X, y = load_data()

    results_df = run_cross_validation(X, y)

    print("\nCross Validation Results:")
    print(results_df)

    results_df.to_csv(
        "models/model_comparison_cv.csv",
        index=False
    )

    best_model_name = results_df.iloc[0]["model"]

    train_best_model(X, y, best_model_name)

    print("\nComparison saved at: models/model_comparison_cv.csv")


if __name__ == "__main__":
    main()