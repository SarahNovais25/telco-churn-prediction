from pathlib import Path
import joblib
import mlflow


MODEL_PATH = Path("models/")


def setup_mlflow(experiment_name: str = "telco-churn-prediction"):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    mlruns_path = MODEL_PATH.resolve() / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")
    mlflow.set_experiment(experiment_name)


def log_cv_run(model_name: str, model, scores: dict, fold_scores: list):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(scores)

        for fold_idx, fold_score in enumerate(fold_scores):
            mlflow.log_metric("cv_roc_auc_fold", fold_score, step=fold_idx)

        mlflow.set_tag("stage", "cross_validation")


def log_final_model(model_name: str, model, pipeline, best_row: dict):
    model_file = MODEL_PATH / "best_model.pkl"
    joblib.dump(pipeline, model_file)

    with mlflow.start_run(run_name=f"{model_name}_final"):
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "cv_accuracy_mean":  best_row["accuracy_mean"],
            "cv_precision_mean": best_row["precision_mean"],
            "cv_recall_mean":    best_row["recall_mean"],
            "cv_f1_mean":        best_row["f1_mean"],
            "cv_roc_auc_mean":   best_row["roc_auc_mean"],
            "cv_roc_auc_std":    best_row["roc_auc_std"],
        })
        mlflow.set_tag("stage", "final_model")
        mlflow.set_tag("best_model", model_name)

        # loga o .pkl como artefato simples — evita PermissionError no Windows
        mlflow.log_artifact(str(model_file))

    print(f"\nBest model trained on full dataset: {model_name}")
    print(f"Saved at: {model_file}")