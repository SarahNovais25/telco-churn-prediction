import mlflow
import mlflow.sklearn


def setup_mlflow(experiment_name: str = "telco-churn-prediction"):
    mlflow.set_experiment(experiment_name)

def set_tracking_uri(uri: str):
    mlflow.set_tracking_uri(uri)

def start_run(run_name: str = None):
    """Retorna o contexto do run do MLflow para usar com o bloco 'with'."""
    return mlflow.start_run(run_name=run_name)

def log_metrics(metrics: dict, step: int = None):
    for name, value in metrics.items():
        if step is not None:
            mlflow.log_metric(name, float(value), step=step)
        else:
            mlflow.log_metric(name, float(value))

def log_params(params: dict):
    for name, value in params.items():
        mlflow.log_param(name, value)

def log_artifacts(local_path: str):
    mlflow.log_artifact(local_path)

def log_sklearn_model(model, artifact_path: str):
    mlflow.sklearn.log_model(model, artifact_path)