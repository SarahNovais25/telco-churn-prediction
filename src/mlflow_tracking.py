import mlflow


def setup_mlflow(experiment_name: str = "telco-churn-prediction"):
    mlflow.set_experiment(experiment_name)


def log_metrics(metrics: dict):
    for name, value in metrics.items():
        mlflow.log_metric(name, float(value))


def log_params(params: dict):
    for name, value in params.items():
        mlflow.log_param(name, value)
