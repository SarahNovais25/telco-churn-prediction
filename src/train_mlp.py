import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_data, preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
MODEL_PATH = Path("models/mlp_churn.pt")
SCALER_PATH = Path("models/mlp_scaler.joblib")
EXPERIMENT_NAME = "telco-churn-prediction"

torch.manual_seed(SEED)
np.random.seed(SEED)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_mlp():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="pytorch_mlp"):
        df = load_data("data/Telco_customer_churn.xlsx")
        X, y = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = ChurnMLP(input_dim=X_train.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        mlflow.log_params(
            {
                "model_type": "PyTorch MLP",
                "hidden_layers": "64,32",
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "test_size": 0.2,
                "random_state": SEED,
                "threshold": 0.5,
                "input_dim": X_train.shape[1],
            }
        )

        best_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_epoch = 0

        for epoch in range(100):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                patience_counter = 0

                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)

                joblib.dump(
                    {
                        "scaler": scaler,
                        "columns": X.columns.tolist(),
                        "input_dim": X_train.shape[1],
                    },
                    SCALER_PATH,
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %s", epoch + 1)
                break

        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        with torch.no_grad():
            logits = model(X_test_tensor)
            probabilities = torch.sigmoid(logits).numpy().ravel()

        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        }

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))

        logger.info("MLP metrics: %s", metrics)

        return model, metrics


if __name__ == "__main__":
    train_mlp()
