import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.mlflow_tracking import log_metrics, log_params, setup_mlflow
from src.train import build_preprocessor, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
MODEL_PATH = Path("models/mlp_churn.pt")
SCALER_PATH = Path("models/mlp_preprocessor.joblib") 
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
    setup_mlflow(EXPERIMENT_NAME)
    
    # Mantendo o tracking uri para salvar corretamente na pasta models/mlruns
    mlruns_path = MODEL_PATH.parent.resolve() / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")

    with mlflow.start_run(run_name="pytorch_mlp"):
        
        # 1. Carregando dados EXATAMENTE como no pipeline de produção (sem leak e sem non-prod cols)
        X, y = load_data()

        # 2. Divisão em Treino, Validação e Teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=SEED, stratify=y_temp
        )

        # 3. Aplicando o ColumnTransformer oficial do projeto
        preprocessor = build_preprocessor(X_train)
        
        # O fit deve ser APENAS no X_train
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # 4. Preparação dos tensores
        X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        
        X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        
        X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), 
            batch_size=64, 
            shuffle=True
            )
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

        # O input_dim agora reflete a saída exata do OneHotEncoder do sklearn
        input_dim = X_train_processed.shape[1]
        model = ChurnMLP(input_dim=input_dim)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        params ={
                "model_type": "PyTorch MLP",
                "hidden_layers": "64,32",
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "test_size": 0.2,
                "val_size": 0.15,
                "random_state": SEED,
                "input_dim": input_dim,
            } 
        log_params(params)
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_epoch = 0

        for epoch in range(100):
            # Treinamento
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_loss += criterion(model(batch_X), batch_y).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            mlflow.log_metrics(
                {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, 
                step=epoch+1
                )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                
                # Agora salvamos o ColumnTransformer inteiro, garantindo consistência
                joblib.dump(
                    {
                        "preprocessor": preprocessor, 
                        "input_dim": input_dim
                    }, 
                    SCALER_PATH
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Teste final
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test_tensor)).numpy().ravel()
        
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
        
        log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))

        logger.info("MLP metrics: %s", metrics)

        return model, metrics

if __name__ == "__main__":
    train_mlp()
