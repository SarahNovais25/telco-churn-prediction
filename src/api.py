import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- 1. Configuração de Logging Estruturado ---
class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Extrai dicionários passados via 'extra={"extra_info": {...}}'
        if hasattr(record, "extra_info"):
            log_record.update(record.extra_info)
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

logger = logging.getLogger("telco_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONLogFormatter())
if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False


# --- 2. Inicialização da Aplicação e Modelo ---
app = FastAPI(title="Telco Churn Prediction API")

MODEL_PATH = Path("models/best_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo carregado com sucesso.", extra={"extra_info": {"model_path": str(MODEL_PATH)}})
except Exception as e:
    logger.error("Falha ao carregar o modelo.", exc_info=True, extra={"extra_info": {"model_path": str(MODEL_PATH)}})
    raise e

THRESHOLD = 0.4


class CustomerData(BaseModel):
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    tenure_months: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


# --- 3. Middleware de Latência ---
@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start_time = time.time()
    
    # Processa a requisição
    response = await call_next(request)
    
    # Calcula a latência
    process_time_ms = round((time.time() - start_time) * 1000, 2)
    
    # Log estruturado da requisição
    logger.info(
        "Requisição processada",
        extra={
            "extra_info": {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": process_time_ms,
                "client_ip": request.client.host if request.client else None,
            }
        }
    )
    
    # Opcional: Adiciona a latência no header da resposta para debug no client
    response.headers["X-Process-Time"] = str(process_time_ms)
    
    return response


# --- 4. Rotas ---
@app.get("/")
def root():
    return {"message": "API running successfully"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        row = pd.DataFrame(
            [
                {
                    "Gender": customer.gender,
                    "Senior Citizen": customer.senior_citizen,
                    "Partner": customer.partner,
                    "Dependents": customer.dependents,
                    "Tenure Months": customer.tenure_months,
                    "Phone Service": customer.phone_service,
                    "Multiple Lines": customer.multiple_lines,
                    "Internet Service": customer.internet_service,
                    "Online Security": customer.online_security,
                    "Online Backup": customer.online_backup,
                    "Device Protection": customer.device_protection,
                    "Tech Support": customer.tech_support,
                    "Streaming TV": customer.streaming_tv,
                    "Streaming Movies": customer.streaming_movies,
                    "Contract": customer.contract,
                    "Paperless Billing": customer.paperless_billing,
                    "Payment Method": customer.payment_method,
                    "Monthly Charges": customer.monthly_charges,
                    "Total Charges": customer.total_charges,
                }
            ]
        )

        prob = model.predict_proba(row)[0][1]
        pred = int(prob >= THRESHOLD)
        
        # Log do negócio (informações relevantes da predição)
        logger.info(
            "Predição realizada", 
            extra={
                "extra_info": {
                    "churn_probability": round(float(prob), 4),
                    "prediction": pred
                }
            }
        )

        return {"churn_probability": round(float(prob), 4), "prediction": pred, "threshold": THRESHOLD}

    except Exception as e:
        logger.error("Erro durante a predição", exc_info=True)
        raise e
