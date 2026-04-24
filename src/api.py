from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Telco Churn Prediction API")

MODEL_PATH = Path("models/random_forest_churn.joblib")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
columns = artifact["columns"]


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


@app.get("/")
def root():
    return {"message": "API running successfully"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    row = pd.DataFrame([{
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
        "Total Charges": customer.total_charges
    }])

    row = pd.get_dummies(row)
    row = row.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(row)[0][1]
    pred = int(prob >= 0.4)

    return {
        "churn_probability": round(float(prob), 4),
        "prediction": pred,
        "threshold": 0.4
    }