from fastapi import FastAPI

app = FastAPI(title="Telco Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    return {
        "churn_probability": 0.75,
        "prediction": 1,
    }