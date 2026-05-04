from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_success():
    # Dados de exemplo baseados no README
    payload = {
        "gender": "Male",
        "senior_citizen": "0",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.5,
        "total_charges": 1050.0
}
    
    response = client.post("/predict", json=payload)
    
    # Validações básicas[cite: 1, 10]
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "threshold" in data
    assert isinstance(data["prediction"], int)
    assert 0 <= data["churn_probability"] <= 1
