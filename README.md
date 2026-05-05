# Telco Customer Churn Prediction

Machine Learning project focused on predicting customer churn using classification models, experiment tracking, API deployment and engineering best practices.

---

## 📌 Project Objective

The goal of this project is to identify customers with a high probability of churn so that business teams can proactively execute retention strategies.

This repository was developed as part of the FIAP Machine Learning Engineer Tech Challenge.

---

## 🧠 Problem Statement

Customer churn directly impacts revenue and growth.

By predicting churn in advance, companies can:

- Reduce customer loss
- Improve retention campaigns
- Prioritize high-risk customers
- Increase lifetime value

---

## 📊 Dataset

**Telco Customer Churn Dataset**

Contains information such as:

- Demographics
- Contract type
- Services subscribed
- Monthly charges
- Tenure
- Payment method
- Churn label

### Target Variable

- `Churn Value`
  - `0` = Stayed
  - `1` = Churned

---

## ⚠️ Data Leakage Prevention

The following columns were removed because they contain future information:

- `Churn Label`
- `Churn Score`
- `Churn Reason`

The following identifier, geographic and non-operational columns were also removed to keep the model aligned with API inference:

- `CustomerID`
- `Count`
- `Country`
- `State`
- `City`
- `Zip Code`
- `Lat Long`
- `Latitude`
- `Longitude`
- `CLTV`

---

## 📁 Project Structure
```
telco-churn-prediction/
│
├── data/
│
├── docs/
│ └── model_card.md
│
├── models/
│ ├── best_model.pkl
│ └── model_comparison_cv.csv
│
├── notebooks/
│
├── src/
│ ├── api.py
│ ├── data.py
│ ├── evaluate.py
│ ├── train.py
│ ├── train_mlp.py
│ └── mlflow_tracking.py
│
├── tests/
│ ├── test_api.py
│ ├── test_schema.py
│ └── test_smoke.py
│
├── pyproject.toml
├── Makefile
├── README.md
└── .gitignore
```
---

## ⚙️ Setup

Clone repository:

git clone https://github.com/SarahNovais25/telco-churn-prediction.git  
cd telco-churn-prediction

Create virtual environment:

python -m venv .venv  

Linux/Mac: 
source .venv/bin/activate

Windows: 
.venv\Scripts\activate

Install dependencies:

make install

---

## 🚀 Available Commands

Install project dependencies:

make install

Run lint validation:

make lint

Run automated tests:

make test

Train and compare models:

make train

Run FastAPI application:

make run

Run MLflow UI:

make mlflow

---

## 🤖 Models Evaluated

### Baseline Models

- Dummy Classifier
- Logistic Regression

### Tree-Based Models

- Decision Tree
- Random Forest
- Gradient Boosting

### Neural Network

- PyTorch MLP
- MLPClassifier

---

## 📈 Cross Validation Results

The models were evaluated using 5-fold stratified cross validation.

| Model                | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean | ROC-AUC Std | PR-AUC Mean | PR-AUC Std |
|----------------------|--------------:|---------------:|------------:|--------:|-------------:|------------:|------------:|-----------:|
| Gradient Boosting    | 0.8099        | 0.6697         | 0.5602      | 0.6100  | 0.8628       | 0.0056      | 0.6869      | 0.0116     |
| Random Forest        | 0.7838        | 0.5717         | 0.7394      | 0.6448  | 0.8588       | 0.0053      | 0.6788      | 0.0103     |
| Logistic Regression  | 0.7589        | 0.5298         | 0.8127      | 0.6414  | 0.8576       | 0.0089      | 0.6733      | 0.0089     |
| Decision Tree        | 0.7537        | 0.5233         | 0.8117      | 0.6361  | 0.8449       | 0.0088      | 0.6194      | 0.0092     |
| MLPClassifier        | 0.7583        | 0.5478         | 0.5131      | 0.5298  | 0.7937       | 0.0075      | 0.5332      | 0.0206     |
---

## 🏆 Final Model

### Gradient Boosting Classifier

The final model selected was **Gradient Boosting Classifier**.

It was selected because it achieved the highest **ROC-AUC Mean (0.8628)** and the highest **Accuracy Mean (0.8099)** during cross validation.

Although Logistic Regression and Random Forest achieved higher recall, Gradient Boosting showed the best overall performance and stable results across folds.

---

## 📦 Model Artifact

After training, the final model is saved as:

models/best_model.pkl

This file contains the trained pipeline, including preprocessing and the final model.

The API loads this file to make predictions without retraining the model every time.

Simple flow:

Historical data → train.py → best_model.pkl → FastAPI → churn prediction

---

## 🌐 API Inference

Run API:

make run

Swagger Docs:

http://127.0.0.1:8000/docs

### Available Endpoints

- `GET /`
- `GET /health`
- `POST /predict`

Example request:

```json
{
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
```

Example response:

```json
{
  "churn_probability": 0.7577,
  "prediction": 1,
  "threshold": 0.4
}
```

---

## 🔬 MLflow Experiment Tracking

This project uses MLflow to track machine learning experiments.

MLflow records:

- Model name
- Model parameters
- Cross-validation metrics
- ROC-AUC
- PR-AUC
- Model artifacts
- Final selected model

Run MLflow UI:

make mlflow

Or:

python -m mlflow ui

Open in browser:

http://127.0.0.1:5000

---

## 🧪 Automated Tests

Implemented tests:

- Smoke test
- Schema validation
- API endpoint validation

Run:

make test

---

## 🧹 Code Quality

Linting with Ruff:

make lint

---

## 📄 Model Governance

See detailed documentation:

docs/model_card.md

Includes:

- Intended use
- Risks
- Bias considerations
- Monitoring plan
- Limitations
- Human oversight

---

## 🔁 Monitoring Recommendations

Monitor monthly:

- Accuracy
- Recall
- Precision
- ROC-AUC
- PR-AUC
- Feature drift
- Prediction distribution
- Business churn rate
- API latency
- API errors

---

## 🛠️ Future Improvements

- Hyperparameter tuning
- Explainability with SHAP
- Docker deployment
- CI/CD pipeline
- Cloud deployment
- Real-time inference
- Threshold optimization
- Structured logging
- Latency middleware
- Production monitoring

---
