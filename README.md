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

---

## 📁 Project Structure

    telco-churn-prediction/
    │
    ├── data/
    ├── docs/
    │   └── model_card.md
    ├── models/
    ├── notebooks/
    ├── src/
    │   ├── api.py
    │   ├── data.py
    │   ├── evaluate.py
    │   ├── train.py
    │   ├── train_mlp.py
    │   └── mlflow_tracking.py
    ├── tests/
    │   ├── test_api.py
    │   ├── test_schema.py
    │   └── test_smoke.py
    ├── pyproject.toml
    ├── Makefile
    ├── README.md
    └── .gitignore

---

## ⚙️ Setup

### Clone repository

    git clone https://github.com/SarahNovais25/telco-churn-prediction.git
    cd telco-churn-prediction

### Create virtual environment

    python3 -m venv .venv
    source .venv/bin/activate

### Install dependencies

    make install

---

## 🚀 Available Commands

### Install project dependencies

    make install

### Run lint validation

    make lint

### Run automated tests

    make test

### Train Random Forest model

    make train

### Run FastAPI application

    make run

---

## 🧠 Neural Network Training (PyTorch MLP)

Run neural network experiment:

    python3 -m src.train_mlp

---

## 🔬 MLflow Experiment Tracking

Run MLflow UI:

    python3 -m mlflow ui

Open in browser:

    http://127.0.0.1:5000

---

## 🤖 Models Evaluated

### Baselines

- Dummy Classifier
- Logistic Regression

### Tree Models

- Decision Tree
- Random Forest

### Neural Network

- PyTorch MLP

---

## 🏆 Final Model

### Random Forest Classifier

Chosen due to strong balance between:

- Predictive performance
- Interpretability
- Fast inference
- Easy deployment

---

## 📈 Final Metrics

| Metric | Value |
|--------|------:|
| Accuracy | 0.79 |
| Precision | 0.62 |
| Recall | 0.61 |
| F1-score | 0.61 |
| ROC-AUC | 0.83 |

### Threshold Used

    0.4

Optimized to improve churn recall.

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

## 🌐 API Inference

Run API:

    make run

Swagger Docs:

    http://127.0.0.1:8000/docs

### Available Endpoints

- `GET /`
- `GET /health`
- `POST /predict`

---

## 🔬 MLflow Tracking

Tracks:

- Parameters
- Metrics
- Loss history
- Model artifacts
- Experiment comparisons

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

---

## 🔁 Monitoring Recommendations

Monitor monthly:

- Accuracy
- Recall
- Precision
- Feature drift
- Prediction distribution
- Business churn rate

---

## 🛠️ Future Improvements

- Hyperparameter tuning
- Explainability with SHAP
- Docker deployment
- CI/CD pipeline
- Cloud deployment
- Real-time inference

---

## 👩‍💻 Author
