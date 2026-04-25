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

### Train models

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

| Model | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean | ROC-AUC Std |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8093 | 0.6805 | 0.5302 | 0.5960 | **0.8617** | 0.0071 |
| Decision Tree | 0.7572 | 0.5279 | **0.8138** | **0.6400** | 0.8478 | 0.0086 |
| Logistic Regression | 0.7693 | 0.5491 | 0.7309 | 0.6270 | 0.8392 | 0.0118 |
| Random Forest | 0.7402 | 0.5066 | 0.8197 | 0.6261 | 0.8375 | 0.0083 |
| MLPClassifier | 0.7126 | 0.4568 | 0.4360 | 0.4458 | 0.7081 | 0.0111 |

---

## 🏆 Final Model

### Gradient Boosting Classifier

Selected after comparing multiple algorithms using **5-fold stratified cross validation**.

Chosen due to:

- Highest ROC-AUC score
- Highest overall accuracy
- Stable performance across folds
- Strong generalization capability

---

## 📌 Final Model Metrics

| Metric | Value |
|---|---:|
| Accuracy Mean | 0.8093 |
| Precision Mean | 0.6805 |
| Recall Mean | 0.5302 |
| F1 Mean | 0.5960 |
| ROC-AUC Mean | 0.8617 |

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
- Model artifacts
- Experiment comparisons
- Training runs

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
- Threshold optimization

---

## 👩‍💻 Author
