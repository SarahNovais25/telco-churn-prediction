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
| Gradient Boosting | 0.8099 | 0.6697 | 0.5602 | 0.6100 | **0.8628** | 0.0056 |
| Random Forest | 0.7821 | 0.5694 | 0.7346 | **0.6415** | 0.8587 | 0.0052 |
| Logistic Regression | 0.7589 | 0.5298 | **0.8127** | 0.6414 | 0.8576 | 0.0089 |
| Decision Tree | 0.7537 | 0.5233 | 0.8117 | 0.6361 | 0.8449 | 0.0088 |
| MLPClassifier | 0.7583 | 0.5478 | 0.5131 | 0.5298 | 0.7937 | 0.0075 |
|

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
