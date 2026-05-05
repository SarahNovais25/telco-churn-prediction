# Model Card — Telco Customer Churn Prediction

---

## 📌 Model Details

- **Project Name:** Telco Customer Churn Prediction
- **Model Version:** v1.0
- **Authors:** Sarah Novais, Caio Terencio, Henrique and Gustavo
- **Model Type:** Gradient Boosting Classifier
- **Primary Objective:** Predict customer churn probability.
- **Output:** Churn probability, binary prediction and decision threshold.
- **Threshold:** 0.4

---

## 🎯 Intended Use

This model was developed to support business teams in identifying customers with a higher probability of churn.

The model should be used as a **decision-support tool** for retention strategies, helping prioritize customers who may require proactive actions.

### Recommended Use Cases

- Retention campaigns
- Customer prioritization
- Churn risk analysis
- Commercial decision support
- Customer relationship management support

### Not Recommended For

- Fully automated decisions without human review
- Penalizing customers
- Sensitive individual assessments
- Use outside the telecom context without revalidation
- Final business decisions without operational analysis

---

## 📊 Training Data

Dataset used: **Telco Customer Churn Dataset**

### Main Data Categories

- Demographic data
- Contract type
- Subscribed services
- Tenure
- Monthly charges
- Payment method
- Churn history

### Target Variable

- `Churn Value`
  - `0` = Customer stayed
  - `1` = Customer churned

---

## ⚠️ Data Leakage Prevention

The following columns were removed because they contain future information and could artificially improve model performance:

- `Churn Label`
- `Churn Score`
- `Churn Reason`

The following identifier, geographic and non-operational columns were also removed to ensure that the model uses only features expected to be available during API inference:

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

## 🧪 Preprocessing

The final model artifact contains a complete machine learning pipeline, including preprocessing and the trained model.

### Numeric Features

- Missing values handled with median imputation
- Standard scaling applied

### Categorical Features

- Missing values handled with most frequent value imputation
- One Hot Encoding applied to transform categories into numerical features

---

## 🤖 Models Evaluated

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- MLPClassifier
- PyTorch MLP Neural Network

---

## 📈 Evaluation Methodology

Model evaluation was performed using **5-fold stratified cross validation**.

This approach reduces dependency on a single train/test split and helps evaluate whether model performance is stable across different data partitions.

Experiments were tracked using **MLflow**, including parameters, metrics and artifacts.

---

## 📊 Cross Validation Results

| Model | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean | ROC-AUC Std | PR-AUC Mean | PR-AUC Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8099 | 0.6697 | 0.5602 | 0.6100 | **0.8628** | 0.0056 | **0.6869** | 0.0116 |
| Random Forest | 0.7838 | 0.5717 | 0.7394 | **0.6448** | 0.8588 | 0.0053 | 0.6788 | 0.0103 |
| Logistic Regression | 0.7589 | 0.5298 | **0.8127** | 0.6414 | 0.8576 | 0.0089 | 0.6733 | 0.0089 |
| Decision Tree | 0.7537 | 0.5233 | 0.8117 | 0.6361 | 0.8449 | 0.0088 | 0.6194 | 0.0092 |
| MLPClassifier | 0.7583 | 0.5478 | 0.5131 | 0.5298 | 0.7937 | 0.0075 | 0.5332 | 0.0206 |

---

## 🏆 Final Model Selection

After comparing all models using stratified cross validation, **Gradient Boosting** was selected as the final model.

It achieved the highest **ROC-AUC Mean (0.8628)** and the highest **PR-AUC Mean (0.6869)**, demonstrating the best overall performance for ranking customers by churn risk.

Although Logistic Regression and Random Forest achieved higher recall, Gradient Boosting was selected because it showed stronger global performance and better balance across evaluation metrics.

---

## 🎚️ Threshold Strategy

The API uses an operational threshold of:

`0.4`

This threshold was selected to increase churn detection sensitivity.

### Threshold Behavior

- Lower threshold → higher recall, more customers flagged as churn risk
- Higher threshold → higher precision, fewer false positives

The threshold may be adjusted depending on business goals, campaign costs and the desired balance between false positives and false negatives.

---

## 📦 Model Artifact

The final trained model is saved as:

`models/best_model.pkl`

This artifact contains the complete machine learning pipeline, including preprocessing and the trained Gradient Boosting model.

The FastAPI application loads this file to generate churn predictions without retraining the model.

---

## 🔬 MLflow Tracking

MLflow is used to track:

- Model parameters
- Cross-validation metrics
- ROC-AUC
- PR-AUC
- Model artifacts
- Final selected model

The MLflow UI can be started with:

`python3 -m mlflow ui`

---

## ⚖️ Trade-offs

Higher recall increases the ability to detect customers likely to churn, but may also generate more false positives.

This can increase retention campaign costs, but may reduce the risk of losing valuable customers.

Gradient Boosting showed the best global performance, while Logistic Regression and Random Forest showed stronger recall.

The final model selection prioritizes overall ranking quality and model stability, while acknowledging that threshold tuning may be required depending on business strategy.

---

## ⚠️ Limitations

- Dataset is limited to a telecom context
- Model may not generalize to other industries without revalidation
- Historical data may contain operational or commercial bias
- Customer behavior may change over time
- Model performance may degrade due to data drift
- The model does not explain causality; it identifies statistical patterns

---

## 🧬 Bias and Fairness Considerations

Possible sources of bias:

- Demographic distribution in the original dataset
- Historical business strategies
- Uneven representation of customer groups
- Differences in product availability or service quality across customer segments

Continuous monitoring by customer segment is recommended to identify potential unfair performance differences.

---

## 🚨 Risks

- False positives: customers may be classified as churn risk unnecessarily
- False negatives: customers likely to churn may not be detected
- Data drift may reduce model performance over time
- Business changes may affect prediction quality
- Incorrect interpretation of predictions may lead to poor business decisions

---

## 🔁 Monitoring Plan

Recommended monthly monitoring:

- Accuracy
- Recall
- Precision
- F1-score
- ROC-AUC
- PR-AUC
- Real churn rate
- Feature drift
- Prediction distribution
- API latency
- API errors

---

## 🛠️ Maintenance

Recommended retraining:

- Monthly or quarterly
- When model performance decreases
- When significant business changes occur
- When data distribution changes
- When churn behavior changes over time

---

## 👩‍💻 Human Oversight

The model should support human decision-making and should not fully replace business or customer relationship teams.

Predictions should be used as decision support, especially for prioritizing retention actions.

Human review is recommended before taking customer-facing actions.

---

## 📌 Final Recommendation

Gradient Boosting presented the best overall predictive performance and stable cross-validation results.

It is recommended for assisted use in customer retention strategies, especially for prioritizing customers with higher churn probability.

The model should be monitored continuously and periodically retrained to maintain performance over time.

---

## 👩‍💻 Authors

Sarah Novais, Caio Terencio, Henrique and Gustavo
