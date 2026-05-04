# Model Card — Telco Customer Churn Prediction

---

## 📌 Model Details

- **Project Name:** Telco Customer Churn Prediction
- **Author:** Sarah Novais
- **Model Type:** Gradient Boosting Classifier
- **Primary Objective:** Predict customer churn probability.

---

## 🎯 Intended Use

This model was developed to support business teams in identifying customers at risk of churn.

### Recommended Use Cases

- Retention campaigns
- Customer prioritization
- Churn risk analysis
- Commercial decision support

### Not Recommended For

- Fully automated decisions without human review
- Penalizing customers
- Sensitive individual assessments
- Use outside the telecom context without revalidation

---

## 📊 Training Data

Dataset used: **Telco Customer Churn Dataset**

### Contains

- Demographic data
- Contract type
- Subscribed services
- Tenure
- Monthly charges
- Payment method
- Churn history

### Target

- `Churn Value`
  - `0` = Customer stayed
  - `1` = Customer churned

---

## ⚠️ Data Leakage Prevention

The following columns were removed because they contain future information:

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

This approach was used to reduce dependency on a single train/test split and to evaluate model stability across different data partitions.

Experiments were tracked using **MLflow**, including parameters, metrics and artifacts.

---

## 📈 Cross Validation Results

| Model | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean | ROC-AUC Std |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8099 | 0.6697 | 0.5602 | 0.6100 | **0.8628** | 0.0056 |
| Random Forest | 0.7821 | 0.5694 | 0.7346 | **0.6415** | 0.8587 | 0.0052 |
| Logistic Regression | 0.7589 | 0.5298 | **0.8127** | 0.6414 | 0.8576 | 0.0089 |
| Decision Tree | 0.7537 | 0.5233 | 0.8117 | 0.6361 | 0.8449 | 0.0088 |
| MLPClassifier | 0.7583 | 0.5478 | 0.5131 | 0.5298 | 0.7937 | 0.0075 |

---

## 🏆 Final Model Selection

After comparing all models using stratified cross validation, **Gradient Boosting** was selected as the final model.

It achieved the highest **ROC-AUC Mean (0.8628)** and the highest **Accuracy Mean (0.8099)**, demonstrating the best overall performance and stable results across folds.

Although Logistic Regression and Random Forest achieved higher recall, Gradient Boosting was selected because it showed stronger global performance for ranking and separating churn risk.

---

## 🎚️ Threshold Strategy

The API uses an operational threshold of:

`0.4`

This threshold was selected to increase churn detection sensitivity.

Threshold behavior:

- Lower threshold → higher recall
- Higher threshold → higher precision

The threshold may be adjusted depending on business goals and campaign costs.

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

---

## ⚠️ Limitations

- Dataset is limited to a telecom context
- Model may not generalize to other industries without revalidation
- Historical data may contain operational or commercial bias
- Customer behavior may change over time
- Model should be monitored after deployment

---

## 🧬 Bias and Fairness Considerations

Possible sources of bias:

- Demographic distribution in the original dataset
- Historical business strategies
- Uneven representation of customer groups

Continuous monitoring by customer segment is recommended.

---

## 🚨 Risks

- False positives: customers may be classified as churn risk unnecessarily
- False negatives: customers likely to churn may not be detected
- Data drift may reduce model performance over time
- Business changes may affect prediction quality

---

## 🔁 Monitoring Plan

Recommended monthly monitoring:

- Accuracy
- Recall
- Precision
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

---

## 👩‍💻 Human Oversight

The model should support human decision-making and should not fully replace business or customer relationship teams.

Predictions should be used as decision support, especially for prioritizing retention actions.

---

## 📌 Final Recommendation

Gradient Boosting presented the best overall predictive performance and stable cross-validation results.

It is recommended for assisted use in customer retention strategies.

---
