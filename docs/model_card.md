# Model Card — Telco Customer Churn Prediction

---

## 📌 Model Details

- **Project Name:** Telco Customer Churn Prediction
- **Author:** 
- **Model Type:** Gradient Boosting Classifier
- **Primary Objective:** Predict customer churn probability.

### Alternative Models Tested

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- MLPClassifier
- PyTorch MLP Neural Network

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

---

## 📈 Performance

## Cross Validation Results

| Model | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean | ROC-AUC Std |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8099 | 0.6697 | 0.5602 | 0.6100 | **0.8628** | 0.0056 |
| Random Forest | 0.7821 | 0.5694 | 0.7346 | **0.6415** | 0.8587 | 0.0052 |
| Logistic Regression | 0.7589 | 0.5298 | **0.8127** | 0.6414 | 0.8576 | 0.0089 |
| Decision Tree | 0.7537 | 0.5233 | 0.8117 | 0.6361 | 0.8449 | 0.0088 |
| MLPClassifier | 0.7583 | 0.5478 | 0.5131 | 0.5298 | 0.7937 | 0.0075 |
|

## Final Model Selection

After comparing all models using **5-fold stratified cross validation**, **Gradient Boosting** was selected as the final model.

It achieved the highest **ROC-AUC Mean (0.8617)** and the highest **Accuracy Mean (0.8093)**, demonstrating the best overall performance and stable results across folds.

---

## 🎚️ Threshold Strategy

Operational threshold may be adjusted depending on business goals.

Example:

- Lower threshold → Higher recall
- Higher threshold → Higher precision

The default classification threshold is `0.50`.

---

## 🧠 Trade-offs

Higher recall increases the ability to detect churners, but may generate more false positives.

This can increase campaign costs, but reduces the risk of losing valuable customers.

---

## ⚠️ Limitations

- Dataset limited to one telecom company
- May not generalize to other industries
- Economic changes may impact future performance
- Historical data may contain operational bias

---

## 🧬 Bias and Fairness Considerations

Possible sources of bias:

- Original dataset demographic profile
- Past business strategies
- Uneven group distribution

Continuous monitoring by segment is recommended.

---

## 🚨 Risks

- False positives: customers identified as churn risk unnecessarily
- False negatives: churners not detected
- Temporal drift in customer behavior

---

## 🔁 Monitoring Plan

Recommended monthly monitoring:

- Accuracy
- Recall
- Precision
- Real churn rate
- Feature drift
- Probability distribution

---

## 🛠️ Maintenance

Recommended retraining:

- Monthly or quarterly
- When performance declines
- When significant business changes occur

---

## 👩‍💻 Human Oversight

The model should support human decisions and not fully replace them.

---

## 📌 Final Recommendation

Gradient Boosting presented the best overall predictive performance and stable validation results.

It is recommended for assisted use in customer retention strategies.

---

## 👩‍💻 Author

