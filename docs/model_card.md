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
| Gradient Boosting | 0.8093 | 0.6805 | 0.5302 | 0.5960 | **0.8617** | 0.0071 |
| Decision Tree | 0.7572 | 0.5279 | **0.8138** | **0.6400** | 0.8478 | 0.0086 |
| Logistic Regression | 0.7693 | 0.5491 | 0.7309 | 0.6270 | 0.8392 | 0.0118 |
| Random Forest | 0.7402 | 0.5066 | 0.8197 | 0.6261 | 0.8375 | 0.0083 |
| MLPClassifier | 0.7126 | 0.4568 | 0.4360 | 0.4458 | 0.7081 | 0.0111 |

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

