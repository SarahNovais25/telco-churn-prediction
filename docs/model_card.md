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

## Final Model Selection

After comparing all models using **5-fold stratified cross validation**, **Gradient Boosting** was selected as the final model.

It achieved the highest **ROC-AUC Mean (0.8617)** and the highest **Accuracy Mean (0.8093)**, demonstrating the best overall performance and stable results across folds.

---

## 🎚️ Threshold Strategy

Operational threshold may be adjusted depending on business goals.

Example:

- Lower threshold → Higher recall
- Higher recall increases the ability to detect churners, but may generate more false positives.
- Higher precision reduces the number of false positives, but may miss some actual churners.

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

## 🚨 Riscos e Cenários de Falha

Este modelo, como qualquer sistema preditivo, está sujeito a riscos e pode falhar em determinadas situações. É crucial entender esses cenários para uma utilização responsável e eficaz.

### Riscos Inerentes

*   **Falsos Positivos:** Clientes identificados como risco de churn que, na verdade, não iriam cancelar. Isso pode levar a custos desnecessários em campanhas de retenção ou a um contato inoportuno com o cliente.
*   **Falsos Negativos:** Clientes que realmente iriam cancelar, mas que o modelo não conseguiu identificar como risco. Isso resulta na perda de clientes valiosos e oportunidades de intervenção.
*   **Drift Temporal (Temporal Drift):** O comportamento do cliente e as condições de mercado podem mudar ao longo do tempo, fazendo com que as relações aprendidas pelo modelo se tornem desatualizadas e sua performance degrade.

### Cenários de Falha Específicos

*   **Mudanças Abruptas no Mercado:** Eventos externos não representados nos dados de treinamento (ex: entrada de um novo concorrente com ofertas agressivas, crises econômicas, novas regulamentações) podem alterar drasticamente o comportamento de churn, tornando as previsões do modelo menos precisas.
*   **Comportamento de Cliente Atípico:** O modelo pode ter dificuldade em prever o churn para segmentos de clientes com histórico limitado ou comportamentos muito distintos da maioria (ex: clientes recém-adquiridos, clientes com serviços muito específicos).
*   **Problemas na Qualidade dos Dados de Entrada:** Erros ou inconsistências nos dados fornecidos à API para inferência (ex: valores ausentes inesperados, formatos incorretos, dados desatualizados) podem levar a previsões errôneas ou à falha da API.
*   **Alterações na Definição de Churn:** Se a operadora mudar a forma como define ou registra o churn, o modelo pode precisar de retreinamento e revalidação para se adaptar à nova definição.
*   **Viés de Seleção:** Se as campanhas de retenção passadas influenciaram os dados de treinamento de forma não controlada, o modelo pode aprender a prever o churn apenas para clientes que não foram alvo dessas campanhas, falhando em identificar outros grupos de risco.

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