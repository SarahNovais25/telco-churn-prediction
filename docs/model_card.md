# Model Card — Telco Customer Churn Prediction

---

## 📌 Model Details

- **Project Name:** Telco Customer Churn Prediction
- **Author:** Sarah Novais
- **Model Type:** Random Forest Classifier
- **Alternative Models Tested:**
  - Dummy Classifier
  - Logistic Regression
  - Decision Tree
  - PyTorch MLP Neural Network

- **Primary Objective:** Predict customer churn probability.

---

## 🎯 Intended Use

Este modelo foi desenvolvido para apoiar equipes de negócio na identificação de clientes com risco de cancelamento.

### Casos de uso recomendados:

- Campanhas de retenção
- Priorização de atendimento
- Análise de risco de churn
- Apoio à tomada de decisão comercial

### Não recomendado para:

- Decisões automáticas sem revisão humana
- Penalização de clientes
- Avaliação individual sensível
- Uso fora do contexto de telecom sem revalidação

---

## 📊 Training Data

Dataset utilizado:

**Telco Customer Churn Dataset**

Contém:

- Dados demográficos
- Tipo de contrato
- Serviços contratados
- Tempo de permanência
- Cobrança mensal
- Histórico de churn

### Target:

- `Churn Value`
  - 0 = cliente permaneceu
  - 1 = cliente saiu

---

## ⚠️ Data Leakage Prevention

As colunas abaixo foram removidas por conter informação futura:

- Churn Label
- Churn Score
- Churn Reason

---

## 📈 Performance

## Modelo Final: Random Forest

| Metric | Value |
|------|------|
| Accuracy | 0.79 |
| Precision | 0.62 |
| Recall | 0.61 |
| F1-score | 0.61 |
| ROC-AUC | 0.83 |

### Threshold Ajustado

Threshold operacional utilizado:

```text
0.4
Objetivo: aumentar recall de churn.

---

## 🧠 Alternative Model Results

| Model | ROC-AUC |
|------|--------:|
| Logistic Regression | 0.77 |
| Decision Tree | 0.83 |
| Random Forest | 0.82 |
| PyTorch MLP | 0.77 |

---

## ⚖️ Trade-offs

Maior recall aumenta a capacidade de detectar churn, porém gera mais falsos positivos.

Isso pode aumentar o custo de campanhas, porém reduz o risco de perder clientes valiosos.

---

## ⚠️ Limitations

- Dataset limitado a uma empresa de telecom  
- Pode não generalizar para outros mercados  
- Mudanças econômicas podem afetar performance futura  
- Dados históricos podem conter vieses operacionais  

---

## 🧬 Bias and Fairness Considerations

Possíveis fontes de viés:

- Perfil demográfico da base original  
- Estratégias comerciais passadas  
- Distribuição desigual entre grupos  

Recomenda-se monitoramento contínuo por segmento.

---

## 🚨 Risks

- Falsos positivos: clientes tratados como churn sem necessidade  
- Falsos negativos: clientes churn não detectados  
- Drift temporal do comportamento dos clientes  

---

## 🔁 Monitoring Plan

Recomenda-se monitorar mensalmente:

- Accuracy  
- Recall  
- Precision  
- Taxa de churn real  
- Drift de features  
- Distribuição das probabilidades  

---

## 🛠️ Maintenance

Recomendado retreinamento:

- Mensal ou trimestral  
- Quando performance cair  
- Quando houver mudança relevante de negócio  

---

## 👩‍💻 Human Oversight

O modelo deve apoiar decisões humanas e não substituí-las integralmente.

---

## 📌 Final Recommendation

O modelo Random Forest apresentou melhor equilíbrio entre performance e interpretabilidade.

É recomendado para uso assistido em estratégias de retenção.

---
Objetivo: aumentar recall de churn.

---

## 🧠 Alternative Model Results

| Model | ROC-AUC |
|------|--------:|
| Logistic Regression | 0.77 |
| Decision Tree | 0.83 |
| Random Forest | 0.82 |
| PyTorch MLP | 0.77 |

---

## ⚖️ Trade-offs

Maior recall aumenta a capacidade de detectar churn, porém gera mais falsos positivos.

Isso pode aumentar o custo de campanhas, porém reduz o risco de perder clientes valiosos.

---

## ⚠️ Limitations

- Dataset limitado a uma empresa de telecom  
- Pode não generalizar para outros mercados  
- Mudanças econômicas podem afetar performance futura  
- Dados históricos podem conter vieses operacionais  

---

## 🧬 Bias and Fairness Considerations

Possíveis fontes de viés:

- Perfil demográfico da base original  
- Estratégias comerciais passadas  
- Distribuição desigual entre grupos  

Recomenda-se monitoramento contínuo por segmento.

---

## 🚨 Risks

- Falsos positivos: clientes tratados como churn sem necessidade  
- Falsos negativos: clientes churn não detectados  
- Drift temporal do comportamento dos clientes  

---

## 🔁 Monitoring Plan

Recomenda-se monitorar mensalmente:

- Accuracy  
- Recall  
- Precision  
- Taxa de churn real  
- Drift de features  
- Distribuição das probabilidades  

---

## 🛠️ Maintenance

Recomendado retreinamento:

- Mensal ou trimestral  
- Quando performance cair  
- Quando houver mudança relevante de negócio  

---

## 👩‍💻 Human Oversight

O modelo deve apoiar decisões humanas e não substituí-las integralmente.

---

## 📌 Final Recommendation

O modelo Random Forest apresentou melhor equilíbrio entre performance e interpretabilidade.

É recomendado para uso assistido em estratégias de retenção.

---

## 👩‍💻 Author

