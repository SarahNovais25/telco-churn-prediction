# Plano de Monitoramento - Modelo de Previsão de Churn

## 1. Introdução

Este documento detalha o plano de monitoramento para o modelo de previsão de churn de clientes Telco. O monitoramento contínuo é essencial para garantir que o modelo mantenha sua performance ao longo do tempo, detecte desvios nos dados ou no comportamento do cliente, e forneça insights acionáveis para a equipe de negócio.

## 2. Objetivos do Monitoramento

*   **Garantir a Performance do Modelo:** Assegurar que o modelo continue a entregar previsões precisas e úteis para o negócio.
*   **Detectar Drift:** Identificar mudanças na distribuição dos dados de entrada (Data Drift) ou na relação entre as features e o target (Concept Drift).
*   **Identificar Vieses:** Monitorar a performance do modelo em diferentes segmentos de clientes para detectar e mitigar vieses.
*   **Fornecer Alertas Proativos:** Notificar as equipes relevantes sobre problemas potenciais antes que impactem significativamente o negócio.
*   **Informar Retreinamentos:** Fornecer dados para decidir quando e como o modelo deve ser retreinado.

## 3. Métricas de Monitoramento

O monitoramento será dividido em três categorias principais: **Métricas de Performance do Modelo**, **Métricas de Qualidade de Dados/Features** e **Métricas de Negócio**.

### 3.1. Métricas de Performance do Modelo

Estas métricas serão calculadas periodicamente (ex: mensalmente) em um conjunto de dados de validação recente, que reflita o comportamento atual dos clientes.

*   **Acurácia (Accuracy):** Proporção de previsões corretas (churners e não-churners).
*   **Precisão (Precision):** Dos clientes que o modelo previu como churners, quantos realmente churnaram.
*   **Recall:** Dos clientes que realmente churnaram, quantos o modelo conseguiu identificar.
*   **F1-Score:** Média harmônica entre Precisão e Recall.
*   **ROC-AUC:** Capacidade do modelo de distinguir entre classes positivas e negativas.
*   **Curva PR-AUC:** Avaliação da performance do modelo focada na classe minoritária (churners), especialmente útil em datasets desbalanceados.
*   **Distribuição de Probabilidades:** Monitorar a distribuição das probabilidades de churn previstas pelo modelo. Mudanças significativas podem indicar drift.

### 3.2. Métricas de Qualidade de Dados e Features (Data Drift)

Estas métricas visam identificar mudanças nas características dos dados de entrada que podem impactar a performance do modelo.

*   **Drift de Features:**
    *   **Distribuição das Features:** Comparar a distribuição das features de entrada (ex: `Tenure`, `MonthlyCharges`, `ContractType`) entre o período de treinamento e o período de inferência. Ferramentas como o KS-Test (Kolmogorov-Smirnov) ou PSI (Population Stability Index) podem ser usadas.
    *   **Valores Ausentes:** Monitorar a proporção de valores ausentes em cada feature. Aumentos inesperados podem indicar problemas na coleta de dados.
    *   **Valores Atípicos (Outliers):** Monitorar a frequência e a magnitude de outliers nas features numéricas.
*   **Integridade dos Dados:**
    *   **Schema Drift:** Verificar se o esquema dos dados de entrada (tipos de dados, nomes de colunas) permanece consistente com o esperado pelo modelo.

### 3.3. Métricas de Negócio

Estas métricas conectam a performance do modelo diretamente aos resultados de negócio.

*   **Taxa de Churn Real vs. Prevista:** Comparar a taxa de churn real observada com a taxa de churn prevista pelo modelo para o mesmo período.
*   **Efetividade das Campanhas de Retenção:** Medir o sucesso das campanhas de retenção que utilizaram as previsões do modelo (ex: redução da taxa de churn no grupo de clientes alvo, ROI das campanhas).
*   **Custo de Churn Evitado:** Estimar o valor financeiro economizado pela empresa ao reter clientes identificados pelo modelo.

## 4. Frequência de Monitoramento

O monitoramento será realizado com a seguinte frequência:

*   **Diário:**
    *   Disponibilidade da API (`/health` endpoint).
    *   Latência da API.
    *   Volume de requisições da API.
    *   Erros da API.
*   **Semanal:**
    *   Drift de Features (distribuição, valores ausentes).
    *   Distribuição das probabilidades de churn.
*   **Mensal:**
    *   Todas as Métricas de Performance do Modelo (Acurácia, Precision, Recall, F1, ROC-AUC, PR-AUC).
    *   Métricas de Negócio (Taxa de Churn Real vs. Prevista, Efetividade das Campanhas).
    *   Análise de Vieses por segmento.

## 5. Alertas e Limiares

Alertas serão configurados para notificar as equipes de MLOps e Negócio quando as métricas monitoradas excederem limites pré-definidos.

*   **Performance do Modelo:**
    *   Queda de **5%** na Acurácia ou ROC-AUC em relação ao baseline de produção.
    *   Queda de **10%** no Recall ou F1-Score para a classe de churn.
*   **Qualidade de Dados:**
    *   Aumento de **15%** na proporção de valores ausentes em qualquer feature crítica.
    *   PSI (Population Stability Index) > **0.2** para qualquer feature importante.
*   **API:**
    *   Latência média > **500ms** por mais de 15 minutos.
    *   Taxa de erro > **1%** por mais de 15 minutos.

## 6. Playbook de Resposta a Alertas

Quando um alerta é disparado, o seguinte playbook será seguido:

1.  **Notificação:** O alerta será enviado para o canal de comunicação da equipe de MLOps (ex: Slack, e-mail) e para os stakeholders de negócio relevantes.
2.  **Investigação Inicial (Equipe MLOps):**
    *   Verificar logs da API para identificar erros ou padrões.
    *   Analisar os dados de entrada recentes para identificar a causa do drift (seja de dados ou de conceito).
    *   Revisar o desempenho do modelo em subgrupos para identificar vieses.
3.  **Análise de Causa Raiz:**
    *   **Se for Data Drift:** Identificar a fonte da mudança nos dados (ex: mudança no sistema de coleta, nova promoção, erro de ETL).
    *   **Se for Concept Drift:** Avaliar se o comportamento do cliente mudou fundamentalmente (ex: entrada de um novo concorrente, mudança econômica).
    *   **Se for Problema de Infraestrutura/API:** Investigar problemas de rede, recursos, ou bugs no código da API.
4.  **Ações Corretivas:**
    *   **Retreinamento do Modelo:** Se o drift for significativo e o modelo estiver degradado, um retreinamento com dados mais recentes será agendado.
    *   **Ajuste de Limiares:** Se a estratégia de negócio mudar, os limiares de classificação podem ser ajustados.
    *   **Rollback:** Em casos de falha crítica, o modelo pode ser revertido para uma versão anterior estável.
    *   **Correção de Dados:** Se a causa for um problema na coleta de dados, a equipe de engenharia de dados será acionada.
    *   **Comunicação:** Manter as equipes de negócio informadas sobre o status e as ações tomadas.

## 7. Ferramentas de Monitoramento (Exemplos)

*   **MLflow:** Para rastrear e comparar a performance de diferentes versões do modelo ao longo do tempo.
*   **Prometheus/Grafana:** Para monitoramento de infraestrutura da API (latência, uso de CPU/memória, taxa de erros) e visualização de métricas de modelo.
*   **Evidently AI / NannyML:** Ferramentas dedicadas para detecção de Data Drift e Concept Drift.
*   **Dashboards Customizados:** Para visualização das métricas de negócio e performance do modelo, acessíveis pelas equipes de negócio.

---