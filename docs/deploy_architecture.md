# Arquitetura de Deploy - Modelo de Previsão de Churn

## 1. Visão Geral da Arquitetura

A arquitetura de deploy para o modelo de previsão de churn do cliente Telco é projetada para ser robusta, escalável e eficiente, dividida em duas fases principais:

1.  **Fase de Treinamento (Batch):** Onde o modelo é treinado e retreinado periodicamente usando dados históricos.
2.  **Fase de Inferência (Real-time via API):** Onde o modelo treinado é servido para realizar previsões em tempo real para novos clientes ou clientes existentes.

Esta abordagem garante que o modelo esteja sempre atualizado com os dados mais recentes e que as previsões estejam disponíveis sob demanda para as aplicações de negócio.

## 2. Fase de Treinamento (Batch)

### 2.1. Componentes e Fluxo

O processo de treinamento é orquestrado por um script principal (`train.py`) e gerenciado por um `Makefile` para simplificar a execução.

*   **Dados Históricos:** O processo inicia com a ingestão de dados históricos de clientes, que contêm informações demográficas, contratuais, serviços e o histórico de churn.
*   **`train.py` (Cérebro do Projeto):** Este script é o coração do processo de treinamento. Ele é responsável por:
    *   **Leitura e Preparação de Dados:** Carrega a base de dados, realiza a limpeza, pré-processamento e engenharia de features.
    *   **Experimentação e Comparação de Modelos:** Testa diversos modelos de Machine Learning (Regressão Logística, Árvore de Decisão, Random Forest, Rede Neural MLP, Gradient Boosting).
    *   **Validação Cruzada:** Utiliza a técnica de Cross Validation (5-fold estratificada) para garantir que o modelo funciona de forma confiável e que os resultados não são baseados em apenas um teste.
    *   **Seleção do Melhor Modelo:** Compara os resultados de todos os modelos testados com base em métricas como ROC-AUC, Accuracy, Precision, Recall e F1-Score.
    *   **Salvamento do Modelo:** Após a seleção, o melhor modelo é serializado e salvo como `best_model.pkl` no diretório `models/`.
*   **MLflow:** Todos os experimentos, incluindo parâmetros, métricas e artefatos (como o próprio modelo `best_model.pkl` e a versão do dataset), são rastreados e registrados no MLflow. Isso garante reprodutibilidade e auditabilidade de cada execução de treinamento.
*   **`Makefile`:** Simplifica a execução do processo de treinamento. O comando `make train` automatiza todas as etapas descritas acima, desde a leitura dos dados até o salvamento do modelo final.

### 2.2. Justificativa para Abordagem Batch

A abordagem de treinamento em batch é ideal para este caso de uso por várias razões:

*   **Frequência de Retreinamento:** A taxa de churn e o comportamento do cliente não mudam drasticamente em questão de horas. Um retreinamento mensal ou trimestral (conforme definido no plano de manutenção) é suficiente para manter o modelo atualizado.
*   **Custo-Benefício:** Treinar modelos complexos como o Gradient Boosting e Redes Neurais pode ser computacionalmente intensivo. Realizar isso em batch fora do caminho crítico de inferência otimiza o uso de recursos.
*   **Consistência:** Garante que o modelo utilizado para inferência seja uma versão estável e validada, com todos os experimentos rastreados.

## 3. Fase de Inferência (Real-time via API)

### 3.1. Componentes e Fluxo

A inferência do modelo é realizada através de uma API RESTful, construída com FastAPI, que permite que outras aplicações de negócio solicitem previsões em tempo real.

*   **API FastAPI:**
    *   **Carregamento do Modelo:** Ao ser iniciada, a API carrega o modelo treinado (`models/best_model.pkl`) na memória. Isso significa que o modelo está pronto para uso imediato, sem a necessidade de ser treinado a cada requisição.
    *   **Endpoint `/predict`:** Recebe requisições HTTP contendo os dados de um novo cliente (ou cliente existente) e retorna a probabilidade de churn. A validação dos dados de entrada é feita usando Pydantic para garantir a integridade.
    *   **Endpoint `/health`:** Fornece um mecanismo para verificar a saúde e disponibilidade da API.
    *   **Logging Estruturado:** Todas as requisições e respostas são logadas de forma estruturada, facilitando o monitoramento e a depuração.
    *   **Middleware de Latência:** Monitora o tempo de resposta das requisições, auxiliando na identificação de gargalos de performance.
*   **`Makefile`:** O comando `make run` é utilizado para iniciar a API, tornando o processo de deploy simples e padronizado.
*   **Consumidores da API:** Aplicações de negócio (ex: CRM, sistemas de campanha de marketing, dashboards de BI) podem integrar-se à API para obter previsões de churn e tomar decisões proativas.

### 3.2. Justificativa para Abordagem Real-time via API

A escolha de servir o modelo via API em tempo real é justificada por:

*   **Tomada de Decisão Proativa:** Permite que as equipes de negócio identifiquem clientes em risco de churn no momento em que interagem com a operadora ou quando seus dados são atualizados, possibilitando ações de retenção imediatas.
*   **Integração Flexível:** Uma API RESTful é um padrão de comunicação amplamente aceito, facilitando a integração com diversos sistemas internos e externos.
*   **Escalabilidade:** Frameworks como FastAPI são projetados para alta performance e podem ser facilmente escalados horizontalmente para lidar com um grande volume de requisições.
*   **Desacoplamento:** Separa a lógica de inferência do modelo das aplicações de negócio, permitindo que ambos evoluam independentemente.

## 4. Fluxo de Trabalho Simplificado

1.  **Treinamento (`make train`):**
    *   `Dados Históricos` → `train.py` (limpeza, preparação, experimentação, validação, seleção) → `best_model.pkl` (salvo no diretório `models/`)
2.  **Uso em Produção (`make run`):**
    *   `API FastAPI` (carrega `models/best_model.pkl`) → `Novo Cliente` (dados de entrada) → `API` → `Previsão de Churn`

## 5. (Opcional) Deploy em Nuvem

Para um deploy em nuvem (AWS, Azure, GCP), a API FastAPI seria empacotada (ex: via Docker) e implantada em um serviço de contêiner (ex: AWS ECS/EKS, Azure Container Instances/AKS, Google Cloud Run/GKE) ou em uma função serverless (ex: AWS Lambda com API Gateway, Azure Functions, Google Cloud Functions). O `best_model.pkl` seria armazenado em um bucket de armazenamento de objetos (ex: S3, Azure Blob Storage, GCS) e carregado pela API na inicialização. Isso garantiria alta disponibilidade, escalabilidade e gerenciamento de infraestrutura simplificado.

---