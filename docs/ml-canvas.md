# ML Canvas – Previsão de Churn (Telco Customer Churn – IBM)
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa de um modelo preditivo de churn que classifique clientes com risco de cancelamento. Assim, o grupo deve construir o projeto do zero ao modelo servido via API, aplicando todas as boas práticas de engenharia de ML aprendidas na Fase 1.

## Problema de Negócio

- Reduzir 20% a taxa de churn trimestral, identificando clientes com risco de cancelamento com antecedencia para ações de retenção.

## Métricas de Negócio (KPI)

KPI de Negócio: Redução da Taxa de Churn (Percentual de Cancelamento)

KPI Secundario: Taxa de retenção após medidas do suporte.

# Levantamento de Requisitos de Restrições

Requisitos do projeto: 
- Modelo pronto em até 4 meses, para temporada de nova campanha.
- Desempenho de 90% de precisão minima.
- Atender a LGPD.

Restrições: 

 - Dados históricos de um contexto específico (não necessariamente generalizados para outras operadoras/países).
 - Possível análise de amostragem (segmentos de clientes sub-representados).
 - Não inclui variáveis ​​externas (concorrência, macroeconomia, campanhas passadas).


## Stakeholders
Gerente de Negócio: Gerencia as previsões dos clientes (Churn).

Equipe de Marketing: Com previsões definidas gera plano de retenção ( campanha e oferta).

 Time de Suporte Técnico / Atendimento: Podem fornecer conhecimento de domínio (especialistas) sobre os motivos mais frequentes de insatisfação dos clientes.

 Equipe de TI: Administradores das bases de dados, essenciais para viabilizar a extração segura das informações históricas e para apoiar o deploy do modelo via API nos sistemas da empresa.

## SLOs (Service Level Objectives)
Recall (Sensibilidade): O modelo deve identificar pelo menos 80% dos clientes que realmente iriam cancelar (True Positives) nos próximos 30 dias.

Precisão (Precision): Manter uma precisão de pelo menos 40% nas predições de churn.

Estabilidade da Performance: "A variação do F1-Score não deve cair mais que 5% em uma janela de 7 dias."

Latência de Inferência: "O modelo deve ser capaz de processar a lista de clientes e retornar as pontuações de risco em até 2 horas após a extração dos dados diários."

## Dados e Aprendizado (Data & Learning)
Fontes de Dados: Histórico de pagamentos, logs de cliques, tempo de tela por perfil e lista de favoritos.

Tarefa de ML: Classificação Binária (O usuário vai cancelar? Sim ou Não).

Frequência de Treino: Retreinar o modelo mensalmente para captar novas tendências dos clientes.

## Avaliação e Deployment (Evaluation & Deployment)

valiação Técnica: Comparação da MLP com os baselines do Scikit-Learn no MLflow. Em problemas de churn, dá-se atenção especial à métrica de Recall (sensibilidade), pois o custo de não identificar um cliente que vai cancelar (falso negativo) costuma ser maior do que focar em um cliente que não iria cancelar (falso positivo).

Avaliação de Negócio: Após a integração da API e início das ações de retenção, a eficácia do projeto será avaliada pelo monitoramento do KPI principal (queda na taxa de cancelamento trimestral) e pela análise de custo-benefício (o valor salvo com a retenção superou os custos de desenvolvimento e das ofertas concedidas).

