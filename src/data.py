import pandas as pd


def load_data(path: str):
    df = pd.read_excel(path)
    return df


def preprocess_data(df: pd.DataFrame):
    # 1. Conversão de Colunas Numéricas
    # 'Tenure Months' é o nome da coluna no dataset para tenure
    num_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
    
    for col in num_cols:
        # errors='coerce' transforma valores inválidos (como espaços em branco) em NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher possíveis NaNs gerados na conversão (comum em Total Charges para novos clientes)
    df["Total Charges"] = df["Total Charges"].fillna(0)

    # 2. Tratamento da coluna Senior Citizen
    # No dataset original, ela vem como "No" ou "Yes"
    df["Senior Citizen"] = df["Senior Citizen"].map({"Yes": 1, "No": 0})

    # 3. Limpeza e Drop de Colunas
    cols_to_drop = ["CustomerID", "Lat Long", "Churn Label", "Churn Score", "Churn Reason"]
    df = df.drop(columns=cols_to_drop)

    # 4. Separação de features e target
    y = df["Churn Value"]
    X = df.drop(columns=["Churn Value"])

    # 5. Encoding de variáveis categóricas
    X = pd.get_dummies(X)

    return X, y