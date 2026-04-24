import pandas as pd


def load_data(path: str):
    df = pd.read_excel(path)
    return df


def preprocess_data(df: pd.DataFrame):
    cols_to_drop = [
        'CustomerID',
        'Lat Long',
        'Churn Label',
        'Churn Score',
        'Churn Reason'
    ]

    df = df.drop(columns=cols_to_drop)

    y = df['Churn Value']
    X = df.drop(columns=['Churn Value'])

    X = pd.get_dummies(X)

    return X, y