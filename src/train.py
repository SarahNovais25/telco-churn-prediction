from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.data import load_data, preprocess_data


def train_model():
    df = load_data("data/Telco_customer_churn.xlsx")
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_model()
    print("Treino concluído")