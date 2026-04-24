from src.data import load_data


def test_dataset_has_target():
    df = load_data("data/Telco_customer_churn.xlsx")
    assert "Churn Value" in df.columns