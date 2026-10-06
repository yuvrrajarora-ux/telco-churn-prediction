import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import clean, split_xy
from src.features import add_features, build_pipeline
from src.generate_data import generate


@pytest.fixture(scope="module")
def data():
    return clean(generate(600, seed=1))


def test_generated_data_schema(data):
    assert "Churn" in data.columns and "customerID" not in data.columns
    assert set(data["Churn"].unique()) <= {0, 1}
    assert data["TotalCharges"].isna().sum() > 0  # blanks became NaN


def test_add_features(data):
    out = add_features(data.drop(columns=["Churn"]))
    for col in ["avg_monthly_spend", "charge_ratio", "is_new_customer", "n_addons"]:
        assert col in out.columns
    assert out["n_addons"].between(0, 6).all()


def test_pipeline_fits_and_predicts(data):
    X, y = split_xy(data)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_pipeline_handles_unseen_category(data):
    X, y = split_xy(data)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"
    assert pipe.predict_proba(row).shape == (1, 2)
