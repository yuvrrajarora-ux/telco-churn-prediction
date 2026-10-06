"""Feature engineering + preprocessing, packaged in one sklearn Pipeline so the
exact same transformations run in training, the API and the app."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tenure = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / tenure
    df["charge_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    present = [c for c in addons if c in df.columns]
    df["n_addons"] = sum((df[c] == "Yes").astype(int) for c in present)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_include=[object, "string"])),
    ])


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", model),
    ])
