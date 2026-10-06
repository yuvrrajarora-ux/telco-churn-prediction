"""Generate a synthetic dataset with the same schema as the IBM Telco Customer
Churn dataset (Kaggle), so the project runs without any download.

To use the real data instead, download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
from Kaggle and save it as data/telco_churn.csv - the pipeline works unchanged.

Usage: python -m src.generate_data --rows 7000
"""
import argparse

import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE


def generate(n: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)

    tenure = np.clip(rng.exponential(30, n).astype(int) + 1, 1, 72)
    internet = pick(["DSL", "Fiber optic", "No"], [0.35, 0.45, 0.20])
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    phone = pick(["Yes", "No"], [0.9, 0.1])

    def addon():
        v = pick(["Yes", "No"])
        return np.where(internet == "No", "No internet service", v)

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": pick(["Male", "Female"]),
        "SeniorCitizen": rng.binomial(1, 0.16, n),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", pick(["Yes", "No"])),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": addon(),
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"], [0.6, 0.4]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
    })

    n_addons = sum((df[c] == "Yes").astype(int) for c in
                   ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"])
    monthly = (20 + 45 * (internet == "Fiber optic") + 25 * (internet == "DSL")
               + 5 * n_addons + rng.normal(0, 3, n))
    df["MonthlyCharges"] = monthly.round(2)
    df["TotalCharges"] = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)

    logit = (-1.0 + 1.1 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
             + 0.7 * (internet == "Fiber optic") - 0.03 * tenure
             + 0.008 * (monthly - 65) + 0.4 * (df["PaymentMethod"] == "Electronic check")
             + 0.3 * (df["PaperlessBilling"] == "Yes") + 0.3 * df["SeniorCitizen"]
             - 0.5 * (df["TechSupport"] == "Yes") - 0.4 * (df["OnlineSecurity"] == "Yes"))
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")

    # Mimic the real dataset's quirk: blank TotalCharges for a few customers
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[rng.choice(n, 10, replace=False), "TotalCharges"] = " "
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=7000)
    args = ap.parse_args()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate(args.rows)
    data.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(data)} rows to {DATA_PATH} | churn rate: {(data.Churn == 'Yes').mean():.1%}")
