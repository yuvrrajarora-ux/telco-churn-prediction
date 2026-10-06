"""Inference helper shared by the API and the Streamlit app."""
from functools import lru_cache

import joblib
import pandas as pd

from .config import MODEL_PATH


@lru_cache(maxsize=1)
def load_bundle(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict(records: list[dict]) -> list[dict]:
    bundle = load_bundle()
    X = pd.DataFrame(records)[bundle["columns"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    thr = bundle["threshold"]
    return [{"churn_probability": round(float(p), 4),
             "churn_prediction": bool(p >= thr),
             "risk": "high" if p >= max(thr, 0.5) else "medium" if p >= thr else "low"}
            for p in proba]
