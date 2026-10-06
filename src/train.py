"""Train, compare, tune and save the churn model.

Usage: python -m src.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

from .config import CV_FOLDS, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from .data import load_data, split_xy
from .evaluate import compute_metrics, save_plots
from .features import build_pipeline

CANDIDATES = {
    "dummy": DummyClassifier(strategy="prior"),
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            n_jobs=-1, random_state=RANDOM_STATE),
    "hist_gb": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

PARAM_SPACES = {
    "logreg": {"model__C": loguniform(1e-2, 1e2)},
    "random_forest": {"model__max_depth": [4, 6, 8, 12, None],
                      "model__min_samples_leaf": [1, 3, 5, 10]},
    "hist_gb": {"model__learning_rate": loguniform(1e-2, 3e-1),
                "model__max_depth": [2, 3, 4, 6],
                "model__max_leaf_nodes": [8, 15, 31],
                "model__l2_regularization": loguniform(1e-3, 10)},
}


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def main():
    df = load_data()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {X_tr.shape} | Test: {X_te.shape} | churn rate: {y.mean():.1%}")

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1) Compare candidate models with cross-validation (training data only)
    rows = []
    for name, model in CANDIDATES.items():
        res = cross_validate(build_pipeline(model), X_tr, y_tr, cv=cv, n_jobs=-1,
                             scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        rows.append({"model": name, "roc_auc": res["test_roc_auc"].mean(),
                     "roc_auc_std": res["test_roc_auc"].std(),
                     "pr_auc": res["test_pr_auc"].mean()})
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    print("\nCross-validated comparison:\n", comparison.round(4).to_string(index=False))

    # 2) Tune the best real model
    best_name = comparison[comparison.model != "dummy"].iloc[0]["model"]
    print(f"\nTuning: {best_name}")
    search = RandomizedSearchCV(build_pipeline(CANDIDATES[best_name]), PARAM_SPACES[best_name],
                                n_iter=20, scoring="roc_auc", cv=cv, n_jobs=-1,
                                random_state=RANDOM_STATE)
    search.fit(X_tr, y_tr)
    pipe = search.best_estimator_
    print(f"Best CV ROC-AUC: {search.best_score_:.4f} | params: {search.best_params_}")

    # 3) Pick decision threshold from out-of-fold predictions (no test leakage)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_tr, oof)

    # 4) Final, one-time evaluation on the held-out test set
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, proba, threshold)
    metrics.update(best_model=best_name,
                   best_params={k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                for k, v in search.best_params_.items()},
                   comparison=comparison.round(4).to_dict(orient="records"))
    print("\nTest metrics:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    save_plots(pipe, X_te, y_te, proba, threshold)

    # 5) Persist model + threshold + schema together
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold, "columns": list(X.columns)}, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
