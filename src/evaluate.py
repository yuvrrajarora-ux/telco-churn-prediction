"""Metrics and diagnostic plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .config import FIGURES_DIR, RANDOM_STATE


def compute_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "threshold": float(threshold),
    }


def save_plots(pipe, X_test, y_test, proba, threshold, out=FIGURES_DIR):
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[0].set_title("ROC curve"); ax[1].set_title("Precision-Recall curve")
    fig.tight_layout(); fig.savefig(out / "roc_pr.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= threshold).astype(int),
                                            ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label="model"); ax.plot([0, 1], [0, 1], "--", label="perfect")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)

    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    idx = np.argsort(imp.importances_mean)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(X_test.columns[idx], imp.importances_mean[idx], xerr=imp.importances_std[idx])
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("Permutation importance (top 12)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)
