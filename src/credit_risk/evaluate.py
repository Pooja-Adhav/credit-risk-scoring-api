"""Credit-risk metrics, cost-based threshold selection, fairness checks and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from credit_risk.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max separation between defaulter and non-defaulter score CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
    }


def expected_cost(y_true, y_prob, threshold: float) -> float:
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_prob) >= threshold
    fn = np.sum(~flagged & (y_true == 1))
    fp = np.sum(flagged & (y_true == 0))
    return float((COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp) / len(y_true))


def best_threshold(y_true, y_prob) -> float:
    """Pick the cut-off that minimises expected business cost (chosen on validation data)."""
    grid = np.linspace(0.05, 0.95, 181)
    costs = [expected_cost(y_true, y_prob, t) for t in grid]
    return round(float(grid[int(np.argmin(costs))]), 3)


def decision_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    flagged = np.asarray(y_prob) >= threshold
    tp = np.sum(flagged & (y_true == 1))
    return {
        "threshold": threshold,
        "flag_rate": round(float(flagged.mean()), 4),
        "precision": round(float(tp / max(flagged.sum(), 1)), 4),
        "recall": round(float(tp / max((y_true == 1).sum(), 1)), 4),
        "expected_cost_per_client": round(expected_cost(y_true, y_prob, threshold), 4),
    }


def fairness_report(y_true, y_prob, threshold: float, groups: pd.Series) -> pd.DataFrame:
    """Approval rate and error rates per group (demographic parity / equal opportunity view)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "flag": np.asarray(y_prob) >= threshold,
                          "group": np.asarray(groups)})
    rows = []
    for g, part in frame.groupby("group"):
        good, bad = part[part.y == 0], part[part.y == 1]
        rows.append({
            "group": g,
            "n": len(part),
            "default_rate": part.y.mean(),
            "approval_rate": 1 - part.flag.mean(),
            "true_positive_rate": bad.flag.mean() if len(bad) else np.nan,
            "false_positive_rate": good.flag.mean() if len(good) else np.nan,
        })
    report = pd.DataFrame(rows).round(4)
    rates = report["approval_rate"]
    report.attrs["approval_ratio"] = round(float(rates.min() / rates.max()), 4) if rates.max() else 1.0
    return report


def save_plots(y_true, probs: dict[str, np.ndarray], importance: pd.Series, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xlabel="Predicted probability of default", ylabel="Observed default rate",
           title="Calibration (test set)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)

    top = importance.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top.index, top.values)
    ax.set(xlabel="Mean |SHAP value|", title="Top drivers of predicted default risk")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
