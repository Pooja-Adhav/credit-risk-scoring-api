"""Train a logistic-regression baseline and an XGBoost model, evaluate, and save the best one.

Usage:
    python -m credit_risk.train
    python -m credit_risk.train --data data/credit_default.csv --no-plots
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk import __version__
from credit_risk.config import FIGURES_DIR, MODEL_PATH, RANDOM_STATE, RAW_DATA_PATH, REPORTS_DIR, TARGET
from credit_risk.data import load
from credit_risk.evaluate import (
    best_threshold,
    classification_metrics,
    decision_metrics,
    fairness_report,
    save_plots,
)
from credit_risk.features import FEATURES, build_features

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "early_stopping_rounds": 100,
    "random_state": RANDOM_STATE,
}


def split(df: pd.DataFrame):
    """60 / 20 / 20 stratified train / validation / test split."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=RANDOM_STATE)
    return train, val, test


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 30, 45, 120], labels=["<30", "30-45", "45+"]).astype(str)


def train(df: pd.DataFrame, model_path: Path = MODEL_PATH, reports_dir: Path | None = REPORTS_DIR,
          plots: bool = True) -> dict:
    train_df, val_df, test_df = split(df)
    X_train, X_val, X_test = (build_features(d) for d in (train_df, val_df, test_df))
    y_train, y_val, y_test = (d[TARGET].to_numpy() for d in (train_df, val_df, test_df))

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_train, y_train)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])

    p_base = baseline.predict_proba(X_test)[:, 1]
    p_xgb = model.predict_proba(X_test)[:, 1]

    contribs = model.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)[:, :-1]
    importance = pd.Series(np.abs(contribs).mean(axis=0), index=FEATURES)

    fairness = {
        "sex": fairness_report(y_test, p_xgb, threshold, test_df["SEX"].map({1: "male", 2: "female"})),
        "age_band": fairness_report(y_test, p_xgb, threshold, age_band(test_df["AGE"])),
    }

    metrics = {
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "default_rate": round(float(df[TARGET].mean()), 4),
        "test": {
            "logistic_regression": classification_metrics(y_test, p_base),
            "xgboost": classification_metrics(y_test, p_xgb),
        },
        "decision": decision_metrics(y_test, p_xgb, threshold),
        "xgboost_best_iteration": int(model.best_iteration),
        "fairness": {
            name: {"approval_ratio": rep.attrs["approval_ratio"], "groups": rep.to_dict("records")}
            for name, rep in fairness.items()
        },
        "top_features": importance.sort_values(ascending=False).head(10).round(4).to_dict(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold,
                 "version": __version__, "metrics": metrics["test"]["xgboost"]}, model_path)

    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if plots:
            save_plots(y_test, {"Logistic regression": p_base, "XGBoost": p_xgb}, importance,
                       reports_dir / FIGURES_DIR.name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit risk model.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    metrics = train(load(args.data), args.model_out, plots=not args.no_plots)
    print(json.dumps({k: metrics[k] for k in ("test", "decision")}, indent=2))
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
