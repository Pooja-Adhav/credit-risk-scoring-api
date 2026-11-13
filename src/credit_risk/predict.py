"""Load a trained model and score applicants with plain-language reason codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from credit_risk.config import MODEL_PATH, RISK_BANDS
from credit_risk.features import FEATURE_DESCRIPTIONS, build_features


def risk_band(probability: float) -> str:
    for upper, band in RISK_BANDS:
        if probability < upper:
            return band
    return RISK_BANDS[-1][1]


@dataclass
class ScoreResult:
    probability_of_default: float
    risk_band: str
    decision: str
    top_risk_drivers: list[dict]


class Scorer:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model: xgb.XGBClassifier = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.threshold: float = bundle["threshold"]
        self.version: str = bundle["version"]

    def score(self, applicants: pd.DataFrame, n_reasons: int = 3) -> list[ScoreResult]:
        X = build_features(applicants)[self.features]
        probs = self.model.predict_proba(X)[:, 1]
        # SHAP values computed natively by XGBoost (last column is the bias term).
        contribs = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]

        results = []
        for prob, row in zip(probs, contribs, strict=True):
            order = np.argsort(row)[::-1][:n_reasons]
            drivers = [
                {"feature": self.features[i],
                 "description": FEATURE_DESCRIPTIONS.get(self.features[i], self.features[i]),
                 "impact": round(float(row[i]), 4)}
                for i in order if row[i] > 0
            ]
            if prob >= self.threshold:
                decision = "decline"
            elif prob >= self.threshold * 0.75:
                decision = "manual_review"
            else:
                decision = "approve"
            results.append(ScoreResult(round(float(prob), 4), risk_band(prob), decision, drivers))
        return results
