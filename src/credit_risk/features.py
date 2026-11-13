"""Feature engineering shared by training and the API, so both see identical inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS

BASE_FEATURES = ["LIMIT_BAL", "EDUCATION", "AGE", *PAY_STATUS_COLS, *BILL_COLS, *PAY_AMT_COLS]

ENGINEERED_FEATURES = [
    "utilization_recent",   # latest bill / credit limit
    "utilization_avg",      # average bill / credit limit over 6 months
    "payment_ratio",        # total paid / total billed
    "months_delayed",       # number of months with a payment delay
    "max_delay",            # worst delay in months
    "bill_trend",           # change in balance over 6 months, relative to limit
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# Plain-language descriptions used in the API's reason codes.
FEATURE_DESCRIPTIONS = {
    "LIMIT_BAL": "credit limit",
    "EDUCATION": "education level",
    "AGE": "age",
    "utilization_recent": "current credit utilisation",
    "utilization_avg": "average credit utilisation",
    "payment_ratio": "share of bills repaid",
    "months_delayed": "number of months with late payments",
    "max_delay": "longest payment delay",
    "bill_trend": "trend in outstanding balance",
    **{c: f"repayment status {i} month(s) ago" for i, c in enumerate(PAY_STATUS_COLS, 1)},
    **{c: f"bill amount {i} month(s) ago" for i, c in enumerate(BILL_COLS, 1)},
    **{c: f"payment amount {i} month(s) ago" for i, c in enumerate(PAY_AMT_COLS, 1)},
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix. Protected attributes are deliberately excluded."""
    out = df[BASE_FEATURES].astype(float).copy()
    limit = out["LIMIT_BAL"].replace(0, np.nan)
    bills = out[BILL_COLS]
    paid_total = out[PAY_AMT_COLS].sum(axis=1)
    billed_total = bills.clip(lower=0).sum(axis=1)
    delays = out[PAY_STATUS_COLS]

    out["utilization_recent"] = out["BILL_AMT1"] / limit
    out["utilization_avg"] = bills.mean(axis=1) / limit
    out["payment_ratio"] = np.where(billed_total > 0, paid_total / billed_total.replace(0, np.nan), 1.0)
    out["months_delayed"] = (delays > 0).sum(axis=1)
    out["max_delay"] = delays.max(axis=1).clip(lower=0)
    out["bill_trend"] = (out["BILL_AMT1"] - out["BILL_AMT6"]) / limit

    out["payment_ratio"] = out["payment_ratio"].clip(0, 5)
    return out[FEATURES].fillna(0.0)
