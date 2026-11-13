import pandas as pd

from credit_risk.config import PROTECTED_COLS
from credit_risk.data import clean
from credit_risk.features import FEATURES, build_features


def test_feature_matrix_has_expected_columns(synthetic_df):
    X = build_features(synthetic_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(synthetic_df)


def test_protected_attributes_are_not_model_inputs():
    assert not set(PROTECTED_COLS) & set(FEATURES)


def test_no_missing_or_infinite_values(synthetic_df):
    X = build_features(synthetic_df)
    assert X.notna().all().all()
    assert (X.abs() < float("inf")).all().all()


def test_zero_limit_does_not_break_features(synthetic_df):
    row = synthetic_df.head(1).copy()
    row["LIMIT_BAL"] = 0
    assert build_features(row).notna().all().all()


def test_clean_maps_undocumented_codes(synthetic_df):
    raw = synthetic_df.head(3).copy()
    raw["EDUCATION"] = [0, 5, 6]
    raw["MARRIAGE"] = [0, 1, 2]
    out = clean(raw)
    assert set(out["EDUCATION"]) == {4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}


def test_engineered_values_on_known_row():
    row = pd.DataFrame([{
        "LIMIT_BAL": 10_000, "EDUCATION": 2, "AGE": 30,
        **{f"PAY_{m}": (2 if m == 1 else 0) for m in range(1, 7)},
        **{f"BILL_AMT{m}": 5_000 for m in range(1, 7)},
        **{f"PAY_AMT{m}": 1_000 for m in range(1, 7)},
    }])
    X = build_features(row).iloc[0]
    assert X["utilization_recent"] == 0.5
    assert X["payment_ratio"] == 0.2
    assert X["months_delayed"] == 1
    assert X["max_delay"] == 2
