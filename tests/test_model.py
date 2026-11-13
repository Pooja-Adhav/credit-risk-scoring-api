import numpy as np

from credit_risk.evaluate import best_threshold, classification_metrics, fairness_report
from credit_risk.predict import Scorer, risk_band


def test_model_beats_random(trained_model):
    _, metrics = trained_model
    assert metrics["test"]["xgboost"]["auc"] > 0.65
    assert metrics["test"]["logistic_regression"]["auc"] > 0.65


def test_scorer_outputs_valid_results(trained_model, synthetic_df):
    scorer = Scorer(trained_model[0])
    results = scorer.score(synthetic_df.head(20))
    assert len(results) == 20
    for r in results:
        assert 0 <= r.probability_of_default <= 1
        assert r.risk_band in {"A", "B", "C", "D", "E"}
        assert r.decision in {"approve", "manual_review", "decline"}
        assert len(r.top_risk_drivers) <= 3


def test_risk_band_boundaries():
    assert risk_band(0.01) == "A"
    assert risk_band(0.15) == "B"
    assert risk_band(0.99) == "E"


def test_metrics_on_perfect_scores():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0


def test_threshold_is_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    assert 0.05 <= best_threshold(y, p) <= 0.95


def test_fairness_report_shape():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.6, 0.2])
    rep = fairness_report(y, p, 0.5, np.array(["a", "a", "b", "b"]))
    assert set(rep["group"]) == {"a", "b"}
    assert 0 <= rep.attrs["approval_ratio"] <= 1
