"""Shared fixtures: tests use synthetic data, so they run offline and in CI."""

import pytest

from credit_risk.data import clean, make_synthetic
from credit_risk.train import train


@pytest.fixture(scope="session")
def synthetic_df():
    return clean(make_synthetic(n=2000))


@pytest.fixture(scope="session")
def trained_model(synthetic_df, tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    metrics = train(synthetic_df, model_path=model_path, reports_dir=None, plots=False)
    return model_path, metrics
