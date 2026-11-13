import pytest
from fastapi.testclient import TestClient

from credit_risk.api import app

APPLICANT = {
    "limit_bal": 50000,
    "education": 2,
    "age": 35,
    "repayment_status": [2, 2, 0, 0, 0, 0],
    "bill_amounts": [48000, 47000, 45000, 40000, 38000, 30000],
    "payment_amounts": [0, 1500, 2000, 1500, 1500, 2000],
}


@pytest.fixture()
def client(trained_model, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model[0]))
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_score_single(client):
    res = client.post("/score", json=APPLICANT)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "manual_review", "decline"}


def test_score_batch(client):
    res = client.post("/score/batch", json={"applicants": [APPLICANT, APPLICANT]})
    assert res.status_code == 200
    assert len(res.json()) == 2


def test_risky_profile_scores_higher(client):
    safe = {**APPLICANT, "repayment_status": [-1] * 6, "payment_amounts": [48000] * 6}
    risky = {**APPLICANT, "repayment_status": [4, 3, 3, 2, 2, 2], "payment_amounts": [0] * 6}
    p_safe = client.post("/score", json=safe).json()["probability_of_default"]
    p_risky = client.post("/score", json=risky).json()["probability_of_default"]
    assert p_risky > p_safe


@pytest.mark.parametrize("field,value", [
    ("age", 10),
    ("limit_bal", -5),
    ("repayment_status", [0, 0, 0]),
    ("education", 9),
])
def test_invalid_input_rejected(client, field, value):
    assert client.post("/score", json={**APPLICANT, field: value}).status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "model_missing"
        assert c.post("/score", json=APPLICANT).status_code == 503
