"""API tests — requires that a model has been trained (`python -m src.train`)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path("models/active.joblib")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="active model missing — run `python -m src.train` first",
)


@pytest.fixture
def client():
    from src.serve import app
    return TestClient(app)


def _valid_record() -> dict:
    return {
        "customerID": "C-API-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 1074.0,
    }


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_metadata(client):
    r = client.get("/model")
    assert r.status_code == 200
    meta = r.json()
    assert "version" in meta
    assert "model_type" in meta


def test_predict_shape(client):
    r = client.post("/predict", json=_valid_record())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["model_version"]
    assert body["latency_ms"] > 0


def test_predict_rejects_invalid_types(client):
    bad = _valid_record()
    bad["tenure"] = -1  # violates Field(ge=0)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict(client):
    payload = {"records": [_valid_record() for _ in range(10)]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 10
    assert len(body["predictions"]) == 10


def test_batch_size_limit(client):
    payload = {"records": [_valid_record() for _ in range(501)]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 413
