"""
FastAPI inference service.

Track B is primarily batch, but exposing a low-QPS API is useful for:
  - Retention agents pulling a single customer's score in a CRM
  - Debugging/what-if analysis
  - Ad-hoc scoring of a new signup before batch runs

Endpoints:
    GET  /health            -> service + model status
    GET  /model             -> active model metadata
    POST /predict           -> single record
    POST /predict/batch     -> array of records (bounded)

Run:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import RAW_SCHEMA, build_features, validate_schema
from src.train import predict_proba


CONFIG_PATH = "config/config.yaml"
MAX_BATCH = 500


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict):
    p = Path(cfg["paths"]["models_dir"]) / "active.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


# ------------------------------------------------------------------
# Pydantic contracts — the API's request schema is derived from the
# same RAW_SCHEMA that ingest.py and features.py enforce.
# ------------------------------------------------------------------
class CustomerRecord(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    customerID: str
    churn_probability: float
    churn_prediction: int
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    records: list[CustomerRecord]


# ------------------------------------------------------------------
app = FastAPI(title="Churn Prediction API", version="0.1.0")
_cfg = _load_cfg()
_bundle: dict[str, Any] | None = _load_model(_cfg)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _bundle is not None else "no_model",
        "model_loaded": _bundle is not None,
    }


@app.get("/model")
def model_meta() -> dict:
    if _bundle is None:
        raise HTTPException(503, "no active model — run training first")
    return _bundle["meta"]


def _score(records: list[dict]) -> list[dict]:
    if _bundle is None:
        raise HTTPException(503, "no active model")
    df = pd.DataFrame(records)
    X = build_features(df)
    proba = predict_proba(_bundle, X)
    threshold = _cfg["serving"]["default_threshold"]
    pred = (proba >= threshold).astype(int)
    meta = _bundle["meta"]
    return [
        {
            "customerID": r["customerID"],
            "churn_probability": float(proba[i]),
            "churn_prediction": int(pred[i]),
            "model_version": meta["version"],
        }
        for i, r in enumerate(records)
    ]


@app.post("/predict", response_model=Prediction)
def predict(rec: CustomerRecord) -> dict:
    t0 = time.perf_counter()
    ok, missing = validate_schema(rec.model_dump())
    if not ok:
        raise HTTPException(400, f"missing fields: {missing}")
    result = _score([rec.model_dump()])[0]
    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    if len(req.records) == 0:
        raise HTTPException(400, "empty batch")
    if len(req.records) > MAX_BATCH:
        raise HTTPException(413, f"batch size {len(req.records)} > max {MAX_BATCH}")
    t0 = time.perf_counter()
    results = _score([r.model_dump() for r in req.records])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "n": len(results),
        "elapsed_ms": elapsed_ms,
        "rows_per_sec": len(results) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else None,
        "predictions": results,
    }
