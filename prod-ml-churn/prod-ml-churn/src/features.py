"""
Shared feature engineering module.

CRITICAL: This module is imported by both training (train.py) and serving
(serve.py, batch_score.py). Using the SAME transformation code in both paths
is the primary defense against training-serving skew.

Contract:
  build_features(df_raw) -> df_features
  - Input:  raw records with the Telco Churn schema (see RAW_SCHEMA below)
  - Output: numeric feature matrix, deterministic column order
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Schema contract (validated by API and ingestion)
# ------------------------------------------------------------------
RAW_SCHEMA = {
    "customerID": "str",
    "gender": "str",
    "SeniorCitizen": "int",
    "Partner": "str",
    "Dependents": "str",
    "tenure": "int",
    "PhoneService": "str",
    "MultipleLines": "str",
    "InternetService": "str",
    "OnlineSecurity": "str",
    "OnlineBackup": "str",
    "DeviceProtection": "str",
    "TechSupport": "str",
    "StreamingTV": "str",
    "StreamingMovies": "str",
    "Contract": "str",
    "PaperlessBilling": "str",
    "PaymentMethod": "str",
    "MonthlyCharges": "float",
    "TotalCharges": "float",
}

ADDON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CONTRACT_RISK_MAP = {"Month-to-month": 3, "One year": 2, "Two year": 1}

# Deterministic feature order — matters for both model input and drift checks
FEATURE_ORDER = [
    # engineered numeric
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_tenure_month",
    "monthly_to_total_ratio",
    "num_addon_services",
    "contract_risk_score",
    "payment_auto_flag",
    "is_high_value",
    "tenure_bucket_ord",
    "SeniorCitizen",
    # one-hots (fixed set — unknowns collapse to base category)
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_Yes",
    "PaperlessBilling_Yes",
    "InternetService_Fiber_optic",
    "InternetService_No",
]


# ------------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------------
def _coerce_total_charges(series: pd.Series) -> pd.Series:
    """TotalCharges sometimes arrives as a whitespace-padded string (raw Telco data)."""
    return pd.to_numeric(series.replace(r"^\s*$", np.nan, regex=True), errors="coerce")


def _tenure_bucket(tenure: pd.Series) -> pd.Series:
    """Ordinal encoding: 0=0-12mo, 1=13-24, 2=25-48, 3=49+."""
    bins = [-1, 12, 24, 48, np.inf]
    return pd.cut(tenure, bins=bins, labels=[0, 1, 2, 3]).astype(int)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic feature construction. No fitting on data — all rules are static
    so that a single row (online) and a full batch (offline) produce identical values.
    """
    df = df_raw.copy()

    # Type coercion
    df["TotalCharges"] = _coerce_total_charges(df["TotalCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])  # new customer edge case
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    # ---- Engineered numeric features ----
    # 1. Normalized spend per month of tenure
    df["charges_per_tenure_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # 2. Recency ratio — high value implies short tenure
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)

    # 3. Count of premium add-ons subscribed
    df["num_addon_services"] = sum(
        (df[c] == "Yes").astype(int) for c in ADDON_SERVICE_COLS
    )

    # 4. Contract risk score (month-to-month = highest churn risk)
    df["contract_risk_score"] = df["Contract"].map(CONTRACT_RISK_MAP).fillna(3).astype(int)

    # 5. Payment auto flag (auto-pay correlates with lower churn)
    df["payment_auto_flag"] = df["PaymentMethod"].str.contains(
        "automatic", case=False, na=False
    ).astype(int)

    # 6. High-value customer flag (fixed threshold — chosen from training mean, hardcoded here
    #    to keep online path stateless; in production this would come from a feature store)
    df["is_high_value"] = (df["MonthlyCharges"] > 70.0).astype(int)

    # 7. Ordinal tenure bucket
    df["tenure_bucket_ord"] = _tenure_bucket(df["tenure"])

    # ---- Categorical to binary (fixed vocab — unknowns collapse to base) ----
    df["gender_Male"] = (df["gender"] == "Male").astype(int)
    df["Partner_Yes"] = (df["Partner"] == "Yes").astype(int)
    df["Dependents_Yes"] = (df["Dependents"] == "Yes").astype(int)
    df["PhoneService_Yes"] = (df["PhoneService"] == "Yes").astype(int)
    df["MultipleLines_Yes"] = (df["MultipleLines"] == "Yes").astype(int)
    df["PaperlessBilling_Yes"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["InternetService_Fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)

    return df[FEATURE_ORDER].astype(float)


def validate_schema(record: dict) -> tuple[bool, list[str]]:
    """Return (ok, missing_fields) — used by the API to reject malformed inputs."""
    missing = [k for k in RAW_SCHEMA if k not in record]
    return len(missing) == 0, missing
