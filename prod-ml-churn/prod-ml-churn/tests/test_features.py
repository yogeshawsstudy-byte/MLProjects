"""Unit tests for shared feature module — pin the training/serving contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_ORDER, RAW_SCHEMA, build_features, validate_schema


def _sample_record() -> dict:
    return {
        "customerID": "C-TEST",
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


def test_feature_order_stable():
    df = pd.DataFrame([_sample_record()])
    X = build_features(df)
    assert list(X.columns) == FEATURE_ORDER, "feature order drift will silently break inference"


def test_single_vs_batch_identical():
    """The core skew defense: 1 row on its own == same row inside a batch."""
    r1, r2, r3 = _sample_record(), _sample_record(), _sample_record()
    r2["tenure"] = 45; r2["Contract"] = "Two year"
    r3["InternetService"] = "No"

    single = build_features(pd.DataFrame([r2]))
    batch = build_features(pd.DataFrame([r1, r2, r3]))

    np.testing.assert_array_almost_equal(single.iloc[0].values, batch.iloc[1].values)


def test_addon_count():
    r = _sample_record()  # has OnlineBackup=Yes, StreamingTV=Yes, StreamingMovies=Yes
    X = build_features(pd.DataFrame([r]))
    assert X["num_addon_services"].iloc[0] == 3.0


def test_contract_risk():
    for contract, expected in [("Month-to-month", 3), ("One year", 2), ("Two year", 1)]:
        r = _sample_record()
        r["Contract"] = contract
        X = build_features(pd.DataFrame([r]))
        assert X["contract_risk_score"].iloc[0] == expected


def test_payment_auto_flag():
    for pm, expected in [
        ("Electronic check", 0),
        ("Mailed check", 0),
        ("Bank transfer (automatic)", 1),
        ("Credit card (automatic)", 1),
    ]:
        r = _sample_record()
        r["PaymentMethod"] = pm
        X = build_features(pd.DataFrame([r]))
        assert X["payment_auto_flag"].iloc[0] == expected


def test_zero_tenure_no_divide_error():
    r = _sample_record()
    r["tenure"] = 0
    r["TotalCharges"] = 0.0
    X = build_features(pd.DataFrame([r]))
    assert np.isfinite(X["charges_per_tenure_month"].iloc[0])


def test_total_charges_whitespace_string():
    """Real Telco CSV has ' ' where TotalCharges is missing for new customers."""
    r = _sample_record()
    df = pd.DataFrame([r])
    df["TotalCharges"] = " "
    X = build_features(df)
    assert np.isfinite(X["TotalCharges"].iloc[0])


def test_validate_schema_missing_field():
    r = _sample_record()
    del r["tenure"]
    ok, missing = validate_schema(r)
    assert not ok
    assert "tenure" in missing


def test_unknown_categorical_collapses():
    """New/unknown categorical values shouldn't crash — they collapse to base category."""
    r = _sample_record()
    r["Contract"] = "Three year"  # unseen level
    X = build_features(pd.DataFrame([r]))
    # Should fall back to default risk (3)
    assert X["contract_risk_score"].iloc[0] == 3
