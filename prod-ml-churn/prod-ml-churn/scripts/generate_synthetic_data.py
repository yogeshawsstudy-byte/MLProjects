"""
Generate synthetic customer records matching the IBM Telco Churn schema.

Produces daily CSV drops in data/raw/. The generator injects a mild but
learnable churn signal (short tenure + month-to-month + fiber = higher churn)
so models can achieve AUC ~ 0.82-0.86 — realistic for a public tutorial dataset.

Usage:
    python scripts/generate_synthetic_data.py --n 5000 --out data/raw/day_2026-07-25.csv
    python scripts/generate_synthetic_data.py --n 500 --drift  # for drift demo
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, size=n)
    phone = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiline_probs = [0.42, 0.42, 0.16]  # No, Yes, No phone service
    multi = rng.choice(["No", "Yes", "No phone service"], size=n, p=multiline_probs)
    multi = np.where(phone == "No", "No phone service", multi)

    if drift:
        # simulate a shift toward fiber optic + higher monthly charges
        internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.20, 0.70, 0.10])
    else:
        internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])

    def _addon(prob_yes: float):
        vals = rng.choice(["Yes", "No", "No internet service"], size=n,
                          p=[prob_yes, 0.9 - prob_yes, 0.10])
        return np.where(internet == "No", "No internet service", vals)

    online_sec = _addon(0.29)
    online_backup = _addon(0.34)
    device = _addon(0.34)
    tech = _addon(0.29)
    stream_tv = _addon(0.38)
    stream_mov = _addon(0.38)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.21, 0.24],
    )
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )

    base_monthly = 20.0
    monthly = base_monthly \
        + 20.0 * (internet == "Fiber optic") \
        + 5.0 * (internet == "DSL") \
        + 6.0 * (online_sec == "Yes") \
        + 5.0 * (online_backup == "Yes") \
        + 5.0 * (device == "Yes") \
        + 5.0 * (tech == "Yes") \
        + 6.0 * (stream_tv == "Yes") \
        + 6.0 * (stream_mov == "Yes") \
        + rng.normal(0, 3.0, size=n)
    if drift:
        monthly = monthly + 8.0  # price hike
    monthly = np.clip(monthly, 18.0, 130.0).round(2)

    total = (monthly * tenure + rng.normal(0, 20, size=n)).clip(min=0).round(2)

    # ---- Churn label ----
    # Higher risk: month-to-month, short tenure, fiber optic, electronic check, no add-ons
    # Signal is strong-ish (noise sigma=0.25) so a well-tuned model can cross AUC 0.80.
    risk = (
        1.6 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.4 * (contract == "One year")
        + 1.5 * (tenure < 6)
        + 0.8 * (tenure < 12)
        - 0.5 * (tenure > 36)
        + 0.9 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
        - 0.4 * ((online_sec == "Yes").astype(int) + (tech == "Yes").astype(int))
        + 0.3 * (senior == 1)
        + 0.02 * (monthly - 65.0)  # subtle interaction — monthly charges influence risk
    )
    logit = risk - 1.4 + rng.normal(0, 0.25, size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"C-{uuid.uuid4().hex[:8]}" for _ in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/day_synthetic.csv")
    ap.add_argument("--drift", action="store_true", help="Inject distribution shift")
    args = ap.parse_args()

    df = generate(args.n, seed=args.seed, drift=args.drift)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
