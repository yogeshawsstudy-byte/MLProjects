"""
Batch inference — Track B primary serving pattern.

Reads a CSV of customer records, scores every row with the active model,
writes predictions + probabilities to a timestamped output CSV.

Usage:
    python -m src.batch_score --input data/training/training.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.features import build_features
from src.train import predict_proba


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV of customers to score")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else cfg["serving"]["default_threshold"]

    model_path = Path(cfg["paths"]["models_dir"]) / "active.joblib"
    if not model_path.exists():
        print(f"[batch_score] No active model at {model_path}. Run `python -m src.train` first.", file=sys.stderr)
        return 1

    bundle = joblib.load(model_path)
    meta = bundle["meta"]
    print(f"[batch_score] Loaded {meta['model_type']} v{meta['version']}")

    df = pd.read_csv(args.input)
    print(f"[batch_score] Scoring {len(df)} rows...")

    t0 = time.perf_counter()
    X = build_features(df)
    proba = predict_proba(bundle, X)
    pred = (proba >= threshold).astype(int)
    elapsed = time.perf_counter() - t0

    id_col = cfg["data"]["id_col"]
    out = pd.DataFrame({
        id_col: df[id_col] if id_col in df.columns else range(len(df)),
        "churn_probability": proba,
        "churn_prediction": pred,
        "model_version": meta["version"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    })

    out_dir = Path(cfg["paths"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"predictions_{ts}.csv"
    out.to_csv(out_path, index=False)

    rate = len(df) / elapsed if elapsed > 0 else float("inf")
    print(f"[batch_score] Done in {elapsed:.3f}s ({rate:,.0f} rows/sec)")
    print(f"[batch_score] Positive rate: {pred.mean():.3f}  (threshold={threshold})")
    print(f"[batch_score] Wrote {out_path}")

    # Log run summary
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "elapsed_sec": elapsed,
        "rows_per_sec": rate,
        "positive_rate": float(pred.mean()),
        "model_version": meta["version"],
        "output": str(out_path),
    }
    with open(out_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
