"""
Monitoring & data quality.

Two lightweight checks that would run daily in production:

  1. Data quality — null rates, out-of-range values, schema conformance
  2. Feature drift — Population Stability Index (PSI) between the training
     reference distribution (saved at train time) and a recent scoring batch

Usage:
    python -m src.monitoring --recent data/raw/day_2026-07-25.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import RAW_SCHEMA, build_features


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Data quality
# ------------------------------------------------------------------
def data_quality_report(df_raw: pd.DataFrame) -> dict:
    """
    Report checked BEFORE feature engineering. Catches upstream issues:
      - Missing/renamed columns (schema drift)
      - Null spikes
      - Out-of-range numeric values
      - New categorical levels
    """
    issues: list[str] = []
    n = len(df_raw)

    # Schema check
    missing_cols = [c for c in RAW_SCHEMA if c not in df_raw.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")

    # Null rates per column
    null_rates = {c: float(df_raw[c].isna().mean()) for c in df_raw.columns if c in RAW_SCHEMA}
    high_null = {c: r for c, r in null_rates.items() if r > 0.05}
    if high_null:
        issues.append(f"null rate >5% in: {high_null}")

    # Range checks
    range_issues: dict[str, str] = {}
    if "tenure" in df_raw.columns:
        bad = ((df_raw["tenure"] < 0) | (df_raw["tenure"] > 120)).sum()
        if bad:
            range_issues["tenure"] = f"{bad} rows outside [0, 120]"
    if "MonthlyCharges" in df_raw.columns:
        bad = ((df_raw["MonthlyCharges"] < 0) | (df_raw["MonthlyCharges"] > 1000)).sum()
        if bad:
            range_issues["MonthlyCharges"] = f"{bad} rows outside [0, 1000]"
    if range_issues:
        issues.append(f"range violations: {range_issues}")

    # New categorical values vs known vocab
    known_vocab = {
        "Contract": {"Month-to-month", "One year", "Two year"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "PaymentMethod": {
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        },
    }
    new_levels: dict[str, list] = {}
    for col, known in known_vocab.items():
        if col in df_raw.columns:
            seen = set(df_raw[col].dropna().unique())
            unknown = list(seen - known)
            if unknown:
                new_levels[col] = unknown
    if new_levels:
        issues.append(f"new categorical values: {new_levels}")

    return {
        "n_rows": n,
        "null_rates": null_rates,
        "issues": issues,
        "status": "ok" if not issues else "warn",
    }


# ------------------------------------------------------------------
# Drift — Population Stability Index (PSI)
# ------------------------------------------------------------------
def psi(reference: np.ndarray, recent: np.ndarray, buckets: int = 10) -> float:
    """
    PSI = sum( (recent% - ref%) * ln(recent% / ref%) ) across buckets.

    Rule of thumb:
      < 0.10  : no significant change
      0.10–0.25: moderate shift, investigate
      > 0.25  : major shift, likely need to retrain
    """
    reference = np.asarray(reference, dtype=float)
    recent = np.asarray(recent, dtype=float)
    reference = reference[~np.isnan(reference)]
    recent = recent[~np.isnan(recent)]
    if len(reference) == 0 or len(recent) == 0:
        return float("nan")

    # Quantile bins from reference — same edges applied to recent
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_hist, _ = np.histogram(reference, bins=edges)
    rec_hist, _ = np.histogram(recent, bins=edges)

    # Laplace smoothing to avoid div-by-zero
    ref_pct = (ref_hist + 1) / (ref_hist.sum() + len(ref_hist))
    rec_pct = (rec_hist + 1) / (rec_hist.sum() + len(rec_hist))
    return float(np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct)))


def drift_report(df_recent: pd.DataFrame, reference_stats: dict, cfg: dict) -> dict:
    """
    Compares recent batch's feature distributions to training reference.

    reference_stats is loaded from models/reference_stats.json (produced by train.py).
    For a demo run without a saved reference we fall back to splitting df_recent
    in half to synthesize a reference vs recent comparison.
    """
    features = build_features(df_recent)
    per_feature: dict[str, dict] = {}

    warn = cfg["monitoring"]["psi_warn"]
    alert = cfg["monitoring"]["psi_alert"]

    for col in cfg["monitoring"]["drift_features"]:
        if col not in features.columns:
            per_feature[col] = {"status": "missing"}
            continue
        recent_vals = features[col].values

        # Reconstitute a reference sample from stored describe() stats
        stats = reference_stats.get(col, {})
        if stats and "mean" in stats and "std" in stats and stats.get("count", 0) > 30:
            rng = np.random.default_rng(42)
            ref_vals = rng.normal(stats["mean"], max(stats["std"], 1e-6), size=int(stats["count"]))
        else:
            # No reference — split recent in half so PSI = ~0 (sanity check)
            mid = len(recent_vals) // 2
            ref_vals = recent_vals[:mid]
            recent_vals = recent_vals[mid:]

        score = psi(ref_vals, recent_vals)
        status = "ok"
        if score >= alert:
            status = "alert"
        elif score >= warn:
            status = "warn"
        per_feature[col] = {
            "psi": score,
            "status": status,
            "recent_mean": float(np.nanmean(recent_vals)),
            "recent_std": float(np.nanstd(recent_vals)),
        }

    n_alerts = sum(1 for v in per_feature.values() if v.get("status") == "alert")
    overall = "alert" if n_alerts > 0 else (
        "warn" if any(v.get("status") == "warn" for v in per_feature.values()) else "ok"
    )
    return {
        "features": per_feature,
        "n_alerts": n_alerts,
        "overall_status": overall,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", required=True, help="CSV of recent scoring batch")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.recent)

    dq = data_quality_report(df)

    ref_path = Path(cfg["paths"]["models_dir"]) / "reference_stats.json"
    if ref_path.exists():
        with open(ref_path) as f:
            reference_stats = json.load(f).get("feature_stats", {})
    else:
        print("[monitor] WARN: no reference_stats.json — run training first for real drift check")
        reference_stats = {}

    drift = drift_report(df, reference_stats, cfg)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "input": args.recent,
        "data_quality": dq,
        "drift": drift,
    }
    out_dir = Path(cfg["paths"]["eval_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"monitor_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable
    print("\n" + "=" * 60)
    print(f"DATA QUALITY: {dq['status']}   ({len(dq['issues'])} issue(s))")
    for issue in dq["issues"]:
        print(f"  - {issue}")
    print(f"\nDRIFT: {drift['overall_status']}   ({drift['n_alerts']} feature alert(s))")
    for feat, res in drift["features"].items():
        psi_val = res.get("psi")
        if psi_val is None:
            print(f"  {feat:<30} MISSING")
        else:
            print(f"  {feat:<30} PSI={psi_val:.4f}  [{res['status']}]")
    print("=" * 60)
    print(f"Report: {out_path}")

    # Non-zero exit on alert — CI/cron can pick this up
    return 2 if drift["overall_status"] == "alert" or dq["status"] != "ok" else 0


if __name__ == "__main__":
    sys.exit(main())
