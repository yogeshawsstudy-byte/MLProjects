"""
Training pipeline: load -> split -> train baseline + candidate -> evaluate -> save.

Usage:
    python -m src.train
    python -m src.train --config config/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import build_features


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def data_hash(df: pd.DataFrame) -> str:
    """Fingerprint of the training data — helps trace which snapshot produced which model."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def train_baseline(X_train, y_train):
    """Baseline: L2-regularized logistic regression on scaled features."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(Xs, y_train)
    return {"scaler": scaler, "model": clf}


def train_candidate(X_train, y_train, params: dict):
    """Candidate: XGBoost with basic class-imbalance handling."""
    pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    clf = XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.9),
        eval_metric=params.get("eval_metric", "logloss"),
        scale_pos_weight=pos_weight,
        random_state=42,
        tree_method="hist",
    )
    clf.fit(X_train, y_train)
    return {"scaler": None, "model": clf}


def predict_proba(bundle, X):
    X_use = bundle["scaler"].transform(X) if bundle["scaler"] is not None else X
    return bundle["model"].predict_proba(X_use)[:, 1]


def evaluate(bundle, X_val, y_val, threshold: float = 0.5) -> dict:
    proba = predict_proba(bundle, X_val)
    pred = (proba >= threshold).astype(int)
    # recall@20% — top 20% highest predicted risk, how many actual churners captured?
    k = max(int(0.20 * len(proba)), 1)
    idx = np.argsort(-proba)[:k]
    recall_at_20 = y_val.iloc[idx].sum() / max(y_val.sum(), 1)
    return {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
        "recall_at_20pct": float(recall_at_20),
        "threshold": threshold,
        "n_samples": int(len(y_val)),
        "positive_rate": float(y_val.mean()),
    }


def promote(candidate_metrics: dict, baseline_metrics: dict, rule: dict) -> tuple[bool, str]:
    min_auc = rule["min_auc"]
    max_reg = rule["max_regression_vs_baseline"]
    if candidate_metrics["roc_auc"] < min_auc:
        return False, f"candidate AUC {candidate_metrics['roc_auc']:.4f} < min_auc {min_auc}"
    if candidate_metrics["roc_auc"] < baseline_metrics["roc_auc"] - max_reg:
        return False, (
            f"candidate AUC {candidate_metrics['roc_auc']:.4f} is worse than baseline "
            f"{baseline_metrics['roc_auc']:.4f} by more than {max_reg}"
        )
    return True, "candidate meets promotion criteria"


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    train_path = Path(cfg["paths"]["training_data"])
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir = Path(cfg["paths"]["eval_dir"])
    target = cfg["data"]["target_col"]

    print(f"[train] Loading {train_path}")
    df = pd.read_csv(train_path)
    print(f"[train] {len(df)} rows loaded")

    # Target encoding: "Yes"/"No" -> 1/0
    y = (df[target] == "Yes").astype(int)
    X = build_features(df)

    # 70/10/20 split — val is used for promotion decision, test held for final report
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    val_frac = cfg["data"]["val_size"] / (1 - cfg["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac,
        random_state=cfg["data"]["random_state"],
        stratify=y_tv,
    )
    print(f"[train] split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # ---- Baseline ----
    print("[train] Training baseline (logistic regression)...")
    baseline = train_baseline(X_train, y_train)
    baseline_val = evaluate(baseline, X_val, y_val)
    baseline_test = evaluate(baseline, X_test, y_test)

    # ---- Candidate ----
    print("[train] Training candidate (xgboost)...")
    candidate = train_candidate(X_train, y_train, cfg["training"]["candidate_params"])
    candidate_val = evaluate(candidate, X_val, y_val)
    candidate_test = evaluate(candidate, X_test, y_test)

    # ---- Promotion decision (on val set) ----
    promoted, reason = promote(candidate_val, baseline_val, cfg["promotion"])
    winner_name = "candidate" if promoted else "baseline"
    winner = candidate if promoted else baseline

    # ---- Save artifacts ----
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    training_hash = data_hash(df)
    bundle_meta = {
        "version": version,
        "model_type": cfg["training"]["candidate_model"] if promoted else cfg["training"]["baseline_model"],
        "training_data_hash": training_hash,
        "n_training_rows": len(df),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": list(X.columns),
    }

    active_path = models_dir / "active.joblib"
    versioned_path = models_dir / f"model_{version}.joblib"
    save_bundle({**winner, "meta": bundle_meta}, versioned_path)
    save_bundle({**winner, "meta": bundle_meta}, active_path)

    # Also save training feature stats — monitoring uses this as reference distribution
    ref_stats = X_train.describe().to_dict()
    with open(models_dir / "reference_stats.json", "w") as f:
        json.dump({"feature_stats": ref_stats, "meta": bundle_meta}, f, indent=2)

    report = {
        "version": version,
        "promoted": promoted,
        "promotion_reason": reason,
        "winner": winner_name,
        "baseline": {"val": baseline_val, "test": baseline_test},
        "candidate": {"val": candidate_val, "test": candidate_test},
        "meta": bundle_meta,
    }
    eval_path = eval_dir / f"eval_{version}.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)
    # Latest pointer
    with open(eval_dir / "latest.json", "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    print("\n" + "=" * 60)
    print(f"Version: {version}   Winner: {winner_name}")
    print(f"Promotion: {'YES' if promoted else 'NO'} — {reason}")
    print(f"{'Metric':<20}{'Baseline':>14}{'Candidate':>14}")
    print("-" * 60)
    for m in ["roc_auc", "pr_auc", "recall_at_20pct", "f1", "precision", "recall"]:
        print(f"{m:<20}{baseline_val[m]:>14.4f}{candidate_val[m]:>14.4f}")
    print("=" * 60)
    print(f"Report: {eval_path}")
    print(f"Model:  {active_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
