"""
Retraining trigger.

Not wired to a scheduler in this repo — but the decision function is real and
takes structured inputs a cron/Airflow job would supply.

Signals (any one -> retrain):
  1. Scheduled: it's been >= N days since the last successful training
  2. Drift:     >= K features breached PSI alert in the last monitor run
  3. Performance: rolling 7-day AUC on labeled feedback dropped below floor

Usage:
    python -m src.retraining --auc 0.72
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def days_since_last_train(eval_dir: Path) -> float | None:
    latest = eval_dir / "latest.json"
    if not latest.exists():
        return None
    with open(latest) as f:
        report = json.load(f)
    trained_at = datetime.fromisoformat(report["meta"]["trained_at"].replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400.0


def latest_monitor_alerts(eval_dir: Path) -> int:
    monitors = sorted(eval_dir.glob("monitor_*.json"))
    if not monitors:
        return 0
    with open(monitors[-1]) as f:
        return json.load(f).get("drift", {}).get("n_alerts", 0)


def should_retrain(
    *,
    days_since_train: float | None,
    n_drift_alerts: int,
    recent_auc: float | None,
    cfg: dict,
) -> tuple[bool, list[str]]:
    """Pure decision function — easy to unit test."""
    reasons: list[str] = []

    if days_since_train is None or days_since_train >= cfg["retraining"]["scheduled_days"]:
        reasons.append(
            f"scheduled: {days_since_train} days since last train "
            f"(threshold {cfg['retraining']['scheduled_days']})"
        )

    if n_drift_alerts >= cfg["retraining"]["drift_feature_count"]:
        reasons.append(
            f"drift: {n_drift_alerts} features breached PSI alert "
            f"(threshold {cfg['retraining']['drift_feature_count']})"
        )

    if recent_auc is not None and recent_auc < cfg["retraining"]["recent_auc_floor"]:
        reasons.append(
            f"performance: recent AUC {recent_auc:.4f} < floor "
            f"{cfg['retraining']['recent_auc_floor']}"
        )

    return len(reasons) > 0, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auc", type=float, default=None, help="Optional recent rolling AUC")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    eval_dir = Path(cfg["paths"]["eval_dir"])

    days = days_since_last_train(eval_dir)
    alerts = latest_monitor_alerts(eval_dir)

    decision, reasons = should_retrain(
        days_since_train=days,
        n_drift_alerts=alerts,
        recent_auc=args.auc,
        cfg=cfg,
    )
    print(f"days_since_train = {days}")
    print(f"drift_alerts     = {alerts}")
    print(f"recent_auc       = {args.auc}")
    print(f"retrain?         = {decision}")
    for r in reasons:
        print(f"  * {r}")
    return 0 if not decision else 3  # non-zero exit signals a scheduler to kick off training


if __name__ == "__main__":
    sys.exit(main())
