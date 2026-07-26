"""
Batch data ingestion.

Reads any *.csv in `data/raw/` that hasn't already been ingested, appends new
rows to the master training file, and writes a JSON-lines log of every run.

Usage:
    python -m src.ingest
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.features import RAW_SCHEMA


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def already_ingested(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    seen = set()
    with open(log_path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def validate_columns(df: pd.DataFrame, required: dict) -> list[str]:
    """Return list of missing columns; empty list means valid."""
    return [c for c in required if c not in df.columns]


def ingest(cfg: dict) -> dict:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    train_path = Path(cfg["paths"]["training_data"])
    log_path = Path(cfg["paths"]["ingest_log"])
    target = cfg["data"]["target_col"]

    train_path.parent.mkdir(parents=True, exist_ok=True)

    seen = already_ingested(log_path)
    csvs = sorted(raw_dir.glob("*.csv"))
    new_files = [p for p in csvs if p.name not in seen]

    if not new_files:
        print(f"[ingest] No new files in {raw_dir}. Seen={len(seen)}")
        return {"files": 0, "rows": 0}

    # Load existing training data if present
    if train_path.exists():
        train_df = pd.read_csv(train_path)
    else:
        train_df = pd.DataFrame()

    total_new = 0
    for path in new_files:
        df = pd.read_csv(path)
        missing = validate_columns(df, {**RAW_SCHEMA, target: "str"})
        if missing:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "file": path.name,
                "status": "rejected",
                "reason": f"missing columns: {missing}",
                "rows": 0,
            }
            append_log(log_path, entry)
            print(f"[ingest] REJECTED {path.name}: {entry['reason']}")
            continue

        train_df = pd.concat([train_df, df], ignore_index=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "file": path.name,
            "status": "ok",
            "rows": len(df),
        }
        append_log(log_path, entry)
        total_new += len(df)
        print(f"[ingest] OK {path.name} — {len(df)} rows")

    # Dedupe by id (last write wins — enables late-arriving corrections)
    id_col = cfg["data"]["id_col"]
    if id_col in train_df.columns:
        before = len(train_df)
        train_df = train_df.drop_duplicates(subset=[id_col], keep="last")
        deduped = before - len(train_df)
        if deduped:
            print(f"[ingest] deduped {deduped} rows on {id_col}")

    train_df.to_csv(train_path, index=False)
    print(f"[ingest] Wrote {len(train_df)} total rows to {train_path}")
    return {"files": len(new_files), "rows": total_new, "total": len(train_df)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ingest(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
