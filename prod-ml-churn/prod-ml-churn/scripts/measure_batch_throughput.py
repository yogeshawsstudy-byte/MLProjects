"""
Measure batch scoring throughput across different batch sizes.

Usage:
    python scripts/measure_batch_throughput.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic_data import generate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.train import predict_proba  # noqa: E402


def main() -> int:
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    bundle = joblib.load(Path(cfg["paths"]["models_dir"]) / "active.joblib")

    print(f"{'Batch size':>12} {'Elapsed (s)':>14} {'Rows/sec':>14}")
    print("-" * 42)
    for n in [100, 1_000, 10_000, 50_000]:
        df = generate(n, seed=n)
        t0 = time.perf_counter()
        X = build_features(df)
        _ = predict_proba(bundle, X)
        elapsed = time.perf_counter() - t0
        print(f"{n:>12,} {elapsed:>14.3f} {n / elapsed:>14,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
