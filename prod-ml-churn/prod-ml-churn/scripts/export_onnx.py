"""
Export the active model to ONNX and validate it produces identical predictions.

Demonstrates M7 (Model Standardization) concretely:
  - Convert framework model → ONNX
  - Load with ONNX Runtime (no XGBoost/scikit-learn needed at inference time)
  - Prove predictions match the original within tolerance
  - Compare inference latency (framework vs ONNX Runtime)

Usage:
    python scripts/export_onnx.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# ---- ONNX conversion libs ----
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_features, FEATURE_ORDER
from scripts.generate_synthetic_data import generate


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert_xgboost(model, n_features: int) -> bytes:
    """XGBoost → ONNX via onnxmltools.

    XGBoost's ONNX exporter expects feature names in the 'f0, f1, ...' pattern,
    but our training uses semantic names ('contract_risk_score', ...). Rewrite
    the booster's feature names before conversion — this doesn't change the
    model's behavior, only its metadata.
    """
    from onnxmltools import convert_xgboost as _conv
    from onnxmltools.convert.common.data_types import FloatTensorType as XgbFloatTensorType

    booster = model.get_booster()
    generic_names = [f"f{i}" for i in range(n_features)]
    booster.feature_names = generic_names

    initial_types = [("input", XgbFloatTensorType([None, n_features]))]
    onnx_model = _conv(model, initial_types=initial_types)
    return onnx_model.SerializeToString()


def convert_sklearn_pipeline(scaler, model, n_features: int) -> bytes:
    """
    LogReg baseline is (scaler, model) — convert as a pipeline so preprocessing
    is baked into the ONNX graph and there's a single artifact to ship.
    """
    from sklearn.pipeline import Pipeline
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType

    pipeline = Pipeline([("scaler", scaler), ("model", model)])
    initial_types = [("input", SkFloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types,
                                  target_opset=15)
    return onnx_model.SerializeToString()


def framework_predict_proba(bundle, X: np.ndarray) -> np.ndarray:
    """Same logic as src.train.predict_proba — kept local to avoid dep."""
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(X)[:, 1]


def onnx_predict_proba(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Extract positive-class probability from ONNX output."""
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X.astype(np.float32)})
    # Convention:
    #   outputs[0] = predicted label
    #   outputs[1] = probabilities (dict-of-array for sklearn, tensor for xgboost)
    proba_out = outputs[1]
    if isinstance(proba_out, list):
        # sklearn wraps per-row as [{0: p0, 1: p1}, ...]
        return np.array([row[1] for row in proba_out])
    else:
        # xgboost returns a [n, 2] tensor
        return proba_out[:, 1]


def bench(fn, n_runs: int = 100) -> float:
    """Return average ms per call."""
    # warmup
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / n_runs


def main() -> int:
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    active_path = models_dir / "active.joblib"

    if not active_path.exists():
        print(f"[export_onnx] No active model at {active_path}. Train first.",
              file=sys.stderr)
        return 1

    bundle = joblib.load(active_path)
    meta = bundle["meta"]
    model_type = meta["model_type"]
    print(f"[export_onnx] Active model: {model_type}  v{meta['version']}")

    n_features = len(FEATURE_ORDER)

    # ---- Convert ----
    print(f"[export_onnx] Converting to ONNX...")
    if model_type == "xgboost":
        onnx_bytes = convert_xgboost(bundle["model"], n_features)
    elif model_type == "logistic_regression":
        onnx_bytes = convert_sklearn_pipeline(bundle["scaler"], bundle["model"],
                                               n_features)
    else:
        print(f"[export_onnx] Unsupported model type: {model_type}",
              file=sys.stderr)
        return 2

    onnx_path = models_dir / "active.onnx"
    onnx_path.write_bytes(onnx_bytes)
    size_kb = len(onnx_bytes) / 1024
    print(f"[export_onnx] Wrote {onnx_path}  ({size_kb:,.1f} KB)")

    # Framework artifact size for comparison
    fw_size_kb = active_path.stat().st_size / 1024
    print(f"[export_onnx] Framework artifact: {fw_size_kb:,.1f} KB  "
          f"({fw_size_kb / size_kb:.1f}× larger)")

    # ---- Parity check: same predictions? ----
    print("\n[export_onnx] Parity check (1000 synthetic rows)...")
    df = generate(1000, seed=777)
    X = build_features(df).values

    fw_proba = framework_predict_proba(bundle, X)
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    onnx_proba = onnx_predict_proba(sess, X)

    max_diff = np.max(np.abs(fw_proba - onnx_proba))
    mean_diff = np.mean(np.abs(fw_proba - onnx_proba))
    print(f"  max |framework - onnx|  = {max_diff:.6e}")
    print(f"  mean |framework - onnx| = {mean_diff:.6e}")

    tolerance = 1e-4
    parity_ok = max_diff < tolerance
    print(f"  parity within {tolerance}: {'PASS' if parity_ok else 'FAIL'}")

    # ---- Latency comparison ----
    print("\n[export_onnx] Latency comparison (single record, 100 runs)...")
    X1 = X[:1]
    fw_ms = bench(lambda: framework_predict_proba(bundle, X1))
    onnx_ms = bench(lambda: onnx_predict_proba(sess, X1))
    speedup = fw_ms / onnx_ms if onnx_ms > 0 else float("inf")

    print(f"  framework:    {fw_ms:.3f} ms/call")
    print(f"  onnxruntime:  {onnx_ms:.3f} ms/call  ({speedup:.2f}× vs framework)")

    # ---- Save report ----
    report = {
        "model_version": meta["version"],
        "model_type": model_type,
        "n_features": n_features,
        "artifact_size_kb": {
            "framework_joblib": round(fw_size_kb, 1),
            "onnx": round(size_kb, 1),
        },
        "parity": {
            "max_abs_diff": float(max_diff),
            "mean_abs_diff": float(mean_diff),
            "tolerance": tolerance,
            "passed": bool(parity_ok),
        },
        "latency_ms_per_call": {
            "framework": round(fw_ms, 3),
            "onnxruntime": round(onnx_ms, 3),
            "speedup_x": round(speedup, 2),
        },
    }
    report_path = Path(cfg["paths"]["eval_dir"]) / "onnx_export.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[export_onnx] Report: {report_path}")

    return 0 if parity_ok else 3


if __name__ == "__main__":
    sys.exit(main())
