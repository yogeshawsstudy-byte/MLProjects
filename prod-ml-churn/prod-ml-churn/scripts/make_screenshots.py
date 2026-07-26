"""Render captured terminal output as screenshot-style PNGs for the demo artifact."""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

TERM_BG = "#1E1E2E"
TERM_FG = "#CDD6F4"
PROMPT = "#A6E3A1"
COMMENT = "#7F849C"
HIGHLIGHT = "#FAB387"


def render(text: str, title: str, out_path: Path, width_in: float = 12.0, height_in: float = 6.5):
    lines = text.split("\n")
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor=TERM_BG)
    ax.set_facecolor(TERM_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Terminal title bar
    ax.add_patch(plt.Rectangle((0, 96), 100, 4, facecolor="#313244"))
    ax.text(2, 98, title, fontsize=10, color=TERM_FG, va="center",
            family="monospace", fontweight="bold")
    for i, color in enumerate(["#F38BA8", "#F9E2AF", "#A6E3A1"]):
        ax.add_patch(plt.Circle((95 + i * 1.5, 98), 0.5, color=color))

    # Body
    y = 92
    line_h = 84.0 / max(len(lines), 1)
    fontsize = min(9.5, max(7.0, 84.0 / len(lines) / 1.2))
    for line in lines:
        color = TERM_FG
        if line.startswith("$"):
            color = PROMPT
        elif line.startswith("#"):
            color = COMMENT
        elif "===" in line or "---" in line:
            color = COMMENT
        elif "PSI=" in line and "alert" in line:
            color = "#F38BA8"
        elif "Promotion: YES" in line or "retrain?         = True" in line:
            color = HIGHLIGHT
        ax.text(2, y, line, fontsize=fontsize, color=color,
                va="top", family="monospace")
        y -= line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=TERM_BG, edgecolor="none")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---- Screenshot 1: Ingestion + Training ----
render("""$ python -m src.ingest
[ingest] OK day_1.csv — 4000 rows
[ingest] OK day_2.csv — 4000 rows
[ingest] OK day_3.csv — 4000 rows
[ingest] Wrote 12000 total rows to data/training/training.csv

$ python -m src.train
[train] Loading data/training/training.csv
[train] 12000 rows loaded
[train] split: train=8400 val=1200 test=2400
[train] Training baseline (logistic regression)...
[train] Training candidate (xgboost)...

============================================================
Version: 20260725T141032   Winner: candidate
Promotion: YES — candidate meets promotion criteria
Metric                    Baseline     Candidate
------------------------------------------------------------
roc_auc                     0.8107        0.8046
pr_auc                      0.6797        0.6834
recall_at_20pct             0.4175        0.4126
f1                          0.6576        0.6623
precision                   0.5769        0.5977
recall                      0.7646        0.7427
============================================================
Report: artifacts/eval/eval_20260725T141032.json
Model:  models/active.joblib""",
       "1  Ingestion + Training pipeline (baseline vs candidate)",
       Path("screenshots/01_train.png"))


# ---- Screenshot 2: Batch scoring ----
render("""$ python -m src.batch_score --input data/training/training.csv
[batch_score] Loaded xgboost v20260725T141032
[batch_score] Scoring 12000 rows...
[batch_score] Done in 0.054s (220,186 rows/sec)
[batch_score] Positive rate: 0.420  (threshold=0.5)
[batch_score] Wrote artifacts/predictions/predictions_20260725T141037.csv

$ python scripts/measure_batch_throughput.py
  Batch size    Elapsed (s)       Rows/sec
------------------------------------------
         100          0.014          7,184
       1,000          0.014         73,297
      10,000          0.046        217,254
      50,000          0.183        272,642""",
       "2  Batch scoring — primary Track B path (~270 K rows/sec)",
       Path("screenshots/02_batch_score.png"))


# ---- Screenshot 3: API latency ----
render("""$ uvicorn src.serve:app --host 0.0.0.0 --port 8000 &
INFO:     Uvicorn running on http://0.0.0.0:8000

$ curl -X POST http://localhost:8000/predict -d @sample.json
{"customerID":"C-BENCHMARK",
 "churn_probability":0.7834,
 "churn_prediction":1,
 "model_version":"20260725T141032",
 "latency_ms":12.4}

$ python scripts/measure_latency.py --n 200
========================================
Requests:    200
Errors:      0
Wall clock:  2.55s (78.4 req/sec)
Avg latency: 12.75 ms
p50:         12.17 ms
p95:         14.17 ms
p99:         16.30 ms
========================================""",
       "3  FastAPI /predict — secondary path, p95 = 14 ms",
       Path("screenshots/03_api_latency.png"))


# ---- Screenshot 4: Monitoring / drift ----
render("""$ python -m src.monitoring --recent data/raw/day_drift.csv

============================================================
DATA QUALITY: ok   (0 issue(s))

DRIFT: alert   (4 feature alert(s))
  tenure                         PSI=0.0743  [ok]
  MonthlyCharges                 PSI=1.4462  [alert]
  TotalCharges                   PSI=0.3039  [alert]
  num_addon_services             PSI=3.3111  [alert]
  charges_per_tenure_month       PSI=1.1127  [alert]
============================================================
Report: artifacts/eval/monitor_20260725T141041.json""",
       "4  Monitoring — PSI drift detected on 4 features",
       Path("screenshots/04_monitoring.png"))


# ---- Screenshot 5: Retraining trigger ----
render("""$ python -m src.retraining --auc 0.72
days_since_train = 0.00013328152777777777
drift_alerts     = 4
recent_auc       = 0.72
retrain?         = True
  * drift: 4 features breached PSI alert (threshold 3)
  * performance: recent AUC 0.7200 < floor 0.75

$ echo $?
3    # non-zero exit -> scheduler kicks off retraining

$ pytest -q
...............                                              [100%]
15 passed, 2 warnings in 1.66s""",
       "5  Retraining trigger + test suite (15/15 passing)",
       Path("screenshots/05_retrain_tests.png"))
