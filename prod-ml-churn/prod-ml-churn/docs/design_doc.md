# Design Document — Mini Production ML System for Customer Churn

**Author:** Yogesh Vasu (BITS Pilani MSc DS&AI) &nbsp;|&nbsp; **Track:** B — Batch/Pipeline &nbsp;|&nbsp; **Use case:** Binary classification (churn) &nbsp;|&nbsp; **Repo:** `prod-ml-churn/`

> Module references (M1–M11) are cited inline to make the mapping to the course explicit.

---

## 1. Problem framing, users, and constraints (M1)

**Problem.** Predict which subscribers of a telecommunications service are most likely to churn (cancel) in the next billing cycle, so the retention team can prioritize outreach to a limited number of at-risk customers.

**Users.**
- **Primary:** the retention/marketing operations team, which consumes a daily ranked CSV of customers with churn probabilities.
- **Secondary:** frontline retention agents, who occasionally need to score a single customer on demand from a CRM.

**Inputs.** One row per customer with 20 fields (demographics, tenure, contract, service subscriptions, billing). See `src/features.RAW_SCHEMA` for the enforced contract.

**Outputs.** A churn probability in `[0, 1]`, a hard label at threshold 0.50, and the `model_version` that produced the score (so downstream systems can audit which model made a decision — M11).

**Production constraints (M1).** This is intentionally a *batch-first* workload:
- **Latency:** the primary path is a nightly batch job; sub-day freshness is acceptable. The API path has a soft SLA of *p95 < 50 ms* for single-record lookups.
- **Throughput:** the batch job must comfortably score 100 K–500 K customers within a nightly window.
- **Cost:** batch scoring on a single VM is orders of magnitude cheaper than an always-on API for the same population; the API is provisioned small for occasional lookups only (M8).
- **Reliability:** if the model fails a promotion gate, the previous `models/active.joblib` continues serving — no automatic rollback needed because promotion is *pull*, not *push*.

This engineering framing distinguishes model engineering from pure data science (M1): the accuracy of the model matters, but so do the pipeline, the artifact provenance, the monitoring loop, and the cost of the deployed system.

---

## 2. Data and feature design (M9, M10)

**Source and schema.** A synthetic generator (`scripts/generate_synthetic_data.py`) produces records that match the IBM Telco Customer Churn schema exactly, so the pipeline is drop-in compatible with the public dataset. The generator injects a realistic learnable signal (short tenure + month-to-month + fiber-optic + electronic check → higher churn probability) with Gaussian noise, yielding a class balance of ~34% churners.

**Feature engineering — seven non-trivial features.** All are computed in the single shared module `src/features.py`:

| # | Feature | Type | Rationale |
|---|---|---|---|
| 1 | `charges_per_tenure_month` | ratio (aggregation over history) | Normalizes lifetime spend by tenure — separates high-spend loyal customers from high-spend new ones |
| 2 | `monthly_to_total_ratio` | ratio | Proxy for recency; new customers have a ratio near 1.0 |
| 3 | `num_addon_services` | count aggregation over 6 columns | Bundle depth is a strong retention signal |
| 4 | `contract_risk_score` | ordinal encoding of `Contract` | Month-to-month = 3 (highest risk) → Two-year = 1 |
| 5 | `payment_auto_flag` | binary derived | Auto-pay correlates with lower churn |
| 6 | `is_high_value` | binary threshold (`MonthlyCharges > 70`) | Distinguishes premium segment for retention prioritization |
| 7 | `tenure_bucket_ord` | ordinal binning | Non-linear tenure effect (0–12 vs 13–24 vs 25–48 vs 49+ months) |

Together with the raw numeric fields and fixed-vocab one-hots, the output is a 19-column deterministic feature matrix in a fixed column order (`FEATURE_ORDER`).

**Offline vs online features (M9).** All seven features above are *point-in-time* — they can be computed from the customer's current row alone, so they are both offline (used at training) and online (used at scoring) with identical code. In a fuller production system, aggregations like *"call-center contacts in the last 30 days"* would live in a feature store (Feast, Tecton, or in-house) with an offline snapshot for training and a low-latency online view for scoring. For this project, the shared Python module is the miniature feature store.

**Training–serving skew defense (M9).** This is the single most important design decision. Both `src/train.py` (offline) and `src/serve.py` + `src/batch_score.py` (online/batch) import `build_features` from the *same* module. The unit test `test_single_vs_batch_identical` asserts that scoring one row alone produces byte-identical features to scoring it as part of a batch — this pins the skew defense to CI. Unknown categorical values collapse deterministically to the base category (rather than crashing or silently producing a NaN downstream).

**Ingestion pipeline (M10).** `src/ingest.py` performs micro-batch ingestion:
1. Watches `data/raw/` for new daily CSV drops.
2. Validates schema (rejects files with missing columns and logs the reason).
3. Merges into `data/training/training.csv`, deduplicating by `customerID` with *last-write-wins* semantics — this handles late-arriving corrections.
4. Appends a structured JSON-lines log entry (`{ts, file, status, rows}`) to `artifacts/ingest_log.jsonl` for auditability.

The conceptual mapping to production tooling (M10): **Kafka** would replace CSV drops as the message bus; **Spark Structured Streaming** or **Beam** would replace the pandas merge for horizontal scaling; **Flink** would handle stream-first workloads. Freshness, completeness, and schema-evolution semantics all live in this layer — our ingest checks are the miniature version.

---

## 3. Model choice, evaluation, and promotion (M1, M4, M6)

**Two candidates.**
- **Baseline:** `LogisticRegression` with L2 regularization, `class_weight='balanced'`, features scaled with `StandardScaler`. Cheap, interpretable, calibration-friendly.
- **Candidate:** `XGBoost` (`hist` tree method, 200 estimators, depth 5) with `scale_pos_weight` set to the class ratio. Better at nonlinear interactions.

**Metric choice.** The primary metric is **ROC-AUC** because the churn label is imbalanced and the retention team uses a ranked list (they act on the top-K customers regardless of a fixed threshold). We also report **PR-AUC** (more informative under imbalance), **F1/precision/recall** at threshold 0.50, and **recall@20% of the ranked list** — a direct business metric answering *"if we call our top 20% highest-risk customers, what fraction of actual churners do we reach?"*.

**Actual results (from `artifacts/eval/latest.json`).**

| Metric (val) | Baseline (LR) | Candidate (XGBoost) |
|---|---:|---:|
| ROC-AUC | 0.8107 | **0.8046** |
| PR-AUC | 0.6797 | **0.6834** |
| Recall @ 20% | 0.4175 | 0.4126 |
| F1 @ 0.5 | 0.6576 | **0.6623** |

**Promotion rule (M6).** Encoded in code (`train.py::promote`):

```
Promote candidate iff
    candidate.roc_auc ≥ 0.80          # absolute quality floor
    AND candidate.roc_auc ≥ baseline.roc_auc − 0.01   # no meaningful regression
```

On the current data snapshot both models cross the 0.80 floor and XGBoost is promoted (within the 0.01 tolerance on AUC while winning on PR-AUC and F1). If the candidate fails, the baseline is written to `models/active.joblib` instead — the system is never left without an active model.

**Artifact tracking (M4).** Each training run writes:
- `models/model_<version>.joblib` — versioned model (`version` = UTC timestamp).
- `models/active.joblib` — the current "production" pointer.
- `models/reference_stats.json` — per-feature `describe()` used as the drift reference.
- `artifacts/eval/eval_<version>.json` and `latest.json` — full metrics report with `training_data_hash` (12-char MD5 of the training frame) so we can trace *which model was trained on which snapshot*. This is the miniature version of MLflow-style lineage tracking.

---

## 4. Serving and inference pattern (M2, M3, M8)

**Choice: hybrid, batch-primary (M2).**

The decision uses the M2 framework:
- Is a human waiting? No — retention outreach is planned weekly.
- What latency is acceptable? Sub-day for the ranked list; ~50 ms for occasional single-record lookups.
- Is the use case naturally batch or streaming? Naturally batch — the population is finite and known, and outreach cadence is discrete.

We therefore run **batch scoring as the primary inference path** (`src/batch_score.py`) and expose a **thin FastAPI service** for occasional online lookups (`src/serve.py`).

**Serving architecture (M3).** A single **microservice** wraps the model behind a **synchronous REST** API — pragmatic for low QPS. gRPC would be an option if this service ever became internal-only and latency-critical; serverless (Lambda/Cloud Run) would be a good match if traffic were spiky and low-average. A `Dockerfile` containerizes the API for portability.

**Contract validation.** The API's request schema (`CustomerRecord`) mirrors `RAW_SCHEMA`; malformed requests get HTTP 422 from Pydantic before touching the model. Batch requests are bounded at 500 records to prevent a single request from starving the server.

**Measured performance (real numbers from this repo).**
- **API latency** (200 requests to `/predict` on localhost): avg 12.75 ms, **p50 12.17 ms, p95 14.17 ms, p99 16.30 ms**. Comfortably under the 50 ms SLA.
- **Batch throughput**: 100 rows → 7 K rows/sec (fixed overhead dominates); **50 000 rows → 273 K rows/sec** (vectorized pandas + XGBoost). A 500 K-customer nightly batch would finish in ~2 s of compute, well within the window.

**Cost implication (M8).** The batch job on a single 2-vCPU VM costs roughly $0.10/day; a 24×7 API sized for the same throughput would cost 20–50×. The hybrid pattern captures both use cases at the cheaper cost profile.

---

## 5. Data pipeline and retraining strategy (M6, M10)

**Closed loop.**
```
new daily CSVs  →  ingest  →  training table  →  train (baseline vs candidate)
                     ↓                                       ↓
              ingest_log.jsonl                         active model
                                                            ↓
                                       batch_score / API  →  predictions
                                                            ↓
                            monitoring (DQ + PSI drift) → retraining trigger
                                                            ↑
                                                   (labels arrive later)
```

**Retraining triggers (M6).** Encoded in `src/retraining.py::should_retrain` as a pure function of three signals (any one fires → retrain):

1. **Scheduled** — `days_since_last_train ≥ 30`
2. **Drift** — `n_features_in_PSI_alert ≥ 3`
3. **Performance** — `recent_rolling_auc < 0.75`

A cron or Airflow DAG would call this function daily and kick off `python -m src.train` on a non-zero exit code. A human sign-off gate (M6) sits between "candidate promoted in staging" and "candidate replaces `active.joblib` in production" — the promotion rule provides evidence for that reviewer.

---

## 6. Monitoring plan and alerts (M5, M11)

**Three tiers of metrics (M5).**

| Tier | Examples | Owner | Alerting example |
|---|---|---|---|
| **Infra** | latency p95/p99, error rate, container OOM | SRE | pager if p95 > 100 ms for 5 min |
| **Data / feature** | null rate per column, out-of-range counts, new categorical values, **PSI drift** | ML platform team | Slack warning at PSI > 0.10; PagerDuty at PSI > 0.25 |
| **Model / business** | rolling AUC on labeled feedback, prediction rate, retention uplift vs holdout | ML + business | Weekly dashboard; email if rolling AUC drops > 5 pts |

**Implemented drift check.** `src/monitoring.py` computes **Population Stability Index (PSI)** per configured feature between the training reference distribution (saved at training time in `reference_stats.json`) and any recent batch. Alerting bands follow the standard convention (<0.10 = ok, 0.10–0.25 = warn, >0.25 = alert). A non-zero exit code lets a cron treat monitoring as CI.

**Data quality check.** Before drift, the same script validates schema conformance, per-column null rates, numeric range bounds, and known-categorical vocab. This catches upstream problems (missing columns, silent renames, new payment methods) *before* they present as drift.

**Demonstrated incident scenario.** Running `python scripts/generate_synthetic_data.py --drift` produces a batch with shifted `InternetService` and `MonthlyCharges` distributions (a simulated pricing change). Monitoring correctly flags **4 features in alert** (PSI up to 3.31 on `num_addon_services`), and the retraining trigger fires on both the drift signal and — separately — a low simulated recent AUC.

**Failure playbook.** An upstream schema change (say a new `PaymentMethod` value) would be caught by the DQ check as "new categorical values." Response: (1) alert on-call, (2) *keep serving the current model* — the feature module collapses unknown categories to base, so predictions remain valid if slightly stale, (3) file a data-team ticket, (4) once the mapping is updated, retrain and promote.

---

## 7. Security, privacy, responsibility (M11)

- **PII minimization.** The pipeline stores only `customerID` (a hash-like handle), not names/emails/addresses. In production the customer ID would be an opaque, non-reversible token.
- **Access control.** `models/`, `data/`, and `artifacts/` are separate directories precisely so IAM policies can differ (data engineers write to `data/raw/`; only training writes to `models/`).
- **Segmented fairness.** The evaluation harness is easy to extend to compute per-segment AUC (e.g., by `SeniorCitizen`, by `Contract` type) — the same pattern would catch fairness regressions across the categories the model is *not* explicitly optimized for.
- **Explainability.** XGBoost surfaces per-record feature importances (`predict(pred_contribs=True)` in a follow-up). The API response already includes `model_version` for downstream audit trail linking.
- **Audit trails.** Ingest logs, versioned model artifacts, `training_data_hash`, and prediction-run summaries together form a reproducible chain: *given a prediction, we can trace it to a model version, to a training data snapshot, to the ingest events that produced that snapshot.*

---

## 8. Trade-offs, limitations, future work (M7, M8)

**Trade-offs made.**
- *Batch-primary vs streaming-first.* Batch is cheaper and simpler; if outreach became event-driven (e.g., cancel-button click), a streaming path would be added.
- *XGBoost vs logistic regression.* XGBoost wins on PR-AUC/F1 within margin, but LR is more interpretable and calibrated. The promotion rule allows either to serve.
- *Filesystem "registry" vs full registry.* Simpler to reason about; a real registry (MLflow, Vertex AI, SageMaker Model Registry) would add stage transitions, approvers, and lineage graphs.

**Limitations.**
- No feature store — online features are limited to point-in-time fields on the incoming record.
- The API loads the model at process start; a model update requires a restart. A blue/green deploy or a periodic reload endpoint would fix this.
- No canary/shadow traffic on new candidates.

**Future work (M7, M8).**
- **ONNX export.** `xgboost` → `onnx` conversion via `onnxmltools`, served through **ONNX Runtime** — typically a 2–4× latency improvement and smaller memory footprint, useful if API QPS grows.
- **Quantization / distillation (M7).** Distill XGBoost predictions into a linear model for even faster serving on the long tail of low-risk customers, gated on maintained AUC.
- **Cost levers (M8).** Move nightly batch onto spot/preemptible instances (~70% cheaper); autoscale the API on QPS; consider serverless if traffic is spiky.
- **Streaming path.** Add a Kafka topic for cancellation-intent events, score on the fly, and merge back into the daily list.
- **Shadow deployment** of any promoted candidate before it replaces `active.joblib`, comparing prediction distributions to the incumbent for a soak period.
