"""Measure API latency using a persistent HTTP session (fast on Windows)."""
import argparse
import statistics
import sys
import time

import requests


SAMPLE = {
    "customerID": "C-BENCHMARK",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5, "TotalCharges": 1074.0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    # Persistent session — reuses one TCP connection instead of opening a new one per request
    session = requests.Session()

    # Warmup
    for _ in range(5):
        session.post(f"{args.url}/predict", json=SAMPLE, timeout=10)

    latencies_ms = []
    errors = 0
    t_start = time.perf_counter()
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = session.post(f"{args.url}/predict", json=SAMPLE, timeout=10)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if r.status_code != 200:
            errors += 1
    total = time.perf_counter() - t_start

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]

    print("=" * 40)
    print(f"Requests:    {args.n}")
    print(f"Errors:      {errors}")
    print(f"Wall clock:  {total:.2f}s ({args.n / total:.1f} req/sec)")
    print(f"Avg latency: {statistics.mean(latencies_ms):.2f} ms")
    print(f"p50:         {p50:.2f} ms")
    print(f"p95:         {p95:.2f} ms")
    print(f"p99:         {p99:.2f} ms")
    print("=" * 40)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())