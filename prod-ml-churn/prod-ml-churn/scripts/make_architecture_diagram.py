"""Generate architecture diagram PNG for the design doc."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Layout parameters
FIG_W, FIG_H = 16, 10

# Color palette by layer
COL_DATA    = "#4C7CB0"
COL_FEATURE = "#7B9CC7"
COL_TRAIN   = "#C89432"
COL_REG     = "#8B6BB1"
COL_SERVE   = "#4A9367"
COL_MON     = "#B8524A"
COL_TEXT    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, label, color, fontsize=10, sublabel=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.2, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color=COL_TEXT)
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                ha="center", va="center", fontsize=fontsize - 2, color=COL_TEXT, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color=COL_TEXT)


def arrow(x1, y1, x2, y2, label=None, color="#333", style="-", lw=1.4):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=8,head_width=6",
        color=color, linewidth=lw, linestyle=style,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                ha="center", va="center", fontsize=8, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


# ---------------- Title ----------------
ax.text(8, 9.55, "Churn Prediction — Production ML Architecture (Track B: Batch/Pipeline)",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(8, 9.15, "M1–M11 mapped: shared feature module, batch primary + API secondary, drift+trigger closed loop",
        ha="center", va="center", fontsize=9.5, style="italic", color="#555")

# ---------------- Layer 1: Data sources ----------------
box(0.3, 7.2, 2.6, 1.3, "Operational\nSystems", COL_DATA, sublabel="CRM · Billing")
box(3.2, 7.2, 2.6, 1.3, "Daily CSV drop", COL_DATA, sublabel="data/raw/*.csv  (M10)")

# ---------------- Layer 2: Ingestion ----------------
box(6.2, 7.2, 3.2, 1.3, "Ingestion", COL_FEATURE, sublabel="src/ingest.py  (M10)")
arrow(2.9, 7.85, 3.2, 7.85)
arrow(5.8, 7.85, 6.2, 7.85)

# ---------------- Layer 3: Training data + Feature module ----------------
box(9.8, 7.2, 2.6, 1.3, "Training Table", COL_FEATURE, sublabel="data/training/training.csv")
arrow(9.4, 7.85, 9.8, 7.85)

box(12.7, 7.2, 3.0, 1.3, "Feature Module\n(SHARED)", COL_FEATURE, sublabel="src/features.py  (M9)")

# ---------------- Layer 4: Training ----------------
box(9.8, 5.2, 2.6, 1.3, "Training Pipeline", COL_TRAIN, sublabel="baseline + candidate  (M4)")
box(12.7, 5.2, 3.0, 1.3, "Offline Eval\n+ Promotion Rule", COL_TRAIN, sublabel="AUC / PR-AUC / R@20  (M6)")
arrow(11.1, 7.2, 11.1, 6.5)
arrow(12.4, 5.85, 12.7, 5.85)
# Feature module feeds training
arrow(13.5, 7.2, 12.7, 6.5, style="--", color="#555")
ax.text(13.7, 6.85, "offline path", fontsize=7.5, color="#555", style="italic")

# ---------------- Layer 5: Registry ----------------
box(9.8, 3.4, 5.9, 1.2, "Model Registry (filesystem)", COL_REG,
    sublabel="models/active.joblib  ·  reference_stats.json  ·  eval/latest.json")
arrow(12.7, 5.2, 12.7, 4.6)
ax.text(12.9, 4.9, "promoted", fontsize=8, color="#555")

# ---------------- Layer 6: Serving ----------------
box(0.3, 3.4, 3.5, 1.4, "Batch Scoring\n(PRIMARY)", COL_SERVE, sublabel="src/batch_score.py  (M2·M8)")
box(4.1, 3.4, 3.5, 1.4, "FastAPI /predict\n(SECONDARY)", COL_SERVE, sublabel="src/serve.py  (M3)")

# Feature module also feeds serving (skew defense)
arrow(13.5, 7.2, 6.0, 4.8, style="--", color="#555")
ax.text(9.3, 6.15, "online path (same code)", fontsize=8, color="#555", style="italic")

# Registry -> serving
arrow(9.8, 4.0, 7.6, 4.0)
arrow(9.8, 4.0, 3.8, 4.0)
ax.text(8.7, 4.15, "load model", fontsize=8, color="#333")

# Batch output
box(0.3, 1.4, 3.5, 1.2, "Predictions CSV", COL_SERVE, sublabel="artifacts/predictions/  → CRM")
arrow(2.05, 3.4, 2.05, 2.6)

# API output
box(4.1, 1.4, 3.5, 1.2, "JSON response", COL_SERVE, sublabel="retention agents / apps")
arrow(5.85, 3.4, 5.85, 2.6)

# ---------------- Layer 7: Monitoring & Retraining ----------------
box(8.0, 1.4, 3.7, 1.4, "Monitoring", COL_MON,
    sublabel="DQ + PSI drift  (M5·M10·M11)")
box(12.0, 1.4, 3.7, 1.4, "Retraining Trigger", COL_MON,
    sublabel="schedule · drift · perf  (M6)")

# Predictions -> monitoring
arrow(3.8, 2.0, 8.0, 2.0)
ax.text(5.9, 2.15, "recent batch stats", fontsize=8, color="#333")

# Monitoring -> trigger
arrow(11.7, 2.1, 12.0, 2.1)
# Feedback loop: trigger -> training
arrow(13.8, 2.8, 11.1, 5.2, style="--", color=COL_MON, lw=1.6)
ax.text(12.5, 4.0, "retrain when\ntriggered", fontsize=8, color=COL_MON, style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

# ---------------- Legend ----------------
legend_items = [
    ("Data",              COL_DATA),
    ("Feature/Ingest",    COL_FEATURE),
    ("Training",          COL_TRAIN),
    ("Registry",          COL_REG),
    ("Serving",           COL_SERVE),
    ("Monitor/Retrain",   COL_MON),
]
lx, ly = 0.3, 0.25
for i, (lbl, col) in enumerate(legend_items):
    x = lx + i * 2.55
    box_patch = FancyBboxPatch((x, ly), 0.35, 0.35, boxstyle="round,pad=0.02",
                                linewidth=0.8, edgecolor="#333", facecolor=col)
    ax.add_patch(box_patch)
    ax.text(x + 0.45, ly + 0.17, lbl, fontsize=9, va="center")

plt.tight_layout()
out_path = Path("docs/architecture.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Wrote {out_path}")
