"""
FG Probability - Basic & Feature Plots
=======================================
Requires: numpy, matplotlib, pandas, joblib, scikit-learn
Run from project root:  python fg_basic_plots.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import joblib
from pathlib import Path

# ── Load training data & model ───────────────────────────────────────────────
from src.models.fg_probability import build_training_data, FEATURE_COLS, MODEL_PATH

print("Loading training data...")
df = build_training_data().to_pandas()

print("Loading model...")
artifact = joblib.load(MODEL_PATH)
model    = artifact["model"]

# ── Shared style ─────────────────────────────────────────────────────────────
DARK   = "#0d1f17"
GREEN  = "#1a472a"
GOLD   = "#f5c518"
ACCENT = "#4ade80"
DANGER = "#f87171"
WHITE  = "#f8f8f2"
MUTED  = "#8aaa96"
BLUES  = "#60a5fa"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   "#0f2219",
    "axes.edgecolor":   "#2d5a3d",
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       WHITE,
    "grid.color":       "#1e4a2e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
    "axes.titlecolor":  WHITE,
    "axes.titlesize":   12,
    "axes.titlepad":    12,
})


# ── PLOT 1: Make Rate by Distance Bucket (Bar Chart) ────────────────────────
print("Plotting 1/5: Make rate by distance bucket...")

# Updated buckets to match new model's finer-grained distance buckets
buckets = ["0-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-58", "59-61", "62-64", "65+"]
bucket_rates  = []
bucket_counts = []
for b in buckets:
    subset = df[df["distance_bucket"] == b]["fg_made"]
    bucket_rates.append(subset.mean() if len(subset) > 0 else 0)
    bucket_counts.append(len(subset))

colors = [ACCENT if r > 0.8 else GOLD if r > 0.6 else DANGER for r in bucket_rates]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(buckets, bucket_rates, color=colors, width=0.65, edgecolor="#2d5a3d", linewidth=0.8)

for bar, rate, count in zip(bars, bucket_rates, bucket_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=WHITE)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f"n={count:,}", ha="center", va="bottom", fontsize=7, color="#4a7a5a")

ax.axhline(df["fg_made"].mean(), color=MUTED, linewidth=1.2, linestyle=":", alpha=0.8)
ax.text(len(buckets) - 0.55, df["fg_made"].mean() + 0.01, "overall avg", color=MUTED, fontsize=8)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.set_xlabel("Distance Bucket (yards)", labelpad=8)
ax.set_ylabel("FG Make Rate", labelpad=8)
ax.set_title("FG Make Rate by Distance Bucket  |  NFL 2016–2024")
ax.grid(True, axis="y")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fg_bucket_bar.png", dpi=150, bbox_inches="tight")
plt.show()


# ── PLOT 2: Make Rate by Season (Line Chart) ─────────────────────────────────
print("Plotting 2/5: Make rate by season...")

season_stats = (
    df.groupby("season")["fg_made"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "make_rate", "count": "attempts"})
)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(season_stats["season"], season_stats["make_rate"], color=GOLD, linewidth=2.5, marker="o", markersize=6, zorder=3)
ax1.fill_between(season_stats["season"], season_stats["make_rate"], alpha=0.1, color=GOLD)
ax2.bar(season_stats["season"], season_stats["attempts"], color=ACCENT, alpha=0.2, width=0.6, zorder=1)

ax1.set_xlabel("Season", labelpad=8)
ax1.set_ylabel("FG Make Rate", color=GOLD, labelpad=8)
ax2.set_ylabel("Total Attempts", color=ACCENT, labelpad=8)
ax1.tick_params(axis="y", colors=GOLD)
ax2.tick_params(axis="y", colors=ACCENT)
ax2.spines["right"].set_edgecolor(ACCENT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax1.set_title("FG Make Rate & Attempt Volume by Season  |  NFL 2016–2024")
ax1.grid(True, axis="y")
ax1.set_axisbelow(True)

# Annotate each point
for _, row in season_stats.iterrows():
    ax1.text(row["season"], row["make_rate"] + 0.004, f"{row['make_rate']:.1%}",
             ha="center", fontsize=8, color=GOLD)

plt.tight_layout()
plt.savefig("fg_season_line.png", dpi=150, bbox_inches="tight")
plt.show()


# ── PLOT 3: Attempts Distribution (Histogram) ────────────────────────────────
print("Plotting 3/5: Attempts distribution...")

made   = df[df["fg_made"] == 1]["kick_distance"]
missed = df[df["fg_made"] == 0]["kick_distance"]

fig, ax = plt.subplots(figsize=(10, 5))
bins = range(18, 65, 2)

ax.hist(made,   bins=bins, color=ACCENT, alpha=0.7, label="Made",   edgecolor="#0f2219", linewidth=0.5)
ax.hist(missed, bins=bins, color=DANGER, alpha=0.7, label="Missed", edgecolor="#0f2219", linewidth=0.5)

ax.set_xlabel("Kick Distance (yards)", labelpad=8)
ax.set_ylabel("Number of Attempts", labelpad=8)
ax.set_title("Distribution of FG Attempts by Distance  |  Made vs Missed  |  NFL 2016–2024")
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE)
ax.grid(True, axis="y")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fg_attempts_hist.png", dpi=150, bbox_inches="tight")
plt.show()


# ── PLOT 4: XGBoost Feature Importance ──────────────────────────────────────
print("Plotting 4/5: XGBoost feature importance...")

# Extract feature importances averaged across calibration folds
try:
    importances = np.mean(
        [cc.estimator.feature_importances_
         for cc in model.calibrated_classifiers_],
        axis=0,
    )
except AttributeError:
    # Fallback: single estimator (non-calibrated)
    importances = model.feature_importances_

# Sort by importance descending
sorted_idx   = np.argsort(importances)
sorted_feats = [FEATURE_COLS[i] for i in sorted_idx]
sorted_imps  = importances[sorted_idx]

# Colour: top 5 gold, rest muted green
bar_colors = [GOLD if i >= len(sorted_feats) - 5 else "#2d5a3d" for i in range(len(sorted_feats))]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(sorted_feats, sorted_imps, color=bar_colors, edgecolor="#1e4a2e", linewidth=0.8)

for bar, val in zip(bars, sorted_imps):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, color=WHITE)

ax.set_xlabel("Feature Importance (XGBoost gain, averaged across calibration folds)", labelpad=8)
ax.set_title("Feature Importance  |  XGBoost FG Probability Model\n(higher = more influential in predictions)")
ax.grid(True, axis="x")
ax.set_axisbelow(True)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GOLD, label="Top 5 features"),
                   Patch(facecolor="#2d5a3d", label="Other features")]
ax.legend(handles=legend_elements, facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, loc="lower right")

plt.tight_layout()
plt.savefig("fg_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# ── PLOT 5: Feature Importance Ranked with % of Total ───────────────────────
print("Plotting 5/5: Feature importance breakdown...")

# Normalise importances to % of total so they're easier to interpret
total        = importances.sum()
pct_imps     = (importances / total) * 100
sorted_pct   = pct_imps[sorted_idx]

# Cumulative line
cumulative = np.cumsum(sorted_pct)

fig, ax1 = plt.subplots(figsize=(10, 7))
ax2 = ax1.twiny()

bar_colors2 = [GOLD if i >= len(sorted_feats) - 5 else ACCENT for i in range(len(sorted_feats))]
ax1.barh(sorted_feats, sorted_pct, color=bar_colors2, edgecolor="#1e4a2e", linewidth=0.8, alpha=0.85)
ax2.plot(cumulative, sorted_feats, color=DANGER, linewidth=2, marker="o", markersize=4, zorder=5)

for i, (feat, pct) in enumerate(zip(sorted_feats, sorted_pct)):
    ax1.text(pct + 0.2, i, f"{pct:.1f}%", va="center", fontsize=9, color=WHITE)

ax1.set_xlabel("% of Total Feature Importance", labelpad=8)
ax2.set_xlabel("Cumulative Importance %", color=DANGER, labelpad=8)
ax2.tick_params(axis="x", colors=DANGER)
ax2.set_xlim(0, 105)
ax1.set_title("Feature Importance Breakdown  |  XGBoost FG Probability Model\n(% of total gain · red line = cumulative)")
ax1.grid(True, axis="x")
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fg_feature_importance_pct.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone! PNG files saved to current directory.")
