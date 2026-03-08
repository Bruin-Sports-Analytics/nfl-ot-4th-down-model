"""
FG Probability - Basic & Feature Plots
=======================================
Requires: numpy, matplotlib, pandas, joblib, scikit-learn
Run from project root:  python fg_basic_plots.py

Outputs saved to: outputs/
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ── Load training data & model ───────────────────────────────────────────────
from src.models.fg_probability import build_training_data, FEATURE_COLS, MODEL_PATH

print("Loading training data...")
df = build_training_data().to_pandas()

print("Loading model...")
artifact = joblib.load(MODEL_PATH)
model    = artifact["model"]

# Recreate the same train/test split used during training
clean        = df[FEATURE_COLS + ["fg_made", "season"]].dropna()
X_all        = clean[FEATURE_COLS].to_numpy()
y_all        = clean["fg_made"].to_numpy()
seasons_arr  = clean["season"].to_numpy()
X_train, X_test, y_train, y_test, _, _ = train_test_split(
    X_all, y_all, seasons_arr,
    test_size=0.2, random_state=42, stratify=seasons_arr,
)

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
DARK   = "#0d1f17"
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


# ── PLOT 1: Make Rate by Distance Bucket ────────────────────────────────────
print("Plotting 1/6: Make rate by distance bucket...")

buckets = ["0-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-58", "59-61", "62-64", "65+"]
bucket_rates  = []
bucket_counts = []
for b in buckets:
    subset = df[df["distance_bucket"] == b]["fg_made"]
    bucket_rates.append(subset.mean() if len(subset) > 0 else 0)
    bucket_counts.append(len(subset))

colors = [ACCENT if r > 0.8 else GOLD if r > 0.6 else DANGER for r in bucket_rates]

fig, ax = plt.subplots(figsize=(12, 5.5))
bars = ax.bar(buckets, bucket_rates, color=colors, width=0.65, edgecolor="#2d5a3d", linewidth=0.8)

for bar, rate, count in zip(bars, bucket_rates, bucket_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.013,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=WHITE)
        ax.text(bar.get_x() + bar.get_width() / 2, 0.022,
                f"n={count:,}", ha="center", va="bottom", fontsize=7.5, color="#4a7a5a")

overall = df["fg_made"].mean()
ax.axhline(overall, color=MUTED, linewidth=1.5, linestyle=":", alpha=0.9)
ax.text(len(buckets) - 0.48, overall + 0.012, f"Overall avg  {overall:.1%}",
        color=MUTED, fontsize=8.5, ha="right")

ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.set_xlabel("Distance Bucket (yards)", labelpad=8)
ax.set_ylabel("FG Make Rate", labelpad=8)
ax.set_title("FG Make Rate by Distance Bucket  |  NFL 2016–2024  (unblocked kicks only)")

legend_elements = [
    mpatches.Patch(facecolor=ACCENT, label="> 80%  (high)"),
    mpatches.Patch(facecolor=GOLD,   label="60–80%  (medium)"),
    mpatches.Patch(facecolor=DANGER, label="< 60%  (low)"),
]
ax.legend(handles=legend_elements, facecolor="#0f2219", edgecolor="#2d5a3d",
          labelcolor=WHITE, loc="upper right")
ax.grid(True, axis="y")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "fg_bucket_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_bucket_bar.png'}")


# ── PLOT 2: Make Rate by Season ──────────────────────────────────────────────
print("Plotting 2/6: Make rate by season...")

season_stats = (
    df.groupby("season")["fg_made"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "make_rate", "count": "attempts"})
)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax2.bar(season_stats["season"], season_stats["attempts"],
        color=ACCENT, alpha=0.18, width=0.7, zorder=1)
ax1.plot(season_stats["season"], season_stats["make_rate"],
         color=GOLD, linewidth=2.5, marker="o", markersize=7, zorder=3)
ax1.fill_between(season_stats["season"], season_stats["make_rate"],
                 alpha=0.1, color=GOLD)

ax1.set_xlabel("Season", labelpad=8)
ax1.set_ylabel("FG Make Rate", color=GOLD, labelpad=8)
ax2.set_ylabel("Total Attempts", color=ACCENT, labelpad=8)
ax1.tick_params(axis="y", colors=GOLD)
ax2.tick_params(axis="y", colors=ACCENT)
ax2.spines["right"].set_edgecolor(ACCENT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax1.set_ylim(0.80, 0.92)
ax1.set_title("FG Make Rate & Attempt Volume by Season  |  NFL 2016–2024")
ax1.grid(True, axis="y")
ax1.set_axisbelow(True)

for _, row in season_stats.iterrows():
    ax1.text(row["season"], row["make_rate"] + 0.003,
             f"{row['make_rate']:.1%}", ha="center", fontsize=8, color=GOLD)

plt.tight_layout()
plt.savefig(OUT / "fg_season_line.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_season_line.png'}")


# ── PLOT 3: Attempts Distribution ────────────────────────────────────────────
print("Plotting 3/6: Attempts distribution...")

made   = df[df["fg_made"] == 1]["kick_distance"]
missed = df[df["fg_made"] == 0]["kick_distance"]

fig, ax = plt.subplots(figsize=(11, 5))
bins = range(18, 66, 2)

ax.hist(missed, bins=bins, color=DANGER, alpha=0.75, label="Missed",
        edgecolor="#0f2219", linewidth=0.5, zorder=2)
ax.hist(made,   bins=bins, color=ACCENT, alpha=0.65, label="Made",
        edgecolor="#0f2219", linewidth=0.5, zorder=3)

ax.set_xlabel("Kick Distance (yards)", labelpad=8)
ax.set_ylabel("Number of Attempts", labelpad=8)
ax.set_title("Distribution of FG Attempts by Distance  |  Made vs Missed  |  NFL 2016–2024")
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, fontsize=10)
ax.grid(True, axis="y")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT / "fg_attempts_hist.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_attempts_hist.png'}")


# ── PLOT 4: Feature Importance — Gain (XGBoost) + Permutation ───────────────
print("Plotting 4/6: Feature importance (gain + permutation)...")

# Gain importance averaged across calibration folds
try:
    gain_imps = np.mean(
        [cc.estimator.feature_importances_ for cc in model.calibrated_classifiers_],
        axis=0,
    )
except AttributeError:
    gain_imps = model.feature_importances_

# Permutation importance on held-out test set (ROC-AUC metric)
print("  Computing permutation importance on test set (20 repeats)...")
perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=20, random_state=42,
    scoring="roc_auc", n_jobs=-1,
)
perm_means = perm_result.importances_mean
perm_stds  = perm_result.importances_std

# Normalise permutation importance so bars are on a comparable scale
total_perm = perm_means.sum()
perm_pct   = perm_means / total_perm * 100 if total_perm > 0 else perm_means * 0

# Feature labels (shortened for readability)
FEAT_LABELS = {
    "kick_distance":           "kick_distance ★",
    "fg_make_rate_roll6":      "fg_make_rate_roll6 ★",
    "kicker_career_make_rate": "kicker_career_make_rate",
    "temp_x_distance":         "temp × distance",
    "kicker_career_attempts":  "kicker_career_attempts",
    "is_dome":                 "is_dome",
    "wind_gust_x_distance":    "wind_gust × distance",
    "wind_gust":               "wind_gust",
    "temp_adj":                "temp_adj",
    "game_seconds_remaining":  "game_seconds_remaining",
    "score_differential":      "score_differential",
    "surface_is_grass":        "surface_is_grass",
    "is_precipitation":        "is_precipitation",
    "altitude_ft":             "altitude_ft ⚠",
    "is_overtime":             "is_overtime ⚠",
}
labels = [FEAT_LABELS.get(f, f) for f in FEATURE_COLS]

# Sort by gain importance descending, then drop zero-importance features
order        = np.argsort(gain_imps)
sorted_feats = [labels[i] for i in order]
sorted_gain  = gain_imps[order]
sorted_perm  = perm_pct[order]

nonzero_mask = sorted_gain >= 0.0001
sorted_feats = [f for f, m in zip(sorted_feats, nonzero_mask) if m]
sorted_gain  = sorted_gain[nonzero_mask]
sorted_perm  = sorted_perm[nonzero_mask]

# Colors: gold = top 5 by gain, green = rest
bar_colors = []
for i, imp in enumerate(sorted_gain):
    if i >= len(sorted_feats) - 5:
        bar_colors.append(GOLD)
    else:
        bar_colors.append("#2d5a3d")

fig, (ax_gain, ax_perm) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.subplots_adjust(wspace=0.04)

# Left: Gain importance
bars_g = ax_gain.barh(sorted_feats, sorted_gain * 100,
                       color=bar_colors, edgecolor="#1e4a2e", linewidth=0.8)
for bar, val in zip(bars_g, sorted_gain):
    ax_gain.text(max(val * 100 + 0.1, 0.3), bar.get_y() + bar.get_height() / 2,
                 f"{val:.3%}", va="center", ha="left", fontsize=8.5, color=WHITE)
ax_gain.set_xlabel("Gain Importance (% of total)", labelpad=8)
ax_gain.set_title("XGBoost Gain Importance\n(how often each feature creates splits)", fontsize=10)
ax_gain.grid(True, axis="x")
ax_gain.set_axisbelow(True)
ax_gain.tick_params(axis="y", labelsize=9)

# Right: Permutation importance
bar_colors_p = []
for i, imp in enumerate(sorted_gain):
    if i >= len(sorted_feats) - 5:
        bar_colors_p.append(GOLD)
    else:
        bar_colors_p.append("#2d5a3d")

perm_sorted = sorted_perm
bars_p = ax_perm.barh(sorted_feats, perm_sorted,
                       color=bar_colors_p, edgecolor="#1e4a2e", linewidth=0.8)
for bar, val in zip(bars_p, perm_sorted):
    ax_perm.text(max(val + 0.1, 0.1), bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color=WHITE)
ax_perm.set_xlabel("Permutation Importance (% of total ROC-AUC drop)", labelpad=8)
ax_perm.set_title("Permutation Importance\n(how much AUC drops when feature is shuffled)", fontsize=10)
ax_perm.grid(True, axis="x")
ax_perm.set_axisbelow(True)

# Shared title & legend
fig.suptitle(
    "XGBoost FG Probability Model — Feature Importance",
    fontsize=11, color=WHITE, y=1.01, ha="center",
)

legend_elements = [
    mpatches.Patch(facecolor=GOLD,      label="Top 5 by gain"),
    mpatches.Patch(facecolor="#2d5a3d", label="Other features"),
]
ax_gain.legend(handles=legend_elements, facecolor="#0f2219", edgecolor="#2d5a3d",
               labelcolor=WHITE, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "fg_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_feature_importance.png'}")


# ── PLOT 5: Feature Sensitivity Sweep ────────────────────────────────────────
# Show how much each feature shifts predictions at average conditions.
# This captures altitude/OT effects the gain metric misses.
print("Plotting 5/6: Feature sensitivity sweep...")

avg_row = pd.DataFrame([dict(zip(FEATURE_COLS, X_train.mean(axis=0)))])

# Sweep each feature across its 5th–95th percentile range
train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)

sweep_results = {}
for feat in FEATURE_COLS:
    lo = np.percentile(train_df[feat], 5)
    hi = np.percentile(train_df[feat], 95)
    if hi <= lo:
        # Binary feature: sweep 0→1
        lo, hi = 0, 1

    row_lo = avg_row.copy(); row_lo[feat] = lo
    row_hi = avg_row.copy(); row_hi[feat] = hi

    p_lo = model.predict_proba(row_lo.to_numpy())[0, 1]
    p_hi = model.predict_proba(row_hi.to_numpy())[0, 1]
    sweep_results[feat] = abs(p_hi - p_lo) * 100  # percentage points

feats_sorted = [(f, e) for f, e in sorted(sweep_results.items(), key=lambda x: x[1]) if e >= 0.01]
feat_names   = [FEAT_LABELS.get(f, f) for f, _ in feats_sorted]
feat_effects = [v for _, v in feats_sorted]

eff_colors = []
for i, (feat, eff) in enumerate(feats_sorted):
    if i >= len(feats_sorted) - 5:
        eff_colors.append(GOLD)
    else:
        eff_colors.append(ACCENT)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(feat_names, feat_effects, color=eff_colors,
               edgecolor="#1e4a2e", linewidth=0.8)

for bar, val in zip(bars, feat_effects):
    ax.text(max(val + 0.1, 0.1), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} pp", va="center", ha="left", fontsize=9, color=WHITE)

ax.set_xlabel("Predicted Probability Shift (percentage points)  |  5th → 95th percentile", labelpad=8)
ax.set_title(
    "Feature Sensitivity — How Much Each Feature Moves P(FG Made)\n"
    "Evaluated at average conditions, sweeping 5th → 95th percentile of training data.",
    fontsize=10, color=WHITE,
)
ax.grid(True, axis="x")
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=9)

legend_elements = [
    mpatches.Patch(facecolor=GOLD,   label="Top 5 movers"),
    mpatches.Patch(facecolor=ACCENT, label="Other features"),
]
ax.legend(handles=legend_elements, facecolor="#0f2219", edgecolor="#2d5a3d",
          labelcolor=WHITE, loc="lower right")

plt.tight_layout()
plt.savefig(OUT / "fg_feature_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_feature_sensitivity.png'}")


# ── PLOT 6: Feature Importance % with Cumulative Line ───────────────────────
print("Plotting 6/6: Feature importance breakdown (%)...")

total_gain  = gain_imps.sum()
pct_imps    = (gain_imps / total_gain) * 100
sorted_pct  = pct_imps[order][nonzero_mask]
cumulative  = np.cumsum(sorted_pct)

bar_colors2 = []
for i, imp in enumerate(sorted_gain):
    if i >= len(sorted_feats) - 5:
        bar_colors2.append(GOLD)
    else:
        bar_colors2.append(ACCENT)

fig, ax1 = plt.subplots(figsize=(11, 7))
ax2 = ax1.twiny()

ax1.barh(sorted_feats, sorted_pct, color=bar_colors2,
         edgecolor="#1e4a2e", linewidth=0.8, alpha=0.88)
ax2.plot(cumulative, sorted_feats, color=DANGER, linewidth=2,
         marker="o", markersize=4, zorder=5)

for i, (feat, pct) in enumerate(zip(sorted_feats, sorted_pct)):
    ax1.text(max(pct + 0.1, 0.1), i, f"{pct:.1f}%", va="center",
             fontsize=9, color=WHITE)

ax1.set_xlabel("% of Total Gain Importance", labelpad=8)
ax2.set_xlabel("Cumulative Importance %", color=DANGER, labelpad=8)
ax2.tick_params(axis="x", colors=DANGER)
ax2.set_xlim(0, 108)
ax1.set_title(
    "Feature Importance Breakdown  |  XGBoost FG Probability Model\n"
    "% of total gain · red line = cumulative",
    fontsize=10, color=WHITE,
)
ax1.grid(True, axis="x")
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig(OUT / "fg_feature_importance_pct.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_feature_importance_pct.png'}")


print(f"\n✓ Done — all 6 plots saved to {OUT}/")
