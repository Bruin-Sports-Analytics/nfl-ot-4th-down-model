"""
FG Probability — Presentation Slides
======================================
Generates 2 publication-quality images for slide decks.

  outputs/slide_feature_importance.png   — What drives the model (no zero features)
  outputs/slide_explainer.png            — Non-technical "what the model does" visual

Run from project root:  python fg_presentation_plots.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
import joblib

from src.models.fg_probability import FEATURE_COLS, MODEL_PATH, predict_fg_prob

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Load model & importances ──────────────────────────────────────────────────
artifact    = joblib.load(MODEL_PATH)
model       = artifact["model"]
importances = np.mean(
    [cc.estimator.feature_importances_ for cc in model.calibrated_classifiers_],
    axis=0,
)

# ── Colour palette ────────────────────────────────────────────────────────────
BG     = "#0d1f17"
PANEL  = "#0f2219"
BORDER = "#2d5a3d"
GRID   = "#1e4a2e"
GOLD   = "#f5c518"
GREEN  = "#4ade80"
RED    = "#f87171"
WHITE  = "#f8f8f2"
MUTED  = "#8aaa96"
BLUE   = "#60a5fa"

# ── Global rcParams for presentation look ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       WHITE,
    "text.color":        WHITE,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlepad":     14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   12,
    "axes.labelsize":    12,
})


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Feature Importance (zero-importance features removed)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating slide 1: Feature Importance...")

# Human-readable names for every feature
FEAT_NAMES = {
    "kick_distance":           "Kick Distance",
    "fg_make_rate_roll6":      "Kicker Recent Form\n(6-game rolling make rate)",
    "kicker_career_make_rate": "Kicker Career Make Rate\n(by distance bucket)",
    "temp_x_distance":         "Cold Air × Distance\n(compound drag effect)",
    "kicker_career_attempts":  "Kicker Experience\n(career attempts in bucket)",
    "is_dome":                 "Indoor / Dome",
    "wind_gust_x_distance":    "Wind Gust × Distance\n(kick-moment physics)",
    "wind_gust":               "Wind Gust Speed",
    "temp_adj":                "Temperature",
    "game_seconds_remaining":  "Time Remaining",
    "score_differential":      "Score Differential",
    "surface_is_grass":        "Natural Grass Surface",
    "is_precipitation":        "Rain / Snow / Fog",
}

# Filter out zero-importance features and sort ascending for horizontal bar
filtered = [(FEAT_NAMES.get(f, f), imp)
            for f, imp in zip(FEATURE_COLS, importances)
            if imp > 0]
filtered.sort(key=lambda x: x[1])

feat_labels = [f for f, _ in filtered]
feat_imps   = np.array([v for _, v in filtered])
feat_pcts   = feat_imps / feat_imps.sum() * 100

# Colour: top 3 gold, next 3 green, rest muted green
n = len(feat_labels)
bar_colors = []
for i in range(n):
    if i >= n - 3:
        bar_colors.append(GOLD)
    elif i >= n - 6:
        bar_colors.append(GREEN)
    else:
        bar_colors.append("#2d5a3d")

fig, ax = plt.subplots(figsize=(14, 7.5))
fig.patch.set_facecolor(BG)

bars = ax.barh(feat_labels, feat_pcts, color=bar_colors,
               edgecolor="#1a3a26", linewidth=0.8, height=0.62)

for bar, pct in zip(bars, feat_pcts):
    ax.text(
        pct + 0.25, bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        va="center", ha="left", fontsize=11, color=WHITE, fontweight="bold",
    )

ax.set_xlim(0, feat_pcts.max() * 1.22)
ax.set_xlabel("Share of Model Importance (%)", labelpad=10, fontsize=12)
ax.set_title(
    "What Drives Field Goal Probability?",
    fontsize=18, fontweight="bold", color=WHITE, pad=16,
)
ax.text(
    0.5, -0.10,
    "XGBoost feature importance (gain) averaged across calibration folds  ·  "
    "NFL 2016–2024  ·  8,742 unblocked FG attempts",
    transform=ax.transAxes, ha="center", fontsize=9, color=MUTED,
)

legend_elements = [
    mpatches.Patch(facecolor=GOLD,      label="Top 3 drivers"),
    mpatches.Patch(facecolor=GREEN,     label="Strong contributors"),
    mpatches.Patch(facecolor="#2d5a3d", label="Minor contributors"),
]
ax.legend(handles=legend_elements, facecolor=PANEL, edgecolor=BORDER,
          labelcolor=WHITE, loc="lower right", fontsize=10)

ax.grid(True, axis="x")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=6)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT / "slide_feature_importance.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  → Saved {OUT / 'slide_feature_importance.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Non-Technical Explainer: "What Does the Model Do?"
# 4-panel layout:
#   A (top-left)  — Distance curve with annotated scenarios
#   B (top-right) — Kicker quality comparison (grouped bars)
#   C (bot-left)  — Wind & cold effect at 45 yards
#   D (bot-right) — Real-game scenario scoreboard
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating slide 2: Non-technical explainer...")

def fg(d, **kw):
    defaults = dict(is_dome=False, wind=8, temp=55, fg_make_rate_roll6=0.80,
                    kicker_career_make_rate=0.81, kicker_career_attempts=50,
                    surface_is_grass=True, altitude_ft=0,
                    game_seconds_remaining=900, score_differential=0,
                    is_overtime=False)
    defaults.update(kw)
    return predict_fg_prob(kick_distance=d, **defaults)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)

gs = GridSpec(
    2, 2, figure=fig,
    left=0.06, right=0.97, top=0.88, bottom=0.07,
    wspace=0.28, hspace=0.46,
)

ax_a = fig.add_subplot(gs[0, 0])   # distance curve
ax_b = fig.add_subplot(gs[0, 1])   # kicker quality
ax_c = fig.add_subplot(gs[1, 0])   # conditions
ax_d = fig.add_subplot(gs[1, 1])   # real scenarios

PCT_FMT = mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%")

for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ── Panel A: Distance → Make Probability ─────────────────────────────────────
distances = np.arange(18, 65)
probs_avg  = [fg(d) for d in distances]

ax_a.plot(distances, probs_avg, color=GOLD, linewidth=3)
ax_a.fill_between(distances, probs_avg, alpha=0.12, color=GOLD)

# Annotate landmark distances
for dist, label, yoff in [
    (20, "Extra point\ndistance", +0.07),
    (38, "Mid-range", -0.14),
    (50, "50-yarder", -0.14),
    (60, "Near\nrecord", +0.07),
]:
    p = fg(dist)
    ax_a.scatter([dist], [p], color=WHITE, s=60, zorder=5)
    ax_a.annotate(
        f"{label}\n{p:.0%}",
        xy=(dist, p), xytext=(dist, p + yoff),
        ha="center", fontsize=8.5, color=WHITE,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )

ax_a.set_ylim(0, 1.12)
ax_a.set_xlim(18, 64)
ax_a.yaxis.set_major_formatter(PCT_FMT)
ax_a.set_xlabel("Kick Distance (yards)", fontsize=11)
ax_a.set_ylabel("Chance of Making It", fontsize=11)
ax_a.set_title("Longer Kicks = Lower Probability", fontsize=13, fontweight="bold", color=WHITE)
ax_a.grid(True, alpha=0.4)
ax_a.tick_params(colors=MUTED)

# ── Panel B: Kicker Quality at 3 distances ────────────────────────────────────
kick_dists  = [35, 45, 52]
dist_labels = ["35 yd", "45 yd", "52 yd"]

elite_probs    = [fg(d, fg_make_rate_roll6=0.95, kicker_career_make_rate=0.96, kicker_career_attempts=120) for d in kick_dists]
avg_probs      = [fg(d, fg_make_rate_roll6=0.80, kicker_career_make_rate=0.81, kicker_career_attempts=50) for d in kick_dists]
struggle_probs = [fg(d, fg_make_rate_roll6=0.62, kicker_career_make_rate=0.65, kicker_career_attempts=15) for d in kick_dists]

x   = np.arange(len(kick_dists))
w   = 0.24

b1 = ax_b.bar(x - w, elite_probs,    width=w, color=GREEN, label="Elite kicker",      edgecolor=PANEL, linewidth=0.5)
b2 = ax_b.bar(x,     avg_probs,      width=w, color=GOLD,  label="Average kicker",    edgecolor=PANEL, linewidth=0.5)
b3 = ax_b.bar(x + w, struggle_probs, width=w, color=RED,   label="Struggling kicker", edgecolor=PANEL, linewidth=0.5)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                  f"{h:.0%}", ha="center", va="bottom", fontsize=9,
                  color=WHITE, fontweight="bold")

ax_b.set_ylim(0, 1.18)
ax_b.set_xticks(x)
ax_b.set_xticklabels(dist_labels, fontsize=11, color=WHITE)
ax_b.yaxis.set_major_formatter(PCT_FMT)
ax_b.set_ylabel("Chance of Making It", fontsize=11)
ax_b.set_title("Kicker Quality Has a Big Impact", fontsize=13, fontweight="bold", color=WHITE)
ax_b.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE, fontsize=9, loc="upper right")
ax_b.grid(True, axis="y", alpha=0.4)
ax_b.tick_params(axis="x", colors=WHITE)
ax_b.tick_params(axis="y", colors=MUTED)

# ── Panel C: Weather / Conditions at 45 yards ─────────────────────────────────
conditions = [
    ("Dome\n(ideal)",        fg(45, is_dome=True, wind=0, temp=72)),
    ("Mild day\n(65°F, 8 mph)", fg(45, temp=65, wind=8)),
    ("Cold\n(32°F, 15 mph)",  fg(45, temp=32, wind=15, wind_gust=20)),
    ("Snow +\ngusts 25 mph",  fg(45, temp=28, wind=20, wind_gust=25, is_precipitation=True)),
]
cond_labels = [c[0] for c in conditions]
cond_probs  = [c[1] for c in conditions]
cond_colors = [BLUE, GREEN, GOLD, RED]

bars_c = ax_c.bar(cond_labels, cond_probs, color=cond_colors,
                  width=0.52, edgecolor=PANEL, linewidth=0.8)
for bar, p in zip(bars_c, cond_probs):
    ax_c.text(bar.get_x() + bar.get_width() / 2, p + 0.018,
              f"{p:.0%}", ha="center", va="bottom", fontsize=13,
              color=WHITE, fontweight="bold")

ax_c.set_ylim(0, 1.18)
ax_c.yaxis.set_major_formatter(PCT_FMT)
ax_c.set_ylabel("Chance of Making It", fontsize=11)
ax_c.set_title("Weather Conditions at 45 Yards", fontsize=13, fontweight="bold", color=WHITE)
ax_c.grid(True, axis="y", alpha=0.4)
ax_c.set_axisbelow(True)
ax_c.tick_params(axis="x", colors=WHITE, labelsize=10)
ax_c.tick_params(axis="y", colors=MUTED)

# ── Panel D: Real-Game Scenario Scoreboard ────────────────────────────────────
game_scenarios = [
    ("Short dome kick\nelite veteran",
     fg(22, is_dome=True, fg_make_rate_roll6=0.96, kicker_career_make_rate=0.97, kicker_career_attempts=130)),
    ("Mid-range kick\nmild conditions",
     fg(38, temp=62, wind=6, fg_make_rate_roll6=0.84, kicker_career_make_rate=0.85, kicker_career_attempts=55)),
    ("47-yarder\ntrailing, 2 min left",
     fg(47, temp=42, wind=12, wind_gust=18, game_seconds_remaining=90, score_differential=-3,
        fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82, kicker_career_attempts=40)),
    ("55-yarder\ncold & snowy",
     fg(55, temp=28, wind=20, wind_gust=28, is_precipitation=True,
        fg_make_rate_roll6=0.72, kicker_career_make_rate=0.74, kicker_career_attempts=22)),
    ("63-yard desperation\nneutral weather",
     fg(63, temp=55, wind=10, wind_gust=14, game_seconds_remaining=5, score_differential=-3,
        fg_make_rate_roll6=0.76, kicker_career_make_rate=0.77, kicker_career_attempts=8)),
]

sc_labels = [s[0] for s in game_scenarios]
sc_probs  = [s[1] for s in game_scenarios]
sc_colors = [GREEN if p > 0.80 else GOLD if p > 0.55 else RED for p in sc_probs]

bars_d = ax_d.barh(sc_labels, sc_probs, color=sc_colors,
                   height=0.52, edgecolor=PANEL, linewidth=0.8)
for bar, p in zip(bars_d, sc_probs):
    ax_d.text(p + 0.012, bar.get_y() + bar.get_height() / 2,
              f"{p:.0%}", va="center", ha="left", fontsize=13,
              color=WHITE, fontweight="bold")

ax_d.set_xlim(0, 1.22)
ax_d.xaxis.set_major_formatter(PCT_FMT)
ax_d.set_xlabel("Model's Predicted Probability", fontsize=11)
ax_d.set_title("Real-Game Scenarios", fontsize=13, fontweight="bold", color=WHITE)
ax_d.axvline(0.80, color=MUTED, linewidth=1.2, linestyle=":", alpha=0.7)
ax_d.text(0.805, -0.7, "80%\nleague\navg", color=MUTED, fontsize=8, va="bottom")
ax_d.grid(True, axis="x", alpha=0.4)
ax_d.set_axisbelow(True)
ax_d.tick_params(axis="y", colors=WHITE, labelsize=9.5, length=0, pad=6)
ax_d.tick_params(axis="x", colors=MUTED)
ax_d.invert_yaxis()

# ── Main title & subtitle ─────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    "NFL Field Goal Probability Model",
    ha="center", va="center", fontsize=22, fontweight="bold", color=WHITE,
)
fig.text(
    0.5, 0.925,
    "Given the kick distance, kicker quality, weather, and game situation — "
    "how likely is this field goal to go through?",
    ha="center", va="center", fontsize=12, color=MUTED,
)

plt.savefig(OUT / "slide_explainer.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  → Saved {OUT / 'slide_explainer.png'}")

print(f"\n✓ Done — presentation slides saved to {OUT}/")
