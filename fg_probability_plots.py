"""
FG Probability - Scenario & Sensitivity Plots
==============================================
Run from project root:  python fg_probability_plots.py

Outputs saved to: outputs/
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from src.models.fg_probability import predict_fg_prob

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Thin wrapper with sensible defaults covering all 15 features.
def fg_prob(
    kick_distance,
    is_dome=False,
    wind=8,
    wind_gust=None,
    temp=55,
    is_precipitation=False,
    fg_make_rate_roll6=0.80,
    kicker_career_make_rate=None,
    kicker_career_attempts=50,
    surface_is_grass=True,
    altitude_ft=0.0,
    game_seconds_remaining=1800.0,
    score_differential=0.0,
    is_overtime=False,
):
    return predict_fg_prob(
        kick_distance=kick_distance,
        is_dome=is_dome,
        wind=wind,
        wind_gust=wind_gust,
        temp=temp,
        is_precipitation=is_precipitation,
        fg_make_rate_roll6=fg_make_rate_roll6,
        kicker_career_make_rate=kicker_career_make_rate,
        kicker_career_attempts=kicker_career_attempts,
        surface_is_grass=surface_is_grass,
        altitude_ft=altitude_ft,
        game_seconds_remaining=game_seconds_remaining,
        score_differential=score_differential,
        is_overtime=is_overtime,
    )


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

PCT_FMT = mticker.FuncFormatter(lambda v, _: f"{int(v*100)}%")


# ── PLOT 1: P(make) vs Distance — Kicker Tiers ──────────────────────────────
print("Plotting 1/6: P(make) vs distance by kicker tier...")

distances = np.arange(18, 64)

fig, ax = plt.subplots(figsize=(11, 5.5))

kicker_tiers = [
    ("Elite  (roll6=95%, career=96%, 120 att)", 0.95, 0.96, 120, ACCENT),
    ("Good   (roll6=87%, career=88%, 60 att)",  0.87, 0.88,  60, GOLD),
    ("Avg    (roll6=80%, career=81%, 40 att)",  0.80, 0.81,  40, WHITE),
    ("Struggling (roll6=65%, career=70%, 15 att)", 0.65, 0.70, 15, DANGER),
]

for label, roll6, career, att, color in kicker_tiers:
    probs = [fg_prob(d, fg_make_rate_roll6=roll6, kicker_career_make_rate=career,
                     kicker_career_attempts=att) for d in distances]
    ax.plot(distances, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(distances, probs, alpha=0.06, color=color)

ax.set_xlabel("Kick Distance (yards)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Distance by Kicker Tier\n"
             "Outdoor · 8 mph wind · 55°F · sea level · neutral game state",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PCT_FMT)
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, fontsize=9)
ax.grid(True)

for yard, lbl in [(30, "30"), (40, "40"), (50, "50"), (55, "55"), (60, "60")]:
    ax.axvline(yard, color="#2d5a3d", linewidth=1, linestyle=":")
    ax.text(yard + 0.4, 0.03, f"{lbl} yd", color="#4a7a5a", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT / "fg_distance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_distance.png'}")


# ── PLOT 2: P(make) vs Wind Gust by Distance ─────────────────────────────────
print("Plotting 2/6: P(make) vs wind gust...")

gusts = np.arange(0, 46)

fig, ax = plt.subplots(figsize=(11, 5.5))

for label, dist, color in [
    ("30 yd kick", 30, ACCENT),
    ("40 yd kick", 40, GOLD),
    ("50 yd kick", 50, WHITE),
    ("55 yd kick", 55, DANGER),
]:
    probs = [fg_prob(dist, wind_gust=g) for g in gusts]
    ax.plot(gusts, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(gusts, probs, alpha=0.06, color=color)

ax.set_xlabel("Wind Gust Speed (mph)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Wind Gust Speed by Kick Distance\n"
             "Outdoor · 55°F · league-avg kicker · neutral game state",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PCT_FMT)
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, fontsize=9)
ax.grid(True)

for g, lbl in [(10, "Breezy"), (20, "Windy"), (30, "Strong"), (40, "Severe")]:
    ax.axvline(g, color="#2d5a3d", linewidth=1, linestyle=":")
    ax.text(g + 0.4, 0.03, lbl, color="#4a7a5a", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT / "fg_wind.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_wind.png'}")


# ── PLOT 3: P(make) vs Temperature by Distance ──────────────────────────────
print("Plotting 3/6: P(make) vs temperature...")

temps = np.arange(-5, 91, 2)

fig, ax = plt.subplots(figsize=(11, 5.5))

for label, dist, color in [
    ("30 yd kick", 30, ACCENT),
    ("40 yd kick", 40, GOLD),
    ("50 yd kick", 50, WHITE),
    ("55 yd kick", 55, DANGER),
]:
    probs = [fg_prob(dist, temp=t) for t in temps]
    ax.plot(temps, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(temps, probs, alpha=0.06, color=color)

ax.set_xlabel("Temperature (°F)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Temperature by Kick Distance\n"
             "Outdoor · 8 mph wind · league-avg kicker · neutral game state",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PCT_FMT)
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, fontsize=9)
ax.grid(True)

ax.axvspan(-5, 32, alpha=0.07, color=BLUES)
ax.text(8, 1.01, "Freezing", color=BLUES, fontsize=8)
ax.axvline(72, color="#2d5a3d", linewidth=1, linestyle=":")
ax.text(72.5, 0.03, "Dome temp", color="#4a7a5a", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT / "fg_temp.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_temp.png'}")


# ── PLOT 4: P(make) vs Career Attempts (kicker experience) ──────────────────
print("Plotting 4/6: P(make) vs kicker career attempts...")

att_range = np.arange(0, 121, 5)

fig, ax = plt.subplots(figsize=(11, 5.5))

for label, dist, roll6, career, color in [
    ("45 yd · elite form (roll6=93%)",  45, 0.93, 0.88, ACCENT),
    ("45 yd · avg form (roll6=80%)",    45, 0.80, 0.81, GOLD),
    ("55 yd · avg form (roll6=80%)",    55, 0.80, 0.81, DANGER),
]:
    probs = [fg_prob(dist, fg_make_rate_roll6=roll6,
                     kicker_career_make_rate=career,
                     kicker_career_attempts=int(a)) for a in att_range]
    ax.plot(att_range, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(att_range, probs, alpha=0.06, color=color)

ax.set_xlabel("Career Attempts in This Distance Bucket", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Kicker Experience (Career Attempts)\n"
             "More attempts → model trusts career rate more → less regression to league avg",
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PCT_FMT)
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE, fontsize=9)
ax.grid(True)

ax.axvline(20,  color="#2d5a3d", linewidth=1, linestyle=":")
ax.text(20.5, 0.03, "Rookie", color="#4a7a5a", fontsize=7.5)
ax.axvline(60,  color="#2d5a3d", linewidth=1, linestyle=":")
ax.text(60.5, 0.03, "Veteran", color="#4a7a5a", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT / "fg_career_attempts.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_career_attempts.png'}")


# ── PLOT 5: Distance × Wind Gust Heatmap ─────────────────────────────────────
print("Plotting 5/6: Distance × wind gust heatmap...")

dist_range = np.arange(18, 63)
gust_range = np.arange(0, 36)

Z = np.array([[fg_prob(d, wind_gust=g) for g in gust_range] for d in dist_range])

cmap = LinearSegmentedColormap.from_list("fg", [DANGER, GOLD, ACCENT])

fig, ax = plt.subplots(figsize=(11, 6.5))

im = ax.imshow(
    Z, aspect="auto", origin="lower", cmap=cmap, vmin=0.3, vmax=1.0,
    extent=[gust_range[0] - 0.5, gust_range[-1] + 0.5,
            dist_range[0] - 0.5, dist_range[-1] + 0.5],
)

cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cbar.ax.yaxis.set_major_formatter(PCT_FMT)
cbar.ax.tick_params(colors=MUTED)
cbar.set_label("P(make)", color=MUTED)

CS = ax.contour(gust_range, dist_range, Z,
                levels=[0.50, 0.65, 0.80, 0.90, 0.95],
                colors=["white"], alpha=0.5, linewidths=1.2)
ax.clabel(CS, fmt=lambda v: f"{int(v*100)}%", fontsize=8.5, colors="white")

ax.set_xlabel("Wind Gust Speed (mph)", labelpad=8)
ax.set_ylabel("Kick Distance (yards)", labelpad=8)
ax.set_title(
    "FG Make Probability Heatmap  |  Distance × Wind Gust\n"
    "55°F · league-avg kicker · neutral game state · sea level",
    fontsize=11,
)

plt.tight_layout()
plt.savefig(OUT / "fg_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_heatmap.png'}")


# ── PLOT 6: Scenario Bar Chart — All 15 Features ─────────────────────────────
print("Plotting 6/6: Scenario comparison bar chart...")

scenarios = [
    (
        "20 yd\nDome · elite vet",
        dict(kick_distance=20, is_dome=True,  wind=0,  wind_gust=0,   temp=72,
             is_precipitation=False, fg_make_rate_roll6=0.95,
             kicker_career_make_rate=0.97, kicker_career_attempts=120,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=1800, score_differential=0, is_overtime=False),
    ),
    (
        "38 yd\nMild · avg kicker",
        dict(kick_distance=38, is_dome=False, wind=5,  wind_gust=8,   temp=65,
             is_precipitation=False, fg_make_rate_roll6=0.83,
             kicker_career_make_rate=0.85, kicker_career_attempts=60,
             surface_is_grass=True, altitude_ft=0,
             game_seconds_remaining=900, score_differential=0, is_overtime=False),
    ),
    (
        "47 yd\nTrailing -3\nfinal drive",
        dict(kick_distance=47, is_dome=False, wind=12, wind_gust=18,  temp=42,
             is_precipitation=False, fg_make_rate_roll6=0.81,
             kicker_career_make_rate=0.82, kicker_career_attempts=40,
             surface_is_grass=True, altitude_ft=0,
             game_seconds_remaining=60, score_differential=-3, is_overtime=False),
    ),
    (
        "47 yd\nTrailing -3\nrookie kicker",
        dict(kick_distance=47, is_dome=False, wind=12, wind_gust=18,  temp=42,
             is_precipitation=False, fg_make_rate_roll6=0.81,
             kicker_career_make_rate=0.82, kicker_career_attempts=2,
             surface_is_grass=True, altitude_ft=0,
             game_seconds_remaining=60, score_differential=-3, is_overtime=False),
    ),
    (
        "50 yd\nSnow · gusts 25\nstruggling",
        dict(kick_distance=50, is_dome=False, wind=18, wind_gust=25,  temp=28,
             is_precipitation=True,  fg_make_rate_roll6=0.71,
             kicker_career_make_rate=0.73, kicker_career_attempts=20,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=180, score_differential=-3, is_overtime=False),
    ),
    (
        "55 yd\nCold · gusts 30\nstruggling",
        dict(kick_distance=55, is_dome=False, wind=20, wind_gust=30,  temp=30,
             is_precipitation=True,  fg_make_rate_roll6=0.72,
             kicker_career_make_rate=0.74, kicker_career_attempts=25,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=120, score_differential=-3, is_overtime=False),
    ),
    (
        "60 yd\nBlizzard\ngusts 35",
        dict(kick_distance=60, is_dome=False, wind=25, wind_gust=35,  temp=20,
             is_precipitation=True,  fg_make_rate_roll6=0.68,
             kicker_career_make_rate=0.70, kicker_career_attempts=10,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=5, score_differential=-3, is_overtime=False),
    ),
    (
        "63 yd\nOT · no history\nneutral wx",
        dict(kick_distance=63, is_dome=False, wind=10, wind_gust=14,  temp=55,
             is_precipitation=False, fg_make_rate_roll6=None,
             kicker_career_make_rate=None, kicker_career_attempts=0,
             surface_is_grass=True, altitude_ft=0,
             game_seconds_remaining=60, score_differential=0, is_overtime=True),
    ),
]

labels = [s[0] for s in scenarios]
probs  = [fg_prob(**s[1]) for s in scenarios]
colors = [ACCENT if p > 0.80 else GOLD if p > 0.55 else DANGER for p in probs]

fig, ax = plt.subplots(figsize=(14, 5.5))

bars = ax.bar(labels, probs, color=colors, width=0.58,
              edgecolor="#2d5a3d", linewidth=0.9)

for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{p:.1%}", ha="center", va="bottom", fontsize=11,
            color=WHITE, fontweight="bold", fontfamily="monospace")

overall_avg = 0.858  # league-wide make rate from training data
ax.axhline(overall_avg, color=MUTED, linewidth=1.5, linestyle=":", alpha=0.85)
ax.text(len(scenarios) - 0.35, overall_avg + 0.012,
        f"League avg  {overall_avg:.1%}", color=MUTED, fontsize=8.5, ha="right")

ax.set_ylim(0, 1.22)
ax.yaxis.set_major_formatter(PCT_FMT)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title(
    "FG Make Probability — Scenario Comparison\n"
    "All 15 model features set explicitly per scenario "
    "(distance · kicker form · weather · game pressure · surface · altitude · OT)",
    fontsize=11,
)

legend_elements = [
    mpatches.Patch(facecolor=ACCENT, label="> 80%  (high confidence)"),
    mpatches.Patch(facecolor=GOLD,   label="55–80%  (moderate)"),
    mpatches.Patch(facecolor=DANGER, label="< 55%  (low confidence)"),
]
ax.legend(handles=legend_elements, facecolor="#0f2219", edgecolor="#2d5a3d",
          labelcolor=WHITE, loc="upper right")
ax.grid(True, axis="y")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT / "fg_scenarios.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  → Saved {OUT / 'fg_scenarios.png'}")


print(f"\n✓ Done — all 6 plots saved to {OUT}/")
