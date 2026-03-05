"""
FG Probability - Static Plots for Jupyter Notebook
====================================================
Run each cell independently or all at once.
Requires: numpy, matplotlib, and your trained model (models/fg_prob_model.pkl)

pip install numpy matplotlib

Make sure you have trained the model first:
    python -m src.models.fg_probability
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.models.fg_probability import predict_fg_prob

# Thin wrapper exposing all 15 model features with sensible defaults.
# wind_gust defaults to wind when not specified (matches predict_fg_prob behaviour).
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
        wind_gust=wind_gust,          # None → falls back to wind inside predict_fg_prob
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


# ── Shared style ────────────────────────────────────────────────────────────
DARK   = "#0d1f17"
GREEN  = "#1a472a"
GOLD   = "#f5c518"
ACCENT = "#4ade80"
DANGER = "#f87171"
WHITE  = "#f8f8f2"
MUTED  = "#8aaa96"

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
})


# ── CELL 1: P(make) vs Distance ─────────────────────────────────────────────
# ============================================================================
distances = np.arange(18, 63)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)

for label, kicker, color in [
    ("Elite kicker (95%)",    0.95, ACCENT),
    ("League avg (80%)",      0.80, GOLD),
    ("Struggling kicker (60%)", 0.60, DANGER),
]:
    probs = [fg_prob(d, fg_make_rate_roll6=kicker) for d in distances]
    ax.plot(distances, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(distances, probs, alpha=0.08, color=color)

ax.set_xlabel("Kick Distance (yards)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Distance  |  Outdoor, 8 mph wind, 55°F", 
             color=WHITE, fontsize=12, pad=12)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE)
ax.grid(True)

# Vertical reference lines for distance buckets
for yard, lbl in [(30, "30"), (40, "40"), (50, "50"), (55, "55")]:
    ax.axvline(yard, color="#2d5a3d", linewidth=1, linestyle=":")
    ax.text(yard + 0.3, 0.03, f"{lbl} yd", color="#4a7a5a", fontsize=8)

plt.tight_layout()
plt.savefig("fg_distance.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 2: P(make) vs Wind Gust ────────────────────────────────────────────
# The model uses wind_gust (peak gust speed) as its primary wind feature.
# When wind_gust is None it falls back to avg wind; here we sweep gust directly.
# ============================================================================
gusts = np.arange(0, 46)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)

for label, dist, color in [
    ("30 yd kick", 30, ACCENT),
    ("45 yd kick", 45, GOLD),
    ("55 yd kick", 55, DANGER),
]:
    probs = [fg_prob(dist, wind_gust=g) for g in gusts]
    ax.plot(gusts, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(gusts, probs, alpha=0.08, color=color)

ax.set_xlabel("Wind Gust Speed (mph)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Wind Gust  |  Outdoor, 55°F, League-avg kicker",
             color=WHITE, fontsize=12, pad=12)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE)
ax.grid(True)

# Gust severity reference lines
for g, lbl in [(15, "Moderate"), (25, "Strong"), (35, "Severe")]:
    ax.axvline(g, color="#2d5a3d", linewidth=1, linestyle=":")
    ax.text(g + 0.3, 0.03, lbl, color="#4a7a5a", fontsize=8)

plt.tight_layout()
plt.savefig("fg_wind.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 3: P(make) vs Temperature ──────────────────────────────────────────
# ============================================================================
temps = np.arange(-10, 91, 2)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)

for label, dist, color in [
    ("30 yd kick", 30, ACCENT),
    ("45 yd kick", 45, GOLD),
    ("55 yd kick", 55, DANGER),
]:
    probs = [fg_prob(dist, temp=t) for t in temps]
    ax.plot(temps, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(temps, probs, alpha=0.08, color=color)

ax.set_xlabel("Temperature (°F)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Temperature  |  8 mph wind, League-avg kicker",
             color=WHITE, fontsize=12, pad=12)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE)
ax.grid(True)

# Freeze zone
ax.axvspan(-10, 32, alpha=0.06, color="#60a5fa")
ax.text(5, 0.97, "Freezing", color="#60a5fa", fontsize=8, va="top")

plt.tight_layout()
plt.savefig("fg_temp.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 4: P(make) vs Kicker Form ──────────────────────────────────────────
# ============================================================================
kicker_rates = np.linspace(0.55, 1.0, 100)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)

for label, dist, color in [
    ("30 yd kick", 30, ACCENT),
    ("45 yd kick", 45, GOLD),
    ("55 yd kick", 55, DANGER),
]:
    probs = [fg_prob(dist, fg_make_rate_roll6=k) for k in kicker_rates]
    ax.plot(kicker_rates * 100, probs, color=color, linewidth=2.5, label=label)
    ax.fill_between(kicker_rates * 100, probs, alpha=0.08, color=color)

ax.set_xlabel("Kicker Rolling 6-Game Make Rate (%)", labelpad=8)
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title("FG Make Probability vs Kicker Form  |  Outdoor, 8 mph wind, 55°F",
             color=WHITE, fontsize=12, pad=12)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor=WHITE)
ax.grid(True)

# League avg line
ax.axvline(80, color=MUTED, linewidth=1, linestyle=":")
ax.text(80.5, 0.03, "League avg", color=MUTED, fontsize=8)

plt.tight_layout()
plt.savefig("fg_kicker.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 5: Distance × Wind Heatmap ─────────────────────────────────────────
# ============================================================================
dist_range = np.arange(18, 63)
wind_range = np.arange(0, 31)

Z = np.array([[fg_prob(d, wind=w) for w in wind_range] for d in dist_range])

cmap = LinearSegmentedColormap.from_list("fg", [DANGER, GOLD, ACCENT])

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK)

im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1,
               extent=[wind_range[0], wind_range[-1], dist_range[0], dist_range[-1]])

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
cbar.ax.tick_params(colors=MUTED)
cbar.set_label("P(make)", color=MUTED)

# Contour lines
CS = ax.contour(wind_range, dist_range, Z, levels=[0.5, 0.65, 0.80, 0.90],
                colors=["white"], alpha=0.4, linewidths=1)
ax.clabel(CS, fmt=lambda v: f"{int(v*100)}%", fontsize=8, colors="white")

ax.set_xlabel("Wind Speed (mph)", labelpad=8)
ax.set_ylabel("Kick Distance (yards)", labelpad=8)
ax.set_title("FG Make Probability Heatmap  |  Distance × Wind  |  55°F, League-avg kicker",
             color=WHITE, fontsize=12, pad=12)

plt.tight_layout()
plt.savefig("fg_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 6: Scenario Bar Chart ───────────────────────────────────────────────
# All 15 model features explicitly set — covers the full input space.
# ============================================================================
scenarios = [
    (
        "20 yd\nDome, elite vet",
        dict(kick_distance=20, is_dome=True,  wind=0,   wind_gust=0,   temp=72, is_precipitation=False,
             fg_make_rate_roll6=0.95, kicker_career_make_rate=0.97, kicker_career_attempts=120,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=1800, score_differential=0,  is_overtime=False),
    ),
    (
        "38 yd\nMild, avg kicker",
        dict(kick_distance=38, is_dome=False, wind=5,   wind_gust=8,   temp=65, is_precipitation=False,
             fg_make_rate_roll6=0.83, kicker_career_make_rate=0.85, kicker_career_attempts=60,
             surface_is_grass=True,  altitude_ft=0,
             game_seconds_remaining=900,  score_differential=0,  is_overtime=False),
    ),
    (
        "47 yd\nTrailing -3\nfinal drive",
        dict(kick_distance=47, is_dome=False, wind=12,  wind_gust=18,  temp=42, is_precipitation=False,
             fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82, kicker_career_attempts=40,
             surface_is_grass=True,  altitude_ft=0,
             game_seconds_remaining=60,   score_differential=-3, is_overtime=False),
    ),
    (
        "47 yd\nDenver altitude",
        dict(kick_distance=47, is_dome=False, wind=12,  wind_gust=18,  temp=42, is_precipitation=False,
             fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82, kicker_career_attempts=40,
             surface_is_grass=True,  altitude_ft=5280,
             game_seconds_remaining=300,  score_differential=-3, is_overtime=False),
    ),
    (
        "55 yd\nCold / snowy\nstruggling",
        dict(kick_distance=55, is_dome=False, wind=20,  wind_gust=30,  temp=30, is_precipitation=True,
             fg_make_rate_roll6=0.72, kicker_career_make_rate=0.74, kicker_career_attempts=25,
             surface_is_grass=False, altitude_ft=0,
             game_seconds_remaining=120,  score_differential=-3, is_overtime=False),
    ),
    (
        "63 yd\nOT, no history\nneutral wx",
        dict(kick_distance=63, is_dome=False, wind=10,  wind_gust=14,  temp=55, is_precipitation=False,
             fg_make_rate_roll6=None, kicker_career_make_rate=None, kicker_career_attempts=0,
             surface_is_grass=True,  altitude_ft=0,
             game_seconds_remaining=60,   score_differential=0,  is_overtime=True),
    ),
]

labels = [s[0] for s in scenarios]
probs  = [fg_prob(**s[1]) for s in scenarios]
colors = [ACCENT if p > 0.75 else GOLD if p > 0.5 else DANGER for p in probs]

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK)

bars = ax.bar(labels, probs, color=colors, width=0.55, edgecolor="#2d5a3d", linewidth=0.8)

for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{int(p*100)}%", ha="center", va="bottom", fontsize=11,
            color=WHITE, fontweight="bold", fontfamily="monospace")

ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
ax.set_ylabel("P(make)", labelpad=8)
ax.set_title(
    "FG Make Probability — Example Scenarios\n"
    "(wind gust · career rate · score diff · time remaining · altitude · OT all included)",
    color=WHITE, fontsize=12, pad=12,
)
ax.axhline(0.80, color=MUTED, linewidth=1, linestyle=":", alpha=0.7)
ax.text(len(scenarios) - 0.45, 0.81, "80% league avg", color=MUTED, fontsize=8, va="bottom")
ax.grid(True, axis="y")
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("fg_scenarios.png", dpi=150, bbox_inches="tight")
plt.show()
