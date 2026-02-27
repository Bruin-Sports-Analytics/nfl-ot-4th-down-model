"""Win Probability Model — OT & Benchmark Evaluation
=====================================================
Evaluates our XGBoost WP model against nflfastR's built-in ``wp`` column
on two data slices:

  • All test plays  (2023-2024)
  • Overtime plays only (2023-2024 — the new both-team-possession rules)

Run from the project root::

    python -m src.visualization.benchmark_eval

Plots saved to ``reports/figures/``:

  1. benchmark_reliability.png   — calibration curves: Ours vs nflfastR (all plays)
  2. ot_reliability.png          — calibration curves: Ours vs nflfastR (OT only)
  3. metrics_comparison.png      — Brier / log-loss / calib-error bar chart
  4. ot_phase_analysis.png       — residuals + calibration by OT possession phase
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.models.win_probability import (
    WinProbabilityModel,
    _compute_ot_possession_number,
    load_wp_model,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIG_DIR = _ROOT / "reports" / "figures"
_DATA_DIR = _ROOT / "data" / "raw"
_MODEL_PATH = _ROOT / "models" / "win_probability_model.pkl"

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_dataframe(
    model: WinProbabilityModel,
    parquet_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build an evaluation dataframe that includes both our model's predictions
    and nflfastR's ``wp`` column for the same plays.

    Applies identical filters to ``build_training_data()`` so the row set
    is exactly the model's test split.  The nflfastR ``wp`` column is the
    pre-play win probability for the possession team (posteam), matching our
    model's target ``offense_team_won_game``.

    Returns
    -------
    pd.DataFrame with columns:
        our_wp, fastr_wp, offense_team_won_game,
        season, quarter, is_overtime, overtime_possession_number,
        score_differential, yardline_100
    """
    if parquet_path is None:
        parquet_path = _DATA_DIR / "pbp_2016_2024.parquet"

    load_cols = [
        "game_id", "posteam", "defteam", "home_team", "away_team",
        "season", "week", "qtr", "game_seconds_remaining",
        "score_differential", "down", "ydstogo", "yardline_100",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "drive", "result",
        "wp",   # nflfastR pre-play WP for posteam
    ]
    df = pd.read_parquet(parquet_path, columns=load_cols)

    # ── Same filters as build_training_data() ─────────────────────────────────
    required_notna = [
        "down", "ydstogo", "yardline_100",
        "game_seconds_remaining", "score_differential",
        "result", "posteam",
    ]
    df = df.dropna(subset=required_notna)
    df = df[df["season"].between(2016, 2024)].copy()
    df = df[df["down"].between(1, 4)].copy()

    # ── Target variable (same as build_training_data()) ────────────────────────
    df["offense_team_won_game"] = (
        ((df["posteam"] == df["home_team"]) & (df["result"] > 0)) |
        ((df["posteam"] == df["away_team"]) & (df["result"] < 0))
    ).astype(int)

    # ── Canonical feature columns ──────────────────────────────────────────────
    df["quarter"]            = df["qtr"].clip(1, 5).astype(int)
    df["seconds_remaining"]  = df["game_seconds_remaining"].clip(lower=0).astype(float)
    df["yardline_100"]       = df["yardline_100"].clip(1, 99).astype(float)
    df["ydstogo"]            = df["ydstogo"].clip(1, 99).astype(float)
    df["down"]               = df["down"].astype(int)
    df["offense_timeouts"]   = df["posteam_timeouts_remaining"].fillna(3).clip(0, 3).astype(int)
    df["defense_timeouts"]   = df["defteam_timeouts_remaining"].fillna(3).clip(0, 3).astype(int)
    df["score_differential"] = df["score_differential"].astype(float)
    df["is_overtime"]        = (df["quarter"] >= 5).astype(int)
    df["offense_team_strength_rating"] = 0.0
    df["defense_team_strength_rating"] = 0.0

    df = _compute_ot_possession_number(df)

    # ── Filter to test seasons only ────────────────────────────────────────────
    df = df[df["season"].isin(model.test_seasons)].copy()

    # ── Drop rows where nflfastR wp is null (can't benchmark those) ────────────
    df = df.dropna(subset=["wp"]).copy()
    df = df.rename(columns={"wp": "fastr_wp"})

    # ── Run our model on the same rows ─────────────────────────────────────────
    df_feat = model._engineer_features(df)
    X = df_feat[model._feature_cols].fillna(0.0).values
    raw = model._base_model.predict_proba(X)[:, 1]
    df["our_wp"] = model._apply_calibrator(raw)

    logger.info(
        "Eval dataframe: %d plays | %d OT plays | seasons %s",
        len(df), df["is_overtime"].sum(), sorted(df["season"].unique()),
    )
    return df


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    n_bins: int = 10,
) -> dict:
    """Return log_loss, brier, mean_calib_err and the calibration curve arrays."""
    ll   = log_loss(y_true, y_pred)
    bs   = brier_score_loss(y_true, y_pred)
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    mce  = float(np.mean(np.abs(frac_pos - mean_pred)))
    logger.info("[%s]  n=%d  log_loss=%.4f  brier=%.4f  mean_calib_err=%.4f",
                label, len(y_true), ll, bs, mce)
    return {
        "label": label, "n": len(y_true),
        "log_loss": ll, "brier": bs, "mean_calib_err": mce,
        "frac_pos": frac_pos, "mean_pred": mean_pred,
    }


# ── Plot 1: Benchmark reliability diagram (all test plays) ────────────────────

def plot_benchmark_reliability(
    df: pd.DataFrame,
    save_path: Path | None = None,
    n_bins: int = 15,
) -> plt.Figure:
    """Overlay calibration curves — Our model vs nflfastR — on all test plays.

    Both curves use the same y_true (``offense_team_won_game``).  The nflfastR
    ``wp`` column is the pre-play WP for the possession team so it is directly
    comparable.  Whichever curve hugs the diagonal more tightly is better
    calibrated on these seasons.
    """
    y_true = df["offense_team_won_game"].values
    frac_ours, pred_ours = calibration_curve(y_true, df["our_wp"].values,   n_bins=n_bins, strategy="uniform")
    frac_fast, pred_fast = calibration_curve(y_true, df["fastr_wp"].values, n_bins=n_bins, strategy="uniform")

    mce_ours = float(np.mean(np.abs(frac_ours - pred_ours)))
    mce_fast = float(np.mean(np.abs(frac_fast - pred_fast)))

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")
    ax.plot(pred_ours, frac_ours, "o-", color="#1565C0", lw=2, ms=5,
            label=f"Our model  (MCE={mce_ours:.4f})")
    ax.plot(pred_fast, frac_fast, "s-", color="#E65100", lw=2, ms=5,
            label=f"nflfastR   (MCE={mce_fast:.4f})")
    ax.set_ylabel("Fraction of wins (actual)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        f"Reliability Diagram — All Test Plays  (n={len(df):,})\n"
        f"Seasons {sorted(df['season'].unique())}",
        pad=10,
    )
    ax.legend(loc="upper left")
    ax.set_xticklabels([])

    ax2 = axes[1]
    ax2.hist(df["our_wp"],   bins=40, color="#1565C0", alpha=0.55, label="Our model", edgecolor="none")
    ax2.hist(df["fastr_wp"], bins=40, color="#E65100", alpha=0.45, label="nflfastR",  edgecolor="none")
    ax2.set_xlabel("Predicted win probability")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved benchmark reliability diagram → %s", save_path)
    return fig


# ── Plot 2: OT-only reliability diagram ──────────────────────────────────────

def plot_ot_reliability(
    df: pd.DataFrame,
    save_path: Path | None = None,
    n_bins: int = 8,
) -> plt.Figure:
    """Calibration curves restricted to overtime plays only.

    OT sample is small (~400 plays) so we use quantile binning (``strategy=
    'quantile'``) with fewer bins to avoid empty or near-empty buckets.
    The 2023-2024 test seasons fall under the new both-team-possession OT
    rules — a regime neither model was explicitly trained on.
    """
    df_ot = df[df["is_overtime"] == 1].copy()
    if len(df_ot) < 30:
        logger.warning("Fewer than 30 OT plays — skipping OT reliability plot.")
        return None

    y_true = df_ot["offense_team_won_game"].values
    frac_ours, pred_ours = calibration_curve(y_true, df_ot["our_wp"].values,   n_bins=n_bins, strategy="quantile")
    frac_fast, pred_fast = calibration_curve(y_true, df_ot["fastr_wp"].values, n_bins=n_bins, strategy="quantile")

    mce_ours = float(np.mean(np.abs(frac_ours - pred_ours)))
    mce_fast = float(np.mean(np.abs(frac_fast - pred_fast)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")
    ax.plot(pred_ours, frac_ours, "o-", color="#1565C0", lw=2, ms=6,
            label=f"Our model  (MCE={mce_ours:.4f})")
    ax.plot(pred_fast, frac_fast, "s-", color="#E65100", lw=2, ms=6,
            label=f"nflfastR   (MCE={mce_fast:.4f})")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Fraction of wins (actual)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        f"Reliability Diagram — Overtime Plays Only  (n={len(df_ot):,})\n"
        f"Seasons {sorted(df['season'].unique())}  |  quantile bins",
        pad=10,
    )
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved OT reliability diagram → %s", save_path)
    return fig


# ── Plot 3: Metrics comparison bar chart ──────────────────────────────────────

def plot_metrics_comparison(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Side-by-side bar chart of Brier score, log-loss, and mean calibration
    error for Our model vs nflfastR across three data slices:
      • All test plays
      • Overtime plays only
      • OT 1st possession  /  OT 2nd possession  /  OT sudden death
    Lower is better for all three metrics.
    """
    def _slice_metrics(mask, tag):
        sub = df[mask]
        if len(sub) < 10:
            return []
        y = sub["offense_team_won_game"].values
        rows = []
        for name, col in [("Our model", "our_wp"), ("nflfastR", "fastr_wp")]:
            m = compute_metrics(y, sub[col].values, f"{tag} | {name}", n_bins=8)
            rows.append({
                "slice": tag, "model": name,
                "Brier score": m["brier"],
                "Log-loss": m["log_loss"],
                "Calib error": m["mean_calib_err"],
            })
        return rows

    ot = df["is_overtime"] == 1
    pn = df["overtime_possession_number"]

    rows = []
    rows += _slice_metrics(slice(None),     "All plays")
    rows += _slice_metrics(ot,              "OT — all")
    rows += _slice_metrics(ot & (pn == 0),  "OT — 1st poss")
    rows += _slice_metrics(ot & (pn == 1),  "OT — 2nd poss")
    rows += _slice_metrics(ot & (pn >= 2),  "OT — sudden death")

    results = pd.DataFrame(rows)
    metrics  = ["Brier score", "Log-loss", "Calib error"]
    slices   = results["slice"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Our Model vs nflfastR — Lower is Better", fontsize=13)

    colours = {"Our model": "#1565C0", "nflfastR": "#E65100"}
    x       = np.arange(len(slices))
    width   = 0.35

    for ax, metric in zip(axes, metrics):
        for i, (model_name, colour) in enumerate(colours.items()):
            vals  = [
                results.loc[(results["slice"] == s) & (results["model"] == model_name), metric].values[0]
                if len(results.loc[(results["slice"] == s) & (results["model"] == model_name)]) > 0
                else np.nan
                for s in slices
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model_name,
                          color=colour, alpha=0.85, edgecolor="none")
            ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(slices, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved metrics comparison chart → %s", save_path)
    return fig


# ── Plot 4: OT phase deep-dive ────────────────────────────────────────────────

def plot_ot_phase_analysis(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Four-panel deep-dive into overtime prediction accuracy by possession phase.

    Top row: mean predicted WP for each phase, split by actual outcome (won/lost).
             A well-calibrated model should show the winning group near 1 and
             the losing group near 0 — but the nflfastR model may be off
             because it was trained under old OT rules.

    Bottom row: mean residual (predicted - actual) per phase for each model.
                Negative = under-predicts win probability; positive = over-predicts.
    """
    df_ot = df[df["is_overtime"] == 1].copy()
    pn = df_ot["overtime_possession_number"]

    phase_map = {0: "1st possession\n(n={n})", 1: "2nd possession\n(n={n})", 2: "Sudden death\n(n={n})"}
    phases = sorted(df_ot["overtime_possession_number"].clip(0, 2).unique())
    phase_labels = []
    for p in phases:
        n = (pn.clip(0, 2) == p).sum()
        phase_labels.append(phase_map[p].format(n=n))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Overtime Deep-Dive by Possession Phase\n"
        f"Test seasons {sorted(df['season'].unique())}  |  n_OT={len(df_ot)}",
        fontsize=13,
    )

    # ── Top-left: Mean predicted WP by phase × outcome (Our model) ────────────
    ax = axes[0, 0]
    for outcome, marker, col in [(1, "o", "#2E7D32"), (0, "s", "#C62828")]:
        label = "Won" if outcome else "Lost"
        vals = []
        for p in phases:
            mask = (pn.clip(0, 2) == p) & (df_ot["offense_team_won_game"] == outcome)
            vals.append(df_ot.loc[mask, "our_wp"].mean() if mask.sum() > 0 else np.nan)
        ax.plot(phase_labels, vals, marker=marker, color=col, lw=2, ms=8, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted WP")
    ax.set_title("Our Model — Mean WP by Phase & Outcome")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Top-right: Mean predicted WP by phase × outcome (nflfastR) ────────────
    ax = axes[0, 1]
    for outcome, marker, col in [(1, "o", "#2E7D32"), (0, "s", "#C62828")]:
        label = "Won" if outcome else "Lost"
        vals = []
        for p in phases:
            mask = (pn.clip(0, 2) == p) & (df_ot["offense_team_won_game"] == outcome)
            vals.append(df_ot.loc[mask, "fastr_wp"].mean() if mask.sum() > 0 else np.nan)
        ax.plot(phase_labels, vals, marker=marker, color=col, lw=2, ms=8, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted WP")
    ax.set_title("nflfastR — Mean WP by Phase & Outcome")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Bottom-left: Mean residual by phase (Our model) ───────────────────────
    ax = axes[1, 0]
    df_ot["our_residual"]   = df_ot["our_wp"]   - df_ot["offense_team_won_game"]
    df_ot["fastr_residual"] = df_ot["fastr_wp"]  - df_ot["offense_team_won_game"]

    means_ours = []
    sems_ours  = []
    for p in phases:
        sub = df_ot.loc[pn.clip(0, 2) == p, "our_residual"]
        means_ours.append(sub.mean())
        sems_ours.append(sub.sem())

    colours_bar = ["#EF5350" if m > 0 else "#42A5F5" for m in means_ours]
    ax.bar(phase_labels, means_ours, color=colours_bar, edgecolor="none", width=0.5)
    ax.errorbar(range(len(phases)), means_ours, yerr=[1.96 * s for s in sems_ours],
                fmt="none", color="black", capsize=5, lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Mean residual (pred − actual)")
    ax.set_title("Our Model — Mean Residual by Phase")
    ax.set_ylim(-0.25, 0.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)

    # ── Bottom-right: Mean residual by phase (nflfastR) ───────────────────────
    ax = axes[1, 1]
    means_fast = []
    sems_fast  = []
    for p in phases:
        sub = df_ot.loc[pn.clip(0, 2) == p, "fastr_residual"]
        means_fast.append(sub.mean())
        sems_fast.append(sub.sem())

    colours_bar = ["#EF5350" if m > 0 else "#42A5F5" for m in means_fast]
    ax.bar(phase_labels, means_fast, color=colours_bar, edgecolor="none", width=0.5)
    ax.errorbar(range(len(phases)), means_fast, yerr=[1.96 * s for s in sems_fast],
                fmt="none", color="black", capsize=5, lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Mean residual (pred − actual)")
    ax.set_title("nflfastR — Mean Residual by Phase")
    ax.set_ylim(-0.25, 0.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved OT phase analysis → %s", save_path)
    return fig


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Run OT evaluation and nflfastR benchmark, save all plots."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Win Probability — OT & Benchmark Evaluation")
    print("=" * 65)

    print("\nLoading model...")
    model = load_wp_model(_MODEL_PATH)

    print("Building evaluation dataframe (test seasons + nflfastR wp)...")
    df = load_eval_dataframe(model)

    n_ot = df["is_overtime"].sum()
    print(f"\n  Test plays : {len(df):,}")
    print(f"  OT plays   : {n_ot:,}")
    pn_counts = df[df["is_overtime"] == 1]["overtime_possession_number"].clip(0, 2).value_counts().sort_index()
    for phase, cnt in pn_counts.items():
        phase_name = {0: "1st possession", 1: "2nd possession", 2: "Sudden death"}.get(phase, str(phase))
        print(f"    {phase_name}: {cnt}")

    # Print headline metrics to console
    print("\n-- Headline metrics (test seasons) -----------------------------")
    for model_name, col in [("Our model", "our_wp"), ("nflfastR ", "fastr_wp")]:
        m = compute_metrics(df["offense_team_won_game"].values, df[col].values,
                            f"{model_name} | all", n_bins=10)
        print(f"  {model_name}  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"calib_err={m['mean_calib_err']:.4f}")

    print("\n-- OT-only metrics ----------------------------------------------")
    df_ot = df[df["is_overtime"] == 1]
    for model_name, col in [("Our model", "our_wp"), ("nflfastR ", "fastr_wp")]:
        m = compute_metrics(df_ot["offense_team_won_game"].values, df_ot[col].values,
                            f"{model_name} | OT", n_bins=8)
        print(f"  {model_name}  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
              f"calib_err={m['mean_calib_err']:.4f}")

    print()
    print("[1/4]  Benchmark reliability diagram (all plays)...")
    plot_benchmark_reliability(df, save_path=_FIG_DIR / "benchmark_reliability.png")

    print("[2/4]  OT-only reliability diagram...")
    plot_ot_reliability(df, save_path=_FIG_DIR / "ot_reliability.png")

    print("[3/4]  Metrics comparison bar chart...")
    plot_metrics_comparison(df, save_path=_FIG_DIR / "metrics_comparison.png")

    print("[4/4]  OT phase deep-dive...")
    plot_ot_phase_analysis(df, save_path=_FIG_DIR / "ot_phase_analysis.png")

    print(f"\nAll plots saved to {_FIG_DIR}")
    plt.close("all")


if __name__ == "__main__":
    main()
