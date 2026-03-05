"""Win Probability Model — Diagnostic Visualizations
=====================================================
Four plots that validate model fit and surface the learned WP surface.

Run from the project root::

    python -m src.visualization.win_probability_plots

Plots saved to ``reports/figures/``:

  1. reliability_diagram.png  — calibration curve (predicted vs actual win rate)
  2. wp_surface_heatmap.png   — WP as a function of score × time remaining
  3. feature_importance.png   — top-20 XGBoost gain importances
  4. residuals_by_state.png   — mean residual by quarter / score group / field position
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

from src.models.win_probability import WinProbabilityModel, build_training_data, load_wp_model

_ROOT = Path(__file__).resolve().parents[2]
_FIG_DIR = _ROOT / "reports" / "figures"
_MODEL_PATH = _ROOT / "models" / "win_probability_model.pkl"

logger = logging.getLogger(__name__)

# ── Shared style ──────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ── Shared data helper ────────────────────────────────────────────────────────

def _get_test_predictions(
    model: WinProbabilityModel,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (y_true, y_pred, df_test) for the model's held-out test seasons.

    OT plays are transformed via OTStateTransformer before prediction so the
    residuals-by-state chart reflects the actual model output rather than
    bypassing the transformer.
    """
    test_mask = df["season"].isin(model.test_seasons)
    df_test = df[test_mask].copy()

    # Apply OT transformation to OT rows before batch feature engineering
    ot_mask = df_test.get("is_overtime", pd.Series(0, index=df_test.index)).astype(bool)
    df_scored = df_test.copy()

    if ot_mask.any():
        ot_rows = df_test[ot_mask].copy()
        transformed = ot_rows.apply(
            lambda row: pd.Series(model._ot_transformer.transform(row.to_dict())),
            axis=1,
        )
        for col in ["is_overtime", "overtime_possession_number", "quarter", "seconds_remaining"]:
            if col in transformed.columns:
                df_scored.loc[ot_mask, col] = transformed[col].values

    df_feat = model._engineer_features(df_scored)
    X = df_feat[model._feature_cols].fillna(0.0).values
    raw = model._base_model.predict_proba(X)[:, 1]
    y_pred = model._apply_calibrator(raw)
    y_true = df_test["offense_team_won_game"].values
    return y_true, y_pred, df_test


# ── Plot 1: Reliability diagram ───────────────────────────────────────────────

def plot_reliability_diagram(
    model: WinProbabilityModel,
    df: pd.DataFrame,
    save_path: Path | None = None,
    n_bins: int = 15,
) -> plt.Figure:
    """Calibration curve (reliability diagram) on the test seasons.

    The upper panel plots mean predicted WP per bin against the actual win
    rate in that bin.  A perfectly calibrated model hugs the diagonal.
    Curves above the diagonal indicate under-confidence; below → over-confidence.
    The lower panel shows the distribution of predicted probabilities so you
    can see where the model is dense vs sparse.
    """
    y_true, y_pred, _ = _get_test_predictions(model, df)
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")
    mce = float(np.mean(np.abs(frac_pos - mean_pred)))

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", alpha=0.55)
    ax.plot(mean_pred, frac_pos, "o-", color="#1565C0", lw=2, ms=6, label="Model")
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.12, color="#1565C0")
    ax.set_ylabel("Fraction of wins (actual)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Reliability Diagram — Test seasons {model.test_seasons}\n"
        f"Mean calibration error = {mce:.4f}",
        pad=10,
    )
    ax.legend(loc="upper left")
    ax.set_xticklabels([])

    ax2 = axes[1]
    ax2.hist(y_pred, bins=40, color="#1565C0", alpha=0.65, edgecolor="none")
    ax2.set_xlabel("Predicted win probability")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))
    ax2.set_xlim(0, 1)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved reliability diagram → %s", save_path)
    return fig


# ── Plot 2: WP surface heatmap ────────────────────────────────────────────────

def plot_wp_surface_heatmap(
    model: WinProbabilityModel,
    save_path: Path | None = None,
) -> plt.Figure:
    """Win probability surface: score differential × time remaining.

    Fixed context: 1st & 10 at own 25 (yardline_100 = 75), two timeouts each,
    regulation play.  Annotates each cell with the predicted WP value so
    the surface is easy to sanity-check against domain intuition.
    """
    score_diffs = np.arange(-21, 22, 3)   # -21 to +21, step 3
    time_points = [3600, 2700, 1800, 900, 600, 300, 120, 60, 30]
    time_labels  = ["Q1\nstart", "Q2\nstart", "Half", "Q4\nstart",
                    "10:00", "5:00", "2:00", "1:00", "0:30"]

    # Quarter derived from game seconds remaining (counts down from 3600)
    def _sec_to_quarter(sec: float) -> int:
        if sec > 2700: return 1
        if sec > 1800: return 2
        if sec > 900:  return 3
        return 4

    wp_grid = np.zeros((len(score_diffs), len(time_points)))
    for i, sd in enumerate(score_diffs):
        for j, sec in enumerate(time_points):
            state = dict(
                score_differential=float(sd),
                quarter=_sec_to_quarter(sec),
                seconds_remaining=float(sec),
                yardline_100=75.0,
                down=1, ydstogo=10.0,
                offense_timeouts=2, defense_timeouts=2,
                is_overtime=0, overtime_possession_number=0,
            )
            wp_grid[i, j] = model.predict_proba(state)

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(wp_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Win probability")

    ax.set_xticks(range(len(time_points)))
    ax.set_xticklabels(time_labels, fontsize=10)
    ax.set_yticks(range(len(score_diffs)))
    ax.set_yticklabels([f"{int(s):+d}" for s in score_diffs])
    ax.set_xlabel("Time remaining")
    ax.set_ylabel("Score differential (offense − defense)")
    ax.set_title(
        "Win Probability Surface\n"
        "1st & 10 at own 25  |  Equal timeouts  |  Regulation",
        pad=10,
    )

    for i in range(len(score_diffs)):
        for j in range(len(time_points)):
            val = wp_grid[i, j]
            text_col = "white" if (val < 0.22 or val > 0.78) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_col, fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved WP surface heatmap → %s", save_path)
    return fig


# ── Plot 3: Feature importance ────────────────────────────────────────────────

def plot_feature_importance(
    model: WinProbabilityModel,
    save_path: Path | None = None,
    top_n: int = 20,
) -> plt.Figure:
    """Horizontal bar chart of top-N XGBoost gain-based feature importances.

    Gain measures the average reduction in loss per split using that feature,
    weighted by the number of samples covered — the most interpretable
    importance metric for understanding which features drive predictions.
    """
    fi = model.feature_importance().head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(fi["feature"], fi["importance"], color="#1E88E5", edgecolor="none", height=0.65)
    ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=9)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Top-{top_n} Feature Importances — XGBoost Gain", pad=10)
    ax.set_xlim(0, fi["importance"].max() * 1.20)
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved feature importance chart → %s", save_path)
    return fig


# ── Plot 4: Residuals by game state ───────────────────────────────────────────

def plot_residuals_by_state(
    model: WinProbabilityModel,
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Mean residual (predicted WP − actual outcome) by three game-state dimensions.

    Panels: quarter, score group (losing badly → winning badly), and field
    position zone.  Error bars show 95 % confidence intervals (±1.96 × SE).

    A well-calibrated model shows near-zero residuals in every bin.
    Positive bars mean the model over-predicted WP in those situations;
    negative bars mean it under-predicted.
    """
    y_true, y_pred, df_test = _get_test_predictions(model, df)

    df_res = df_test.copy()
    df_res["residual"] = y_pred - y_true

    # Quarter label
    qtr_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "OT"}
    df_res["quarter_label"] = df_res["quarter"].map(qtr_map).fillna("OT")

    # Score group
    def _score_group(sd: float) -> str:
        if sd <= -8:  return "Down 8+"
        if sd <= -1:  return "Down 1-7"
        if sd == 0:   return "Tied"
        if sd <= 7:   return "Up 1-7"
        return "Up 8+"
    df_res["score_group"] = df_res["score_differential"].apply(_score_group)

    # Field position zone
    def _field_zone(yl: float) -> str:
        if yl >= 60:  return "Own\n(60–99)"
        if yl >= 41:  return "Midfield\n(41–59)"
        if yl >= 21:  return "Opp\n(21–40)"
        return "Red zone\n(≤20)"
    df_res["field_zone"] = df_res["yardline_100"].apply(_field_zone)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Mean Residual (Predicted WP − Actual) by Game State\n"
        f"Test seasons {model.test_seasons}  |  n = {len(df_res):,}",
        fontsize=13,
    )

    panel_specs = [
        ("quarter_label", ["Q1", "Q2", "Q3", "Q4", "OT"], "By Quarter", 0),
        ("score_group",   ["Down 8+", "Down 1-7", "Tied", "Up 1-7", "Up 8+"], "By Score Group", 15),
        ("field_zone",    ["Own\n(60–99)", "Midfield\n(41–59)", "Opp\n(21–40)", "Red zone\n(≤20)"],
         "By Field Position", 0),
    ]

    for ax, (col, order, title, rotation) in zip(axes, panel_specs):
        stats = (
            df_res.groupby(col)["residual"]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reindex(order)
            .reset_index()
        )
        colors = ["#EF5350" if m > 0 else "#42A5F5" for m in stats["mean"]]
        ax.bar(stats[col], stats["mean"], color=colors, edgecolor="none", width=0.6)
        ax.errorbar(
            range(len(stats)), stats["mean"], yerr=1.96 * stats["sem"],
            fmt="none", color="black", capsize=4, lw=1.2,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(stats)))
        if rotation:
            ax.set_xticklabels(stats[col], rotation=rotation, ha="right", fontsize=9)
        else:
            ax.set_xticklabels(stats[col], fontsize=9)
        ax.set_ylabel("Mean residual" if ax is axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_ylim(-0.06, 0.06)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved residuals chart → %s", save_path)
    return fig


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Generate all four diagnostic plots and save to reports/figures/."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Win Probability Model — Diagnostic Visualizations")
    print("=" * 65)

    print("\nLoading model and play-by-play data...")
    model = load_wp_model(_MODEL_PATH)
    df = build_training_data()

    print("\n[1/4]  Reliability diagram (calibration curve)...")
    plot_reliability_diagram(model, df, save_path=_FIG_DIR / "reliability_diagram.png")

    print("[2/4]  Win probability surface heatmap...")
    plot_wp_surface_heatmap(model, save_path=_FIG_DIR / "wp_surface_heatmap.png")

    print("[3/4]  Feature importance chart...")
    plot_feature_importance(model, save_path=_FIG_DIR / "feature_importance.png")

    print("[4/4]  Residuals by game state...")
    plot_residuals_by_state(model, df, save_path=_FIG_DIR / "residuals_by_state.png")

    print(f"\nAll plots saved to {_FIG_DIR}")
    plt.close("all")


if __name__ == "__main__":
    main()
