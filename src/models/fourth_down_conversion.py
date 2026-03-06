"""
Fourth Down Conversion Probability Model
=========================================
feature/4th-down-conversion branch

Estimates P(4th-down conversion) given the game situation and rolling team stats.

Inputs (for inference)
----------------------
  -- Situation --
  ydstogo                     : yards needed for first down  (PRIMARY DRIVER)
  qtr                         : quarter (1-4; 5 = OT)
  yardline_100                : yards from opponent end zone
  score_differential          : posteam score minus defteam score
  game_seconds_remaining      : seconds left in the full game
  half_seconds_remaining      : seconds left in the current half
  wp                          : pre-snap win probability (nflfastR model)
  posteam_timeouts_remaining  : offense timeouts remaining
  defteam_timeouts_remaining  : defense timeouts remaining
  -- Environment --
  temp                        : temperature (°F)
  -- Play design --
  shotgun                     : 1 if shotgun formation, 0 if under-center
  no_huddle                   : 1 if hurry-up / no-huddle play
  -- Offensive rolling stats (last 15 games, shift(1) — no leakage) --
  epa_per_game_roll15         : EPA per drive allowed
  success_rate_roll15         : offensive success rate
  points_per_game_roll15      : points scored per game
  go_conv_rate                : rolling 4th-down conversion rate
  -- Defensive rolling stats (last 15 games, shift(1) — no leakage) --
  def_epa_per_game_roll15     : EPA per drive allowed by defense
  def_success_rate_roll15     : success rate allowed by defense
  def_points_per_game_roll15  : points allowed per game
  go_stop_rate                : defense's rolling 4th-down stop rate

Output
------
  P(conversion)  in [0, 1]

Monotone constraints enforced
-----------------------------
  ydstogo               : -1  (more yards needed = strictly lower probability)
  epa_per_game_roll15   : +1  (better offense EPA = higher probability)
  success_rate_roll15   : +1  (higher offense success rate = higher probability)
  def_epa_per_game_roll15 : +1  (more EPA allowed = worse defense = higher probability)
  def_success_rate_roll15 : +1  (more success rate allowed = easier to convert)

Training Data
-------------
  NFL 4th-down go-for-it attempts (ydstogo <= 10, unfiltered by result).
  Rolling offensive and defensive stats joined from drive-level summaries.
  Temporal train/test split: most recent ~20% of seasons held out.

Model
-----
  XGBoost Classifier with isotonic-regression calibration (CalibratedClassifierCV).
  Hyperparameters tuned via RandomizedSearchCV (60 iterations, 5-fold CV,
  optimising Brier score loss).

Run order
---------
  1. python src/data/build_features.py
  2. python -m src.models.fourth_down_conversion
  3. streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).resolve().parents[2]   # nfl-ot-4th-down-model/
DATA_DIR   = _ROOT / "data"
PROCESSED  = DATA_DIR / "processed"
MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODELS_DIR / "fourth_down_conversion.pkl"
DATA_PATH       = PROCESSED / "fourth_down_enhanced.csv"
TEAM_STATS_PATH = PROCESSED / "team_stats_snapshot.csv"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# All candidate features — the model will use whichever are present in the data.
# This allows the same training code to work even if optional columns (wp,
# no_huddle, timeout counts) are missing from the CSV.
CANDIDATE_FEATURE_COLS: list[str] = [
    # Situation
    "ydstogo",                      # PRIMARY: yards to go for first down
    "qtr",                          # quarter (1-4, 5=OT)
    "yardline_100",                 # distance from opponent end zone
    "score_differential",           # posteam - defteam score
    "game_seconds_remaining",       # total seconds remaining in game
    "half_seconds_remaining",       # seconds remaining in current half
    "wp",                           # pre-snap win probability (nflfastR)
    "posteam_timeouts_remaining",   # offense timeouts left
    "defteam_timeouts_remaining",   # defense timeouts left
    # Environment
    "temp",                         # temperature (°F)
    # Play design
    "shotgun",                      # shotgun formation indicator (0/1)
    "no_huddle",                    # hurry-up / no-huddle indicator (0/1)
    # Offensive rolling stats (15-game rolling, shift(1) — no leakage)
    "epa_per_game_roll15",          # EPA per drive
    "success_rate_roll15",          # play success rate
    "points_per_game_roll15",       # points scored per game
    # NOTE: go_conv_rate and go_stop_rate intentionally excluded.
    # They are season-level 4th-down rates with tiny sample sizes (many teams have
    # 0 or 1 attempts per snapshot), dominate the model (70% combined importance),
    # and produce out-of-distribution predictions (0% or 100%) in the app.
    # Team quality is already captured by the rolling offensive/defensive stats below.
    # Defensive rolling stats
    "def_epa_per_game_roll15",      # EPA per drive allowed
    "def_success_rate_roll15",      # success rate allowed
    "def_points_per_game_roll15",   # points allowed per game
]

TARGET = "target"

# Monotone constraints: features where the direction of effect is physically certain.
# +1 = probability must increase as feature increases
# -1 = probability must decrease as feature increases
#  0 = no constraint (learned freely by XGBoost)
_MONOTONE_MAP: dict[str, int] = {
    "ydstogo":               -1,  # more yards to go = strictly harder
    "epa_per_game_roll15":   +1,  # better offense = more likely to convert
    "success_rate_roll15":   +1,  # better offense success rate = more likely
    "def_epa_per_game_roll15": +1, # worse defense (more EPA allowed) = easier to convert
    "def_success_rate_roll15": +1, # more success rate allowed = easier to convert
}

# League-average fallbacks used by predict_conversion_prob() when rolling stats
# are not available for a team (e.g., week 1 of season).
_LEAGUE_AVG: dict[str, float] = {
    "epa_per_game_roll15":          0.00,
    "success_rate_roll15":          0.42,
    "points_per_game_roll15":      23.0,
    "def_epa_per_game_roll15":      0.00,
    "def_success_rate_roll15":      0.42,
    "def_points_per_game_roll15":  23.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_features(df: pd.DataFrame) -> list[str]:
    """Return CANDIDATE_FEATURE_COLS intersected with df's actual columns."""
    present = set(df.columns)
    return [f for f in CANDIDATE_FEATURE_COLS if f in present]


def _make_constraints(features: list[str]) -> tuple[int, ...]:
    """Build the monotone_constraints tuple matching the given feature order."""
    return tuple(_MONOTONE_MAP.get(f, 0) for f in features)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the enhanced 4th-down dataset and prepare it for training.

    Filters to go-for-it attempts only (ydstogo <= 10), creates the binary
    target, and returns a clean DataFrame with CANDIDATE_FEATURE_COLS + [TARGET, "season"].

    Returns
    -------
    pd.DataFrame  — one row per go-for-it attempt.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Enhanced dataset not found at {data_path}.\n"
            f"Run:  python src/build_features.py"
        )

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['season'].unique())}")

    # Filter to actual go-for-it attempts
    go_mask = (
        df["go_converted"].fillna(False).astype(bool) |
        df["go_failed"].fillna(False).astype(bool)
    )
    df = df[go_mask].copy()
    print(f"Go-for-it attempts (all):      {len(df):,}")

    df = df[df["ydstogo"] <= 10].copy()
    print(f"Go-for-it attempts (<=10 yds): {len(df):,}")

    df[TARGET] = df["go_converted"].fillna(0).astype(int)

    conv_rate = df[TARGET].mean()
    print(f"Conversion rate: {conv_rate:.1%}  ({df[TARGET].sum():,} converted / {len(df):,} attempts)")

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    n_search_iter: int = 80,
) -> tuple[object, list[str], dict]:
    """
    Tune and train the 4th-down conversion model.

    Uses RandomizedSearchCV (60 iterations, 5-fold, Brier score) to select
    XGBoost hyperparameters, then wraps the best estimator with isotonic
    calibration (CalibratedClassifierCV, 5-fold) to produce well-calibrated
    probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_training_data().
    n_search_iter : int
        Number of RandomizedSearchCV iterations (default 60).

    Returns
    -------
    (model, feature_cols, metrics_dict)
    """
    features = _available_features(df)
    print(f"\nFeatures used ({len(features)}): {features}")

    keep_cols = features + [TARGET, "season"]
    clean = df[[c for c in keep_cols if c in df.columns]].dropna()
    clean[TARGET] = clean[TARGET].astype(int)

    rows_before = len(df)
    print(f"Rows after dropna: {len(clean):,}  (dropped {rows_before - len(clean):,})")
    print(f"Class balance — 0: {(clean[TARGET]==0).sum():,}  |  1: {(clean[TARGET]==1).sum():,}")

    # Temporal train / test split: hold out the most recent ~20% of seasons
    all_seasons = sorted(clean["season"].unique())
    n_test_seasons = max(1, int(round(len(all_seasons) * 0.20)))
    test_seasons  = set(all_seasons[-n_test_seasons:])
    train_seasons = set(all_seasons) - test_seasons

    train_df = clean[clean["season"].isin(train_seasons)]
    test_df  = clean[clean["season"].isin(test_seasons)]

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test  = test_df[features]
    y_test  = test_df[TARGET]

    print(f"\nTrain seasons: {sorted(train_seasons)}  — {len(X_train):,} plays")
    print(f"Test  seasons: {sorted(test_seasons)}   — {len(X_test):,} plays")

    # Monotone constraints (physically grounded directions only)
    constraints = _make_constraints(features)
    constrained = {f: c for f, c in zip(features, constraints) if c != 0}
    print(f"\nMonotone constraints applied: {constrained}")

    # ── Hyperparameter search ──────────────────────────────────────────────
    print(f"\nRunning RandomizedSearchCV ({n_search_iter} iterations, 5-fold, Brier score)...")
    base = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        monotone_constraints=constraints,
        n_jobs=1,
    )
    param_dist = {
        "n_estimators":     [400, 600, 800, 1000],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.10],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [3, 5, 10, 20],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
        "gamma":            [0, 0.1, 0.3],
    }
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_search_iter,
        scoring="neg_brier_score",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best CV Brier score:  {-search.best_score_:.4f}")

    # ── Calibrate ────────────────────────────────────────────────────────
    print("\nFitting isotonic calibration (5-fold)...")
    best_base = XGBClassifier(
        **search.best_params_,
        random_state=42,
        eval_metric="logloss",
        monotone_constraints=constraints,
        n_jobs=1,
    )
    model = CalibratedClassifierCV(best_base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    # ── Cross-validation on training set ─────────────────────────────────
    print("\nCross-validating on training data (5-fold)...")
    auc_cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    brier_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_brier_score")
    print(f"  CV ROC-AUC    : {auc_cv.mean():.4f}  ±  {auc_cv.std():.4f}")
    print(f"  CV Brier score: {(-brier_cv).mean():.4f}  ±  {(-brier_cv).std():.4f}")

    # ── Hold-out evaluation ───────────────────────────────────────────────
    y_prob    = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_prob >= 0.5).astype(int)
    baseline  = np.full(len(y_test), y_train.mean())

    metrics = {
        "test_log_loss"     : log_loss(y_test, y_prob),
        "test_log_loss_base": log_loss(y_test, baseline),
        "test_accuracy"     : accuracy_score(y_test, y_pred),
        "test_roc_auc"      : roc_auc_score(y_test, y_prob),
        "test_avg_precision": average_precision_score(y_test, y_prob),
        "test_brier_score"  : brier_score_loss(y_test, y_prob),
        "test_n"            : len(y_test),
        "test_conv_rate"    : float(y_test.mean()),
    }

    print("\n" + "=" * 56)
    print("HOLD-OUT TEST RESULTS")
    print(f"  Log Loss  (model)    : {metrics['test_log_loss']:.4f}")
    print(f"  Log Loss  (baseline) : {metrics['test_log_loss_base']:.4f}")
    print(f"  Accuracy             : {metrics['test_accuracy']:.4f}")
    print(f"  ROC-AUC              : {metrics['test_roc_auc']:.4f}")
    print(f"  Avg Precision (PR)   : {metrics['test_avg_precision']:.4f}")
    print(f"  Brier Score          : {metrics['test_brier_score']:.4f}")
    print("=" * 56)

    # ── Sanity check: probability must decrease with ydstogo ──────────────
    print("\nSanity check — avg team stats, sweeping ydstogo 1–10:")
    avg_row = X_train.mean()
    for ytg in range(1, 11):
        row = avg_row.copy(); row["ydstogo"] = ytg
        p = model.predict_proba(pd.DataFrame([row]))[0][1]
        bar = "█" * int(p * 30)
        print(f"  4th & {ytg:2d}: {p:.1%}  {bar}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _make_plots(model, X_train, X_test, y_test, y_prob, features)

    return model, features, metrics


def _make_plots(model, X_train, X_test, y_test, y_prob, features):
    """Generate and save diagnostic plots."""
    DARK   = "#0d1f17"
    ACCENT = "#4ade80"
    GOLD   = "#f5c518"
    DANGER = "#f87171"
    MUTED  = "#8aaa96"
    GRID   = "#1e4a2e"

    plt.rcParams.update({
        "figure.facecolor": DARK, "axes.facecolor": "#0f2219",
        "axes.edgecolor": "#2d5a3d", "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": "#f8f8f2", "grid.color": GRID,
        "grid.linestyle": "--", "grid.alpha": 0.6, "font.family": "monospace",
    })

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Plot 1: Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, marker="o", lw=2.5, color=GOLD, label="Calibrated model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction converted")
    ax.set_title("Calibration Curve — 4th Down Conversion Model")
    ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor="#f8f8f2")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_dir / "4th_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Feature importance (averaged across calibration folds)
    try:
        importances = np.mean(
            [cc.estimator.feature_importances_ for cc in model.calibrated_classifiers_],
            axis=0,
        )
    except AttributeError:
        importances = model.feature_importances_

    order      = np.argsort(importances)
    bar_colors = [GOLD if i >= len(features) - 5 else "#2d5a3d" for i in range(len(features))]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([features[i] for i in order], importances[order],
            color=[bar_colors[i] for i in order], edgecolor="#1e4a2e", linewidth=0.8)
    for bar, val in zip(ax.patches, importances[order]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Feature importance (XGBoost gain, avg across calibration folds)")
    ax.set_title("Feature Importance — 4th Down Conversion Model")
    ax.grid(True, axis="x")
    plt.tight_layout()
    fig.savefig(out_dir / "4th_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Observed conversion rate vs ydstogo (test set)
    tmp = X_test[["ydstogo"]].copy()
    tmp["converted"] = y_test.values
    rate = (tmp.groupby("ydstogo")
               .agg(conv_rate=("converted", "mean"), n=("converted", "count"))
               .reset_index())
    rate = rate[rate["n"] >= 10]
    colors = [ACCENT if r > 0.55 else GOLD if r > 0.45 else DANGER for r in rate["conv_rate"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(rate["ydstogo"], rate["conv_rate"],
                  color=colors, edgecolor="#2d5a3d", linewidth=0.8, width=0.65)
    for bar, row in zip(bars, rate.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                f"{row.conv_rate:.0%}", ha="center", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f"n={row.n}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.12); ax.set_xlabel("Yards To Go"); ax.set_ylabel("Conversion Rate")
    ax.set_title("4th Down Conversion Rate vs Yards To Go  |  Test Seasons")
    ax.set_xticks(rate["ydstogo"])
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "4th_conversion_by_ydstogo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Predicted probability vs ydstogo for strong / avg / weak offenses
    avg = X_train.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, q, color in [
        ("Strong offense (75th pct EPA)", 0.75, ACCENT),
        ("Average offense",               0.50, GOLD),
        ("Weak offense (25th pct EPA)",   0.25, DANGER),
    ]:
        epa_val = X_train["epa_per_game_roll15"].quantile(q)
        probs = []
        for ytg in range(1, 11):
            row = avg.copy(); row["ydstogo"] = ytg; row["epa_per_game_roll15"] = epa_val
            probs.append(model.predict_proba(pd.DataFrame([row]))[0][1])
        ax.plot(range(1, 11), probs, marker="o", lw=2.5, label=label, color=color)
    ax.set_xlabel("Yards To Go"); ax.set_ylabel("P(conversion)")
    ax.set_title("Predicted Conversion Probability vs Yards To Go by Offense Strength")
    ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor="#f8f8f2")
    ax.set_xticks(range(1, 11)); ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_dir / "4th_prob_by_ydstogo_offense.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: Predicted probability vs ydstogo for different game situations
    avg2 = X_train.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    situations = [
        ("Leading +7, 4th qtr",  {"score_differential":  7, "game_seconds_remaining": 300},  ACCENT),
        ("Tied, mid-game",        {"score_differential":  0, "game_seconds_remaining": 1800}, GOLD),
        ("Trailing -7, 2-min",    {"score_differential": -7, "game_seconds_remaining": 120},  DANGER),
    ]
    for label, overrides, color in situations:
        probs = []
        for ytg in range(1, 11):
            row = avg2.copy(); row["ydstogo"] = ytg
            for k, v in overrides.items():
                if k in row.index:
                    row[k] = v
            probs.append(model.predict_proba(pd.DataFrame([row]))[0][1])
        ax.plot(range(1, 11), probs, marker="o", lw=2.5, label=label, color=color)
    ax.set_xlabel("Yards To Go"); ax.set_ylabel("P(conversion)")
    ax.set_title("Predicted Conversion Probability vs Yards To Go by Game Situation")
    ax.legend(facecolor="#0f2219", edgecolor="#2d5a3d", labelcolor="#f8f8f2")
    ax.set_xticks(range(1, 11)); ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_dir / "4th_prob_by_ydstogo_situation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, feature_cols: list[str], path: Path = MODEL_PATH) -> None:
    joblib.dump({"model": model, "features": feature_cols, "target": TARGET}, path)
    print(f"Model saved → {path}")


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            f"Run:  python -m src.models.fourth_down_conversion"
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------

def predict_conversion_prob(
    ydstogo: float,
    yardline_100: float,
    score_differential: float,
    game_seconds_remaining: float,
    qtr: int = 4,
    half_seconds_remaining: float = 300.0,
    wp: float = 0.5,
    posteam_timeouts_remaining: int = 2,
    defteam_timeouts_remaining: int = 2,
    temp: float = 65.0,
    shotgun: bool = True,
    no_huddle: bool = False,
    epa_per_game_roll15: Optional[float] = None,
    success_rate_roll15: Optional[float] = None,
    points_per_game_roll15: Optional[float] = None,
    def_epa_per_game_roll15: Optional[float] = None,
    def_success_rate_roll15: Optional[float] = None,
    def_points_per_game_roll15: Optional[float] = None,
    model_path: Path = MODEL_PATH,
) -> float:
    """
    Predict 4th-down conversion probability for a live game situation.

    Rolling team stats default to league averages when not provided.

    Parameters
    ----------
    ydstogo : float
        Yards needed for a first down.
    yardline_100 : float
        Yards from opponent end zone (1 = goal line, 99 = own 1-yd line).
    score_differential : float
        Offense score minus defense score.
    game_seconds_remaining : float
        Seconds left in the game.
    qtr : int, optional
        Quarter (1-4; 5 = overtime). Default 4.
    half_seconds_remaining : float, optional
        Seconds left in the current half. Default 300.
    wp : float, optional
        Pre-snap win probability (0-1). Default 0.5.
    posteam_timeouts_remaining : int, optional
        Offense timeouts remaining (0-3). Default 2.
    defteam_timeouts_remaining : int, optional
        Defense timeouts remaining (0-3). Default 2.
    temp : float, optional
        Temperature in °F. Default 65.
    shotgun : bool, optional
        True if shotgun formation. Default True.
    no_huddle : bool, optional
        True if hurry-up / no-huddle. Default False.
    epa_per_game_roll15, success_rate_roll15, ... : float, optional
        Rolling offensive/defensive stats. Default to league averages.

    Returns
    -------
    float
        Estimated probability of converting the 4th down, in [0, 1].

    Examples
    --------
    >>> p = predict_conversion_prob(
    ...     ydstogo=2, yardline_100=35, score_differential=-3,
    ...     game_seconds_remaining=120, shotgun=True,
    ...     epa_per_game_roll15=0.12, def_epa_per_game_roll15=0.05,
    ... )
    >>> print(f"Conversion probability: {p:.1%}")
    """
    artifact     = load_model(model_path)
    model        = artifact["model"]
    feature_cols = artifact["features"]

    # Build a dict of all possible values
    all_vals: dict[str, float] = {
        "ydstogo":                    float(ydstogo),
        "qtr":                        float(qtr),
        "yardline_100":               float(yardline_100),
        "score_differential":         float(score_differential),
        "game_seconds_remaining":     float(game_seconds_remaining),
        "half_seconds_remaining":     float(half_seconds_remaining),
        "wp":                         float(wp),
        "posteam_timeouts_remaining": float(posteam_timeouts_remaining),
        "defteam_timeouts_remaining": float(defteam_timeouts_remaining),
        "temp":                       float(temp),
        "shotgun":                    float(shotgun),
        "no_huddle":                  float(no_huddle),
        "epa_per_game_roll15":
            float(epa_per_game_roll15) if epa_per_game_roll15 is not None
            else _LEAGUE_AVG["epa_per_game_roll15"],
        "success_rate_roll15":
            float(success_rate_roll15) if success_rate_roll15 is not None
            else _LEAGUE_AVG["success_rate_roll15"],
        "points_per_game_roll15":
            float(points_per_game_roll15) if points_per_game_roll15 is not None
            else _LEAGUE_AVG["points_per_game_roll15"],
        "def_epa_per_game_roll15":
            float(def_epa_per_game_roll15) if def_epa_per_game_roll15 is not None
            else _LEAGUE_AVG["def_epa_per_game_roll15"],
        "def_success_rate_roll15":
            float(def_success_rate_roll15) if def_success_rate_roll15 is not None
            else _LEAGUE_AVG["def_success_rate_roll15"],
        "def_points_per_game_roll15":
            float(def_points_per_game_roll15) if def_points_per_game_roll15 is not None
            else _LEAGUE_AVG["def_points_per_game_roll15"],
    }

    # Select only the features the saved model was trained on (in order)
    row = {f: all_vals[f] for f in feature_cols if f in all_vals}
    X   = pd.DataFrame([row])[feature_cols]

    return float(model.predict_proba(X)[0][1])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("4th Down Conversion Probability Model — Training")
    print("=" * 60)

    print("\n[1/3] Loading training data...")
    df = build_training_data()

    print("\n[2/3] Training (hyperparameter search + calibration)...")
    model, feature_cols, _ = train(df)
    save_model(model, feature_cols)

    print("\n[3/3] Example predictions (from saved model):")
    examples = [
        dict(ydstogo=1,  yardline_100=2,  score_differential=0,  game_seconds_remaining=900,
             shotgun=True,  wp=0.55, label="4th & 1 at the 2-yd line — tied, 2nd half"),
        dict(ydstogo=2,  yardline_100=35, score_differential=-3, game_seconds_remaining=120,
             shotgun=True,  wp=0.35, label="4th & 2 at opp 35 — trailing by 3, 2-min"),
        dict(ydstogo=4,  yardline_100=40, score_differential=0,  game_seconds_remaining=600,
             shotgun=True,  wp=0.50, label="4th & 4 at opp 40 — tied, 4th quarter"),
        dict(ydstogo=6,  yardline_100=50, score_differential=7,  game_seconds_remaining=1200,
             shotgun=False, wp=0.65, label="4th & 6 at midfield — leading by 7"),
        dict(ydstogo=10, yardline_100=60, score_differential=-7, game_seconds_remaining=60,
             shotgun=True,  wp=0.20, label="4th & 10 at own 40 — desperate, last min"),
        dict(ydstogo=1,  yardline_100=1,  score_differential=-1, game_seconds_remaining=5,
             shotgun=False, wp=0.45, label="4th & goal from 1 — down 1, final play"),
    ]

    print(f"\n  {'Scenario':<55} {'P(convert)':>10}")
    print("  " + "-" * 67)
    for ex in examples:
        label = ex.pop("label")
        prob  = predict_conversion_prob(**ex)
        print(f"  {label:<55} {prob:>9.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()