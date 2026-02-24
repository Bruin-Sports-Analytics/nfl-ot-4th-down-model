"""
Field Goal Make Probability Model
==================================
feature/fg-probability-model branch

Estimates P(FG made) given kick situation and kicker form.

Inputs (for inference)
----------------------
  kick_distance       : yards (int/float)
  is_dome             : bool  -- closed/dome stadium
  wind                : MPH   -- 0 for dome
  temp                : °F    -- 72 for dome
  fg_make_rate_roll6  : float -- kicker's rolling 6-game make rate (by distance bucket)
  surface_is_grass    : bool  -- natural grass vs. artificial

Output
------
  P(FG made)  in [0, 1]

Training Data
-------------
  All regular-season FG attempts 2016-2024 (from fg_attempts.parquet).
  Weather and stadium features joined from raw play-by-play.
  Rolling kicker make rate joined from team_rolling_kicker.parquet.

Usage
-----
  # Train and save model
  python -m src.models.fg_probability

  # Inference in Python
  from src.models.fg_probability import predict_fg_prob
  prob = predict_fg_prob(kick_distance=47, is_dome=False, wind=12,
                         temp=42, fg_make_rate_roll6=0.81)
  print(f"FG probability: {prob:.3f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data"
RAW = DATA_DIR / "raw" / "pbp_2016_2024.parquet"
PROCESSED = DATA_DIR / "processed"
MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "fg_prob_model.pkl"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "kick_distance",      # primary driver of FG probability
    "is_dome",            # controlled environment removes wind/temp effects
    "wind_adj",           # MPH; 0 for dome
    "temp_adj",           # °F;  72 for dome
    "fg_make_rate_roll6", # kicker rolling 6-game make rate at this distance range
    "surface_is_grass",   # natural grass vs artificial turf
]

TARGET = "fg_made"

# League-average FG make rate used when rolling kicker stat is unavailable
# (e.g., early season with no prior attempts in that distance bucket)
_LEAGUE_AVG_MAKE_RATE = 0.80

# Distance bucket boundaries must match make_special_teams.py
_DISTANCE_BUCKETS = [
    (0,  30, "0-30"),
    (31, 40, "31-40"),
    (41, 50, "41-50"),
    (51, 55, "51-55"),
    (56, 999, "56+"),
]


def _distance_to_bucket(yards: float) -> str:
    """Map a kick distance to the same bucket labels used in make_special_teams.py."""
    for lo, hi, label in _DISTANCE_BUCKETS:
        if lo <= yards <= hi:
            return label
    return "56+"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_VIEW = DATA_DIR / "view"


def _load_table(name: str) -> pl.DataFrame:
    """
    Load a processed table by name.  Tries data/processed/<name>.parquet first;
    falls back to data/view/<name>.csv if the parquet is missing (e.g. the
    special-teams parquets haven't been regenerated on this branch yet).
    """
    parquet_path = PROCESSED / f"{name}.parquet"
    if parquet_path.exists():
        return pl.read_parquet(str(parquet_path))

    csv_path = _VIEW / f"{name}.csv"
    if csv_path.exists():
        print(f"  [fallback] {name}.parquet not found — reading {csv_path.name}")
        return pl.read_csv(str(csv_path))

    raise FileNotFoundError(
        f"Neither {parquet_path} nor {csv_path} exist.\n"
        f"Run:  python -m src.data.make_special_teams"
    )


# ---------------------------------------------------------------------------
# Data loading / feature engineering
# ---------------------------------------------------------------------------

def build_training_data() -> pl.DataFrame:
    """
    Join FG attempts with weather, stadium type, and rolling kicker stats
    to produce the full model-ready training frame.

    Returns
    -------
    pl.DataFrame
        One row per FG attempt with FEATURE_COLS + TARGET populated.
    """
    # -- FG attempts (parquet preferred, CSV fallback) -----------------------
    fg = _load_table("fg_attempts")

    # Normalise fg_made: CSV stores it as "true"/"false" strings
    if fg["fg_made"].dtype == pl.String:
        fg = fg.with_columns(
            (pl.col("fg_made").str.to_lowercase() == "true").alias("fg_made")
        )

    # -- Weather / stadium from raw PBP --------------------------------------
    # Only scan the columns we need to keep memory usage low
    weather_want = ["game_id", "wind", "temp", "roof", "surface"]
    raw_schema = set(pl.scan_parquet(str(RAW)).collect_schema().names())
    weather_want = [c for c in weather_want if c in raw_schema]

    weather = (
        pl.scan_parquet(str(RAW))
        .select(weather_want)
        .unique(subset=["game_id"])
        .collect()
    )

    # -- Rolling kicker stats (parquet preferred, CSV fallback) --------------
    kicker_roll = _load_table("team_rolling_kicker")

    # Rename so join keys align with fg_attempts column names
    kicker_roll = kicker_roll.rename({"team": "posteam"})

    # -- Join ----------------------------------------------------------------
    df = fg.join(weather, on="game_id", how="left")

    df = df.join(
        kicker_roll.select([
            "season", "week", "game_id", "posteam", "distance_bucket",
            "fg_make_rate_roll6",
        ]),
        on=["season", "week", "game_id", "posteam", "distance_bucket"],
        how="left",
    )

    # -- Feature engineering -------------------------------------------------
    df = df.with_columns([
        # is_dome: fully enclosed dome or retractable roof closed
        pl.col("roof")
          .is_in(["dome", "closed"])
          .fill_null(False)
          .cast(pl.Int8)
          .alias("is_dome"),

        # Natural grass indicator
        pl.col("surface")
          .str.to_lowercase()
          .str.contains("grass")
          .fill_null(False)
          .cast(pl.Int8)
          .alias("surface_is_grass"),

        # Target variable as integer
        pl.col("fg_made").cast(pl.Int8),
    ])

    # Dome-adjusted weather: 0 mph wind, 72°F inside domes
    df = df.with_columns([
        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(0.0))
          .otherwise(pl.col("wind").cast(pl.Float64))
          .alias("wind_adj"),

        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(72.0))
          .otherwise(pl.col("temp").cast(pl.Float64))
          .alias("temp_adj"),

        # Fill missing rolling rate with league average
        pl.col("fg_make_rate_roll6")
          .fill_null(_LEAGUE_AVG_MAKE_RATE)
          .alias("fg_make_rate_roll6"),
    ])

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(
    df: pl.DataFrame,
    model_type: Literal["logistic", "gradient_boost"] = "logistic",
    test_seasons: list[int] | None = None,
) -> tuple:
    """
    Train the FG probability model and report evaluation metrics.

    Parameters
    ----------
    df : pl.DataFrame
        Output of build_training_data().
    model_type : {"logistic", "gradient_boost"}
        Which base learner to use.
    test_seasons : list[int] | None
        Seasons to hold out for final evaluation (temporal split).
        Defaults to [2023, 2024].

    Returns
    -------
    (model, metrics_dict)
    """
    if test_seasons is None:
        test_seasons = [2023, 2024]

    required = FEATURE_COLS + [TARGET, "season"]
    available = [c for c in required if c in df.columns]
    clean = df.select(available).drop_nulls()

    print(f"Total FG attempts after dropping nulls: {len(clean):,}")
    print(f"Overall make rate: {clean[TARGET].mean():.3f}")

    # Temporal train / test split
    train_mask = ~clean["season"].is_in(test_seasons)
    train_df = clean.filter(train_mask)
    test_df  = clean.filter(~train_mask)

    print(f"\nTrain seasons  : {clean.filter(train_mask)['season'].min()}"
          f" – {train_df['season'].max()}  ({len(train_df):,} attempts)")
    print(f"Test seasons   : {test_seasons}  ({len(test_df):,} attempts)")

    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df[TARGET].to_numpy()
    X_test  = test_df.select(FEATURE_COLS).to_numpy()
    y_test  = test_df[TARGET].to_numpy()

    # -- Build model ---------------------------------------------------------
    if model_type == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ])
    elif model_type == "gradient_boost":
        # Wrap with Platt scaling for well-calibrated probabilities
        base = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # -- Cross-validation on training set ------------------------------------
    print(f"\nCross-validating {model_type} model (5-fold, training seasons)...")
    auc_cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    brier_cv = cross_val_score(model, X_train, y_train, cv=5,
                               scoring="neg_brier_score")
    print(f"  CV ROC-AUC    : {auc_cv.mean():.4f}  ±  {auc_cv.std():.4f}")
    print(f"  CV Brier score: {(-brier_cv).mean():.4f}  ±  {(-brier_cv).std():.4f}")

    # -- Fit on all training data -------------------------------------------
    model.fit(X_train, y_train)

    # -- Hold-out evaluation ------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_roc_auc"    : roc_auc_score(y_test, y_prob),
        "test_brier_score": brier_score_loss(y_test, y_prob),
        "test_log_loss"   : log_loss(y_test, y_prob),
        "test_n"          : len(y_test),
        "test_make_rate"  : float(y_test.mean()),
    }
    print(f"\nHold-out test ({test_seasons}):")
    print(f"  ROC-AUC     : {metrics['test_roc_auc']:.4f}")
    print(f"  Brier score : {metrics['test_brier_score']:.4f}")
    print(f"  Log-loss    : {metrics['test_log_loss']:.4f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: Path = MODEL_PATH) -> None:
    """Persist model and feature list to disk."""
    joblib.dump(
        {"model": model, "features": FEATURE_COLS, "target": TARGET},
        path,
    )
    print(f"\nModel saved → {path}")


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load persisted model artifact."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------

def predict_fg_prob(
    kick_distance: float,
    is_dome: bool,
    wind: float,
    temp: float,
    fg_make_rate_roll6: float = _LEAGUE_AVG_MAKE_RATE,
    surface_is_grass: bool = True,
    model_path: Path = MODEL_PATH,
) -> float:
    """
    Predict field goal make probability for a live game situation.

    Parameters
    ----------
    kick_distance : float
        Distance of the attempt in yards (e.g., 47 for a kick from the 30).
    is_dome : bool
        True if the stadium has a fixed dome or retractable roof currently
        closed (i.e., wind and temperature are controlled).
    wind : float
        Wind speed in MPH. Pass 0 if is_dome is True.
    temp : float
        Temperature in °F. Pass 72 (or any comfortable indoor temp)
        if is_dome is True.
    fg_make_rate_roll6 : float, optional
        Kicker's rolling 6-game FG make rate for the matching distance bucket.
        Look up from team_rolling_kicker.parquet using (season, week, team,
        distance_bucket).  Defaults to the league average (0.80).
    surface_is_grass : bool, optional
        True for natural grass, False for artificial turf. Default True.
    model_path : Path, optional
        Path to the saved model artifact (pkl).

    Returns
    -------
    float
        Estimated probability of making the field goal, in [0, 1].

    Examples
    --------
    >>> from src.models.fg_probability import predict_fg_prob
    >>> predict_fg_prob(kick_distance=47, is_dome=False, wind=12,
    ...                 temp=42, fg_make_rate_roll6=0.81)
    0.742   # example output
    """
    artifact = load_model(model_path)
    model    = artifact["model"]

    # Dome: zero out environmental factors
    wind_adj = 0.0  if is_dome else float(wind)
    temp_adj = 72.0 if is_dome else float(temp)

    X = np.array([[
        float(kick_distance),
        int(is_dome),
        wind_adj,
        temp_adj,
        float(fg_make_rate_roll6),
        int(surface_is_grass),
    ]])

    return float(model.predict_proba(X)[0, 1])


def predict_fg_prob_batch(
    situations: list[dict],
    model_path: Path = MODEL_PATH,
) -> list[float]:
    """
    Vectorized version of predict_fg_prob for multiple situations at once.

    Parameters
    ----------
    situations : list[dict]
        Each dict must contain the same keys as predict_fg_prob parameters.

    Returns
    -------
    list[float]
        P(FG made) for each situation.

    Examples
    --------
    >>> probs = predict_fg_prob_batch([
    ...     {"kick_distance": 25, "is_dome": True,  "wind": 0,  "temp": 72},
    ...     {"kick_distance": 55, "is_dome": False, "wind": 20, "temp": 30},
    ... ])
    """
    artifact = load_model(model_path)
    model    = artifact["model"]

    rows = []
    for s in situations:
        dome     = s.get("is_dome", False)
        wind_adj = 0.0  if dome else float(s.get("wind", 0))
        temp_adj = 72.0 if dome else float(s.get("temp", 72))
        rows.append([
            float(s["kick_distance"]),
            int(dome),
            wind_adj,
            temp_adj,
            float(s.get("fg_make_rate_roll6", _LEAGUE_AVG_MAKE_RATE)),
            int(s.get("surface_is_grass", True)),
        ])

    X = np.array(rows)
    return model.predict_proba(X)[:, 1].tolist()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Field Goal Probability Model — Training")
    print("=" * 60)

    print("\n[1/3] Building training dataset...")
    df = build_training_data()
    print(f"      Dataset shape: {df.shape}")

    print("\n[2/3] Training Logistic Regression model...")
    model, metrics = train(df, model_type="logistic", test_seasons=[2023, 2024])
    save_model(model)

    print("\n[3/3] Example predictions (from saved model):")
    examples = [
        dict(kick_distance=20, is_dome=True,  wind=0,  temp=72,
             fg_make_rate_roll6=0.95, surface_is_grass=False,
             label="Short indoor 20 yd — elite kicker"),
        dict(kick_distance=38, is_dome=False, wind=5,  temp=65,
             fg_make_rate_roll6=0.83, surface_is_grass=True,
             label="Mid-range outdoor 38 yd — avg kicker, mild day"),
        dict(kick_distance=47, is_dome=False, wind=12, temp=42,
             fg_make_rate_roll6=0.81, surface_is_grass=True,
             label="Mid-long outdoor 47 yd — avg kicker, cool/windy"),
        dict(kick_distance=55, is_dome=False, wind=20, temp=30,
             fg_make_rate_roll6=0.72, surface_is_grass=False,
             label="Long outdoor 55 yd — struggling kicker, cold/windy"),
        dict(kick_distance=62, is_dome=False, wind=25, temp=20,
             fg_make_rate_roll6=0.68, surface_is_grass=False,
             label="Very long 62 yd — cold blizzard game"),
    ]

    print(f"\n  {'Scenario':<52} {'P(make)':>8}")
    print("  " + "-" * 62)
    for ex in examples:
        label = ex.pop("label")
        prob  = predict_fg_prob(**ex)
        print(f"  {label:<52} {prob:>7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
