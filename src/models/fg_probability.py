"""
Field Goal Make Probability Model
==================================
feature/fg-probability-model branch

Estimates P(FG made) given kick situation and kicker form.

Inputs (for inference)
----------------------
  kick_distance           : yards (int/float)
  is_dome                 : bool  -- closed/dome stadium
  wind                    : MPH   -- average wind speed; 0 for dome
  wind_gust               : MPH   -- peak gust speed; defaults to wind if unknown
  temp                    : °F    -- 72 for dome
  is_precipitation        : bool  -- rain / snow / fog present
  fg_make_rate_roll6      : float -- kicker's rolling 6-game make rate (by distance bucket)
  kicker_career_make_rate : float -- kicker's career make rate (by distance bucket)
  surface_is_grass        : bool  -- natural grass vs. artificial
  altitude_ft             : float -- stadium altitude in feet (default 0 = sea level)
  game_seconds_remaining  : float -- seconds left in game
  score_differential      : float -- posteam score minus defteam score
  is_overtime             : bool  -- True if attempt is in OT
  kicking_at_home         : bool  -- kicking team is the home team
  iced_kicker             : bool  -- defense called timeout on immediately prior play

Output
------
  P(FG made)  in [0, 1]

Training Data
-------------
  All regular-season unblocked FG attempts 2016-2024 (from fg_attempts.parquet).
  Blocked kicks are excluded: a block is a defensive event that does not reflect
  kicker accuracy or situational factors the model cares about.
  Weather and stadium features joined from raw play-by-play.
  Rolling kicker make rate joined from team_rolling_kicker.parquet.
  Career kicker make rate joined from kicker_career_make_rate.parquet.

  Train/test split: stratified by season — each season contributes
  ~80% to train and ~20% to test, so every era of play is represented
  in both sets.

Model
-----
  Gradient Boosting Classifier with Platt scaling calibration.
  Hyperparameters selected via RandomizedSearchCV (30 iterations, 5-fold CV,
  optimising Brier score).

Usage
-----
  # Rebuild processed data first (if buckets or icing changed)
  python -m src.data.make_special_teams

  # Train and save model
  python -m src.models.fg_probability

  # Inference in Python
  from src.models.fg_probability import predict_fg_prob
  prob = predict_fg_prob(kick_distance=47, is_dome=False, wind=12,
                         temp=42, fg_make_rate_roll6=0.81)
  print(f"FG probability: {prob:.3f}")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
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
# Feature configuration  (17 features)
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "kick_distance",            # primary driver of FG probability
    "is_dome",                  # controlled environment removes wind/temp effects
    "wind_adj",                 # avg wind MPH; 0 for dome
    "wind_x_distance",          # avg wind × distance interaction
    "wind_gust",                # peak gust MPH (parsed from weather text); 0 for dome
    "wind_gust_x_distance",     # gust × distance interaction — the kick-moment physics
    "temp_adj",                 # °F; 72 for dome
    "is_precipitation",         # rain / snow / fog (0/1)
    "fg_make_rate_roll6",       # kicker rolling 6-game make rate (recent form)
    "kicker_career_make_rate",  # kicker career make rate (stable baseline)
    "surface_is_grass",         # natural grass vs artificial turf
    "altitude_ft",              # stadium altitude in feet (Denver = 5,280; others ≈ 0)
    "game_seconds_remaining",   # pressure: time remaining in game
    "score_differential",       # pressure: trailing/leading situation
    "is_overtime",              # maximum pressure — sudden death
    "kicking_at_home",          # kicking team is playing at home
    "iced_kicker",              # defense called timeout on immediately prior play
]

TARGET = "fg_made"

# ---------------------------------------------------------------------------
# Distance bucket configuration
# ---------------------------------------------------------------------------
# 5-yard wide in short/mid range; 3-yard wide at 56+ so that long-range
# buckets are still granular while having enough sample for rolling rates.
# Must stay in sync with make_special_teams.py.
_DISTANCE_BUCKETS = [
    (0,  30, "0-30"),
    (31, 35, "31-35"),
    (36, 40, "36-40"),
    (41, 45, "41-45"),
    (46, 50, "46-50"),
    (51, 55, "51-55"),
    (56, 58, "56-58"),
    (59, 61, "59-61"),
    (62, 64, "62-64"),
    (65, 999, "65+"),
]

# Distance-aware league-average make rates used when rolling/career kicker stat is
# unavailable (e.g., early season / first attempt in a bucket).
_LEAGUE_AVG_BY_BUCKET: dict[str, float] = {
    "0-30":  0.94,
    "31-35": 0.92,
    "36-40": 0.88,
    "41-45": 0.83,
    "46-50": 0.77,
    "51-55": 0.67,
    "56-58": 0.57,
    "59-61": 0.46,
    "62-64": 0.34,
    "65+":   0.22,
}
_LEAGUE_AVG_MAKE_RATE = 0.80  # global fallback if bucket not recognised

# ---------------------------------------------------------------------------
# Altitude lookup: only Denver has a meaningful effect (~5,280 ft).
# ---------------------------------------------------------------------------
_HOME_TEAM_ALTITUDE_FT: dict[str, float] = {
    "DEN": 5280.0,
}

# Regex for parsing gust speed and precipitation from the weather text column.
# Format examples:
#   "Clear Temp: 79° F, Humidity: 40 %, Wind: NW 17 mph gusting to 28 mph"
#   "Rain Temp: 52° F, Humidity: 88%, Wind: N 12 mph"
_GUST_RE = re.compile(r"gusting to (\d+)", re.IGNORECASE)
_PRECIP_RE = re.compile(r"rain|snow|sleet|fog|drizzle|precip|flurr", re.IGNORECASE)


def _distance_to_bucket(yards: float) -> str:
    for lo, hi, label in _DISTANCE_BUCKETS:
        if lo <= yards <= hi:
            return label
    return "65+"


def _bucket_league_avg(bucket: str) -> float:
    return _LEAGUE_AVG_BY_BUCKET.get(bucket, _LEAGUE_AVG_MAKE_RATE)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_VIEW = DATA_DIR / "view"


def _load_table(name: str) -> pl.DataFrame:
    """
    Load a processed table by name.  Tries data/processed/<name>.parquet first;
    falls back to data/view/<name>.csv if the parquet is missing.
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
    Join FG attempts with weather, stadium, rolling kicker stats, and career stats
    to produce the full model-ready training frame.

    Blocked kicks are excluded: a block is a defensive event (bad snap timing,
    fast rush) and should not count against the kicker's accuracy model.

    Returns
    -------
    pl.DataFrame  — one row per unblocked FG attempt with FEATURE_COLS + TARGET.
    """
    # -- FG attempts (parquet preferred, CSV fallback) -----------------------
    fg = _load_table("fg_attempts")

    # Normalise fg_made: CSV stores it as "true"/"false" strings
    if fg["fg_made"].dtype == pl.String:
        fg = fg.with_columns(
            (pl.col("fg_made").str.to_lowercase() == "true").alias("fg_made")
        )
    if fg["fg_blocked"].dtype == pl.String:
        fg = fg.with_columns(
            (pl.col("fg_blocked").str.to_lowercase() == "true").alias("fg_blocked")
        )

    # Exclude blocked kicks — they are defensive events, not kicker failures
    fg = fg.filter(~pl.col("fg_blocked"))

    # Safety: add iced_kicker = 0 if the column is absent (old CSV fallback)
    if "iced_kicker" not in fg.columns:
        fg = fg.with_columns(pl.lit(0).cast(pl.Int8).alias("iced_kicker"))

    # -- Weather / stadium from raw PBP --------------------------------------
    weather_want = ["game_id", "wind", "temp", "roof", "surface", "weather"]
    raw_schema = set(pl.scan_parquet(str(RAW)).collect_schema().names())
    weather_want = [c for c in weather_want if c in raw_schema]

    weather = (
        pl.scan_parquet(str(RAW))
        .select(weather_want)
        .unique(subset=["game_id"])
        .collect()
    )

    # -- Rolling kicker stats -----------------------------------------------
    kicker_roll = _load_table("team_rolling_kicker")
    kicker_roll = kicker_roll.rename({"team": "posteam"})

    # -- Career kicker stats ------------------------------------------------
    career_stats_path = PROCESSED / "kicker_career_make_rate.parquet"
    if career_stats_path.exists():
        career = pl.read_parquet(str(career_stats_path))
    else:
        print("  [warning] kicker_career_make_rate.parquet not found; career rate will use league avg")
        career = None

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

    if career is not None:
        df = df.join(
            career.select(["game_id", "play_id", "kicker_career_make_rate"]),
            on=["game_id", "play_id"],
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("kicker_career_make_rate"))

    # -- Core feature engineering -------------------------------------------
    df = df.with_columns([
        # Dome indicator
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

        # Overtime flag (qtr == 5)
        (pl.col("qtr") == 5)
          .fill_null(False)
          .cast(pl.Int8)
          .alias("is_overtime"),

        # Home team extracted from game_id: format "YYYY_WW_AWAY_HOME"
        pl.col("game_id").str.split("_").list.get(3).alias("home_team"),
    ])

    # -- Dome-adjusted weather -----------------------------------------------
    df = df.with_columns([
        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(0.0))
          .otherwise(pl.col("wind").cast(pl.Float64))
          .alias("wind_adj"),

        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(72.0))
          .otherwise(pl.col("temp").cast(pl.Float64))
          .alias("temp_adj"),
    ])

    # -- Wind gust and precipitation (parsed from weather text) --------------
    # wind_gust: peak gust from "gusting to X mph"; falls back to avg wind if absent.
    # is_precipitation: any mention of rain / snow / fog in the weather description.
    df = df.with_columns([
        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(0.0))
          .when(
              pl.col("weather").is_not_null() &
              pl.col("weather").str.contains(r"(?i)gusting to \d+")
          )
          .then(
              pl.col("weather")
                .str.extract(r"gusting to (\d+)", 1)
                .cast(pl.Float64)
          )
          .otherwise(pl.col("wind_adj"))
          .alias("wind_gust"),

        pl.when(pl.col("is_dome") == 1)
          .then(pl.lit(0).cast(pl.Int8))
          .otherwise(
              pl.col("weather")
                .str.to_lowercase()
                .str.contains(r"rain|snow|sleet|fog|drizzle|precip|flurr")
                .fill_null(False)
                .cast(pl.Int8)
          )
          .alias("is_precipitation"),
    ])

    # -- Interaction terms ---------------------------------------------------
    df = df.with_columns([
        (pl.col("wind_adj")  * pl.col("kick_distance")).alias("wind_x_distance"),
        (pl.col("wind_gust") * pl.col("kick_distance")).alias("wind_gust_x_distance"),
    ])

    # -- Altitude -----------------------------------------------------------
    df = df.with_columns([
        pl.col("home_team")
          .map_elements(
              lambda t: _HOME_TEAM_ALTITUDE_FT.get(t, 0.0),
              return_dtype=pl.Float64,
          )
          .alias("altitude_ft"),
    ])

    # -- Kicking at home ---------------------------------------------------
    df = df.with_columns([
        (pl.col("posteam") == pl.col("home_team"))
          .cast(pl.Int8)
          .alias("kicking_at_home"),
    ])

    # -- Distance-aware fills for missing kicker rates ----------------------
    df = df.with_columns([
        pl.struct(["fg_make_rate_roll6", "distance_bucket"])
          .map_elements(
              lambda r: r["fg_make_rate_roll6"]
              if r["fg_make_rate_roll6"] is not None
              else _bucket_league_avg(r["distance_bucket"]),
              return_dtype=pl.Float64,
          )
          .alias("fg_make_rate_roll6"),

        pl.struct(["kicker_career_make_rate", "distance_bucket"])
          .map_elements(
              lambda r: r["kicker_career_make_rate"]
              if r["kicker_career_make_rate"] is not None
              else _bucket_league_avg(r["distance_bucket"]),
              return_dtype=pl.Float64,
          )
          .alias("kicker_career_make_rate"),
    ])

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(
    df: pl.DataFrame,
    model_type: Literal["logistic", "gradient_boost"] = "gradient_boost",
    test_size: float = 0.2,
) -> tuple:
    """
    Train the FG probability model and report evaluation metrics.

    Parameters
    ----------
    df : pl.DataFrame
        Output of build_training_data().
    model_type : {"logistic", "gradient_boost"}
        Which base learner to use. Defaults to gradient_boost.
    test_size : float
        Fraction of each season's attempts held out for evaluation.

    Returns
    -------
    (model, metrics_dict)
    """
    required = FEATURE_COLS + [TARGET, "season"]
    available = [c for c in required if c in df.columns]
    clean = df.select(available).drop_nulls()

    print(f"Total unblocked FG attempts after dropping nulls: {len(clean):,}")
    print(f"Overall make rate: {clean[TARGET].mean():.3f}")

    # Stratified split: each season contributes ~80% train / ~20% test
    X_all = clean.select(FEATURE_COLS).to_numpy()
    y_all = clean[TARGET].to_numpy()
    seasons_arr = clean["season"].to_numpy()

    X_train, X_test, y_train, y_test, _, seasons_test = train_test_split(
        X_all, y_all, seasons_arr,
        test_size=test_size,
        random_state=42,
        stratify=seasons_arr,
    )

    print(f"\nTrain: {len(y_train):,} attempts  |  Test: {len(y_test):,} attempts")
    print("Season distribution in test set:")
    unique_seasons, counts = np.unique(seasons_test, return_counts=True)
    for s, c in zip(unique_seasons, counts):
        print(f"  {s}: {c} attempts")

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
        print(f"\nRunning RandomizedSearchCV (30 iterations, 5-fold) to tune GBM...")
        base = GradientBoostingClassifier(random_state=42)
        param_dist = {
            "n_estimators":     [300, 400, 500, 600],
            "max_depth":        [2, 3, 4],
            "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample":        [0.7, 0.8, 0.9],
            "min_samples_leaf": [10, 15, 20, 30],
            "max_features":     [0.6, 0.7, 0.8, "sqrt"],
        }
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=30,
            scoring="neg_brier_score",
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"Best hyperparameters: {search.best_params_}")
        print(f"Best CV Brier score:  {-search.best_score_:.4f}")

        best_base = GradientBoostingClassifier(**search.best_params_, random_state=42)
        model = CalibratedClassifierCV(best_base, cv=5, method="sigmoid")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # -- Cross-validation on training set ------------------------------------
    print(f"\nCross-validating {model_type} model (5-fold, training data)...")
    auc_cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    brier_cv = cross_val_score(model, X_train, y_train, cv=5,
                               scoring="neg_brier_score")
    print(f"  CV ROC-AUC    : {auc_cv.mean():.4f}  ±  {auc_cv.std():.4f}")
    print(f"  CV Brier score: {(-brier_cv).mean():.4f}  ±  {(-brier_cv).std():.4f}")

    # -- Fit on all training data -------------------------------------------
    model.fit(X_train, y_train)

    # -- Feature importance (GBM only) --------------------------------------
    if model_type == "gradient_boost" and hasattr(model, "calibrated_classifiers_"):
        importances = np.mean(
            [cc.estimator.feature_importances_
             for cc in model.calibrated_classifiers_],
            axis=0,
        )
        print("\nFeature importances (GBM, averaged across calibration folds):")
        for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 200)
            print(f"  {feat:<30} {imp:.4f}  {bar}")

    # -- Hold-out evaluation ------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_roc_auc"    : roc_auc_score(y_test, y_prob),
        "test_brier_score": brier_score_loss(y_test, y_prob),
        "test_log_loss"   : log_loss(y_test, y_prob),
        "test_n"          : len(y_test),
        "test_make_rate"  : float(y_test.mean()),
    }
    print(f"\nHold-out test (stratified 20% across all seasons):")
    print(f"  ROC-AUC     : {metrics['test_roc_auc']:.4f}")
    print(f"  Brier score : {metrics['test_brier_score']:.4f}")
    print(f"  Log-loss    : {metrics['test_log_loss']:.4f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: Path = MODEL_PATH) -> None:
    joblib.dump(
        {"model": model, "features": FEATURE_COLS, "target": TARGET},
        path,
    )
    print(f"\nModel saved → {path}")


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------

def predict_fg_prob(
    kick_distance: float,
    is_dome: bool,
    wind: float,
    temp: float,
    fg_make_rate_roll6: float | None = None,
    surface_is_grass: bool = True,
    altitude_ft: float = 0.0,
    game_seconds_remaining: float = 1800.0,
    score_differential: float = 0.0,
    is_overtime: bool = False,
    wind_gust: float | None = None,
    is_precipitation: bool = False,
    kicker_career_make_rate: float | None = None,
    kicking_at_home: bool = False,
    iced_kicker: bool = False,
    model_path: Path = MODEL_PATH,
) -> float:
    """
    Predict field goal make probability for a live game situation.

    Parameters
    ----------
    kick_distance : float
        Distance of the attempt in yards.
    is_dome : bool
        True if the stadium has a fixed dome or retractable roof closed.
    wind : float
        Average wind speed in MPH. Pass 0 if is_dome is True.
    temp : float
        Temperature in °F. Pass 72 if is_dome is True.
    fg_make_rate_roll6 : float, optional
        Kicker's rolling 6-game FG make rate. If None, distance-aware league avg used.
    surface_is_grass : bool, optional
        True for natural grass, False for artificial turf.
    altitude_ft : float, optional
        Stadium altitude in feet. Default 0. Use 5280 for Denver.
    game_seconds_remaining : float, optional
        Seconds remaining in the game. Default 1800 (mid-game).
    score_differential : float, optional
        Kicking team score minus opponent score. Negative = trailing.
    is_overtime : bool, optional
        True if the attempt is in overtime.
    wind_gust : float, optional
        Peak gust speed in MPH. Defaults to wind (no gust info available).
    is_precipitation : bool, optional
        True if rain, snow, fog, or similar conditions are present.
    kicker_career_make_rate : float, optional
        Kicker's career FG make rate for this distance bucket. If None, league avg used.
    kicking_at_home : bool, optional
        True if the kicking team is the home team.
    iced_kicker : bool, optional
        True if the defense called timeout on the immediately preceding play.
    model_path : Path, optional
        Path to the saved model artifact.

    Returns
    -------
    float
        Estimated probability of making the field goal, in [0, 1].

    Examples
    --------
    >>> prob = predict_fg_prob(kick_distance=47, is_dome=False, wind=12, temp=42,
    ...                        fg_make_rate_roll6=0.81, wind_gust=20)
    """
    artifact = load_model(model_path)
    model    = artifact["model"]

    wind_adj      = 0.0  if is_dome else float(wind)
    temp_adj      = 72.0 if is_dome else float(temp)
    wind_gust_val = 0.0  if is_dome else float(wind_gust if wind_gust is not None else wind)
    wind_x_distance      = wind_adj      * float(kick_distance)
    wind_gust_x_distance = wind_gust_val * float(kick_distance)

    bucket = _distance_to_bucket(float(kick_distance))
    if fg_make_rate_roll6 is None:
        fg_make_rate_roll6 = _bucket_league_avg(bucket)
    if kicker_career_make_rate is None:
        kicker_career_make_rate = _bucket_league_avg(bucket)

    X = np.array([[
        float(kick_distance),
        int(is_dome),
        wind_adj,
        wind_x_distance,
        wind_gust_val,
        wind_gust_x_distance,
        temp_adj,
        int(is_precipitation and not is_dome),
        float(fg_make_rate_roll6),
        float(kicker_career_make_rate),
        int(surface_is_grass),
        float(altitude_ft),
        float(game_seconds_remaining),
        float(score_differential),
        int(is_overtime),
        int(kicking_at_home),
        int(iced_kicker),
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
        Each dict may contain the same keys as predict_fg_prob parameters.
        Missing keys use the same defaults as predict_fg_prob.

    Returns
    -------
    list[float]  — P(FG made) for each situation.
    """
    artifact = load_model(model_path)
    model    = artifact["model"]

    rows = []
    for s in situations:
        dome      = s.get("is_dome", False)
        dist      = float(s["kick_distance"])
        wind_adj  = 0.0  if dome else float(s.get("wind", 0))
        temp_adj  = 72.0 if dome else float(s.get("temp", 72))
        gust_raw  = s.get("wind_gust", s.get("wind", 0))
        gust      = 0.0  if dome else float(gust_raw)
        wind_x_distance      = wind_adj * dist
        wind_gust_x_distance = gust     * dist

        bucket = _distance_to_bucket(dist)
        roll   = s.get("fg_make_rate_roll6")
        career = s.get("kicker_career_make_rate")
        if roll   is None: roll   = _bucket_league_avg(bucket)
        if career is None: career = _bucket_league_avg(bucket)

        rows.append([
            dist,
            int(dome),
            wind_adj,
            wind_x_distance,
            gust,
            wind_gust_x_distance,
            temp_adj,
            int(s.get("is_precipitation", False) and not dome),
            float(roll),
            float(career),
            int(s.get("surface_is_grass", True)),
            float(s.get("altitude_ft", 0.0)),
            float(s.get("game_seconds_remaining", 1800.0)),
            float(s.get("score_differential", 0.0)),
            int(s.get("is_overtime", False)),
            int(s.get("kicking_at_home", False)),
            int(s.get("iced_kicker", False)),
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

    print("\n[2/3] Training Gradient Boosting model (with hyperparameter tuning)...")
    model, metrics = train(df, model_type="gradient_boost")
    save_model(model)

    print("\n[3/3] Example predictions (from saved model):")
    examples = [
        dict(kick_distance=20, is_dome=True,  wind=0,  wind_gust=0,  temp=72,
             is_precipitation=False, fg_make_rate_roll6=0.95, kicker_career_make_rate=0.97,
             surface_is_grass=False, altitude_ft=0, game_seconds_remaining=1800,
             score_differential=0, kicking_at_home=True, iced_kicker=False,
             label="Short indoor 20 yd — elite kicker"),
        dict(kick_distance=38, is_dome=False, wind=5,  wind_gust=8,  temp=65,
             is_precipitation=False, fg_make_rate_roll6=0.83, kicker_career_make_rate=0.85,
             surface_is_grass=True,  altitude_ft=0, game_seconds_remaining=900,
             score_differential=0, kicking_at_home=True, iced_kicker=False,
             label="Mid-range 38 yd — avg kicker, mild day"),
        dict(kick_distance=47, is_dome=False, wind=12, wind_gust=18, temp=42,
             is_precipitation=False, fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82,
             surface_is_grass=True,  altitude_ft=0, game_seconds_remaining=300,
             score_differential=-3, kicking_at_home=False, iced_kicker=False,
             label="47 yd — trailing by 3, final mins, away"),
        dict(kick_distance=47, is_dome=False, wind=12, wind_gust=18, temp=42,
             is_precipitation=False, fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82,
             surface_is_grass=True,  altitude_ft=5280, game_seconds_remaining=300,
             score_differential=-3, kicking_at_home=True, iced_kicker=False,
             label="Same 47 yd — but Denver (altitude + home)"),
        dict(kick_distance=47, is_dome=False, wind=12, wind_gust=18, temp=42,
             is_precipitation=False, fg_make_rate_roll6=0.81, kicker_career_make_rate=0.82,
             surface_is_grass=True,  altitude_ft=0, game_seconds_remaining=5,
             score_differential=-3, kicking_at_home=False, iced_kicker=True,
             label="Same 47 yd — ICED, last play of game"),
        dict(kick_distance=55, is_dome=False, wind=20, wind_gust=30, temp=30,
             is_precipitation=True,  fg_make_rate_roll6=0.72, kicker_career_make_rate=0.74,
             surface_is_grass=False, altitude_ft=0, game_seconds_remaining=120,
             score_differential=-3, kicking_at_home=False, iced_kicker=False,
             label="55 yd — cold/snowy, gusting to 30, struggling kicker"),
        dict(kick_distance=60, is_dome=False, wind=25, wind_gust=35, temp=20,
             is_precipitation=True,  fg_make_rate_roll6=0.68, kicker_career_make_rate=0.70,
             surface_is_grass=False, altitude_ft=0, game_seconds_remaining=5,
             score_differential=-3, kicking_at_home=False, iced_kicker=False,
             label="60 yd — blizzard, gusting to 35, last play"),
        dict(kick_distance=63, is_dome=False, wind=10, wind_gust=14, temp=55,
             is_precipitation=False, fg_make_rate_roll6=None, kicker_career_make_rate=None,
             surface_is_grass=True,  altitude_ft=0, game_seconds_remaining=60,
             score_differential=0, kicking_at_home=False, iced_kicker=False, is_overtime=True,
             label="63 yd OT — no kicker history, neutral weather"),
    ]

    print(f"\n  {'Scenario':<58} {'P(make)':>8}")
    print("  " + "-" * 68)
    for ex in examples:
        label = ex.pop("label")
        prob  = predict_fg_prob(**ex)
        print(f"  {label:<58} {prob:>7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
