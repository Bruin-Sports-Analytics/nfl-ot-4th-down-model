"""
Punt Outcome Model
==================
submodel branch

Predicts the opponent's starting field position (yardline_100) after a punt,
given the punt location and rolling punter quality metrics.

Inputs (for inference)
----------------------
  yardline_100              : int/float — yards from opponent end zone (where the punt occurs)
  punt_distance_roll6       : float — team's rolling 6-game average punt distance
  inside_twenty_rate_roll6  : float — team's rolling 6-game rate of punts landing inside the 20

Output
------
  opponent_start : float — predicted opponent starting yardline_100 after the punt

Training Data
-------------
  All regular-season punt attempts from fourth_down_with_features.csv (2016-2024).
  Punts with kick_distance <= 0 or opponent_start > 100 are excluded.
  Rolling punter stats are computed per team (posteam), shifted by 1 game
  to prevent leakage, with a 6-game rolling window (min_periods=1).

  Train/test split: seasons <= 2022 for training, >= 2023 for testing.

Model
-----
  XGBRegressor (n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8).

Results
-------
  RMSE: 7.51 yards
  MAE:  5.91 yards
  Feature importance: yardline_100 (94.3%), punt_distance_roll6 (3.2%),
                      inside_twenty_rate_roll6 (2.5%)

Run order
---------
  1. python src/data/make_tables.py
  2. python -m src.models.punt_outcome
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "data" / "view"
MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "punt_outcome_xgb.json"
DATA_PATH = DATA_DIR / "fourth_down_with_features.csv"

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
FEATURE_COLS = ["yardline_100", "punt_distance_roll6", "inside_twenty_rate_roll6"]
TARGET = "opponent_start"

# League-average fallbacks for inference when rolling stats are unavailable
_LEAGUE_AVG = {
    "punt_distance_roll6": 45.9,
    "inside_twenty_rate_roll6": 0.44,
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_punt_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load fourth-down data, filter to punts, compute target and rolling features.

    Returns a DataFrame with FEATURE_COLS + [TARGET, "season"] ready for training.
    """
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Filter to punt attempts
    punts = df[df["punt_attempt"] == 1].copy()
    print(f"Punt attempts: {len(punts):,}")

    # Compute target: opponent starting field position
    punts[TARGET] = 100 - punts["yardline_100"] + punts["kick_distance"]
    punts.loc[punts["touchback"] == 1, TARGET] = 80

    # Remove invalid punts
    punts = punts[(punts["kick_distance"] > 0) & (punts[TARGET] <= 100)].copy()
    print(f"Valid punts: {len(punts):,}")

    # Compute per-team, per-game punt stats
    game_stats = punts.groupby(["posteam", "season", "week"]).agg(
        avg_punt_distance=("kick_distance", "mean"),
        inside_twenty_rate=(TARGET, lambda x: (x > 80).mean()),
    ).reset_index()

    game_stats = game_stats.sort_values(["posteam", "season", "week"]).copy()

    # Rolling 6-game averages (shifted by 1 to prevent leakage)
    game_stats["punt_distance_roll6"] = (
        game_stats.groupby("posteam")["avg_punt_distance"]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )
    game_stats["inside_twenty_rate_roll6"] = (
        game_stats.groupby("posteam")["inside_twenty_rate"]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )

    # Merge rolling features back onto each punt
    punts = punts.merge(
        game_stats[["posteam", "season", "week",
                     "punt_distance_roll6", "inside_twenty_rate_roll6"]],
        on=["posteam", "season", "week"],
        how="left",
    )

    # Drop rows without rolling stats (first game of a team's history)
    punts = punts.dropna(subset=["punt_distance_roll6", "inside_twenty_rate_roll6"]).copy()
    print(f"Punts with rolling features: {len(punts):,}")

    return punts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train the punt outcome XGBoost model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_punt_data().

    Returns
    -------
    (model, metrics_dict)
    """
    train_df = df[df["season"] <= 2022]
    test_df = df[df["season"] >= 2023]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

    print(f"\nTrain seasons: {sorted(train_df['season'].unique())}  — {len(X_train):,} punts")
    print(f"Test  seasons: {sorted(test_df['season'].unique())}  — {len(X_test):,} punts")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "train_n": len(X_train),
        "test_n": len(X_test),
    }

    print(f"\nRMSE: {rmse:.2f} yards")
    print(f"MAE:  {mae:.2f} yards")

    print("\nFeature importance:")
    for name, score in zip(FEATURE_COLS, model.feature_importances_):
        print(f"  {name}: {score:.3f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: xgb.XGBRegressor, path: Path = MODEL_PATH) -> None:
    """Save the trained model to JSON."""
    model.save_model(str(path))
    print(f"Model saved → {path}")


def load_model(path: Path = MODEL_PATH) -> xgb.XGBRegressor:
    """Load a previously saved model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            f"Run:  python -m src.models.punt_outcome"
        )
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


# ---------------------------------------------------------------------------
# Live inference
# ---------------------------------------------------------------------------

def predict_opponent_start(
    yardline_100: float,
    punt_distance_roll6: float | None = None,
    inside_twenty_rate_roll6: float | None = None,
    model_path: Path = MODEL_PATH,
) -> float:
    """
    Predict the opponent's starting field position after a punt.

    Parameters
    ----------
    yardline_100 : float
        Yards from opponent end zone where the punt occurs.
    punt_distance_roll6 : float, optional
        Team's rolling 6-game average punt distance. Defaults to league average.
    inside_twenty_rate_roll6 : float, optional
        Team's rolling 6-game inside-20 rate. Defaults to league average.

    Returns
    -------
    float
        Predicted opponent starting yardline_100.
    """
    model = load_model(model_path)

    if punt_distance_roll6 is None:
        punt_distance_roll6 = _LEAGUE_AVG["punt_distance_roll6"]
    if inside_twenty_rate_roll6 is None:
        inside_twenty_rate_roll6 = _LEAGUE_AVG["inside_twenty_rate_roll6"]

    X = pd.DataFrame([{
        "yardline_100": yardline_100,
        "punt_distance_roll6": punt_distance_roll6,
        "inside_twenty_rate_roll6": inside_twenty_rate_roll6,
    }])

    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Punt Outcome Model — Training")
    print("=" * 60)

    print("\n[1/3] Preparing data...")
    df = prepare_punt_data()

    print("\n[2/3] Training model...")
    model, _ = train(df)
    save_model(model)

    print("\n[3/3] Example predictions (from saved model):")
    examples = [
        dict(yardline_100=35, label="Own 35 (avg punter)"),
        dict(yardline_100=45, label="Own 45 (avg punter)"),
        dict(yardline_100=55, label="Midfield (avg punter)"),
        dict(yardline_100=70, label="Own 30 (avg punter)"),
        dict(yardline_100=45, punt_distance_roll6=50.0, inside_twenty_rate_roll6=0.60,
             label="Own 45 (elite punter)"),
        dict(yardline_100=45, punt_distance_roll6=40.0, inside_twenty_rate_roll6=0.25,
             label="Own 45 (weak punter)"),
    ]

    print(f"\n  {'Scenario':<35} {'Predicted opp start':>20}")
    print("  " + "-" * 57)
    for ex in examples:
        label = ex.pop("label")
        pred = predict_opponent_start(**ex)
        print(f"  {label:<35} {pred:>18.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
