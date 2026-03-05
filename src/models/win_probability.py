"""Win Probability Model
=======================
XGBoost + isotonic-calibration model that estimates the probability that
the *offensive team on the current play* wins the game, given a pre-play
game state.

Why XGBoost (not logistic regression):
  Win probability is a non-linear, high-interaction surface.  A 3-point lead
  with 60 seconds left carries a very different win probability than the same
  lead with 30 minutes left — and that interaction is compounded by timeouts,
  field position, down, and whether the game is in overtime.  XGBoost finds
  these interaction effects automatically through tree splits without any
  manual feature crossing.  Logistic regression would require O(n²) hand-
  crafted interaction terms just to approximate the same surface, and still
  cannot model the discontinuities introduced by rule changes (e.g. the 2023
  NFL playoff overtime rule where both teams are guaranteed one possession).

Why calibration is required:
  Raw XGBoost scores optimise log-loss but are not guaranteed to be
  well-calibrated probabilities.  An IsotonicRegression calibrator fits a
  monotone mapping from raw model output to empirical win rates, so that a
  predicted 70 % corresponds to an observed ~70 % win rate.  Calibration is
  critical for an expected-value calculation like:
      wp_go = p_conv * wp(success_state) + (1 - p_conv) * wp(failure_state)

Why OT plays are excluded from training:
  Overtime rules changed three times within the 2016-2024 window:
    • 2012 rule (pre-2023 playoffs): first score of any kind wins on
      the first possession; only sudden death from the second possession on.
    • 2023 playoff rule: both teams guaranteed one possession before
      sudden death.
    • 2025 regular season: both teams guaranteed one possession for ALL games.
  Because the mapping from game state to win probability differs across eras,
  including OT plays in training creates contradictory signal — the same
  state yields different outcomes depending on the year.  Excluding OT from
  training lets the model fit a clean regulation surface.  OT inference is
  then handled by OTStateTransformer, which re-encodes each OT state into a
  regulation-equivalent state the model already understands.
  (nflfastR uses the same design choice.)

Leakage prevention:
  Only pre-play state variables are used.  Features that are realised *during*
  or *after* the play (EPA, yards_gained, wpa, result of the current down) are
  never included.  A strict temporal split is used: models are evaluated on
  future seasons that were never seen during training or calibration.

New features vs. previous version:
  diff_time_ratio  = score_differential * exp(4 * elapsed_share)
      The primary WP signal used by nflfastR.  Score differential is
      exponentially amplified as the game progresses: a 3-point deficit in
      Q4 is far more damaging than in Q1.
  spread_time      = posteam_spread * exp(-4 * elapsed_share)
      The Vegas pre-game spread decays to zero as the game progresses,
      reflecting that the market's prior belief becomes irrelevant once
      most of the game has been played.
  half_seconds_remaining
      Seconds left in the current half (resets at half-time).  Captures
      end-of-half urgency that game_seconds_remaining misses (e.g. a team
      trailing by 7 with 30 seconds LEFT IN THE HALF plays very differently
      from 30 seconds left IN THE GAME).

Validation approach:
  Hold-out test seasons (default 2023-2024) are never seen during tree
  learning or calibration.  The most recent non-test seasons are reserved for
  early-stopping and isotonic calibration fitting.  Reported metrics:
    • Log-loss   (lower is better; logistic baseline ~0.65)
    • Brier score (lower is better; climatology baseline ~0.25)
    • Mean calibration error: average |predicted − actual win rate| per bin

Kickoff touchback yardlines (for state construction after FG / kickoff):
  2016-2023: 25-yard line (yardline_100 = 75 for receiving team)
  2024:      30-yard line (yardline_100 = 70) — dynamic kickoff rule
  2025+:     35-yard line (yardline_100 = 65)
  Use FourthDownDecisionEngine.kickoff_touchback_yardline(season) to look
  up the correct value for a given season.
  Punt touchbacks are always the 20-yard line (yardline_100 = 80).

Required columns for training:
  score_differential, quarter, seconds_remaining, yardline_100, down, ydstogo,
  offense_timeouts, defense_timeouts, is_overtime, overtime_possession_number,
  offense_team_won_game (target), season

Optional columns (default to 0 if absent):
  home           – 1 if possession team is the home team
  posteam_spread – point spread from possession team's view (neg = favored)
  guaranteed_possession – 1 if both teams are guaranteed a possession in OT
                          (2023+ playoffs, 2025+ regular season)

Usage
-----
  # ── Build & train ─────────────────────────────────────────────────────────
  from src.models.win_probability import WinProbabilityModel, build_training_data

  df   = build_training_data()          # reads data/raw/pbp_2016_2024.parquet
  wp   = WinProbabilityModel()
  wp.fit(df)
  wp.save()                             # → models/win_probability_model.pkl

  # ── Inference (regulation) ─────────────────────────────────────────────────
  wp = WinProbabilityModel.load("models/win_probability_model.pkl")

  state = dict(
      score_differential=0, quarter=4, seconds_remaining=120,
      yardline_100=35, down=4, ydstogo=2,
      offense_timeouts=2, defense_timeouts=1,
      is_overtime=0, overtime_possession_number=0,
  )
  wp.predict_proba(state)               # e.g. 0.52

  # ── Inference (overtime — auto-routed through OTStateTransformer) ──────────
  ot_state = dict(
      score_differential=0, quarter=5, seconds_remaining=420,
      yardline_100=75, down=1, ydstogo=10,
      offense_timeouts=2, defense_timeouts=2,
      is_overtime=1, overtime_possession_number=0,
      guaranteed_possession=0,          # regular season pre-2025
  )
  wp.predict_proba(ot_state)            # transformer maps to regulation space

  # ── 2025 season (guaranteed possession rule) ───────────────────────────────
  ot_state_2025 = dict(
      ...,
      is_overtime=1, overtime_possession_number=1,
      guaranteed_possession=1,          # 2025+ reg season or 2023+ playoffs
  )
  wp.predict_proba(ot_state_2025)

  # ── 4th-down decision engine ───────────────────────────────────────────────
  from src.models.win_probability import FourthDownDecisionEngine
  engine = FourthDownDecisionEngine(wp_model=wp)
  result = engine.recommend(
      current_state=state,
      p_conv=0.58,   success_state={...}, failure_state={...},
      p_fg=0.72,     fg_make_state={...}, fg_miss_state={...},
      punt_state={...},
  )
  print(result["decision"], result["wp_go"], result["wp_fg"], result["wp_punt"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models"
_DATA_DIR = _ROOT / "data" / "raw"

logger = logging.getLogger(__name__)

# ── Feature schema ─────────────────────────────────────────────────────────────

REQUIRED_STATE_KEYS: list[str] = [
    "score_differential",           # int | float  (offense minus defense)
    "quarter",                       # int  1–4 regular, 5 overtime
    "seconds_remaining",             # float  seconds left in game (0–3900)
    "yardline_100",                  # float  yards to opponent end zone (1–99)
    "down",                          # int  1–4
    "ydstogo",                       # float  yards to first down (1–99)
    "offense_timeouts",              # int  0–3
    "defense_timeouts",              # int  0–3
    "is_overtime",                   # int  0 or 1
    "overtime_possession_number",    # int  0=first poss, 1=second, 2+=sudden death
]

OPTIONAL_STATE_KEYS: list[str] = [
    # Team strength — default 0 (neutral)
    "offense_team_strength_rating",
    "defense_team_strength_rating",
    # Game context — default 0 if not supplied
    "home",                          # 1 if possession team is the home team
    "posteam_spread",                # Vegas spread from possession team's view
    #                                  (negative = favored, positive = underdog)
    "guaranteed_possession",         # 1 if both OT teams get a possession before
    #                                  sudden death (2023+ playoffs, 2025+ reg)
]

# Raw state features passed directly to the model
_BASE_FEATURES: list[str] = [
    "score_differential",
    "quarter",
    "seconds_remaining",
    "yardline_100",
    "down",
    "ydstogo",
    "offense_timeouts",
    "defense_timeouts",
    "is_overtime",
    "overtime_possession_number",
]

# Derived non-linear and interaction features (see _engineer_features)
_ENGINEERED_FEATURES: list[str] = [
    # Non-linear time effects
    "seconds_remaining_sqrt",
    "seconds_remaining_log1p",
    # Half-time context (resets at halftime — captures end-of-half urgency)
    "half_seconds_remaining",
    # Game progress fraction (0 = kickoff, 1 = final whistle)
    "elapsed_share",
    # Primary nflfastR-style WP signal: score amplified by elapsed time
    "diff_time_ratio",
    # Vegas spread decaying to 0 as game progresses
    "spread_time",
    # Classic score × time interaction
    "score_x_time",
    "urgency",
    "clock_leverage",
    # Score state
    "abs_score_diff",
    "score_differential_sq",
    # Timeout advantage
    "timeout_diff",
    "total_timeouts",
    # Down & distance
    "ydstogo_log1p",
    "short_yardage",
    # Field position zones
    "fg_range",
    "red_zone",
    "scoring_position",
    # Overtime phase flags
    "ot_first_poss",
    "ot_second_poss",
    "ot_sudden_death",
    "ot_must_score",
    "ot_leading_first_poss",
]

_ALL_FEATURES: list[str] = _BASE_FEATURES + _ENGINEERED_FEATURES


# ── Data helpers ───────────────────────────────────────────────────────────────

def _compute_ot_possession_number(df: pd.DataFrame) -> pd.DataFrame:
    """Assign overtime possession number (0-indexed) to every OT play.

    Uses dense ranking of drive numbers within OT (``qtr >= 5``) per game:
      0 → first team's possession
      1 → second team's possession
      2 → third+ (sudden death, clipped at 2)

    Regulation plays are left at 0 (the column defaults to 0 for non-OT rows
    and is masked by ``is_overtime`` in downstream feature engineering).
    """
    df = df.copy()
    df["overtime_possession_number"] = 0

    ot_mask = df["is_overtime"] == 1
    if ot_mask.sum() == 0:
        return df

    ot_idx = df.index[ot_mask]
    ot_sub = df.loc[ot_idx, ["game_id", "drive"]].copy()

    ot_sub["ot_drive_rank"] = (
        ot_sub.groupby("game_id")["drive"]
        .transform(lambda s: s.rank(method="dense") - 1)
        .fillna(0)
        .clip(0, 2)
        .astype(int)
    )
    df.loc[ot_idx, "overtime_possession_number"] = ot_sub["ot_drive_rank"]
    return df


def build_training_data(
    parquet_path: Path | str | None = None,
    min_season: int = 2016,
    max_season: int = 2024,
) -> pd.DataFrame:
    """Load and prepare a training DataFrame from nflfastR play-by-play data.

    Reads the raw parquet file, filters to scrimmage plays with complete state
    information, derives ``offense_team_won_game``, and renames columns to the
    model's canonical schema.

    Returns ALL plays (including OT).  The ``fit()`` method filters OT out of
    the training and calibration splits; OT plays are retained for evaluation
    in the test set so we can measure OT accuracy.

    Parameters
    ----------
    parquet_path:
        Path to the nflfastR ``*.parquet`` file.  Defaults to
        ``data/raw/pbp_2016_2024.parquet`` relative to the project root.
    min_season, max_season:
        Inclusive season range to include in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per scrimmage play.  Contains all ``REQUIRED_STATE_KEYS``,
        available ``OPTIONAL_STATE_KEYS``, ``offense_team_won_game``, and
        ``season`` / ``game_id`` for temporal splitting.
    """
    if parquet_path is None:
        parquet_path = _DATA_DIR / "pbp_2016_2024.parquet"
    parquet_path = Path(parquet_path)

    logger.info("Loading PBP data from %s", parquet_path)

    load_cols = [
        "game_id", "posteam", "defteam", "home_team", "away_team",
        "season", "week", "qtr", "game_seconds_remaining",
        "score_differential",
        "down", "ydstogo", "yardline_100",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "drive", "result",
    ]

    # Optional enrichment columns — load if present, silently skip if absent
    optional_cols = ["spread_line", "home_opening_kickoff"]

    df_full = pd.read_parquet(parquet_path, columns=None)
    available = set(df_full.columns)
    extra = [c for c in optional_cols if c in available]
    cols_to_load = load_cols + extra
    df = df_full[cols_to_load].copy()
    del df_full

    # ── Filter: keep only plays with a full pre-play state ────────────────────
    required_notna = [
        "down", "ydstogo", "yardline_100",
        "game_seconds_remaining", "score_differential",
        "result", "posteam",
    ]
    df = df.dropna(subset=required_notna)
    df = df[df["season"].between(min_season, max_season)].copy()
    df = df[df["down"].between(1, 4)].copy()

    # ── Target variable ───────────────────────────────────────────────────────
    # ``result`` = home_final_score − away_final_score (from nflfastR).
    df["offense_team_won_game"] = (
        ((df["posteam"] == df["home_team"]) & (df["result"] > 0)) |
        ((df["posteam"] == df["away_team"]) & (df["result"] < 0))
    ).astype(int)

    # ── Rename / clip to canonical schema ────────────────────────────────────
    df["quarter"] = df["qtr"].clip(1, 5).astype(int)
    df["seconds_remaining"] = df["game_seconds_remaining"].clip(lower=0).astype(float)
    df["yardline_100"] = df["yardline_100"].clip(1, 99).astype(float)
    df["ydstogo"] = df["ydstogo"].clip(1, 99).astype(float)
    df["down"] = df["down"].astype(int)
    df["offense_timeouts"] = df["posteam_timeouts_remaining"].fillna(3).clip(0, 3).astype(int)
    df["defense_timeouts"] = df["defteam_timeouts_remaining"].fillna(3).clip(0, 3).astype(int)
    df["score_differential"] = df["score_differential"].astype(float)
    df["is_overtime"] = (df["quarter"] >= 5).astype(int)

    # ── Optional: home indicator and Vegas spread ─────────────────────────────
    df["home"] = (df["posteam"] == df["home_team"]).astype(float)

    if "spread_line" in df.columns:
        # spread_line in nflfastR is from the HOME team's perspective
        # (negative = home team favoured).  Flip sign for away possession.
        df["posteam_spread"] = np.where(
            df["posteam"] == df["home_team"],
            df["spread_line"].fillna(0.0),
            -df["spread_line"].fillna(0.0),
        )
    else:
        df["posteam_spread"] = 0.0

    # guaranteed_possession flag — not available in raw data, default 0
    # (Callers set this at inference time for 2023+ playoffs / 2025+ reg)
    df["guaranteed_possession"] = 0.0

    # Strength ratings — default neutral
    df["offense_team_strength_rating"] = 0.0
    df["defense_team_strength_rating"] = 0.0

    # ── Overtime possession number ────────────────────────────────────────────
    df = _compute_ot_possession_number(df)

    logger.info(
        "Training data ready: %d plays | %d seasons | avg WP target = %.3f",
        len(df), df["season"].nunique(), df["offense_team_won_game"].mean(),
    )
    return df


# ── Model class ───────────────────────────────────────────────────────────────

class WinProbabilityModel:
    """XGBoost win probability model with post-hoc probability calibration.

    The model predicts P(offensive team wins the game | current game state).
    It is trained on regulation plays only (OT excluded from training to avoid
    contradictory signal across rule eras).  Overtime inference is handled
    by the OTStateTransformer, which maps each OT state to a regulation-
    equivalent state before calling predict_proba().

    Parameters
    ----------
    n_estimators:
        Maximum number of trees.  Early stopping halts before this ceiling.
    max_depth:
        Maximum tree depth.  6 captures rich interactions without overfitting.
    learning_rate:
        Shrinkage factor per tree.  0.02 with 2000 estimators and early
        stopping gives a better-regularised fit than 0.05/600.
    subsample, colsample_bytree:
        Row- and column-sampling fractions for stochastic gradient boosting.
    min_child_weight:
        Minimum sum of instance weights in a leaf.  10 provides regularisation
        without being too restrictive (was 20, relaxed slightly to let the
        model capture real sub-population patterns).
    gamma:
        Minimum loss reduction for a further split.
    reg_alpha, reg_lambda:
        L1 / L2 regularisation on leaf weights.
    calibration_method:
        ``"isotonic"`` (non-parametric, preferred for large data) or
        ``"sigmoid"`` (Platt scaling).
    test_seasons:
        Seasons held out for final evaluation.  Never used in training or
        calibration.  Default: [2023, 2024].
    random_state:
        Seed for reproducibility.
    season_decay:
        Exponential decay factor applied to sample weights by season age.
        0.80 weights one season back at 0.80×, two back at 0.64×, etc.
    n_calib_seasons:
        Number of most-recent non-test seasons used for calibration and
        early stopping.  3 seasons gives the isotonic calibrator more diverse
        data and reduces overfitting to a single season's distribution.
    """

    def __init__(
        self,
        n_estimators: int = 2000,
        max_depth: int = 6,
        learning_rate: float = 0.02,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 10,
        gamma: float = 1.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        calibration_method: str = "isotonic",
        test_seasons: list[int] | None = None,
        random_state: int = 42,
        season_decay: float = 0.80,
        n_calib_seasons: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.calibration_method = calibration_method
        self.test_seasons: list[int] = test_seasons if test_seasons is not None else [2023, 2024]
        self.random_state = random_state
        self.season_decay = season_decay
        self.n_calib_seasons = n_calib_seasons

        # Internal state — populated by fit()
        self._base_model: XGBClassifier | None = None
        self._calibrator: IsotonicRegression | LogisticRegression | None = None
        self._feature_cols: list[str] = []
        self._is_fitted: bool = False
        self._ot_transformer: OTStateTransformer = OTStateTransformer()

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive non-linear and interaction features from raw state columns.

        All transforms are deterministic and applied identically at train time
        and inference time, preventing any train-serve skew.

        Key derived features
        --------------------
        diff_time_ratio:
            score_differential * exp(4 * elapsed_share).  The primary WP
            signal from nflfastR.  Exponentially amplifies the score as the
            game progresses: a 7-point lead means much more with 2 minutes
            left than with 30 minutes left.

        spread_time:
            posteam_spread * exp(-4 * elapsed_share).  The Vegas pre-game
            spread decays from its full value at kickoff toward 0 at game
            end, reflecting that the prior becomes irrelevant as evidence
            accumulates.

        half_seconds_remaining:
            Seconds left in the current half (resets at halftime).  Captures
            end-of-half urgency (two-minute drill, prevent defence) that the
            full-game clock misses.

        elapsed_share:
            (3600 - seconds_remaining) / 3600, clipped to [0, 1].  The
            fraction of regulation time that has been played.
        """
        out = df.copy()
        sec = out["seconds_remaining"].astype(float)
        sd = out["score_differential"].astype(float)
        ot = out["is_overtime"].astype(int)
        pn = out["overtime_possession_number"].astype(int)

        # Game progress
        elapsed = ((3600.0 - sec) / 3600.0).clip(0.0, 1.0)
        out["elapsed_share"] = elapsed

        # Half-time context: time remaining in the current half
        # First half: game_seconds_remaining runs from 3600 → 1800
        #   → half_seconds = seconds_remaining - 1800
        # Second half: game_seconds_remaining runs from 1800 → 0
        #   → half_seconds = seconds_remaining
        out["half_seconds_remaining"] = np.where(sec > 1800.0, sec - 1800.0, sec)

        # Primary nflfastR-style WP signal
        out["diff_time_ratio"] = sd * np.exp(4.0 * elapsed)

        # Vegas spread decay
        spread = out.get("posteam_spread", pd.Series(0.0, index=out.index))
        if "posteam_spread" in out.columns:
            spread = out["posteam_spread"].fillna(0.0)
        out["spread_time"] = spread * np.exp(-4.0 * elapsed)

        # Non-linear time
        out["seconds_remaining_sqrt"] = np.sqrt(sec)
        out["seconds_remaining_log1p"] = np.log1p(sec)

        # Classic score × time interaction
        out["score_x_time"] = sd * sec / 3600.0
        out["urgency"] = np.abs(sd) / (sec + 1.0)
        out["clock_leverage"] = 1.0 / (np.abs(sd) + 1.0) / (sec + 60.0)

        # Score state
        out["abs_score_diff"] = np.abs(sd)
        out["score_differential_sq"] = sd ** 2

        # Timeouts
        out["timeout_diff"] = out["offense_timeouts"] - out["defense_timeouts"]
        out["total_timeouts"] = out["offense_timeouts"] + out["defense_timeouts"]

        # Down & distance
        out["ydstogo_log1p"] = np.log1p(out["ydstogo"])
        out["short_yardage"] = (out["ydstogo"] <= 2).astype(int)

        # Field position
        out["fg_range"] = (out["yardline_100"] <= 35).astype(int)
        out["red_zone"] = (out["yardline_100"] <= 20).astype(int)
        out["scoring_position"] = (out["yardline_100"] <= 10).astype(int)

        # Overtime phase dummies
        out["ot_first_poss"] = (ot & (pn == 0)).astype(int)
        out["ot_second_poss"] = (ot & (pn == 1)).astype(int)
        out["ot_sudden_death"] = (ot & (pn >= 2)).astype(int)
        out["ot_must_score"] = ((ot == 1) & (pn >= 1) & (sd < 0)).astype(int)
        out["ot_leading_first_poss"] = ((ot == 1) & (pn == 0) & (sd > 0)).astype(int)

        return out

    # ── State dict helpers ────────────────────────────────────────────────────

    def _validate_state(self, state_dict: dict[str, Any]) -> None:
        """Raise ValueError if any required key is absent from *state_dict*."""
        missing = [k for k in REQUIRED_STATE_KEYS if k not in state_dict]
        if missing:
            raise ValueError(
                f"State dict is missing required keys: {missing}\n"
                f"Required: {REQUIRED_STATE_KEYS}"
            )

    def _state_to_features(self, state_dict: dict[str, Any]) -> np.ndarray:
        """Convert a state dictionary to a (1, n_features) numpy array."""
        self._validate_state(state_dict)
        row = {k: state_dict.get(k, 0) for k in REQUIRED_STATE_KEYS}
        for k in OPTIONAL_STATE_KEYS:
            row[k] = float(state_dict.get(k, 0.0))

        df_row = pd.DataFrame([row])
        df_row = self._engineer_features(df_row)
        return df_row[self._feature_cols].fillna(0.0).values

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "WinProbabilityModel":
        """Train the Win Probability model on regulation plays only.

        OT plays are excluded from training and calibration because OT rules
        changed across the 2016-2024 data window, creating contradictory
        signal.  OT plays in the test seasons ARE evaluated so we can measure
        OT accuracy separately.

        Temporal split strategy
        -----------------------
        1. ``test_seasons``         – held out entirely; reported at the end.
        2. ``calib_seasons``        – most recent n_calib_seasons non-test
                                      seasons; used for XGBoost early-stopping
                                      and isotonic calibration fitting.
        3. remaining fit seasons    – regulation plays only for tree learning.
        """
        if "offense_team_won_game" not in df.columns:
            raise ValueError("df must contain 'offense_team_won_game' (binary target).")
        if "season" not in df.columns:
            raise ValueError("df must contain 'season' for temporal train/test split.")

        logger.info("Engineering features for %d plays...", len(df))
        df_feat = self._engineer_features(df)

        # Determine which features to use
        self._feature_cols = list(_ALL_FEATURES)
        for opt in OPTIONAL_STATE_KEYS:
            if opt in df_feat.columns and df_feat[opt].notna().any():
                self._feature_cols.append(opt)

        # ── Temporal split ────────────────────────────────────────────────────
        all_seasons = sorted(df_feat["season"].unique())
        train_seasons = [s for s in all_seasons if s not in self.test_seasons]
        test_seasons_present = [s for s in self.test_seasons if s in all_seasons]

        if not train_seasons:
            raise ValueError(
                f"No training seasons remain after reserving test seasons "
                f"{self.test_seasons}.  Available: {all_seasons}"
            )

        n_calib = min(self.n_calib_seasons, len(train_seasons) - 1)
        if n_calib < 1:
            logger.warning(
                "Only one non-test season available. "
                "Using it for both fitting and calibration."
            )
            n_calib = 1
        calib_seasons_used = train_seasons[-n_calib:]
        fit_seasons = [s for s in train_seasons if s not in calib_seasons_used]
        if not fit_seasons:
            fit_seasons = list(calib_seasons_used)

        logger.info(
            "Fit seasons: %s | Calibration seasons: %s | Test seasons: %s",
            fit_seasons, calib_seasons_used, test_seasons_present,
        )

        # ── OT exclusion for training and calibration ─────────────────────────
        # OT plays are excluded here (not in build_training_data) so that the
        # full dataset (including OT) is still available for test evaluation.
        reg_mask = df_feat["is_overtime"] == 0
        logger.info(
            "Regulation plays for training/calibration: %d / %d total "
            "(%.1f%% OT excluded)",
            reg_mask.sum(), len(df_feat),
            100.0 * (1 - reg_mask.mean()),
        )

        fit_mask = df_feat["season"].isin(fit_seasons) & reg_mask
        X_fit = df_feat.loc[fit_mask, self._feature_cols].fillna(0.0).values
        y_fit = df_feat.loc[fit_mask, "offense_team_won_game"].values

        # Recency weights: exponential decay by season age
        max_fit_season = int(max(fit_seasons))
        season_ages = (max_fit_season - df_feat.loc[fit_mask, "season"].values).astype(float)
        sample_weights = self.season_decay ** season_ages

        calib_mask = df_feat["season"].isin(calib_seasons_used) & reg_mask
        X_calib = df_feat.loc[calib_mask, self._feature_cols].fillna(0.0).values
        y_calib = df_feat.loc[calib_mask, "offense_team_won_game"].values

        # ── Base XGBoost ──────────────────────────────────────────────────────
        base = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=self.random_state,
        )

        base.fit(
            X_fit, y_fit,
            sample_weight=sample_weights,
            eval_set=[(X_calib, y_calib)],
            verbose=False,
        )
        logger.info("XGBoost best iteration: %d", base.best_iteration)
        self._base_model = base

        # ── Calibration ───────────────────────────────────────────────────────
        raw_calib = base.predict_proba(X_calib)[:, 1]

        if self.calibration_method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_calib, y_calib)
        else:
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs")
            self._calibrator.fit(raw_calib.reshape(-1, 1), y_calib)

        self._is_fitted = True

        # ── Metrics ───────────────────────────────────────────────────────────
        calib_probs = self._apply_calibrator(raw_calib)
        logger.info("-- Calibration seasons %s metrics --", calib_seasons_used)
        self.calib_metrics_ = self._report_metrics(
            y_calib, calib_probs, split=f"calib({calib_seasons_used})"
        )
        self.calib_seasons_used_ = calib_seasons_used

        self.test_metrics_: dict[str, float] | None = None
        self.test_seasons_present_: list[int] = test_seasons_present
        if test_seasons_present:
            # Test evaluation uses ALL plays (including OT) so we can measure
            # both regulation and OT accuracy in the benchmark script.
            test_mask = df_feat["season"].isin(test_seasons_present)
            X_test = df_feat.loc[test_mask, self._feature_cols].fillna(0.0).values
            y_test = df_feat.loc[test_mask, "offense_team_won_game"].values
            raw_test = self._base_model.predict_proba(X_test)[:, 1]
            test_probs = self._apply_calibrator(raw_test)
            logger.info("-- Test seasons %s metrics --", test_seasons_present)
            self.test_metrics_ = self._report_metrics(
                y_test, test_probs, split=f"test({test_seasons_present})"
            )

        return self

    # ── Calibration helper ────────────────────────────────────────────────────

    def _apply_calibrator(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw XGBoost probabilities through the fitted calibrator."""
        if isinstance(self._calibrator, IsotonicRegression):
            return self._calibrator.predict(raw_probs)
        else:
            return self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, state_dict: dict[str, Any]) -> float:
        """Return P(offensive team wins | current game state).

        For overtime states (``is_overtime=1``), the state is first passed
        through OTStateTransformer, which maps it to a regulation-equivalent
        state that the regulation-trained model can interpret.

        Parameters
        ----------
        state_dict:
            Dict with all keys in ``REQUIRED_STATE_KEYS``.
            Optional keys from ``OPTIONAL_STATE_KEYS`` default to 0.
            For OT inference, include ``guaranteed_possession`` (0 or 1)
            to specify which rule set applies.

        Returns
        -------
        float in [0, 1]
            Calibrated win probability for the offensive team.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # Route OT states through the transformer
        if state_dict.get("is_overtime", 0):
            state_dict = self._ot_transformer.transform(state_dict)

        X = self._state_to_features(state_dict)
        raw = self._base_model.predict_proba(X)[0, 1]
        return float(self._apply_calibrator(np.array([raw]))[0])

    def simulate_state(self, state_dict: dict[str, Any]) -> float:
        """Return WP for a hypothetical (counterfactual) game state.

        Semantically identical to ``predict_proba`` but signals to the reader
        that the state was *constructed* to represent a post-play outcome
        (e.g. after a successful conversion, made FG, or punt).

        Used inside the 4th-down decision engine:

            wp_go = p_conv * wp.simulate_state(success_state)
                  + (1 - p_conv) * wp.simulate_state(failure_state)
        """
        return self.predict_proba(state_dict)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Serialise the fitted model to disk with joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.  Call fit() first.")

        if path is None:
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = _MODEL_DIR / "win_probability_model.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "base_model": self._base_model,
            "calibrator": self._calibrator,
            "feature_cols": self._feature_cols,
            "hyperparams": {
                k: getattr(self, k)
                for k in (
                    "n_estimators", "max_depth", "learning_rate",
                    "subsample", "colsample_bytree", "min_child_weight",
                    "gamma", "reg_alpha", "reg_lambda",
                    "calibration_method", "test_seasons", "random_state",
                    "season_decay", "n_calib_seasons",
                )
            },
        }
        joblib.dump(payload, path)
        logger.info("Model saved -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "WinProbabilityModel":
        """Load a previously saved model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        instance = cls(**payload.get("hyperparams", {}))
        instance._base_model = payload["base_model"]
        instance._calibrator = payload["calibrator"]
        instance._feature_cols = payload["feature_cols"]
        instance._is_fitted = True
        logger.info("Model loaded <- %s", path)
        return instance

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _report_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        split: str = "eval",
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Compute and log log-loss, Brier score, mean calibration error, and accuracy."""
        ll = log_loss(y_true, y_prob)
        bs = brier_score_loss(y_true, y_prob)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        mce = float(np.mean(np.abs(frac_pos - mean_pred)))
        accuracy = float(((y_prob >= 0.5).astype(int) == y_true).mean())

        logger.info(
            "[%s]  n=%d  log_loss=%.4f  brier=%.4f  mean_calib_err=%.4f  accuracy=%.4f",
            split, len(y_true), ll, bs, mce, accuracy,
        )
        return {"log_loss": ll, "brier_score": bs, "mean_calib_error": mce, "accuracy": accuracy}

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of XGBoost gain-based feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        scores = self._base_model.get_booster().get_score(importance_type="gain")
        rows = [
            {"feature": self._feature_cols[int(k.replace("f", ""))], "importance": v}
            for k, v in scores.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_wp_model(path: Path | str | None = None) -> WinProbabilityModel:
    """Load the saved Win Probability model from disk."""
    if path is None:
        path = _MODEL_DIR / "win_probability_model.pkl"
    return WinProbabilityModel.load(path)


# ── OT State Transformer ───────────────────────────────────────────────────────

class OTStateTransformer:
    """Maps overtime game states to regulation-equivalent states.

    The win probability model is trained on regulation plays only.  To get a
    meaningful probability for overtime situations, we re-encode the OT context
    into the signals the model already understands: score differential, time
    pressure, field position, and urgency.

    Transformation logic
    --------------------
    1.  OT clock → compressed regulation clock.
        A full overtime period (600 s since 2017) is mapped to 300 s of
        regulation-equivalent time (the final 5 minutes of Q4).  This
        preserves the relative urgency within OT while placing it in the
        range where the regulation model has good density.

    2.  Quarter → 4 (late regulation, maximum leverage zone).

    3.  Score differential: kept as-is.  In pre-play overtime states the
        score is almost always tied (0), which is correct — both teams
        entered OT equal.  On second/sudden-death possessions the score
        may reflect what the first team scored; that differential is
        preserved so the model captures the must-score urgency.

    4.  ``is_overtime`` → 0  (so the regulation model's OT phase dummies
        are all zero and it uses the time/score signal instead).

    NFL rule timeline (affects ``guaranteed_possession`` flag)
    ----------------------------------------------------------
    2016–2022 regular season:
        First score of any type wins immediately (pure sudden death from
        the start for FGs; TD on first possession wins outright).
        guaranteed_possession = 0

    2023+ playoffs / 2022 season playoffs:
        Both teams guaranteed one possession.  After both teams possess,
        sudden death.
        guaranteed_possession = 1

    2025+ regular season:
        Both teams guaranteed one possession in ALL games.
        guaranteed_possession = 1

    Usage
    -----
    Callers should set ``guaranteed_possession`` in the state dict:

        # Regular season 2016-2024
        state["guaranteed_possession"] = 0

        # 2025 regular season or 2023+ playoffs
        state["guaranteed_possession"] = 1

    The transformer uses ``guaranteed_possession`` only for documentation
    and potential future adjustments to the score differential mapping.
    The core time compression applies regardless.
    """

    # OT period length in seconds (10 min since 2017 playoffs / 2018 regular season;
    # was 15 min = 900 s in earlier eras, but our data starts at 2016 so edge case)
    _OT_DURATION: float = 600.0

    # Regulation seconds that a full OT period maps to (final 5 min of Q4)
    _REG_EQUIV_SECONDS: float = 300.0

    # Minimum compressed clock (avoids exact-zero edge case)
    _MIN_SECONDS: float = 10.0

    def transform(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a regulation-equivalent state dict for model inference.

        Parameters
        ----------
        state:
            A game state dict with ``is_overtime=1``.  Must include all
            ``REQUIRED_STATE_KEYS``.  ``guaranteed_possession`` (optional)
            should be 1 for 2023+ playoffs or 2025+ regular season.

        Returns
        -------
        dict
            A new state dict with ``is_overtime=0``, ``quarter=4``, and a
            compressed ``seconds_remaining``.  All other keys are preserved.
        """
        ot_seconds = float(state.get("seconds_remaining", self._OT_DURATION))

        # Compress the OT clock into the regulation model's Q4 range.
        # No artificial cap on sudden-death time — if there are 8 minutes
        # left and both teams are in sudden death, that urgency is real and
        # should be reflected via the actual clock.
        scale = self._REG_EQUIV_SECONDS / self._OT_DURATION
        reg_seconds = max(self._MIN_SECONDS, ot_seconds * scale)

        new_state = dict(state)
        new_state["is_overtime"] = 0
        new_state["overtime_possession_number"] = 0
        new_state["quarter"] = 4
        new_state["seconds_remaining"] = reg_seconds

        return new_state


# ── 4th-down decision engine ──────────────────────────────────────────────────

class FourthDownDecisionEngine:
    """Integrates the Win Probability model with sub-model probabilities to
    recommend the optimal 4th-down decision via expected-value maximisation.

    The three decisions and their expected WP:

        wp_go   = p_conv * wp(success_state) + (1-p_conv) * [1 - wp(failure_state)]
        wp_fg   = p_make * [1 - wp(make_state)] + (1-p_make) * [1 - wp(miss_state)]
        wp_punt = 1 - wp(punt_state)

        decision = argmax(wp_go, wp_fg, wp_punt)

    All post-play state dicts are from the NEW POSSESSION TEAM's perspective.
    ``success_state`` keeps the original team (no flip); all other states
    represent the opponent taking over, so the engine applies ``1 - wp(...)``
    to convert from the opponent's WP back to the original team's WP.

    Sub-model interface
    -------------------
    * Punt outcome model (models/punt_outcome_xgb.json):
        Outputs ``opponent_start`` = new possession team's ``yardline_100``.
        Pass directly as ``punt_state["yardline_100"]``.

    * FG probability model (src/models/fg_probability.py):
        Call: ``predict_fg_prob(kick_distance=FourthDownDecisionEngine.fg_kick_distance(yl), ...)``
        where ``yl`` = current ``yardline_100``.

    * Go-conversion model (pending — placeholder):
        The ``p_conv`` parameter is a caller-supplied float from whatever
        conversion model is available.  Pass the model output directly.
        (Interface is already compatible; no changes needed when the model
        is ready.)

    Parameters
    ----------
    wp_model:
        A fitted ``WinProbabilityModel`` instance.
    """

    # Kickoff touchback yardline by season (receiving team's yardline_100).
    # Key = first season the rule applies; dict is traversed newest-first.
    _KICKOFF_TOUCHBACK: dict[int, float] = {
        2025: 65.0,   # 35-yard line (dynamic kickoff 2025 adjustment)
        2024: 70.0,   # 30-yard line (dynamic kickoff introduced 2024)
        2016: 75.0,   # 25-yard line (2016-2023)
    }

    # Punt touchback yardline — always the 20-yard line, never changed.
    PUNT_TOUCHBACK_YARDLINE_100: float = 80.0  # 80 yards from opp end zone

    def __init__(self, wp_model: WinProbabilityModel) -> None:
        if not wp_model._is_fitted:
            raise ValueError("wp_model must be fitted before use.")
        self.wp = wp_model

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def kickoff_touchback_yardline(season: int) -> float:
        """Return the touchback start yardline_100 for kickoffs in a given season.

        Parameters
        ----------
        season:
            The NFL season year (e.g. 2024).

        Returns
        -------
        float
            The receiving team's ``yardline_100`` after a touchback.
            (e.g. 70.0 for the 2024 rule = starts at own 30-yard line)
        """
        table = FourthDownDecisionEngine._KICKOFF_TOUCHBACK
        for cutoff in sorted(table.keys(), reverse=True):
            if season >= cutoff:
                return table[cutoff]
        return 75.0  # fallback: pre-2024 rule

    @staticmethod
    def fg_kick_distance(yardline_100: float) -> float:
        """Return the field goal kick distance for a given line of scrimmage.

        Standard NFL formula: kick spot is 7 yards behind the line of
        scrimmage (center snap) and the goal posts are at the back of the
        end zone (10 yards deep).

            kick_distance = yardline_100 + 17

        Parameters
        ----------
        yardline_100:
            Yards from the line of scrimmage to the opponent's end zone.

        Returns
        -------
        float
            Kick distance in yards (what the FG probability model expects).
        """
        return float(yardline_100) + 17.0

    @staticmethod
    def _flip_possession(state: dict[str, Any]) -> dict[str, Any]:
        """Return a new state with possession flipped to the other team.

        Used when the ball changes hands (failed conversion, punt, missed FG).
        The new state represents the *opponent's* next possession:
          - score_differential negated
          - timeouts swapped
          - home indicator flipped
          - posteam_spread negated
          - overtime_possession_number incremented (if in OT)

        The caller is still responsible for setting the correct
        ``yardline_100``, ``down``, ``ydstogo``, and ``seconds_remaining``.
        """
        new = dict(state)
        new["score_differential"] = -state.get("score_differential", 0.0)
        new["offense_timeouts"] = state.get("defense_timeouts", 3)
        new["defense_timeouts"] = state.get("offense_timeouts", 3)
        new["home"] = 1.0 - float(state.get("home", 0.0))
        new["posteam_spread"] = -float(state.get("posteam_spread", 0.0))
        if state.get("is_overtime", 0):
            new["overtime_possession_number"] = (
                int(state.get("overtime_possession_number", 0)) + 1
            )
        return new

    @staticmethod
    def _same_possession(state: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of *state* (no possession flip).

        Used when the same team retains possession after a play (successful
        conversion).  The caller updates yardline_100, down, ydstogo, and
        seconds_remaining as needed.
        """
        return dict(state)

    # ── Decision engine ───────────────────────────────────────────────────────

    def recommend(
        self,
        *,
        current_state: dict[str, Any],
        # GO
        p_conv: float,
        success_state: dict[str, Any],
        failure_state: dict[str, Any],
        # FG
        p_fg: float,
        fg_make_state: dict[str, Any],
        fg_miss_state: dict[str, Any],
        # PUNT
        punt_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute expected WP for each option and return the recommendation.

        State convention
        ----------------
        ALL state dicts are from the **new possession team's perspective**
        (the team that will hold the ball after the play):

        * ``success_state``: original team keeps the ball (same team).
          ``wp(success_state)`` = P(original team wins).  No flip needed.

        * ``failure_state``, ``fg_make_state``, ``fg_miss_state``,
          ``punt_state``: opponent receives the ball.
          ``wp(opponent_state)`` = P(opponent wins), so the engine
          returns ``1 - wp(opponent_state)`` = P(original team wins).

        Build states with ``_flip_possession(current_state)`` for any play
        where the ball changes hands, then update field position, down, and
        clock.  Use ``_same_possession(current_state)`` for the success case.

        Parameters
        ----------
        current_state:
            The current pre-play game state (used for context / logging only).
        p_conv:
            P(4th-down conversion succeeds).  Supply from the go-conversion
            sub-model.  Interface is ready — plug in model output directly
            when the model is available.
        success_state:
            State after a successful conversion (same team keeps ball).
            Use ``_same_possession(current_state)`` as a base, then update
            ``yardline_100``, ``down=1``, ``ydstogo=10``, ``seconds_remaining``.
        failure_state:
            State after a failed conversion (opponent takes over).
            Use ``_flip_possession(current_state)`` as a base, then set
            ``yardline_100 = 100 - current_yardline_100``,
            ``down=1``, ``ydstogo=10``.
        p_fg:
            P(field goal made).  From the FG probability sub-model:
            ``predict_fg_prob(kick_distance=fg_kick_distance(yardline_100), ...)``
        fg_make_state:
            State after a made FG (opponent receives kickoff).
            Use ``_flip_possession(current_state)`` as a base, then add 3
            to the negated score_differential and set
            ``yardline_100=kickoff_touchback_yardline(season)``.
        fg_miss_state:
            State after a missed FG (opponent takes over at line of scrimmage).
            Use ``_flip_possession(current_state)`` as a base, then set
            ``yardline_100 = max(80, 100 - current_yardline_100)``.
        punt_state:
            State after the punt (opponent receives).
            Use ``_flip_possession(current_state)`` as a base, then set
            ``yardline_100`` to the punt sub-model's ``opponent_start``.

        Returns
        -------
        dict with keys:
            decision           -- "GO" | "FG" | "PUNT"
            wp_go              -- expected WP for going for it
            wp_fg              -- expected WP for attempting FG
            wp_punt            -- expected WP for punting
            margin_go_vs_fg    -- wp_go - wp_fg
            margin_go_vs_punt  -- wp_go - wp_punt
        """
        # success: original team keeps ball → wp is from their perspective
        # failure/fg/punt: opponent has ball → 1 - wp(opponent) = original team's WP
        wp_go = (
            p_conv * self.wp.simulate_state(success_state)
            + (1.0 - p_conv) * (1.0 - self.wp.simulate_state(failure_state))
        )
        wp_fg = (
            p_fg * (1.0 - self.wp.simulate_state(fg_make_state))
            + (1.0 - p_fg) * (1.0 - self.wp.simulate_state(fg_miss_state))
        )
        wp_punt = 1.0 - self.wp.simulate_state(punt_state)

        options = {"GO": wp_go, "FG": wp_fg, "PUNT": wp_punt}
        decision = max(options, key=options.__getitem__)

        return {
            "decision": decision,
            "wp_go": round(wp_go, 4),
            "wp_fg": round(wp_fg, 4),
            "wp_punt": round(wp_punt, 4),
            "margin_go_vs_fg": round(wp_go - wp_fg, 4),
            "margin_go_vs_punt": round(wp_go - wp_punt, 4),
        }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Train and save the Win Probability model.

    Run from the project root::

        python -m src.models.win_probability
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("  NFL Win Probability Model -- Training Pipeline")
    print("=" * 65)

    # Step 1 -- Load training data
    print("\n[1/4]  Loading play-by-play data...")
    df = build_training_data()
    n_reg = (df["is_overtime"] == 0).sum()
    n_ot = (df["is_overtime"] == 1).sum()
    print(f"       {len(df):,} plays loaded from {df['season'].nunique()} seasons.")
    print(f"       {n_reg:,} regulation plays  |  {n_ot:,} OT plays (excluded from training)")

    # Step 2 -- Train model
    print("\n[2/4]  Training XGBoost + calibration model (regulation plays only)...")
    model = WinProbabilityModel(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        min_child_weight=10,
        calibration_method="isotonic",
        test_seasons=[2023, 2024],
        n_calib_seasons=3,
    )
    model.fit(df)

    # Print metrics summary
    cm = model.calib_metrics_
    print(f"\n       Calibration {model.calib_seasons_used_}:  "
          f"log_loss={cm['log_loss']:.4f}  brier={cm['brier_score']:.4f}  "
          f"mean_calib_err={cm['mean_calib_error']:.4f}  accuracy={cm['accuracy']:.4f}")
    if model.test_metrics_ is not None:
        tm = model.test_metrics_
        print(f"       Test {model.test_seasons_present_}:         "
              f"log_loss={tm['log_loss']:.4f}  brier={tm['brier_score']:.4f}  "
              f"mean_calib_err={tm['mean_calib_error']:.4f}  accuracy={tm['accuracy']:.4f}")

    # Step 3 -- Show feature importance
    print("\n[3/4]  Top 10 feature importances (gain):")
    try:
        fi = model.feature_importance().head(10)
        for _, row in fi.iterrows():
            print(f"       {row['feature']:<35s}  {row['importance']:>10.1f}")
    except Exception as exc:
        print(f"       (skipped: {exc})")

    # Step 4 -- Save
    print("\n[4/4]  Saving model...")
    saved_path = model.save()
    print(f"       Saved -> {saved_path}")

    # ── Smoke test: 4th-down decision engine example ──────────────────────────
    print("\n" + "=" * 65)
    print("  Smoke test -- 4th-down decision engine example")
    print("=" * 65)

    wp = WinProbabilityModel.load(saved_path)
    engine = FourthDownDecisionEngine(wp_model=wp)

    # Scenario: 4th-and-2 from opponent 17, tied game, 4Q 2:00 left (2024 season)
    # Kickoff touchback in 2024 = 30-yard line = yardline_100 = 70
    SEASON = 2024
    kickoff_tb = engine.kickoff_touchback_yardline(SEASON)

    current = dict(
        score_differential=0, quarter=4, seconds_remaining=120,
        yardline_100=17, down=4, ydstogo=2,
        offense_timeouts=1, defense_timeouts=2,
        is_overtime=0, overtime_possession_number=0,
    )

    # GO: p_conv = 0.65 (short yardage near goal line)
    # success: gained the first down, original team at opp-15, 1st & 10
    # (same team keeps ball — use _same_possession, no possession flip)
    success = engine._same_possession(current)
    success.update(dict(
        yardline_100=15, down=1, ydstogo=10, seconds_remaining=112,
    ))

    # failure: opponent takes over at opp-17 → from their view: own 83
    # (ball changes hands — use _flip_possession)
    failure = engine._flip_possession(current)
    failure.update(dict(
        yardline_100=100 - current["yardline_100"], down=1, ydstogo=10,
        seconds_remaining=112,
    ))

    # FG: p_fg = 0.88 (34-yard FG: yardline_100=17 + 17 = 34 yards)
    # make: +3, opponent receives kickoff at own 30 (yardline_100=70 in 2024)
    # Build from opponent's perspective: they are down 3
    fg_make = engine._flip_possession(current)
    fg_make["score_differential"] = -(current["score_differential"] + 3)
    fg_make.update(dict(yardline_100=kickoff_tb, down=1, ydstogo=10, seconds_remaining=112))

    # miss: opponent takes over at the line of scrimmage.
    # If attempted from inside own 20 (yl < 20), spot at own 20 (yl=80).
    fg_miss = engine._flip_possession(current)
    fg_miss.update(dict(
        yardline_100=max(80.0, 100.0 - current["yardline_100"]),
        down=1, ydstogo=10, seconds_remaining=112,
    ))

    # PUNT from opp-17 → likely touchback (punt into end zone) → opponent at own 20
    # opponent_start = 80 (punt touchback, always 20-yard line)
    punt = engine._flip_possession(current)
    punt.update(dict(
        yardline_100=FourthDownDecisionEngine.PUNT_TOUCHBACK_YARDLINE_100,
        down=1, ydstogo=10, seconds_remaining=112,
    ))

    result = engine.recommend(
        current_state=current,
        p_conv=0.65, success_state=success, failure_state=failure,
        p_fg=0.88, fg_make_state=fg_make, fg_miss_state=fg_miss,
        punt_state=punt,
    )

    print(f"\n  Situation: 4th & 2 from opp-17, tied, 2:00 Q4  (season={SEASON})")
    print(f"  Kickoff touchback spot: yardline_100 = {kickoff_tb} (own {100 - kickoff_tb:.0f})")
    print(f"  p_conv = 0.65  |  p_fg = 0.88")
    print(f"\n  WP (GO)   = {result['wp_go']:.3f}")
    print(f"  WP (FG)   = {result['wp_fg']:.3f}")
    print(f"  WP (PUNT) = {result['wp_punt']:.3f}")
    print(f"\n  >> Recommended decision: {result['decision']}")
    print(f"  GO vs FG   margin = {result['margin_go_vs_fg']:+.3f}")
    print(f"  GO vs PUNT margin = {result['margin_go_vs_punt']:+.3f}")
    print()

    # ── OT smoke test: verify transformer routing ─────────────────────────────
    print("  OT smoke test (pre-2025 regular season):")
    ot_state = dict(
        score_differential=0, quarter=5, seconds_remaining=480,
        yardline_100=75, down=1, ydstogo=10,
        offense_timeouts=2, defense_timeouts=2,
        is_overtime=1, overtime_possession_number=0,
        guaranteed_possession=0,
    )
    ot_wp = wp.predict_proba(ot_state)
    print(f"  Tied, OT first possession, 8:00 left  -> WP = {ot_wp:.3f}  (expect ~0.50)")

    ot_state_2025 = dict(ot_state, guaranteed_possession=1, overtime_possession_number=1,
                         score_differential=-3)
    ot_wp_2025 = wp.predict_proba(ot_state_2025)
    print(f"  Down 3, OT second possession (2025 rules), 8:00 left -> WP = {ot_wp_2025:.3f}  (expect <0.50)")
    print()


if __name__ == "__main__":
    main()
