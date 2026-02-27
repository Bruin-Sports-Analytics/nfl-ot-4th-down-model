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
  cannot model discontinuities introduced by rule changes (e.g. the 2023 NFL
  overtime rule where both teams are guaranteed one possession).

Why calibration is required:
  Raw XGBoost scores optimise log-loss but are not guaranteed to be
  well-calibrated probabilities.  CalibratedClassifierCV with isotonic
  regression fits a monotone mapping from raw model output to empirical win
  rates, so that a predicted 70 % corresponds to an observed ≈70 % win rate.
  Calibration is critical for an expected-value calculation like:
      wp_go = p_conv * wp(success_state) + (1 - p_conv) * wp(failure_state)

Leakage prevention:
  Only pre-play state variables are used.  Features that are realised *during*
  or *after* the play (EPA, yards_gained, wpa, result of the current down) are
  never included.  A strict temporal split is used: models are evaluated on
  future seasons that were never seen during training or calibration.

Overtime (2025 NFL rules):
  Starting with the 2025 regular season:
    • Both teams are guaranteed one possession before sudden death.
    • Overtime is shortened to 10 minutes.
  The model uses `is_overtime` and `overtime_possession_number` (0=first
  team's possession, 1=second team's, 2+=sudden death) to represent these
  phases.  Note: training data from 2016-2021 uses older OT rules (first
  score wins).  The model upweights recent seasons automatically via the
  temporal structure; for high-stakes OT calibration consider filtering
  OT plays to 2022+ when both-team-possession rules were first applied in
  playoffs.

Validation approach:
  Hold-out test seasons (default 2023-2024) are never seen during tree
  learning or calibration.  The most recent non-test season is reserved for
  early-stopping and isotonic calibration fitting.  Reported metrics:
    • Log-loss   (lower is better; logistic baseline ≈ 0.65)
    • Brier score (lower is better; climatology baseline ≈ 0.25)
    • Mean calibration error: average |predicted − actual win rate| per bin

Required columns for training:
  score_differential, quarter, seconds_remaining, yardline_100, down, ydstogo,
  offense_timeouts, defense_timeouts, is_overtime, overtime_possession_number,
  offense_team_won_game (target), season

Optional columns (default to 0 if absent):
  offense_team_strength_rating, defense_team_strength_rating

Usage
-----
  # ── Build & train ─────────────────────────────────────────────────────────
  from src.models.win_probability import WinProbabilityModel, build_training_data

  df   = build_training_data()          # reads data/raw/pbp_2016_2024.parquet
  wp   = WinProbabilityModel()
  wp.fit(df)
  wp.save()                             # → models/win_probability_model.pkl

  # ── Inference ──────────────────────────────────────────────────────────────
  wp = WinProbabilityModel.load("models/win_probability_model.pkl")

  state = dict(
      score_differential=0, quarter=4, seconds_remaining=120,
      yardline_100=35, down=4, ydstogo=2,
      offense_timeouts=2, defense_timeouts=1,
      is_overtime=0, overtime_possession_number=0,
  )
  wp.predict_proba(state)               # e.g. 0.52

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
    "offense_team_strength_rating",  # float, 0-centred (default 0)
    "defense_team_strength_rating",  # float, 0-centred (default 0)
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
    # Non-linear time effects — captures increasing leverage of each second
    "seconds_remaining_sqrt",
    "seconds_remaining_log1p",
    # Score × time interaction — the core WP signal
    "score_x_time",
    "urgency",
    "clock_leverage",
    # Score state dummies
    "is_winning",
    "is_tied",
    "is_losing",
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
    # Overtime phase flags (2025 rules)
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
      2 → third+ (sudden death under 2025 rules, clipped at 2)

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
        ``OPTIONAL_STATE_KEYS`` (as zeros), ``offense_team_won_game``, and
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

    df = pd.read_parquet(parquet_path, columns=load_cols)

    # ── Filter: keep only plays with a full pre-play state ────────────────────
    # Missing `result` means the game record is incomplete — drop those rows.
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
    # We want P(posteam wins), so:
    #   posteam is home  AND result > 0  → posteam won (1)
    #   posteam is away  AND result < 0  → posteam won (1)
    #   all other cases (tied or losing) → 0
    # Ties are treated as losses (WP interpretation: ties benefit neither side).
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

    # Optional strength ratings — default to neutral (0) if not supplied
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
    It is designed to be called programmatically with arbitrary hypothetical
    states, making it suitable for use inside a 4th-down decision engine.

    Parameters
    ----------
    n_estimators:
        Maximum number of trees.  Early stopping typically halts well before
        this ceiling when a calibration season is available.
    max_depth:
        Maximum tree depth.  6 captures rich interactions without overfitting.
    learning_rate:
        Shrinkage factor per tree.  0.05 with early stopping is a good default.
    subsample, colsample_bytree:
        Row- and column-sampling fractions for stochastic gradient boosting.
        Both at 0.8 add mild regularisation without sacrificing accuracy.
    min_child_weight:
        Minimum sum of instance weights in a leaf.  Set to 20 to prevent the
        model fitting to tiny sub-populations (e.g. very rare OT states).
    gamma:
        Minimum loss reduction for a further split.  Effectively a pruning
        threshold; 1.0 keeps the tree focused on splits that matter.
    reg_alpha, reg_lambda:
        L1 / L2 regularisation on leaf weights.
    calibration_method:
        ``"isotonic"`` (non-parametric, better with large data) or
        ``"sigmoid"`` (Platt scaling, better with small data).
    test_seasons:
        Seasons held out for final evaluation.  Never used in training or
        calibration.  Default: [2023, 2024].
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 20,
        gamma: float = 1.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        calibration_method: str = "isotonic",
        test_seasons: list[int] | None = None,
        random_state: int = 42,
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

        # Internal state — populated by fit()
        self._base_model: XGBClassifier | None = None
        self._calibrator: IsotonicRegression | LogisticRegression | None = None
        self._feature_cols: list[str] = []
        self._is_fitted: bool = False

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive non-linear and interaction features from raw state columns.

        All transforms are deterministic and applied identically at train time
        and inference time, preventing any train-serve skew.

        Key derived features
        --------------------
        seconds_remaining_sqrt:
            sqrt-scale captures the accelerating leverage of each second as
            the clock winds down (a 1-second difference at 0:05 matters far
            more than at 30:00).

        score_x_time:
            score_differential × seconds_remaining / 3600.  The primary WP
            signal: a 3-point deficit means very different things in the first
            quarter vs. the final minute.

        urgency:
            |score_differential| / (seconds_remaining + 1).  High when a
            large lead exists with little time — near-certain win.

        clock_leverage:
            1 / (|score_differential| + 1) / (seconds_remaining + 60).  High
            when the game is close AND late — maximum uncertainty.

        ot_*:
            Overtime phase dummies derived from ``is_overtime`` and
            ``overtime_possession_number``.  Reflect 2025 NFL rules where
            both teams receive one possession before sudden death.
        """
        out = df.copy()
        sec = out["seconds_remaining"].astype(float)
        sd = out["score_differential"].astype(float)
        ot = out["is_overtime"].astype(int)
        pn = out["overtime_possession_number"].astype(int)

        # Non-linear time
        out["seconds_remaining_sqrt"] = np.sqrt(sec)
        out["seconds_remaining_log1p"] = np.log1p(sec)

        # Score × time (core WP signal)
        out["score_x_time"] = sd * sec / 3600.0
        out["urgency"] = np.abs(sd) / (sec + 1.0)
        out["clock_leverage"] = 1.0 / (np.abs(sd) + 1.0) / (sec + 60.0)

        # Score state
        out["is_winning"] = (sd > 0).astype(int)
        out["is_tied"] = (sd == 0).astype(int)
        out["is_losing"] = (sd < 0).astype(int)
        out["abs_score_diff"] = np.abs(sd)
        out["score_differential_sq"] = sd ** 2

        # Timeouts
        out["timeout_diff"] = out["offense_timeouts"] - out["defense_timeouts"]
        out["total_timeouts"] = out["offense_timeouts"] + out["defense_timeouts"]

        # Down & distance
        out["ydstogo_log1p"] = np.log1p(out["ydstogo"])
        out["short_yardage"] = (out["ydstogo"] <= 2).astype(int)

        # Field position
        out["fg_range"] = (out["yardline_100"] <= 35).astype(int)   # ≈52-yd FG
        out["red_zone"] = (out["yardline_100"] <= 20).astype(int)
        out["scoring_position"] = (out["yardline_100"] <= 10).astype(int)

        # Overtime phase (2025 rules: 0=first poss, 1=second poss, 2+=sudden death)
        out["ot_first_poss"] = (ot & (pn == 0)).astype(int)
        out["ot_second_poss"] = (ot & (pn == 1)).astype(int)
        out["ot_sudden_death"] = (ot & (pn >= 2)).astype(int)
        # On the second+ possession, trailing team MUST score to survive
        out["ot_must_score"] = ((ot == 1) & (pn >= 1) & (sd < 0)).astype(int)
        # Leading on the first OT possession: opponent still gets a chance (2025)
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
        """Train the Win Probability model.

        Parameters
        ----------
        df:
            One row per play.  Must contain all ``REQUIRED_STATE_KEYS``,
            ``"offense_team_won_game"`` (binary target), and ``"season"``
            (int, used for temporal splitting).

        Returns
        -------
        self  (chainable)

        Temporal split strategy
        -----------------------
        1. ``test_seasons``         – completely held out; reported at the end.
        2. ``calib_season``         – most recent non-test season; used for
                                      XGBoost early-stopping and calibration.
        3. remaining train seasons  – used for tree learning only.

        This ensures zero information leakage between splits: calibration
        cannot overfit to training data, and test metrics are unbiased.
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

        # Most recent training season → calibration + early-stopping validation
        calib_season = max(train_seasons)
        fit_seasons = [s for s in train_seasons if s != calib_season]

        if not fit_seasons:
            logger.warning(
                "Only one non-test season available (%s). "
                "Using it for both fitting and calibration.",
                calib_season,
            )
            fit_seasons = train_seasons

        logger.info(
            "Fit seasons: %s | Calibration season: %s | Test seasons: %s",
            fit_seasons, calib_season, test_seasons_present,
        )

        X_fit = df_feat.loc[df_feat["season"].isin(fit_seasons), self._feature_cols].fillna(0.0).values
        y_fit = df_feat.loc[df_feat["season"].isin(fit_seasons), "offense_team_won_game"].values

        X_calib = df_feat.loc[df_feat["season"] == calib_season, self._feature_cols].fillna(0.0).values
        y_calib = df_feat.loc[df_feat["season"] == calib_season, "offense_team_won_game"].values

        # ── Base XGBoost ──────────────────────────────────────────────────────
        # XGBoost is appropriate because:
        #  • Non-linear WP surface (score × time interactions are automatic)
        #  • Handles mixed-type features without normalisation
        #  • Robust to irrelevant features via regularisation (gamma, reg_lambda)
        #  • Early stopping prevents overfitting on large play-by-play data
        #
        # Logistic regression is insufficient because:
        #  • It cannot model the discontinuous WP surface (same score diff
        #    has radically different meaning at different times/OT phases)
        #  • Would require hundreds of hand-crafted interaction terms
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
            eval_set=[(X_calib, y_calib)],
            verbose=False,
        )
        logger.info("XGBoost best iteration: %d", base.best_iteration)
        self._base_model = base

        # ── Calibration ───────────────────────────────────────────────────────
        # We fit calibration manually rather than via CalibratedClassifierCV
        # because that class's cv="prefit" mode was removed in scikit-learn 1.6+.
        # The approach: get raw XGBoost probabilities on the held-out calibration
        # season, then fit a monotone mapping (isotonic) or logistic mapping
        # (Platt / sigmoid) from raw scores → calibrated probabilities.
        raw_calib = base.predict_proba(X_calib)[:, 1]

        if self.calibration_method == "isotonic":
            # Non-parametric monotone mapping; preferred for large datasets
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_calib, y_calib)
        else:
            # Platt scaling: logistic regression on the raw score (sigmoid)
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs")
            self._calibrator.fit(raw_calib.reshape(-1, 1), y_calib)

        self._is_fitted = True

        # ── Metrics ───────────────────────────────────────────────────────────
        calib_probs = self._apply_calibrator(raw_calib)
        logger.info("── Calibration season (%d) metrics ──────", calib_season)
        self.calib_metrics_ = self._report_metrics(y_calib, calib_probs, split=f"calib({calib_season})")
        self.calib_season_ = calib_season

        self.test_metrics_: dict[str, float] | None = None
        self.test_seasons_present_: list[int] = test_seasons_present
        if test_seasons_present:
            X_test = df_feat.loc[
                df_feat["season"].isin(test_seasons_present), self._feature_cols
            ].fillna(0.0).values
            y_test = df_feat.loc[
                df_feat["season"].isin(test_seasons_present), "offense_team_won_game"
            ].values
            raw_test = self._base_model.predict_proba(X_test)[:, 1]
            test_probs = self._apply_calibrator(raw_test)
            logger.info("── Test seasons %s metrics ──────────", test_seasons_present)
            self.test_metrics_ = self._report_metrics(y_test, test_probs, split=f"test({test_seasons_present})")

        return self

    # ── Calibration helper ────────────────────────────────────────────────────

    def _apply_calibrator(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw XGBoost probabilities through the fitted calibrator."""
        if isinstance(self._calibrator, IsotonicRegression):
            return self._calibrator.predict(raw_probs)
        else:
            # LogisticRegression (Platt scaling): expects 2-D input
            return self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, state_dict: dict[str, Any]) -> float:
        """Return P(offensive team wins | current game state).

        Parameters
        ----------
        state_dict:
            Dict with all keys in ``REQUIRED_STATE_KEYS``.
            Optional keys from ``OPTIONAL_STATE_KEYS`` default to 0.

        Returns
        -------
        float in [0, 1]
            Calibrated win probability for the offensive team.

        Example
        -------
        >>> wp.predict_proba(dict(
        ...     score_differential=0, quarter=4, seconds_remaining=120,
        ...     yardline_100=35, down=4, ydstogo=2,
        ...     offense_timeouts=2, defense_timeouts=1,
        ...     is_overtime=0, overtime_possession_number=0,
        ... ))
        0.518
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._state_to_features(state_dict)
        raw = self._base_model.predict_proba(X)[0, 1]
        return float(self._apply_calibrator(np.array([raw]))[0])

    def simulate_state(self, state_dict: dict[str, Any]) -> float:
        """Return WP for a hypothetical (counterfactual) game state.

        Semantically identical to ``predict_proba`` but intended for states
        that are *constructed* to represent the outcome of a hypothetical play
        (e.g. post-conversion, post-FG, or post-punt field position).

        This is the primary method used inside a 4th-down decision engine
        where the caller assembles each post-play state and queries the model:

            wp_go = p_conv * wp.simulate_state(success_state)
                  + (1 - p_conv) * wp.simulate_state(failure_state)

        Parameters
        ----------
        state_dict:
            The hypothetical game state *after* the play.  Must contain all
            ``REQUIRED_STATE_KEYS``.

        Returns
        -------
        float in [0, 1]

        Example
        -------
        >>> current = dict(score_differential=-3, quarter=4,
        ...               seconds_remaining=90, yardline_100=4,
        ...               down=4, ydstogo=4,
        ...               offense_timeouts=1, defense_timeouts=2,
        ...               is_overtime=0, overtime_possession_number=0)

        >>> # State after a successful 4th-down TD conversion
        >>> success_state = {**current,
        ...     score_differential=4,   # -3 + 7
        ...     yardline_100=75,        # opponent kicks from end zone ~25 yd line
        ...     down=1, ydstogo=10,
        ...     seconds_remaining=84,
        ... }
        >>> wp.simulate_state(success_state)
        0.89
        """
        return self.predict_proba(state_dict)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Serialise the fitted model to disk with joblib.

        Parameters
        ----------
        path:
            Destination file.  Defaults to
            ``<project_root>/models/win_probability_model.pkl``.

        Returns
        -------
        Path  – the file that was written.
        """
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
                )
            },
        }
        joblib.dump(payload, path)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "WinProbabilityModel":
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            Path to a ``.pkl`` file produced by ``save()``.

        Returns
        -------
        WinProbabilityModel  – fitted and ready for inference.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        instance = cls(**payload.get("hyperparams", {}))
        instance._base_model = payload["base_model"]
        instance._calibrator = payload["calibrator"]
        instance._feature_cols = payload["feature_cols"]
        instance._is_fitted = True
        logger.info("Model loaded ← %s", path)
        return instance

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _report_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        split: str = "eval",
        n_bins: int = 10,
    ) -> dict[str, float]:
        """Compute and log log-loss, Brier score, and calibration error.

        Calibration curve:
          ``calibration_curve`` bins the predicted probabilities and computes
          the actual win rate in each bin.  The mean absolute deviation between
          the two curves is the ``mean_calib_error``.  Values < 0.02 are
          considered well-calibrated for WP models.

        Returns
        -------
        dict with keys: log_loss, brier_score, mean_calib_error
        """
        ll = log_loss(y_true, y_prob)
        bs = brier_score_loss(y_true, y_prob)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        mce = float(np.mean(np.abs(frac_pos - mean_pred)))

        logger.info(
            "[%s]  n=%d  log_loss=%.4f  brier=%.4f  mean_calib_err=%.4f",
            split, len(y_true), ll, bs, mce,
        )
        return {"log_loss": ll, "brier_score": bs, "mean_calib_error": mce}

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances from the base XGBoost model.

        Retrieves ``gain``-based importance (average reduction in loss per
        split, weighted by number of samples) from the fitted XGBoost
        estimator inside the calibrated wrapper.

        Returns
        -------
        pd.DataFrame sorted descending by ``importance``.
        """
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
    """Load the saved Win Probability model from disk.

    Parameters
    ----------
    path:
        Path to ``.pkl`` file.  Defaults to
        ``<project_root>/models/win_probability_model.pkl``.
    """
    if path is None:
        path = _MODEL_DIR / "win_probability_model.pkl"
    return WinProbabilityModel.load(path)


# ── 4th-down decision engine integration ──────────────────────────────────────

class FourthDownDecisionEngine:
    """Integrates the Win Probability model with sub-model probabilities to
    recommend the optimal 4th-down decision via expected-value maximisation.

    The three decisions and their expected WP:

        wp_go   = p_conv * wp(success_state) + (1 - p_conv) * wp(failure_state)
        wp_fg   = p_make * wp(make_state)    + (1 - p_make) * wp(miss_state)
        wp_punt = wp(punt_state)

        decision = argmax(wp_go, wp_fg, wp_punt)

    Each ``*_state`` dict represents the game state *after* the hypothetical
    outcome.  The caller is responsible for constructing these states using
    their conversion / FG / punt sub-models.

    Parameters
    ----------
    wp_model:
        A fitted ``WinProbabilityModel`` instance.
    """

    def __init__(self, wp_model: WinProbabilityModel) -> None:
        if not wp_model._is_fitted:
            raise ValueError("wp_model must be fitted before use.")
        self.wp = wp_model

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

        Parameters
        ----------
        current_state:
            The current pre-play game state (used only for context / logging).
        p_conv:
            P(4th-down conversion succeeds) from the conversion sub-model.
        success_state:
            Game state *after* a successful conversion (down=1, ydstogo=10,
            yardline advanced, clock ticked).
        failure_state:
            Game state *after* a failed conversion (opponent takes over at
            current yardline, possessions flipped — note: state should reflect
            *opponent's* possession, so score_differential is negated).
        p_fg:
            P(field goal made) from the FG sub-model.
        fg_make_state:
            Game state after a made FG (score_differential += 3, kickoff).
        fg_miss_state:
            Game state after a missed FG (opponent takes over at ~spot of kick).
        punt_state:
            Game state after the punt (opponent takes over at expected field
            position from punt sub-model).

        Returns
        -------
        dict with keys:
            decision  – "GO" | "FG" | "PUNT"
            wp_go     – expected WP for going for it
            wp_fg     – expected WP for attempting FG
            wp_punt   – expected WP for punting
            margin_go_vs_fg    – wp_go   - wp_fg
            margin_go_vs_punt  – wp_go   - wp_punt

        Notes
        -----
        *Post-play states represent the opponent's next possession.*
        When the offense fails to convert or punts, the other team takes over
        — their WP is 1 - wp_model.predict_proba(opponent_state).  The caller
        must negate ``score_differential`` and flip timeouts in the post-play
        state to reflect the perspective change, OR construct the state from
        the new offensive team's point of view and the method will call
        ``simulate_state`` on it directly.
        """
        wp_go = (
            p_conv * self.wp.simulate_state(success_state)
            + (1.0 - p_conv) * self.wp.simulate_state(failure_state)
        )
        wp_fg = (
            p_fg * self.wp.simulate_state(fg_make_state)
            + (1.0 - p_fg) * self.wp.simulate_state(fg_miss_state)
        )
        wp_punt = self.wp.simulate_state(punt_state)

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


# ── CLI entry point ─────────────────────────────────────────────────────────

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
    print("  NFL Win Probability Model — Training Pipeline")
    print("=" * 65)

    # Step 1 – Load training data
    print("\n[1/4]  Loading play-by-play data...")
    df = build_training_data()
    print(f"       {len(df):,} plays loaded from {df['season'].nunique()} seasons.")

    # Step 2 – Train model
    print("\n[2/4]  Training XGBoost + calibration model...")
    model = WinProbabilityModel(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        calibration_method="isotonic",
        test_seasons=[2023, 2024],
    )
    model.fit(df)

    # Print metrics summary
    cm = model.calib_metrics_
    print(f"\n       Calibration ({model.calib_season_}):  "
          f"log_loss={cm['log_loss']:.4f}  brier={cm['brier_score']:.4f}  "
          f"mean_calib_err={cm['mean_calib_error']:.4f}")
    if model.test_metrics_ is not None:
        tm = model.test_metrics_
        print(f"       Test {model.test_seasons_present_}:         "
              f"log_loss={tm['log_loss']:.4f}  brier={tm['brier_score']:.4f}  "
              f"mean_calib_err={tm['mean_calib_error']:.4f}")

    # Step 3 – Show feature importance
    print("\n[3/4]  Top 10 feature importances (gain):")
    try:
        fi = model.feature_importance().head(10)
        for _, row in fi.iterrows():
            print(f"       {row['feature']:<35s}  {row['importance']:>10.1f}")
    except Exception as exc:
        print(f"       (skipped: {exc})")

    # Step 4 – Save
    print("\n[4/4]  Saving model...")
    saved_path = model.save()
    print(f"       Saved → {saved_path}")

    # ── Smoke test: 4th-down decision engine example ──────────────────────────
    print("\n" + "=" * 65)
    print("  Smoke test — 4th-down decision engine example")
    print("=" * 65)

    wp = WinProbabilityModel.load(saved_path)
    engine = FourthDownDecisionEngine(wp_model=wp)

    # Scenario: 4th-and-2 from opponent 17, tied game, 4Q 2:00 left
    current = dict(
        score_differential=0, quarter=4, seconds_remaining=120,
        yardline_100=17, down=4, ydstogo=2,
        offense_timeouts=1, defense_timeouts=2,
        is_overtime=0, overtime_possession_number=0,
    )

    # p_conv = 0.65 (short yardage near goal line)
    # success: TD scored, opponent kicks off from ~35, 1st & 10 at own 25
    success = dict(
        score_differential=7, quarter=4, seconds_remaining=112,
        yardline_100=75, down=1, ydstogo=10,
        offense_timeouts=1, defense_timeouts=2,
        is_overtime=0, overtime_possession_number=0,
    )
    # failure: opponent takes over at own 17, down by 0 (score_differential negated)
    failure = dict(
        score_differential=0, quarter=4, seconds_remaining=112,
        yardline_100=83, down=1, ydstogo=10,
        offense_timeouts=2, defense_timeouts=1,
        is_overtime=0, overtime_possession_number=0,
    )

    # p_fg = 0.88 (34-yard FG)
    # make: +3, opponent kicks off
    fg_make = dict(
        score_differential=3, quarter=4, seconds_remaining=112,
        yardline_100=75, down=1, ydstogo=10,
        offense_timeouts=1, defense_timeouts=2,
        is_overtime=0, overtime_possession_number=0,
    )
    # miss: opponent takes over at ~spot (own 24)
    fg_miss = dict(
        score_differential=0, quarter=4, seconds_remaining=112,
        yardline_100=76, down=1, ydstogo=10,
        offense_timeouts=2, defense_timeouts=1,
        is_overtime=0, overtime_possession_number=0,
    )

    # punt: opponent starts at own 8
    punt = dict(
        score_differential=0, quarter=4, seconds_remaining=112,
        yardline_100=92, down=1, ydstogo=10,
        offense_timeouts=2, defense_timeouts=1,
        is_overtime=0, overtime_possession_number=0,
    )

    result = engine.recommend(
        current_state=current,
        p_conv=0.65, success_state=success, failure_state=failure,
        p_fg=0.88, fg_make_state=fg_make, fg_miss_state=fg_miss,
        punt_state=punt,
    )

    print(f"\n  Situation: 4th & 2 from opp-17, tied, 2:00 Q4")
    print(f"  p_conv = 0.65  |  p_fg = 0.88")
    print(f"\n  WP (GO)   = {result['wp_go']:.3f}")
    print(f"  WP (FG)   = {result['wp_fg']:.3f}")
    print(f"  WP (PUNT) = {result['wp_punt']:.3f}")
    print(f"\n  ► Recommended decision: {result['decision']}")
    print(f"  GO vs FG   margin = {result['margin_go_vs_fg']:+.3f}")
    print(f"  GO vs PUNT margin = {result['margin_go_vs_punt']:+.3f}")
    print()


if __name__ == "__main__":
    main()
