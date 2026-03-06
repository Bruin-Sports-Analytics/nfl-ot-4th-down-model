"""
build_features.py
=================
Computes offensive and defensive rolling stats from drive_summary.csv
and merges them into the enhanced fourth-down dataset.

Outputs (written to data/processed/):
  fourth_down_enhanced.csv    -- main training dataset for the model
  team_stats_snapshot.csv     -- latest rolling values per team (used by app.py)

Run this once before training:
    python src/data/build_features.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (project-relative — no hardcoded machine paths)
# ---------------------------------------------------------------------------
_ROOT          = Path(__file__).resolve().parents[2]   # nfl-ot-4th-down-model/
DATA_DIR       = _ROOT / "data"
PROCESSED      = DATA_DIR / "processed"
DRIVE_PATH     = PROCESSED / "drive_summary.parquet"
FOURTH_PATH    = PROCESSED / "fourth_down_with_features.parquet"
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT_FOURTH     = PROCESSED / "fourth_down_enhanced.csv"
OUT_TEAM_STATS = PROCESSED / "team_stats_snapshot.csv"

ROLL_WINDOW = 15   # rolling game window
MIN_PERIODS = 3    # minimum games before producing a value

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading data...")
ds = pd.read_parquet(DRIVE_PATH)
fd = pd.read_parquet(FOURTH_PATH)
print(f"  drive_summary shape         : {ds.shape}")
print(f"  fourth_down_features shape  : {fd.shape}")
print(f"  drive_summary columns       : {list(ds.columns)}")

# ---------------------------------------------------------------------------
# 2. Offensive rolling stats (per posteam, per game)
# ---------------------------------------------------------------------------
print(f"\nBuilding offensive rolling {ROLL_WINDOW}-game stats...")

# Core drive-level aggregations (always available)
off_agg: dict = {
    "game_epa":     ("drive_epa",          "mean"),
    "game_success": ("drive_success_rate", "mean"),
    "game_points":  ("drive_points",       "sum"),
}

# Optional columns: add if present in drive_summary
if "drive_yards" in ds.columns and "drive_plays" in ds.columns:
    off_agg["game_yards"]  = ("drive_yards", "sum")
    off_agg["game_plays"]  = ("drive_plays", "sum")
    print("  → yards_per_play features will be computed (drive_yards + drive_plays found)")

if "drive_turnovers" in ds.columns:
    off_agg["game_to"] = ("drive_turnovers", "sum")
    print("  → turnover_rate feature will be computed (drive_turnovers found)")

game_off = (
    ds.groupby(["season", "week", "game_id", "posteam"])
    .agg(**off_agg)
    .reset_index()
    .sort_values(["posteam", "season", "week"])
    .reset_index(drop=True)
)

# Always-computed rolling features
for new_col, src_col in [
    ("epa_per_game_roll15",    "game_epa"),
    ("success_rate_roll15",    "game_success"),
    ("points_per_game_roll15", "game_points"),
]:
    game_off[new_col] = (
        game_off.groupby("posteam")[src_col]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

# Optional rolling features (if source columns were available)
if "game_yards" in game_off.columns and "game_plays" in game_off.columns:
    game_off["game_ypp"] = game_off["game_yards"] / game_off["game_plays"].replace(0, np.nan)
    game_off["yards_per_play_roll15"] = (
        game_off.groupby("posteam")["game_ypp"]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

if "game_to" in game_off.columns:
    game_off["turnover_rate_roll15"] = (
        game_off.groupby("posteam")["game_to"]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

# ---------------------------------------------------------------------------
# 3. Defensive rolling stats (what offenses scored against each defteam)
# ---------------------------------------------------------------------------
print(f"Building defensive rolling {ROLL_WINDOW}-game stats...")

def_agg: dict = {
    "def_game_epa":     ("drive_epa",          "mean"),
    "def_game_success": ("drive_success_rate", "mean"),
    "def_game_points":  ("drive_points",       "sum"),
}

if "drive_yards" in ds.columns and "drive_plays" in ds.columns:
    def_agg["def_game_yards"] = ("drive_yards", "sum")
    def_agg["def_game_plays"] = ("drive_plays", "sum")

game_def = (
    ds.groupby(["season", "week", "game_id", "defteam"])
    .agg(**def_agg)
    .reset_index()
    .rename(columns={"defteam": "team"})
    .sort_values(["team", "season", "week"])
    .reset_index(drop=True)
)

for new_col, src_col in [
    ("def_epa_per_game_roll15",    "def_game_epa"),
    ("def_success_rate_roll15",    "def_game_success"),
    ("def_points_per_game_roll15", "def_game_points"),
]:
    game_def[new_col] = (
        game_def.groupby("team")[src_col]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

if "def_game_yards" in game_def.columns and "def_game_plays" in game_def.columns:
    game_def["def_game_ypp"] = (
        game_def["def_game_yards"] / game_def["def_game_plays"].replace(0, np.nan)
    )
    game_def["def_yards_per_play_roll15"] = (
        game_def.groupby("team")["def_game_ypp"]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

# ---------------------------------------------------------------------------
# 4. Merge into fourth_down dataset
# ---------------------------------------------------------------------------
print("Merging into fourth_down dataset...")

off_rolling_cols = [
    "epa_per_game_roll15", "success_rate_roll15", "points_per_game_roll15",
]
if "yards_per_play_roll15" in game_off.columns:
    off_rolling_cols.append("yards_per_play_roll15")
if "turnover_rate_roll15" in game_off.columns:
    off_rolling_cols.append("turnover_rate_roll15")

def_rolling_cols = [
    "def_epa_per_game_roll15", "def_success_rate_roll15", "def_points_per_game_roll15",
]
if "def_yards_per_play_roll15" in game_def.columns:
    def_rolling_cols.append("def_yards_per_play_roll15")

off_merge = game_off[["game_id", "posteam"] + off_rolling_cols]
def_merge = (
    game_def[["game_id", "team"] + def_rolling_cols]
    .rename(columns={"team": "defteam"})
)

fd_enhanced = fd.merge(off_merge, on=["game_id", "posteam"], how="left")
fd_enhanced = fd_enhanced.merge(def_merge, on=["game_id", "defteam"], how="left")

all_new_cols = off_rolling_cols + def_rolling_cols
print(f"  Enhanced dataset shape : {fd_enhanced.shape}")
print(f"  Null counts (new rolling columns):")
for col in all_new_cols:
    if col in fd_enhanced.columns:
        n_null = fd_enhanced[col].isnull().sum()
        pct    = n_null / len(fd_enhanced) * 100
        print(f"    {col:<35}: {n_null:,}  ({pct:.1f}%)")

fd_enhanced.to_csv(OUT_FOURTH, index=False)
print(f"\nSaved enhanced dataset → {OUT_FOURTH}")

# ---------------------------------------------------------------------------
# 5. Team stats snapshot (latest rolling values — used by app.py)
# ---------------------------------------------------------------------------
print("\nBuilding team stats snapshot...")

snap_off_cols = ["epa_per_game_roll15", "success_rate_roll15", "points_per_game_roll15"]
if "yards_per_play_roll15" in game_off.columns:
    snap_off_cols.append("yards_per_play_roll15")

off_snap = (
    game_off.sort_values(["posteam", "season", "week"])
    .groupby("posteam")
    .last()[snap_off_cols]
    .reset_index()
    .rename(columns={"posteam": "team"})
)

# 4th-down go/stop rates: take each team's latest values from the enhanced dataset
fd_rates_cols = ["posteam"]
for rate_col in ["go_conv_rate", "go_stop_rate"]:
    if rate_col in fd_enhanced.columns:
        fd_rates_cols.append(rate_col)

if len(fd_rates_cols) > 1:
    fd_rates = (
        fd_enhanced[fd_rates_cols]
        .dropna()
        .groupby("posteam")
        .last()
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    off_snap = off_snap.merge(fd_rates, on="team", how="left")

snap_def_cols = ["def_epa_per_game_roll15", "def_success_rate_roll15", "def_points_per_game_roll15"]
if "def_yards_per_play_roll15" in game_def.columns:
    snap_def_cols.append("def_yards_per_play_roll15")

def_snap = (
    game_def.sort_values(["team", "season", "week"])
    .groupby("team")
    .last()[snap_def_cols]
    .reset_index()
)

team_stats = off_snap.merge(def_snap, on="team", how="outer").sort_values("team").reset_index(drop=True)

print(f"  Teams in snapshot : {len(team_stats)}")
print(team_stats.to_string())
team_stats.to_csv(OUT_TEAM_STATS, index=False)
print(f"\nSaved team stats snapshot → {OUT_TEAM_STATS}")
print("\nDone. Now run:  python -m src.models.fourth_down_conversion")
