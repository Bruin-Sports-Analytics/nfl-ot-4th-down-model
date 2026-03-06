"""
build_features.py
=================
Computes offensive and defensive rolling 15-game stats from drive_summary.csv
and merges them into fourth_down_with_features.csv.

Run this once before training:
    python src/build_features.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# Paths
# =============================================================================
BASE_DIR       = Path(r"C:\Users\andre\OneDrive\Desktop\BSA 4th Down Model")
DRIVE_PATH     = BASE_DIR / "data" / "drive_summary.csv"
FOURTH_PATH    = BASE_DIR / "data" / "fourth_down_with_features.csv"
OUT_FOURTH     = BASE_DIR / "data" / "fourth_down_enhanced.csv"
OUT_TEAM_STATS = BASE_DIR / "data" / "team_stats_snapshot.csv"

ROLL_WINDOW = 15   # rolling game window
MIN_PERIODS = 3    # minimum games needed before producing a value

# =============================================================================
# 1. Load
# =============================================================================
print("Loading data ...")
ds = pd.read_csv(DRIVE_PATH)
fd = pd.read_csv(FOURTH_PATH)
print(f"  drive_summary shape        : {ds.shape}")
print(f"  fourth_down_features shape : {fd.shape}")

# =============================================================================
# 2. Offensive rolling stats (per posteam, per game)
# =============================================================================
print(f"\nBuilding offensive rolling {ROLL_WINDOW}-game stats ...")
game_off = (
    ds.groupby(['season', 'week', 'game_id', 'posteam'])
    .agg(
        game_epa     = ('drive_epa',          'mean'),
        game_success = ('drive_success_rate', 'mean'),
        game_points  = ('drive_points',       'sum'),
    )
    .reset_index()
    .sort_values(['posteam', 'season', 'week'])
    .reset_index(drop=True)
)

for new_col, src_col in [
    ('epa_per_game_roll15',    'game_epa'),
    ('success_rate_roll15',    'game_success'),
    ('points_per_game_roll15', 'game_points'),
]:
    game_off[new_col] = (
        game_off.groupby('posteam')[src_col]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

# =============================================================================
# 3. Defensive rolling stats (what offenses did against each defteam)
# =============================================================================
print(f"Building defensive rolling {ROLL_WINDOW}-game stats ...")
game_def = (
    ds.groupby(['season', 'week', 'game_id', 'defteam'])
    .agg(
        def_game_epa     = ('drive_epa',          'mean'),
        def_game_success = ('drive_success_rate', 'mean'),
        def_game_points  = ('drive_points',       'sum'),
    )
    .reset_index()
    .rename(columns={'defteam': 'team'})
    .sort_values(['team', 'season', 'week'])
    .reset_index(drop=True)
)

for new_col, src_col in [
    ('def_epa_per_game_roll15',    'def_game_epa'),
    ('def_success_rate_roll15',    'def_game_success'),
    ('def_points_per_game_roll15', 'def_game_points'),
]:
    game_def[new_col] = (
        game_def.groupby('team')[src_col]
        .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean())
    )

# =============================================================================
# 4. Merge defensive stats into fourth_down dataset
# =============================================================================
print("Merging into fourth_down dataset ...")
def_merge = game_def[[
    'game_id', 'team',
    'def_epa_per_game_roll15',
    'def_success_rate_roll15',
    'def_points_per_game_roll15',
]].rename(columns={'team': 'defteam'})

# Also merge offensive rolling stats (to replace old roll6 columns)
off_merge = game_off[[
    'game_id', 'posteam',
    'epa_per_game_roll15',
    'success_rate_roll15',
    'points_per_game_roll15',
]]

fd_enhanced = fd.merge(off_merge, on=['game_id', 'posteam'], how='left')
fd_enhanced = fd_enhanced.merge(def_merge, on=['game_id', 'defteam'], how='left')

print(f"  Enhanced dataset shape : {fd_enhanced.shape}")
print(f"  Null counts (new columns):")
new_cols = ['epa_per_game_roll15','success_rate_roll15','points_per_game_roll15',
            'def_epa_per_game_roll15','def_success_rate_roll15','def_points_per_game_roll15']
print(fd_enhanced[new_cols].isnull().sum())

fd_enhanced.to_csv(OUT_FOURTH, index=False)
print(f"\nSaved enhanced dataset -> {OUT_FOURTH}")

# =============================================================================
# 5. Team stats snapshot (latest rolling values — used by app.py)
# =============================================================================
print("\nBuilding team stats snapshot ...")

off_snap = (
    game_off.sort_values(['posteam', 'season', 'week'])
    .groupby('posteam')
    .last()[['epa_per_game_roll15', 'success_rate_roll15', 'points_per_game_roll15']]
    .reset_index()
    .rename(columns={'posteam': 'team'})
)

# Latest 4th down go/stop rates per team
fd_rates = (
    fd[['posteam', 'go_conv_rate', 'go_stop_rate']]
    .dropna()
    .groupby('posteam')
    .last()
    .reset_index()
    .rename(columns={'posteam': 'team'})
)
off_snap = off_snap.merge(fd_rates, on='team', how='left')

def_snap = (
    game_def.sort_values(['team', 'season', 'week'])
    .groupby('team')
    .last()[['def_epa_per_game_roll15', 'def_success_rate_roll15', 'def_points_per_game_roll15']]
    .reset_index()
)

team_stats = off_snap.merge(def_snap, on='team', how='outer')
team_stats = team_stats.sort_values('team').reset_index(drop=True)

print(team_stats.to_string())
team_stats.to_csv(OUT_TEAM_STATS, index=False)
print(f"\nSaved team stats snapshot ({len(team_stats)} teams) -> {OUT_TEAM_STATS}")
print("\nDone. Now run model.py to retrain.")
