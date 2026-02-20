"""
Defense feature extraction for the NFL 4th-down / OT model.
Group 3 (Defense) -- feature/defense branch

Outputs (all written to data/processed/):
  defense_drive_summary.parquet   -- drive-level: EPA allowed, yards allowed, stop rate, points allowed
  team_rolling_defense.parquet    -- rolling 6-game defensive averages (leakage-safe)
  team_rolling_pressure.parquet   -- rolling sack rate, QB-hit rate, TFL rate
  team_rolling_turnovers.parquet  -- rolling interceptions + forced fumbles

Usage:
  python3 -m src.data.make_defense
"""

from pathlib import Path
import polars as pl

RAW = Path("data/raw/pbp_2016_2024.parquet")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DEFENSE_COLS = [
    "play_id", "game_id", "season", "season_type", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    "qtr", "down", "ydstogo", "yardline_100",
    "game_seconds_remaining", "half_seconds_remaining",
    "posteam_score", "posteam_score_post",
    "defteam_score", "defteam_score_post",
    "score_differential",
    "epa", "wp", "wpa",
    "success",
    "yards_gained",
    "rush_attempt", "pass_attempt",
    "sack", "qb_hit", "tackled_for_loss",
    "interception", "fumble_forced", "fumble_lost",
    "fourth_down_converted", "fourth_down_failed",
    "field_goal_attempt", "punt_attempt",
    "fixed_drive", "fixed_drive_result",
    "drive_play_count", "drive_first_downs",
    "drive_ended_with_score",
]


def scan_defense():
    lf = pl.scan_parquet(str(RAW))
    schema_cols = set(lf.collect_schema().names())
    return lf.select([c for c in DEFENSE_COLS if c in schema_cols])


def make_defense_drive_summary():
    """
    Drive-level defensive stats: EPA allowed, yards allowed, points allowed, stop rate.
    One row per (game_id, defteam, fixed_drive).

    Fixes applied:
    - Drop "End of half" / "End of game" drives (clock-kill plays, not real defensive stops)
    - Drop 3 drives with null fixed_drive_result (phantom data gaps)
    - drive_points_allowed capped at 7 to exclude 2-pt conversion inflation
    """
    df = scan_defense()

    drives = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            pl.col("defteam").is_not_null() &
            pl.col("posteam").is_not_null() &
            # Fix 2 & 3: drop end-of-half/game drives and null drive results
            pl.col("fixed_drive_result").is_not_null() &
            ~pl.col("fixed_drive_result").is_in(["End of half", "End of game"])
        )
        .group_by([
            "game_id", "season", "week", "game_date",
            "defteam", "posteam", "home_team", "away_team", "fixed_drive",
        ])
        .agg([
            pl.sum("epa").alias("drive_epa_allowed"),
            pl.col("success").cast(pl.Float64).mean().alias("drive_offense_success_rate"),
            pl.sum("yards_gained").alias("drive_yards_allowed"),
            # points allowed = how much the offense scored on this drive
            # Fix 4: cap at 7 to prevent 2-pt conversion plays inflating score to 8
            (pl.col("posteam_score_post").max() - pl.col("posteam_score").min())
                .clip(lower_bound=0, upper_bound=7).alias("drive_points_allowed"),
            pl.first("fixed_drive_result").alias("fixed_drive_result"),
        ])
        .with_columns([
            # Fix 5: null success rate = pick-6 / special teams TD drives (0 yards, 0 pts)
            # The offense was fully stopped, so stop_rate = 1.0
            (1.0 - pl.col("drive_offense_success_rate")).fill_null(1.0).alias("drive_stop_rate"),
        ])
    )

    out_path = OUT / "defense_drive_summary.parquet"
    drives.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_team_rolling_defense(window_games: int = 6):
    """
    Rolling defensive metrics over the last N games (shift-1 before rolling = leakage-safe).
    Mirrors make_team_rolling_offense() on the offense branch.

    Columns in output:
      epa_allowed_roll6        -- avg EPA allowed per drive over last 6 games
      stop_rate_roll6          -- avg stop rate (1 - offense success rate) over last 6
      yards_allowed_roll6      -- avg yards allowed per drive over last 6
      points_allowed_roll6     -- avg points allowed per drive over last 6
    """
    drives = pl.scan_parquet(str(OUT / "defense_drive_summary.parquet"))

    game_def = (
        drives
        .group_by(["season", "week", "game_date", "game_id", "defteam"])
        .agg([
            pl.mean("drive_epa_allowed").alias("epa_allowed_per_drive"),
            pl.mean("drive_stop_rate").alias("stop_rate"),
            pl.mean("drive_yards_allowed").alias("yards_allowed_per_drive"),
            pl.mean("drive_points_allowed").alias("points_allowed_per_drive"),
            pl.len().alias("n_drives"),
        ])
        .rename({"defteam": "team"})
        .sort(["team", "season", "week", "game_date"])
    )

    rolled = (
        game_def
        .with_columns([
            pl.col("epa_allowed_per_drive").shift(1).over(["team", "season"]).alias("epa_lag1"),
            pl.col("stop_rate").shift(1).over(["team", "season"]).alias("stop_lag1"),
            pl.col("yards_allowed_per_drive").shift(1).over(["team", "season"]).alias("yards_lag1"),
            pl.col("points_allowed_per_drive").shift(1).over(["team", "season"]).alias("pts_lag1"),
        ])
        .with_columns([
            pl.col("epa_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"epa_allowed_roll{window_games}"),
            pl.col("stop_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"stop_rate_roll{window_games}"),
            pl.col("yards_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"yards_allowed_roll{window_games}"),
            pl.col("pts_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"points_allowed_roll{window_games}"),
        ])
        .select([
            "season", "week", "game_date", "game_id", "team",
            f"epa_allowed_roll{window_games}",
            f"stop_rate_roll{window_games}",
            f"yards_allowed_roll{window_games}",
            f"points_allowed_roll{window_games}",
        ])
    )

    out_path = OUT / "team_rolling_defense.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_defense_pressure_stats():
    """
    Rolling pass-rush and run-stop pressure metrics.

    Fix 1: tfl_rate was always 0 because tackled_for_loss is only set on rush plays,
    but we were filtering to pass plays only. Now computed separately:
    - sack_rate, qb_hit_rate: per pass play faced (unchanged, correct denominator)
    - tfl_rate: TFLs per rush play faced (correct denominator)
    Both are then joined and rolled together.
    """
    df = scan_defense()

    # Pass pressure: sacks and QB hits (pass plays only)
    pass_plays = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            pl.col("defteam").is_not_null() &
            (
                (pl.col("pass_attempt").fill_null(0).cast(pl.Int8) == 1) |
                (pl.col("sack").fill_null(0).cast(pl.Int8) == 1)
            )
        )
        .group_by(["game_id", "season", "week", "defteam"])
        .agg([
            pl.len().alias("pass_plays_faced"),
            pl.col("sack").fill_null(0).cast(pl.Float64).sum().alias("sacks"),
            pl.col("qb_hit").fill_null(0).cast(pl.Float64).sum().alias("qb_hits"),
        ])
        .with_columns([
            (pl.col("sacks") / pl.col("pass_plays_faced")).alias("sack_rate"),
            (pl.col("qb_hits") / pl.col("pass_plays_faced")).alias("qb_hit_rate"),
        ])
    )

    # Fix 1: TFL rate on rush plays only (that's where tackled_for_loss is flagged)
    rush_plays = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            pl.col("defteam").is_not_null() &
            (pl.col("rush_attempt").fill_null(0).cast(pl.Int8) == 1)
        )
        .group_by(["game_id", "defteam"])
        .agg([
            pl.len().alias("rush_plays_faced"),
            pl.col("tackled_for_loss").fill_null(0).cast(pl.Float64).sum().alias("tfl"),
        ])
        .with_columns([
            (pl.col("tfl") / pl.col("rush_plays_faced")).alias("tfl_rate"),
        ])
    )

    game_pressure = (
        pass_plays
        .join(rush_plays, on=["game_id", "defteam"], how="left")
        .rename({"defteam": "team"})
        .sort(["team", "season", "week"])
    )

    rolled = (
        game_pressure
        .with_columns([
            pl.col("sack_rate").shift(1).over(["team", "season"]).alias("sack_lag1"),
            pl.col("qb_hit_rate").shift(1).over(["team", "season"]).alias("qb_hit_lag1"),
            pl.col("tfl_rate").shift(1).over(["team", "season"]).alias("tfl_lag1"),
        ])
        .with_columns([
            pl.col("sack_lag1").rolling_mean(6).over(["team", "season"]).alias("sack_rate_roll6"),
            pl.col("qb_hit_lag1").rolling_mean(6).over(["team", "season"]).alias("qb_hit_rate_roll6"),
            pl.col("tfl_lag1").rolling_mean(6).over(["team", "season"]).alias("tfl_rate_roll6"),
        ])
        .select([
            "season", "week", "game_id", "team",
            "sack_rate_roll6", "qb_hit_rate_roll6", "tfl_rate_roll6",
        ])
    )

    out_path = OUT / "team_rolling_pressure.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_defense_turnover_stats():
    """
    Rolling turnover-generation metrics: interceptions, forced fumbles, total turnovers.
    """
    df = scan_defense()

    game_to = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            pl.col("defteam").is_not_null()
        )
        .group_by(["game_id", "season", "week", "defteam"])
        .agg([
            pl.col("interception").fill_null(0).cast(pl.Float64).sum().alias("interceptions"),
            pl.col("fumble_forced").fill_null(0).cast(pl.Float64).sum().alias("fumbles_forced"),
        ])
        .with_columns([
            (pl.col("interceptions") + pl.col("fumbles_forced")).alias("total_turnovers"),
        ])
        .rename({"defteam": "team"})
        .sort(["team", "season", "week"])
    )

    rolled = (
        game_to
        .with_columns([
            pl.col("total_turnovers").shift(1).over(["team", "season"]).alias("to_lag1"),
        ])
        .with_columns([
            pl.col("to_lag1").rolling_mean(6).over(["team", "season"]).alias("turnovers_roll6"),
        ])
        .select([
            "season", "week", "game_id", "team",
            "interceptions", "fumbles_forced", "total_turnovers", "turnovers_roll6",
        ])
    )

    out_path = OUT / "team_rolling_turnovers.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def main():
    print("Building defense drive summary...")
    make_defense_drive_summary()

    print("Building rolling defense metrics...")
    make_team_rolling_defense(window_games=6)

    print("Building pressure stats...")
    make_defense_pressure_stats()

    print("Building turnover stats...")
    make_defense_turnover_stats()

    print("Done! Outputs in data/processed/")


if __name__ == "__main__":
    main()
