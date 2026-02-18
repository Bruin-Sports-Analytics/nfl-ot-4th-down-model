"""
Special teams feature extraction for the NFL 4th-down / OT model.
feature/special-teams branch

Outputs (all written to data/processed/):
  fg_attempts.parquet                 -- every field goal attempt with result and context
  team_rolling_kicker.parquet         -- rolling 6-game FG make rate by distance bucket
  punt_attempts.parquet               -- every punt with distance and field position context
  team_rolling_punter.parquet         -- rolling 6-game avg punt distance and inside-20 rate
  team_rolling_kickoff.parquet        -- rolling 6-game touchback rate and return yards allowed

Usage:
  python3 -m src.data.make_special_teams
"""

from pathlib import Path
import polars as pl

RAW = Path("data/raw/pbp_2016_2024.parquet")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SPECIAL_TEAMS_COLS = [
    "play_id", "game_id", "season", "season_type", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    "qtr", "ydstogo", "yardline_100",
    "game_seconds_remaining", "score_differential",
    "wp",
    # field goals
    "field_goal_attempt", "field_goal_result", "kick_distance",
    "fg_prob", "opp_fg_prob",
    "kicker_player_id", "kicker_player_name",
    # extra points
    "extra_point_attempt", "extra_point_result",
    # punts
    "punt_attempt", "punt_blocked", "punt_downed",
    "punt_fair_catch", "punt_in_endzone", "punt_inside_twenty",
    "punt_out_of_bounds", "touchback",
    "punter_player_id", "punter_player_name",
    "return_yards",
    # kickoffs
    "kickoff_attempt", "kickoff_in_endzone", "kickoff_inside_twenty",
    "kickoff_fair_catch", "kickoff_out_of_bounds",
    "return_touchdown",
    "epa",
]


def scan_special_teams():
    lf = pl.scan_parquet(str(RAW))
    schema_cols = set(lf.collect_schema().names())
    return lf.select([c for c in SPECIAL_TEAMS_COLS if c in schema_cols])


def make_fg_attempts():
    """
    Every regular-season field goal attempt with result, distance bucket, and context.
    """
    df = scan_special_teams()

    fg = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("field_goal_attempt").fill_null(0).cast(pl.Int8) == 1)
        )
        .with_columns([
            (pl.col("field_goal_result") == "made").alias("fg_made"),
            (pl.col("field_goal_result") == "blocked").alias("fg_blocked"),

            # distance buckets relevant for 4th down FG decisions
            pl.when(pl.col("kick_distance") <= 30).then(pl.lit("0-30"))
             .when(pl.col("kick_distance") <= 40).then(pl.lit("31-40"))
             .when(pl.col("kick_distance") <= 50).then(pl.lit("41-50"))
             .when(pl.col("kick_distance") <= 55).then(pl.lit("51-55"))
             .otherwise(pl.lit("56+"))
             .alias("distance_bucket"),
        ])
        .select([
            "game_id", "season", "week", "game_date",
            "posteam", "defteam",
            "qtr", "yardline_100", "kick_distance", "distance_bucket",
            "game_seconds_remaining", "score_differential", "wp",
            "fg_prob",
            "fg_made", "fg_blocked", "field_goal_result",
            "kicker_player_id", "kicker_player_name",
            "epa", "play_id",
        ])
    )

    out_path = OUT / "fg_attempts.parquet"
    fg.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_team_rolling_kicker(window_games: int = 6):
    """
    Rolling FG make rate per team by distance bucket (leakage-safe).
    Key feature for 4th down FG decisions: how reliable is this kicker right now?
    """
    fg = pl.scan_parquet(str(OUT / "fg_attempts.parquet"))

    game_fg = (
        fg
        .group_by(["season", "week", "game_id", "posteam", "distance_bucket"])
        .agg([
            pl.len().alias("fg_attempts"),
            pl.col("fg_made").cast(pl.Float64).mean().alias("fg_make_rate"),
            pl.col("fg_blocked").cast(pl.Float64).mean().alias("fg_block_rate"),
        ])
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
    )

    rolled = (
        game_fg
        .with_columns([
            pl.col("fg_make_rate").shift(1).over(["team", "season", "distance_bucket"]).alias("make_lag1"),
        ])
        .with_columns([
            pl.col("make_lag1").rolling_mean(window_games)
                .over(["team", "season", "distance_bucket"])
                .alias(f"fg_make_rate_roll{window_games}"),
        ])
        .select([
            "season", "week", "game_id", "team", "distance_bucket",
            f"fg_make_rate_roll{window_games}",
        ])
    )

    out_path = OUT / "team_rolling_kicker.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_punt_attempts():
    """
    Every regular-season punt with distance, field position, and outcome context.
    """
    df = scan_special_teams()

    punts = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("punt_attempt").fill_null(0).cast(pl.Int8) == 1)
        )
        .with_columns([
            pl.col("punt_inside_twenty").fill_null(0).cast(pl.Boolean).alias("inside_twenty"),
            pl.col("punt_blocked").fill_null(0).cast(pl.Boolean).alias("blocked"),
            pl.col("touchback").fill_null(0).cast(pl.Boolean).alias("touchback"),
            pl.col("return_yards").fill_null(0).alias("return_yards"),
        ])
        .select([
            "game_id", "season", "week", "game_date",
            "posteam", "defteam",
            "qtr", "yardline_100", "ydstogo",
            "game_seconds_remaining", "score_differential", "wp",
            "kick_distance",
            "inside_twenty", "blocked", "touchback", "return_yards",
            "punter_player_id", "punter_player_name",
            "epa", "play_id",
        ])
    )

    out_path = OUT / "punt_attempts.parquet"
    punts.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_team_rolling_punter(window_games: int = 6):
    """
    Rolling punt distance and inside-20 rate per team (leakage-safe).
    Useful for 4th down punt decisions: how good is this team's punter right now?
    """
    punts = pl.scan_parquet(str(OUT / "punt_attempts.parquet"))

    game_punt = (
        punts
        .group_by(["season", "week", "game_id", "posteam"])
        .agg([
            pl.len().alias("n_punts"),
            pl.col("kick_distance").mean().alias("avg_punt_distance"),
            pl.col("inside_twenty").cast(pl.Float64).mean().alias("inside_twenty_rate"),
            pl.col("blocked").cast(pl.Float64).mean().alias("block_rate"),
            pl.col("return_yards").mean().alias("avg_return_yards"),
        ])
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
    )

    rolled = (
        game_punt
        .with_columns([
            pl.col("avg_punt_distance").shift(1).over(["team", "season"]).alias("dist_lag1"),
            pl.col("inside_twenty_rate").shift(1).over(["team", "season"]).alias("i20_lag1"),
            pl.col("avg_return_yards").shift(1).over(["team", "season"]).alias("ret_lag1"),
        ])
        .with_columns([
            pl.col("dist_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"punt_distance_roll{window_games}"),
            pl.col("i20_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"inside_twenty_rate_roll{window_games}"),
            pl.col("ret_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"return_yards_allowed_roll{window_games}"),
        ])
        .select([
            "season", "week", "game_id", "team",
            f"punt_distance_roll{window_games}",
            f"inside_twenty_rate_roll{window_games}",
            f"return_yards_allowed_roll{window_games}",
        ])
    )

    out_path = OUT / "team_rolling_punter.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_team_rolling_kickoff(window_games: int = 6):
    """
    Rolling kickoff touchback rate and return yards allowed per team (leakage-safe).
    """
    df = scan_special_teams()

    kickoffs = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("kickoff_attempt").fill_null(0).cast(pl.Int8) == 1)
        )
        .group_by(["season", "week", "game_id", "posteam"])
        .agg([
            pl.len().alias("n_kickoffs"),
            pl.col("touchback").fill_null(0).cast(pl.Float64).mean().alias("touchback_rate"),
            pl.col("return_yards").fill_null(0).mean().alias("avg_return_yards"),
            pl.col("return_touchdown").fill_null(0).cast(pl.Float64).mean().alias("return_td_rate"),
        ])
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
    )

    rolled = (
        kickoffs
        .with_columns([
            pl.col("touchback_rate").shift(1).over(["team", "season"]).alias("tb_lag1"),
            pl.col("avg_return_yards").shift(1).over(["team", "season"]).alias("ret_lag1"),
        ])
        .with_columns([
            pl.col("tb_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"touchback_rate_roll{window_games}"),
            pl.col("ret_lag1").rolling_mean(window_games).over(["team", "season"])
                .alias(f"return_yards_roll{window_games}"),
        ])
        .select([
            "season", "week", "game_id", "team",
            f"touchback_rate_roll{window_games}",
            f"return_yards_roll{window_games}",
        ])
    )

    out_path = OUT / "team_rolling_kickoff.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def main():
    print("Building FG attempts table...")
    make_fg_attempts()

    print("Building rolling kicker stats...")
    make_team_rolling_kicker(window_games=6)

    print("Building punt attempts table...")
    make_punt_attempts()

    print("Building rolling punter stats...")
    make_team_rolling_punter(window_games=6)

    print("Building rolling kickoff stats...")
    make_team_rolling_kickoff(window_games=6)

    print("Done! Outputs in data/processed/")


if __name__ == "__main__":
    main()
