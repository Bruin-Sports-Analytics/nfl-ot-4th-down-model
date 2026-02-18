"""
Situational feature extraction for the NFL 4th-down / OT model.
Group 1 (Situational) -- feature/situational branch

Outputs (all written to data/processed/):
  situational_4th_down.parquet       -- every 4th down with rich situational bins + flags
  ot_situations.parquet              -- OT plays with situational context
  team_situational_tendencies.parquet -- rolling 6-game go/FG/punt rates by situation bucket

Usage:
  python3 -m src.data.make_situational
"""

from pathlib import Path
import polars as pl

RAW = Path("data/raw/pbp_2016_2024.parquet")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SITUATIONAL_COLS = [
    "play_id", "game_id", "season", "season_type", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam", "posteam_type",
    "qtr", "down", "ydstogo", "yardline_100",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "posteam_score", "defteam_score", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "wp", "vegas_wp", "wpa",
    "epa",
    "shotgun", "no_huddle",
    "roof", "temp", "wind",
    "field_goal_attempt", "field_goal_result", "kick_distance",
    "punt_attempt",
    "fourth_down_converted", "fourth_down_failed",
    "qb_spike", "qb_kneel",
    "fixed_drive", "order_sequence",
]


def scan_situational():
    lf = pl.scan_parquet(str(RAW))
    schema_cols = set(lf.collect_schema().names())
    return lf.select([c for c in SITUATIONAL_COLS if c in schema_cols])


def make_situational_4th_down():
    """
    Every regular-season 4th down with situational bins and context flags.
    This is the core situational table — joins cleanly with offense/defense tables on game_id.
    """
    df = scan_situational()

    fourth = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("down") == 4) &
            pl.col("posteam").is_not_null() &
            pl.col("defteam").is_not_null() &
            pl.col("ydstogo").is_not_null() &
            pl.col("yardline_100").is_not_null()
        )
        # remove junk plays
        .filter(pl.col("qb_spike").fill_null(0).cast(pl.Int8) == 0)
        .filter(pl.col("qb_kneel").fill_null(0).cast(pl.Int8) == 0)

        # decision label
        .with_columns([
            pl.when(pl.col("field_goal_attempt") == 1).then(pl.lit("FG"))
             .when(pl.col("punt_attempt") == 1).then(pl.lit("PUNT"))
             .otherwise(pl.lit("GO"))
             .alias("decision"),

            (pl.col("fourth_down_converted") == 1).alias("go_converted"),
            (pl.col("field_goal_result") == "made").alias("fg_made"),

            # is the posteam the home team?
            (pl.col("posteam") == pl.col("home_team")).alias("is_home"),

            # dome/weather flag
            pl.col("roof").is_in(["dome", "closed"]).alias("is_dome"),
        ])

        # situational bins
        .with_columns([
            # distance bucket (same bins as offense/defense tables)
            pl.when(pl.col("ydstogo") <= 1).then(pl.lit("1"))
             .when(pl.col("ydstogo") <= 2).then(pl.lit("2"))
             .when(pl.col("ydstogo") <= 5).then(pl.lit("3-5"))
             .when(pl.col("ydstogo") <= 10).then(pl.lit("6-10"))
             .otherwise(pl.lit("11+"))
             .alias("distance_bucket"),

            # field position bucket
            pl.when(pl.col("yardline_100") <= 20).then(pl.lit("red_zone"))
             .when(pl.col("yardline_100") <= 40).then(pl.lit("opp_territory"))
             .when(pl.col("yardline_100") <= 60).then(pl.lit("midfield"))
             .when(pl.col("yardline_100") <= 80).then(pl.lit("own_territory"))
             .otherwise(pl.lit("backed_up"))
             .alias("field_position_bucket"),

            # score bucket
            pl.when(pl.col("score_differential") < -14).then(pl.lit("big_deficit"))
             .when(pl.col("score_differential") < -3).then(pl.lit("deficit"))
             .when(pl.col("score_differential") <= 3).then(pl.lit("close"))
             .when(pl.col("score_differential") <= 14).then(pl.lit("ahead"))
             .otherwise(pl.lit("big_lead"))
             .alias("score_bucket"),

            # time bucket
            pl.when(pl.col("qtr") >= 5).then(pl.lit("OT"))
             .when(pl.col("game_seconds_remaining") < 300).then(pl.lit("crunch_time"))
             .when(pl.col("game_seconds_remaining") < 840).then(pl.lit("late"))
             .when(pl.col("game_seconds_remaining") < 1680).then(pl.lit("mid"))
             .otherwise(pl.lit("early"))
             .alias("time_bucket"),

            # clutch flag: 4th qtr or OT, within 8 pts, under 5 min
            (
                (pl.col("qtr") >= 4) &
                (pl.col("score_differential").abs() <= 8) &
                (pl.col("game_seconds_remaining") < 300)
            ).alias("is_clutch"),
        ])

        .select([
            "game_id", "season", "week", "game_date",
            "home_team", "away_team", "posteam", "defteam",
            "qtr", "game_seconds_remaining", "half_seconds_remaining",
            "yardline_100", "ydstogo", "score_differential",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "wp", "vegas_wp", "wpa", "epa",
            "shotgun", "no_huddle",
            "roof", "temp", "wind", "is_dome",
            "is_home", "is_clutch",
            "distance_bucket", "field_position_bucket", "score_bucket", "time_bucket",
            "decision", "go_converted", "fg_made", "kick_distance",
            "fixed_drive", "order_sequence", "play_id",
        ])
    )

    out_path = OUT / "situational_4th_down.parquet"
    fourth.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_ot_situations():
    """
    All OT plays (qtr >= 5) with situational context.
    Flags the first possession drive per game.
    """
    df = scan_situational()

    ot = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("qtr") >= 5) &
            pl.col("posteam").is_not_null()
        )
        .with_columns([
            pl.when(pl.col("field_goal_attempt") == 1).then(pl.lit("FG"))
             .when(pl.col("punt_attempt") == 1).then(pl.lit("PUNT"))
             .otherwise(pl.lit("GO"))
             .alias("decision"),
            (pl.col("fourth_down_converted") == 1).alias("go_converted"),
            (pl.col("field_goal_result") == "made").alias("fg_made"),
            (pl.col("posteam") == pl.col("home_team")).alias("is_home"),
        ])
    )

    # flag first OT possession per game
    first_drive = (
        ot.group_by("game_id")
          .agg(pl.min("fixed_drive").alias("first_ot_drive"))
    )

    ot_with_flag = (
        ot.join(first_drive, on="game_id", how="left")
          .with_columns([
              (pl.col("fixed_drive") == pl.col("first_ot_drive")).alias("is_first_ot_possession"),
          ])
          .drop("first_ot_drive")
    )

    out_path = OUT / "ot_situations.parquet"
    ot_with_flag.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def make_team_situational_tendencies(window_games: int = 6):
    """
    Rolling 6-game go/FG/punt rates per team broken down by situation bucket.
    Useful as a feature: does this team tend to go for it in this situation?
    """
    fourth = pl.scan_parquet(str(OUT / "situational_4th_down.parquet"))

    # game-level decision counts per team per situation bucket combo
    game_tend = (
        fourth
        .group_by(["season", "week", "game_id", "posteam", "distance_bucket", "field_position_bucket"])
        .agg([
            pl.len().alias("n_plays"),
            (pl.col("decision") == "GO").cast(pl.Float64).mean().alias("go_rate"),
            (pl.col("decision") == "FG").cast(pl.Float64).mean().alias("fg_rate"),
            (pl.col("decision") == "PUNT").cast(pl.Float64).mean().alias("punt_rate"),
        ])
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
    )

    # rolling (shift-1 for leakage safety)
    rolled = (
        game_tend
        .with_columns([
            pl.col("go_rate").shift(1).over(["team", "season", "distance_bucket", "field_position_bucket"]).alias("go_lag1"),
            pl.col("fg_rate").shift(1).over(["team", "season", "distance_bucket", "field_position_bucket"]).alias("fg_lag1"),
            pl.col("punt_rate").shift(1).over(["team", "season", "distance_bucket", "field_position_bucket"]).alias("punt_lag1"),
        ])
        .with_columns([
            pl.col("go_lag1").rolling_mean(window_games)
                .over(["team", "season", "distance_bucket", "field_position_bucket"])
                .alias(f"go_rate_roll{window_games}"),
            pl.col("fg_lag1").rolling_mean(window_games)
                .over(["team", "season", "distance_bucket", "field_position_bucket"])
                .alias(f"fg_rate_roll{window_games}"),
            pl.col("punt_lag1").rolling_mean(window_games)
                .over(["team", "season", "distance_bucket", "field_position_bucket"])
                .alias(f"punt_rate_roll{window_games}"),
        ])
        .select([
            "season", "week", "game_id", "team",
            "distance_bucket", "field_position_bucket",
            f"go_rate_roll{window_games}",
            f"fg_rate_roll{window_games}",
            f"punt_rate_roll{window_games}",
        ])
    )

    out_path = OUT / "team_situational_tendencies.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")


def main():
    print("Building situational 4th down table...")
    make_situational_4th_down()

    print("Building OT situations table...")
    make_ot_situations()

    print("Building team situational tendencies...")
    make_team_situational_tendencies(window_games=6)

    print("Done! Outputs in data/processed/")


if __name__ == "__main__":
    main()
