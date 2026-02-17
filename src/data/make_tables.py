from pathlib import Path
import polars as pl

RAW = Path("data/raw/pbp_2016_2024.parquet")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

CORE_COLS = [
    "play_id","game_id","season","season_type","week","game_date",
    "home_team","away_team","posteam","defteam",
    "qtr","down","ydstogo","yardline_100",
    "quarter_seconds_remaining","half_seconds_remaining","game_seconds_remaining",
    "posteam_score","posteam_score_post","defteam_score","score_differential","defteam_score","score_differential",
    "epa","wp","wpa",
    "field_goal_attempt","field_goal_result","kick_distance",
    "punt_attempt","yards_gained","touchback",
    "fourth_down_converted","fourth_down_failed",
    "shotgun","no_huddle","qb_dropback","qb_kneel","qb_spike","qb_scramble",
    "roof","temp","wind",
    "drive","fixed_drive","order_sequence",
    # drive summary fields you already have (useful for drive-level features)
    "drive_play_count","drive_time_of_possession","drive_first_downs",
    "drive_start_yard_line","drive_end_yard_line","drive_ended_with_score",
    "drive_result","fixed_drive_result",
    # may or may not be populated but safe
    "success",
]
# de-duplicate while preserving order (prevents Polars DuplicateError)
CORE_COLS = list(dict.fromkeys(CORE_COLS))


def scan_core():
    lf = pl.scan_parquet(str(RAW))
    cols = set(lf.collect_schema().names())  # schema-only, no full collect
    return lf.select([c for c in CORE_COLS if c in cols])


def make_fourth_down_decisions():
    df = scan_core()

    fourth = (
        df
        .filter(
            (pl.col("season_type") == "REG") &
            (pl.col("down") == 4) &
            (pl.col("posteam").is_not_null()) &
            (pl.col("defteam").is_not_null()) &
            (pl.col("ydstogo").is_not_null()) &
            (pl.col("yardline_100").is_not_null())
        )
        .with_columns([
            pl.when(pl.col("field_goal_attempt") == 1).then(pl.lit("FG"))
             .when(pl.col("punt_attempt") == 1).then(pl.lit("PUNT"))
             .otherwise(pl.lit("GO"))
             .alias("decision"),

            (pl.col("fourth_down_converted") == 1).alias("go_converted"),
            (pl.col("fourth_down_failed") == 1).alias("go_failed"),
            (pl.col("field_goal_result") == "made").alias("fg_made"),
        ])
        # optional: remove obvious junk (spikes/kneels on 4th that are not real decisions)
        # qb_spike/qb_kneel are often numeric 0/1 floats -> cast before using boolean logic
        .filter(pl.col("qb_spike").fill_null(0).cast(pl.Int8) == 0)
        .filter(pl.col("qb_kneel").fill_null(0).cast(pl.Int8) == 0)

        .select([
            "game_id","season","week","game_date",
            "home_team","away_team","posteam","defteam",
            "qtr","game_seconds_remaining",
            "yardline_100","ydstogo","score_differential",
            "roof","temp","wind",
            "shotgun","no_huddle","qb_dropback","qb_scramble",
            "epa","wp","wpa",
            "decision","go_converted","go_failed","fg_made","kick_distance",
            "punt_attempt","field_goal_attempt","yards_gained","touchback",
            "drive","fixed_drive","order_sequence","play_id",
        ])
    )

    out_path = OUT / "fourth_down_decisions.parquet"
    fourth.sink_parquet(out_path)   
    print(f"Wrote {out_path}")

def make_team_season_4th_rates():
    fourth = pl.scan_parquet(str(OUT / "fourth_down_decisions.parquet"))

    # Keep GO attempts only for conversion rates
    go = fourth.filter(pl.col("decision") == "GO")

    # Bins
    go_binned = go.with_columns([
        pl.when(pl.col("ydstogo") <= 1).then(pl.lit("1"))
         .when(pl.col("ydstogo") <= 2).then(pl.lit("2"))
         .when(pl.col("ydstogo") <= 5).then(pl.lit("3-5"))
         .when(pl.col("ydstogo") <= 10).then(pl.lit("6-10"))
         .otherwise(pl.lit("11+")).alias("ydstogo_bin"),

        pl.when(pl.col("yardline_100") <= 20).then(pl.lit("0-20"))
         .when(pl.col("yardline_100") <= 40).then(pl.lit("21-40"))
         .when(pl.col("yardline_100") <= 60).then(pl.lit("41-60"))
         .when(pl.col("yardline_100") <= 80).then(pl.lit("61-80"))
         .otherwise(pl.lit("81-100")).alias("yardline_bin"),
    ])

    # Offense rates (posteam)
    off = (
        go_binned
        .group_by(["season","posteam","ydstogo_bin","yardline_bin"])
        .agg([
            pl.len().alias("go_attempts"),
            pl.col("go_converted").cast(pl.Float64).mean().alias("go_conv_rate"),
        ])
        .rename({"posteam": "team"})
        .with_columns([pl.lit("offense").alias("side")])
    )


    # Defense stop rates (defteam)
    # stop rate = 1 - offense conversion rate allowed
    d = (
        go_binned
        .group_by(["season","defteam","ydstogo_bin","yardline_bin"])
        .agg([
            pl.len().alias("go_faced"),
            (1 - pl.col("go_converted").cast(pl.Float64).mean()).alias("go_stop_rate"),
        ])
        .rename({"defteam": "team"})
        .with_columns([pl.lit("defense").alias("side")])
    )


    out_path = OUT / "team_season_4th_rates.parquet"
    pl.concat([off, d], how="diagonal").sink_parquet(out_path)
    print(f"Wrote {out_path}")

def make_drive_summary():
    df = scan_core()

    drives = (
        df
        .filter((pl.col("season_type") == "REG") & pl.col("posteam").is_not_null())
        .group_by(["game_id","season","week","game_date","posteam","defteam","home_team","away_team","fixed_drive"])
        .agg([
            pl.sum("epa").alias("drive_epa"),
            pl.col("success").cast(pl.Float64).mean().alias("drive_success_rate"),
            pl.min("posteam_score").alias("drive_posteam_score_start"),
            pl.max("posteam_score_post").alias("drive_posteam_score_end"),
            pl.first("drive_start_yard_line").alias("drive_start_yard_line"),
            pl.first("fixed_drive_result").alias("fixed_drive_result"),
        ])

        .with_columns([
            (pl.col("drive_posteam_score_end") - pl.col("drive_posteam_score_start")).alias("drive_points")
        ])
    )

    out_path = OUT / "drive_summary.parquet"
    drives.sink_parquet(out_path)
    print(f"Wrote {out_path}")

def make_team_rolling_offense(window_games: int = 6):
    drives = pl.scan_parquet(str(OUT / "drive_summary.parquet"))

    # game-level offense summaries
    game_off = (
        drives
        .group_by(["season","week","game_date","game_id","posteam"])
        .agg([
            pl.mean("drive_epa").alias("epa_per_drive"),
            pl.mean("drive_success_rate").alias("success_rate"),
            pl.mean("drive_points").alias("points_per_drive"),
            pl.len().alias("n_drives"),
        ])
        .rename({"posteam": "team"})
        .sort(["team","season","week","game_date"])
    )

    # rolling over last N games (excluding current game) -> leakage-safe
    # polars rolling supports time-based; here do group-wise shift then rolling_mean
    rolled = (
        game_off
        .with_columns([
            pl.col("epa_per_drive").shift(1).over(["team","season"]).alias("epa_pd_lag1"),
            pl.col("success_rate").shift(1).over(["team","season"]).alias("sr_lag1"),
            pl.col("points_per_drive").shift(1).over(["team","season"]).alias("ppd_lag1"),
        ])
        .with_columns([
            pl.col("epa_pd_lag1").rolling_mean(window_games).over(["team","season"]).alias(f"epa_per_drive_roll{window_games}"),
            pl.col("sr_lag1").rolling_mean(window_games).over(["team","season"]).alias(f"success_rate_roll{window_games}"),
            pl.col("ppd_lag1").rolling_mean(window_games).over(["team","season"]).alias(f"points_per_drive_roll{window_games}"),
        ])
        .select([
            "season","week","game_date","game_id","team",
            f"epa_per_drive_roll{window_games}",
            f"success_rate_roll{window_games}",
            f"points_per_drive_roll{window_games}",
        ])
    )

    out_path = OUT / "team_rolling_offense.parquet"
    rolled.sink_parquet(out_path)
    print(f"Wrote {out_path}")

def make_fourth_down_with_features():
    fourth = pl.scan_parquet(str(OUT / "fourth_down_decisions.parquet"))
    rates  = pl.scan_parquet(str(OUT / "team_season_4th_rates.parquet"))
    roll   = pl.scan_parquet(str(OUT / "team_rolling_offense.parquet"))

    # add bins to fourth to match rates table
    fourth_b = fourth.with_columns([
        pl.when(pl.col("ydstogo") <= 1).then(pl.lit("1"))
         .when(pl.col("ydstogo") <= 2).then(pl.lit("2"))
         .when(pl.col("ydstogo") <= 5).then(pl.lit("3-5"))
         .when(pl.col("ydstogo") <= 10).then(pl.lit("6-10"))
         .otherwise(pl.lit("11+")).alias("ydstogo_bin"),

        pl.when(pl.col("yardline_100") <= 20).then(pl.lit("0-20"))
         .when(pl.col("yardline_100") <= 40).then(pl.lit("21-40"))
         .when(pl.col("yardline_100") <= 60).then(pl.lit("41-60"))
         .when(pl.col("yardline_100") <= 80).then(pl.lit("61-80"))
         .otherwise(pl.lit("81-100")).alias("yardline_bin"),
    ])

    # offense conversion rate feature
    off_rates = (
        rates.filter(pl.col("side") == "offense")
            .select(["season","team","ydstogo_bin","yardline_bin","go_conv_rate","go_attempts"])
    )

    def_rates = (
        rates.filter(pl.col("side") == "defense")
            .select(["season","team","ydstogo_bin","yardline_bin","go_stop_rate","go_faced"])
    )

    out = (
        fourth_b
        .join(
            off_rates,
            left_on=["season","posteam","ydstogo_bin","yardline_bin"],
            right_on=["season","team","ydstogo_bin","yardline_bin"],
            how="left",
            suffix="_off",
        )
        .join(
            def_rates,
            left_on=["season","defteam","ydstogo_bin","yardline_bin"],
            right_on=["season","team","ydstogo_bin","yardline_bin"],
            how="left",
            suffix="_def",
        )
        .join(
            roll.select(["game_id","team","epa_per_drive_roll6","success_rate_roll6","points_per_drive_roll6"]),
            left_on=["game_id","defteam"],
            right_on=["game_id","team"],
            how="left",
            suffix="_roll",
        )
    )




    out_path = OUT / "fourth_down_with_features.parquet"
    out.sink_parquet(out_path)
    print(f"Wrote {out_path}")

def make_ot_first_possession_fourth_downs():
    df = scan_core()

    ot = (
        df
        .filter((pl.col("season_type")=="REG") & (pl.col("qtr") >= 5))
        .filter(pl.col("posteam").is_not_null())
        .select(["game_id","posteam","fixed_drive","order_sequence","down","ydstogo","yardline_100",
                 "game_seconds_remaining","score_differential",
                 "field_goal_attempt","punt_attempt","fourth_down_converted","fourth_down_failed",
                 "field_goal_result","kick_distance",
                 "epa","wp","wpa",
                 "season","week","game_date","home_team","away_team","defteam"])
    )

    # identify first OT drive per game (min fixed_drive)
    first_drive = (
        ot.group_by("game_id")
          .agg(pl.min("fixed_drive").alias("first_ot_drive"))
    )

    first_ot_plays = ot.join(first_drive, on="game_id", how="inner") \
                      .filter(pl.col("fixed_drive") == pl.col("first_ot_drive"))

    # now keep 4th down plays on that drive and label decision
    first_ot_fourth = (
        first_ot_plays
        .filter(pl.col("down") == 4)
        .with_columns([
            pl.when(pl.col("field_goal_attempt") == 1).then(pl.lit("FG"))
             .when(pl.col("punt_attempt") == 1).then(pl.lit("PUNT"))
             .otherwise(pl.lit("GO"))
             .alias("decision"),
            (pl.col("fourth_down_converted")==1).alias("go_converted"),
            (pl.col("field_goal_result")=="made").alias("fg_made"),
        ])
        .sort(["game_id","order_sequence"])
    )

    out_path = OUT / "ot_first_possession_4th.parquet"
    first_ot_fourth.sink_parquet(out_path)

    print(f"Wrote {out_path}")

def main():
    make_fourth_down_decisions()
    make_team_season_4th_rates()
    make_drive_summary()
    make_team_rolling_offense(window_games=6)
    make_fourth_down_with_features()
    make_ot_first_possession_fourth_downs()

if __name__ == "__main__":
    main()
